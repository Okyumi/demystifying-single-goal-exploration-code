"""
Tabular SGCRL agent adapted for the ContinualMaze environment.

This is a modular extraction of the SGCRLAgent from tabular_maze.ipynb,
generalised so the step function is provided by a ContinualMaze instance
rather than a module-level global.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any

from .envs.continual_maze import ContinualMaze


class TabularSGCRLAgent:
    """Tabular Single-Goal Contrastive RL agent.

    Parameters
    ----------
    env : ContinualMaze
        The maze environment (provides step(), start_state, goal_state, etc.).
    rep_dim : int
        Dimensionality of the ψ embeddings.
    lr_psi : float
        Learning rate for embedding updates.
    batch_size : int
        Number of (s, s⁺) pairs per contrastive update.
    replay_capacity : int
        Maximum number of trajectories in the replay buffer.
    max_steps : int
        Maximum steps per episode.
    gamma : float
        Discount factor for geometric future-state sampling.
    entropy_coeff : float
        Temperature for softmax action selection (lower = greedier).
    episodes_per_update : int
        How often to run a contrastive update (in episodes).
    normalize : bool
        Whether to L2-normalize embeddings after each update.
    """

    def __init__(self, env: ContinualMaze, *,
                 rep_dim: int = 16,
                 lr_psi: float = 1e-2,
                 batch_size: int = 128,
                 replay_capacity: int = 1000,
                 max_steps: int = 50,
                 gamma: float = 0.99,
                 entropy_coeff: float = 0.1,
                 episodes_per_update: int = 1,
                 normalize: bool = True):
        self.env = env
        self.rep_dim = rep_dim
        self.lr_psi = lr_psi
        self.batch_size = batch_size
        self.replay_capacity = replay_capacity
        self.max_steps = max_steps
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.episodes_per_update = episodes_per_update
        self.normalize = normalize

        self.goal = env.goal_state
        self.start = env.start_state

        # ψ embeddings — initialised near the goal embedding
        self.psi = np.empty((env.num_states, rep_dim))
        self._init_psi()

        # Replay buffer: list of state-index trajectories
        self.replay: List[List[int]] = []

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_psi(self):
        """Initialise all ψ embeddings near a random goal vector."""
        psi_goal = np.random.randn(self.rep_dim) * 0.1
        self.psi[self.goal] = psi_goal
        if self.normalize:
            self.psi[self.goal] /= np.linalg.norm(self.psi[self.goal]) + 1e-8
        psi_goal_normed = self.psi[self.goal].copy()
        for s in range(self.env.num_states):
            if s != self.goal:
                self.psi[s] = psi_goal_normed + np.random.randn(self.rep_dim) * 0.1
        if self.normalize:
            norms = np.linalg.norm(self.psi, axis=1, keepdims=True) + 1e-8
            self.psi /= norms

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state: int) -> int:
        """Softmax (exploratory) action selection based on ψ(s')·ψ(g)."""
        goal_vec = self.psi[self.goal]
        sims = np.array([
            self.psi[self.env.step(state, a)] @ goal_vec
            for a in range(self.env.NUM_ACTIONS)
        ])
        logits = sims / self.entropy_coeff
        logits -= logits.max()
        exp_l = np.exp(logits)
        probs = exp_l / exp_l.sum()
        return int(np.random.choice(self.env.NUM_ACTIONS, p=probs))

    def eval_action(self, state: int) -> int:
        """Greedy (deterministic) action selection."""
        goal_vec = self.psi[self.goal]
        best_a, best_val = 0, -np.inf
        for a in range(self.env.NUM_ACTIONS):
            ns = self.env.step(state, a)
            val = self.psi[ns] @ goal_vec
            if val > best_val:
                best_val = val
                best_a = a
        return best_a

    # ------------------------------------------------------------------
    # Episode collection
    # ------------------------------------------------------------------

    def collect_episode(self) -> Tuple[List[int], bool]:
        """Run one training episode. Returns (trajectory, reached_goal)."""
        traj = [self.start]
        reached_goal = False
        for _ in range(self.max_steps):
            a = self.select_action(traj[-1])
            ns = self.env.step(traj[-1], a)
            traj.append(ns)
            if self.env.is_goal(ns) or self.env.is_near_goal(ns):
                reached_goal = True
        # Add to replay
        self.replay.append(traj)
        if len(self.replay) > self.replay_capacity:
            self.replay.pop(0)
        return traj, reached_goal

    def run_eval_episode(self) -> Tuple[List[int], bool]:
        """Run one deterministic evaluation episode."""
        traj = [self.start]
        reached_goal = False
        for _ in range(self.max_steps):
            a = self.eval_action(traj[-1])
            ns = self.env.step(traj[-1], a)
            traj.append(ns)
            if self.env.is_goal(ns) or self.env.is_near_goal(ns):
                reached_goal = True
        return traj, reached_goal

    # ------------------------------------------------------------------
    # Contrastive representation update
    # ------------------------------------------------------------------

    def update_representations(self) -> float:
        """Vectorised InfoNCE contrastive update on ψ embeddings.

        Returns the mean negative log-likelihood (loss).
        """
        if len(self.replay) < 2:
            return 0.0

        traj_ids = np.random.choice(len(self.replay), self.batch_size, replace=True)
        s_list, sp_list = [], []
        for idx in traj_ids:
            traj = self.replay[idx]
            if len(traj) < 2:
                continue
            i = np.random.randint(0, len(traj) - 1)
            remaining = len(traj) - i
            w = self.gamma ** np.arange(remaining)
            w /= w.sum()
            j = i + np.random.choice(remaining, p=w)
            s_list.append(traj[i])
            sp_list.append(traj[j])

        if not s_list:
            return 0.0

        s_batch = np.asarray(s_list, dtype=np.int32)
        sp_batch = np.asarray(sp_list, dtype=np.int32)
        B = len(s_batch)

        psi_s = self.psi[s_batch]    # (B, D)
        psi_p = self.psi[sp_batch]   # (B, D)

        # Column-wise softmax
        dots = psi_s @ psi_p.T
        dots -= dots.max(axis=0, keepdims=True)
        exp_d = np.exp(dots)
        P = exp_d / exp_d.sum(axis=0, keepdims=True)

        nll = -np.mean(np.log(np.diag(P) + 1e-12))

        # Gradient updates
        coeff = np.eye(B) - P
        np.add.at(self.psi, s_batch, self.lr_psi * (coeff @ psi_p))
        np.add.at(self.psi, sp_batch, self.lr_psi * (psi_s - P.T @ psi_s))

        if self.normalize:
            norms = np.linalg.norm(self.psi, axis=1, keepdims=True) + 1e-8
            self.psi /= norms

        return nll

    # ------------------------------------------------------------------
    # Snapshot / restore
    # ------------------------------------------------------------------

    def get_psi_snapshot(self) -> np.ndarray:
        """Return a copy of the current ψ embedding table."""
        return self.psi.copy()

    def get_similarity_map(self) -> np.ndarray:
        """Compute ψ(s)·ψ(g) for every state, reshaped to (H, W)."""
        goal_vec = self.psi[self.goal]
        g_norm = np.linalg.norm(goal_vec) + 1e-8
        s_norms = np.linalg.norm(self.psi, axis=1) + 1e-8
        sims = (self.psi @ goal_vec) / (s_norms * g_norm)
        return sims.reshape(self.env.height, self.env.width)

    def clear_replay(self):
        """Empty the replay buffer (useful at phase transitions)."""
        self.replay.clear()
