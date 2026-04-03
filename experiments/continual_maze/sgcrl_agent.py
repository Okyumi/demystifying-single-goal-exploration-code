"""
Tabular SGCRL agent adapted for the ContinualMaze environment.

Ported from tabular_maze.ipynb with modifications:
- Uses ContinualMaze for transitions instead of a global step() function
- Tracks per-phase metrics and phase transitions
- Supports configurable replay buffer handling at phase transitions
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .envs.continual_maze import ContinualMaze


@dataclass
class PhaseSnapshot:
    """Snapshot of agent state at a phase transition boundary."""
    episode: int
    phase_idx: int
    psi: np.ndarray          # copy of embedding table
    psi_goal: np.ndarray     # copy of goal embedding


class SGCRLAgent:
    """Tabular SGCRL agent for ContinualMaze.

    Parameters match the original tabular_maze.ipynb defaults where possible.
    """

    def __init__(
        self,
        env: ContinualMaze,
        rep_dim: int = 16,
        lr_psi: float = 1e-2,
        batch_size: int = 128,
        replay_capacity: int = 1000,
        max_steps: int = 50,
        gamma: float = 0.99,
        entropy_coeff: float = 0.1,
        episodes_per_upd: int = 1,
        eval_freq: int = 1,
        norm: bool = True,
        clear_replay_on_phase_change: bool = False,
        seed: Optional[int] = None,
    ):
        self.env = env
        self.rep_dim = rep_dim
        self.lr_psi = lr_psi
        self.batch_size = batch_size
        self.replay_capacity = replay_capacity
        self.max_steps = max_steps
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.episodes_per_upd = episodes_per_upd
        self.eval_freq = eval_freq
        self.norm = norm
        self.clear_replay_on_phase_change = clear_replay_on_phase_change

        if seed is not None:
            np.random.seed(seed)

        # Number of states / actions from environment
        self.n_states = env.num_states
        self.n_actions = env.NUM_ACTIONS
        self.goal = env.goal_state
        self.start = env.start_state

        # ---- Initialise psi embeddings (same scheme as notebook) ----
        self.psi = np.empty((self.n_states, rep_dim))
        self.psi_goal = np.random.randn(rep_dim) * 0.1
        self.psi[self.goal] = self.psi_goal.copy()
        if self.norm:
            n = np.linalg.norm(self.psi[self.goal]) + 1e-8
            self.psi[self.goal] /= n
            self.psi_goal = self.psi[self.goal].copy()

        for s in range(self.n_states):
            if s != self.goal:
                self.psi[s] = self.psi_goal + np.random.randn(rep_dim) * 0.1
        if self.norm:
            norms = np.linalg.norm(self.psi, axis=1, keepdims=True) + 1e-8
            self.psi /= norms

        # Replay buffer
        self.replay: List[List[int]] = []

        # ---- Tracking / metrics ----
        # Per-episode
        self.train_successes: List[int] = []
        self.eval_successes: List[int] = []
        self.eval_trajectories: List[List[int]] = []
        self.train_trajectories: List[List[int]] = []
        self.loss_history: List[float] = []
        self.phase_at_episode: List[int] = []

        # Snapshots at evaluation points
        self.psi_snapshots: List[Tuple[int, int, np.ndarray]] = []
        # (episode, phase_idx, psi copy)

        # Phase transition snapshots
        self.phase_snapshots: List[PhaseSnapshot] = []

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state: int) -> int:
        """Softmax action selection: p(a) ∝ exp(ψ(s')·ψ(g) / τ)."""
        goal_vec = self.psi[self.goal]
        sims = np.empty(self.n_actions)
        for a in range(self.n_actions):
            ns = self.env.step(state, a)
            sims[a] = self.psi[ns] @ goal_vec
        logits = sims / self.entropy_coeff
        logits -= logits.max()
        exp_l = np.exp(logits)
        probs = exp_l / exp_l.sum()
        return int(np.random.choice(self.n_actions, p=probs))

    def eval_action(self, state: int) -> int:
        """Greedy action: argmax_a ψ(s')·ψ(g)."""
        goal_vec = self.psi[self.goal]
        best_a, best_v = 0, -np.inf
        for a in range(self.n_actions):
            ns = self.env.step(state, a)
            v = self.psi[ns] @ goal_vec
            if v > best_v:
                best_v = v
                best_a = a
        return best_a

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def collect_episode(self) -> Tuple[List[int], bool]:
        """Run one training episode; return (trajectory, reached_goal)."""
        traj = [self.start]
        reached = False
        for _ in range(self.max_steps):
            a = self.select_action(traj[-1])
            ns = self.env.step(traj[-1], a)
            traj.append(ns)
            if ns == self.goal or self.env.is_near_goal(ns):
                reached = True
        self.replay.append(traj)
        if len(self.replay) > self.replay_capacity:
            self.replay.pop(0)
        return traj, reached

    def run_eval_episode(self) -> Tuple[List[int], bool]:
        """Greedy evaluation episode."""
        traj = [self.start]
        reached = False
        for _ in range(self.max_steps):
            a = self.eval_action(traj[-1])
            ns = self.env.step(traj[-1], a)
            traj.append(ns)
            if ns == self.goal or self.env.is_near_goal(ns):
                reached = True
        return traj, reached

    # ------------------------------------------------------------------
    # Contrastive update (vectorised InfoNCE)
    # ------------------------------------------------------------------

    def update_representations(self) -> float:
        """One contrastive update step.  Returns NLL loss."""
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
            j = i + int(np.random.choice(remaining, p=w))
            s_list.append(traj[i])
            sp_list.append(traj[j])

        if not s_list:
            return 0.0

        s_batch = np.asarray(s_list, dtype=np.int32)
        sp_batch = np.asarray(sp_list, dtype=np.int32)
        B = len(s_batch)

        psi_s = self.psi[s_batch]
        psi_p = self.psi[sp_batch]

        dots = psi_s @ psi_p.T
        dots -= dots.max(axis=0, keepdims=True)
        exp_logits = np.exp(dots)
        P = exp_logits / exp_logits.sum(axis=0, keepdims=True)

        diag_P = np.diag(P)
        nll = -np.mean(np.log(diag_P + 1e-12))

        coeff = np.eye(B) - P
        np.add.at(self.psi, s_batch, self.lr_psi * (coeff @ psi_p))

        expected_anchor = P.T @ psi_s
        np.add.at(self.psi, sp_batch, self.lr_psi * (psi_s - expected_anchor))

        if self.norm:
            norms = np.linalg.norm(self.psi, axis=1, keepdims=True) + 1e-8
            self.psi /= norms

        return nll

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, verbose: bool = True) -> Dict:
        """Run full training across all phases.

        Returns a dict with collected metrics.
        """
        total_eps = self.env.total_num_episodes()
        iterator = range(total_eps)
        if verbose:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Training")

        # Take initial snapshot
        self.psi_snapshots.append(
            (0, self.env.phase_idx, self.psi.copy())
        )
        self.phase_snapshots.append(PhaseSnapshot(
            episode=0, phase_idx=0,
            psi=self.psi.copy(), psi_goal=self.psi_goal.copy(),
        ))

        for ep in iterator:
            self.phase_at_episode.append(self.env.phase_idx)

            # Collect training episode
            traj, reached = self.collect_episode()
            self.train_successes.append(int(reached))
            self.train_trajectories.append(traj)

            # Contrastive update
            if ep % self.episodes_per_upd == 0:
                loss = self.update_representations()
                self.loss_history.append(loss)

            # Evaluation
            if ep % self.eval_freq == 0:
                eval_traj, eval_reached = self.run_eval_episode()
                self.eval_successes.append(int(eval_reached))
                self.eval_trajectories.append(eval_traj)

                # Save psi snapshot every 50 episodes
                if ep % 50 == 0:
                    self.psi_snapshots.append(
                        (ep, self.env.phase_idx, self.psi.copy())
                    )

            # Advance environment episode counter & detect phase change
            phase_changed = self.env.advance_episode()
            if phase_changed:
                # Snapshot after transition
                self.phase_snapshots.append(PhaseSnapshot(
                    episode=ep + 1,
                    phase_idx=self.env.phase_idx,
                    psi=self.psi.copy(),
                    psi_goal=self.psi_goal.copy(),
                ))
                if self.clear_replay_on_phase_change:
                    self.replay.clear()
                if verbose:
                    print(f"\n>> Phase transition at ep {ep+1}: "
                          f"now phase {self.env.phase_idx} "
                          f"({self.env.current_phase.name})")

        # Final snapshot
        self.psi_snapshots.append(
            (total_eps, self.env.phase_idx, self.psi.copy())
        )

        return self._collect_results()

    def _collect_results(self) -> Dict:
        """Package all tracked data into a results dict."""
        return {
            "train_successes": self.train_successes,
            "eval_successes": self.eval_successes,
            "eval_trajectories": self.eval_trajectories,
            "train_trajectories": self.train_trajectories,
            "loss_history": self.loss_history,
            "phase_at_episode": self.phase_at_episode,
            "phase_log": self.env.phase_log,
            "psi_snapshots": self.psi_snapshots,
            "phase_snapshots": self.phase_snapshots,
        }
