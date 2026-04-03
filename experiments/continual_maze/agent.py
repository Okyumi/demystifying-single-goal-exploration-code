"""
Tabular SGCRL agents for ContinualMaze and StochasticSuccessMaze.

Two agent classes:
1. TabularSGCRLAgent — deterministic maze, success = reaching the goal cell
2. StochasticSGCRLAgent — success depends on approach direction (inherits from above)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any


class TabularSGCRLAgent:
    """Tabular Single-Goal Contrastive RL agent.

    Parameters
    ----------
    env : ContinualMaze or StochasticSuccessMaze
    rep_dim : dimensionality of ψ embeddings
    lr_psi : learning rate for embedding updates
    batch_size : pairs per contrastive update
    replay_capacity : max trajectories in replay buffer
    max_steps : max steps per episode
    gamma : discount for geometric future-state sampling
    entropy_coeff : softmax temperature (lower = greedier)
    episodes_per_update : contrastive updates every N episodes
    normalize : L2-normalize embeddings after each update
    """

    def __init__(self, env, *,
                 rep_dim: int = 16,
                 lr_psi: float = 1e-2,
                 batch_size: int = 128,
                 replay_capacity: int = 1000,
                 max_steps: int = 80,
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

        # ψ embeddings
        self.psi = np.empty((env.num_states, rep_dim))
        self._init_psi()

        # Replay buffer
        self.replay: List[List[int]] = []

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_psi(self):
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
        """Softmax action selection based on ψ(s')·ψ(g)."""
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
        """Greedy action selection."""
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
    # Policy distribution (for adaptation metrics)
    # ------------------------------------------------------------------

    def get_policy_distribution(self) -> np.ndarray:
        """Return π(a|s) for all states as (num_states, NUM_ACTIONS) array."""
        goal_vec = self.psi[self.goal]
        policy = np.zeros((self.env.num_states, self.env.NUM_ACTIONS))
        for s in range(self.env.num_states):
            sims = np.array([
                self.psi[self.env.step(s, a)] @ goal_vec
                for a in range(self.env.NUM_ACTIONS)
            ])
            logits = sims / self.entropy_coeff
            logits -= logits.max()
            exp_l = np.exp(logits)
            policy[s] = exp_l / exp_l.sum()
        return policy

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
        self.replay.append(traj)
        if len(self.replay) > self.replay_capacity:
            self.replay.pop(0)
        return traj, reached_goal

    def run_eval_episode(self) -> Tuple[List[int], bool]:
        """Greedy evaluation episode."""
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
    # Contrastive update
    # ------------------------------------------------------------------

    def update_representations(self) -> float:
        """InfoNCE contrastive update. Returns mean NLL loss."""
        if len(self.replay) < 2:
            return 0.0

        traj_ids = np.random.choice(len(self.replay), self.batch_size,
                                     replace=True)
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

        psi_s = self.psi[s_batch]
        psi_p = self.psi[sp_batch]

        dots = psi_s @ psi_p.T
        dots -= dots.max(axis=0, keepdims=True)
        exp_d = np.exp(dots)
        P = exp_d / exp_d.sum(axis=0, keepdims=True)

        nll = -np.mean(np.log(np.diag(P) + 1e-12))

        coeff = np.eye(B) - P
        np.add.at(self.psi, s_batch, self.lr_psi * (coeff @ psi_p))
        np.add.at(self.psi, sp_batch, self.lr_psi * (psi_s - P.T @ psi_s))

        if self.normalize:
            norms = np.linalg.norm(self.psi, axis=1, keepdims=True) + 1e-8
            self.psi /= norms

        return nll

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    def get_psi_snapshot(self) -> np.ndarray:
        return self.psi.copy()

    def get_similarity_map(self) -> np.ndarray:
        """ψ(s)·ψ(g) for every state, shaped (H, W)."""
        goal_vec = self.psi[self.goal]
        g_norm = np.linalg.norm(goal_vec) + 1e-8
        s_norms = np.linalg.norm(self.psi, axis=1) + 1e-8
        sims = (self.psi @ goal_vec) / (s_norms * g_norm)
        return sims.reshape(self.env.height, self.env.width)

    def clear_replay(self):
        self.replay.clear()


# ---------------------------------------------------------------------------
# StochasticSGCRLAgent
# ---------------------------------------------------------------------------

class StochasticSGCRLAgent(TabularSGCRLAgent):
    """SGCRL agent for StochasticSuccessMaze.

    Same contrastive representation learning, but success is stochastic:
    the agent only gets a "success" signal probabilistically when it reaches
    the goal, depending on the approach direction.

    Only successful trajectories are added to the replay buffer (the agent
    learns from trajectories that led to "actual" success).
    """

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        # Track approach statistics
        self.approach_counts: Dict[str, int] = {}
        self.approach_successes: Dict[str, int] = {}

    def collect_episode(self) -> Tuple[List[int], bool, Optional[str]]:
        """Run one episode. Returns (trajectory, success, approach_direction).

        The agent reaches the goal cell deterministically, but success is
        stochastic — depends on the direction of the final approach step.
        Only successful episodes are added to the replay buffer.
        """
        traj = [self.start]
        reached_goal_cell = False
        success = False
        approach_dir = None

        for step_idx in range(self.max_steps):
            a = self.select_action(traj[-1])
            ns = self.env.step(traj[-1], a)
            traj.append(ns)

            if self.env.is_goal(ns):
                reached_goal_cell = True
                # Check stochastic success
                prev = traj[-2]
                success, approach_dir, prob = \
                    self.env.check_stochastic_success(prev, ns)

                # Track approach stats
                self.approach_counts[approach_dir] = \
                    self.approach_counts.get(approach_dir, 0) + 1
                if success:
                    self.approach_successes[approach_dir] = \
                        self.approach_successes.get(approach_dir, 0) + 1
                break

        # Only add successful episodes to replay
        if success:
            self.replay.append(traj)
            if len(self.replay) > self.replay_capacity:
                self.replay.pop(0)

        return traj, success, approach_dir

    def run_eval_episode(self) -> Tuple[List[int], bool, Optional[str]]:
        """Greedy evaluation with stochastic success."""
        traj = [self.start]
        success = False
        approach_dir = None

        for _ in range(self.max_steps):
            a = self.eval_action(traj[-1])
            ns = self.env.step(traj[-1], a)
            traj.append(ns)
            if self.env.is_goal(ns):
                prev = traj[-2]
                success, approach_dir, _ = \
                    self.env.check_stochastic_success(prev, ns)
                break

        return traj, success, approach_dir

    def get_approach_stats(self) -> Dict[str, Dict[str, Any]]:
        """Return approach direction statistics."""
        stats = {}
        for d in self.approach_counts:
            attempts = self.approach_counts[d]
            successes = self.approach_successes.get(d, 0)
            stats[d] = {
                "attempts": attempts,
                "successes": successes,
                "empirical_rate": successes / attempts if attempts > 0 else 0.0,
            }
        return stats
