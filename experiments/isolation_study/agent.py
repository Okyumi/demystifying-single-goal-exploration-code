"""
Self-contained SGCRL agents for the isolation study.

Four agent classes:
1. TabularSGCRLAgent — deterministic maze, all trajectories in replay
2. StochasticSGCRLAgent_AllReplay — stochastic success, ALL trajectories in replay
3. StochasticSGCRLAgent_SuccessOnly — stochastic success, only successes in replay
4. (base class shared logic)

The key difference from the continual_maze agents: the stochastic variants come
in two flavors to test whether replay buffer composition matters for contrastive
learning under stochastic success.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any


class TabularSGCRLAgent:
    """Tabular Single-Goal Contrastive RL agent.

    Parameters
    ----------
    env : ContinualMaze or StochasticSuccessMaze
    rep_dim : dimensionality of psi embeddings
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
                 replay_capacity: int = 2000,
                 max_steps: int = 120,
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

        self.psi = np.empty((env.num_states, rep_dim))
        self._init_psi()

        self.replay: List[List[int]] = []

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

    def select_action(self, state: int) -> int:
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
        goal_vec = self.psi[self.goal]
        best_a, best_val = 0, -np.inf
        for a in range(self.env.NUM_ACTIONS):
            ns = self.env.step(state, a)
            val = self.psi[ns] @ goal_vec
            if val > best_val:
                best_val = val
                best_a = a
        return best_a

    def get_policy_distribution(self) -> np.ndarray:
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

    def collect_episode(self) -> Tuple[List[int], bool]:
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
        traj = [self.start]
        reached_goal = False
        for _ in range(self.max_steps):
            a = self.eval_action(traj[-1])
            ns = self.env.step(traj[-1], a)
            traj.append(ns)
            if self.env.is_goal(ns) or self.env.is_near_goal(ns):
                reached_goal = True
        return traj, reached_goal

    def greedy_rollout(self) -> List[int]:
        """Roll out the greedy policy from start. Returns list of states."""
        traj = [self.start]
        for _ in range(self.max_steps):
            a = self.eval_action(traj[-1])
            ns = self.env.step(traj[-1], a)
            traj.append(ns)
            if self.env.is_goal(ns):
                break
        return traj

    def update_representations(self) -> float:
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

    def get_psi_snapshot(self) -> np.ndarray:
        return self.psi.copy()

    def get_similarity_map(self) -> np.ndarray:
        goal_vec = self.psi[self.goal]
        g_norm = np.linalg.norm(goal_vec) + 1e-8
        s_norms = np.linalg.norm(self.psi, axis=1) + 1e-8
        sims = (self.psi @ goal_vec) / (s_norms * g_norm)
        return sims.reshape(self.env.height, self.env.width)

    def clear_replay(self):
        self.replay.clear()


# ---------------------------------------------------------------------------
# StochasticSGCRLAgent_AllReplay — adds ALL trajectories to replay
# ---------------------------------------------------------------------------

class StochasticSGCRLAgent_AllReplay(TabularSGCRLAgent):
    """SGCRL agent for StochasticSuccessMaze that adds ALL trajectories to replay.

    The contrastive critic learns state-to-future-state reachability from ALL
    transitions, regardless of whether the episode was a "success". Stochastic
    success only affects which episodes count as successful for metrics.
    """

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.approach_counts: Dict[str, int] = {}
        self.approach_successes: Dict[str, int] = {}

    def collect_episode(self) -> Tuple[List[int], bool, Optional[str]]:
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
                prev = traj[-2]
                success, approach_dir, prob = \
                    self.env.check_stochastic_success(prev, ns)
                self.approach_counts[approach_dir] = \
                    self.approach_counts.get(approach_dir, 0) + 1
                if success:
                    self.approach_successes[approach_dir] = \
                        self.approach_successes.get(approach_dir, 0) + 1
                break

        # ALL trajectories go into replay (key difference)
        self.replay.append(traj)
        if len(self.replay) > self.replay_capacity:
            self.replay.pop(0)

        return traj, success, approach_dir

    def run_eval_episode(self) -> Tuple[List[int], bool, Optional[str]]:
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


# ---------------------------------------------------------------------------
# StochasticSGCRLAgent_SuccessOnly — adds only successes to replay
# ---------------------------------------------------------------------------

class StochasticSGCRLAgent_SuccessOnly(TabularSGCRLAgent):
    """SGCRL agent for StochasticSuccessMaze that adds ONLY successful
    trajectories to replay (original behavior from continual_maze).

    This means the contrastive critic only learns from trajectories
    that happened to get a stochastic success signal.
    """

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.approach_counts: Dict[str, int] = {}
        self.approach_successes: Dict[str, int] = {}

    def collect_episode(self) -> Tuple[List[int], bool, Optional[str]]:
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
                prev = traj[-2]
                success, approach_dir, prob = \
                    self.env.check_stochastic_success(prev, ns)
                self.approach_counts[approach_dir] = \
                    self.approach_counts.get(approach_dir, 0) + 1
                if success:
                    self.approach_successes[approach_dir] = \
                        self.approach_successes.get(approach_dir, 0) + 1
                break

        # Only successful episodes go into replay (key difference)
        if success:
            self.replay.append(traj)
            if len(self.replay) > self.replay_capacity:
                self.replay.pop(0)

        return traj, success, approach_dir

    def run_eval_episode(self) -> Tuple[List[int], bool, Optional[str]]:
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
