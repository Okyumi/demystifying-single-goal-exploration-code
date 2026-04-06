"""
Metrics for the isolation study — 9 metrics (M1-M9).

Each metric has a detailed docstring covering:
  - What it measures
  - Why it's useful
  - How to interpret it
  - What behavior it captures
"""

import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trajectory_to_cell_set(traj: List[int]) -> frozenset:
    return frozenset(traj)


def _jaccard_distance(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / len(a | b)


def _cluster_trajectories(cell_sets: List[frozenset],
                          threshold: float = 0.5) -> List[int]:
    """Single-linkage clustering by Jaccard distance."""
    n = len(cell_sets)
    if n == 0:
        return []
    labels = list(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if _jaccard_distance(cell_sets[i], cell_sets[j]) < threshold:
                old, new = max(labels[i], labels[j]), min(labels[i], labels[j])
                labels = [new if l == old else l for l in labels]
    unique = sorted(set(labels))
    remap = {v: k for k, v in enumerate(unique)}
    return [remap[l] for l in labels]


def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Mean KL(p || q) across all states. p, q shape: (S, A)."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    kl_per_state = np.sum(p * np.log(p / q), axis=1)
    return float(np.mean(kl_per_state))


# ---------------------------------------------------------------------------
# MetricsCollector
# ---------------------------------------------------------------------------

class MetricsCollector:
    """Collects all 9 metrics (M1-M9) during training."""

    def __init__(self, num_states: int, maze_shape: Tuple[int, int]):
        self.num_states = num_states
        self.maze_shape = maze_shape

        self.episode_success: List[bool] = []
        self.episode_phase: List[int] = []
        self.episode_trajectories: List[List[int]] = []

        self.psi_snapshots: List[Tuple[int, int, np.ndarray]] = []
        self.psi_full_snapshots: List[Tuple[int, int, np.ndarray]] = []
        self.psi_transition_snapshots: List[Tuple[int, str, np.ndarray]] = []
        self.phase_transition_episodes: List[int] = []
        self.policy_snapshots: List[Tuple[int, int, np.ndarray]] = []

        # Greedy trajectory snapshots for M8
        self.greedy_trajectory_snapshots: List[Tuple[int, int, List[int]]] = []

        # Approach direction data for M6/M7
        self.approach_history: List[Tuple[int, Optional[str], bool]] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_episode(self, episode: int, phase_idx: int,
                       trajectory: List[int], success: bool):
        self.episode_success.append(success)
        self.episode_phase.append(phase_idx)
        self.episode_trajectories.append(trajectory)

    def record_psi_snapshot(self, episode: int, phase_idx: int,
                            similarity_map: np.ndarray):
        self.psi_snapshots.append((episode, phase_idx, similarity_map.copy()))

    def record_psi_full_snapshot(self, episode: int, phase_idx: int,
                                 psi: np.ndarray):
        self.psi_full_snapshots.append((episode, phase_idx, psi.copy()))

    def record_phase_transition(self, episode: int, psi_before: np.ndarray,
                                psi_after: np.ndarray, phase_idx: int):
        self.phase_transition_episodes.append(episode)
        self.psi_transition_snapshots.append(
            (phase_idx, "before", psi_before.copy()))
        self.psi_transition_snapshots.append(
            (phase_idx, "after", psi_after.copy()))

    def record_policy_snapshot(self, episode: int, phase_idx: int,
                               policy: np.ndarray):
        self.policy_snapshots.append((episode, phase_idx, policy.copy()))

    def record_greedy_trajectory(self, episode: int, phase_idx: int,
                                  trajectory: List[int]):
        self.greedy_trajectory_snapshots.append(
            (episode, phase_idx, list(trajectory)))

    def record_approach(self, episode: int, direction: Optional[str],
                        success: bool):
        self.approach_history.append((episode, direction, success))

    # ------------------------------------------------------------------
    # M1: Success Rate (rolling window=50)
    # ------------------------------------------------------------------

    def m1_rolling_success_rate(self, window: int = 50) -> np.ndarray:
        """M1: Success Rate (rolling window).

        What: Fraction of episodes reaching the goal (deterministic) or
              achieving stochastic success, computed over a sliding window.
        Why: Basic performance measure. The most fundamental metric for
             evaluating whether the agent is learning.
        Interpret: Higher = better. Drops after phase transitions indicate
                   the cost of adapting to new dynamics. Plateaus indicate
                   convergence. A flat zero means the agent is stuck.
        Captures: Learning speed, adaptation speed, steady-state performance.
        """
        arr = np.array(self.episode_success, dtype=float)
        if len(arr) < window:
            return np.cumsum(arr) / (np.arange(len(arr)) + 1)
        kernel = np.ones(window) / window
        return np.convolve(arr, kernel, mode="same")

    def m1_success_rate_per_phase(self) -> Dict[int, float]:
        """Per-phase variant of M1."""
        phase_successes: Dict[int, List[bool]] = defaultdict(list)
        for p, s in zip(self.episode_phase, self.episode_success):
            phase_successes[p].append(s)
        return {p: float(np.mean(v)) for p, v in sorted(phase_successes.items())}

    # ------------------------------------------------------------------
    # M2: Path Diversity (Jaccard + clustering)
    # ------------------------------------------------------------------

    def m2_path_diversity(self, cluster_threshold: float = 0.5
                          ) -> Dict[int, Dict[str, float]]:
        """M2: Path Diversity (Jaccard distance + route clustering + entropy).

        What: Mean pairwise Jaccard distance between successful trajectories;
              number of distinct route clusters (single-linkage, threshold=0.5);
              Shannon entropy over cluster distribution.
        Why: Low diversity + high exploitation ratio = overexploitation. This
             metric reveals whether the agent explores multiple strategies or
             locks onto a single route.
        Interpret: High mean_jaccard = diverse paths. High entropy = even
                   distribution across routes. Low values = one dominant path.
                   Many clusters = many distinct routes discovered.
        Captures: Whether the agent explores multiple strategies or locks
                  onto one. Route variety and exploration breadth.
        """
        phase_trajs: Dict[int, List[List[int]]] = defaultdict(list)
        for p, traj, succ in zip(self.episode_phase,
                                 self.episode_trajectories,
                                 self.episode_success):
            if succ:
                phase_trajs[p].append(traj)

        results = {}
        for p, trajs in sorted(phase_trajs.items()):
            if len(trajs) < 2:
                results[p] = {
                    "mean_jaccard": 0.0,
                    "num_clusters": 1 if trajs else 0,
                    "entropy": 0.0,
                    "num_successful": len(trajs),
                }
                continue

            cell_sets = [_trajectory_to_cell_set(t) for t in trajs]
            n = len(cell_sets)
            dists = []
            for i in range(n):
                for j in range(i + 1, n):
                    dists.append(_jaccard_distance(cell_sets[i], cell_sets[j]))
            mean_jaccard = float(np.mean(dists)) if dists else 0.0

            labels = _cluster_trajectories(cell_sets, cluster_threshold)
            num_clusters = len(set(labels))
            counts = np.bincount(labels)
            probs = counts / counts.sum()
            entropy = float(-np.sum(probs * np.log(probs + 1e-12)))

            results[p] = {
                "mean_jaccard": mean_jaccard,
                "num_clusters": num_clusters,
                "entropy": entropy,
                "num_successful": len(trajs),
            }
        return results

    # ------------------------------------------------------------------
    # M3: Exploitation Ratio
    # ------------------------------------------------------------------

    def m3_exploitation_ratio(self, cluster_threshold: float = 0.5
                              ) -> Dict[int, float]:
        """M3: Exploitation Ratio.

        What: Fraction of successful trajectories in the largest route cluster.
        Why: Directly measures how much the agent relies on a single route.
             Combined with M2, reveals the full picture of route preference.
        Interpret: Close to 1.0 = extreme overexploitation (one dominant path).
                   0.5 = two balanced routes. Lower = more distributed usage.
        Captures: Route preference, overexploitation, whether agent has
                  discovered multiple viable paths.
        """
        phase_trajs: Dict[int, List[List[int]]] = defaultdict(list)
        for p, traj, succ in zip(self.episode_phase,
                                 self.episode_trajectories,
                                 self.episode_success):
            if succ:
                phase_trajs[p].append(traj)

        results = {}
        for p, trajs in sorted(phase_trajs.items()):
            if not trajs:
                results[p] = 0.0
                continue
            cell_sets = [_trajectory_to_cell_set(t) for t in trajs]
            labels = _cluster_trajectories(cell_sets, cluster_threshold)
            counts = np.bincount(labels)
            results[p] = float(np.max(counts) / counts.sum())
        return results

    # ------------------------------------------------------------------
    # M4: Policy Adaptation Index (KL divergence)
    # ------------------------------------------------------------------

    def m4_policy_adaptation_index(self) -> List[Dict[str, Any]]:
        """M4: Policy Adaptation Index (KL divergence).

        What: KL(pi_t || pi_{t-delta}) — KL divergence between the current
              policy and a snapshot from delta episodes ago, computed over
              all states.
        Why: Measures how much the actor's behavior changed between
             snapshots. Essential for detecting whether the agent adapts
             after dynamics changes.
        Interpret: Spikes at phase transitions = actor is adapting.
                   Flat within phases = stable policy.
                   Flat at transitions = stuck on old policy (bad).
        Captures: Policy adaptation to dynamics changes, learning stability.
        """
        results = []
        for i in range(1, len(self.policy_snapshots)):
            ep_prev, phase_prev, pol_prev = self.policy_snapshots[i - 1]
            ep_curr, phase_curr, pol_curr = self.policy_snapshots[i]
            kl = _kl_divergence(pol_curr, pol_prev)
            results.append({
                "episode_from": int(ep_prev),
                "episode_to": int(ep_curr),
                "phase_from": int(phase_prev),
                "phase_to": int(phase_curr),
                "kl_divergence": kl,
                "is_phase_transition": phase_prev != phase_curr,
            })
        return results

    # ------------------------------------------------------------------
    # M5: Representation Adaptation Rate (psi L2 velocity)
    # ------------------------------------------------------------------

    def m5_representation_adaptation_rate(self) -> List[Dict[str, Any]]:
        """M5: Representation Adaptation Rate (psi L2 velocity).

        What: Mean L2 distance between psi(s) at time t and psi(s) at
              time t-delta, averaged over all states.
        Why: Measures how fast the critic is reshaping the representation
             landscape. Complementary to M4 which measures policy change.
        Interpret: High early = active learning. Decreasing = settling.
                   Spike at phase transition = re-learning (good).
                   No spike at transition = stuck representation.
        Captures: Representation plasticity, critic adaptation speed.
        """
        results = []
        for i in range(1, len(self.psi_full_snapshots)):
            ep_prev, phase_prev, psi_prev = self.psi_full_snapshots[i - 1]
            ep_curr, phase_curr, psi_curr = self.psi_full_snapshots[i]
            l2 = float(np.mean(np.linalg.norm(psi_curr - psi_prev, axis=1)))
            dots = np.sum(psi_curr * psi_prev, axis=1)
            norms = (np.linalg.norm(psi_curr, axis=1) *
                     np.linalg.norm(psi_prev, axis=1) + 1e-12)
            cos_dist = float(1.0 - np.mean(dots / norms))
            results.append({
                "episode_from": int(ep_prev),
                "episode_to": int(ep_curr),
                "phase_from": int(phase_prev),
                "phase_to": int(phase_curr),
                "mean_l2_rate": l2,
                "mean_cosine_distance": cos_dist,
            })
        return results

    # ------------------------------------------------------------------
    # M6: Suboptimality Gap (stochastic experiments)
    # ------------------------------------------------------------------

    def m6_suboptimality_gap(self, true_probs: Dict[str, float],
                              window: int = 50) -> np.ndarray:
        """M6: Suboptimality Gap (stochastic experiments).

        What: Difference between the optimal approach direction's success rate
              and the agent's actual approach direction's success rate, computed
              over a rolling window.
        Why: Directly measures whether the agent found the best route. A gap
             of zero means the agent is optimally exploiting the highest-
             probability direction.
        Interpret: 0 = optimal (agent uses the best direction consistently).
                   >0 = agent is using a worse direction.
                   gap = p_best - p_agent_route.
        Captures: Whether agent converges to optimal strategy under
                  stochastic success, overexploitation cost.
        """
        if not self.approach_history:
            return np.array([])

        best_prob = max(v for k, v in true_probs.items() if k != "stay")
        # For each episode, compute the true prob of the approach used
        ep_probs = []
        for ep, direction, success in self.approach_history:
            if direction is not None:
                ep_probs.append(true_probs.get(direction, 0.0))
            else:
                ep_probs.append(0.0)

        gaps = best_prob - np.array(ep_probs)
        if len(gaps) < window:
            return np.cumsum(gaps) / (np.arange(len(gaps)) + 1)
        kernel = np.ones(window) / window
        return np.convolve(gaps, kernel, mode="same")

    # ------------------------------------------------------------------
    # M7: Approach Direction Entropy (stochastic experiments)
    # ------------------------------------------------------------------

    def m7_approach_direction_entropy(self, window: int = 100) -> np.ndarray:
        """M7: Approach Direction Entropy (stochastic experiments).

        What: Shannon entropy over the distribution of approach directions
              used by the agent, computed over a rolling window.
        Why: High entropy = trying many directions (exploration).
             Low entropy = locked onto one direction (exploitation).
        Interpret: Maximum entropy (~log(4) = 1.386) = uniform exploration
                   of all 4 directions. 0 = always same direction.
                   Decreasing entropy over time = convergence to a strategy.
        Captures: Exploration breadth, sensitivity to stochastic success,
                  whether the agent systematically tries alternatives.
        """
        if not self.approach_history:
            return np.array([])

        dir_labels = ["from_above", "from_below", "from_left", "from_right"]
        dir_to_idx = {d: i for i, d in enumerate(dir_labels)}

        entropies = []
        for t in range(len(self.approach_history)):
            start = max(0, t - window + 1)
            counts = np.zeros(len(dir_labels))
            for i in range(start, t + 1):
                _, d, _ = self.approach_history[i]
                if d in dir_to_idx:
                    counts[dir_to_idx[d]] += 1
            total = counts.sum()
            if total == 0:
                entropies.append(0.0)
            else:
                probs = counts / total
                entropy = -np.sum(probs * np.log(probs + 1e-12))
                entropies.append(entropy)

        return np.array(entropies)

    # ------------------------------------------------------------------
    # M8: Representation-Route Alignment
    # ------------------------------------------------------------------

    def m8_representation_route_alignment(self) -> List[Dict[str, Any]]:
        """M8: Representation-Route Alignment.

        What: For each snapshot, compute the mean psi-similarity of states
              ON the greedy policy trajectory vs states NOT on it. The
              alignment score is the difference (on_route - off_route).
        Why: Shows whether the psi-similarity landscape has formed a "ridge"
             along the current route. A strong ridge means the representation
             is tightly locked to a specific path.
        Interpret: Large positive gap = strong psi-trace along the route
                   (exploitation, representation locked to path).
                   Small gap = flat landscape (exploration or no learning).
                   Negative gap = representation favors alternative paths.
        Captures: How tightly the representation is locked to a specific
                  route, representation-behavior coupling.
        """
        results = []
        for (ep_psi, phase_psi, sim_map), (ep_traj, phase_traj, traj) in zip(
                self.psi_snapshots, self.greedy_trajectory_snapshots):
            h, w = sim_map.shape
            route_states = set(traj)
            on_route_sims = []
            off_route_sims = []
            for s in range(h * w):
                r, c = np.unravel_index(s, (h, w))
                val = sim_map[r, c]
                if np.isnan(val):
                    continue
                if s in route_states:
                    on_route_sims.append(val)
                else:
                    off_route_sims.append(val)

            mean_on = float(np.mean(on_route_sims)) if on_route_sims else 0.0
            mean_off = float(np.mean(off_route_sims)) if off_route_sims else 0.0
            results.append({
                "episode": int(ep_psi),
                "phase": int(phase_psi),
                "mean_on_route": mean_on,
                "mean_off_route": mean_off,
                "alignment_gap": mean_on - mean_off,
                "route_length": len(route_states),
            })
        return results

    # ------------------------------------------------------------------
    # M9: Phase-Transition Recovery Time
    # ------------------------------------------------------------------

    def m9_phase_transition_recovery(self, threshold_frac: float = 0.5,
                                      window: int = 50) -> List[Dict[str, Any]]:
        """M9: Phase-Transition Recovery Time.

        What: Number of episodes after a phase transition until the rolling
              success rate recovers to X% of the pre-transition level
              (X=50% by default).
        Why: Measures adaptation speed in concrete terms. More intuitive
             than M4/M5 for understanding practical performance impact.
        Interpret: Short recovery = fast adaptation. Long = slow adaptation.
                   Infinite (None) = never recovered.
        Captures: Agent's ability to recover from dynamics changes,
                  representation plasticity in practical terms.
        """
        if not self.phase_transition_episodes:
            return []

        rolling = self.m1_rolling_success_rate(window)
        results = []
        for trans_ep in self.phase_transition_episodes:
            # Pre-transition level: average of window before transition
            pre_start = max(0, trans_ep - window)
            pre_level = float(np.mean(rolling[pre_start:trans_ep])) if trans_ep > 0 else 0.0
            target = pre_level * threshold_frac

            # Find recovery point
            recovery_ep = None
            search_end = min(len(rolling), trans_ep + 2000)
            for ep in range(trans_ep, search_end):
                if rolling[ep] >= target:
                    recovery_ep = ep - trans_ep
                    break

            results.append({
                "transition_episode": int(trans_ep),
                "pre_transition_level": pre_level,
                "target_level": target,
                "recovery_episodes": recovery_ep,
            })
        return results

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self, true_probs: Optional[Dict[str, float]] = None
                ) -> Dict[str, Any]:
        result = {
            "m1_rolling_success_rate": self.m1_rolling_success_rate().tolist(),
            "m1_success_rate_per_phase": self.m1_success_rate_per_phase(),
            "m2_path_diversity": {
                str(k): v for k, v in self.m2_path_diversity().items()
            },
            "m3_exploitation_ratio": {
                str(k): v for k, v in self.m3_exploitation_ratio().items()
            },
            "m4_policy_adaptation_index": self.m4_policy_adaptation_index(),
            "m5_representation_adaptation_rate":
                self.m5_representation_adaptation_rate(),
            "m8_representation_route_alignment":
                self.m8_representation_route_alignment(),
            "m9_phase_transition_recovery":
                self.m9_phase_transition_recovery(),
            "psi_snapshots": [
                {"episode": ep, "phase": ph, "similarity_map": sm.tolist()}
                for ep, ph, sm in self.psi_snapshots
            ],
            "total_episodes": len(self.episode_success),
            "phase_transition_episodes": self.phase_transition_episodes,
        }
        if true_probs:
            result["m6_suboptimality_gap"] = \
                self.m6_suboptimality_gap(true_probs).tolist()
            result["m7_approach_direction_entropy"] = \
                self.m7_approach_direction_entropy().tolist()
        return result
