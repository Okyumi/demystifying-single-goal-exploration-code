"""
Metrics for continual maze and stochastic-success experiments.

Tracks:
  1. Success rate (per-phase, rolling)
  2. Path diversity (Jaccard distance, route clustering, entropy)
  3. Trajectory preference / dominant path fraction
  4. Adaptation speed (episodes to first success after phase change)
  5. ψ-similarity evolution (snapshots over training)
  6. Representation drift (L2 / cosine across phase transitions)
  7. Exploitation ratio
  8. **Policy adaptation index** — KL divergence of the policy between
     successive snapshots, measuring how much the actor changes its behavior
  9. **Representation adaptation rate** — rolling speed of ψ change
"""

import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass


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
    """Collects all metrics during training."""

    def __init__(self, num_states: int, maze_shape: Tuple[int, int]):
        self.num_states = num_states
        self.maze_shape = maze_shape

        # Per-episode
        self.episode_success: List[bool] = []
        self.episode_phase: List[int] = []
        self.episode_trajectories: List[List[int]] = []

        # ψ-similarity snapshots: (episode, phase, sim_map)
        self.psi_snapshots: List[Tuple[int, int, np.ndarray]] = []

        # Full ψ embedding snapshots: (episode, phase, psi_copy)
        self.psi_full_snapshots: List[Tuple[int, int, np.ndarray]] = []

        # Phase transition data
        self.psi_transition_snapshots: List[Tuple[int, str, np.ndarray]] = []
        self.phase_transition_episodes: List[int] = []

        # Policy snapshots for adaptation index: (episode, phase, policy)
        self.policy_snapshots: List[Tuple[int, int, np.ndarray]] = []

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
        """Record π(a|s) for all states at a given training step."""
        self.policy_snapshots.append((episode, phase_idx, policy.copy()))

    # ------------------------------------------------------------------
    # Standard metrics
    # ------------------------------------------------------------------

    def success_rate_per_phase(self) -> Dict[int, float]:
        phase_successes: Dict[int, List[bool]] = defaultdict(list)
        for p, s in zip(self.episode_phase, self.episode_success):
            phase_successes[p].append(s)
        return {p: float(np.mean(v)) for p, v in sorted(phase_successes.items())}

    def rolling_success_rate(self, window: int = 50) -> np.ndarray:
        arr = np.array(self.episode_success, dtype=float)
        if len(arr) < window:
            return np.cumsum(arr) / (np.arange(len(arr)) + 1)
        kernel = np.ones(window) / window
        return np.convolve(arr, kernel, mode="same")

    def path_diversity_per_phase(self, cluster_threshold: float = 0.5
                                 ) -> Dict[int, Dict[str, float]]:
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

    def trajectory_preference(self, cluster_threshold: float = 0.5
                              ) -> Dict[int, Dict[str, Any]]:
        phase_trajs: Dict[int, List[List[int]]] = defaultdict(list)
        for p, traj, succ in zip(self.episode_phase,
                                 self.episode_trajectories,
                                 self.episode_success):
            if succ:
                phase_trajs[p].append(traj)

        results = {}
        for p, trajs in sorted(phase_trajs.items()):
            if not trajs:
                results[p] = {"cluster_fractions": [], "dominant_fraction": 0.0}
                continue
            cell_sets = [_trajectory_to_cell_set(t) for t in trajs]
            labels = _cluster_trajectories(cell_sets, cluster_threshold)
            counts = np.bincount(labels)
            fracs = (counts / counts.sum()).tolist()
            results[p] = {
                "cluster_fractions": fracs,
                "dominant_fraction": float(max(fracs)),
            }
        return results

    def adaptation_speed(self) -> Dict[int, Optional[int]]:
        results = {}
        transitions = [0] + self.phase_transition_episodes
        for idx, start_ep in enumerate(transitions):
            end_ep = (transitions[idx + 1] if idx + 1 < len(transitions)
                      else len(self.episode_success))
            first_success = None
            for ep in range(start_ep, end_ep):
                if ep < len(self.episode_success) and self.episode_success[ep]:
                    first_success = ep - start_ep
                    break
            results[idx] = first_success
        return results

    def representation_drift(self) -> List[Dict[str, Any]]:
        results = []
        before_map: Dict[int, np.ndarray] = {}
        after_map: Dict[int, np.ndarray] = {}
        for phase_idx, direction, psi in self.psi_transition_snapshots:
            if direction == "before":
                before_map[phase_idx] = psi
            else:
                after_map[phase_idx] = psi

        for phase_idx in sorted(set(before_map) & set(after_map)):
            psi_b = before_map[phase_idx]
            psi_a = after_map[phase_idx]
            l2 = float(np.mean(np.linalg.norm(psi_a - psi_b, axis=1)))
            dots = np.sum(psi_a * psi_b, axis=1)
            norms = (np.linalg.norm(psi_a, axis=1) *
                     np.linalg.norm(psi_b, axis=1) + 1e-12)
            cos_sim = float(np.mean(dots / norms))
            results.append({
                "phase_idx": phase_idx,
                "mean_l2_distance": l2,
                "mean_cosine_similarity": cos_sim,
            })
        return results

    def exploitation_ratio(self, cluster_threshold: float = 0.5
                           ) -> Dict[int, float]:
        prefs = self.trajectory_preference(cluster_threshold)
        return {p: v["dominant_fraction"] for p, v in prefs.items()}

    # ------------------------------------------------------------------
    # Adaptation-specific metrics
    # ------------------------------------------------------------------

    def policy_adaptation_index(self) -> List[Dict[str, Any]]:
        """KL divergence between successive policy snapshots.

        Measures how much the actor's policy changed between two snapshots.
        Large KL at phase transitions = the agent is adapting its behavior.
        Small KL at phase transitions = the agent is stuck on old policy.
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

    def representation_adaptation_rate(self) -> List[Dict[str, Any]]:
        """Rolling rate of ψ change between successive full snapshots.

        Computes mean L2 distance of ψ vectors between consecutive snapshots.
        High rate = representations are actively being reshaped.
        Low rate = representations have settled / agent stopped adapting.
        """
        results = []
        for i in range(1, len(self.psi_full_snapshots)):
            ep_prev, phase_prev, psi_prev = self.psi_full_snapshots[i - 1]
            ep_curr, phase_curr, psi_curr = self.psi_full_snapshots[i]
            l2 = float(np.mean(np.linalg.norm(psi_curr - psi_prev, axis=1)))
            # Also compute cosine distance
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
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success_rate_per_phase": self.success_rate_per_phase(),
            "rolling_success_rate": self.rolling_success_rate().tolist(),
            "path_diversity_per_phase": self.path_diversity_per_phase(),
            "trajectory_preference": {
                str(k): v for k, v in self.trajectory_preference().items()
            },
            "adaptation_speed": {
                str(k): v for k, v in self.adaptation_speed().items()
            },
            "representation_drift": self.representation_drift(),
            "exploitation_ratio": {
                str(k): v for k, v in self.exploitation_ratio().items()
            },
            "policy_adaptation_index": self.policy_adaptation_index(),
            "representation_adaptation_rate":
                self.representation_adaptation_rate(),
            "psi_snapshots": [
                {"episode": ep, "phase": ph, "similarity_map": sm.tolist()}
                for ep, ph, sm in self.psi_snapshots
            ],
            "total_episodes": len(self.episode_success),
            "phase_transition_episodes": self.phase_transition_episodes,
        }
