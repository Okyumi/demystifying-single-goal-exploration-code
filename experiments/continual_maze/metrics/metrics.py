"""
Metrics collection for continual maze experiments.

Tracks:
  1. Success rate (per-phase, rolling)
  2. Path diversity (Jaccard distance, route clustering, entropy)
  3. Trajectory preference (fraction per dominant path)
  4. Adaptation speed (episodes until first success after phase change)
  5. ψ-similarity evolution (snapshots over training)
  6. Representation drift (L2 / cosine distance across phase transitions)
  7. Exploitation ratio (fraction using dominant path)
"""

import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field


def _trajectory_to_cell_set(traj: List[int]) -> frozenset:
    """Convert a trajectory (list of state indices) to a set of visited cells."""
    return frozenset(traj)


def _jaccard_distance(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / len(a | b)


def _cluster_trajectories(cell_sets: List[frozenset],
                          threshold: float = 0.5) -> List[int]:
    """Simple single-linkage clustering of trajectories by Jaccard distance.

    Returns a cluster-id list (same length as cell_sets).
    """
    n = len(cell_sets)
    if n == 0:
        return []
    labels = list(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            if _jaccard_distance(cell_sets[i], cell_sets[j]) < threshold:
                # Merge clusters
                old, new = max(labels[i], labels[j]), min(labels[i], labels[j])
                labels = [new if l == old else l for l in labels]

    # Re-index to 0..K-1
    unique = sorted(set(labels))
    remap = {v: k for k, v in enumerate(unique)}
    return [remap[l] for l in labels]


class MetricsCollector:
    """Collects and stores all experiment metrics.

    Call the `record_*` methods during training; call `finalise()` at the end
    to compute aggregate statistics.
    """

    def __init__(self, num_states: int, maze_shape: Tuple[int, int]):
        self.num_states = num_states
        self.maze_shape = maze_shape

        # Per-episode tracking
        self.episode_success: List[bool] = []
        self.episode_phase: List[int] = []
        self.episode_trajectories: List[List[int]] = []

        # ψ-similarity snapshots: list of (episode, phase, similarity_map)
        self.psi_snapshots: List[Tuple[int, int, np.ndarray]] = []

        # ψ full embedding snapshots at phase transitions
        # list of (phase_idx, direction, psi_copy)
        # direction: "before" or "after"
        self.psi_transition_snapshots: List[Tuple[int, str, np.ndarray]] = []

        # Phase transition episodes (for adaptation speed)
        self.phase_transition_episodes: List[int] = []

    # ------------------------------------------------------------------
    # Recording methods (called during training)
    # ------------------------------------------------------------------

    def record_episode(self, episode: int, phase_idx: int,
                       trajectory: List[int], success: bool):
        self.episode_success.append(success)
        self.episode_phase.append(phase_idx)
        self.episode_trajectories.append(trajectory)

    def record_psi_snapshot(self, episode: int, phase_idx: int,
                            similarity_map: np.ndarray):
        self.psi_snapshots.append((episode, phase_idx, similarity_map.copy()))

    def record_phase_transition(self, episode: int, psi_before: np.ndarray,
                                psi_after: np.ndarray, phase_idx: int):
        self.phase_transition_episodes.append(episode)
        self.psi_transition_snapshots.append((phase_idx, "before", psi_before.copy()))
        self.psi_transition_snapshots.append((phase_idx, "after", psi_after.copy()))

    # ------------------------------------------------------------------
    # Computed metrics
    # ------------------------------------------------------------------

    def success_rate_per_phase(self) -> Dict[int, float]:
        """Mean success rate for each phase."""
        phase_successes: Dict[int, List[bool]] = defaultdict(list)
        for p, s in zip(self.episode_phase, self.episode_success):
            phase_successes[p].append(s)
        return {p: float(np.mean(v)) for p, v in sorted(phase_successes.items())}

    def rolling_success_rate(self, window: int = 50) -> np.ndarray:
        """Smoothed success rate curve."""
        arr = np.array(self.episode_success, dtype=float)
        if len(arr) < window:
            return np.cumsum(arr) / (np.arange(len(arr)) + 1)
        kernel = np.ones(window) / window
        return np.convolve(arr, kernel, mode="same")

    def path_diversity_per_phase(self, cluster_threshold: float = 0.5
                                 ) -> Dict[int, Dict[str, float]]:
        """Per-phase path diversity metrics for *successful* trajectories."""
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

            # Mean pairwise Jaccard distance
            n = len(cell_sets)
            dists = []
            for i in range(n):
                for j in range(i + 1, n):
                    dists.append(_jaccard_distance(cell_sets[i], cell_sets[j]))
            mean_jaccard = float(np.mean(dists)) if dists else 0.0

            # Clustering
            labels = _cluster_trajectories(cell_sets, cluster_threshold)
            num_clusters = len(set(labels))

            # Entropy over cluster distribution
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
        """Per-phase: fraction of successful episodes using each route cluster."""
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
        """Episodes from phase transition to first success, per phase."""
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
        """L2 and cosine drift of ψ across each phase transition."""
        results = []
        # Pair up (before, after) for each transition
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
            # Mean L2 distance
            l2 = float(np.mean(np.linalg.norm(psi_a - psi_b, axis=1)))
            # Mean cosine similarity
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
        """Per-phase fraction of successful episodes using the dominant path."""
        prefs = self.trajectory_preference(cluster_threshold)
        return {p: v["dominant_fraction"] for p, v in prefs.items()}

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise all computed metrics to a JSON-friendly dict."""
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
            "psi_snapshots": [
                {"episode": ep, "phase": ph, "similarity_map": sm.tolist()}
                for ep, ph, sm in self.psi_snapshots
            ],
            "total_episodes": len(self.episode_success),
            "phase_transition_episodes": self.phase_transition_episodes,
        }
