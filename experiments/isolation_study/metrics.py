"""
Metrics for the isolation study — 9 metrics (M1-M9).

HPC additions over the original version:
  - MetricsCollector.save(output_dir)  — writes each metric to its own file
  - Streaming M1 via _RollingSuccessRate — O(1) memory, no stored bool list
  - m0_episode_log  — list of dicts appended per episode; serialised to
    episode_log.jsonl (one JSON object per line, appendable mid-run)
  - get_state() / from_state() — for checkpoint resume

All 9 original metrics (M1-M9) are preserved unchanged.
"""

from __future__ import annotations

import json
import os
import numpy as np
from collections import defaultdict, deque
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
                old = max(labels[i], labels[j])
                new = min(labels[i], labels[j])
                labels = [new if l == old else l for l in labels]
    unique = sorted(set(labels))
    remap = {v: k for k, v in enumerate(unique)}
    return [remap[l] for l in labels]


def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Mean KL(p || q) across all states.  p, q shape: (S, A)."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.mean(np.sum(p * np.log(p / q), axis=1)))


class _RollingSuccessRate:
    """Streaming rolling-window success rate (O(window) memory).

    Maintains a deque of the last `window` booleans and tracks the running
    sum for O(1) mean computation.
    """

    def __init__(self, window: int = 50):
        self.window = window
        self._buf: deque = deque(maxlen=window)
        self._total_episodes = 0
        self._running_sum = 0.0
        # For full array output we keep a compact list
        self._rates: List[float] = []

    def update(self, success: bool):
        val = float(success)
        if len(self._buf) == self.window:
            self._running_sum -= self._buf[0]
        self._buf.append(val)
        self._running_sum += val
        self._total_episodes += 1
        if self._total_episodes <= self.window:
            # Early episodes: use expanding mean
            self._rates.append(self._running_sum / len(self._buf))
        else:
            self._rates.append(self._running_sum / self.window)

    def current(self) -> float:
        if not self._buf:
            return 0.0
        return self._running_sum / len(self._buf)

    def as_array(self) -> np.ndarray:
        return np.array(self._rates, dtype=np.float32)

    def get_state(self) -> Dict[str, Any]:
        return {
            "window": self.window,
            "buf": list(self._buf),
            "total_episodes": self._total_episodes,
            "running_sum": self._running_sum,
            "rates": self._rates,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "_RollingSuccessRate":
        obj = cls(window=state["window"])
        obj._buf = deque(state["buf"], maxlen=state["window"])
        obj._total_episodes = state["total_episodes"]
        obj._running_sum = state["running_sum"]
        obj._rates = state["rates"]
        return obj


# ---------------------------------------------------------------------------
# MetricsCollector
# ---------------------------------------------------------------------------

class MetricsCollector:
    """Collects all 9 metrics (M1-M9) during training.

    HPC additions
    -------------
    * Streaming M1 via _RollingSuccessRate (no full bool list retained in RAM)
    * m0_episode_log: list of lightweight dicts, one per episode
    * save(output_dir): write individual .npy / .json files
    * get_state() / from_state(): checkpoint/resume
    """

    def __init__(self, num_states: int, maze_shape: Tuple[int, int],
                 m1_window: int = 50):
        self.num_states = num_states
        self.maze_shape = maze_shape

        # Streaming M1
        self._rolling = _RollingSuccessRate(window=m1_window)

        # Full episode lists (kept for M2/M3/M8/M9 which need trajectory data)
        self.episode_success:      List[bool]        = []
        self.episode_phase:        List[int]         = []
        self.episode_trajectories: List[List[int]]   = []

        # Snapshot lists
        self.psi_snapshots:             List[Tuple[int, int, np.ndarray]] = []
        self.psi_full_snapshots:        List[Tuple[int, int, np.ndarray]] = []
        self.psi_transition_snapshots:  List[Tuple[int, str, np.ndarray]] = []
        self.phase_transition_episodes: List[int]                          = []
        self.policy_snapshots:          List[Tuple[int, int, np.ndarray]] = []
        self.greedy_trajectory_snapshots: List[Tuple[int, int, List[int]]] = []

        # Approach direction data (Study A, stochastic)
        self.approach_history: List[Tuple[int, Optional[str], bool]] = []

        # M0: lightweight per-episode log (for append-mode JSONL writing)
        self.m0_episode_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_episode(self, episode: int, phase_idx: int,
                       trajectory: List[int], success: bool,
                       extra: Optional[Dict[str, Any]] = None):
        self.episode_success.append(success)
        self.episode_phase.append(phase_idx)
        self.episode_trajectories.append(trajectory)
        self._rolling.update(success)

        log_entry: Dict[str, Any] = {
            "episode":  episode,
            "phase":    phase_idx,
            "success":  bool(success),
            "steps":    len(trajectory) - 1,
        }
        if extra:
            log_entry.update(extra)
        self.m0_episode_log.append(log_entry)

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
            (phase_idx, "after",  psi_after.copy()))

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
    # M1: Success Rate
    # ------------------------------------------------------------------

    def m1_rolling_success_rate(self, window: int = 50) -> np.ndarray:
        """M1: Rolling success rate.

        Uses streaming _RollingSuccessRate if the window matches; otherwise
        falls back to convolving the stored bool array (for arbitrary windows).

        What: Fraction of episodes reaching the goal / achieving stochastic
              success, computed over a sliding window.
        Why:  Basic performance measure.
        Interpret: Higher = better.  Drops after phase transitions indicate
                   adaptation cost.  Plateaus = convergence.
        Captures: Learning speed, adaptation speed, steady-state performance.
        """
        if window == self._rolling.window:
            return self._rolling.as_array()
        # Fallback for arbitrary window
        arr = np.array(self.episode_success, dtype=float)
        if len(arr) == 0:
            return arr
        if len(arr) < window:
            return np.cumsum(arr) / (np.arange(len(arr)) + 1)
        kernel = np.ones(window) / window
        return np.convolve(arr, kernel, mode="same")

    def m1_rolling_current(self) -> float:
        """Current (most recent) rolling success rate value."""
        return self._rolling.current()

    def m1_success_rate_per_phase(self) -> Dict[int, float]:
        """Per-phase variant of M1."""
        phase_successes: Dict[int, List[bool]] = defaultdict(list)
        for p, s in zip(self.episode_phase, self.episode_success):
            phase_successes[p].append(s)
        return {p: float(np.mean(v)) for p, v in sorted(phase_successes.items())}

    # ------------------------------------------------------------------
    # M2: Path Diversity
    # ------------------------------------------------------------------

    def m2_path_diversity(self, cluster_threshold: float = 0.5,
                           max_sample: int = 200) -> Dict[int, Dict[str, float]]:
        """M2: Path Diversity (Jaccard distance + route clustering + entropy).

        What: Mean pairwise Jaccard distance between successful trajectories;
              number of distinct route clusters; Shannon entropy over clusters.
        Why:  Low diversity + high exploitation = overexploitation.
        Interpret: High mean_jaccard = diverse paths.  High entropy = even
                   distribution.  Many clusters = many distinct routes found.
        Captures: Route variety, exploration breadth, overexploitation.
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
                    "mean_jaccard":   0.0,
                    "num_clusters":   1 if trajs else 0,
                    "entropy":        0.0,
                    "num_successful": len(trajs),
                }
                continue

            if len(trajs) > max_sample:
                idx     = np.random.choice(len(trajs), max_sample, replace=False)
                sampled = [trajs[i] for i in idx]
            else:
                sampled = trajs

            cell_sets = [_trajectory_to_cell_set(t) for t in sampled]
            n = len(cell_sets)
            dists = [_jaccard_distance(cell_sets[i], cell_sets[j])
                     for i in range(n) for j in range(i + 1, n)]
            mean_jaccard = float(np.mean(dists)) if dists else 0.0

            labels      = _cluster_trajectories(cell_sets, cluster_threshold)
            num_clusters = len(set(labels))
            counts      = np.bincount(labels)
            probs       = counts / counts.sum()
            entropy     = float(-np.sum(probs * np.log(probs + 1e-12)))

            results[p] = {
                "mean_jaccard":   mean_jaccard,
                "num_clusters":   num_clusters,
                "entropy":        entropy,
                "num_successful": len(trajs),
            }
        return results

    # ------------------------------------------------------------------
    # M3: Exploitation Ratio
    # ------------------------------------------------------------------

    def m3_exploitation_ratio(self, cluster_threshold: float = 0.5,
                               max_sample: int = 200) -> Dict[int, float]:
        """M3: Exploitation Ratio.

        What: Fraction of successful trajectories in the largest route cluster.
        Why:  Directly measures reliance on a single route.
        Interpret: ~1.0 = extreme overexploitation.  0.5 = two balanced routes.
        Captures: Route preference, overexploitation.
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
            if len(trajs) > max_sample:
                idx     = np.random.choice(len(trajs), max_sample, replace=False)
                sampled = [trajs[i] for i in idx]
            else:
                sampled = trajs
            cell_sets = [_trajectory_to_cell_set(t) for t in sampled]
            labels    = _cluster_trajectories(cell_sets, cluster_threshold)
            counts    = np.bincount(labels)
            results[p] = float(np.max(counts) / counts.sum())
        return results

    # ------------------------------------------------------------------
    # M4: Policy Adaptation Index
    # ------------------------------------------------------------------

    def m4_policy_adaptation_index(self) -> List[Dict[str, Any]]:
        """M4: Policy Adaptation Index (KL divergence).

        What: KL(pi_t || pi_{t-delta}) between consecutive policy snapshots.
        Why:  Measures how much the actor's behaviour changed.
        Interpret: Spikes at phase transitions = adapting.  Flat = stable.
        Captures: Policy adaptation speed, learning stability.
        """
        results = []
        for i in range(1, len(self.policy_snapshots)):
            ep_prev, phase_prev, pol_prev = self.policy_snapshots[i - 1]
            ep_curr, phase_curr, pol_curr = self.policy_snapshots[i]
            kl = _kl_divergence(pol_curr, pol_prev)
            results.append({
                "episode_from":       int(ep_prev),
                "episode_to":         int(ep_curr),
                "phase_from":         int(phase_prev),
                "phase_to":           int(phase_curr),
                "kl_divergence":      kl,
                "is_phase_transition": phase_prev != phase_curr,
            })
        return results

    # ------------------------------------------------------------------
    # M5: Representation Adaptation Rate
    # ------------------------------------------------------------------

    def m5_representation_adaptation_rate(self) -> List[Dict[str, Any]]:
        """M5: Representation Adaptation Rate (psi L2 velocity).

        What: Mean L2 distance between psi(s) at t and t-delta across all states.
        Why:  Measures how fast the critic reshapes the representation landscape.
        Interpret: High early = active learning.  Spike at phase = re-learning.
        Captures: Representation plasticity.
        """
        results = []
        for i in range(1, len(self.psi_full_snapshots)):
            ep_prev,   phase_prev, psi_prev = self.psi_full_snapshots[i - 1]
            ep_curr,   phase_curr, psi_curr = self.psi_full_snapshots[i]
            l2 = float(np.mean(np.linalg.norm(psi_curr - psi_prev, axis=1)))
            dots = np.sum(psi_curr * psi_prev, axis=1)
            norms = (np.linalg.norm(psi_curr, axis=1) *
                     np.linalg.norm(psi_prev, axis=1) + 1e-12)
            cos_dist = float(1.0 - np.mean(dots / norms))
            results.append({
                "episode_from":       int(ep_prev),
                "episode_to":         int(ep_curr),
                "phase_from":         int(phase_prev),
                "phase_to":           int(phase_curr),
                "mean_l2_rate":       l2,
                "mean_cosine_distance": cos_dist,
            })
        return results

    # ------------------------------------------------------------------
    # M6: Suboptimality Gap (stochastic)
    # ------------------------------------------------------------------

    def m6_suboptimality_gap(self, true_probs: Dict[str, float],
                              window: int = 50) -> np.ndarray:
        """M6: Suboptimality Gap.

        What: p_best - p_agent_direction, rolling average over `window` episodes.
        Why:  Directly measures whether the agent found the best-probability route.
        Interpret: 0 = optimal.  >0 = using a worse direction.
        Captures: Convergence to optimal strategy under stochastic success.
        """
        if not self.approach_history:
            return np.array([])

        best_prob = max(v for k, v in true_probs.items() if k != "stay")
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
    # M7: Approach Direction Entropy (stochastic)
    # ------------------------------------------------------------------

    def m7_approach_direction_entropy(self, window: int = 100) -> np.ndarray:
        """M7: Approach Direction Entropy.

        What: Shannon entropy over approach-direction distribution in a rolling window.
        Why:  High = trying many directions (exploration).  Low = locked in.
        Interpret: max ~log(4)=1.386 (uniform).  0 = always same direction.
        Captures: Exploration breadth, sensitivity to stochastic success.
        """
        if not self.approach_history:
            return np.array([])

        dir_labels  = ["from_above", "from_below", "from_left", "from_right"]
        dir_to_idx  = {d: i for i, d in enumerate(dir_labels)}
        entropies   = []
        for t in range(len(self.approach_history)):
            start  = max(0, t - window + 1)
            counts = np.zeros(len(dir_labels))
            for i in range(start, t + 1):
                _, d, _ = self.approach_history[i]
                if d in dir_to_idx:
                    counts[dir_to_idx[d]] += 1
            total = counts.sum()
            if total == 0:
                entropies.append(0.0)
            else:
                probs   = counts / total
                entropy = -np.sum(probs * np.log(probs + 1e-12))
                entropies.append(float(entropy))
        return np.array(entropies)

    # ------------------------------------------------------------------
    # M8: Representation-Route Alignment
    # ------------------------------------------------------------------

    def m8_representation_route_alignment(self) -> List[Dict[str, Any]]:
        """M8: Representation-Route Alignment.

        What: Mean psi-similarity of states ON the greedy trajectory vs OFF it.
              Alignment gap = on_route - off_route.
        Why:  Shows whether psi has formed a 'ridge' along the current route.
        Interpret: Large positive gap = tight lock-in.  Small gap = flat landscape.
        Captures: Representation-behaviour coupling, how locked-in the psi is.
        """
        results = []
        for (ep_psi, phase_psi, sim_map), (ep_traj, phase_traj, traj) in zip(
                self.psi_snapshots, self.greedy_trajectory_snapshots):
            h, w = sim_map.shape
            route_states = set(traj)
            on_route_sims  = []
            off_route_sims = []
            for s in range(h * w):
                r, c = np.unravel_index(s, (h, w))
                val  = sim_map[r, c]
                if np.isnan(val):
                    continue
                if s in route_states:
                    on_route_sims.append(val)
                else:
                    off_route_sims.append(val)

            mean_on  = float(np.mean(on_route_sims))  if on_route_sims  else 0.0
            mean_off = float(np.mean(off_route_sims)) if off_route_sims else 0.0
            results.append({
                "episode":       int(ep_psi),
                "phase":         int(phase_psi),
                "mean_on_route": mean_on,
                "mean_off_route": mean_off,
                "alignment_gap": mean_on - mean_off,
                "route_length":  len(route_states),
            })
        return results

    # ------------------------------------------------------------------
    # M9: Phase-Transition Recovery Time
    # ------------------------------------------------------------------

    def m9_phase_transition_recovery(self, threshold_frac: float = 0.5,
                                      window: int = 50) -> List[Dict[str, Any]]:
        """M9: Phase-Transition Recovery Time.

        What: Episodes after a phase transition until rolling success rate
              recovers to threshold_frac * pre-transition level.
        Why:  Measures adaptation speed in concrete terms.
        Interpret: Short = fast adaptation.  None = never recovered.
        Captures: Practical adaptation cost.
        """
        if not self.phase_transition_episodes:
            return []

        rolling  = self.m1_rolling_success_rate(window)
        results  = []
        for trans_ep in self.phase_transition_episodes:
            pre_start = max(0, trans_ep - window)
            pre_level = float(np.mean(rolling[pre_start:trans_ep])) if trans_ep > 0 else 0.0
            target    = pre_level * threshold_frac

            recovery_ep = None
            search_end  = min(len(rolling), trans_ep + 2000)
            for ep in range(trans_ep, search_end):
                if rolling[ep] >= target:
                    recovery_ep = ep - trans_ep
                    break

            results.append({
                "transition_episode":  int(trans_ep),
                "pre_transition_level": pre_level,
                "target_level":        target,
                "recovery_episodes":   recovery_ep,
            })
        return results

    # ------------------------------------------------------------------
    # Checkpoint state
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        """Return serialisable state for checkpoint/resume."""
        return {
            "rolling_state":             self._rolling.get_state(),
            "episode_success":           [bool(x) for x in self.episode_success],
            "episode_phase":             list(self.episode_phase),
            "phase_transition_episodes": list(self.phase_transition_episodes),
            "approach_history": [
                (int(ep), d, bool(s))
                for ep, d, s in self.approach_history
            ],
            "m0_episode_log": list(self.m0_episode_log),
            # NOTE: snapshot arrays are saved separately as .npy files
            # and listed in snapshots/index.json — they are NOT embedded
            # in the checkpoint to keep file sizes manageable.
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any],
                   num_states: int, maze_shape: Tuple[int, int]) -> "MetricsCollector":
        """Restore a MetricsCollector from a checkpoint state dict."""
        obj = cls(num_states=num_states, maze_shape=maze_shape,
                  m1_window=state["rolling_state"]["window"])
        obj._rolling = _RollingSuccessRate.from_state(state["rolling_state"])
        obj.episode_success           = list(state["episode_success"])
        obj.episode_phase             = list(state["episode_phase"])
        obj.phase_transition_episodes = list(state["phase_transition_episodes"])
        obj.approach_history = [
            (ep, d, s)
            for ep, d, s in state.get("approach_history", [])
        ]
        obj.m0_episode_log = list(state.get("m0_episode_log", []))
        # episode_trajectories and snapshot lists are NOT restored from
        # checkpoint (would be too large); they remain empty.
        return obj

    # ------------------------------------------------------------------
    # Save to output directory
    # ------------------------------------------------------------------

    def save(self, output_dir: str,
             true_probs: Optional[Dict[str, float]] = None):
        """Write all metrics to individual files under output_dir/metrics/.

        Files written:
          all_metrics.json          — full serialised metrics dict
          m1_rolling_success_rate.npy
          m4_policy_adaptation.json
          m5_representation_rate.json
          m8_route_alignment.json
          m9_recovery.json
          m6_suboptimality_gap.npy   (if true_probs provided)
          m7_approach_entropy.npy    (if true_probs provided)
          approach_stats.json        (if approach_history present)
        """
        metrics_dir = os.path.join(output_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)

        # M1 rolling array
        m1_arr = self.m1_rolling_success_rate()
        np.save(os.path.join(metrics_dir, "m1_rolling_success_rate.npy"), m1_arr)

        # M4 policy adaptation
        m4 = self.m4_policy_adaptation_index()
        with open(os.path.join(metrics_dir, "m4_policy_adaptation.json"), "w") as f:
            json.dump(m4, f)

        # M5 representation rate
        m5 = self.m5_representation_adaptation_rate()
        with open(os.path.join(metrics_dir, "m5_representation_rate.json"), "w") as f:
            json.dump(m5, f)

        # M8 alignment
        m8 = self.m8_representation_route_alignment()
        with open(os.path.join(metrics_dir, "m8_route_alignment.json"), "w") as f:
            json.dump(m8, f)

        # M9 recovery
        m9 = self.m9_phase_transition_recovery()
        with open(os.path.join(metrics_dir, "m9_recovery.json"), "w") as f:
            json.dump(m9, f)

        # Stochastic metrics
        if true_probs is not None:
            m6 = self.m6_suboptimality_gap(true_probs)
            np.save(os.path.join(metrics_dir, "m6_suboptimality_gap.npy"), m6)
            m7 = self.m7_approach_direction_entropy()
            np.save(os.path.join(metrics_dir, "m7_approach_entropy.npy"), m7)

        if self.approach_history:
            dir_counts: Dict[str, int]          = defaultdict(int)
            dir_successes: Dict[str, int]        = defaultdict(int)
            for _, direction, success in self.approach_history:
                if direction is not None:
                    dir_counts[direction]    += 1
                    if success:
                        dir_successes[direction] += 1
            approach_stats = {
                d: {
                    "attempts":       dir_counts[d],
                    "successes":      dir_successes.get(d, 0),
                    "empirical_rate": dir_successes.get(d, 0) / dir_counts[d]
                                      if dir_counts[d] > 0 else 0.0,
                }
                for d in dir_counts
            }
            with open(os.path.join(metrics_dir, "approach_stats.json"), "w") as f:
                json.dump(approach_stats, f, indent=2)

        # Summary all_metrics.json
        summary: Dict[str, Any] = {
            "total_episodes":          len(self.episode_success),
            "phase_transition_episodes": self.phase_transition_episodes,
            "m1_success_rate_per_phase": {
                str(k): v
                for k, v in self.m1_success_rate_per_phase().items()
            },
            "m1_final_rolling":        float(m1_arr[-1]) if len(m1_arr) > 0 else 0.0,
            "m2_path_diversity":       {
                str(k): v
                for k, v in self.m2_path_diversity().items()
            },
            "m3_exploitation_ratio":   {
                str(k): v
                for k, v in self.m3_exploitation_ratio().items()
            },
            "m4_num_snapshots":        len(m4),
            "m5_num_snapshots":        len(m5),
            "m8_num_snapshots":        len(m8),
            "m9_recovery":             m9,
            "psi_snapshot_episodes":   [
                {"episode": ep, "phase": ph}
                for ep, ph, _ in self.psi_snapshots
            ],
        }
        if true_probs is not None:
            m6_arr = self.m6_suboptimality_gap(true_probs)
            summary["m6_final_gap"] = float(m6_arr[-1]) if len(m6_arr) > 0 else 0.0

        with open(os.path.join(metrics_dir, "all_metrics.json"), "w") as f:
            json.dump(summary, f, indent=2)

    # ------------------------------------------------------------------
    # Legacy serialisation (used by old run_single.py / generate_plots.py)
    # ------------------------------------------------------------------

    def to_dict(self, true_probs: Optional[Dict[str, float]] = None
                ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "m1_rolling_success_rate":    self.m1_rolling_success_rate().tolist(),
            "m1_success_rate_per_phase":  self.m1_success_rate_per_phase(),
            "m2_path_diversity":          {
                str(k): v for k, v in self.m2_path_diversity().items()
            },
            "m3_exploitation_ratio":      {
                str(k): v for k, v in self.m3_exploitation_ratio().items()
            },
            "m4_policy_adaptation_index": self.m4_policy_adaptation_index(),
            "m5_representation_adaptation_rate":
                self.m5_representation_adaptation_rate(),
            "m8_representation_route_alignment":
                self.m8_representation_route_alignment(),
            "m9_phase_transition_recovery":
                self.m9_phase_transition_recovery(),
            "psi_snapshot_episodes": [
                {"episode": ep, "phase": ph}
                for ep, ph, _ in self.psi_snapshots
            ],
            "total_episodes":            len(self.episode_success),
            "phase_transition_episodes": self.phase_transition_episodes,
        }
        if true_probs:
            result["m6_suboptimality_gap"] = \
                self.m6_suboptimality_gap(true_probs).tolist()
            result["m7_approach_direction_entropy"] = \
                self.m7_approach_direction_entropy().tolist()
        return result
