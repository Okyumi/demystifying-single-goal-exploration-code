"""
Metric computation for the continual maze experiments.

Metrics implemented:
1. Success rate (per-phase)
2. Path diversity (Jaccard distance, cluster count, entropy)
3. Trajectory preference (fraction per route)
4. Adaptation speed (steps to first success after phase change)
5. ψ-similarity evolution (per-state similarity to goal)
6. Representation drift (L2 and cosine at phase transitions)
7. Exploitation ratio (fraction using dominant path)
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _traj_to_cell_set(traj: List[int]) -> frozenset:
    """Convert trajectory to set of visited cells."""
    return frozenset(traj)


def _jaccard_distance(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / len(a | b)


def _classify_route(traj: List[int], maze_shape: Tuple[int, int],
                    goal_state: int) -> str:
    """Classify a trajectory by the quadrant sequence it passes through.

    We label each state by its quadrant (TL, TR, BL, BR) and record the
    sequence of *unique* quadrant transitions.  This gives a compact
    route fingerprint.
    """
    h, w = maze_shape
    mid_r, mid_c = h // 2, w // 2
    labels = []
    for s in traj:
        r, c = np.unravel_index(s, maze_shape)
        if r < mid_r:
            label = "TL" if c < mid_c else "TR"
        else:
            label = "BL" if c < mid_c else "BR"
        if not labels or labels[-1] != label:
            labels.append(label)
    return "->".join(labels)


# -----------------------------------------------------------------------
# 1. Success rate
# -----------------------------------------------------------------------

def compute_success_rate(
    successes: List[int],
    phase_at_episode: List[int],
) -> Dict:
    """Per-phase and overall success rate."""
    successes = np.array(successes)
    phases = np.array(phase_at_episode)

    overall = float(successes.mean()) if len(successes) > 0 else 0.0
    per_phase = {}
    for p in np.unique(phases):
        mask = phases == p
        per_phase[int(p)] = float(successes[mask].mean())

    return {"overall": overall, "per_phase": per_phase}


# -----------------------------------------------------------------------
# 2. Path diversity
# -----------------------------------------------------------------------

def compute_path_diversity(
    trajectories: List[List[int]],
    successes: List[int],
    phase_at_episode: List[int],
    maze_shape: Tuple[int, int],
    goal_state: int,
) -> Dict:
    """Compute Jaccard-based path diversity for successful trajectories."""
    successes = np.array(successes)
    phases = np.array(phase_at_episode)
    n = len(trajectories)

    result: Dict = {"per_phase": {}}

    for p in np.unique(phases):
        mask = (phases == p) & (successes == 1)
        idxs = np.where(mask)[0]
        if len(idxs) < 2:
            result["per_phase"][int(p)] = {
                "mean_jaccard": 0.0,
                "num_distinct_routes": int(len(idxs)),
                "route_entropy": 0.0,
            }
            continue

        # Cell-set representation
        cell_sets = [_traj_to_cell_set(trajectories[i]) for i in idxs]

        # Mean pairwise Jaccard distance
        dists = []
        for i in range(len(cell_sets)):
            for j in range(i + 1, len(cell_sets)):
                dists.append(_jaccard_distance(cell_sets[i], cell_sets[j]))
        mean_jacc = float(np.mean(dists)) if dists else 0.0

        # Route classification and entropy
        routes = [_classify_route(trajectories[i], maze_shape, goal_state)
                  for i in idxs]
        counter = Counter(routes)
        total = sum(counter.values())
        probs = np.array([c / total for c in counter.values()])
        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))

        result["per_phase"][int(p)] = {
            "mean_jaccard": mean_jacc,
            "num_distinct_routes": len(counter),
            "route_entropy": entropy,
            "route_counts": dict(counter),
        }

    return result


# -----------------------------------------------------------------------
# 3. Trajectory preference
# -----------------------------------------------------------------------

def compute_trajectory_preference(
    trajectories: List[List[int]],
    successes: List[int],
    phase_at_episode: List[int],
    maze_shape: Tuple[int, int],
    goal_state: int,
) -> Dict:
    """Fraction of successful episodes using each route, per phase."""
    successes = np.array(successes)
    phases = np.array(phase_at_episode)
    result: Dict = {"per_phase": {}}

    for p in np.unique(phases):
        mask = (phases == p) & (successes == 1)
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            result["per_phase"][int(p)] = {}
            continue
        routes = [_classify_route(trajectories[i], maze_shape, goal_state)
                  for i in idxs]
        counter = Counter(routes)
        total = sum(counter.values())
        fracs = {r: c / total for r, c in counter.items()}
        result["per_phase"][int(p)] = fracs

    return result


# -----------------------------------------------------------------------
# 4. Adaptation speed
# -----------------------------------------------------------------------

def compute_adaptation_speed(
    successes: List[int],
    phase_log: List[Tuple[int, int]],
) -> Dict:
    """Episodes from phase transition to first success."""
    result = {}
    for transition_ep, phase_idx in phase_log:
        if phase_idx == 0:
            continue  # skip initial phase
        first_success = None
        for ep_offset in range(len(successes) - transition_ep):
            ep = transition_ep + ep_offset
            if ep >= len(successes):
                break
            if successes[ep]:
                first_success = ep_offset
                break
        result[int(phase_idx)] = {
            "transition_episode": int(transition_ep),
            "episodes_to_first_success": first_success,
        }
    return result


# -----------------------------------------------------------------------
# 5. ψ-similarity evolution
# -----------------------------------------------------------------------

def compute_psi_similarity_snapshots(
    psi_snapshots: List[Tuple[int, int, np.ndarray]],
    goal_state: int,
    maze_shape: Tuple[int, int],
) -> Dict:
    """Compute per-state ψ(s)·ψ(g) similarity at each snapshot.

    Also computes mean similarity per quadrant.
    """
    h, w = maze_shape
    mid_r, mid_c = h // 2, w // 2
    n_states = h * w

    snapshots = []
    for episode, phase_idx, psi in psi_snapshots:
        goal_vec = psi[goal_state]
        sims = psi @ goal_vec  # (n_states,)

        # Per-quadrant means
        quad_sims = {"TL": [], "TR": [], "BL": [], "BR": []}
        for s in range(n_states):
            r, c = np.unravel_index(s, maze_shape)
            if r < mid_r:
                q = "TL" if c < mid_c else "TR"
            else:
                q = "BL" if c < mid_c else "BR"
            quad_sims[q].append(sims[s])

        quad_means = {q: float(np.mean(v)) if v else 0.0
                      for q, v in quad_sims.items()}

        snapshots.append({
            "episode": int(episode),
            "phase_idx": int(phase_idx),
            "similarity_map": sims.tolist(),
            "quadrant_means": quad_means,
        })

    return {"snapshots": snapshots}


# -----------------------------------------------------------------------
# 6. Representation drift
# -----------------------------------------------------------------------

def compute_representation_drift(
    phase_snapshots,  # List[PhaseSnapshot]
) -> Dict:
    """L2 and cosine drift of ψ at phase transitions."""
    result = {}
    for i in range(1, len(phase_snapshots)):
        prev = phase_snapshots[i - 1]
        curr = phase_snapshots[i]

        psi_prev = prev.psi
        psi_curr = curr.psi

        # Mean L2 distance per state
        l2 = np.linalg.norm(psi_curr - psi_prev, axis=1)
        mean_l2 = float(l2.mean())

        # Mean cosine similarity per state
        dots = np.sum(psi_curr * psi_prev, axis=1)
        norm_prev = np.linalg.norm(psi_prev, axis=1) + 1e-8
        norm_curr = np.linalg.norm(psi_curr, axis=1) + 1e-8
        cosines = dots / (norm_prev * norm_curr)
        mean_cos = float(cosines.mean())

        result[int(curr.phase_idx)] = {
            "from_episode": int(prev.episode),
            "to_episode": int(curr.episode),
            "mean_l2_drift": mean_l2,
            "mean_cosine_similarity": mean_cos,
        }

    return result


# -----------------------------------------------------------------------
# 7. Exploitation ratio
# -----------------------------------------------------------------------

def compute_exploitation_ratio(
    trajectories: List[List[int]],
    successes: List[int],
    phase_at_episode: List[int],
    maze_shape: Tuple[int, int],
    goal_state: int,
) -> Dict:
    """Fraction of successful episodes using the dominant route."""
    successes = np.array(successes)
    phases = np.array(phase_at_episode)
    result: Dict = {"per_phase": {}}

    for p in np.unique(phases):
        mask = (phases == p) & (successes == 1)
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            result["per_phase"][int(p)] = {
                "exploitation_ratio": 0.0,
                "dominant_route": None,
            }
            continue
        routes = [_classify_route(trajectories[i], maze_shape, goal_state)
                  for i in idxs]
        counter = Counter(routes)
        dominant_route, dominant_count = counter.most_common(1)[0]
        ratio = dominant_count / len(idxs)
        result["per_phase"][int(p)] = {
            "exploitation_ratio": float(ratio),
            "dominant_route": dominant_route,
            "total_successful": int(len(idxs)),
        }

    return result


# -----------------------------------------------------------------------
# Aggregate
# -----------------------------------------------------------------------

def compute_all_metrics(
    results: Dict,
    maze_shape: Tuple[int, int],
    goal_state: int,
) -> Dict:
    """Compute all metrics from a training results dict."""
    train_succ = results["train_successes"]
    eval_succ = results["eval_successes"]
    train_traj = results["train_trajectories"]
    eval_traj = results["eval_trajectories"]
    phase_at_ep = results["phase_at_episode"]
    phase_log = results["phase_log"]
    psi_snaps = results["psi_snapshots"]
    phase_snaps = results["phase_snapshots"]

    # Use eval trajectories for path metrics (greedy policy)
    # But we need eval_phase_at_episode aligned with eval indices
    eval_phases = phase_at_ep[::1]  # eval_freq=1 means same length
    # Trim to match eval length
    eval_phases = eval_phases[:len(eval_succ)]

    return {
        "success_rate": compute_success_rate(train_succ, phase_at_ep),
        "eval_success_rate": compute_success_rate(eval_succ, eval_phases),
        "path_diversity": compute_path_diversity(
            eval_traj, eval_succ, eval_phases, maze_shape, goal_state),
        "trajectory_preference": compute_trajectory_preference(
            eval_traj, eval_succ, eval_phases, maze_shape, goal_state),
        "adaptation_speed": compute_adaptation_speed(train_succ, phase_log),
        "psi_similarity": compute_psi_similarity_snapshots(
            psi_snaps, goal_state, maze_shape),
        "representation_drift": compute_representation_drift(phase_snaps),
        "exploitation_ratio": compute_exploitation_ratio(
            eval_traj, eval_succ, eval_phases, maze_shape, goal_state),
    }
