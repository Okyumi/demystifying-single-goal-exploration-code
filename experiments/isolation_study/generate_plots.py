"""
generate_plots.py — Generate all 14+ plots for the SGCRL isolation study.

HPC additions over the original version:
  - CLI: --results_dir, --seeds, --output_dir, --condition filter
  - Multi-seed aggregation: mean ± std for M1 success-rate plots
  - Graceful handling of missing data (skips missing conditions)
  - Backward compatibility: reads both old snapshots.pkl format and
    new individual .npy format (snapshots/index.json)

Usage:
  # Single-seed (uses results dir default):
  python generate_plots.py

  # Multi-seed with explicit seeds:
  python generate_plots.py \\
      --results_dir results/study_a \\
      --seeds 42 43 44 45 46 \\
      --output_dir results/figures

  # Only Study A plots:
  python generate_plots.py --condition a1 a2ar a2so

  # Skip condition if data not present (default behaviour):
  python generate_plots.py --skip_missing
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SCRIPT_DIR = Path(__file__).resolve().parent

# ── style ──────────────────────────────────────────────────────────────────────
COL_BASE    = "#1976D2"
COL_TREAT   = "#E64A19"
COL_THIRD   = "#388E3C"
DPI         = 150
WALL_COLOR  = "#333333"
ROUTE_COLOR = "#1565C0"
SUCCESS_COLOR = "#27ae60"
FAILURE_COLOR = "#c0392b"
COL_SHADE_BASE  = "#BBDEFB"
COL_SHADE_TREAT = "#FFCCBC"
COL_SHADE_THIRD = "#C8E6C9"

# ── condition name mapping ─────────────────────────────────────────────────────
_COND_DIR = {
    "a1":   "A1-baseline",
    "a2ar": "A2-allreplay",
    "a2so": "A2-successonly",
    "b1":   "B1-baseline",
    "b2":   "B2-changing",
}


# ── data loading helpers ───────────────────────────────────────────────────────

def load_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_pkl(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    result = np.full(len(arr), np.nan)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        result[i] = np.mean(arr[start:i + 1])
    return result


def get_walls(env_config: dict, phase_idx: int = 0) -> np.ndarray:
    if "phases" in env_config:
        walls = env_config["phases"][min(phase_idx, len(env_config["phases"]) - 1)]["walls"]
    else:
        walls = env_config["walls"]
    return np.array(walls, dtype=bool)


def draw_walls(ax, walls: np.ndarray, alpha: float = 1.0):
    rows, cols = np.where(walls)
    for r, c in zip(rows, cols):
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                   color=WALL_COLOR, alpha=alpha, zorder=2))


def draw_route(ax, traj_states: List[int], width: int = 11,
               color: str = ROUTE_COLOR, lw: float = 2.0):
    if not traj_states or len(traj_states) < 2:
        return
    rows = [s // width for s in traj_states]
    cols = [s % width  for s in traj_states]
    ax.plot(cols, rows, color=color, linewidth=lw, zorder=5, alpha=0.85)
    n    = len(cols)
    step = max(1, n // 6)
    for i in range(0, n - 1, step):
        ax.annotate(
            "", xy=(cols[i + 1], rows[i + 1]), xytext=(cols[i], rows[i]),
            arrowprops=dict(arrowstyle="->", color=color, lw=lw), zorder=6,
        )


def psi_heatmap_ax(ax, psi: np.ndarray, walls: np.ndarray,
                   traj_states=None, title: str = "", width: int = 11):
    display = psi.copy().astype(float)
    ax.imshow(display, cmap="RdYlGn", vmin=0.7, vmax=1.0,
              origin="upper", aspect="equal", zorder=1)
    draw_walls(ax, walls.astype(bool))
    if traj_states is not None:
        draw_route(ax, traj_states, width=width)
    ax.set_title(title, fontsize=8, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])


# ── multi-seed aggregation ─────────────────────────────────────────────────────

def load_m1_seeds(cond_dir_root: str, seeds: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load M1 rolling success rate for each seed; return (mean, std, eps_arr)."""
    arrays = []
    for seed in seeds:
        seed_path = os.path.join(cond_dir_root, f"seed_{seed}", "metrics.json")
        m = load_json(seed_path)
        if m is None:
            # Try metrics/all_metrics.json + m1_rolling_success_rate.npy
            npy_path = os.path.join(
                cond_dir_root, f"seed_{seed}", "metrics", "m1_rolling_success_rate.npy")
            if os.path.exists(npy_path):
                arrays.append(np.load(npy_path))
        else:
            arrays.append(np.array(m["m1_rolling_success_rate"]))
    if not arrays:
        return None, None, None
    min_len = min(len(a) for a in arrays)
    mat = np.stack([a[:min_len] for a in arrays], axis=0)  # (S, T)
    return mat.mean(0), mat.std(0), np.arange(1, min_len + 1)


# ── condition directory resolver ───────────────────────────────────────────────

class _ResultsLocator:
    """Finds the per-seed result directories for a condition."""

    def __init__(self, results_dir: str, seeds: List[int]):
        self.results_dir = results_dir
        self.seeds = seeds

    def seed_dir(self, condition: str, seed: int) -> str:
        """Return the seed directory path (may not exist)."""
        cond_name = _COND_DIR[condition]
        return os.path.join(self.results_dir, cond_name, f"seed_{seed}")

    def first_seed_dir(self, condition: str) -> Optional[str]:
        """Return first existing seed directory, or None."""
        for seed in self.seeds:
            d = self.seed_dir(condition, seed)
            if os.path.isdir(d):
                return d
        return None

    def cond_root(self, condition: str) -> str:
        return os.path.join(self.results_dir, _COND_DIR[condition])

    def has_condition(self, condition: str) -> bool:
        return self.first_seed_dir(condition) is not None


# ── Study A plots ──────────────────────────────────────────────────────────────

def plot_study_a_m1_success_rate(loc: _ResultsLocator, out_dir: str,
                                  seeds: List[int]):
    """Plot M1 rolling success rate for all 3 Study A conditions (multi-seed)."""
    combos = [
        ("a1",   COL_BASE,  COL_SHADE_BASE,  "A1 - Baseline (deterministic)"),
        ("a2ar", COL_TREAT, COL_SHADE_TREAT, "A2 - Stochastic (all replay)"),
        ("a2so", COL_THIRD, COL_SHADE_THIRD, "A2 - Stochastic (success only)"),
    ]
    fig, ax = plt.subplots(figsize=(14, 5))
    any_data = False
    for cond, color, shade, label in combos:
        if not loc.has_condition(cond):
            continue
        mean, std, eps = load_m1_seeds(loc.cond_root(cond), seeds)
        if mean is None:
            continue
        any_data = True
        ax.plot(eps, mean, color=color, linewidth=1.4, label=label)
        if std is not None and len(seeds) > 1:
            ax.fill_between(eps, mean - std, mean + std,
                            alpha=0.18, color=shade)

    if not any_data:
        plt.close(fig)
        return
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Rolling Success Rate (window=50)", fontsize=12)
    ax.set_title("Study A: Rolling Success Rate Comparison", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "study_a_m1_success_rate.png"), dpi=DPI)
    plt.close(fig)
    print("  Saved: study_a_m1_success_rate.png")


def plot_approach_stats(stats: dict, config: dict, outname: str, label: str,
                         out_dir: str):
    """Bar chart: attempts per direction + true vs empirical rates."""
    directions  = list(stats.keys())
    attempts    = [stats[d]["attempts"]       for d in directions]
    empirical   = [stats[d]["empirical_rate"] for d in directions]
    approach_probs = config.get("approach_probs", {})
    true_rates  = [approach_probs.get(d, np.nan) for d in directions]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    bars = ax.bar(directions, attempts, color=COL_TREAT, alpha=0.8, edgecolor="white")
    ax.set_title(f"{label}: Attempts per Approach Direction", fontsize=12)
    ax.set_xlabel("Approach Direction", fontsize=11)
    ax.set_ylabel("Number of Attempts", fontsize=11)
    max_att = max(attempts) if attempts else 1
    for bar, val in zip(bars, attempts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max_att * 0.01,
                str(val), ha="center", va="bottom", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    ax2 = axes[1]
    x = np.arange(len(directions))
    w = 0.35
    ax2.bar(x - w / 2, true_rates, w, label="True Success Rate",
            color=COL_BASE, alpha=0.85, edgecolor="white")
    ax2.bar(x + w / 2, empirical, w, label="Empirical Success Rate",
            color=COL_TREAT, alpha=0.85, edgecolor="white")
    ax2.set_title(f"{label}: True vs. Empirical Success Rates", fontsize=12)
    ax2.set_xlabel("Approach Direction", fontsize=11)
    ax2.set_ylabel("Success Rate", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(directions)
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=10)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, outname), dpi=DPI)
    plt.close(fig)
    print(f"  Saved: {outname}")


def plot_study_a_approach_allreplay(loc: _ResultsLocator, out_dir: str):
    d = loc.first_seed_dir("a2ar")
    if d is None:
        return
    stats  = load_json(os.path.join(d, "approach_stats.json"))
    config = load_json(os.path.join(d, "run_config.json")) or \
             load_json(os.path.join(d, "config.json")) or {}
    if stats:
        plot_approach_stats(stats, config,
                            "study_a_approach_allreplay.png", "A2-allreplay", out_dir)


def plot_study_a_approach_successonly(loc: _ResultsLocator, out_dir: str):
    d = loc.first_seed_dir("a2so")
    if d is None:
        return
    stats  = load_json(os.path.join(d, "approach_stats.json"))
    config = load_json(os.path.join(d, "run_config.json")) or \
             load_json(os.path.join(d, "config.json")) or {}
    if stats:
        plot_approach_stats(stats, config,
                            "study_a_approach_successonly.png", "A2-successonly", out_dir)


def plot_study_a_approach_over_time(loc: _ResultsLocator, out_dir: str):
    d = loc.first_seed_dir("a2ar")
    if d is None:
        return
    # Try episode_log.jsonl (new format) first
    jsonl_path = os.path.join(d, "episode_log.jsonl")
    ah_path    = os.path.join(d, "approach_history.json")

    history = []
    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                entry = json.loads(line.strip())
                if "approach_dir" in entry:
                    history.append({
                        "episode":   entry["episode"],
                        "direction": entry.get("approach_dir"),
                        "success":   entry.get("success", False),
                    })
    elif os.path.exists(ah_path):
        history = load_json(ah_path) or []

    if not history:
        return

    WINDOW  = 200
    all_dirs = sorted(set(h["direction"] for h in history if h["direction"]))
    episodes  = [h["episode"] for h in history]
    dir_seq   = [h["direction"] for h in history]
    n         = len(history)

    dir_arrays = {
        d: np.array([1.0 if dir_seq[i] == d else 0.0 for i in range(n)])
        for d in all_dirs
    }
    has_dir = np.array([1.0 if dir_seq[i] is not None else 0.0 for i in range(n)])

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = plt.cm.tab10(np.linspace(0, 0.7, len(all_dirs)))
    for idx, d in enumerate(all_dirs):
        roll_dir   = np.array([np.sum(dir_arrays[d][max(0, i-WINDOW+1):i+1]) for i in range(n)])
        roll_total = np.array([np.sum(has_dir[max(0, i-WINDOW+1):i+1])        for i in range(n)])
        frac = np.where(roll_total > 0, roll_dir / roll_total, np.nan)
        ax.plot(episodes, frac, label=d, linewidth=1.2, color=colors[idx])

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel(f"Rolling Fraction (window={WINDOW})", fontsize=12)
    ax.set_title("Study A (A2-allreplay): Approach Direction Fractions Over Training",
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, max(episodes) if episodes else 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "study_a_approach_over_time.png"), dpi=DPI)
    plt.close(fig)
    print("  Saved: study_a_approach_over_time.png")


# ── Study B plots ──────────────────────────────────────────────────────────────

def _add_phase_markers(ax, phase_info: dict, total: int):
    for ep in phase_info.get("transitions", []):
        ax.axvline(x=ep, color="gray", linestyle="--", linewidth=1.2, alpha=0.7, zorder=3)
    names = phase_info.get("names", [])
    if names:
        boundaries = [0] + list(phase_info["transitions"]) + [total]
        for i, name in enumerate(names):
            mid = (boundaries[i] + boundaries[i + 1]) / 2
            ymin, ymax = ax.get_ylim()
            ax.text(mid, ymax * 0.97, name, ha="center", va="top", fontsize=9,
                    color="dimgray",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="lightgray", alpha=0.8))


def plot_study_b_m1_success_rate(loc: _ResultsLocator, out_dir: str,
                                   seeds: List[int]):
    if not loc.has_condition("b1") and not loc.has_condition("b2"):
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    any_data = False

    for cond, color, shade, label in [
        ("b1", COL_BASE,  COL_SHADE_BASE,  "B1 - Static baseline"),
        ("b2", COL_TREAT, COL_SHADE_TREAT, "B2 - Changing environment"),
    ]:
        if not loc.has_condition(cond):
            continue
        mean, std, eps = load_m1_seeds(loc.cond_root(cond), seeds)
        if mean is None:
            continue
        any_data = True
        ax.plot(eps, mean, color=color, linewidth=1.4, label=label)
        if std is not None and len(seeds) > 1:
            ax.fill_between(eps, mean - std, mean + std, alpha=0.18, color=shade)

    if not any_data:
        plt.close(fig)
        return

    # Phase markers from B2
    b2_dir = loc.first_seed_dir("b2")
    if b2_dir:
        phase_info = load_json(os.path.join(b2_dir, "phase_info.json"))
        if phase_info:
            total = (len(mean) if mean is not None else
                     phase_info["transitions"][-1] + 1) if phase_info.get("transitions") else 1
            for ep in phase_info.get("transitions", []):
                ax.axvline(x=ep, color="gray", linestyle="--", linewidth=1.4, alpha=0.7)
            names = phase_info.get("names", [])
            if names and phase_info.get("transitions"):
                boundaries = [0] + phase_info["transitions"] + [total]
                for i, name in enumerate(names):
                    mid = (boundaries[i] + boundaries[i+1]) / 2
                    ax.text(mid, 1.02, name, ha="center", va="bottom", fontsize=9,
                            color="dimgray",
                            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                      ec="lightgray", alpha=0.8))

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Rolling Success Rate (window=50)", fontsize=12)
    ax.set_title("Study B: Rolling Success Rate Comparison", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "study_b_m1_success_rate.png"), dpi=DPI)
    plt.close(fig)
    print("  Saved: study_b_m1_success_rate.png")


def _extract_m4(m: dict):
    eps = [(e["episode_from"] + e["episode_to"]) / 2
           for e in m.get("m4_policy_adaptation_index", [])]
    kl  = [e["kl_divergence"] for e in m.get("m4_policy_adaptation_index", [])]
    return np.array(eps), np.array(kl)


def _extract_m5(m: dict):
    eps = [(e["episode_from"] + e["episode_to"]) / 2
           for e in m.get("m5_representation_adaptation_rate", [])]
    l2  = [e["mean_l2_rate"] for e in m.get("m5_representation_adaptation_rate", [])]
    return np.array(eps), np.array(l2)


def _extract_m8(m: dict):
    eps = [e["episode"]       for e in m.get("m8_representation_route_alignment", [])]
    gap = [e["alignment_gap"] for e in m.get("m8_representation_route_alignment", [])]
    return np.array(eps), np.array(gap)


def _load_first_metrics(loc: _ResultsLocator, cond: str) -> Optional[dict]:
    d = loc.first_seed_dir(cond)
    if d is None:
        return None
    return load_json(os.path.join(d, "metrics.json"))


def _load_phase_info(loc: _ResultsLocator, cond: str) -> Optional[dict]:
    d = loc.first_seed_dir(cond)
    if d is None:
        return None
    return load_json(os.path.join(d, "phase_info.json"))


def _phase_vlines_and_labels(ax, phase_info: dict, eps_max: float,
                               label_y_frac: float = 0.97):
    for ep in phase_info.get("transitions", []):
        ax.axvline(x=ep, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
    names = phase_info.get("names", [])
    if names and phase_info.get("transitions"):
        boundaries = [0] + list(phase_info["transitions"]) + [eps_max]
        ymin, ymax = ax.get_ylim()
        for i, name in enumerate(names):
            mid = (boundaries[i] + boundaries[i + 1]) / 2
            ax.text(mid, ymax * label_y_frac, name, ha="center", va="top",
                    fontsize=9, color="dimgray",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec="lightgray", alpha=0.8))


def plot_study_b_m4_adaptation(loc: _ResultsLocator, out_dir: str):
    m_b1 = _load_first_metrics(loc, "b1")
    m_b2 = _load_first_metrics(loc, "b2")
    if m_b1 is None and m_b2 is None:
        return
    phase_info = _load_phase_info(loc, "b2") or {}
    fig, ax = plt.subplots(figsize=(14, 5))
    if m_b1:
        eps_b1, kl_b1 = _extract_m4(m_b1)
        ax.plot(eps_b1, kl_b1, color=COL_BASE, linewidth=1.3, marker="o",
                markersize=3, label="B1 - Static baseline")
    if m_b2:
        eps_b2, kl_b2 = _extract_m4(m_b2)
        ax.plot(eps_b2, kl_b2, color=COL_TREAT, linewidth=1.3, marker="o",
                markersize=3, label="B2 - Changing environment")
        eps_max = float(eps_b2.max()) if len(eps_b2) > 0 else 0
        _phase_vlines_and_labels(ax, phase_info, eps_max)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("KL Divergence (M4)", fontsize=12)
    ax.set_title("Study B: Policy Adaptation Index (M4 – KL Divergence)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "study_b_m4_adaptation.png"), dpi=DPI)
    plt.close(fig)
    print("  Saved: study_b_m4_adaptation.png")


def plot_study_b_m5_representation(loc: _ResultsLocator, out_dir: str):
    m_b1 = _load_first_metrics(loc, "b1")
    m_b2 = _load_first_metrics(loc, "b2")
    if m_b1 is None and m_b2 is None:
        return
    phase_info = _load_phase_info(loc, "b2") or {}
    fig, ax = plt.subplots(figsize=(14, 5))
    if m_b1:
        eps, l2 = _extract_m5(m_b1)
        ax.plot(eps, l2, color=COL_BASE, linewidth=1.3, marker="o",
                markersize=3, label="B1 - Static baseline")
    if m_b2:
        eps, l2 = _extract_m5(m_b2)
        ax.plot(eps, l2, color=COL_TREAT, linewidth=1.3, marker="o",
                markersize=3, label="B2 - Changing environment")
        if len(eps) > 0:
            _phase_vlines_and_labels(ax, phase_info, float(eps.max()))
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Mean L2 Change Rate (M5)", fontsize=12)
    ax.set_title("Study B: Representation Adaptation Rate (M5 – Mean L2 Change)",
                 fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "study_b_m5_representation.png"), dpi=DPI)
    plt.close(fig)
    print("  Saved: study_b_m5_representation.png")


def plot_study_b_m8_alignment(loc: _ResultsLocator, out_dir: str):
    m_b1 = _load_first_metrics(loc, "b1")
    m_b2 = _load_first_metrics(loc, "b2")
    if m_b1 is None and m_b2 is None:
        return
    phase_info = _load_phase_info(loc, "b2") or {}
    fig, ax = plt.subplots(figsize=(14, 5))
    if m_b1:
        eps, gap = _extract_m8(m_b1)
        ax.plot(eps, gap, color=COL_BASE, linewidth=1.3, marker="o",
                markersize=3, label="B1 - Static baseline")
    if m_b2:
        eps, gap = _extract_m8(m_b2)
        ax.plot(eps, gap, color=COL_TREAT, linewidth=1.3, marker="o",
                markersize=3, label="B2 - Changing environment")
        if len(eps) > 0:
            _phase_vlines_and_labels(ax, phase_info, float(eps.max()))
    ax.axhline(y=0, color="black", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Alignment Gap (M8)", fontsize=12)
    ax.set_title("Study B: Representation–Route Alignment Gap (M8)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "study_b_m8_alignment.png"), dpi=DPI)
    plt.close(fig)
    print("  Saved: study_b_m8_alignment.png")


# ── psi grids ─────────────────────────────────────────────────────────────────

def _load_snapshots(seed_dir: str):
    """Load snapshot data from either new (.npy index) or old (pkl) format.

    Returns (psi_snaps, traj_snaps) where each is a list of (ep, phase, data).
    """
    # New format: snapshots/index.json + individual .npy files
    index_path = os.path.join(seed_dir, "snapshots", "index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            entries = json.load(f)
        snap_dir   = os.path.join(seed_dir, "snapshots")
        psi_snaps  = []
        traj_snaps = []
        for e in entries:
            ep    = e["episode"]
            phase = e["phase"]
            sim_f = os.path.join(snap_dir, e.get("sim_file", f"sim_ep{ep:06d}.npy"))
            rt_f  = os.path.join(snap_dir, e.get("route_file", f"route_ep{ep:06d}.npy"))
            if os.path.exists(sim_f):
                psi_snaps.append((ep, phase, np.load(sim_f)))
            if os.path.exists(rt_f):
                traj_snaps.append((ep, phase, np.load(rt_f).tolist()))
        return psi_snaps, traj_snaps

    # Old format: snapshots.pkl
    pkl_path = os.path.join(seed_dir, "snapshots.pkl")
    if os.path.exists(pkl_path):
        data = load_pkl(pkl_path)
        if data:
            return (data.get("psi_snapshots", []),
                    data.get("greedy_trajectory_snapshots", []))
    return [], []


def make_psi_grid(seed_dir: str, out_dir: str, outname: str, title: str,
                  n_rows: int = 4, n_cols: int = 4):
    env_config = (load_json(os.path.join(seed_dir, "env_config.json"))
                  or load_json(os.path.join(seed_dir, "config.json")))
    if env_config is None:
        print(f"  Skipping psi grid (no env_config.json): {seed_dir}")
        return

    psi_snaps, traj_snaps = _load_snapshots(seed_dir)
    if not psi_snaps:
        print(f"  Skipping psi grid (no snapshots): {seed_dir}")
        return

    total  = len(psi_snaps)
    n_cells = n_rows * n_cols
    if total <= n_cells:
        indices = list(range(total))
    else:
        indices = [int(round(i * (total - 1) / (n_cells - 1)))
                   for i in range(n_cells)]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 16))
    for ax in axes.flatten():
        ax.set_visible(False)

    width = env_config.get("width", 11)
    n_phases_env = len(env_config.get("phases", [1]))

    for plot_idx, snap_idx in enumerate(indices):
        if plot_idx >= n_cells:
            break
        ax = axes.flatten()[plot_idx]
        ax.set_visible(True)
        ep, phase, psi_arr = psi_snaps[snap_idx]
        traj = traj_snaps[snap_idx][2] if snap_idx < len(traj_snaps) else None
        walls = get_walls(env_config, min(phase, n_phases_env - 1))
        psi_heatmap_ax(ax, psi_arr, walls, traj_states=traj,
                       title=f"ep={ep} phase={phase}", width=width)

    fig.suptitle(title, fontsize=16, y=1.01)
    fig.tight_layout(pad=0.5)
    fig.savefig(os.path.join(out_dir, outname), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outname}")


def plot_study_a_psi_grid_baseline(loc: _ResultsLocator, out_dir: str):
    d = loc.first_seed_dir("a1")
    if d:
        make_psi_grid(d, out_dir, "study_a_psi_grid_baseline.png",
                      "Study A – A1 Baseline: Psi Similarity Grid")


def plot_study_a_psi_grid_allreplay(loc: _ResultsLocator, out_dir: str):
    d = loc.first_seed_dir("a2ar")
    if d:
        make_psi_grid(d, out_dir, "study_a_psi_grid_allreplay.png",
                      "Study A – A2 All-Replay: Psi Similarity Grid")


def plot_study_b_psi_grid_changing(loc: _ResultsLocator, out_dir: str):
    seed_dir = loc.first_seed_dir("b2")
    if seed_dir is None:
        return

    env_config = (load_json(os.path.join(seed_dir, "env_config.json"))
                  or load_json(os.path.join(seed_dir, "config.json")))
    if env_config is None:
        return

    psi_snaps, traj_snaps = _load_snapshots(seed_dir)
    if not psi_snaps:
        return

    by_phase: Dict[int, List[int]] = defaultdict(list)
    for i, (ep, ph, _) in enumerate(psi_snaps):
        by_phase[ph].append(i)

    n_rows, n_cols = 4, 4
    n_cells = n_rows * n_cols
    n_phases = len(by_phase)
    per_phase = max(1, n_cells // max(n_phases, 1))

    indices = []
    for ph in sorted(by_phase.keys()):
        phase_idx_list = by_phase[ph]
        n = len(phase_idx_list)
        if n <= per_phase:
            sel = phase_idx_list
        else:
            sel = [phase_idx_list[int(round(i * (n - 1) / (per_phase - 1)))]
                   for i in range(per_phase)]
        indices.extend(sel)
    indices = indices[:n_cells]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 16))
    for ax in axes.flatten():
        ax.set_visible(False)

    width    = env_config.get("width", 11)
    n_phases_env = len(env_config.get("phases", [1]))

    for plot_idx, snap_idx in enumerate(indices):
        ax = axes.flatten()[plot_idx]
        ax.set_visible(True)
        ep, phase, psi_arr = psi_snaps[snap_idx]
        traj = traj_snaps[snap_idx][2] if snap_idx < len(traj_snaps) else None
        walls = get_walls(env_config, min(phase, n_phases_env - 1))
        phase_name = (env_config["phases"][phase]["name"]
                      if "phases" in env_config and phase < len(env_config["phases"])
                      else str(phase))
        psi_heatmap_ax(ax, psi_arr, walls, traj_states=traj,
                       title=f"ep={ep} ({phase_name})", width=width)

    fig.suptitle("Study B – B2 Changing: Psi Similarity Grid (all 3 phases)",
                 fontsize=16, y=1.01)
    fig.tight_layout(pad=0.5)
    fig.savefig(os.path.join(out_dir, "study_b_psi_grid_changing.png"),
                dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: study_b_psi_grid_changing.png")


# ── trajectory overlays ────────────────────────────────────────────────────────

def _plot_trajectories_inner(trajs, succs, env_config, walls,
                              title, outpath):
    width  = env_config.get("width",  11)
    height = env_config.get("height", 11)
    goal_state  = env_config.get("goal_state",  120)
    start_state = env_config.get("start_state", 0)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_facecolor("#F5F5F5")
    for r in range(height):
        for c in range(width):
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                       color="white", zorder=0))
    draw_walls(ax, walls, alpha=0.9)

    for traj, suc in zip(trajs, succs):
        color = SUCCESS_COLOR if suc else FAILURE_COLOR
        rows = [s // width for s in traj]
        cols = [s % width  for s in traj]
        ax.plot(cols, rows, color=color, linewidth=0.7, alpha=0.35, zorder=3)

    gr, gc = divmod(goal_state,  width)
    sr, sc = divmod(start_state, width)
    ax.plot(gc, gr, "*", color="gold",  markersize=15, zorder=7,
            markeredgecolor="black", markeredgewidth=0.5)
    ax.plot(sc, sr, "s", color="cyan",  markersize=10, zorder=7,
            markeredgecolor="black", markeredgewidth=0.5)

    ax.set_xlim(-0.5, width  - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11)

    legend_patches = [
        mpatches.Patch(color=SUCCESS_COLOR, alpha=0.7, label="Success"),
        mpatches.Patch(color=FAILURE_COLOR, alpha=0.7, label="Failure"),
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="gold",
                   markersize=12, markeredgecolor="black", label="Goal"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="cyan",
                   markersize=9,  markeredgecolor="black", label="Start"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=DPI)
    plt.close(fig)


def plot_trajectories(seed_dir: str, out_dir: str, outname: str, title: str,
                      n_last: int = 100, phase_idx: int = 0):
    env_config = (load_json(os.path.join(seed_dir, "env_config.json"))
                  or load_json(os.path.join(seed_dir, "config.json")))
    if env_config is None:
        return

    # Try old pkl first, then episode_log.jsonl
    pkl = load_pkl(os.path.join(seed_dir, "snapshots.pkl"))
    if pkl and "episode_trajectories" in pkl:
        trajs = pkl["episode_trajectories"][-n_last:]
        succs = pkl["episode_success"][-n_last:]
    else:
        print(f"  No trajectory data for {outname} (snapshots.pkl missing)")
        return

    walls = get_walls(env_config, phase_idx)
    _plot_trajectories_inner(trajs, succs, env_config, walls, title,
                              os.path.join(out_dir, outname))
    print(f"  Saved: {outname}")


def plot_study_a_trajectories_baseline(loc: _ResultsLocator, out_dir: str):
    d = loc.first_seed_dir("a1")
    if d:
        plot_trajectories(d, out_dir,
                          "study_a_trajectories_baseline.png",
                          "Study A – A1 Baseline: Last 100 Episode Trajectories")


def plot_study_a_trajectories_allreplay(loc: _ResultsLocator, out_dir: str):
    d = loc.first_seed_dir("a2ar")
    if d:
        plot_trajectories(d, out_dir,
                          "study_a_trajectories_allreplay.png",
                          "Study A – A2 All-Replay: Last 100 Episode Trajectories")


def plot_study_b_trajectories_changing(loc: _ResultsLocator, out_dir: str):
    seed_dir = loc.first_seed_dir("b2")
    if seed_dir is None:
        return
    env_config = (load_json(os.path.join(seed_dir, "env_config.json"))
                  or load_json(os.path.join(seed_dir, "config.json")))
    if env_config is None:
        return
    pkl = load_pkl(os.path.join(seed_dir, "snapshots.pkl"))
    if not pkl or "episode_trajectories" not in pkl:
        return
    trajs = pkl["episode_trajectories"][-100:]
    succs = pkl["episode_success"][-100:]
    phase_idx = len(env_config.get("phases", [1])) - 1
    walls = get_walls(env_config, phase_idx)
    phase_name = (env_config["phases"][phase_idx]["name"]
                  if "phases" in env_config else "")
    _plot_trajectories_inner(
        trajs, succs, env_config, walls,
        f"Study B – B2 Changing: Last 100 Trajectories (final phase: {phase_name})",
        os.path.join(out_dir, "study_b_trajectories_changing.png"),
    )
    print("  Saved: study_b_trajectories_changing.png")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _build_parser():
    p = argparse.ArgumentParser(
        description="Generate plots for the SGCRL isolation study.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results_dir", default=None,
                   help="Root results directory.  If not given, uses "
                        "<script_dir>/results.  Can point to study_a or a "
                        "directory containing both study_a/ and study_b/.")
    p.add_argument("--study_a_dir", default=None,
                   help="Explicit Study A results root (overrides --results_dir).")
    p.add_argument("--study_b_dir", default=None,
                   help="Explicit Study B results root (overrides --results_dir).")
    p.add_argument("--seeds", type=int, nargs="+", default=[42],
                   help="Seeds to aggregate over.")
    p.add_argument("--output_dir", default=None,
                   help="Directory to write figures into.  Defaults to "
                        "<results_dir>/figures.")
    p.add_argument("--condition", nargs="*", default=None,
                   choices=["a1", "a2ar", "a2so", "b1", "b2"],
                   help="Only generate plots for these conditions.")
    p.add_argument("--skip_missing", action="store_true",
                   help="Silently skip plots when result files do not exist.")
    return p


def _resolve_dirs(args):
    """Return (study_a_results_dir, study_b_results_dir, figures_dir)."""
    base = Path(args.results_dir) if args.results_dir else SCRIPT_DIR / "results"

    # Study A
    if args.study_a_dir:
        sa_dir = Path(args.study_a_dir)
    elif (base / "study_a").is_dir():
        sa_dir = base / "study_a"
    elif any((base / _COND_DIR[c]).is_dir() for c in ["a1", "a2ar", "a2so"]):
        sa_dir = base
    else:
        # Old single-seed layout: base / study_a_seed_42 / A1-baseline
        sa_dir = base

    # Study B
    if args.study_b_dir:
        sb_dir = Path(args.study_b_dir)
    elif (base / "study_b").is_dir():
        sb_dir = base / "study_b"
    else:
        sb_dir = base

    # Figures
    if args.output_dir:
        fig_dir = Path(args.output_dir)
    else:
        fig_dir = base / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    return str(sa_dir), str(sb_dir), str(fig_dir)


def main():
    parser = _build_parser()
    args   = parser.parse_args()

    sa_dir, sb_dir, fig_dir = _resolve_dirs(args)
    seeds = args.seeds
    cond_filter = set(args.condition) if args.condition else None

    def _want(c: str) -> bool:
        return cond_filter is None or c in cond_filter

    loc_a = _ResultsLocator(sa_dir, seeds)
    loc_b = _ResultsLocator(sb_dir, seeds)

    print("=== Generating Study A Comparison Plots ===")
    if _want("a1") or _want("a2ar") or _want("a2so"):
        plot_study_a_m1_success_rate(loc_a, fig_dir, seeds)
    if _want("a2ar"):
        plot_study_a_approach_allreplay(loc_a, fig_dir)
        plot_study_a_approach_over_time(loc_a, fig_dir)
    if _want("a2so"):
        plot_study_a_approach_successonly(loc_a, fig_dir)

    print("\n=== Generating Study B Comparison Plots ===")
    if _want("b1") or _want("b2"):
        plot_study_b_m1_success_rate(loc_b, fig_dir, seeds)
        plot_study_b_m4_adaptation(loc_b, fig_dir)
        plot_study_b_m5_representation(loc_b, fig_dir)
        plot_study_b_m8_alignment(loc_b, fig_dir)

    print("\n=== Generating Psi Similarity Grids ===")
    if _want("a1"):
        plot_study_a_psi_grid_baseline(loc_a, fig_dir)
    if _want("a2ar"):
        plot_study_a_psi_grid_allreplay(loc_a, fig_dir)
    if _want("b2"):
        plot_study_b_psi_grid_changing(loc_b, fig_dir)

    print("\n=== Generating Trajectory Overlays ===")
    if _want("a1"):
        plot_study_a_trajectories_baseline(loc_a, fig_dir)
    if _want("a2ar"):
        plot_study_a_trajectories_allreplay(loc_a, fig_dir)
    if _want("b2"):
        plot_study_b_trajectories_changing(loc_b, fig_dir)

    print("\n=== All plots generated ===")
    files = sorted(os.listdir(fig_dir))
    print(f"\nFiles in {fig_dir}:")
    for fname in files:
        fpath = os.path.join(fig_dir, fname)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  {fname}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
