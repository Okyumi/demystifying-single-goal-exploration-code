"""
Generate all plots for the isolation study experiments.
"""

import json
import pickle
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from collections import Counter

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")
FIGURES = os.path.join(RESULTS, "figures")
os.makedirs(FIGURES, exist_ok=True)

# Study A directories
A1 = os.path.join(RESULTS, "study_a_seed_42", "A1-baseline")
A2_ALL = os.path.join(RESULTS, "study_a_seed_42", "A2-stochastic-allreplay")
A2_SUC = os.path.join(RESULTS, "study_a_seed_42", "A2-stochastic-successonly")

# Study B directories
B1 = os.path.join(RESULTS, "study_b_seed_42", "B1-baseline")
B2 = os.path.join(RESULTS, "study_b_seed_42", "B2-changing")

# ─── Style constants ──────────────────────────────────────────────────────────
COL_BASE = "#1976D2"
COL_TREAT = "#E64A19"
COL_THIRD = "#388E3C"
DPI = 150
WALL_COLOR = "#333333"
ROUTE_COLOR = "#1565C0"
SUCCESS_COLOR = "#27ae60"
FAILURE_COLOR = "#c0392b"


# ─── Helper functions ─────────────────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        return json.load(f)

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def rolling_mean(arr, window):
    result = np.full(len(arr), np.nan)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        result[i] = np.mean(arr[start:i+1])
    return result

def state_to_rc(state, width=11):
    """Convert state int to (row, col)."""
    return divmod(state, width)

def get_walls(env_config, phase_idx=0):
    """Return 11x11 boolean wall array for a given phase index.
    Handles both flat-walls format and phases-list format.
    """
    if "phases" in env_config:
        walls = env_config["phases"][phase_idx]["walls"]
    else:
        # Flat format: single walls array
        walls = env_config["walls"]
    return np.array(walls, dtype=bool)

def draw_walls(ax, walls, alpha=1.0):
    """Fill wall cells with dark gray."""
    rows, cols = np.where(walls)
    for r, c in zip(rows, cols):
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                   color=WALL_COLOR, alpha=alpha, zorder=2))

def draw_route(ax, traj_states, width=11, color=ROUTE_COLOR, lw=2):
    """Draw policy route as a line with arrows."""
    if not traj_states or len(traj_states) < 2:
        return
    rows = [s // width for s in traj_states]
    cols = [s % width for s in traj_states]
    ax.plot(cols, rows, color=color, linewidth=lw, zorder=5, alpha=0.85)
    # Draw arrows at regular intervals
    n = len(cols)
    step = max(1, n // 6)
    for i in range(0, n - 1, step):
        dc = cols[i+1] - cols[i]
        dr = rows[i+1] - rows[i]
        ax.annotate("", xy=(cols[i+1], rows[i+1]), xytext=(cols[i], rows[i]),
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw),
                    zorder=6)

def psi_heatmap_ax(ax, psi, walls, traj_states=None, title="", width=11):
    """Render psi similarity heatmap on ax, with optional route overlay."""
    display = psi.copy().astype(float)
    # Mask wall cells
    wall_mask = walls.astype(bool)

    img = ax.imshow(display, cmap="RdYlGn", vmin=0.7, vmax=1.0,
                    origin="upper", aspect="equal", zorder=1)
    # Overlay walls
    draw_walls(ax, wall_mask)
    # Route overlay
    if traj_states is not None:
        draw_route(ax, traj_states, width=width)
    ax.set_title(title, fontsize=8, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    return img


# ═══════════════════════════════════════════════════════════════════════════════
# STUDY A COMPARISON PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_study_a_m1_success_rate():
    """Plot 1: Study A rolling success rate (3 lines)."""
    m_a1 = load_json(os.path.join(A1, "metrics.json"))
    m_ar = load_json(os.path.join(A2_ALL, "metrics.json"))
    m_as = load_json(os.path.join(A2_SUC, "metrics.json"))

    r_a1 = np.array(m_a1["m1_rolling_success_rate"])
    r_ar = np.array(m_ar["m1_rolling_success_rate"])
    r_as = np.array(m_as["m1_rolling_success_rate"])
    eps = np.arange(1, len(r_a1) + 1)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(eps, r_a1, color=COL_BASE, linewidth=1.2, label="A1 - Baseline (deterministic)")
    ax.plot(eps, r_ar, color=COL_TREAT, linewidth=1.2, label="A2 - Stochastic (all replay)")
    ax.plot(eps, r_as, color=COL_THIRD, linewidth=1.2, label="A2 - Stochastic (success only)")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Rolling Success Rate (window=50)", fontsize=12)
    ax.set_title("Study A: Rolling Success Rate Comparison", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xlim(1, len(r_a1))
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "study_a_m1_success_rate.png"), dpi=DPI)
    plt.close(fig)
    print("  Saved: study_a_m1_success_rate.png")


def plot_approach_stats(stats, config, outname, label):
    """Plot 2/3: Approach bar chart with attempts + true vs empirical rates."""
    directions = list(stats.keys())
    attempts = [stats[d]["attempts"] for d in directions]
    empirical = [stats[d]["empirical_rate"] for d in directions]

    # True rates from config
    approach_probs = config.get("approach_probs", {})
    true_rates = [approach_probs.get(d, np.nan) for d in directions]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: attempts bar chart
    ax = axes[0]
    bars = ax.bar(directions, attempts, color=COL_TREAT, alpha=0.8, edgecolor="white")
    ax.set_title(f"{label}: Attempts per Approach Direction", fontsize=12)
    ax.set_xlabel("Approach Direction", fontsize=11)
    ax.set_ylabel("Number of Attempts", fontsize=11)
    for bar, val in zip(bars, attempts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(attempts) * 0.01,
                str(val), ha="center", va="bottom", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    # Right: true vs empirical success rates
    ax2 = axes[1]
    x = np.arange(len(directions))
    width = 0.35
    rects1 = ax2.bar(x - width / 2, true_rates, width, label="True Success Rate",
                     color=COL_BASE, alpha=0.85, edgecolor="white")
    rects2 = ax2.bar(x + width / 2, empirical, width, label="Empirical Success Rate",
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
    fig.savefig(os.path.join(FIGURES, outname), dpi=DPI)
    plt.close(fig)
    print(f"  Saved: {outname}")


def plot_study_a_approach_allreplay():
    stats = load_json(os.path.join(A2_ALL, "approach_stats.json"))
    config = load_json(os.path.join(A2_ALL, "config.json"))
    plot_approach_stats(stats, config, "study_a_approach_allreplay.png", "A2-allreplay")


def plot_study_a_approach_successonly():
    stats = load_json(os.path.join(A2_SUC, "approach_stats.json"))
    config = load_json(os.path.join(A2_SUC, "config.json"))
    plot_approach_stats(stats, config, "study_a_approach_successonly.png", "A2-successonly")


def plot_study_a_approach_over_time():
    """Plot 4: Rolling fraction of each approach direction over training."""
    history = load_json(os.path.join(A2_ALL, "approach_history.json"))
    WINDOW = 200

    # Collect all directions
    all_dirs = sorted(set(h["direction"] for h in history if h["direction"] is not None))
    episodes = [h["episode"] for h in history]
    directions_seq = [h["direction"] for h in history]

    # Build per-direction binary arrays
    n = len(history)
    dir_arrays = {}
    for d in all_dirs:
        arr = np.array([1.0 if directions_seq[i] == d else 0.0 for i in range(n)])
        dir_arrays[d] = arr

    # Rolling fraction (fraction among episodes with a direction assigned)
    has_dir = np.array([1.0 if directions_seq[i] is not None else 0.0 for i in range(n)])

    eps_arr = np.array(episodes)

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = plt.cm.tab10(np.linspace(0, 0.7, len(all_dirs)))
    for idx, d in enumerate(all_dirs):
        # Rolling sum for direction / rolling sum of episodes with direction
        roll_dir = np.array([
            np.sum(dir_arrays[d][max(0, i - WINDOW + 1):i + 1])
            for i in range(n)
        ])
        roll_total = np.array([
            np.sum(has_dir[max(0, i - WINDOW + 1):i + 1])
            for i in range(n)
        ])
        frac = np.where(roll_total > 0, roll_dir / roll_total, np.nan)
        ax.plot(eps_arr, frac, label=d, linewidth=1.2, color=colors[idx])

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel(f"Rolling Fraction (window={WINDOW})", fontsize=12)
    ax.set_title("Study A (A2-allreplay): Approach Direction Fractions Over Training", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, max(episodes))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "study_a_approach_over_time.png"), dpi=DPI)
    plt.close(fig)
    print("  Saved: study_a_approach_over_time.png")


# ═══════════════════════════════════════════════════════════════════════════════
# STUDY B COMPARISON PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def add_phase_markers(ax, phase_names=None, transitions=(5000, 10000), total=15000):
    """Add vertical lines and phase labels to a Study B plot."""
    for ep in transitions:
        ax.axvline(x=ep, color="gray", linestyle="--", linewidth=1.2, alpha=0.7, zorder=3)
    if phase_names:
        boundaries = [0] + list(transitions) + [total]
        for i, name in enumerate(phase_names):
            mid = (boundaries[i] + boundaries[i + 1]) / 2
            ymin, ymax = ax.get_ylim()
            ax.text(mid, ymax * 0.97, name, ha="center", va="top",
                    fontsize=9, color="gray",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6))


def plot_study_b_m1_success_rate():
    """Plot 5: Study B rolling success rate (B1 vs B2) with phase markers."""
    m_b1 = load_json(os.path.join(B1, "metrics.json"))
    m_b2 = load_json(os.path.join(B2, "metrics.json"))
    phase_info = load_json(os.path.join(B2, "phase_info.json"))

    r_b1 = np.array(m_b1["m1_rolling_success_rate"])
    r_b2 = np.array(m_b2["m1_rolling_success_rate"])
    eps = np.arange(1, len(r_b1) + 1)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(eps, r_b1, color=COL_BASE, linewidth=1.2, label="B1 - Static baseline")
    ax.plot(eps, r_b2, color=COL_TREAT, linewidth=1.2, label="B2 - Changing environment")

    for ep in phase_info["transitions"]:
        ax.axvline(x=ep, color="gray", linestyle="--", linewidth=1.4, alpha=0.7, zorder=3)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Rolling Success Rate (window=50)", fontsize=12)
    ax.set_title("Study B: Rolling Success Rate Comparison", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xlim(1, len(r_b1))
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)

    # Phase labels
    boundaries = [0] + phase_info["transitions"] + [len(r_b1)]
    for i, name in enumerate(phase_info["names"]):
        mid = (boundaries[i] + boundaries[i + 1]) / 2
        ax.text(mid, 1.02, name, ha="center", va="bottom", fontsize=9, color="dimgray",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="lightgray", alpha=0.8))

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "study_b_m1_success_rate.png"), dpi=DPI)
    plt.close(fig)
    print("  Saved: study_b_m1_success_rate.png")


def plot_study_b_m4_adaptation():
    """Plot 6: KL divergence (M4) comparison B1 vs B2."""
    m_b1 = load_json(os.path.join(B1, "metrics.json"))
    m_b2 = load_json(os.path.join(B2, "metrics.json"))
    phase_info = load_json(os.path.join(B2, "phase_info.json"))

    def extract_m4(m):
        eps_mid = [(e["episode_from"] + e["episode_to"]) / 2 for e in m["m4_policy_adaptation_index"]]
        kl = [e["kl_divergence"] for e in m["m4_policy_adaptation_index"]]
        return np.array(eps_mid), np.array(kl)

    eps_b1, kl_b1 = extract_m4(m_b1)
    eps_b2, kl_b2 = extract_m4(m_b2)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(eps_b1, kl_b1, color=COL_BASE, linewidth=1.3, marker="o", markersize=3,
            label="B1 - Static baseline")
    ax.plot(eps_b2, kl_b2, color=COL_TREAT, linewidth=1.3, marker="o", markersize=3,
            label="B2 - Changing environment")

    for ep in phase_info["transitions"]:
        ax.axvline(x=ep, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("KL Divergence (M4)", fontsize=12)
    ax.set_title("Study B: Policy Adaptation Index (M4 – KL Divergence)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Phase labels
    boundaries = [0] + phase_info["transitions"] + [max(eps_b1.max(), eps_b2.max())]
    for i, name in enumerate(phase_info["names"]):
        mid = (boundaries[i] + boundaries[i + 1]) / 2
        ax.text(mid, ax.get_ylim()[1] * 0.97, name, ha="center", va="top",
                fontsize=9, color="dimgray",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="lightgray", alpha=0.8))

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "study_b_m4_adaptation.png"), dpi=DPI)
    plt.close(fig)
    print("  Saved: study_b_m4_adaptation.png")


def plot_study_b_m5_representation():
    """Plot 7: Mean L2 change (M5) comparison B1 vs B2."""
    m_b1 = load_json(os.path.join(B1, "metrics.json"))
    m_b2 = load_json(os.path.join(B2, "metrics.json"))
    phase_info = load_json(os.path.join(B2, "phase_info.json"))

    def extract_m5(m):
        eps_mid = [(e["episode_from"] + e["episode_to"]) / 2 for e in m["m5_representation_adaptation_rate"]]
        l2 = [e["mean_l2_rate"] for e in m["m5_representation_adaptation_rate"]]
        return np.array(eps_mid), np.array(l2)

    eps_b1, l2_b1 = extract_m5(m_b1)
    eps_b2, l2_b2 = extract_m5(m_b2)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(eps_b1, l2_b1, color=COL_BASE, linewidth=1.3, marker="o", markersize=3,
            label="B1 - Static baseline")
    ax.plot(eps_b2, l2_b2, color=COL_TREAT, linewidth=1.3, marker="o", markersize=3,
            label="B2 - Changing environment")

    for ep in phase_info["transitions"]:
        ax.axvline(x=ep, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Mean L2 Change Rate (M5)", fontsize=12)
    ax.set_title("Study B: Representation Adaptation Rate (M5 – Mean L2 Change)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    boundaries = [0] + phase_info["transitions"] + [max(eps_b1.max(), eps_b2.max())]
    for i, name in enumerate(phase_info["names"]):
        mid = (boundaries[i] + boundaries[i + 1]) / 2
        ax.text(mid, ax.get_ylim()[1] * 0.97, name, ha="center", va="top",
                fontsize=9, color="dimgray",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="lightgray", alpha=0.8))

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "study_b_m5_representation.png"), dpi=DPI)
    plt.close(fig)
    print("  Saved: study_b_m5_representation.png")


def plot_study_b_m8_alignment():
    """Plot 8: Alignment gap (M8) comparison B1 vs B2."""
    m_b1 = load_json(os.path.join(B1, "metrics.json"))
    m_b2 = load_json(os.path.join(B2, "metrics.json"))
    phase_info = load_json(os.path.join(B2, "phase_info.json"))

    def extract_m8(m):
        eps = [e["episode"] for e in m["m8_representation_route_alignment"]]
        gap = [e["alignment_gap"] for e in m["m8_representation_route_alignment"]]
        return np.array(eps), np.array(gap)

    eps_b1, gap_b1 = extract_m8(m_b1)
    eps_b2, gap_b2 = extract_m8(m_b2)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(eps_b1, gap_b1, color=COL_BASE, linewidth=1.3, marker="o", markersize=3,
            label="B1 - Static baseline")
    ax.plot(eps_b2, gap_b2, color=COL_TREAT, linewidth=1.3, marker="o", markersize=3,
            label="B2 - Changing environment")
    ax.axhline(y=0, color="black", linewidth=0.8, linestyle=":", alpha=0.5)

    for ep in phase_info["transitions"]:
        ax.axvline(x=ep, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Alignment Gap (M8)", fontsize=12)
    ax.set_title("Study B: Representation–Route Alignment Gap (M8)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    boundaries = [0] + phase_info["transitions"] + [max(eps_b1.max(), eps_b2.max())]
    ymin, ymax = ax.get_ylim()
    for i, name in enumerate(phase_info["names"]):
        mid = (boundaries[i] + boundaries[i + 1]) / 2
        ax.text(mid, ymax * 0.97, name, ha="center", va="top",
                fontsize=9, color="dimgray",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="lightgray", alpha=0.8))

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "study_b_m8_alignment.png"), dpi=DPI)
    plt.close(fig)
    print("  Saved: study_b_m8_alignment.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PSI SIMILARITY GRIDS
# ═══════════════════════════════════════════════════════════════════════════════

def make_psi_grid(snap_dir, env_config_path, outname, title, n_rows=4, n_cols=4):
    """
    Generic 4x4 psi similarity grid with policy route overlay.
    Selects up to 16 snapshots evenly spaced across time.
    """
    snap = load_pkl(os.path.join(snap_dir, "snapshots.pkl"))
    env_config = load_json(env_config_path)

    psi_snaps = snap["psi_snapshots"]        # list of (ep, phase, 11x11 ndarray)
    traj_snaps = snap["greedy_trajectory_snapshots"]  # list of (ep, phase, list)

    total = len(psi_snaps)
    n_cells = n_rows * n_cols
    # Evenly spaced indices
    if total <= n_cells:
        indices = list(range(total))
    else:
        indices = [int(round(i * (total - 1) / (n_cells - 1))) for i in range(n_cells)]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 16))
    axes_flat = axes.flatten()

    # Hide unused cells
    for ax in axes_flat:
        ax.set_visible(False)

    for plot_idx, snap_idx in enumerate(indices):
        if plot_idx >= n_cells:
            break
        ax = axes_flat[plot_idx]
        ax.set_visible(True)

        ep, phase, psi_arr = psi_snaps[snap_idx]
        _, _, traj = traj_snaps[snap_idx]

        n_phases = len(env_config["phases"]) if "phases" in env_config else 1
        walls = get_walls(env_config, phase_idx=min(phase, n_phases - 1))

        psi_heatmap_ax(ax, psi_arr, walls, traj_states=traj,
                       title=f"ep={ep} phase={phase}", width=env_config["width"])

    fig.suptitle(title, fontsize=16, y=1.01)
    fig.tight_layout(pad=0.5)
    fig.savefig(os.path.join(FIGURES, outname), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outname}")


def plot_study_a_psi_grid_baseline():
    make_psi_grid(A1, os.path.join(A1, "env_config.json"),
                  "study_a_psi_grid_baseline.png",
                  "Study A – A1 Baseline: Psi Similarity Grid")


def plot_study_a_psi_grid_allreplay():
    make_psi_grid(A2_ALL, os.path.join(A2_ALL, "env_config.json"),
                  "study_a_psi_grid_allreplay.png",
                  "Study A – A2 All-Replay: Psi Similarity Grid")


def plot_study_b_psi_grid_changing():
    """
    Plot 11: B2-changing psi grid. Shows all 3 phases.
    Select ~5-6 snapshots per phase for a 4x4 = 16 total grid.
    """
    snap = load_pkl(os.path.join(B2, "snapshots.pkl"))
    env_config = load_json(os.path.join(B2, "env_config.json"))

    psi_snaps = snap["psi_snapshots"]
    traj_snaps = snap["greedy_trajectory_snapshots"]

    # Group by phase
    from collections import defaultdict
    by_phase = defaultdict(list)
    for i, (ep, ph, arr) in enumerate(psi_snaps):
        by_phase[ph].append(i)

    n_rows, n_cols = 4, 4
    n_cells = n_rows * n_cols
    n_phases = len(by_phase)
    per_phase = n_cells // n_phases  # 5 per phase, with 1 leftover

    indices = []
    for ph in sorted(by_phase.keys()):
        phase_indices = by_phase[ph]
        n = len(phase_indices)
        if n <= per_phase:
            sel = phase_indices
        else:
            sel = [phase_indices[int(round(i * (n - 1) / (per_phase - 1)))] for i in range(per_phase)]
        indices.extend(sel)
    # Trim or pad to 16
    indices = indices[:n_cells]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 16))
    axes_flat = axes.flatten()

    for ax in axes_flat:
        ax.set_visible(False)

    for plot_idx, snap_idx in enumerate(indices):
        ax = axes_flat[plot_idx]
        ax.set_visible(True)

        ep, phase, psi_arr = psi_snaps[snap_idx]
        _, _, traj = traj_snaps[snap_idx]

        n_phases = len(env_config["phases"]) if "phases" in env_config else 1
        walls = get_walls(env_config, phase_idx=min(phase, n_phases - 1))
        phase_name = env_config["phases"][phase]["name"]
        psi_heatmap_ax(ax, psi_arr, walls, traj_states=traj,
                       title=f"ep={ep} ({phase_name})", width=env_config["width"])

    fig.suptitle("Study B – B2 Changing: Psi Similarity Grid (all 3 phases)", fontsize=16, y=1.01)
    fig.tight_layout(pad=0.5)
    fig.savefig(os.path.join(FIGURES, "study_b_psi_grid_changing.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: study_b_psi_grid_changing.png")


# ═══════════════════════════════════════════════════════════════════════════════
# TRAJECTORY OVERLAYS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_trajectories(snap_dir, env_config_path, outname, title,
                      n_last=100, phase_idx=0):
    """
    Draw last N episode trajectories on the maze.
    Green = success, Red = failure.
    """
    snap = load_pkl(os.path.join(snap_dir, "snapshots.pkl"))
    env_config = load_json(env_config_path)

    trajs = snap["episode_trajectories"]   # list of lists of state ints
    succs = snap["episode_success"]        # list of bools

    # Take last n_last
    trajs = trajs[-n_last:]
    succs = succs[-n_last:]

    walls = get_walls(env_config, phase_idx=phase_idx)
    width = env_config["width"]
    height = env_config["height"]
    goal_state = env_config["goal_state"]
    start_state = env_config["start_state"]

    fig, ax = plt.subplots(figsize=(7, 7))

    # Background grid
    ax.set_facecolor("#F5F5F5")
    for r in range(height):
        for c in range(width):
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                       color="white", zorder=0))

    # Draw walls
    draw_walls(ax, walls, alpha=0.9)

    # Draw trajectories
    for traj, suc in zip(trajs, succs):
        color = SUCCESS_COLOR if suc else FAILURE_COLOR
        rows = [s // width for s in traj]
        cols = [s % width for s in traj]
        ax.plot(cols, rows, color=color, linewidth=0.7, alpha=0.35, zorder=3)

    # Mark start and goal
    gr, gc = divmod(goal_state, width)
    sr, sc = divmod(start_state, width)
    ax.plot(gc, gr, "*", color="gold", markersize=15, zorder=7, markeredgecolor="black",
            markeredgewidth=0.5, label="Goal")
    ax.plot(sc, sr, "s", color="cyan", markersize=10, zorder=7, markeredgecolor="black",
            markeredgewidth=0.5, label="Start")

    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)  # invert y for row=0 at top
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=13)

    # Legend
    legend_patches = [
        mpatches.Patch(color=SUCCESS_COLOR, alpha=0.7, label="Success"),
        mpatches.Patch(color=FAILURE_COLOR, alpha=0.7, label="Failure"),
    ]
    ax.legend(handles=legend_patches + [
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="gold",
                   markersize=12, markeredgecolor="black", label="Goal"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="cyan",
                   markersize=9, markeredgecolor="black", label="Start"),
    ], loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, outname), dpi=DPI)
    plt.close(fig)
    print(f"  Saved: {outname}")


def plot_study_a_trajectories_baseline():
    plot_trajectories(A1, os.path.join(A1, "env_config.json"),
                      "study_a_trajectories_baseline.png",
                      "Study A – A1 Baseline: Last 100 Episode Trajectories")


def plot_study_a_trajectories_allreplay():
    plot_trajectories(A2_ALL, os.path.join(A2_ALL, "env_config.json"),
                      "study_a_trajectories_allreplay.png",
                      "Study A – A2 All-Replay: Last 100 Episode Trajectories")


def plot_study_b_trajectories_changing():
    """Last 100 trajectories for B2-changing (final phase walls)."""
    snap = load_pkl(os.path.join(B2, "snapshots.pkl"))
    env_config = load_json(os.path.join(B2, "env_config.json"))

    trajs = snap["episode_trajectories"][-100:]
    succs = snap["episode_success"][-100:]

    # Use last phase walls
    phase_idx = len(env_config["phases"]) - 1
    walls = get_walls(env_config, phase_idx=phase_idx)
    width = env_config["width"]
    height = env_config["height"]
    goal_state = env_config["goal_state"]
    start_state = env_config["start_state"]

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
        cols = [s % width for s in traj]
        ax.plot(cols, rows, color=color, linewidth=0.7, alpha=0.35, zorder=3)

    gr, gc = divmod(goal_state, width)
    sr, sc = divmod(start_state, width)
    ax.plot(gc, gr, "*", color="gold", markersize=15, zorder=7, markeredgecolor="black",
            markeredgewidth=0.5)
    ax.plot(sc, sr, "s", color="cyan", markersize=10, zorder=7, markeredgecolor="black",
            markeredgewidth=0.5)

    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    phase_name = env_config["phases"][phase_idx]["name"]
    ax.set_title(f"Study B – B2 Changing: Last 100 Trajectories (final phase: {phase_name})",
                 fontsize=11)

    legend_patches = [
        mpatches.Patch(color=SUCCESS_COLOR, alpha=0.7, label="Success"),
        mpatches.Patch(color=FAILURE_COLOR, alpha=0.7, label="Failure"),
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="gold",
                   markersize=12, markeredgecolor="black", label="Goal"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="cyan",
                   markersize=9, markeredgecolor="black", label="Start"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "study_b_trajectories_changing.png"), dpi=DPI)
    plt.close(fig)
    print("  Saved: study_b_trajectories_changing.png")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Generating Study A Comparison Plots ===")
    plot_study_a_m1_success_rate()
    plot_study_a_approach_allreplay()
    plot_study_a_approach_successonly()
    plot_study_a_approach_over_time()

    print("\n=== Generating Study B Comparison Plots ===")
    plot_study_b_m1_success_rate()
    plot_study_b_m4_adaptation()
    plot_study_b_m5_representation()
    plot_study_b_m8_alignment()

    print("\n=== Generating Psi Similarity Grids ===")
    plot_study_a_psi_grid_baseline()
    plot_study_a_psi_grid_allreplay()
    plot_study_b_psi_grid_changing()

    print("\n=== Generating Trajectory Overlays ===")
    plot_study_a_trajectories_baseline()
    plot_study_a_trajectories_allreplay()
    plot_study_b_trajectories_changing()

    print("\n=== All plots generated ===")
    # List output files
    files = sorted(os.listdir(FIGURES))
    print(f"\nFiles in {FIGURES}:")
    for f in files:
        fpath = os.path.join(FIGURES, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  {f}  ({size_kb:.1f} KB)")
