"""
Visualization for isolation study experiments.

Key feature: psi-similarity heatmaps with greedy policy route overlay.

Produces:
1. V1: psi-similarity heatmap WITH policy route overlay
2. V2: Trajectory overlay per condition
3. V3: Metric comparison plots (baseline vs treatment)
4. V4: psi-similarity evolution video WITH route overlay
5. V5: Approach direction analysis (Study A only)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from typing import List, Tuple, Dict, Any, Optional
import subprocess


# ---------------------------------------------------------------------------
# Maze grid renderer
# ---------------------------------------------------------------------------

def _draw_maze(ax, walls: np.ndarray, start: Tuple[int, int],
               goal: Tuple[int, int], title: str = ""):
    h, w = walls.shape
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10, pad=8)
    for r in range(h):
        for c in range(w):
            if walls[r, c] == 1:
                ax.add_patch(patches.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    facecolor="#333333", edgecolor="#444444", linewidth=0.5))
    for r in range(h + 1):
        ax.axhline(r - 0.5, color="#cccccc", linewidth=0.3)
    for c in range(w + 1):
        ax.axvline(c - 0.5, color="#cccccc", linewidth=0.3)
    sr, sc = start
    gr, gc = goal
    ax.plot(sc, sr, "o", color="#2ecc71", markersize=10, zorder=5, label="Start")
    ax.plot(gc, gr, "*", color="#e74c3c", markersize=14, zorder=5, label="Goal")
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_trajectory(ax, traj_coords: List[Tuple[int, int]],
                     color: str = "blue", alpha: float = 0.4,
                     linewidth: float = 1.5):
    rows = [c[0] for c in traj_coords]
    cols = [c[1] for c in traj_coords]
    ax.plot(cols, rows, "-", color=color, alpha=alpha, linewidth=linewidth,
            zorder=3)


def _draw_greedy_route(ax, route_coords: List[Tuple[int, int]],
                       color: str = "#1565C0", linewidth: float = 3.0,
                       alpha: float = 0.9):
    """Draw the greedy policy trajectory as a thick line with arrows."""
    if len(route_coords) < 2:
        return
    rows = [c[0] for c in route_coords]
    cols = [c[1] for c in route_coords]
    # Draw thick line
    ax.plot(cols, rows, "-", color=color, alpha=alpha, linewidth=linewidth,
            zorder=6, solid_capstyle="round")
    # Draw arrows every few steps
    step = max(1, len(route_coords) // 10)
    for i in range(0, len(route_coords) - 1, step):
        dr = rows[i + 1] - rows[i]
        dc = cols[i + 1] - cols[i]
        if dr == 0 and dc == 0:
            continue
        ax.annotate("", xy=(cols[i + 1], rows[i + 1]),
                     xytext=(cols[i], rows[i]),
                     arrowprops=dict(arrowstyle="->", color=color,
                                     lw=linewidth * 0.7, alpha=alpha),
                     zorder=7)


# ---------------------------------------------------------------------------
# V1: psi-similarity heatmap WITH policy route overlay
# ---------------------------------------------------------------------------

def render_psi_heatmap_with_route(
    sim_map: np.ndarray,
    walls: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    greedy_route_coords: Optional[List[Tuple[int, int]]] = None,
    episode: int = 0,
    phase_name: str = "",
    vmin: float = -1.0,
    vmax: float = 1.0,
    title: Optional[str] = None,
) -> plt.Figure:
    """Render psi-similarity heatmap with greedy policy route overlay.

    This is the KEY visualization — shows representation AND induced behavior together.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    h, w = sim_map.shape

    masked = sim_map.copy()
    masked[walls == 1] = np.nan

    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad("#333333")
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")

    # Grid
    for r in range(h + 1):
        ax.axhline(r - 0.5, color="#666666", linewidth=0.3)
    for c in range(w + 1):
        ax.axvline(c - 0.5, color="#666666", linewidth=0.3)

    # Start and goal
    sr, sc = start
    gr, gc = goal
    ax.plot(sc, sr, "o", color="#2196F3", markersize=10, zorder=8)
    ax.plot(gc, gr, "*", color="#FF5722", markersize=14, zorder=8)

    # Greedy route overlay
    if greedy_route_coords:
        _draw_greedy_route(ax, greedy_route_coords)

    if title:
        ax.set_title(title, fontsize=11, pad=10)
    else:
        ax.set_title(f"psi-similarity  |  Ep {episode}  |  {phase_name}",
                     fontsize=11, pad=10)
    ax.set_xticks([])
    ax.set_yticks([])

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("psi(s) . psi(g)", fontsize=9)

    fig.tight_layout()
    return fig


def plot_psi_heatmap_grid_with_routes(
    psi_snapshots: List[Tuple[int, int, np.ndarray]],
    greedy_trajectories: List[Tuple[int, int, List[int]]],
    env_config: dict,
    savepath: str,
    max_frames: int = 16,
):
    """Grid of psi-similarity heatmaps with greedy route overlays."""
    n = len(psi_snapshots)
    if n == 0:
        return
    step = max(1, n // max_frames)
    sel_psi = psi_snapshots[::step][:max_frames]
    sel_traj = greedy_trajectories[::step][:max_frames]

    h = env_config["height"]
    w = env_config["width"]
    phases = env_config.get("phases", [])

    ncols = min(4, len(sel_psi))
    nrows = (len(sel_psi) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    all_sims = np.concatenate([sm.ravel() for _, _, sm in sel_psi])
    valid = all_sims[~np.isnan(all_sims)] if np.any(np.isnan(all_sims)) else all_sims
    vmin = float(np.percentile(valid, 2))
    vmax = float(np.percentile(valid, 98))

    for idx in range(len(sel_psi)):
        ep, phase, sm = sel_psi[idx]
        _, _, traj = sel_traj[idx] if idx < len(sel_traj) else (0, 0, [])
        r_idx, c_idx = divmod(idx, ncols)
        ax = axes[r_idx, c_idx]

        if phase < len(phases):
            walls = np.array(phases[phase]["walls"])
            name = phases[phase]["name"]
        else:
            walls = np.zeros((h, w), dtype=int)
            name = f"phase_{phase}"

        masked = sm.copy()
        masked[walls == 1] = np.nan
        cmap = plt.cm.RdYlGn.copy()
        cmap.set_bad("#333333")
        ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax,
                  interpolation="nearest")
        ax.set_title(f"Ep {ep} ({name})", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

        # Route overlay
        if traj:
            route_coords = [np.unravel_index(s, (h, w)) for s in traj]
            _draw_greedy_route(ax, route_coords, linewidth=2.0)

    for idx in range(len(sel_psi), nrows * ncols):
        r_idx, c_idx = divmod(idx, ncols)
        axes[r_idx, c_idx].axis("off")

    fig.suptitle("psi(s).psi(g) Similarity + Greedy Route Over Training",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# V2: Trajectory overlay
# ---------------------------------------------------------------------------

def plot_trajectory_overlay(walls: np.ndarray,
                            start: Tuple[int, int],
                            goal: Tuple[int, int],
                            trajectories: List[List[Tuple[int, int]]],
                            successes: List[bool],
                            title: str = "",
                            max_trajs: int = 100,
                            savepath: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(7, 7))
    _draw_maze(ax, walls, start, goal, title)

    indices = list(range(len(trajectories)))
    if len(indices) > max_trajs:
        indices = list(np.random.choice(indices, max_trajs, replace=False))

    for idx in indices:
        traj = trajectories[idx]
        color = "#27ae60" if successes[idx] else "#c0392b"
        alpha = 0.5 if successes[idx] else 0.15
        _draw_trajectory(ax, traj, color=color, alpha=alpha)

    legend_elements = [
        Line2D([0], [0], color="#27ae60", linewidth=2, label="Success"),
        Line2D([0], [0], color="#c0392b", linewidth=2, alpha=0.3, label="Failure"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    fig.tight_layout()
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# V3: Metric comparison plots
# ---------------------------------------------------------------------------

def plot_metric_comparison(
    baseline_data: np.ndarray,
    treatment_data: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    savepath: str,
    baseline_label: str = "Baseline",
    treatment_label: str = "Treatment",
    phase_transitions: Optional[List[int]] = None,
    phase_names: Optional[List[str]] = None,
):
    """Plot baseline and treatment on the same axes for comparison."""
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(baseline_data, linewidth=1.5, color="#1976D2", alpha=0.8,
            label=baseline_label)
    ax.plot(treatment_data, linewidth=1.5, color="#E64A19", alpha=0.8,
            label=treatment_label)

    if phase_transitions:
        for t in phase_transitions:
            ax.axvline(t, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        if phase_names:
            boundaries = [0] + phase_transitions + [max(len(baseline_data), len(treatment_data))]
            for i in range(len(boundaries) - 1):
                if i < len(phase_names):
                    mid = (boundaries[i] + boundaries[i + 1]) / 2
                    ax.text(mid, 0.97, phase_names[i], ha="center", va="top",
                            fontsize=8, fontweight="bold", color="gray",
                            transform=ax.get_xaxis_transform())

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_bar_comparison(
    baseline_vals: Dict[int, float],
    treatment_vals: Dict[int, float],
    ylabel: str,
    title: str,
    savepath: str,
    baseline_label: str = "Baseline",
    treatment_label: str = "Treatment",
    phase_names: Optional[List[str]] = None,
):
    """Bar chart comparison of baseline vs treatment per phase."""
    phases = sorted(set(baseline_vals.keys()) | set(treatment_vals.keys()))
    if phase_names:
        labels = [phase_names[p] if p < len(phase_names) else f"P{p}" for p in phases]
    else:
        labels = [f"Phase {p}" for p in phases]

    x = np.arange(len(phases))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    b_vals = [baseline_vals.get(p, 0.0) for p in phases]
    t_vals = [treatment_vals.get(p, 0.0) for p in phases]

    ax.bar(x - width / 2, b_vals, width, label=baseline_label, color="#1976D2")
    ax.bar(x + width / 2, t_vals, width, label=treatment_label, color="#E64A19")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_adaptation_comparison(
    baseline_ai: List[Dict], treatment_ai: List[Dict],
    baseline_rate: List[Dict], treatment_rate: List[Dict],
    savepath: str,
    baseline_label: str = "Baseline",
    treatment_label: str = "Treatment",
    phase_transitions: Optional[List[int]] = None,
):
    """Combined adaptation metrics comparison (M4 + M5)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # M4: Policy KL
    if baseline_ai:
        eps_b = [(d["episode_from"] + d["episode_to"]) / 2 for d in baseline_ai]
        kls_b = [d["kl_divergence"] for d in baseline_ai]
        ax1.plot(eps_b, kls_b, linewidth=1.5, color="#1976D2", alpha=0.8,
                 label=baseline_label)
    if treatment_ai:
        eps_t = [(d["episode_from"] + d["episode_to"]) / 2 for d in treatment_ai]
        kls_t = [d["kl_divergence"] for d in treatment_ai]
        ax1.plot(eps_t, kls_t, linewidth=1.5, color="#E64A19", alpha=0.8,
                 label=treatment_label)
    ax1.set_ylabel("KL Divergence", fontsize=11)
    ax1.set_title("M4: Policy Adaptation Index", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # M5: Representation L2
    if baseline_rate:
        eps_b = [(d["episode_from"] + d["episode_to"]) / 2 for d in baseline_rate]
        l2s_b = [d["mean_l2_rate"] for d in baseline_rate]
        ax2.plot(eps_b, l2s_b, linewidth=1.5, color="#1976D2", alpha=0.8,
                 label=baseline_label)
    if treatment_rate:
        eps_t = [(d["episode_from"] + d["episode_to"]) / 2 for d in treatment_rate]
        l2s_t = [d["mean_l2_rate"] for d in treatment_rate]
        ax2.plot(eps_t, l2s_t, linewidth=1.5, color="#E64A19", alpha=0.8,
                 label=treatment_label)
    ax2.set_ylabel("Mean L2 Change", fontsize=11)
    ax2.set_title("M5: Representation Adaptation Rate", fontsize=11)
    ax2.set_xlabel("Episode", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    if phase_transitions:
        for ax in (ax1, ax2):
            for t in phase_transitions:
                ax.axvline(t, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    fig.tight_layout()
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_m8_alignment_comparison(
    baseline_align: List[Dict], treatment_align: List[Dict],
    savepath: str,
    baseline_label: str = "Baseline",
    treatment_label: str = "Treatment",
    phase_transitions: Optional[List[int]] = None,
):
    """M8: Representation-Route Alignment comparison."""
    fig, ax = plt.subplots(figsize=(14, 5))

    if baseline_align:
        eps_b = [d["episode"] for d in baseline_align]
        gap_b = [d["alignment_gap"] for d in baseline_align]
        ax.plot(eps_b, gap_b, linewidth=1.5, color="#1976D2", alpha=0.8,
                label=baseline_label)
    if treatment_align:
        eps_t = [d["episode"] for d in treatment_align]
        gap_t = [d["alignment_gap"] for d in treatment_align]
        ax.plot(eps_t, gap_t, linewidth=1.5, color="#E64A19", alpha=0.8,
                label=treatment_label)

    if phase_transitions:
        for t in phase_transitions:
            ax.axvline(t, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Alignment Gap (on-route - off-route)", fontsize=11)
    ax.set_title("M8: Representation-Route Alignment", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# V4: psi-similarity evolution video WITH route overlay
# ---------------------------------------------------------------------------

def generate_psi_video_with_routes(
    psi_snapshots: List[Tuple[int, int, np.ndarray]],
    greedy_trajectories: List[Tuple[int, int, List[int]]],
    env_config: dict,
    output_path: str,
    fps: int = 6,
):
    """Generate MP4 video of psi-similarity evolution with greedy route overlay."""
    h = env_config["height"]
    w = env_config["width"]
    start = np.unravel_index(env_config["start_state"], (h, w))
    goal = np.unravel_index(env_config["goal_state"], (h, w))
    phases = env_config.get("phases", [])

    frame_dir = output_path + "_frames"
    os.makedirs(frame_dir, exist_ok=True)

    all_sims = np.concatenate([sm.ravel() for _, _, sm in psi_snapshots])
    valid = all_sims[~np.isnan(all_sims)] if np.any(np.isnan(all_sims)) else all_sims
    vmin = float(np.percentile(valid, 2))
    vmax = float(np.percentile(valid, 98))

    for idx in range(len(psi_snapshots)):
        ep, phase, sim_map = psi_snapshots[idx]

        if phase < len(phases):
            walls = np.array(phases[phase]["walls"])
            name = phases[phase]["name"]
        else:
            walls = np.zeros((h, w), dtype=int)
            name = f"phase_{phase}"

        # Get matching greedy trajectory
        route_coords = None
        if idx < len(greedy_trajectories):
            _, _, traj = greedy_trajectories[idx]
            if traj:
                route_coords = [np.unravel_index(s, (h, w)) for s in traj]

        fig = render_psi_heatmap_with_route(
            sim_map, walls, start, goal, route_coords,
            episode=ep, phase_name=name, vmin=vmin, vmax=vmax)
        fig.savefig(os.path.join(frame_dir, f"frame_{idx:05d}.png"),
                    dpi=100, bbox_inches="tight")
        plt.close(fig)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(frame_dir, "frame_%05d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        output_path,
    ]
    subprocess.run(cmd, capture_output=True)

    for f in os.listdir(frame_dir):
        os.remove(os.path.join(frame_dir, f))
    os.rmdir(frame_dir)
    print(f"  Video saved to {output_path}")


# ---------------------------------------------------------------------------
# V5: Approach direction analysis (Study A only)
# ---------------------------------------------------------------------------

def plot_approach_direction_stats(approach_stats: Dict[str, Dict],
                                  true_probs: Dict[str, float],
                                  savepath: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    dirs = sorted(approach_stats.keys())
    attempts = [approach_stats[d]["attempts"] for d in dirs]
    emp_rates = [approach_stats[d]["empirical_rate"] for d in dirs]
    true_rates = [true_probs.get(d, 0.0) for d in dirs]

    x = np.arange(len(dirs))
    ax1.bar(x, attempts, color="#42A5F5")
    ax1.set_xticks(x)
    ax1.set_xticklabels(dirs, rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel("Number of Attempts")
    ax1.set_title("Approach Direction Attempts")

    width = 0.35
    ax2.bar(x - width / 2, true_rates, width, label="True prob", color="#66BB6A")
    ax2.bar(x + width / 2, emp_rates, width, label="Empirical rate", color="#EF5350")
    ax2.set_xticks(x)
    ax2.set_xticklabels(dirs, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("Success Probability")
    ax2.set_title("True vs Empirical Success Rate")
    ax2.legend()

    fig.suptitle("Stochastic Success: Approach Direction Analysis",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_approach_over_time(approach_history: List[Tuple[int, Optional[str], bool]],
                            savepath: str, window: int = 50):
    if not approach_history:
        return

    dirs = sorted(set(d for _, d, _ in approach_history if d is not None))
    episodes = [e for e, d, s in approach_history]
    n = len(episodes)

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]

    for i, d in enumerate(dirs):
        binary = np.array([1 if dir_ == d else 0
                          for _, dir_, _ in approach_history], dtype=float)
        if len(binary) >= window:
            kernel = np.ones(window) / window
            smoothed = np.convolve(binary, kernel, mode="same")
        else:
            smoothed = np.cumsum(binary) / (np.arange(len(binary)) + 1)
        ax.plot(episodes, smoothed, label=d,
                color=colors[i % len(colors)], linewidth=1.5)

    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Fraction (rolling {window})")
    ax.set_title("Approach Direction Preference Over Time")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Single-condition standard plots
# ---------------------------------------------------------------------------

def plot_success_rate(rolling: np.ndarray,
                     phase_transitions: List[int],
                     phase_names: List[str],
                     savepath: str):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(rolling, linewidth=1.5, color="steelblue")
    for t in phase_transitions:
        ax.axvline(t, color="red", linestyle="--", alpha=0.6, linewidth=1)
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Success Rate (rolling)", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    boundaries = [0] + phase_transitions + [len(rolling)]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0",
              "#00BCD4", "#E91E63"]
    for i in range(len(boundaries) - 1):
        mid = (boundaries[i] + boundaries[i + 1]) / 2
        label = phase_names[i] if i < len(phase_names) else f"Phase {i}"
        ax.text(mid, 0.95, label, ha="center", va="top", fontsize=9,
                fontweight="bold", color=colors[i % len(colors)],
                transform=ax.get_xaxis_transform())

    fig.suptitle("Success Rate Over Training", fontsize=13,
                 fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
