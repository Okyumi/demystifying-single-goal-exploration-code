"""
Visualization for continual maze experiments.

Produces:
1. Trajectory overlay plots (paths on maze grid)
2. ψ-similarity evolution video (heatmaps over training)
3. Standard metric figures (success rate, exploitation, etc.)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from typing import List, Tuple, Dict, Any, Optional
import subprocess


# ---------------------------------------------------------------------------
# Maze grid renderer
# ---------------------------------------------------------------------------

def _draw_maze(ax, walls: np.ndarray, start: Tuple[int, int],
               goal: Tuple[int, int], title: str = ""):
    """Draw a maze grid with walls, start (green), and goal (red)."""
    h, w = walls.shape
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10, pad=8)

    # Draw walls
    for r in range(h):
        for c in range(w):
            if walls[r, c] == 1:
                ax.add_patch(patches.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    facecolor="#333333", edgecolor="#444444", linewidth=0.5))

    # Grid lines
    for r in range(h + 1):
        ax.axhline(r - 0.5, color="#cccccc", linewidth=0.3)
    for c in range(w + 1):
        ax.axvline(c - 0.5, color="#cccccc", linewidth=0.3)

    # Start and goal
    sr, sc = start
    gr, gc = goal
    ax.plot(sc, sr, "o", color="#2ecc71", markersize=10, zorder=5, label="Start")
    ax.plot(gc, gr, "*", color="#e74c3c", markersize=14, zorder=5, label="Goal")

    ax.set_xticks([])
    ax.set_yticks([])


def _draw_trajectory(ax, traj_coords: List[Tuple[int, int]],
                     color: str = "blue", alpha: float = 0.4,
                     linewidth: float = 1.5):
    """Overlay a trajectory on an existing maze plot."""
    rows = [c[0] for c in traj_coords]
    cols = [c[1] for c in traj_coords]
    ax.plot(cols, rows, "-", color=color, alpha=alpha, linewidth=linewidth,
            zorder=3)


# ---------------------------------------------------------------------------
# 1. Trajectory overlay plots
# ---------------------------------------------------------------------------

def plot_trajectory_overlay(walls: np.ndarray,
                            start: Tuple[int, int],
                            goal: Tuple[int, int],
                            trajectories: List[List[Tuple[int, int]]],
                            successes: List[bool],
                            title: str = "",
                            max_trajs: int = 100,
                            savepath: Optional[str] = None):
    """Plot maze with trajectories overlaid. Green=success, red=failure."""
    fig, ax = plt.subplots(figsize=(7, 7))
    _draw_maze(ax, walls, start, goal, title)

    # Sample if too many
    indices = list(range(len(trajectories)))
    if len(indices) > max_trajs:
        indices = list(np.random.choice(indices, max_trajs, replace=False))

    for idx in indices:
        traj = trajectories[idx]
        color = "#27ae60" if successes[idx] else "#c0392b"
        alpha = 0.5 if successes[idx] else 0.15
        _draw_trajectory(ax, traj, color=color, alpha=alpha)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#27ae60", linewidth=2, label="Success"),
        Line2D([0], [0], color="#c0392b", linewidth=2, alpha=0.3,
               label="Failure"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    fig.tight_layout()
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_per_phase_trajectories(env_config: dict,
                                episode_phases: List[int],
                                episode_trajectories: List[List[int]],
                                episode_successes: List[bool],
                                savedir: str,
                                max_per_phase: int = 80):
    """Generate one trajectory overlay plot per phase."""
    h = env_config["height"]
    w = env_config["width"]
    start = np.unravel_index(env_config["start_state"], (h, w))
    goal = np.unravel_index(env_config["goal_state"], (h, w))

    phases = env_config.get("phases", [])
    phase_ids = sorted(set(episode_phases))

    for pid in phase_ids:
        # Get walls for this phase
        if pid < len(phases):
            walls = np.array(phases[pid]["walls"])
            name = phases[pid]["name"]
        else:
            walls = np.zeros((h, w), dtype=int)
            name = f"phase_{pid}"

        # Collect trajectories for this phase
        trajs_coords = []
        succs = []
        for ep_phase, traj, succ in zip(
                episode_phases, episode_trajectories, episode_successes):
            if ep_phase == pid:
                coords = [np.unravel_index(s, (h, w)) for s in traj]
                trajs_coords.append(coords)
                succs.append(succ)

        plot_trajectory_overlay(
            walls, start, goal, trajs_coords, succs,
            title=f"Phase {pid}: {name}",
            max_trajs=max_per_phase,
            savepath=os.path.join(savedir, f"trajectories_phase{pid}_{name}.png"),
        )


# ---------------------------------------------------------------------------
# 2. ψ-similarity evolution video
# ---------------------------------------------------------------------------

def render_psi_frame(sim_map: np.ndarray, walls: np.ndarray,
                     start: Tuple[int, int], goal: Tuple[int, int],
                     episode: int, phase_name: str,
                     vmin: float = -1.0, vmax: float = 1.0) -> plt.Figure:
    """Render a single frame of the ψ-similarity video."""
    fig, ax = plt.subplots(figsize=(6, 6))
    h, w = sim_map.shape

    # Mask walls
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

    sr, sc = start
    gr, gc = goal
    ax.plot(sc, sr, "o", color="#2196F3", markersize=10, zorder=5)
    ax.plot(gc, gr, "*", color="#FF5722", markersize=14, zorder=5)

    ax.set_title(f"ψ-similarity  |  Ep {episode}  |  {phase_name}",
                 fontsize=11, pad=10)
    ax.set_xticks([])
    ax.set_yticks([])

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("ψ(s)·ψ(g)", fontsize=9)

    fig.tight_layout()
    return fig


def generate_psi_video(psi_snapshots: List[Tuple[int, int, np.ndarray]],
                       env_config: dict,
                       output_path: str,
                       fps: int = 4):
    """Generate an MP4 video of ψ-similarity evolution over training.

    Each frame is a heatmap of ψ(s)·ψ(g) for all states.
    """
    h = env_config["height"]
    w = env_config["width"]
    start = np.unravel_index(env_config["start_state"], (h, w))
    goal = np.unravel_index(env_config["goal_state"], (h, w))
    phases = env_config.get("phases", [])

    # Render frames to temp directory
    frame_dir = output_path + "_frames"
    os.makedirs(frame_dir, exist_ok=True)

    # Find global min/max for consistent colormap
    all_sims = np.concatenate([sm.ravel() for _, _, sm in psi_snapshots])
    vmin = float(np.percentile(all_sims, 2))
    vmax = float(np.percentile(all_sims, 98))

    for idx, (ep, phase, sim_map) in enumerate(psi_snapshots):
        if phase < len(phases):
            walls = np.array(phases[phase]["walls"])
            name = phases[phase]["name"]
        else:
            walls = np.zeros((h, w), dtype=int)
            name = f"phase_{phase}"

        fig = render_psi_frame(sim_map, walls, start, goal, ep, name,
                               vmin=vmin, vmax=vmax)
        fig.savefig(os.path.join(frame_dir, f"frame_{idx:05d}.png"),
                    dpi=100, bbox_inches="tight")
        plt.close(fig)

    # Encode video with ffmpeg
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

    # Clean up frames
    for f in os.listdir(frame_dir):
        os.remove(os.path.join(frame_dir, f))
    os.rmdir(frame_dir)

    print(f"  Video saved to {output_path}")


# ---------------------------------------------------------------------------
# 3. Standard metric figures
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
              "#00BCD4", "#E91E63", "#795548", "#607D8B"]
    for i in range(len(boundaries) - 1):
        mid = (boundaries[i] + boundaries[i + 1]) / 2
        label = phase_names[i] if i < len(phase_names) else f"Phase {i}"
        ax.text(mid, 0.95, label, ha="center", va="top", fontsize=9,
                fontweight="bold",
                color=colors[i % len(colors)],
                transform=ax.get_xaxis_transform())

    fig.suptitle("Success Rate Over Training", fontsize=13,
                 fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_exploitation_ratio(exploit_per_phase: Dict[int, float],
                            phase_names: List[str],
                            savepath: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    phases = sorted(exploit_per_phase.keys())
    labels = [phase_names[p] if p < len(phase_names) else f"Phase {p}"
              for p in phases]
    vals = [exploit_per_phase[p] for p in phases]
    ax.bar(labels, vals, color="#FF9800")
    ax.set_ylabel("Exploitation Ratio")
    ax.set_title("Dominant Path Usage per Phase")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_path_diversity(diversity: Dict[int, Dict[str, float]],
                        phase_names: List[str],
                        savepath: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    phases = sorted(diversity.keys())
    labels = [phase_names[p] if p < len(phase_names) else f"P{p}"
              for p in phases]

    axes[0].bar(labels, [diversity[p]["mean_jaccard"] for p in phases],
                color="#009688")
    axes[0].set_ylabel("Mean Jaccard Distance")
    axes[0].set_title("Path Diversity (Jaccard)")

    axes[1].bar(labels, [diversity[p]["num_clusters"] for p in phases],
                color="#FF7043")
    axes[1].set_ylabel("# Route Clusters")
    axes[1].set_title("Route Clusters")

    axes[2].bar(labels, [diversity[p]["entropy"] for p in phases],
                color="#7E57C2")
    axes[2].set_ylabel("Shannon Entropy")
    axes[2].set_title("Route Distribution Entropy")

    fig.suptitle("Path Diversity Metrics", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_adaptation_metrics(policy_ai: List[Dict],
                            rep_rate: List[Dict],
                            phase_transitions: List[int],
                            phase_names: List[str],
                            savepath: str):
    """Plot policy adaptation index and representation adaptation rate."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Policy KL divergence
    if policy_ai:
        eps = [(d["episode_from"] + d["episode_to"]) / 2 for d in policy_ai]
        kls = [d["kl_divergence"] for d in policy_ai]
        ax1.plot(eps, kls, linewidth=1.5, color="#E91E63")
        ax1.set_ylabel("KL Divergence", fontsize=11)
        ax1.set_title("Policy Adaptation Index (KL between successive snapshots)",
                       fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Representation adaptation rate
    if rep_rate:
        eps = [(d["episode_from"] + d["episode_to"]) / 2 for d in rep_rate]
        l2s = [d["mean_l2_rate"] for d in rep_rate]
        ax2.plot(eps, l2s, linewidth=1.5, color="#3F51B5")
        ax2.set_ylabel("Mean L2 Change", fontsize=11)
        ax2.set_title("Representation Adaptation Rate (ψ L2 between snapshots)",
                       fontsize=11)
    ax2.set_xlabel("Episode", fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Phase transitions
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0",
              "#00BCD4", "#E91E63", "#795548", "#607D8B"]
    for ax in (ax1, ax2):
        for t in phase_transitions:
            ax.axvline(t, color="red", linestyle="--", alpha=0.6, linewidth=1)

    fig.tight_layout()
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_psi_similarity_grid(psi_snapshots: List[Tuple[int, int, np.ndarray]],
                             env_config: dict,
                             savepath: str,
                             max_frames: int = 16):
    """Grid of ψ-similarity heatmaps sampled across training."""
    n = len(psi_snapshots)
    if n == 0:
        return
    step = max(1, n // max_frames)
    selected = psi_snapshots[::step][:max_frames]

    h = env_config["height"]
    w = env_config["width"]
    phases = env_config.get("phases", [])

    ncols = min(4, len(selected))
    nrows = (len(selected) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    all_sims = np.concatenate([sm.ravel() for _, _, sm in selected])
    vmin = float(np.percentile(all_sims, 2))
    vmax = float(np.percentile(all_sims, 98))

    for idx, (ep, phase, sm) in enumerate(selected):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]

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

    # Hide unused axes
    for idx in range(len(selected), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis("off")

    fig.suptitle("ψ(s)·ψ(g) Similarity Over Training",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Stochastic experiment specific
# ---------------------------------------------------------------------------

def plot_approach_direction_stats(approach_stats: Dict[str, Dict],
                                  true_probs: Dict[str, float],
                                  savepath: str):
    """Bar chart of approach direction statistics."""
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
    ax2.bar(x - width / 2, true_rates, width, label="True prob",
            color="#66BB6A")
    ax2.bar(x + width / 2, emp_rates, width, label="Empirical rate",
            color="#EF5350")
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


def plot_approach_over_time(approach_history: List[Tuple[int, str, bool]],
                            savepath: str,
                            window: int = 50):
    """Rolling fraction of each approach direction over training."""
    if not approach_history:
        return

    dirs = sorted(set(d for _, d, _ in approach_history if d is not None))
    episodes = [e for e, d, s in approach_history]
    n = len(episodes)

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]

    for i, d in enumerate(dirs):
        # Binary: was this direction used in each episode?
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
