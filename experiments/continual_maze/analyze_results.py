#!/usr/bin/env python3
"""
Analyse and visualise results from a continual maze experiment.

Usage:
    python experiments/continual_maze/analyze_results.py results/seed_42
    python experiments/continual_maze/analyze_results.py results/seed_42 --no-show
"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path

# Optional: matplotlib may not be installed in headless environments
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend by default
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def load_results(result_dir: str):
    with open(os.path.join(result_dir, "metrics.json")) as f:
        metrics = json.load(f)
    with open(os.path.join(result_dir, "config.json")) as f:
        config = json.load(f)
    env_config_path = os.path.join(result_dir, "env_config.json")
    env_config = None
    if os.path.exists(env_config_path):
        with open(env_config_path) as f:
            env_config = json.load(f)
    return metrics, config, env_config


def print_summary(metrics: dict, config: dict, env_config: dict | None):
    print("=" * 60)
    print("CONTINUAL MAZE EXPERIMENT — RESULTS SUMMARY")
    print("=" * 60)

    print(f"\nSeed: {config.get('seed')}")
    print(f"Total episodes: {metrics['total_episodes']}")
    if env_config:
        phase_names = [p["name"] for p in env_config["phases"]]
        print(f"Phases: {phase_names}")
    print(f"Phase transitions at episodes: "
          f"{metrics['phase_transition_episodes']}")

    # Success rate per phase
    print("\n--- Success Rate per Phase ---")
    sr = metrics["success_rate_per_phase"]
    for p, rate in sorted(sr.items(), key=lambda x: int(x[0])):
        print(f"  Phase {p}: {rate:.3f}")

    # Adaptation speed
    print("\n--- Adaptation Speed (episodes to first success) ---")
    adapt = metrics["adaptation_speed"]
    for p, speed in sorted(adapt.items(), key=lambda x: int(x[0])):
        print(f"  Phase {p}: {speed}")

    # Path diversity
    print("\n--- Path Diversity (successful trajectories) ---")
    pd = metrics["path_diversity_per_phase"]
    for p, info in sorted(pd.items(), key=lambda x: int(x[0])):
        print(f"  Phase {p}: jaccard={info['mean_jaccard']:.3f}  "
              f"clusters={info['num_clusters']}  "
              f"entropy={info['entropy']:.3f}  "
              f"n_success={info['num_successful']}")

    # Exploitation ratio
    print("\n--- Exploitation Ratio (dominant path fraction) ---")
    er = metrics["exploitation_ratio"]
    for p, ratio in sorted(er.items(), key=lambda x: int(x[0])):
        print(f"  Phase {p}: {ratio:.3f}")

    # Representation drift
    print("\n--- Representation Drift (at phase transitions) ---")
    for d in metrics["representation_drift"]:
        print(f"  Phase {d['phase_idx']} transition: "
              f"L2={d['mean_l2_distance']:.4f}  "
              f"cos_sim={d['mean_cosine_similarity']:.4f}")


def plot_results(metrics: dict, config: dict, env_config: dict | None,
                 output_dir: str, show: bool = False):
    if not HAS_MPL:
        print("matplotlib not available — skipping plots")
        return

    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    transitions = metrics["phase_transition_episodes"]
    phase_names = []
    if env_config:
        phase_names = [p["name"] for p in env_config["phases"]]

    # ---- 1. Rolling success rate ----
    fig, ax = plt.subplots(figsize=(10, 4))
    rolling = np.array(metrics["rolling_success_rate"])
    ax.plot(rolling, linewidth=1.5, color="steelblue")
    for t in transitions:
        ax.axvline(t, color="red", linestyle="--", alpha=0.6, linewidth=1)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Rate (rolling)")
    ax.set_title("Success Rate Over Training")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # Add phase labels
    boundaries = [0] + transitions + [len(rolling)]
    for i in range(len(boundaries) - 1):
        mid = (boundaries[i] + boundaries[i + 1]) / 2
        label = phase_names[i] if i < len(phase_names) else f"Phase {i}"
        ax.text(mid, 1.02, label, ha="center", va="bottom", fontsize=9,
                transform=ax.get_xaxis_transform())

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "success_rate.png"), dpi=150)
    if show:
        plt.show()
    plt.close(fig)

    # ---- 2. ψ-similarity heatmaps ----
    snapshots = metrics.get("psi_snapshots", [])
    if snapshots:
        # Pick up to 12 evenly spaced snapshots
        n = len(snapshots)
        indices = np.linspace(0, n - 1, min(n, 12), dtype=int)
        ncols = 4
        nrows = int(np.ceil(len(indices) / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(3.5 * ncols, 3.2 * nrows))
        axes = np.atleast_2d(axes)
        for idx, ax in enumerate(axes.flat):
            if idx >= len(indices):
                ax.axis("off")
                continue
            snap = snapshots[indices[idx]]
            ep = snap["episode"]
            phase = snap["phase"]
            sim_map = np.array(snap["similarity_map"])
            im = ax.imshow(sim_map, cmap="viridis", origin="lower",
                           vmin=min(0, sim_map.min()), vmax=1)
            label = phase_names[phase] if phase < len(phase_names) else f"P{phase}"
            ax.set_title(f"Ep {ep} ({label})", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle("ψ(s)·ψ(g) Similarity Over Training", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "psi_similarity_evolution.png"),
                    dpi=150)
        if show:
            plt.show()
        plt.close(fig)

    # ---- 3. Path diversity bar chart ----
    pd = metrics["path_diversity_per_phase"]
    if pd:
        phases_k = sorted(pd.keys(), key=int)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        labels = [phase_names[int(k)] if int(k) < len(phase_names) else f"P{k}"
                  for k in phases_k]
        x = np.arange(len(phases_k))

        # Jaccard
        vals = [pd[k]["mean_jaccard"] for k in phases_k]
        axes[0].bar(x, vals, color="teal")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels, rotation=30)
        axes[0].set_ylabel("Mean Jaccard Distance")
        axes[0].set_title("Path Diversity (Jaccard)")

        # Clusters
        vals = [pd[k]["num_clusters"] for k in phases_k]
        axes[1].bar(x, vals, color="coral")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=30)
        axes[1].set_ylabel("# Route Clusters")
        axes[1].set_title("Route Clusters")

        # Entropy
        vals = [pd[k]["entropy"] for k in phases_k]
        axes[2].bar(x, vals, color="mediumpurple")
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(labels, rotation=30)
        axes[2].set_ylabel("Shannon Entropy")
        axes[2].set_title("Route Distribution Entropy")

        fig.suptitle("Path Diversity Metrics", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "path_diversity.png"), dpi=150)
        if show:
            plt.show()
        plt.close(fig)

    # ---- 4. Exploitation ratio ----
    er = metrics["exploitation_ratio"]
    if er:
        phases_k = sorted(er.keys(), key=int)
        labels = [phase_names[int(k)] if int(k) < len(phase_names) else f"P{k}"
                  for k in phases_k]
        vals = [er[k] for k in phases_k]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(range(len(vals)), vals, color="darkorange")
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, rotation=30)
        ax.set_ylabel("Exploitation Ratio")
        ax.set_title("Dominant Path Usage per Phase")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "exploitation_ratio.png"), dpi=150)
        if show:
            plt.show()
        plt.close(fig)

    print(f"Figures saved to {fig_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyse continual maze experiment results")
    parser.add_argument("result_dir", type=str,
                        help="Path to results directory (e.g. results/seed_42)")
    parser.add_argument("--show", action="store_true",
                        help="Display plots interactively")
    parser.add_argument("--no-show", action="store_true",
                        help="Suppress interactive display (default)")
    args = parser.parse_args()

    metrics, config, env_config = load_results(args.result_dir)
    print_summary(metrics, config, env_config)
    plot_results(metrics, config, env_config, args.result_dir,
                 show=args.show and not args.no_show)


if __name__ == "__main__":
    main()
