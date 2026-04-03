#!/usr/bin/env python3
"""
Analysis and visualization for continual maze experiment results.

Usage:
    python experiments/continual_maze/analyze_results.py results/run_XXX/
    python experiments/continual_maze/analyze_results.py results/run_XXX/ --no-show
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not found; plots will be skipped.")


# -----------------------------------------------------------------------
# Loaders
# -----------------------------------------------------------------------

def load_results(results_dir: Path) -> Dict:
    """Load summary, per-seed metrics, and psi snapshots."""
    with open(results_dir / "summary.json") as f:
        summary = json.load(f)

    per_seed = []
    for p in sorted(results_dir.glob("metrics_seed_*.json")):
        with open(p) as f:
            per_seed.append(json.load(f))

    psi_snapshots = {}
    for p in sorted(results_dir.glob("psi_snapshots_seed_*.npz")):
        seed = int(p.stem.split("_")[-1])
        data = np.load(p)
        psi_snapshots[seed] = {
            "similarity_maps": data["similarity_maps"],
            "episodes": data["episodes"],
            "phases": data["phases"],
        }

    return {
        "summary": summary,
        "per_seed": per_seed,
        "psi_snapshots": psi_snapshots,
    }


# -----------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------

def plot_success_rate(per_seed: List[Dict], config: Dict,
                      out_dir: Path):
    """Plot per-phase success rate over episodes (smoothed)."""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    window = 50
    epp = config["episodes_per_phase"]

    for ax, key, title in [
        (axes[0], "success_rate", "Training Success Rate"),
        (axes[1], "eval_success_rate", "Evaluation Success Rate"),
    ]:
        for seed_data in per_seed:
            seed = seed_data["seed"]
            per_phase = seed_data["metrics"][key]["per_phase"]

            # We don't have per-episode data in saved metrics,
            # but we have per-phase averages — plot as bars
            phases = sorted(per_phase.keys(), key=int)
            vals = [per_phase[p] for p in phases]
            x = [int(p) for p in phases]
            ax.bar([xi + 0.15 * per_seed.index(seed_data) for xi in x],
                   vals, width=0.15, label=f"seed {seed}", alpha=0.8)

        ax.set_xlabel("Phase")
        ax.set_ylabel("Success Rate")
        ax.set_title(title)
        ax.set_xticks(range(len(phases)))
        ax.set_xticklabels([f"Phase {p}" for p in phases])
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "success_rate.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {out_dir / 'success_rate.png'}")


def plot_exploitation_ratio(per_seed: List[Dict], out_dir: Path):
    """Plot exploitation ratio per phase."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for seed_data in per_seed:
        seed = seed_data["seed"]
        exploit = seed_data["metrics"]["exploitation_ratio"]["per_phase"]
        phases = sorted(exploit.keys(), key=int)
        vals = [exploit[p]["exploitation_ratio"] for p in phases]
        ax.plot([int(p) for p in phases], vals, "o-",
                label=f"seed {seed}", markersize=8)

    ax.set_xlabel("Phase")
    ax.set_ylabel("Exploitation Ratio")
    ax.set_title("Exploitation Ratio per Phase\n"
                 "(fraction of successful episodes using dominant route)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(out_dir / "exploitation_ratio.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {out_dir / 'exploitation_ratio.png'}")


def plot_path_diversity(per_seed: List[Dict], out_dir: Path):
    """Plot path diversity metrics per phase."""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for seed_data in per_seed:
        seed = seed_data["seed"]
        div = seed_data["metrics"]["path_diversity"]["per_phase"]
        phases = sorted(div.keys(), key=int)

        jacc = [div[p]["mean_jaccard"] for p in phases]
        routes = [div[p]["num_distinct_routes"] for p in phases]
        ent = [div[p]["route_entropy"] for p in phases]

        x = [int(p) for p in phases]
        axes[0].plot(x, jacc, "o-", label=f"seed {seed}")
        axes[1].plot(x, routes, "o-", label=f"seed {seed}")
        axes[2].plot(x, ent, "o-", label=f"seed {seed}")

    axes[0].set_title("Mean Pairwise Jaccard Distance")
    axes[1].set_title("Number of Distinct Routes")
    axes[2].set_title("Route Entropy")

    for ax in axes:
        ax.set_xlabel("Phase")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "path_diversity.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {out_dir / 'path_diversity.png'}")


def plot_psi_similarity_evolution(psi_snapshots: Dict, config: Dict,
                                  out_dir: Path):
    """Plot ψ-similarity heatmaps at key training moments."""
    if not HAS_MPL:
        return

    h, w = config["height"], config["width"]

    for seed, data in psi_snapshots.items():
        sim_maps = data["similarity_maps"]
        episodes = data["episodes"]
        phases = data["phases"]

        # Select up to 8 snapshots spread across training
        n_snaps = len(episodes)
        if n_snaps <= 8:
            idxs = list(range(n_snaps))
        else:
            idxs = np.linspace(0, n_snaps - 1, 8, dtype=int).tolist()

        n_plots = len(idxs)
        fig, axes = plt.subplots(1, n_plots, figsize=(3 * n_plots, 3))
        if n_plots == 1:
            axes = [axes]

        for ax_i, snap_i in enumerate(idxs):
            sim = sim_maps[snap_i].reshape(h, w)
            ep = episodes[snap_i]
            ph = phases[snap_i]

            im = axes[ax_i].imshow(sim, cmap="viridis", origin="lower",
                                   vmin=-0.5, vmax=1.0)
            axes[ax_i].set_title(f"Ep {ep}\nPhase {ph}", fontsize=9)
            axes[ax_i].set_xticks([])
            axes[ax_i].set_yticks([])

        fig.suptitle(f"ψ-Similarity Evolution (seed {seed})", fontsize=13)
        fig.colorbar(im, ax=axes, shrink=0.8, label="ψ(s)·ψ(g)")
        plt.tight_layout()
        fig.savefig(out_dir / f"psi_evolution_seed_{seed}.png", dpi=150)
        plt.close(fig)
        print(f"  Saved {out_dir / f'psi_evolution_seed_{seed}.png'}")


def plot_representation_drift(per_seed: List[Dict], out_dir: Path):
    """Plot representation drift at phase transitions."""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for seed_data in per_seed:
        seed = seed_data["seed"]
        drift = seed_data["metrics"]["representation_drift"]
        if not drift:
            continue
        phases = sorted(drift.keys(), key=int)
        l2 = [drift[p]["mean_l2_drift"] for p in phases]
        cos = [drift[p]["mean_cosine_similarity"] for p in phases]

        x = [int(p) for p in phases]
        axes[0].plot(x, l2, "o-", label=f"seed {seed}")
        axes[1].plot(x, cos, "o-", label=f"seed {seed}")

    axes[0].set_title("Mean L2 Drift at Phase Transitions")
    axes[0].set_ylabel("Mean L2 Distance")
    axes[1].set_title("Mean Cosine Similarity at Phase Transitions")
    axes[1].set_ylabel("Cosine Similarity")

    for ax in axes:
        ax.set_xlabel("Phase")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "representation_drift.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {out_dir / 'representation_drift.png'}")


def print_summary(summary: Dict):
    """Print a textual summary of results."""
    print("\n" + "=" * 60)
    print("  EXPERIMENT SUMMARY")
    print("=" * 60)

    print(f"\nSeeds: {summary['seeds']}")
    epp = summary["config"]["episodes_per_phase"]
    print(f"Episodes per phase: {epp}")

    print("\nPer-Phase Training Success Rate:")
    for p, s in summary["per_phase_train_success"].items():
        print(f"  Phase {p}: {s['mean']:.3f} ± {s['std']:.3f} "
              f"(min={s['min']:.3f}, max={s['max']:.3f})")

    print("\nPer-Phase Evaluation Success Rate:")
    for p, s in summary["per_phase_eval_success"].items():
        print(f"  Phase {p}: {s['mean']:.3f} ± {s['std']:.3f}")

    print("\nExploitation Ratio:")
    for p, s in summary["per_phase_exploitation_ratio"].items():
        print(f"  Phase {p}: {s['mean']:.3f} ± {s['std']:.3f}")

    print("\nAdaptation Speed (episodes to first success after transition):")
    for p, s in summary["adaptation_speed"].items():
        mean_str = f"{s['mean']:.1f}" if s["mean"] >= 0 else "never"
        print(f"  Phase {p}: {mean_str} ± {s['std']:.1f}")


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze continual maze experiment results")
    parser.add_argument("results_dir", type=str,
                        help="Path to results directory")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't try to display plots (save only)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: {results_dir} does not exist")
        sys.exit(1)

    data = load_results(results_dir)
    config = data["summary"]["config"]
    per_seed = data["per_seed"]

    # Print summary
    print_summary(data["summary"])

    # Generate plots
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print(f"\nGenerating plots in {plots_dir} ...")
    plot_success_rate(per_seed, config, plots_dir)
    plot_exploitation_ratio(per_seed, plots_dir)
    plot_path_diversity(per_seed, plots_dir)
    plot_psi_similarity_evolution(data["psi_snapshots"], config, plots_dir)
    plot_representation_drift(per_seed, plots_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
