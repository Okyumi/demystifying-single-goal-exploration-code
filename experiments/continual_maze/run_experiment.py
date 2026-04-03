#!/usr/bin/env python3
"""
Main experiment runner for the Continual Maze SGCRL experiment.

Usage:
    python experiments/continual_maze/run_experiment.py
    python experiments/continual_maze/run_experiment.py --config experiments/continual_maze/configs/quick.json
    python experiments/continual_maze/run_experiment.py --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.continual_maze.envs.continual_maze import (
    ContinualMaze,
    MazePhase,
    build_blocked_walls,
    build_fourrooms_walls,
    build_shortcut_walls,
)
from experiments.continual_maze.metrics.metrics import compute_all_metrics
from experiments.continual_maze.sgcrl_agent import SGCRLAgent


# -----------------------------------------------------------------------
# Default configuration
# -----------------------------------------------------------------------

DEFAULT_CONFIG = {
    # Environment
    "height": 10,
    "width": 10,
    "episodes_per_phase": 500,
    # Agent
    "rep_dim": 16,
    "lr_psi": 1e-2,
    "batch_size": 128,
    "replay_capacity": 1000,
    "max_steps": 50,
    "gamma": 0.99,
    "entropy_coeff": 0.1,
    "episodes_per_upd": 1,
    "eval_freq": 1,
    "norm": True,
    "clear_replay_on_phase_change": False,
    # Experiment
    "seeds": [42],
}


# -----------------------------------------------------------------------
# Experiment
# -----------------------------------------------------------------------

def build_env(config: Dict) -> ContinualMaze:
    """Build the 3-phase ContinualMaze from config."""
    h, w = config["height"], config["width"]
    epp = config["episodes_per_phase"]

    phases = [
        MazePhase(
            name="fourrooms",
            walls=build_fourrooms_walls(h, w),
            num_episodes=epp,
            description="Standard FourRooms layout",
        ),
        MazePhase(
            name="shortcut",
            walls=build_shortcut_walls(h, w),
            num_episodes=epp,
            description="Shortcut opened in vertical wall (bottom half)",
        ),
        MazePhase(
            name="blocked",
            walls=build_blocked_walls(h, w),
            num_episodes=epp,
            description="Bottom path blocked, top path opened",
        ),
    ]

    start = int(np.ravel_multi_index((0, 0), (h, w)))
    goal = int(np.ravel_multi_index((h - 1, w - 1), (h, w)))

    return ContinualMaze(
        height=h, width=w,
        start_state=start, goal_state=goal,
        phases=phases,
    )


def run_single_seed(config: Dict, seed: int, verbose: bool = True) -> Dict:
    """Run one experiment with a given seed.  Returns results + metrics."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Seed {seed}")
        print(f"{'='*60}")

    env = build_env(config)
    agent = SGCRLAgent(
        env=env,
        rep_dim=config["rep_dim"],
        lr_psi=config["lr_psi"],
        batch_size=config["batch_size"],
        replay_capacity=config["replay_capacity"],
        max_steps=config["max_steps"],
        gamma=config["gamma"],
        entropy_coeff=config["entropy_coeff"],
        episodes_per_upd=config["episodes_per_upd"],
        eval_freq=config["eval_freq"],
        norm=config["norm"],
        clear_replay_on_phase_change=config["clear_replay_on_phase_change"],
        seed=seed,
    )

    t0 = time.time()
    results = agent.train(verbose=verbose)
    elapsed = time.time() - t0

    if verbose:
        print(f"  Training completed in {elapsed:.1f}s")

    metrics = compute_all_metrics(
        results,
        maze_shape=(config["height"], config["width"]),
        goal_state=env.goal_state,
    )

    return {
        "seed": seed,
        "config": config,
        "metrics": metrics,
        "elapsed_s": elapsed,
        "phase_log": [(int(e), int(p)) for e, p in env.phase_log],
    }


def save_results(all_results: List[Dict], out_dir: Path):
    """Save experiment results to disk."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save per-seed results (metrics only — trajectories are too large)
    for res in all_results:
        seed = res["seed"]
        fname = out_dir / f"metrics_seed_{seed}.json"

        # Strip non-serialisable numpy from psi_similarity snapshots
        metrics_copy = _make_serialisable(res["metrics"])
        payload = {
            "seed": seed,
            "config": res["config"],
            "metrics": metrics_copy,
            "elapsed_s": res["elapsed_s"],
            "phase_log": res["phase_log"],
        }
        with open(fname, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  Saved {fname}")

    # Save psi similarity snapshots as numpy arrays
    for res in all_results:
        seed = res["seed"]
        snaps = res["metrics"]["psi_similarity"]["snapshots"]
        sim_maps = np.array([s["similarity_map"] for s in snaps])
        episodes = np.array([s["episode"] for s in snaps])
        phases = np.array([s["phase_idx"] for s in snaps])
        np.savez_compressed(
            out_dir / f"psi_snapshots_seed_{seed}.npz",
            similarity_maps=sim_maps,
            episodes=episodes,
            phases=phases,
        )

    # Save aggregated summary
    summary = _aggregate_summary(all_results)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved {out_dir / 'summary.json'}")


def _make_serialisable(obj):
    """Recursively convert numpy types to Python types."""
    if isinstance(obj, dict):
        return {k: _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serialisable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _aggregate_summary(all_results: List[Dict]) -> Dict:
    """Create a cross-seed summary of key metrics."""
    seeds = [r["seed"] for r in all_results]
    config = all_results[0]["config"]

    # Gather per-phase success rates across seeds
    phase_successes = {}
    for r in all_results:
        for p, v in r["metrics"]["success_rate"]["per_phase"].items():
            phase_successes.setdefault(str(p), []).append(v)

    phase_eval_successes = {}
    for r in all_results:
        for p, v in r["metrics"]["eval_success_rate"]["per_phase"].items():
            phase_eval_successes.setdefault(str(p), []).append(v)

    # Gather exploitation ratios
    phase_exploit = {}
    for r in all_results:
        for p, v in r["metrics"]["exploitation_ratio"]["per_phase"].items():
            phase_exploit.setdefault(str(p), []).append(v["exploitation_ratio"])

    # Gather adaptation speed
    adaptation = {}
    for r in all_results:
        for p, v in r["metrics"]["adaptation_speed"].items():
            val = v["episodes_to_first_success"]
            adaptation.setdefault(str(p), []).append(val if val is not None else -1)

    def _stats(vals):
        arr = np.array(vals, dtype=float)
        return {"mean": float(arr.mean()), "std": float(arr.std()),
                "min": float(arr.min()), "max": float(arr.max())}

    return {
        "seeds": seeds,
        "config": config,
        "per_phase_train_success": {p: _stats(v) for p, v in phase_successes.items()},
        "per_phase_eval_success": {p: _stats(v) for p, v in phase_eval_successes.items()},
        "per_phase_exploitation_ratio": {p: _stats(v) for p, v in phase_exploit.items()},
        "adaptation_speed": {p: _stats(v) for p, v in adaptation.items()},
    }


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run continual maze SGCRL experiment")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON config file")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Random seeds (overrides config)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    # Load / merge config
    config = dict(DEFAULT_CONFIG)
    if args.config:
        with open(args.config) as f:
            config.update(json.load(f))
    if args.seeds:
        config["seeds"] = args.seeds

    # Output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(__file__).parent / "results" / f"run_{int(time.time())}"

    verbose = not args.quiet

    if verbose:
        print("Continual Maze SGCRL Experiment")
        print(f"  Phases: {config['episodes_per_phase']} episodes each")
        print(f"  Seeds: {config['seeds']}")
        print(f"  Output: {out_dir}")

    # Save config
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Run experiments
    all_results = []
    for seed in config["seeds"]:
        res = run_single_seed(config, seed, verbose=verbose)
        all_results.append(res)

    # Save
    save_results(all_results, out_dir)

    if verbose:
        print(f"\nAll results saved to {out_dir}")
        # Print quick summary
        summary = _aggregate_summary(all_results)
        print("\n--- Quick Summary ---")
        for p, stats in summary["per_phase_train_success"].items():
            print(f"  Phase {p} train success: "
                  f"{stats['mean']:.3f} ± {stats['std']:.3f}")
        for p, stats in summary["per_phase_eval_success"].items():
            print(f"  Phase {p} eval success:  "
                  f"{stats['mean']:.3f} ± {stats['std']:.3f}")
        for p, stats in summary["per_phase_exploitation_ratio"].items():
            print(f"  Phase {p} exploit ratio: "
                  f"{stats['mean']:.3f} ± {stats['std']:.3f}")
        for p, stats in summary["adaptation_speed"].items():
            print(f"  Phase {p} adapt speed:   "
                  f"{stats['mean']:.1f} ± {stats['std']:.1f} episodes")


if __name__ == "__main__":
    main()
