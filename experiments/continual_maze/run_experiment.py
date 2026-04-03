#!/usr/bin/env python3
"""
Main experiment runner for the Continual Maze SGCRL experiments.

Usage:
    python experiments/continual_maze/run_experiment.py
    python experiments/continual_maze/run_experiment.py --config experiments/continual_maze/configs/default.json
    python experiments/continual_maze/run_experiment.py --seed 42 --episodes_per_phase 500

The script:
  1. Builds a ContinualMaze with configurable phases
  2. Trains a tabular SGCRL agent, switching maze layouts on schedule
  3. Collects all metrics (success rate, path diversity, ψ-similarity, etc.)
  4. Saves results to JSON + numpy files
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from pathlib import Path

# Ensure the repo root is on the path so the package is importable
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.continual_maze.envs.continual_maze import (
    ContinualMaze, MazePhase, make_default_continual_maze,
    FOUR_ROOMS_STANDARD, FOUR_ROOMS_SHORTCUT, FOUR_ROOMS_REROUTE,
)
from experiments.continual_maze.agent import TabularSGCRLAgent
from experiments.continual_maze.metrics.metrics import MetricsCollector


# -----------------------------------------------------------------------
# Default configuration
# -----------------------------------------------------------------------

DEFAULT_CONFIG = {
    # Agent hyperparameters
    "rep_dim": 16,
    "lr_psi": 1e-2,
    "batch_size": 128,
    "replay_capacity": 1000,
    "max_steps": 50,
    "gamma": 0.99,
    "entropy_coeff": 0.1,
    "episodes_per_update": 1,
    "normalize": True,

    # Phase schedule
    "episodes_per_phase": 500,

    # Experiment control
    "seed": 42,
    "eval_freq": 5,          # run eval episode every N episodes
    "psi_snapshot_freq": 25,  # save ψ snapshot every N episodes
    "clear_replay_on_transition": False,  # whether to empty buffer on phase change

    # Output
    "output_dir": None,  # set below if not provided
}


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def merge_configs(base: dict, overrides: dict) -> dict:
    merged = dict(base)
    for k, v in overrides.items():
        if v is not None:
            merged[k] = v
    return merged


# -----------------------------------------------------------------------
# Experiment runner
# -----------------------------------------------------------------------

def run_experiment(config: dict):
    seed = config["seed"]
    np.random.seed(seed)

    # Build environment
    env = make_default_continual_maze(
        episodes_per_phase=config["episodes_per_phase"])

    # Build agent
    agent = TabularSGCRLAgent(
        env,
        rep_dim=config["rep_dim"],
        lr_psi=config["lr_psi"],
        batch_size=config["batch_size"],
        replay_capacity=config["replay_capacity"],
        max_steps=config["max_steps"],
        gamma=config["gamma"],
        entropy_coeff=config["entropy_coeff"],
        episodes_per_update=config["episodes_per_update"],
        normalize=config["normalize"],
    )

    # Metrics
    metrics = MetricsCollector(
        num_states=env.num_states,
        maze_shape=(env.height, env.width),
    )

    # Total episodes across all phases
    total_episodes = sum(p.num_episodes for p in env.phases)
    phase_boundaries = []
    cumulative = 0
    for p in env.phases:
        phase_boundaries.append(cumulative)
        cumulative += p.num_episodes

    current_phase_idx = 0
    env.set_phase(0)

    # Track ψ at start of each phase for drift measurement
    psi_at_phase_start: dict = {0: agent.get_psi_snapshot()}

    print(f"=== Continual Maze Experiment ===")
    print(f"Seed: {seed}")
    print(f"Total episodes: {total_episodes}")
    print(f"Phases: {[p.name for p in env.phases]}")
    print(f"Episodes per phase: {[p.num_episodes for p in env.phases]}")
    print()

    t0 = time.time()

    for ep in range(total_episodes):
        # Check for phase transition
        next_phase_idx = current_phase_idx
        for i, boundary in enumerate(phase_boundaries):
            if ep >= boundary:
                next_phase_idx = i

        if next_phase_idx != current_phase_idx:
            # "before" = ψ at END of the previous phase (after training)
            psi_before = agent.get_psi_snapshot()
            # "after" = ψ at START of previous phase (before training)
            # Drift = how much ψ changed during the previous phase
            psi_phase_start = psi_at_phase_start.get(current_phase_idx,
                                                      psi_before)
            metrics.record_phase_transition(ep, psi_phase_start, psi_before,
                                            current_phase_idx)

            env.set_phase(next_phase_idx)
            current_phase_idx = next_phase_idx
            psi_at_phase_start[current_phase_idx] = agent.get_psi_snapshot()

            if config["clear_replay_on_transition"]:
                agent.clear_replay()

            print(f"  [Episode {ep}] Phase transition -> "
                  f"'{env.current_phase.name}'")

        # Collect training episode
        traj, success = agent.collect_episode()
        metrics.record_episode(ep, current_phase_idx, traj, success)

        # Contrastive update
        if ep % agent.episodes_per_update == 0:
            agent.update_representations()

        # Periodic eval + ψ snapshot
        if ep % config["eval_freq"] == 0:
            eval_traj, eval_success = agent.run_eval_episode()
            # (eval episodes are recorded separately for logging only)

        if ep % config["psi_snapshot_freq"] == 0:
            sim_map = agent.get_similarity_map()
            metrics.record_psi_snapshot(ep, current_phase_idx, sim_map)

        # Progress report
        if (ep + 1) % 100 == 0:
            recent = metrics.episode_success[max(0, ep - 99):ep + 1]
            rate = np.mean(recent)
            elapsed = time.time() - t0
            print(f"  [Episode {ep + 1}/{total_episodes}]  "
                  f"phase={env.current_phase.name}  "
                  f"success(last100)={rate:.2f}  "
                  f"elapsed={elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    output_dir = config.get("output_dir") or os.path.join(
        str(REPO_ROOT), "experiments", "continual_maze", "results",
        f"seed_{seed}")
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    config_save = dict(config)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_save, f, indent=2)

    # Save metrics
    metrics_dict = metrics.to_dict()
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=2)

    # Save ψ snapshots as numpy arrays (more compact than JSON)
    psi_dir = os.path.join(output_dir, "psi_snapshots")
    os.makedirs(psi_dir, exist_ok=True)
    for ep, phase, sim_map in metrics.psi_snapshots:
        np.save(os.path.join(psi_dir, f"psi_sim_ep{ep:05d}_phase{phase}.npy"),
                sim_map)

    # Save final ψ embeddings
    np.save(os.path.join(output_dir, "psi_final.npy"), agent.psi)

    # Save environment config
    with open(os.path.join(output_dir, "env_config.json"), "w") as f:
        json.dump(env.get_config(), f, indent=2)

    print(f"Results saved to {output_dir}")

    # Print summary
    print("\n=== Results Summary ===")
    sr = metrics.success_rate_per_phase()
    for p_idx, rate in sr.items():
        print(f"  Phase {p_idx} ({env.phases[p_idx].name}): "
              f"success={rate:.3f}")

    adapt = metrics.adaptation_speed()
    for p_idx, steps in adapt.items():
        label = env.phases[p_idx].name if p_idx < len(env.phases) else "?"
        print(f"  Phase {p_idx} ({label}): "
              f"adaptation_speed={steps} episodes")

    drift = metrics.representation_drift()
    for d in drift:
        print(f"  Phase {d['phase_idx']} transition: "
              f"L2={d['mean_l2_distance']:.4f}  "
              f"cos_sim={d['mean_cosine_similarity']:.4f}")

    exploit = metrics.exploitation_ratio()
    for p_idx, ratio in exploit.items():
        label = env.phases[p_idx].name if p_idx < len(env.phases) else "?"
        print(f"  Phase {p_idx} ({label}): "
              f"exploitation_ratio={ratio:.3f}")

    return metrics_dict


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run continual maze SGCRL experiment")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON config file")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episodes_per_phase", type=int, default=None)
    parser.add_argument("--rep_dim", type=int, default=None)
    parser.add_argument("--lr_psi", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--entropy_coeff", type=float, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--clear_replay_on_transition",
                        action="store_true", default=None)
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    if args.config:
        config = merge_configs(config, load_config(args.config))

    # CLI overrides
    cli_overrides = {k: v for k, v in vars(args).items()
                     if k != "config" and v is not None}
    config = merge_configs(config, cli_overrides)

    run_experiment(config)


if __name__ == "__main__":
    main()
