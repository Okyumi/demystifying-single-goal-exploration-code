#!/usr/bin/env python3
"""
Main experiment runner for continual maze SGCRL experiments.

Two experiment modes:
  1. continual — 6-phase maze with changing walls, fixed goal
  2. stochastic — fixed maze, direction-dependent success probability

Usage:
  python experiments/continual_maze/run_experiment.py --mode continual
  python experiments/continual_maze/run_experiment.py --mode stochastic
  python experiments/continual_maze/run_experiment.py --mode continual --episodes_per_phase 800 --seed 42
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.continual_maze.envs.continual_maze import (
    ContinualMaze, MazePhase, make_continual_maze_v2, make_stochastic_maze,
    StochasticSuccessMaze,
)
from experiments.continual_maze.agent import TabularSGCRLAgent, StochasticSGCRLAgent
from experiments.continual_maze.metrics.metrics import MetricsCollector
from experiments.continual_maze.visualization import (
    plot_success_rate, plot_exploitation_ratio, plot_path_diversity,
    plot_adaptation_metrics, plot_psi_similarity_grid,
    plot_per_phase_trajectories, generate_psi_video,
    plot_approach_direction_stats, plot_approach_over_time,
)


# -----------------------------------------------------------------------
# Default configs
# -----------------------------------------------------------------------

CONTINUAL_CONFIG = {
    "mode": "continual",
    "rep_dim": 16,
    "lr_psi": 1e-2,
    "batch_size": 128,
    "replay_capacity": 1500,
    "max_steps": 80,
    "gamma": 0.99,
    "entropy_coeff": 0.1,
    "episodes_per_update": 1,
    "normalize": True,
    "episodes_per_phase": 600,
    "seed": 42,
    "eval_freq": 5,
    "psi_snapshot_freq": 10,
    "policy_snapshot_freq": 25,
    "clear_replay_on_transition": False,
    "output_dir": None,
}

STOCHASTIC_CONFIG = {
    "mode": "stochastic",
    "rep_dim": 16,
    "lr_psi": 1e-2,
    "batch_size": 128,
    "replay_capacity": 1500,
    "max_steps": 80,
    "gamma": 0.99,
    "entropy_coeff": 0.1,
    "episodes_per_update": 1,
    "normalize": True,
    "total_episodes": 4000,
    "seed": 42,
    "eval_freq": 5,
    "psi_snapshot_freq": 10,
    "policy_snapshot_freq": 25,
    "approach_probs": {
        "from_above": 0.10,
        "from_left":  0.25,
        "from_right": 0.50,
        "from_below": 0.80,
        "stay":       0.00,
    },
    "output_dir": None,
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
# Continual experiment
# -----------------------------------------------------------------------

def run_continual(config: dict):
    seed = config["seed"]
    np.random.seed(seed)

    env = make_continual_maze_v2(
        episodes_per_phase=config["episodes_per_phase"])

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

    metrics = MetricsCollector(
        num_states=env.num_states,
        maze_shape=(env.height, env.width),
    )

    total_episodes = sum(p.num_episodes for p in env.phases)
    phase_boundaries = []
    cumulative = 0
    for p in env.phases:
        phase_boundaries.append(cumulative)
        cumulative += p.num_episodes

    current_phase_idx = 0
    env.set_phase(0)
    psi_at_phase_start = {0: agent.get_psi_snapshot()}

    print(f"=== Continual Maze Experiment (v2, 6 phases) ===")
    print(f"Seed: {seed}")
    print(f"Total episodes: {total_episodes}")
    print(f"Phases: {[p.name for p in env.phases]}")
    print(f"Episodes per phase: {config['episodes_per_phase']}")
    print()

    t0 = time.time()

    for ep in range(total_episodes):
        # Phase transition check
        next_phase_idx = current_phase_idx
        for i, boundary in enumerate(phase_boundaries):
            if ep >= boundary:
                next_phase_idx = i

        if next_phase_idx != current_phase_idx:
            psi_before = agent.get_psi_snapshot()
            psi_phase_start = psi_at_phase_start.get(
                current_phase_idx, psi_before)
            metrics.record_phase_transition(
                ep, psi_phase_start, psi_before, current_phase_idx)

            env.set_phase(next_phase_idx)
            current_phase_idx = next_phase_idx
            psi_at_phase_start[current_phase_idx] = agent.get_psi_snapshot()

            if config["clear_replay_on_transition"]:
                agent.clear_replay()

            print(f"  [Episode {ep}] Phase → '{env.current_phase.name}'")

        # Collect episode
        traj, success = agent.collect_episode()
        metrics.record_episode(ep, current_phase_idx, traj, success)

        # Contrastive update
        if ep % agent.episodes_per_update == 0:
            agent.update_representations()

        # Snapshots
        if ep % config["psi_snapshot_freq"] == 0:
            sim_map = agent.get_similarity_map()
            metrics.record_psi_snapshot(ep, current_phase_idx, sim_map)
            metrics.record_psi_full_snapshot(
                ep, current_phase_idx, agent.get_psi_snapshot())

        if ep % config["policy_snapshot_freq"] == 0:
            policy = agent.get_policy_distribution()
            metrics.record_policy_snapshot(ep, current_phase_idx, policy)

        # Progress
        if (ep + 1) % 100 == 0:
            recent = metrics.episode_success[max(0, ep - 99):ep + 1]
            rate = np.mean(recent)
            elapsed = time.time() - t0
            print(f"  [{ep + 1}/{total_episodes}]  "
                  f"phase={env.current_phase.name}  "
                  f"success(100)={rate:.2f}  "
                  f"elapsed={elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")

    # Save results
    output_dir = config.get("output_dir") or os.path.join(
        str(REPO_ROOT), "experiments", "continual_maze", "results",
        f"continual_seed_{seed}")
    _save_continual_results(config, env, agent, metrics, output_dir)

    return metrics


def _save_continual_results(config, env, agent, metrics, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    metrics_dict = metrics.to_dict()
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=2)

    env_config = env.get_config()
    with open(os.path.join(output_dir, "env_config.json"), "w") as f:
        json.dump(env_config, f, indent=2)

    np.save(os.path.join(output_dir, "psi_final.npy"), agent.psi)

    # Figures
    fig_dir = os.path.join(output_dir, "figures")
    phase_names = [p.name for p in env.phases]

    print("\nGenerating figures...")

    plot_success_rate(
        np.array(metrics_dict["rolling_success_rate"]),
        metrics.phase_transition_episodes,
        phase_names,
        os.path.join(fig_dir, "success_rate.png"),
    )

    plot_exploitation_ratio(
        {int(k): v for k, v in metrics_dict["exploitation_ratio"].items()},
        phase_names,
        os.path.join(fig_dir, "exploitation_ratio.png"),
    )

    plot_path_diversity(
        {int(k): v for k, v in metrics_dict["path_diversity_per_phase"].items()},
        phase_names,
        os.path.join(fig_dir, "path_diversity.png"),
    )

    plot_adaptation_metrics(
        metrics_dict["policy_adaptation_index"],
        metrics_dict["representation_adaptation_rate"],
        metrics.phase_transition_episodes,
        phase_names,
        os.path.join(fig_dir, "adaptation_metrics.png"),
    )

    plot_psi_similarity_grid(
        metrics.psi_snapshots, env_config,
        os.path.join(fig_dir, "psi_similarity_grid.png"),
        max_frames=24,
    )

    # Trajectory overlays per phase
    plot_per_phase_trajectories(
        env_config,
        metrics.episode_phase,
        metrics.episode_trajectories,
        metrics.episode_success,
        os.path.join(fig_dir, "trajectories"),
    )

    # ψ video
    print("  Generating ψ-similarity video...")
    generate_psi_video(
        metrics.psi_snapshots, env_config,
        os.path.join(output_dir, "psi_evolution.mp4"),
        fps=6,
    )

    # Print summary
    print(f"\nResults saved to {output_dir}")
    print("\n=== Summary ===")
    sr = metrics.success_rate_per_phase()
    for p_idx, rate in sr.items():
        print(f"  Phase {p_idx} ({phase_names[p_idx]}): success={rate:.3f}")

    adapt = metrics.adaptation_speed()
    for p_idx, steps in adapt.items():
        label = phase_names[p_idx] if p_idx < len(phase_names) else "?"
        print(f"  Phase {p_idx} ({label}): adapt_speed={steps} eps")

    exploit = metrics.exploitation_ratio()
    for p_idx, ratio in exploit.items():
        label = phase_names[p_idx] if p_idx < len(phase_names) else "?"
        print(f"  Phase {p_idx} ({label}): exploit_ratio={ratio:.3f}")

    pai = metrics.policy_adaptation_index()
    if pai:
        # Mean KL at transitions vs within-phase
        trans_kl = [d["kl_divergence"] for d in pai if d["is_phase_transition"]]
        within_kl = [d["kl_divergence"] for d in pai
                     if not d["is_phase_transition"]]
        if trans_kl:
            print(f"\n  Policy KL at transitions: {np.mean(trans_kl):.4f}")
        if within_kl:
            print(f"  Policy KL within phases: {np.mean(within_kl):.4f}")


# -----------------------------------------------------------------------
# Stochastic success experiment
# -----------------------------------------------------------------------

def run_stochastic(config: dict):
    seed = config["seed"]
    np.random.seed(seed)

    env = make_stochastic_maze(
        approach_probs=config.get("approach_probs"),
    )

    agent = StochasticSGCRLAgent(
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

    metrics = MetricsCollector(
        num_states=env.num_states,
        maze_shape=(env.height, env.width),
    )

    total_episodes = config["total_episodes"]
    approach_history = []  # (episode, direction, success)

    print(f"=== Stochastic Success Experiment ===")
    print(f"Seed: {seed}")
    print(f"Total episodes: {total_episodes}")
    print(f"Approach probs: {env.approach_probs}")
    print()

    t0 = time.time()

    for ep in range(total_episodes):
        traj, success, approach_dir = agent.collect_episode()
        metrics.record_episode(ep, 0, traj, success)
        approach_history.append((ep, approach_dir, success))

        if ep % agent.episodes_per_update == 0:
            agent.update_representations()

        if ep % config["psi_snapshot_freq"] == 0:
            sim_map = agent.get_similarity_map()
            metrics.record_psi_snapshot(ep, 0, sim_map)
            metrics.record_psi_full_snapshot(
                ep, 0, agent.get_psi_snapshot())

        if ep % config["policy_snapshot_freq"] == 0:
            policy = agent.get_policy_distribution()
            metrics.record_policy_snapshot(ep, 0, policy)

        if (ep + 1) % 200 == 0:
            recent = metrics.episode_success[max(0, ep - 199):ep + 1]
            rate = np.mean(recent)
            elapsed = time.time() - t0
            print(f"  [{ep + 1}/{total_episodes}]  "
                  f"success(200)={rate:.2f}  "
                  f"elapsed={elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")

    # Save
    output_dir = config.get("output_dir") or os.path.join(
        str(REPO_ROOT), "experiments", "continual_maze", "results",
        f"stochastic_seed_{seed}")
    _save_stochastic_results(config, env, agent, metrics,
                              approach_history, output_dir)


def _save_stochastic_results(config, env, agent, metrics,
                              approach_history, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    metrics_dict = metrics.to_dict()
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=2)

    env_config = env.get_config()
    with open(os.path.join(output_dir, "env_config.json"), "w") as f:
        json.dump(env_config, f, indent=2)

    np.save(os.path.join(output_dir, "psi_final.npy"), agent.psi)

    # Approach history
    ah_data = [{"episode": e, "direction": d, "success": bool(s)}
               for e, d, s in approach_history]
    with open(os.path.join(output_dir, "approach_history.json"), "w") as f:
        json.dump(ah_data, f, indent=2)

    # Figures
    fig_dir = os.path.join(output_dir, "figures")

    print("\nGenerating figures...")

    # Success rate
    plot_success_rate(
        np.array(metrics_dict["rolling_success_rate"]),
        [], ["stochastic"],
        os.path.join(fig_dir, "success_rate.png"),
    )

    # Approach direction analysis
    plot_approach_direction_stats(
        agent.get_approach_stats(),
        env.approach_probs,
        os.path.join(fig_dir, "approach_direction_stats.png"),
    )

    # Approach preference over time
    plot_approach_over_time(
        approach_history,
        os.path.join(fig_dir, "approach_over_time.png"),
        window=100,
    )

    # ψ-similarity grid
    plot_psi_similarity_grid(
        metrics.psi_snapshots, env_config,
        os.path.join(fig_dir, "psi_similarity_grid.png"),
        max_frames=16,
    )

    # Adaptation metrics
    plot_adaptation_metrics(
        metrics_dict["policy_adaptation_index"],
        metrics_dict["representation_adaptation_rate"],
        [], ["stochastic"],
        os.path.join(fig_dir, "adaptation_metrics.png"),
    )

    # Trajectory overlays (sample from all episodes)
    h, w = env.height, env.width
    start = np.unravel_index(env.start_state, (h, w))
    goal = np.unravel_index(env.goal_state, (h, w))

    from experiments.continual_maze.visualization import plot_trajectory_overlay
    # Successful trajectories
    succ_trajs = [(t, s) for t, s in
                  zip(metrics.episode_trajectories, metrics.episode_success)]
    succ_coords = []
    succ_flags = []
    for traj, succ in succ_trajs[-200:]:
        coords = [np.unravel_index(s, (h, w)) for s in traj]
        succ_coords.append(coords)
        succ_flags.append(succ)

    plot_trajectory_overlay(
        env.walls, start, goal, succ_coords, succ_flags,
        title="Stochastic Success: Last 200 Episodes",
        max_trajs=200,
        savepath=os.path.join(fig_dir, "trajectories_last200.png"),
    )

    # ψ video
    print("  Generating ψ-similarity video...")
    generate_psi_video(
        metrics.psi_snapshots, env_config,
        os.path.join(output_dir, "psi_evolution.mp4"),
        fps=6,
    )

    # Summary
    print(f"\nResults saved to {output_dir}")
    print("\n=== Approach Direction Summary ===")
    stats = agent.get_approach_stats()
    for d in sorted(stats.keys()):
        s = stats[d]
        true_p = env.approach_probs.get(d, 0.0)
        print(f"  {d}: attempts={s['attempts']}  "
              f"successes={s['successes']}  "
              f"empirical={s['empirical_rate']:.3f}  "
              f"true={true_p:.3f}")

    overall_sr = np.mean(metrics.episode_success)
    print(f"\n  Overall success rate: {overall_sr:.3f}")

    # Key question: which direction dominates?
    if stats:
        dominant = max(stats, key=lambda d: stats[d]["attempts"])
        print(f"  Most-attempted direction: {dominant} "
              f"({stats[dominant]['attempts']} attempts)")
        best_dir = max(env.approach_probs, key=env.approach_probs.get)
        print(f"  Highest-probability direction: {best_dir} "
              f"(p={env.approach_probs[best_dir]:.2f})")
        if dominant != best_dir:
            print(f"  ** OVEREXPLOITATION DETECTED: agent prefers "
                  f"'{dominant}' over optimal '{best_dir}' **")
        else:
            print(f"  Agent correctly identified the best direction.")


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run continual maze or stochastic success experiment")
    parser.add_argument("--mode", type=str, default="continual",
                        choices=["continual", "stochastic"],
                        help="Experiment mode")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episodes_per_phase", type=int, default=None)
    parser.add_argument("--total_episodes", type=int, default=None)
    parser.add_argument("--rep_dim", type=int, default=None)
    parser.add_argument("--lr_psi", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--entropy_coeff", type=float, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--clear_replay_on_transition",
                        action="store_true", default=None)
    args = parser.parse_args()

    if args.mode == "stochastic":
        config = dict(STOCHASTIC_CONFIG)
    else:
        config = dict(CONTINUAL_CONFIG)

    if args.config:
        config = merge_configs(config, load_config(args.config))

    cli_overrides = {k: v for k, v in vars(args).items()
                     if k != "config" and v is not None}
    config = merge_configs(config, cli_overrides)

    if config["mode"] == "stochastic":
        run_stochastic(config)
    else:
        run_continual(config)


if __name__ == "__main__":
    main()
