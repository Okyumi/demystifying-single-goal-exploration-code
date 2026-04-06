#!/usr/bin/env python3
"""
Study A: Stochastic Success Isolation.

Same FourRooms maze (11x11), same goal (10,10), same dynamics.
Compares:
  A1-baseline: Deterministic success (p=1.0 from any direction)
  A2-stochastic-allreplay: Direction-dependent success, ALL trajectories in replay
  A2-stochastic-successonly: Direction-dependent success, only successes in replay

3000 episodes each condition, max_steps=120.

Usage:
  python experiments/isolation_study/run_study_a.py --seed 42
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from envs import (
    make_fourrooms_maze, make_stochastic_fourrooms_maze, create_phase0_walls,
)
from agent import (
    TabularSGCRLAgent,
    StochasticSGCRLAgent_AllReplay,
    StochasticSGCRLAgent_SuccessOnly,
)
from metrics import MetricsCollector
from visualization import (
    plot_success_rate, plot_trajectory_overlay,
    render_psi_heatmap_with_route, plot_psi_heatmap_grid_with_routes,
    generate_psi_video_with_routes,
    plot_approach_direction_stats, plot_approach_over_time,
)

CONFIG = {
    "rep_dim": 16,
    "lr_psi": 0.01,
    "batch_size": 128,
    "replay_capacity": 2000,
    "max_steps": 120,
    "gamma": 0.99,
    "entropy_coeff": 0.1,
    "episodes_per_update": 1,
    "normalize": True,
    "total_episodes": 3000,
    "psi_snapshot_freq": 25,
    "policy_snapshot_freq": 25,
    "approach_probs": {
        "from_above": 0.10,
        "from_left":  0.25,
        "from_right": 0.50,
        "from_below": 0.80,
        "stay":       0.00,
    },
}


def agent_kwargs(cfg):
    return {
        "rep_dim": cfg["rep_dim"],
        "lr_psi": cfg["lr_psi"],
        "batch_size": cfg["batch_size"],
        "replay_capacity": cfg["replay_capacity"],
        "max_steps": cfg["max_steps"],
        "gamma": cfg["gamma"],
        "entropy_coeff": cfg["entropy_coeff"],
        "episodes_per_update": cfg["episodes_per_update"],
        "normalize": cfg["normalize"],
    }


# -----------------------------------------------------------------------
# A1: Baseline (deterministic)
# -----------------------------------------------------------------------

def run_a1_baseline(cfg, seed, output_dir):
    print("=" * 60)
    print("Study A — Condition A1: Deterministic Baseline")
    print("=" * 60)
    np.random.seed(seed)

    env = make_fourrooms_maze()
    agent = TabularSGCRLAgent(env, **agent_kwargs(cfg))
    metrics = MetricsCollector(env.num_states, (env.height, env.width))

    total = cfg["total_episodes"]
    t0 = time.time()

    for ep in range(total):
        traj, success = agent.collect_episode()
        metrics.record_episode(ep, 0, traj, success)

        if ep % agent.episodes_per_update == 0:
            agent.update_representations()

        if ep % cfg["psi_snapshot_freq"] == 0:
            sim_map = agent.get_similarity_map()
            metrics.record_psi_snapshot(ep, 0, sim_map)
            metrics.record_psi_full_snapshot(ep, 0, agent.get_psi_snapshot())
            greedy_traj = agent.greedy_rollout()
            metrics.record_greedy_trajectory(ep, 0, greedy_traj)

        if ep % cfg["policy_snapshot_freq"] == 0:
            policy = agent.get_policy_distribution()
            metrics.record_policy_snapshot(ep, 0, policy)

        if (ep + 1) % 500 == 0:
            recent = metrics.episode_success[max(0, ep - 99):ep + 1]
            print(f"  [{ep + 1}/{total}] success(100)={np.mean(recent):.2f} "
                  f"elapsed={time.time() - t0:.1f}s")

    print(f"Done in {time.time() - t0:.1f}s\n")
    _save_results(cfg, env, agent, metrics, output_dir, "A1-baseline")
    return metrics


# -----------------------------------------------------------------------
# A2: Stochastic (AllReplay)
# -----------------------------------------------------------------------

def run_a2_stochastic_allreplay(cfg, seed, output_dir):
    print("=" * 60)
    print("Study A — Condition A2: Stochastic (AllReplay)")
    print("=" * 60)
    np.random.seed(seed)

    env = make_stochastic_fourrooms_maze(cfg["approach_probs"])
    agent = StochasticSGCRLAgent_AllReplay(env, **agent_kwargs(cfg))
    metrics = MetricsCollector(env.num_states, (env.height, env.width))

    total = cfg["total_episodes"]
    approach_history = []
    t0 = time.time()

    for ep in range(total):
        traj, success, approach_dir = agent.collect_episode()
        metrics.record_episode(ep, 0, traj, success)
        metrics.record_approach(ep, approach_dir, success)
        approach_history.append((ep, approach_dir, success))

        if ep % agent.episodes_per_update == 0:
            agent.update_representations()

        if ep % cfg["psi_snapshot_freq"] == 0:
            sim_map = agent.get_similarity_map()
            metrics.record_psi_snapshot(ep, 0, sim_map)
            metrics.record_psi_full_snapshot(ep, 0, agent.get_psi_snapshot())
            greedy_traj = agent.greedy_rollout()
            metrics.record_greedy_trajectory(ep, 0, greedy_traj)

        if ep % cfg["policy_snapshot_freq"] == 0:
            policy = agent.get_policy_distribution()
            metrics.record_policy_snapshot(ep, 0, policy)

        if (ep + 1) % 500 == 0:
            recent = metrics.episode_success[max(0, ep - 99):ep + 1]
            print(f"  [{ep + 1}/{total}] success(100)={np.mean(recent):.2f} "
                  f"elapsed={time.time() - t0:.1f}s")

    print(f"Done in {time.time() - t0:.1f}s\n")
    _save_results(cfg, env, agent, metrics, output_dir,
                  "A2-stochastic-allreplay",
                  approach_history=approach_history,
                  true_probs=cfg["approach_probs"])
    return metrics


# -----------------------------------------------------------------------
# A2: Stochastic (SuccessOnly)
# -----------------------------------------------------------------------

def run_a2_stochastic_successonly(cfg, seed, output_dir):
    print("=" * 60)
    print("Study A — Condition A2: Stochastic (SuccessOnly)")
    print("=" * 60)
    np.random.seed(seed)

    env = make_stochastic_fourrooms_maze(cfg["approach_probs"])
    agent = StochasticSGCRLAgent_SuccessOnly(env, **agent_kwargs(cfg))
    metrics = MetricsCollector(env.num_states, (env.height, env.width))

    total = cfg["total_episodes"]
    approach_history = []
    t0 = time.time()

    for ep in range(total):
        traj, success, approach_dir = agent.collect_episode()
        metrics.record_episode(ep, 0, traj, success)
        metrics.record_approach(ep, approach_dir, success)
        approach_history.append((ep, approach_dir, success))

        if ep % agent.episodes_per_update == 0:
            agent.update_representations()

        if ep % cfg["psi_snapshot_freq"] == 0:
            sim_map = agent.get_similarity_map()
            metrics.record_psi_snapshot(ep, 0, sim_map)
            metrics.record_psi_full_snapshot(ep, 0, agent.get_psi_snapshot())
            greedy_traj = agent.greedy_rollout()
            metrics.record_greedy_trajectory(ep, 0, greedy_traj)

        if ep % cfg["policy_snapshot_freq"] == 0:
            policy = agent.get_policy_distribution()
            metrics.record_policy_snapshot(ep, 0, policy)

        if (ep + 1) % 500 == 0:
            recent = metrics.episode_success[max(0, ep - 99):ep + 1]
            print(f"  [{ep + 1}/{total}] success(100)={np.mean(recent):.2f} "
                  f"elapsed={time.time() - t0:.1f}s")

    print(f"Done in {time.time() - t0:.1f}s\n")
    _save_results(cfg, env, agent, metrics, output_dir,
                  "A2-stochastic-successonly",
                  approach_history=approach_history,
                  true_probs=cfg["approach_probs"])
    return metrics


# -----------------------------------------------------------------------
# Save results
# -----------------------------------------------------------------------

def _save_results(cfg, env, agent, metrics, output_dir, condition_name,
                  approach_history=None, true_probs=None):
    cond_dir = os.path.join(output_dir, condition_name)
    os.makedirs(cond_dir, exist_ok=True)

    with open(os.path.join(cond_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    metrics_dict = metrics.to_dict(true_probs=true_probs)
    with open(os.path.join(cond_dir, "metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=2)

    env_config = env.get_config()
    with open(os.path.join(cond_dir, "env_config.json"), "w") as f:
        json.dump(env_config, f, indent=2)

    np.save(os.path.join(cond_dir, "psi_final.npy"), agent.psi)

    if approach_history:
        ah_data = [{"episode": e, "direction": d, "success": bool(s)}
                   for e, d, s in approach_history]
        with open(os.path.join(cond_dir, "approach_history.json"), "w") as f:
            json.dump(ah_data, f, indent=2)

    # Figures
    fig_dir = os.path.join(cond_dir, "figures")
    print(f"  Generating figures for {condition_name}...")

    # Success rate
    rolling = np.array(metrics_dict["m1_rolling_success_rate"])
    plot_success_rate(rolling, [], [condition_name],
                      os.path.join(fig_dir, "success_rate.png"))

    # Psi heatmap grid with routes
    plot_psi_heatmap_grid_with_routes(
        metrics.psi_snapshots,
        metrics.greedy_trajectory_snapshots,
        env_config,
        os.path.join(fig_dir, "psi_similarity_grid.png"),
        max_frames=16,
    )

    # Trajectory overlay
    h, w = env.height, env.width
    start = np.unravel_index(env.start_state, (h, w))
    goal = np.unravel_index(env.goal_state, (h, w))
    walls = env.walls if hasattr(env, '_walls') else np.array(env_config.get("walls", np.zeros((h, w))))

    succ_coords = []
    succ_flags = []
    for traj, succ in zip(metrics.episode_trajectories[-200:],
                          metrics.episode_success[-200:]):
        coords = [np.unravel_index(s, (h, w)) for s in traj]
        succ_coords.append(coords)
        succ_flags.append(succ)
    plot_trajectory_overlay(
        walls, start, goal, succ_coords, succ_flags,
        title=f"{condition_name}: Last 200 Episodes",
        max_trajs=200,
        savepath=os.path.join(fig_dir, "trajectories_last200.png"),
    )

    # Approach direction stats (stochastic only)
    if approach_history and true_probs and hasattr(agent, 'get_approach_stats'):
        plot_approach_direction_stats(
            agent.get_approach_stats(), true_probs,
            os.path.join(fig_dir, "approach_direction_stats.png"))
        plot_approach_over_time(
            approach_history,
            os.path.join(fig_dir, "approach_over_time.png"), window=100)

    # Video
    print(f"  Generating psi video for {condition_name}...")
    generate_psi_video_with_routes(
        metrics.psi_snapshots,
        metrics.greedy_trajectory_snapshots,
        env_config,
        os.path.join(cond_dir, "psi_evolution.mp4"),
        fps=6,
    )

    print(f"  Results saved to {cond_dir}")

    # Summary
    sr = metrics.m1_success_rate_per_phase()
    for p, rate in sr.items():
        print(f"    Phase {p}: success={rate:.3f}")

    if hasattr(agent, 'get_approach_stats') and agent.approach_counts:
        stats = agent.get_approach_stats()
        print(f"    Approach stats:")
        for d in sorted(stats):
            s = stats[d]
            tp = true_probs.get(d, 0.0) if true_probs else 0.0
            print(f"      {d}: attempts={s['attempts']} "
                  f"empirical={s['empirical_rate']:.3f} true={tp:.3f}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Study A: Stochastic Success Isolation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--conditions", nargs="+",
                        default=["baseline", "allreplay", "successonly"],
                        help="Which conditions to run")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(
        str(SCRIPT_DIR), "results", f"study_a_seed_{args.seed}")

    print(f"\n{'='*60}")
    print(f"STUDY A: Stochastic Success Isolation")
    print(f"Seed: {args.seed}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    if "baseline" in args.conditions:
        run_a1_baseline(CONFIG, args.seed, output_dir)
    if "allreplay" in args.conditions:
        run_a2_stochastic_allreplay(CONFIG, args.seed, output_dir)
    if "successonly" in args.conditions:
        run_a2_stochastic_successonly(CONFIG, args.seed, output_dir)

    print(f"\nAll Study A conditions complete. Results in {output_dir}")


if __name__ == "__main__":
    main()
