#!/usr/bin/env python3
"""
Study B: Changing Dynamics Isolation.

Goal fixed at (10,10). Compares:
  B1-baseline: Static FourRooms for all 15000 episodes
  B2-changing: 3 phases x 5000 episodes (fourrooms -> corridor_shift -> l_wall)

15000 total episodes, max_steps=120.

Usage:
  python experiments/isolation_study/run_study_b.py --seed 42
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

from envs import make_static_fourrooms, make_changing_dynamics_maze
from agent import TabularSGCRLAgent
from metrics import MetricsCollector
from visualization import (
    plot_success_rate, plot_trajectory_overlay,
    plot_psi_heatmap_grid_with_routes,
    generate_psi_video_with_routes,
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
    "total_episodes": 15000,
    "episodes_per_phase": 5000,
    "psi_snapshot_freq": 50,
    "policy_snapshot_freq": 50,
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
# B1: Static Baseline
# -----------------------------------------------------------------------

def run_b1_baseline(cfg, seed, output_dir):
    print("=" * 60)
    print("Study B — Condition B1: Static FourRooms Baseline")
    print("=" * 60)
    np.random.seed(seed)

    env = make_static_fourrooms()
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
            policy_traj = agent.policy_rollout()
            metrics.record_greedy_trajectory(ep, 0, policy_traj)

        if ep % cfg["policy_snapshot_freq"] == 0:
            policy = agent.get_policy_distribution()
            metrics.record_policy_snapshot(ep, 0, policy)

        if (ep + 1) % 500 == 0:
            recent = metrics.episode_success[max(0, ep - 99):ep + 1]
            print(f"  [{ep + 1}/{total}] success(100)={np.mean(recent):.2f} "
                  f"elapsed={time.time() - t0:.1f}s")

    print(f"Done in {time.time() - t0:.1f}s\n")
    _save_results(cfg, env, agent, metrics, output_dir, "B1-baseline")
    return metrics


# -----------------------------------------------------------------------
# B2: Changing Dynamics
# -----------------------------------------------------------------------

def run_b2_changing(cfg, seed, output_dir):
    print("=" * 60)
    print("Study B — Condition B2: Changing Dynamics (3 phases)")
    print("=" * 60)
    np.random.seed(seed)

    env = make_changing_dynamics_maze(cfg["episodes_per_phase"])
    agent = TabularSGCRLAgent(env, **agent_kwargs(cfg))
    metrics = MetricsCollector(env.num_states, (env.height, env.width))

    total = sum(p.num_episodes for p in env.phases)
    phase_boundaries = []
    cumulative = 0
    for p in env.phases:
        phase_boundaries.append(cumulative)
        cumulative += p.num_episodes

    current_phase_idx = 0
    env.set_phase(0)
    psi_at_phase_start = {0: agent.get_psi_snapshot()}

    print(f"  Phases: {[p.name for p in env.phases]}")
    print(f"  Episodes per phase: {cfg['episodes_per_phase']}")
    t0 = time.time()

    for ep in range(total):
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
            print(f"  [Episode {ep}] Phase -> '{env.current_phase.name}'")

        traj, success = agent.collect_episode()
        metrics.record_episode(ep, current_phase_idx, traj, success)

        if ep % agent.episodes_per_update == 0:
            agent.update_representations()

        if ep % cfg["psi_snapshot_freq"] == 0:
            sim_map = agent.get_similarity_map()
            metrics.record_psi_snapshot(ep, current_phase_idx, sim_map)
            metrics.record_psi_full_snapshot(
                ep, current_phase_idx, agent.get_psi_snapshot())
            policy_traj = agent.policy_rollout()
            metrics.record_greedy_trajectory(ep, current_phase_idx, policy_traj)

        if ep % cfg["policy_snapshot_freq"] == 0:
            policy = agent.get_policy_distribution()
            metrics.record_policy_snapshot(ep, current_phase_idx, policy)

        if (ep + 1) % 500 == 0:
            recent = metrics.episode_success[max(0, ep - 99):ep + 1]
            print(f"  [{ep + 1}/{total}] phase={env.current_phase.name} "
                  f"success(100)={np.mean(recent):.2f} "
                  f"elapsed={time.time() - t0:.1f}s")

    print(f"Done in {time.time() - t0:.1f}s\n")
    _save_results(cfg, env, agent, metrics, output_dir, "B2-changing",
                  phase_names=[p.name for p in env.phases])
    return metrics


# -----------------------------------------------------------------------
# Save results
# -----------------------------------------------------------------------

def _save_results(cfg, env, agent, metrics, output_dir, condition_name,
                  phase_names=None):
    cond_dir = os.path.join(output_dir, condition_name)
    os.makedirs(cond_dir, exist_ok=True)

    with open(os.path.join(cond_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    metrics_dict = metrics.to_dict()
    with open(os.path.join(cond_dir, "metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=2)

    env_config = env.get_config()
    with open(os.path.join(cond_dir, "env_config.json"), "w") as f:
        json.dump(env_config, f, indent=2)

    np.save(os.path.join(cond_dir, "psi_final.npy"), agent.psi)

    fig_dir = os.path.join(cond_dir, "figures")
    pnames = phase_names or [condition_name]
    print(f"  Generating figures for {condition_name}...")

    rolling = np.array(metrics_dict["m1_rolling_success_rate"])
    plot_success_rate(rolling, metrics.phase_transition_episodes, pnames,
                      os.path.join(fig_dir, "success_rate.png"))

    plot_psi_heatmap_grid_with_routes(
        metrics.psi_snapshots,
        metrics.greedy_trajectory_snapshots,
        env_config,
        os.path.join(fig_dir, "psi_similarity_grid.png"),
        max_frames=16,
    )

    # Trajectory overlays
    h, w = env.height, env.width
    start = np.unravel_index(env.start_state, (h, w))
    goal = np.unravel_index(env.goal_state, (h, w))

    # Per-phase trajectories
    phase_ids = sorted(set(metrics.episode_phase))
    for pid in phase_ids:
        if pid < len(env_config.get("phases", [])):
            walls = np.array(env_config["phases"][pid]["walls"])
            pname = env_config["phases"][pid]["name"]
        else:
            walls = np.zeros((h, w), dtype=int)
            pname = f"phase_{pid}"

        trajs_coords = []
        succs = []
        for ep_phase, traj, succ in zip(metrics.episode_phase,
                                        metrics.episode_trajectories,
                                        metrics.episode_success):
            if ep_phase == pid:
                coords = [np.unravel_index(s, (h, w)) for s in traj]
                trajs_coords.append(coords)
                succs.append(succ)

        plot_trajectory_overlay(
            walls, start, goal, trajs_coords, succs,
            title=f"{condition_name} — Phase {pid}: {pname}",
            max_trajs=100,
            savepath=os.path.join(fig_dir, f"trajectories_phase{pid}_{pname}.png"),
        )

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

    sr = metrics.m1_success_rate_per_phase()
    for p, rate in sr.items():
        pn = phase_names[p] if phase_names and p < len(phase_names) else f"phase_{p}"
        print(f"    Phase {p} ({pn}): success={rate:.3f}")

    recovery = metrics.m9_phase_transition_recovery()
    for r in recovery:
        print(f"    Recovery at ep {r['transition_episode']}: "
              f"{r['recovery_episodes']} episodes "
              f"(pre={r['pre_transition_level']:.3f})")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Study B: Changing Dynamics Isolation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--conditions", nargs="+",
                        default=["baseline", "changing"],
                        help="Which conditions to run")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(
        str(SCRIPT_DIR), "results", f"study_b_seed_{args.seed}")

    print(f"\n{'='*60}")
    print(f"STUDY B: Changing Dynamics Isolation")
    print(f"Seed: {args.seed}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    if "baseline" in args.conditions:
        run_b1_baseline(CONFIG, args.seed, output_dir)
    if "changing" in args.conditions:
        run_b2_changing(CONFIG, args.seed, output_dir)

    print(f"\nAll Study B conditions complete. Results in {output_dir}")


if __name__ == "__main__":
    main()
