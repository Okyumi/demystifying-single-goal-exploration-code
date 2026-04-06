#!/usr/bin/env python3
"""
Fast all-in-one runner for isolation studies A and B.

Strategy: run ALL training first (fast), save lightweight snapshots,
then generate all visuals at the end. This avoids matplotlib overhead
during the training loop.

Usage:
  python experiments/isolation_study/run_all_fast.py --seed 42
"""

import argparse
import json
import os
import sys
import time
import pickle
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from envs import (
    make_fourrooms_maze, make_stochastic_fourrooms_maze,
    make_static_fourrooms, make_changing_dynamics_maze,
    create_phase0_walls,
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

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------

STUDY_A_EPISODES = 10000
STUDY_B_EPISODES_PER_PHASE = 5000
PSI_SNAPSHOT_FREQ = 100      # less frequent = much faster
POLICY_SNAPSHOT_FREQ = 100
VIDEO_MAX_FRAMES = 60        # cap video frames

AGENT_CFG = {
    "rep_dim": 16,
    "lr_psi": 0.01,
    "batch_size": 128,
    "replay_capacity": 2000,
    "max_steps": 120,
    "gamma": 0.99,
    "entropy_coeff": 0.1,
    "episodes_per_update": 1,
    "normalize": True,
}

APPROACH_PROBS = {
    "from_above": 0.10,
    "from_left":  0.25,
    "from_right": 0.50,
    "from_below": 0.80,
    "stay":       0.00,
}


def agent_kwargs():
    return {k: v for k, v in AGENT_CFG.items()}


# -----------------------------------------------------------------------
# Training (no visualization, fast)
# -----------------------------------------------------------------------

def train_condition(agent, env, total_episodes, metrics, label,
                    is_stochastic=False, approach_history=None):
    """Pure training loop. No matplotlib calls."""
    t0 = time.time()
    for ep in range(total_episodes):
        if is_stochastic:
            traj, success, approach_dir = agent.collect_episode()
            metrics.record_approach(ep, approach_dir, success)
            if approach_history is not None:
                approach_history.append((ep, approach_dir, success))
        else:
            traj, success = agent.collect_episode()

        metrics.record_episode(ep, 0, traj, success)

        if ep % agent.episodes_per_update == 0:
            agent.update_representations()

        if ep % PSI_SNAPSHOT_FREQ == 0:
            sim_map = agent.get_similarity_map()
            metrics.record_psi_snapshot(ep, 0, sim_map)
            metrics.record_psi_full_snapshot(ep, 0, agent.get_psi_snapshot())
            policy_traj = agent.policy_rollout(n_rollouts=5)
            metrics.record_greedy_trajectory(ep, 0, policy_traj)

        if ep % POLICY_SNAPSHOT_FREQ == 0:
            policy = agent.get_policy_distribution()
            metrics.record_policy_snapshot(ep, 0, policy)

        if (ep + 1) % 1000 == 0:
            recent = metrics.episode_success[max(0, ep - 99):ep + 1]
            elapsed = time.time() - t0
            print(f"  [{label}] {ep+1}/{total_episodes} "
                  f"success(100)={np.mean(recent):.2f} "
                  f"elapsed={elapsed:.1f}s")

    print(f"  [{label}] Done in {time.time() - t0:.1f}s")
    return metrics


def train_condition_b2(agent, env, metrics, label, episodes_per_phase):
    """Training loop for B2-changing with phase transitions."""
    t0 = time.time()
    total = env.total_episodes
    current_phase_idx = 0

    for ep in range(total):
        new_phase = env.get_phase_for_episode(ep)
        if new_phase != current_phase_idx:
            current_phase_idx = new_phase
            agent.set_walls(env.current_walls)
            print(f"  [{label}] Phase transition at ep {ep} -> "
                  f"{env.phases[current_phase_idx].name}")

        env.set_episode(ep)
        traj, success = agent.collect_episode()
        metrics.record_episode(ep, current_phase_idx, traj, success)

        if ep % agent.episodes_per_update == 0:
            agent.update_representations()

        if ep % PSI_SNAPSHOT_FREQ == 0:
            sim_map = agent.get_similarity_map()
            metrics.record_psi_snapshot(ep, current_phase_idx, sim_map)
            metrics.record_psi_full_snapshot(
                ep, current_phase_idx, agent.get_psi_snapshot())
            policy_traj = agent.policy_rollout(n_rollouts=5)
            metrics.record_greedy_trajectory(
                ep, current_phase_idx, policy_traj)

        if ep % POLICY_SNAPSHOT_FREQ == 0:
            policy = agent.get_policy_distribution()
            metrics.record_policy_snapshot(ep, current_phase_idx, policy)

        if (ep + 1) % 1000 == 0:
            recent = metrics.episode_success[max(0, ep - 99):ep + 1]
            elapsed = time.time() - t0
            print(f"  [{label}] {ep+1}/{total} "
                  f"success(100)={np.mean(recent):.2f} "
                  f"elapsed={elapsed:.1f}s")

    print(f"  [{label}] Done in {time.time() - t0:.1f}s")
    return metrics


def train_condition_b1(agent, env, metrics, label, total_episodes):
    """Training loop for B1-static baseline (single phase)."""
    t0 = time.time()
    for ep in range(total_episodes):
        traj, success = agent.collect_episode()
        metrics.record_episode(ep, 0, traj, success)

        if ep % agent.episodes_per_update == 0:
            agent.update_representations()

        if ep % PSI_SNAPSHOT_FREQ == 0:
            sim_map = agent.get_similarity_map()
            metrics.record_psi_snapshot(ep, 0, sim_map)
            metrics.record_psi_full_snapshot(ep, 0, agent.get_psi_snapshot())
            policy_traj = agent.policy_rollout(n_rollouts=5)
            metrics.record_greedy_trajectory(ep, 0, policy_traj)

        if ep % POLICY_SNAPSHOT_FREQ == 0:
            policy = agent.get_policy_distribution()
            metrics.record_policy_snapshot(ep, 0, policy)

        if (ep + 1) % 1000 == 0:
            recent = metrics.episode_success[max(0, ep - 99):ep + 1]
            elapsed = time.time() - t0
            print(f"  [{label}] {ep+1}/{total_episodes} "
                  f"success(100)={np.mean(recent):.2f} "
                  f"elapsed={elapsed:.1f}s")

    print(f"  [{label}] Done in {time.time() - t0:.1f}s")
    return metrics


# -----------------------------------------------------------------------
# Visualization (runs after all training)
# -----------------------------------------------------------------------

def save_results(cfg, env, agent, metrics, cond_dir, condition_name,
                 approach_history=None, true_probs=None,
                 phase_transitions=None, phase_names=None):
    """Save all results and generate all figures/videos for one condition."""
    os.makedirs(cond_dir, exist_ok=True)

    # Save config
    with open(os.path.join(cond_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # Save metrics
    metrics_dict = metrics.to_dict(true_probs=true_probs)
    with open(os.path.join(cond_dir, "metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=2)

    # Save env config
    env_config = env.get_config()
    with open(os.path.join(cond_dir, "env_config.json"), "w") as f:
        json.dump(env_config, f, indent=2)

    # Save final psi
    np.save(os.path.join(cond_dir, "psi_final.npy"), agent.psi)

    # Save approach history if applicable
    if approach_history:
        ah_data = [{"episode": e, "direction": d, "success": bool(s)}
                   for e, d, s in approach_history]
        with open(os.path.join(cond_dir, "approach_history.json"), "w") as f:
            json.dump(ah_data, f, indent=2)

    # --- Generate figures ---
    fig_dir = os.path.join(cond_dir, "figures")
    print(f"  Generating figures for {condition_name}...")

    # Success rate
    rolling = np.array(metrics_dict["m1_rolling_success_rate"])
    pt = phase_transitions or []
    pn = phase_names or [condition_name]
    plot_success_rate(rolling, pt, pn,
                      os.path.join(fig_dir, "success_rate.png"))

    # Psi heatmap grid with routes
    plot_psi_heatmap_grid_with_routes(
        metrics.psi_snapshots,
        metrics.greedy_trajectory_snapshots,
        env_config,
        os.path.join(fig_dir, "psi_similarity_grid.png"),
        max_frames=16,
    )

    # Trajectory overlay (last 200)
    h, w = env.height, env.width
    start = np.unravel_index(env.start_state, (h, w))
    goal = np.unravel_index(env.goal_state, (h, w))

    # Get walls for current phase
    if hasattr(env, 'current_walls'):
        walls = env.current_walls
    elif hasattr(env, '_walls'):
        walls = env._walls
    else:
        walls = np.array(env_config.get("walls",
            env_config.get("phases", [{}])[0].get("walls",
                np.zeros((h, w)))))

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

    # Phase-specific trajectory overlays (for changing dynamics)
    if phase_names and phase_transitions:
        boundaries = [0] + phase_transitions + [len(metrics.episode_success)]
        for i, pname in enumerate(phase_names):
            start_ep = boundaries[i]
            end_ep = boundaries[i + 1] if i + 1 < len(boundaries) else len(metrics.episode_success)
            phase_trajs = metrics.episode_trajectories[start_ep:end_ep]
            phase_succ = metrics.episode_success[start_ep:end_ep]

            # Get walls for this phase
            if i < len(env_config.get("phases", [])):
                phase_walls = np.array(env_config["phases"][i]["walls"])
            else:
                phase_walls = walls

            p_coords = []
            p_flags = []
            # Sample last 200 from each phase
            for traj, succ in zip(phase_trajs[-200:], phase_succ[-200:]):
                coords = [np.unravel_index(s, (h, w)) for s in traj]
                p_coords.append(coords)
                p_flags.append(succ)
            if p_coords:
                plot_trajectory_overlay(
                    phase_walls, start, goal, p_coords, p_flags,
                    title=f"{condition_name}: {pname}",
                    max_trajs=200,
                    savepath=os.path.join(fig_dir, f"trajectories_phase{i}_{pname}.png"),
                )

    # Approach direction plots
    if approach_history and true_probs and hasattr(agent, 'get_approach_stats'):
        stats = agent.get_approach_stats()
        if stats:
            plot_approach_direction_stats(
                stats, true_probs,
                os.path.join(fig_dir, "approach_direction_stats.png"))
            plot_approach_over_time(
                approach_history,
                os.path.join(fig_dir, "approach_over_time.png"), window=200)

    # Video (cap frame count)
    print(f"  Generating psi video for {condition_name}...")
    snap_count = len(metrics.psi_snapshots)
    if snap_count > VIDEO_MAX_FRAMES:
        step = snap_count // VIDEO_MAX_FRAMES
        vid_psi = metrics.psi_snapshots[::step][:VIDEO_MAX_FRAMES]
        vid_traj = metrics.greedy_trajectory_snapshots[::step][:VIDEO_MAX_FRAMES]
    else:
        vid_psi = metrics.psi_snapshots
        vid_traj = metrics.greedy_trajectory_snapshots

    generate_psi_video_with_routes(
        vid_psi, vid_traj, env_config,
        os.path.join(cond_dir, "psi_evolution.mp4"),
        fps=6,
    )

    # Print summary
    sr = metrics.m1_success_rate_per_phase()
    for p, rate in sr.items():
        print(f"    Phase {p}: success={rate:.3f}")

    if hasattr(agent, 'get_approach_stats') and hasattr(agent, 'approach_counts') and agent.approach_counts:
        stats = agent.get_approach_stats()
        print(f"    Approach stats:")
        for d in sorted(stats):
            s = stats[d]
            tp = true_probs.get(d, 0.0) if true_probs else 0.0
            print(f"      {d}: attempts={s['attempts']} "
                  f"empirical={s['empirical_rate']:.3f} true={tp:.3f}")

    print(f"  Results saved to {cond_dir}")
    return metrics_dict, env_config


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--studies", nargs="+", default=["a", "b"],
                        help="Which studies to run: a, b, or both")
    args = parser.parse_args()

    base_dir = args.output_dir or str(SCRIPT_DIR / "results")

    full_cfg = {
        **AGENT_CFG,
        "psi_snapshot_freq": PSI_SNAPSHOT_FREQ,
        "policy_snapshot_freq": POLICY_SNAPSHOT_FREQ,
        "approach_probs": APPROACH_PROBS,
    }

    # ==================================================================
    # STUDY A
    # ==================================================================
    if "a" in args.studies:
        print("\n" + "=" * 70)
        print("STUDY A: STOCHASTIC SUCCESS ISOLATION")
        print(f"Episodes: {STUDY_A_EPISODES} | Seed: {args.seed}")
        print("=" * 70)

        out_a = os.path.join(base_dir, f"study_a_seed_{args.seed}")
        cfg_a = {**full_cfg, "total_episodes": STUDY_A_EPISODES}

        # --- A1: Baseline ---
        print("\n>>> Training A1-baseline (deterministic)")
        np.random.seed(args.seed)
        env_a1 = make_fourrooms_maze()
        agent_a1 = TabularSGCRLAgent(env_a1, **agent_kwargs())
        metrics_a1 = MetricsCollector(env_a1.num_states,
                                       (env_a1.height, env_a1.width))
        train_condition(agent_a1, env_a1, STUDY_A_EPISODES, metrics_a1,
                        "A1-baseline")

        # --- A2: AllReplay ---
        print("\n>>> Training A2-stochastic-allreplay")
        np.random.seed(args.seed)
        env_a2a = make_stochastic_fourrooms_maze(APPROACH_PROBS)
        agent_a2a = StochasticSGCRLAgent_AllReplay(env_a2a, **agent_kwargs())
        metrics_a2a = MetricsCollector(env_a2a.num_states,
                                        (env_a2a.height, env_a2a.width))
        ah_allreplay = []
        train_condition(agent_a2a, env_a2a, STUDY_A_EPISODES, metrics_a2a,
                        "A2-allreplay", is_stochastic=True,
                        approach_history=ah_allreplay)

        # --- A2: SuccessOnly ---
        print("\n>>> Training A2-stochastic-successonly")
        np.random.seed(args.seed)
        env_a2s = make_stochastic_fourrooms_maze(APPROACH_PROBS)
        agent_a2s = StochasticSGCRLAgent_SuccessOnly(env_a2s, **agent_kwargs())
        metrics_a2s = MetricsCollector(env_a2s.num_states,
                                        (env_a2s.height, env_a2s.width))
        ah_successonly = []
        train_condition(agent_a2s, env_a2s, STUDY_A_EPISODES, metrics_a2s,
                        "A2-successonly", is_stochastic=True,
                        approach_history=ah_successonly)

        # --- Generate all Study A visuals ---
        print("\n>>> Generating Study A visuals...")
        t_viz = time.time()

        md_a1, ec_a1 = save_results(
            cfg_a, env_a1, agent_a1, metrics_a1,
            os.path.join(out_a, "A1-baseline"), "A1-baseline")

        md_a2a, ec_a2a = save_results(
            cfg_a, env_a2a, agent_a2a, metrics_a2a,
            os.path.join(out_a, "A2-stochastic-allreplay"),
            "A2-stochastic-allreplay",
            approach_history=ah_allreplay, true_probs=APPROACH_PROBS)

        md_a2s, ec_a2s = save_results(
            cfg_a, env_a2s, agent_a2s, metrics_a2s,
            os.path.join(out_a, "A2-stochastic-successonly"),
            "A2-stochastic-successonly",
            approach_history=ah_successonly, true_probs=APPROACH_PROBS)

        print(f"  Study A visuals done in {time.time() - t_viz:.1f}s")

        # --- Comparison plots ---
        print("\n>>> Generating Study A comparison plots...")
        generate_study_a_comparisons(md_a1, md_a2a, md_a2s, out_a)

    # ==================================================================
    # STUDY B
    # ==================================================================
    if "b" in args.studies:
        total_b = STUDY_B_EPISODES_PER_PHASE * 3
        print("\n" + "=" * 70)
        print("STUDY B: CHANGING DYNAMICS ISOLATION")
        print(f"Episodes: {total_b} ({STUDY_B_EPISODES_PER_PHASE}/phase) | Seed: {args.seed}")
        print("=" * 70)

        out_b = os.path.join(base_dir, f"study_b_seed_{args.seed}")
        cfg_b = {**full_cfg,
                 "total_episodes": total_b,
                 "episodes_per_phase": STUDY_B_EPISODES_PER_PHASE}

        # --- B1: Static baseline ---
        print("\n>>> Training B1-baseline (static fourrooms)")
        np.random.seed(args.seed)
        env_b1 = make_static_fourrooms()
        agent_b1 = TabularSGCRLAgent(env_b1, **agent_kwargs())
        metrics_b1 = MetricsCollector(env_b1.num_states,
                                       (env_b1.height, env_b1.width))
        train_condition_b1(agent_b1, env_b1, total_b, metrics_b1, "B1-static")

        # --- B2: Changing dynamics ---
        print("\n>>> Training B2-changing (fourrooms -> corridor_shift -> l_wall)")
        np.random.seed(args.seed)
        env_b2 = make_changing_dynamics_maze(STUDY_B_EPISODES_PER_PHASE)
        agent_b2 = TabularSGCRLAgent(env_b2, **agent_kwargs())
        metrics_b2 = MetricsCollector(env_b2.num_states,
                                       (env_b2.height, env_b2.width))
        train_condition_b2(agent_b2, env_b2, metrics_b2, "B2-changing",
                           STUDY_B_EPISODES_PER_PHASE)

        # --- Generate all Study B visuals ---
        print("\n>>> Generating Study B visuals...")
        t_viz = time.time()

        phase_transitions_b = [STUDY_B_EPISODES_PER_PHASE,
                                STUDY_B_EPISODES_PER_PHASE * 2]
        phase_names_b = ["fourrooms", "corridor_shift", "l_wall"]

        md_b1, ec_b1 = save_results(
            cfg_b, env_b1, agent_b1, metrics_b1,
            os.path.join(out_b, "B1-baseline"), "B1-static",
            phase_names=["fourrooms"])

        md_b2, ec_b2 = save_results(
            cfg_b, env_b2, agent_b2, metrics_b2,
            os.path.join(out_b, "B2-changing"), "B2-changing",
            phase_transitions=phase_transitions_b,
            phase_names=phase_names_b)

        print(f"  Study B visuals done in {time.time() - t_viz:.1f}s")

        # --- Comparison plots ---
        print("\n>>> Generating Study B comparison plots...")
        generate_study_b_comparisons(
            md_b1, md_b2, out_b,
            phase_transitions_b, phase_names_b)

    print("\n" + "=" * 70)
    print("ALL DONE")
    print("=" * 70)


# -----------------------------------------------------------------------
# Comparison plot generators
# -----------------------------------------------------------------------

def generate_study_a_comparisons(md_a1, md_a2a, md_a2s, out_dir):
    """Generate all Study A comparison plots."""
    from visualization import (plot_metric_comparison,
                               plot_adaptation_comparison,
                               plot_m8_alignment_comparison)

    comp_dir = os.path.join(out_dir, "comparisons")
    os.makedirs(comp_dir, exist_ok=True)

    r1 = np.array(md_a1["m1_rolling_success_rate"])
    r2a = np.array(md_a2a["m1_rolling_success_rate"])
    r2s = np.array(md_a2s["m1_rolling_success_rate"])

    # M1: Success rate
    plot_metric_comparison(r1, r2a, "Episode", "Success Rate (rolling 50)",
        "M1: Success Rate — Baseline vs Allreplay",
        os.path.join(comp_dir, "m1_success_rate_vs_allreplay.png"),
        "A1-baseline", "A2-allreplay")

    plot_metric_comparison(r1, r2s, "Episode", "Success Rate (rolling 50)",
        "M1: Success Rate — Baseline vs Successonly",
        os.path.join(comp_dir, "m1_success_rate_vs_successonly.png"),
        "A1-baseline", "A2-successonly")

    plot_metric_comparison(r2a, r2s, "Episode", "Success Rate (rolling 50)",
        "M1: Success Rate — Allreplay vs Successonly",
        os.path.join(comp_dir, "m1_allreplay_vs_successonly.png"),
        "A2-allreplay", "A2-successonly")

    # M3: Exploitation ratio
    for tag, md in [("allreplay", md_a2a), ("successonly", md_a2s)]:
        if "m3_exploitation_ratio" in md:
            er_base = np.array(md_a1.get("m3_exploitation_ratio", []))
            er_treat = np.array(md.get("m3_exploitation_ratio", []))
            if len(er_base) > 0 and len(er_treat) > 0:
                plot_metric_comparison(
                    er_base, er_treat, "Episode", "Exploitation Ratio",
                    f"M3: Exploitation Ratio — Baseline vs {tag.title()}",
                    os.path.join(comp_dir, f"m3_exploitation_ratio_vs_{tag}.png"),
                    "A1-baseline", f"A2-{tag}")

    # M4/M5: Adaptation
    for tag, md in [("allreplay", md_a2a), ("successonly", md_a2s)]:
        plot_adaptation_comparison(
            md_a1.get("m4_adaptation_index", []),
            md.get("m4_adaptation_index", []),
            md_a1.get("m5_representation_rate", []),
            md.get("m5_representation_rate", []),
            os.path.join(comp_dir, f"m4_m5_adaptation_vs_{tag}.png"),
            "A1-baseline", f"A2-{tag}")

    # M6: Suboptimality gap
    for tag, md in [("allreplay", md_a2a), ("successonly", md_a2s)]:
        if "m6_suboptimality_gap" in md:
            sg_base = np.array(md_a1.get("m6_suboptimality_gap", []))
            sg_treat = np.array(md.get("m6_suboptimality_gap", []))
            if len(sg_base) > 0 and len(sg_treat) > 0:
                plot_metric_comparison(
                    sg_base, sg_treat, "Episode", "Step Overhead",
                    f"M6: Suboptimality Gap — Baseline vs {tag.title()}",
                    os.path.join(comp_dir, f"m6_suboptimality_gap_{tag}.png"),
                    "A1-baseline", f"A2-{tag}")

    # M7: Approach entropy
    for tag, md in [("allreplay", md_a2a), ("successonly", md_a2s)]:
        if "m7_approach_entropy" in md:
            ae = np.array(md.get("m7_approach_entropy", []))
            if len(ae) > 0:
                from visualization import plot_metric_comparison as pmc
                fig_path = os.path.join(comp_dir, f"m7_approach_entropy_{tag}.png")
                # Plot just the treatment entropy
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(14, 5))
                ax.plot(ae, linewidth=1.5, color="#E64A19")
                ax.set_xlabel("Episode")
                ax.set_ylabel("Entropy")
                ax.set_title(f"M7: Approach Direction Entropy — {tag.title()}")
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(fig_path, dpi=150, bbox_inches="tight")
                plt.close(fig)

    # M8: Alignment
    for tag, md in [("allreplay", md_a2a), ("successonly", md_a2s)]:
        plot_m8_alignment_comparison(
            md_a1.get("m8_alignment", []),
            md.get("m8_alignment", []),
            os.path.join(comp_dir, f"m8_alignment_vs_{tag}.png"),
            "A1-baseline", f"A2-{tag}")

    print(f"  Study A comparisons saved to {comp_dir}")


def generate_study_b_comparisons(md_b1, md_b2, out_dir,
                                  phase_transitions, phase_names):
    """Generate all Study B comparison plots."""
    from visualization import (plot_metric_comparison,
                               plot_adaptation_comparison,
                               plot_m8_alignment_comparison)

    comp_dir = os.path.join(out_dir, "comparisons")
    os.makedirs(comp_dir, exist_ok=True)

    r1 = np.array(md_b1["m1_rolling_success_rate"])
    r2 = np.array(md_b2["m1_rolling_success_rate"])

    # M1: Success rate
    plot_metric_comparison(r1, r2, "Episode", "Success Rate (rolling 50)",
        "M1: Success Rate — Static vs Changing Dynamics",
        os.path.join(comp_dir, "m1_success_rate.png"),
        "B1-static", "B2-changing",
        phase_transitions=phase_transitions, phase_names=phase_names)

    # M2: Path diversity
    for key, ylabel, title in [
        ("m2_path_diversity", "Unique States per Episode", "M2: Path Diversity"),
    ]:
        d1 = np.array(md_b1.get(key, []))
        d2 = np.array(md_b2.get(key, []))
        if len(d1) > 0 and len(d2) > 0:
            plot_metric_comparison(d1, d2, "Episode", ylabel, title,
                os.path.join(comp_dir, f"{key}.png"),
                "B1-static", "B2-changing",
                phase_transitions=phase_transitions,
                phase_names=phase_names)

    # M3: Exploitation ratio
    d1 = np.array(md_b1.get("m3_exploitation_ratio", []))
    d2 = np.array(md_b2.get("m3_exploitation_ratio", []))
    if len(d1) > 0 and len(d2) > 0:
        plot_metric_comparison(d1, d2, "Episode", "Exploitation Ratio",
            "M3: Exploitation Ratio",
            os.path.join(comp_dir, "m3_exploitation_ratio.png"),
            "B1-static", "B2-changing",
            phase_transitions=phase_transitions,
            phase_names=phase_names)

    # M4/M5: Adaptation
    plot_adaptation_comparison(
        md_b1.get("m4_adaptation_index", []),
        md_b2.get("m4_adaptation_index", []),
        md_b1.get("m5_representation_rate", []),
        md_b2.get("m5_representation_rate", []),
        os.path.join(comp_dir, "m4_m5_adaptation.png"),
        "B1-static", "B2-changing",
        phase_transitions=phase_transitions)

    # M8: Alignment
    plot_m8_alignment_comparison(
        md_b1.get("m8_alignment", []),
        md_b2.get("m8_alignment", []),
        os.path.join(comp_dir, "m8_alignment.png"),
        "B1-static", "B2-changing",
        phase_transitions=phase_transitions)

    # M9: Recovery time
    if "m9_recovery_time" in md_b2:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        recov = md_b2["m9_recovery_time"]
        if recov:
            fig, ax = plt.subplots(figsize=(10, 5))
            labels = [f"Ep {r['transition_episode']}" for r in recov]
            vals = [r.get("recovery_episodes", 0) for r in recov]
            x = range(len(labels))
            ax.bar(x, vals, color="#E64A19")
            ax.set_xticks(list(x))
            ax.set_xticklabels(labels)
            ax.set_ylabel("Episodes to Recover")
            ax.set_title("M9: Phase-Transition Recovery Time (B2-changing)")
            fig.tight_layout()
            os.makedirs(comp_dir, exist_ok=True)
            fig.savefig(os.path.join(comp_dir, "m9_recovery.png"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

    print(f"  Study B comparisons saved to {comp_dir}")


if __name__ == "__main__":
    main()
