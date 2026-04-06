#!/usr/bin/env python3
"""
Run a single condition, save metrics as pickle + JSON. No visualization.
Viz is done separately with plot_results.py.

Usage:
  python run_single.py --condition a1 --seed 42
  python run_single.py --condition a2ar --seed 42
  python run_single.py --condition a2so --seed 42
  python run_single.py --condition b1 --seed 42
  python run_single.py --condition b2 --seed 42
"""

import argparse, json, os, sys, time, pickle
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from envs import (make_fourrooms_maze, make_stochastic_fourrooms_maze,
                   make_static_fourrooms, make_changing_dynamics_maze)
from agent import (TabularSGCRLAgent, StochasticSGCRLAgent_AllReplay,
                   StochasticSGCRLAgent_SuccessOnly)
from metrics import MetricsCollector

STUDY_A_EPS = 10000
STUDY_B_EPS_PER_PHASE = 5000
SNAP_FREQ = 200

AGENT_CFG = dict(rep_dim=16, lr_psi=0.01, batch_size=128, replay_capacity=2000,
                 max_steps=120, gamma=0.99, entropy_coeff=0.1,
                 episodes_per_update=1, normalize=True)
APPROACH_PROBS = dict(from_above=0.10, from_left=0.25, from_right=0.50,
                      from_below=0.80, stay=0.00)


def run_a1(seed, out):
    np.random.seed(seed)
    env = make_fourrooms_maze()
    agent = TabularSGCRLAgent(env, **AGENT_CFG)
    m = MetricsCollector(env.num_states, (env.height, env.width))
    t0 = time.time()
    for ep in range(STUDY_A_EPS):
        traj, success = agent.collect_episode()
        m.record_episode(ep, 0, traj, success)
        if ep % agent.episodes_per_update == 0: agent.update_representations()
        if ep % SNAP_FREQ == 0:
            m.record_psi_snapshot(ep, 0, agent.get_similarity_map())
            m.record_psi_full_snapshot(ep, 0, agent.get_psi_snapshot())
            m.record_greedy_trajectory(ep, 0, agent.policy_rollout(n_rollouts=5))
            m.record_policy_snapshot(ep, 0, agent.get_policy_distribution())
        if (ep+1)%2000==0:
            r = m.episode_success[max(0,ep-99):ep+1]
            print(f"  {ep+1}/{STUDY_A_EPS} sr={np.mean(r):.2f} t={time.time()-t0:.0f}s")
    print(f"  Done {time.time()-t0:.0f}s")
    _save(out, env, agent, m, {"condition":"A1-baseline","total_episodes":STUDY_A_EPS})

def run_a2ar(seed, out):
    np.random.seed(seed)
    env = make_stochastic_fourrooms_maze(APPROACH_PROBS)
    agent = StochasticSGCRLAgent_AllReplay(env, **AGENT_CFG)
    m = MetricsCollector(env.num_states, (env.height, env.width))
    ah = []
    t0 = time.time()
    for ep in range(STUDY_A_EPS):
        traj, success, adir = agent.collect_episode()
        m.record_episode(ep, 0, traj, success)
        m.record_approach(ep, adir, success)
        ah.append((ep, adir, success))
        if ep % agent.episodes_per_update == 0: agent.update_representations()
        if ep % SNAP_FREQ == 0:
            m.record_psi_snapshot(ep, 0, agent.get_similarity_map())
            m.record_psi_full_snapshot(ep, 0, agent.get_psi_snapshot())
            m.record_greedy_trajectory(ep, 0, agent.policy_rollout(n_rollouts=5))
            m.record_policy_snapshot(ep, 0, agent.get_policy_distribution())
        if (ep+1)%2000==0:
            r = m.episode_success[max(0,ep-99):ep+1]
            print(f"  {ep+1}/{STUDY_A_EPS} sr={np.mean(r):.2f} t={time.time()-t0:.0f}s")
    print(f"  Done {time.time()-t0:.0f}s")
    _save(out, env, agent, m, {"condition":"A2-allreplay","total_episodes":STUDY_A_EPS},
          ah=ah, true_probs=APPROACH_PROBS)

def run_a2so(seed, out):
    np.random.seed(seed)
    env = make_stochastic_fourrooms_maze(APPROACH_PROBS)
    agent = StochasticSGCRLAgent_SuccessOnly(env, **AGENT_CFG)
    m = MetricsCollector(env.num_states, (env.height, env.width))
    ah = []
    t0 = time.time()
    for ep in range(STUDY_A_EPS):
        traj, success, adir = agent.collect_episode()
        m.record_episode(ep, 0, traj, success)
        m.record_approach(ep, adir, success)
        ah.append((ep, adir, success))
        if ep % agent.episodes_per_update == 0: agent.update_representations()
        if ep % SNAP_FREQ == 0:
            m.record_psi_snapshot(ep, 0, agent.get_similarity_map())
            m.record_psi_full_snapshot(ep, 0, agent.get_psi_snapshot())
            m.record_greedy_trajectory(ep, 0, agent.policy_rollout(n_rollouts=5))
            m.record_policy_snapshot(ep, 0, agent.get_policy_distribution())
        if (ep+1)%2000==0:
            r = m.episode_success[max(0,ep-99):ep+1]
            print(f"  {ep+1}/{STUDY_A_EPS} sr={np.mean(r):.2f} t={time.time()-t0:.0f}s")
    print(f"  Done {time.time()-t0:.0f}s")
    _save(out, env, agent, m, {"condition":"A2-successonly","total_episodes":STUDY_A_EPS},
          ah=ah, true_probs=APPROACH_PROBS)

def run_b1(seed, out):
    total = STUDY_B_EPS_PER_PHASE * 3
    np.random.seed(seed)
    env = make_static_fourrooms()
    agent = TabularSGCRLAgent(env, **AGENT_CFG)
    m = MetricsCollector(env.num_states, (env.height, env.width))
    t0 = time.time()
    for ep in range(total):
        traj, success = agent.collect_episode()
        m.record_episode(ep, 0, traj, success)
        if ep % agent.episodes_per_update == 0: agent.update_representations()
        if ep % SNAP_FREQ == 0:
            m.record_psi_snapshot(ep, 0, agent.get_similarity_map())
            m.record_psi_full_snapshot(ep, 0, agent.get_psi_snapshot())
            m.record_greedy_trajectory(ep, 0, agent.policy_rollout(n_rollouts=5))
            m.record_policy_snapshot(ep, 0, agent.get_policy_distribution())
        if (ep+1)%2000==0:
            r = m.episode_success[max(0,ep-99):ep+1]
            print(f"  {ep+1}/{total} sr={np.mean(r):.2f} t={time.time()-t0:.0f}s")
    print(f"  Done {time.time()-t0:.0f}s")
    _save(out, env, agent, m, {"condition":"B1-static","total_episodes":total})

def run_b2(seed, out):
    epp = STUDY_B_EPS_PER_PHASE
    np.random.seed(seed)
    env = make_changing_dynamics_maze(epp)
    agent = TabularSGCRLAgent(env, **AGENT_CFG)
    m = MetricsCollector(env.num_states, (env.height, env.width))
    t0 = time.time()
    total = sum(p.num_episodes for p in env.phases)
    # Compute phase boundaries
    phase_boundaries = []
    cumulative = 0
    for p in env.phases:
        phase_boundaries.append(cumulative)
        cumulative += p.num_episodes
    cur_phase = 0
    env.set_phase(0)
    for ep in range(total):
        # Phase transition check
        next_phase = cur_phase
        for i, boundary in enumerate(phase_boundaries):
            if ep >= boundary:
                next_phase = i
        if next_phase != cur_phase:
            env.set_phase(next_phase)
            cur_phase = next_phase
            print(f"  Phase -> {env.current_phase.name} at ep {ep}")
        traj, success = agent.collect_episode()
        m.record_episode(ep, cur_phase, traj, success)
        if ep % agent.episodes_per_update == 0: agent.update_representations()
        if ep % SNAP_FREQ == 0:
            m.record_psi_snapshot(ep, cur_phase, agent.get_similarity_map())
            m.record_psi_full_snapshot(ep, cur_phase, agent.get_psi_snapshot())
            m.record_greedy_trajectory(ep, cur_phase, agent.policy_rollout(n_rollouts=5))
            m.record_policy_snapshot(ep, cur_phase, agent.get_policy_distribution())
        if (ep+1)%2000==0:
            r = m.episode_success[max(0,ep-99):ep+1]
            print(f"  {ep+1}/{total} sr={np.mean(r):.2f} t={time.time()-t0:.0f}s")
    print(f"  Done {time.time()-t0:.0f}s")
    _save(out, env, agent, m,
          {"condition":"B2-changing","total_episodes":total,"episodes_per_phase":epp},
          phase_trans=[epp, epp*2], phase_names=["fourrooms","corridor_shift","l_wall"])


def _save(out_dir, env, agent, metrics, extra_cfg, ah=None, true_probs=None,
          phase_trans=None, phase_names=None):
    os.makedirs(out_dir, exist_ok=True)
    cfg = {**AGENT_CFG, "approach_probs": APPROACH_PROBS,
           "snap_freq": SNAP_FREQ, **extra_cfg}
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    md = metrics.to_dict(true_probs=true_probs)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(md, f, indent=2)
    ec = env.get_config()
    with open(os.path.join(out_dir, "env_config.json"), "w") as f:
        json.dump(ec, f, indent=2)
    np.save(os.path.join(out_dir, "psi_final.npy"), agent.psi)
    # Save snapshots for viz
    with open(os.path.join(out_dir, "snapshots.pkl"), "wb") as f:
        pickle.dump({
            "psi_snapshots": metrics.psi_snapshots,
            "greedy_trajectory_snapshots": metrics.greedy_trajectory_snapshots,
            "episode_trajectories": metrics.episode_trajectories[-200:],
            "episode_success": metrics.episode_success[-200:],
        }, f)
    if ah:
        with open(os.path.join(out_dir, "approach_history.json"), "w") as f:
            json.dump([{"episode":e,"direction":d,"success":bool(s)} for e,d,s in ah], f)
    if phase_trans:
        with open(os.path.join(out_dir, "phase_info.json"), "w") as f:
            json.dump({"transitions":phase_trans,"names":phase_names}, f)
    # Save approach stats for stochastic agents
    if hasattr(agent, 'get_approach_stats') and hasattr(agent, 'approach_counts') and agent.approach_counts:
        with open(os.path.join(out_dir, "approach_stats.json"), "w") as f:
            json.dump(agent.get_approach_stats(), f, indent=2)
    print(f"  Saved to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True, choices=["a1","a2ar","a2so","b1","b2"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default=None)
    args = p.parse_args()

    defaults = {
        "a1": f"study_a_seed_{args.seed}/A1-baseline",
        "a2ar": f"study_a_seed_{args.seed}/A2-stochastic-allreplay",
        "a2so": f"study_a_seed_{args.seed}/A2-stochastic-successonly",
        "b1": f"study_b_seed_{args.seed}/B1-baseline",
        "b2": f"study_b_seed_{args.seed}/B2-changing",
    }
    out = args.output_dir or os.path.join(str(SCRIPT_DIR/"results"), defaults[args.condition])

    {"a1":run_a1, "a2ar":run_a2ar, "a2so":run_a2so, "b1":run_b1, "b2":run_b2}[args.condition](args.seed, out)
