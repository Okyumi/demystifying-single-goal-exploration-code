#!/usr/bin/env python3
"""
Lean runner: train everything first, generate only essential plots.
No videos. Reduced snapshot freq. Fast.

Usage:
  python experiments/isolation_study/run_lean.py --seed 42
"""

import argparse, json, os, sys, time
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from envs import (
    make_fourrooms_maze, make_stochastic_fourrooms_maze,
    make_static_fourrooms, make_changing_dynamics_maze,
)
from agent import (
    TabularSGCRLAgent,
    StochasticSGCRLAgent_AllReplay,
    StochasticSGCRLAgent_SuccessOnly,
)
from metrics import MetricsCollector

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------

STUDY_A_EPS = 10000
STUDY_B_EPS_PER_PHASE = 5000
SNAP_FREQ = 200  # minimal snapshots

AGENT_CFG = dict(
    rep_dim=16, lr_psi=0.01, batch_size=128, replay_capacity=2000,
    max_steps=120, gamma=0.99, entropy_coeff=0.1,
    episodes_per_update=1, normalize=True,
)
APPROACH_PROBS = dict(
    from_above=0.10, from_left=0.25, from_right=0.50,
    from_below=0.80, stay=0.00,
)


# -----------------------------------------------------------------------
# Training loops (no viz)
# -----------------------------------------------------------------------

def train_standard(agent, env, total, metrics, label, is_stochastic=False):
    """Train a single-phase condition."""
    ah = [] if is_stochastic else None
    t0 = time.time()
    for ep in range(total):
        if is_stochastic:
            traj, success, adir = agent.collect_episode()
            metrics.record_approach(ep, adir, success)
            ah.append((ep, adir, success))
        else:
            traj, success = agent.collect_episode()

        metrics.record_episode(ep, 0, traj, success)
        if ep % agent.episodes_per_update == 0:
            agent.update_representations()

        if ep % SNAP_FREQ == 0:
            sm = agent.get_similarity_map()
            metrics.record_psi_snapshot(ep, 0, sm)
            metrics.record_psi_full_snapshot(ep, 0, agent.get_psi_snapshot())
            pt = agent.policy_rollout(n_rollouts=5)
            metrics.record_greedy_trajectory(ep, 0, pt)

        if ep % SNAP_FREQ == 0:
            metrics.record_policy_snapshot(ep, 0, agent.get_policy_distribution())

        if (ep+1) % 2000 == 0:
            r = metrics.episode_success[max(0,ep-99):ep+1]
            print(f"  [{label}] {ep+1}/{total} sr={np.mean(r):.2f} t={time.time()-t0:.0f}s")

    print(f"  [{label}] done {time.time()-t0:.0f}s")
    return ah


def train_b2(agent, env, metrics, label):
    """Train B2-changing with phase transitions."""
    t0 = time.time()
    total = env.total_episodes
    cur_phase = 0
    for ep in range(total):
        new_p = env.get_phase_for_episode(ep)
        if new_p != cur_phase:
            cur_phase = new_p
            agent.set_walls(env.current_walls)
            print(f"  [{label}] Phase -> {env.phases[cur_phase].name} at ep {ep}")

        env.set_episode(ep)
        traj, success = agent.collect_episode()
        metrics.record_episode(ep, cur_phase, traj, success)
        if ep % agent.episodes_per_update == 0:
            agent.update_representations()

        if ep % SNAP_FREQ == 0:
            sm = agent.get_similarity_map()
            metrics.record_psi_snapshot(ep, cur_phase, sm)
            metrics.record_psi_full_snapshot(ep, cur_phase, agent.get_psi_snapshot())
            pt = agent.policy_rollout(n_rollouts=5)
            metrics.record_greedy_trajectory(ep, cur_phase, pt)
            metrics.record_policy_snapshot(ep, cur_phase, agent.get_policy_distribution())

        if (ep+1) % 2000 == 0:
            r = metrics.episode_success[max(0,ep-99):ep+1]
            print(f"  [{label}] {ep+1}/{total} sr={np.mean(r):.2f} t={time.time()-t0:.0f}s")

    print(f"  [{label}] done {time.time()-t0:.0f}s")


# -----------------------------------------------------------------------
# Plotting (after training)
# -----------------------------------------------------------------------

def save_condition(cfg, env, agent, metrics, cond_dir, name,
                   ah=None, true_probs=None, phase_trans=None, phase_names=None):
    """Save metrics + essential plots (no video)."""
    os.makedirs(cond_dir, exist_ok=True)

    with open(os.path.join(cond_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    md = metrics.to_dict(true_probs=true_probs)
    with open(os.path.join(cond_dir, "metrics.json"), "w") as f:
        json.dump(md, f, indent=2)

    ec = env.get_config()
    with open(os.path.join(cond_dir, "env_config.json"), "w") as f:
        json.dump(ec, f, indent=2)

    np.save(os.path.join(cond_dir, "psi_final.npy"), agent.psi)

    if ah:
        with open(os.path.join(cond_dir, "approach_history.json"), "w") as f:
            json.dump([{"episode":e,"direction":d,"success":bool(s)} for e,d,s in ah], f)

    fig_dir = os.path.join(cond_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 1. Success rate
    rolling = np.array(md["m1_rolling_success_rate"])
    fig, ax = plt.subplots(figsize=(14,5))
    ax.plot(rolling, lw=1.5, color="steelblue")
    if phase_trans:
        for t in phase_trans:
            ax.axvline(t, color="red", ls="--", alpha=0.6)
        bounds = [0] + phase_trans + [len(rolling)]
        for i, pn in enumerate(phase_names or []):
            mid = (bounds[i] + bounds[i+1]) / 2
            ax.text(mid, 0.95, pn, ha="center", va="top", fontsize=9,
                    fontweight="bold", color="gray",
                    transform=ax.get_xaxis_transform())
    ax.set_xlabel("Episode"); ax.set_ylabel("Success Rate (rolling 50)")
    ax.set_ylim(-0.05, 1.05); ax.set_title(f"Success Rate — {name}")
    ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "success_rate.png"), dpi=150); plt.close(fig)

    # 2. Psi similarity grid (4x4 max)
    h, w = ec["height"], ec["width"]
    phases = ec.get("phases", [])
    snaps = metrics.psi_snapshots
    trajs = metrics.greedy_trajectory_snapshots
    n = len(snaps)
    step = max(1, n // 16)
    sel = list(range(0, n, step))[:16]

    ncols = min(4, len(sel))
    nrows = max(1, (len(sel) + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    if nrows == 1 and ncols == 1: axes = np.array([[axes]])
    elif nrows == 1: axes = axes[np.newaxis, :]
    elif ncols == 1: axes = axes[:, np.newaxis]

    all_sims = np.concatenate([snaps[i][2].ravel() for i in sel])
    vmin, vmax = np.percentile(all_sims, 2), np.percentile(all_sims, 98)

    for idx_pos, idx in enumerate(sel):
        ep, phase, sm = snaps[idx]
        r_idx, c_idx = divmod(idx_pos, ncols)
        ax = axes[r_idx, c_idx]

        walls = np.array(phases[phase]["walls"]) if phase < len(phases) else np.zeros((h,w))
        pname = phases[phase]["name"] if phase < len(phases) else f"phase_{phase}"
        masked = sm.copy(); masked[walls == 1] = np.nan
        cmap = plt.cm.RdYlGn.copy(); cmap.set_bad("#333333")
        ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(f"Ep {ep} ({pname})", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

        # Route overlay
        if idx < len(trajs):
            _, _, traj = trajs[idx]
            if traj and len(traj) > 1:
                rc = [np.unravel_index(s, (h,w)) for s in traj]
                rows = [c[0] for c in rc]; cols = [c[1] for c in rc]
                ax.plot(cols, rows, "-", color="#1565C0", lw=2.0, alpha=0.9, zorder=6,
                        solid_capstyle="round")
                # Arrows
                arrow_step = max(1, len(rc)//8)
                for ai in range(0, len(rc)-1, arrow_step):
                    dr = rows[ai+1]-rows[ai]; dc = cols[ai+1]-cols[ai]
                    if dr == 0 and dc == 0: continue
                    ax.annotate("", xy=(cols[ai+1],rows[ai+1]),
                                xytext=(cols[ai],rows[ai]),
                                arrowprops=dict(arrowstyle="->", color="#1565C0",
                                                lw=1.4, alpha=0.9), zorder=7)

    for idx_pos in range(len(sel), nrows*ncols):
        r_idx, c_idx = divmod(idx_pos, ncols)
        axes[r_idx, c_idx].axis("off")

    fig.suptitle(f"ψ(s)·ψ(g) Similarity + Policy Route — {name}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(os.path.join(fig_dir, "psi_similarity_grid.png"), dpi=150); plt.close(fig)

    # 3. Trajectory overlay (last 200)
    last_trajs = metrics.episode_trajectories[-200:]
    last_succ = metrics.episode_success[-200:]
    walls_arr = np.array(phases[0]["walls"]) if phases else np.zeros((h,w))
    if phase_names and len(phases) > 1:
        # Use last phase walls
        walls_arr = np.array(phases[-1]["walls"])

    start = np.unravel_index(ec["start_state"], (h,w))
    goal = np.unravel_index(ec["goal_state"], (h,w))

    fig, ax = plt.subplots(figsize=(7,7))
    ax.set_xlim(-0.5, w-0.5); ax.set_ylim(h-0.5, -0.5); ax.set_aspect("equal")
    for r in range(h):
        for c in range(w):
            if walls_arr[r,c] == 1:
                ax.add_patch(plt.Rectangle((c-0.5,r-0.5),1,1,fc="#333",ec="#444",lw=0.5))
    ax.plot(start[1], start[0], "o", color="#2ecc71", ms=10, zorder=5)
    ax.plot(goal[1], goal[0], "*", color="#e74c3c", ms=14, zorder=5)

    for traj, succ in zip(last_trajs[-100:], last_succ[-100:]):
        coords = [np.unravel_index(s,(h,w)) for s in traj]
        rs = [c[0] for c in coords]; cs = [c[1] for c in coords]
        color = "#27ae60" if succ else "#c0392b"
        alpha = 0.5 if succ else 0.15
        ax.plot(cs, rs, "-", color=color, alpha=alpha, lw=1)

    ax.set_title(f"{name}: Last 100 Episodes", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "trajectories_last.png"), dpi=150); plt.close(fig)

    # 4. Approach direction (stochastic only)
    if ah and true_probs and hasattr(agent, 'get_approach_stats'):
        stats = agent.get_approach_stats()
        if stats:
            dirs = sorted(stats.keys())
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            attempts = [stats[d]["attempts"] for d in dirs]
            emp = [stats[d]["empirical_rate"] for d in dirs]
            true = [true_probs.get(d, 0) for d in dirs]
            x = np.arange(len(dirs))
            ax1.bar(x, attempts, color="#42A5F5")
            ax1.set_xticks(x); ax1.set_xticklabels(dirs, rotation=30, ha="right")
            ax1.set_ylabel("Attempts"); ax1.set_title("Approach Direction Attempts")
            wid = 0.35
            ax2.bar(x-wid/2, true, wid, label="True prob", color="#66BB6A")
            ax2.bar(x+wid/2, emp, wid, label="Empirical rate", color="#EF5350")
            ax2.set_xticks(x); ax2.set_xticklabels(dirs, rotation=30, ha="right")
            ax2.set_ylabel("Success Probability"); ax2.set_title("True vs Empirical")
            ax2.legend()
            fig.suptitle(f"Approach Direction Analysis — {name}", fontsize=13, fontweight="bold")
            fig.tight_layout(rect=[0,0,1,0.92])
            fig.savefig(os.path.join(fig_dir, "approach_direction_stats.png"), dpi=150)
            plt.close(fig)

            # Approach over time
            dirs_all = sorted(set(d for _,d,_ in ah if d))
            fig, ax = plt.subplots(figsize=(12,5))
            colors = ["#2196F3","#4CAF50","#FF9800","#E91E63","#9C27B0"]
            for i, d in enumerate(dirs_all):
                binary = np.array([1 if dd==d else 0 for _,dd,_ in ah], dtype=float)
                win = 200
                if len(binary) >= win:
                    k = np.ones(win)/win
                    smoothed = np.convolve(binary, k, mode="same")
                else:
                    smoothed = np.cumsum(binary)/(np.arange(len(binary))+1)
                ax.plot([e for e,_,_ in ah], smoothed, label=d, color=colors[i%5], lw=1.5)
            ax.set_xlabel("Episode"); ax.set_ylabel(f"Fraction (rolling 200)")
            ax.set_title("Approach Direction Over Time"); ax.legend(); ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, "approach_over_time.png"), dpi=150)
            plt.close(fig)

    print(f"  {name} saved to {cond_dir}")
    return md, ec


# -----------------------------------------------------------------------
# Comparison plots
# -----------------------------------------------------------------------

def comparison_line(d1, d2, ylabel, title, path, l1, l2, pt=None, pn=None):
    fig, ax = plt.subplots(figsize=(14,5))
    ax.plot(d1, lw=1.5, color="#1976D2", alpha=0.8, label=l1)
    ax.plot(d2, lw=1.5, color="#E64A19", alpha=0.8, label=l2)
    if pt:
        for t in pt:
            ax.axvline(t, color="gray", ls="--", alpha=0.5)
        if pn:
            bounds = [0] + pt + [max(len(d1),len(d2))]
            for i,n in enumerate(pn):
                if i < len(bounds)-1:
                    mid = (bounds[i]+bounds[i+1])/2
                    ax.text(mid, 0.97, n, ha="center", va="top", fontsize=8,
                            fontweight="bold", color="gray",
                            transform=ax.get_xaxis_transform())
    ax.set_xlabel("Episode"); ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150); plt.close(fig)


def comparison_adapt(md1, md2, path, l1, l2, pt=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for md, lbl, c in [(md1, l1, "#1976D2"), (md2, l2, "#E64A19")]:
        ai = md.get("m4_adaptation_index", [])
        if ai:
            eps = [(d["episode_from"]+d["episode_to"])/2 for d in ai]
            kls = [d["kl_divergence"] for d in ai]
            ax1.plot(eps, kls, lw=1.5, color=c, alpha=0.8, label=lbl)
        rr = md.get("m5_representation_rate", [])
        if rr:
            eps = [(d["episode_from"]+d["episode_to"])/2 for d in rr]
            l2s = [d["mean_l2_rate"] for d in rr]
            ax2.plot(eps, l2s, lw=1.5, color=c, alpha=0.8, label=lbl)
    ax1.set_ylabel("KL Divergence"); ax1.set_title("M4: Policy Adaptation Index")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.set_ylabel("Mean L2 Change"); ax2.set_title("M5: Representation Adaptation Rate")
    ax2.set_xlabel("Episode"); ax2.legend(); ax2.grid(True, alpha=0.3)
    if pt:
        for ax in (ax1,ax2):
            for t in pt: ax.axvline(t, color="gray", ls="--", alpha=0.5)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150); plt.close(fig)


def comparison_align(md1, md2, path, l1, l2, pt=None):
    fig, ax = plt.subplots(figsize=(14,5))
    for md, lbl, c in [(md1, l1, "#1976D2"), (md2, l2, "#E64A19")]:
        al = md.get("m8_alignment", [])
        if al:
            eps = [d["episode"] for d in al]
            gaps = [d["alignment_gap"] for d in al]
            ax.plot(eps, gaps, lw=1.5, color=c, alpha=0.8, label=lbl)
    if pt:
        for t in pt: ax.axvline(t, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Episode"); ax.set_ylabel("Alignment Gap")
    ax.set_title("M8: Representation-Route Alignment", fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150); plt.close(fig)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--studies", nargs="+", default=["a","b"])
    args = parser.parse_args()

    base = str(SCRIPT_DIR / "results")

    full_cfg = {**AGENT_CFG, "approach_probs": APPROACH_PROBS,
                "psi_snapshot_freq": SNAP_FREQ, "policy_snapshot_freq": SNAP_FREQ}

    # ==================================================================
    # STUDY A
    # ==================================================================
    if "a" in args.studies:
        print("\n" + "="*70)
        print(f"STUDY A: Stochastic Success | {STUDY_A_EPS} episodes | seed {args.seed}")
        print("="*70)

        out = os.path.join(base, f"study_a_seed_{args.seed}")
        cfg = {**full_cfg, "total_episodes": STUDY_A_EPS}

        # Train all 3 conditions
        print("\n[A1-baseline]")
        np.random.seed(args.seed)
        env1 = make_fourrooms_maze(); agent1 = TabularSGCRLAgent(env1, **AGENT_CFG)
        m1 = MetricsCollector(env1.num_states, (env1.height, env1.width))
        train_standard(agent1, env1, STUDY_A_EPS, m1, "A1")

        print("\n[A2-allreplay]")
        np.random.seed(args.seed)
        env2a = make_stochastic_fourrooms_maze(APPROACH_PROBS)
        agent2a = StochasticSGCRLAgent_AllReplay(env2a, **AGENT_CFG)
        m2a = MetricsCollector(env2a.num_states, (env2a.height, env2a.width))
        ah2a = train_standard(agent2a, env2a, STUDY_A_EPS, m2a, "A2ar", True)

        print("\n[A2-successonly]")
        np.random.seed(args.seed)
        env2s = make_stochastic_fourrooms_maze(APPROACH_PROBS)
        agent2s = StochasticSGCRLAgent_SuccessOnly(env2s, **AGENT_CFG)
        m2s = MetricsCollector(env2s.num_states, (env2s.height, env2s.width))
        ah2s = train_standard(agent2s, env2s, STUDY_A_EPS, m2s, "A2so", True)

        # Generate visuals
        print("\n>>> Study A visuals")
        t0 = time.time()
        md1, _ = save_condition(cfg, env1, agent1, m1,
            os.path.join(out,"A1-baseline"), "A1-baseline")
        md2a, _ = save_condition(cfg, env2a, agent2a, m2a,
            os.path.join(out,"A2-stochastic-allreplay"), "A2-allreplay",
            ah=ah2a, true_probs=APPROACH_PROBS)
        md2s, _ = save_condition(cfg, env2s, agent2s, m2s,
            os.path.join(out,"A2-stochastic-successonly"), "A2-successonly",
            ah=ah2s, true_probs=APPROACH_PROBS)

        # Comparisons
        comp = os.path.join(out, "comparisons")
        r1 = np.array(md1["m1_rolling_success_rate"])
        r2a = np.array(md2a["m1_rolling_success_rate"])
        r2s = np.array(md2s["m1_rolling_success_rate"])

        comparison_line(r1, r2a, "Success Rate (rolling 50)",
            "M1: Success Rate — Baseline vs Allreplay",
            os.path.join(comp,"m1_success_rate_vs_allreplay.png"), "A1-baseline","A2-allreplay")
        comparison_line(r1, r2s, "Success Rate (rolling 50)",
            "M1: Success Rate — Baseline vs Successonly",
            os.path.join(comp,"m1_success_rate_vs_successonly.png"), "A1-baseline","A2-successonly")
        comparison_line(r2a, r2s, "Success Rate (rolling 50)",
            "M1: Allreplay vs Successonly",
            os.path.join(comp,"m1_allreplay_vs_successonly.png"), "A2-allreplay","A2-successonly")

        for tag, md in [("allreplay",md2a),("successonly",md2s)]:
            comparison_adapt(md1, md, os.path.join(comp,f"m4_m5_adaptation_vs_{tag}.png"),
                             "A1-baseline", f"A2-{tag}")
            comparison_align(md1, md, os.path.join(comp,f"m8_alignment_vs_{tag}.png"),
                             "A1-baseline", f"A2-{tag}")
            for key in ["m3_exploitation_ratio","m6_suboptimality_gap"]:
                d1 = np.array(md1.get(key,[])); d2 = np.array(md.get(key,[]))
                if len(d1)>0 and len(d2)>0:
                    comparison_line(d1, d2, key.split("_",1)[1].replace("_"," ").title(),
                        f"{key.split('_')[0].upper()}: {key.split('_',1)[1].replace('_',' ').title()}",
                        os.path.join(comp,f"{key}_vs_{tag}.png"),
                        "A1-baseline", f"A2-{tag}")

        print(f"  Study A total viz: {time.time()-t0:.0f}s")

    # ==================================================================
    # STUDY B
    # ==================================================================
    if "b" in args.studies:
        total_b = STUDY_B_EPS_PER_PHASE * 3
        print("\n" + "="*70)
        print(f"STUDY B: Changing Dynamics | {total_b} episodes | seed {args.seed}")
        print("="*70)

        out = os.path.join(base, f"study_b_seed_{args.seed}")
        cfg = {**full_cfg, "total_episodes": total_b,
               "episodes_per_phase": STUDY_B_EPS_PER_PHASE}
        pt = [STUDY_B_EPS_PER_PHASE, STUDY_B_EPS_PER_PHASE*2]
        pn = ["fourrooms","corridor_shift","l_wall"]

        print("\n[B1-static]")
        np.random.seed(args.seed)
        env1 = make_static_fourrooms()
        agent1 = TabularSGCRLAgent(env1, **AGENT_CFG)
        m1 = MetricsCollector(env1.num_states, (env1.height, env1.width))
        train_standard(agent1, env1, total_b, m1, "B1")

        print("\n[B2-changing]")
        np.random.seed(args.seed)
        env2 = make_changing_dynamics_maze(STUDY_B_EPS_PER_PHASE)
        agent2 = TabularSGCRLAgent(env2, **AGENT_CFG)
        m2 = MetricsCollector(env2.num_states, (env2.height, env2.width))
        train_b2(agent2, env2, m2, "B2")

        print("\n>>> Study B visuals")
        t0 = time.time()
        mdb1, _ = save_condition(cfg, env1, agent1, m1,
            os.path.join(out,"B1-baseline"), "B1-static", phase_names=["fourrooms"])
        mdb2, _ = save_condition(cfg, env2, agent2, m2,
            os.path.join(out,"B2-changing"), "B2-changing",
            phase_trans=pt, phase_names=pn)

        comp = os.path.join(out, "comparisons")
        rb1 = np.array(mdb1["m1_rolling_success_rate"])
        rb2 = np.array(mdb2["m1_rolling_success_rate"])

        comparison_line(rb1, rb2, "Success Rate (rolling 50)",
            "M1: Success Rate — Static vs Changing",
            os.path.join(comp,"m1_success_rate.png"), "B1-static","B2-changing", pt, pn)
        comparison_adapt(mdb1, mdb2, os.path.join(comp,"m4_m5_adaptation.png"),
                         "B1-static","B2-changing", pt)
        comparison_align(mdb1, mdb2, os.path.join(comp,"m8_alignment.png"),
                         "B1-static","B2-changing", pt)

        for key in ["m2_path_diversity","m3_exploitation_ratio"]:
            d1 = np.array(mdb1.get(key,[])); d2 = np.array(mdb2.get(key,[]))
            if len(d1)>0 and len(d2)>0:
                comparison_line(d1, d2, key.split("_",1)[1].replace("_"," ").title(),
                    f"{key.split('_')[0].upper()}: {key.split('_',1)[1].replace('_',' ').title()}",
                    os.path.join(comp,f"{key}.png"), "B1-static","B2-changing", pt, pn)

        print(f"  Study B total viz: {time.time()-t0:.0f}s")

    print("\n" + "="*70 + "\nALL DONE\n" + "="*70)


if __name__ == "__main__":
    main()
