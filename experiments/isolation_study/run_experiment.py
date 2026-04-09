#!/usr/bin/env python3
"""
run_experiment.py — HPC-ready experiment runner for the SGCRL isolation study.

Replaces the old run_single.py.  Key improvements:
  - All hyperparameters configurable via CLI
  - SLURM preemption-safe (catches SIGTERM/SIGINT, saves checkpoint before exit)
  - Checkpoint/resume support (--resume PATH)
  - Per-episode JSONL log (appendable mid-run)
  - Snapshots saved as individual .npy files (not one big pickle)
  - run_config.json and run_log.txt written at start
  - tqdm progress bar
  - Explicit device control (--device cuda/cpu)

Usage:
  python run_experiment.py --study A --condition a1 --seed 42 \\
      --episodes 50000 --device cuda --output_dir results/study_a/A1-baseline/seed_42

  python run_experiment.py --condition b2 --seed 0 \\
      --episodes_per_phase 25000 --device cpu --output_dir /tmp/b2_s0
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import pickle
import numpy as np
from datetime import datetime
from pathlib import Path

# ── local imports ─────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from envs import (
    make_fourrooms_maze, make_stochastic_fourrooms_maze,
    make_static_fourrooms, make_changing_dynamics_maze,
)
from agent import (
    NeuralSGCRLAgent,
    StochasticNeuralAgent_AllReplay,
    StochasticNeuralAgent_SuccessOnly,
)
from metrics import MetricsCollector

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

import torch


# ── helpers ───────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class _Logger:
    """Writes timestamped lines to both stdout and a log file."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._f = open(log_path, "a")

    def log(self, msg: str):
        line = f"[{_ts()}] {msg}"
        print(line, flush=True)
        self._f.write(line + "\n")
        self._f.flush()

    def close(self):
        self._f.close()


class _EpisodeJSONL:
    """Append-mode JSONL writer (one JSON object per line)."""

    def __init__(self, path: str, resume: bool = False):
        self.path = path
        mode = "a" if resume else "w"
        self._f = open(path, mode)

    def write(self, obj: dict):
        self._f.write(json.dumps(obj) + "\n")
        self._f.flush()

    def close(self):
        self._f.close()


class _SnapshotIndex:
    """Maintains snapshots/index.json listing all saved snapshot files."""

    def __init__(self, snap_dir: str, resume: bool = False):
        self.snap_dir  = snap_dir
        self.index_path = os.path.join(snap_dir, "index.json")
        os.makedirs(snap_dir, exist_ok=True)
        if resume and os.path.exists(self.index_path):
            with open(self.index_path) as f:
                self._entries = json.load(f)
        else:
            self._entries = []

    def add(self, entry: dict):
        self._entries.append(entry)
        with open(self.index_path, "w") as f:
            json.dump(self._entries, f, indent=2)

    def entries(self):
        return list(self._entries)


def _auto_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ── argument parsing ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SGCRL Isolation Study — HPC experiment runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Experiment identity
    p.add_argument("--study", choices=["A", "B"], default=None,
                   help="Study A or B.  Inferred from condition if not given.")
    p.add_argument("--condition", required=True,
                   choices=["a1", "a2ar", "a2so", "b1", "b2"],
                   help="Condition identifier.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--output_dir", default=None,
                   help="Output directory (auto-named if not given).")

    # Training duration
    p.add_argument("--episodes", type=int, default=50000,
                   help="Total training episodes (Study A conditions).")
    p.add_argument("--episodes_per_phase", type=int, default=25000,
                   help="Episodes per phase for Study B conditions.")

    # Representation
    p.add_argument("--rep_dim", type=int, default=64,
                   help="Embedding / psi dimensionality.")
    p.add_argument("--use_mlp", action="store_true",
                   help="Add a 2-layer MLP projection head on top of embedding.")

    # Optimisation
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Adam learning rate.")
    p.add_argument("--batch_size", type=int, default=512,
                   help="InfoNCE mini-batch size (state pairs).")
    p.add_argument("--temperature", type=float, default=0.07,
                   help="InfoNCE temperature.")
    p.add_argument("--grad_clip", type=float, default=1.0,
                   help="Gradient norm clip (0 = disabled).")

    # Replay
    p.add_argument("--replay_capacity", type=int, default=5000,
                   help="Maximum number of trajectories in replay buffer.")
    p.add_argument("--gamma", type=float, default=0.99,
                   help="Geometric discount for future-state sampling.")

    # Episode
    p.add_argument("--max_steps", type=int, default=200,
                   help="Maximum steps per episode.")

    # Device
    p.add_argument("--device", default=None,
                   help="Compute device: cuda/cpu.  Auto-detected if not given.")

    # Logging / checkpointing
    p.add_argument("--log_freq", type=int, default=100,
                   help="Print progress every N episodes.")
    p.add_argument("--snap_freq", type=int, default=500,
                   help="Save psi/policy snapshot every N episodes.")
    p.add_argument("--checkpoint_freq", type=int, default=5000,
                   help="Save checkpoint every N episodes.")
    p.add_argument("--n_rollouts", type=int, default=5,
                   help="Number of rollouts for policy_rollout evaluation.")

    # Resume
    p.add_argument("--resume", default=None,
                   help="Path to a checkpoint .pt file to resume from.")

    # Study A stochastic approach probs
    p.add_argument("--approach_probs", default=None,
                   help="JSON string or path to JSON file with approach_probs "
                        "(for Study A stochastic conditions).  "
                        "Default: from_above=0.10, from_left=0.25, "
                        "from_right=0.50, from_below=0.80.")
    return p


# ── default output directory logic ───────────────────────────────────────────

_CONDITION_NAMES = {
    "a1":   "A1-baseline",
    "a2ar": "A2-allreplay",
    "a2so": "A2-successonly",
    "b1":   "B1-baseline",
    "b2":   "B2-changing",
}

_STUDY_MAP = {
    "a1": "A", "a2ar": "A", "a2so": "A",
    "b1": "B", "b2":   "B",
}

DEFAULT_APPROACH_PROBS = {
    "from_above": 0.10,
    "from_left":  0.25,
    "from_right": 0.50,
    "from_below": 0.80,
    "stay":       0.00,
}


def _resolve_approach_probs(args) -> dict:
    if args.approach_probs is None:
        return DEFAULT_APPROACH_PROBS
    s = args.approach_probs
    if os.path.exists(s):
        with open(s) as f:
            return json.load(f)
    return json.loads(s)


# ── agent factory ─────────────────────────────────────────────────────────────

def _make_agent(condition: str, env, cfg: dict) -> NeuralSGCRLAgent:
    kwargs = dict(
        rep_dim         = cfg["rep_dim"],
        lr              = cfg["lr"],
        batch_size      = cfg["batch_size"],
        replay_capacity = cfg["replay_capacity"],
        max_steps       = cfg["max_steps"],
        gamma           = cfg["gamma"],
        temperature     = cfg["temperature"],
        grad_clip       = cfg["grad_clip"],
        use_mlp         = cfg["use_mlp"],
        device          = cfg["device"],
    )
    if condition in ("a2ar",):
        return StochasticNeuralAgent_AllReplay(env, **kwargs)
    elif condition in ("a2so",):
        return StochasticNeuralAgent_SuccessOnly(env, **kwargs)
    else:
        return NeuralSGCRLAgent(env, **kwargs)


# ── core training loop ────────────────────────────────────────────────────────

def run_experiment(args):
    # ── resolve study / output dir ───────────────────────────────────────────
    study     = args.study or _STUDY_MAP[args.condition]
    cond_name = _CONDITION_NAMES[args.condition]
    if args.output_dir is None:
        results_root = SCRIPT_DIR / "results" / f"study_{study.lower()}"
        args.output_dir = str(results_root / cond_name / f"seed_{args.seed}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or _auto_device()
    approach_probs = _resolve_approach_probs(args)

    # ── config dict ──────────────────────────────────────────────────────────
    cfg = {
        "study":            study,
        "condition":        args.condition,
        "cond_name":        cond_name,
        "seed":             args.seed,
        "device":           device,
        "rep_dim":          args.rep_dim,
        "lr":               args.lr,
        "batch_size":       args.batch_size,
        "temperature":      args.temperature,
        "grad_clip":        args.grad_clip,
        "replay_capacity":  args.replay_capacity,
        "gamma":            args.gamma,
        "max_steps":        args.max_steps,
        "use_mlp":          args.use_mlp,
        "episodes":         args.episodes,
        "episodes_per_phase": args.episodes_per_phase,
        "snap_freq":        args.snap_freq,
        "checkpoint_freq":  args.checkpoint_freq,
        "log_freq":         args.log_freq,
        "n_rollouts":       args.n_rollouts,
        "approach_probs":   approach_probs,
        "timestamp":        _ts(),
        "resumed_from":     args.resume,
    }

    # ── logger ───────────────────────────────────────────────────────────────
    logger = _Logger(str(out_dir / "run_log.txt"))
    logger.log(f"Starting experiment: study={study} condition={args.condition} "
               f"seed={args.seed} device={device}")

    # ── write run_config.json (idempotent) ───────────────────────────────────
    cfg_path = out_dir / "run_config.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    logger.log(f"Config written to {cfg_path}")

    # ── set seeds ────────────────────────────────────────────────────────────
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── build environment ─────────────────────────────────────────────────────
    if args.condition == "a1":
        env = make_fourrooms_maze()
    elif args.condition in ("a2ar", "a2so"):
        env = make_stochastic_fourrooms_maze(approach_probs)
    elif args.condition == "b1":
        env = make_static_fourrooms()
    elif args.condition == "b2":
        env = make_changing_dynamics_maze(args.episodes_per_phase)
    else:
        raise ValueError(f"Unknown condition: {args.condition}")

    # Write env config
    env_cfg_path = out_dir / "env_config.json"
    with open(env_cfg_path, "w") as f:
        json.dump(env.get_config(), f, indent=2)

    # ── build agent ───────────────────────────────────────────────────────────
    agent = _make_agent(args.condition, env, cfg)
    logger.log(f"Agent: {type(agent).__name__}  rep_dim={args.rep_dim}  "
               f"params={sum(p.numel() for p in agent.embedding.parameters())} emb params")

    # ── metrics collector ─────────────────────────────────────────────────────
    metrics = MetricsCollector(env.num_states, (env.height, env.width))

    # ── snapshot index / episode log ──────────────────────────────────────────
    snap_dir  = str(out_dir / "snapshots")
    ckpt_dir  = str(out_dir / "checkpoints")
    os.makedirs(snap_dir,  exist_ok=True)
    os.makedirs(ckpt_dir,  exist_ok=True)

    snap_index = _SnapshotIndex(snap_dir, resume=args.resume is not None)
    ep_log     = _EpisodeJSONL(
        str(out_dir / "episode_log.jsonl"),
        resume=args.resume is not None,
    )

    # ── resume from checkpoint ─────────────────────────────────────────────────
    start_ep  = 0
    cur_phase = 0
    if args.resume is not None:
        logger.log(f"Resuming from checkpoint: {args.resume}")
        payload  = agent.load_checkpoint(args.resume)
        start_ep = payload.get("episode", 0) + 1
        if "metrics_state" in payload:
            metrics = MetricsCollector.from_state(
                payload["metrics_state"], env.num_states, (env.height, env.width))
        if "cur_phase" in payload:
            cur_phase = payload["cur_phase"]
            if hasattr(env, "set_phase"):
                env.set_phase(cur_phase)
        logger.log(f"Resumed from episode {start_ep}")

    # ── Study B: compute phase boundaries ─────────────────────────────────────
    phase_boundaries = []
    if args.condition == "b2":
        cumulative = 0
        for ph in env.phases:
            phase_boundaries.append(cumulative)
            cumulative += ph.num_episodes
        if cur_phase < len(env.phases):
            env.set_phase(cur_phase)

    # ── total episodes ────────────────────────────────────────────────────────
    if args.condition in ("a1", "a2ar", "a2so"):
        total_eps = args.episodes
    else:  # b1, b2
        total_eps = sum(p.num_episodes for p in env.phases)

    # ── SIGTERM / SIGINT handler (SLURM preemption) ───────────────────────────
    _interrupted = [False]

    def _signal_handler(signum, frame):
        logger.log(f"Signal {signum} received — saving emergency checkpoint…")
        _interrupted[0] = True

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT,  _signal_handler)

    # ── helper: save checkpoint ───────────────────────────────────────────────
    def _save_ckpt(ep: int, tag: str = ""):
        fname = f"ep_{ep:06d}{tag}.pt"
        path  = os.path.join(ckpt_dir, fname)
        extra = {
            "episode":       ep,
            "cur_phase":     cur_phase,
            "metrics_state": metrics.get_state(),
            "config":        cfg,
        }
        extra.update(agent.get_state_for_checkpoint())
        agent.save_checkpoint(path, extra=extra)
        logger.log(f"Checkpoint saved: {path}")

    # ── helper: save snapshots ────────────────────────────────────────────────
    def _save_snaps(ep: int, phase: int):
        psi_snap  = agent.get_psi_snapshot()
        sim_map   = agent.get_similarity_map()
        route     = agent.policy_rollout(n_rollouts=args.n_rollouts)
        pol       = agent.get_policy_distribution()

        psi_file    = f"psi_ep{ep:06d}.npy"
        sim_file    = f"sim_ep{ep:06d}.npy"
        route_file  = f"route_ep{ep:06d}.npy"
        policy_file = f"policy_ep{ep:06d}.npy"

        np.save(os.path.join(snap_dir, psi_file),    psi_snap)
        np.save(os.path.join(snap_dir, sim_file),    sim_map)
        np.save(os.path.join(snap_dir, route_file),  np.array(route, dtype=np.int32))
        np.save(os.path.join(snap_dir, policy_file), pol)

        snap_index.add({
            "episode":      ep,
            "phase":        phase,
            "psi_file":     psi_file,
            "sim_file":     sim_file,
            "route_file":   route_file,
            "policy_file":  policy_file,
        })

        # Also record in MetricsCollector for M4/M5/M8
        metrics.record_psi_snapshot(ep, phase, sim_map)
        metrics.record_psi_full_snapshot(ep, phase, psi_snap)
        metrics.record_greedy_trajectory(ep, phase, route)
        metrics.record_policy_snapshot(ep, phase, pol)

    # ── training loop ─────────────────────────────────────────────────────────
    logger.log(f"Training: {start_ep} → {total_eps} episodes")
    t0 = time.time()

    ep_iter = range(start_ep, total_eps)
    if HAS_TQDM:
        ep_iter = tqdm(ep_iter, initial=start_ep, total=total_eps,
                       desc=f"{cond_name}/seed{args.seed}", dynamic_ncols=True)

    for ep in ep_iter:
        # ── Study B: phase transition logic ──────────────────────────────────
        if args.condition == "b2" and phase_boundaries:
            new_phase = cur_phase
            for i, boundary in enumerate(phase_boundaries):
                if ep >= boundary:
                    new_phase = i
            if new_phase != cur_phase:
                psi_before = agent.get_psi_snapshot()
                env.set_phase(new_phase)
                cur_phase = new_phase
                psi_after = agent.get_psi_snapshot()
                metrics.record_phase_transition(ep, psi_before, psi_after,
                                                cur_phase)
                logger.log(f"Phase transition → {env.current_phase.name} "
                           f"at episode {ep}")

        # ── collect episode ───────────────────────────────────────────────────
        if args.condition in ("a2ar", "a2so"):
            traj, success, adir = agent.collect_episode()
            metrics.record_approach(ep, adir, success)
            extra_log = {"approach_dir": adir}
        else:
            traj, success = agent.collect_episode()
            adir      = None
            extra_log = {}

        metrics.record_episode(ep, cur_phase, traj, success, extra=extra_log)

        # Write to episode JSONL
        ep_log.write({
            "episode": ep,
            "phase":   cur_phase,
            "success": bool(success),
            "steps":   len(traj) - 1,
            **extra_log,
        })

        # ── contrastive update ────────────────────────────────────────────────
        agent.update_representations()

        # ── snapshots ─────────────────────────────────────────────────────────
        if ep % args.snap_freq == 0:
            _save_snaps(ep, cur_phase)

        # ── checkpoint ───────────────────────────────────────────────────────
        if args.checkpoint_freq > 0 and ep > start_ep and ep % args.checkpoint_freq == 0:
            _save_ckpt(ep)

        # ── progress log ─────────────────────────────────────────────────────
        if ep % args.log_freq == 0 and ep > start_ep:
            window_start = max(0, ep - 99)
            sr = float(np.mean(metrics.episode_success[window_start:ep + 1]))
            elapsed = time.time() - t0
            msg = (f"ep={ep}/{total_eps}  "
                   f"phase={cur_phase}  "
                   f"sr100={sr:.3f}  "
                   f"replay={len(agent.replay)}  "
                   f"elapsed={elapsed:.0f}s")
            logger.log(msg)
            if HAS_TQDM and isinstance(ep_iter, tqdm):
                ep_iter.set_postfix(sr=f"{sr:.3f}", phase=cur_phase)

        # ── interrupt check ───────────────────────────────────────────────────
        if _interrupted[0]:
            logger.log("Interrupted — saving emergency checkpoint.")
            _save_ckpt(ep, tag="_interrupted")
            break

    # ── end of training ────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    logger.log(f"Training complete. Total time: {elapsed:.1f}s")

    # Final checkpoint
    _save_ckpt(total_eps - 1, tag="")
    final_path = os.path.join(ckpt_dir, "final.pt")
    agent.save_checkpoint(final_path, extra={
        "episode":       total_eps - 1,
        "cur_phase":     cur_phase,
        "metrics_state": metrics.get_state(),
        "config":        cfg,
    })
    logger.log(f"Final checkpoint: {final_path}")

    # Final snapshot
    _save_snaps(total_eps - 1, cur_phase)

    # Save metrics
    is_stochastic = args.condition in ("a2ar", "a2so")
    metrics.save(str(out_dir), true_probs=approach_probs if is_stochastic else None)

    # Save legacy metrics.json for backward compat with old generate_plots.py
    md = metrics.to_dict(true_probs=approach_probs if is_stochastic else None)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(md, f, indent=2)

    # Save final psi
    np.save(str(out_dir / "psi_final.npy"), agent.get_psi_snapshot())

    # Save legacy snapshots.pkl for backward compat with old generate_plots.py
    _save_legacy_pkl(out_dir, metrics)

    # Save approach stats (stochastic conditions)
    if is_stochastic and hasattr(agent, "get_approach_stats"):
        stats = agent.get_approach_stats()
        with open(out_dir / "approach_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

    # Phase info for Study B
    if args.condition == "b2":
        phase_info = {
            "transitions": [int(b) for b in phase_boundaries[1:]],
            "names":       [p.name for p in env.phases],
        }
        with open(out_dir / "phase_info.json", "w") as f:
            json.dump(phase_info, f, indent=2)

    # Summary
    final_sr = metrics.m1_rolling_current()
    logger.log(f"Final rolling success rate: {final_sr:.4f}")
    logger.log(f"All outputs in: {out_dir}")

    ep_log.close()
    logger.close()
    return out_dir


def _save_legacy_pkl(out_dir: Path, metrics: MetricsCollector):
    """Write snapshots.pkl in the same format as the old run_single.py.

    This keeps generate_plots.py's old code path working.
    """
    import pickle
    psi_snaps = [(ep, ph, arr) for ep, ph, arr in metrics.psi_snapshots]
    gt_snaps  = [(ep, ph, t)   for ep, ph, t  in metrics.greedy_trajectory_snapshots]
    pkl_data  = {
        "psi_snapshots":              psi_snaps,
        "greedy_trajectory_snapshots": gt_snaps,
        "episode_trajectories":        metrics.episode_trajectories[-200:],
        "episode_success":             metrics.episode_success[-200:],
    }
    with open(out_dir / "snapshots.pkl", "wb") as f:
        pickle.dump(pkl_data, f)


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = _build_parser()
    args   = parser.parse_args()
    run_experiment(args)
