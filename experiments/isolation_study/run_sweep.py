#!/usr/bin/env python3
"""
run_sweep.py — Multi-seed sweep launcher for the SGCRL isolation study.

Detects whether it is running inside a SLURM job.  Outside SLURM it runs seeds
sequentially using subprocess.  Inside SLURM it prints the array-submission
commands rather than launching them (submission is done by run_array.slurm).

Usage:
  python run_sweep.py \\
      --study A \\
      --condition a1 \\
      --seeds 42 43 44 45 46 \\
      --episodes 50000 \\
      --device cuda \\
      --output_dir results/study_a

  # Only run specific seeds
  python run_sweep.py --condition b2 --seeds 42 43 --episodes_per_phase 25000

  # Dry-run (print commands only, do not execute)
  python run_sweep.py --condition a1 --seeds 42 43 --dry_run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RUN_EXP    = str(SCRIPT_DIR / "run_experiment.py")

# ── condition → study mapping ─────────────────────────────────────────────────
_STUDY_MAP = {
    "a1": "A", "a2ar": "A", "a2so": "A",
    "b1": "B", "b2":   "B",
}
_COND_NAMES = {
    "a1":   "A1-baseline",
    "a2ar": "A2-allreplay",
    "a2so": "A2-successonly",
    "b1":   "B1-baseline",
    "b2":   "B2-changing",
}


def _build_parser():
    p = argparse.ArgumentParser(
        description="Multi-seed sweep launcher for the SGCRL isolation study.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--study", choices=["A", "B"], default=None)
    p.add_argument("--condition", required=True,
                   choices=["a1", "a2ar", "a2so", "b1", "b2"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46],
                   help="List of seeds to run.")
    p.add_argument("--output_dir", default=None,
                   help="Root results directory.  Seed sub-dirs are created inside.")

    # Pass-through to run_experiment.py
    p.add_argument("--episodes",           type=int,   default=50000)
    p.add_argument("--episodes_per_phase", type=int,   default=25000)
    p.add_argument("--device",             default=None)
    p.add_argument("--rep_dim",            type=int,   default=64)
    p.add_argument("--lr",                 type=float, default=1e-3)
    p.add_argument("--batch_size",         type=int,   default=512)
    p.add_argument("--temperature",        type=float, default=0.07)
    p.add_argument("--replay_capacity",    type=int,   default=5000)
    p.add_argument("--max_steps",          type=int,   default=200)
    p.add_argument("--checkpoint_freq",    type=int,   default=5000)
    p.add_argument("--snap_freq",          type=int,   default=500)
    p.add_argument("--log_freq",           type=int,   default=100)
    p.add_argument("--n_rollouts",         type=int,   default=5)
    p.add_argument("--use_mlp",            action="store_true")
    p.add_argument("--approach_probs",     default=None)

    # Sweep control
    p.add_argument("--dry_run", action="store_true",
                   help="Print commands without executing them.")
    p.add_argument("--sequential", action="store_true",
                   help="Force sequential execution even if on SLURM.")
    return p


def _build_cmd(args, seed: int, out_dir: str) -> list[str]:
    """Build the command-line list for one seed."""
    study = args.study or _STUDY_MAP[args.condition]
    cmd   = [
        sys.executable, RUN_EXP,
        "--study",           study,
        "--condition",       args.condition,
        "--seed",            str(seed),
        "--output_dir",      out_dir,
        "--episodes",        str(args.episodes),
        "--episodes_per_phase", str(args.episodes_per_phase),
        "--rep_dim",         str(args.rep_dim),
        "--lr",              str(args.lr),
        "--batch_size",      str(args.batch_size),
        "--temperature",     str(args.temperature),
        "--replay_capacity", str(args.replay_capacity),
        "--max_steps",       str(args.max_steps),
        "--checkpoint_freq", str(args.checkpoint_freq),
        "--snap_freq",       str(args.snap_freq),
        "--log_freq",        str(args.log_freq),
        "--n_rollouts",      str(args.n_rollouts),
    ]
    if args.device:
        cmd += ["--device", args.device]
    if args.use_mlp:
        cmd.append("--use_mlp")
    if args.approach_probs:
        cmd += ["--approach_probs", args.approach_probs]
    return cmd


def main():
    parser = _build_parser()
    args   = parser.parse_args()

    study     = args.study or _STUDY_MAP[args.condition]
    cond_name = _COND_NAMES[args.condition]

    # Determine results root
    if args.output_dir is None:
        root = SCRIPT_DIR / "results" / f"study_{study.lower()}"
    else:
        root = Path(args.output_dir)

    on_slurm = "SLURM_JOB_ID" in os.environ
    print(f"Sweep: study={study} condition={cond_name} "
          f"seeds={args.seeds} on_slurm={on_slurm}")

    if on_slurm and not args.sequential:
        # Inside a SLURM job: just print what we would do.
        # Actual parallel execution is via sbatch run_array.slurm.
        print("Running inside SLURM — printing seed commands (execution via SLURM array):")
        for seed in args.seeds:
            seed_dir = str(root / cond_name / f"seed_{seed}")
            cmd = _build_cmd(args, seed, seed_dir)
            print("  " + " ".join(cmd))
        return

    # ── Sequential execution (outside SLURM or --sequential) ──────────────────
    for i, seed in enumerate(args.seeds):
        seed_dir = str(root / cond_name / f"seed_{seed}")
        cmd = _build_cmd(args, seed, seed_dir)

        print(f"\n[{i+1}/{len(args.seeds)}] Seed {seed}")
        print("  CMD:", " ".join(cmd))

        if args.dry_run:
            print("  (dry-run — skipping)")
            continue

        ret = subprocess.call(cmd)
        if ret != 0:
            print(f"  WARNING: seed {seed} exited with code {ret}")
        else:
            print(f"  Seed {seed} completed successfully.")

    print("\nSweep done.")


if __name__ == "__main__":
    main()
