#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_array.slurm — SLURM array job: run 5 seeds of ONE condition in parallel
#
# Usage:
#   sbatch slurm/run_array.slurm --condition a1
#   sbatch slurm/run_array.slurm --condition b2 --episodes_per_phase 25000
#
# SLURM_ARRAY_TASK_ID maps to seeds: 0→42, 1→43, 2→44, 3→45, 4→46
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=sgcrl_array
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%A_%a_%x.out
#SBATCH --error=logs/%A_%a_%x.err
#SBATCH --mail-type=END,FAIL
# Add to receive email:
#   #SBATCH --mail-user=netid@nyu.edu

set -euo pipefail

# ── Seed mapping ──────────────────────────────────────────────────────────────
SEEDS=(42 43 44 45 46)
SEED="${SEEDS[${SLURM_ARRAY_TASK_ID}]}"

echo "[$(date)] Array job ${SLURM_ARRAY_JOB_ID}[${SLURM_ARRAY_TASK_ID}]"
echo "Seed: ${SEED}  |  Arguments: $@"

# ── Environment setup ─────────────────────────────────────────────────────────
module purge
module load anaconda3/2023.03 2>/dev/null || true
module load cuda/11.8          2>/dev/null || true

conda activate sgcrl || source activate sgcrl || {
    echo "ERROR: could not activate conda env 'sgcrl'"; exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${SCRIPT_DIR}/logs"
cd "${SCRIPT_DIR}"

echo "Python:     $(which python)"
echo "CUDA avail: $(python -c 'import torch; print(torch.cuda.is_available())')"

# ── Run ───────────────────────────────────────────────────────────────────────
python run_experiment.py \
    --seed "${SEED}" \
    --device cuda \
    --episodes 50000 \
    --episodes_per_phase 25000 \
    --rep_dim 64 \
    --lr 1e-3 \
    --batch_size 512 \
    --temperature 0.07 \
    --replay_capacity 5000 \
    --max_steps 200 \
    --checkpoint_freq 5000 \
    --snap_freq 500 \
    --log_freq 100 \
    "$@"

echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID} complete (seed=${SEED})."
