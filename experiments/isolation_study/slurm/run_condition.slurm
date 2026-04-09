#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_condition.slurm — Run one condition, one seed of the SGCRL isolation study
#
# Usage:
#   sbatch slurm/run_condition.slurm --condition a1 --seed 42
#   sbatch slurm/run_condition.slurm --condition b2 --seed 43 --episodes_per_phase 25000
#
# Pass any extra arguments for run_experiment.py after the sbatch flags.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=sgcrl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
#SBATCH --mail-type=END,FAIL
# To receive email notifications, add:
#   #SBATCH --mail-user=netid@nyu.edu

set -euo pipefail

# ── Environment setup ─────────────────────────────────────────────────────────
echo "[$(date)] Starting job ${SLURM_JOB_ID} on $(hostname)"
echo "Arguments: $@"

# Load required modules on NYU Greene / Torch HPC
module purge
module load anaconda3/2023.03 2>/dev/null || true
module load cuda/11.8          2>/dev/null || true

# Activate conda environment
conda activate sgcrl || source activate sgcrl || {
    echo "ERROR: could not activate conda env 'sgcrl'"; exit 1
}

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOGS_DIR}"

cd "${SCRIPT_DIR}"

echo "Script dir: ${SCRIPT_DIR}"
echo "Python:     $(which python)"
echo "PyTorch:    $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA avail: $(python -c 'import torch; print(torch.cuda.is_available())')"

# ── Run ───────────────────────────────────────────────────────────────────────
# All extra arguments passed to this script are forwarded to run_experiment.py
python run_experiment.py \
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

echo "[$(date)] Job complete."
