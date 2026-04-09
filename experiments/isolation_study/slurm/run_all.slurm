#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_all.slurm — Submit all 5 conditions (a1, a2ar, a2so, b1, b2),
#                 each as a 5-seed array job.
#
# Usage:
#   bash slurm/run_all.slurm
#   bash slurm/run_all.slurm --dry_run     # print commands, don't submit
#
# This script calls sbatch to submit 5 independent array jobs.
# Each array job runs 5 seeds in parallel using run_array.slurm.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLURM_DIR="${SCRIPT_DIR}/slurm"
DRY_RUN=0

# Parse --dry_run flag
for arg in "$@"; do
    if [[ "$arg" == "--dry_run" ]]; then
        DRY_RUN=1
    fi
done

mkdir -p "${SCRIPT_DIR}/logs"

echo "=== Submitting all SGCRL isolation study conditions ==="
echo "Script dir: ${SCRIPT_DIR}"
echo "Dry run:    ${DRY_RUN}"
echo ""

submit_array() {
    local condition="$1"
    local extra_args="${2:-}"
    local cmd="sbatch ${SLURM_DIR}/run_array.slurm --condition ${condition} ${extra_args}"
    echo "  ${cmd}"
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        eval "${cmd}"
    fi
}

# Study A conditions (50 000 episodes each)
echo "--- Study A ---"
submit_array "a1"   "--episodes 50000"
submit_array "a2ar" "--episodes 50000"
submit_array "a2so" "--episodes 50000"

echo ""

# Study B conditions (3 × 25 000 episodes each)
echo "--- Study B ---"
submit_array "b1" "--episodes_per_phase 25000"
submit_array "b2" "--episodes_per_phase 25000"

echo ""
echo "=== All jobs submitted. ==="
echo "Monitor with:  squeue -u \$USER"
echo "Cancel all:    scancel -u \$USER"
