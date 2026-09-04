#!/bin/bash
# =============================================================================
# ASMR v2 -- score the discovered library and build the winner table.
#
#   python merge_asmr_library.py --domain Mammals   # first, on the login node
#   sbatch asmr_v2_evaluate.sbatch
#
# or chained straight onto the array so it starts by itself:
#
#   JOBID=$(sbatch --parsable asmr_v2_chains.sbatch)
#   sbatch --dependency=afterok:$JOBID asmr_v2_evaluate.sbatch
#
# (with the dependency form, uncomment the merge line below -- the array must have
# finished before the shards can be merged).
#
# No GPU: this is scipy optimisation only, every library member x every participant
# x 5 folds, which is embarrassingly parallel over participants. So it takes a whole
# cpu node and puts all 96 cores into --n-jobs.
# =============================================================================

#SBATCH --job-name=asmr_eval
#SBATCH --partition=cpu
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=180000
#SBATCH --output=logs/asmr_eval_%j.out
#SBATCH --error=logs/asmr_eval_%j.err

set -euo pipefail

DOMAIN=${DOMAIN:-Mammals}

# --------------------------------------------------------------- environment
# EDIT ME -- keep this in sync with asmr_v2_chains.sbatch (no CUDA needed here).
module purge
source "$HOME/venvs/asmr/bin/activate"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# -------------------------------------------------------------------- run it
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "job ${SLURM_JOB_ID} on $(hostname), started $(date -Is)"

# python merge_asmr_library.py --domain "$DOMAIN"

python run_asmr_evaluate.py \
    --domain "$DOMAIN" \
    --n-jobs "$SLURM_CPUS_PER_TASK"

echo "finished $(date -Is)"
