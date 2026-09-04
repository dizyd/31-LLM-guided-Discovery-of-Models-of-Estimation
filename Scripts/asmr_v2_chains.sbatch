#!/bin/bash
# =============================================================================
# ASMR v2 -- the 20 discovery chains, as a Slurm job array on bwUniCluster 3.0.
#
#   sbatch asmr_v2_chains.sbatch
#
# One array task = one chain = one (seed model, run id) pair, 5 iterations long.
# The array is 0-19 because the grid is 4 seeds x 5 runs; run_asmr_chain.py reads
# $SLURM_ARRAY_TASK_ID and $SLURM_ARRAY_TASK_COUNT and takes chains[task::count],
# so the array size and the grid stay in sync on their own -- change --n-runs and
# you only have to change the --array range to match.
#
# gpu_h100 allows 72 h, against JupyterHub's 4 h. 12 h is set here because a single
# chain should need 1-3 h; a shorter walltime queues sooner. Resubmitting the exact
# same command is safe and is how you finish an interrupted run: every chain that
# already wrote a DONE.json is skipped.
#
# Afterwards:
#   python merge_asmr_library.py --domain Mammals
#   sbatch asmr_v2_evaluate.sbatch
# =============================================================================

#SBATCH --job-name=asmr_chains
#SBATCH --partition=gpu_h100
#SBATCH --array=0-19%8
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=180000
#SBATCH --output=logs/asmr_chain_%A_%a.out
#SBATCH --error=logs/asmr_chain_%A_%a.err
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=david.izydorczyk@gmail.com

set -euo pipefail

DOMAIN=${DOMAIN:-Mammals}
N_ITERATIONS=${N_ITERATIONS:-5}
N_RUNS=${N_RUNS:-5}
LLM=${LLM:-unsloth/Qwen3-32B-bnb-4bit}

# --------------------------------------------------------------- environment
# EDIT ME. This is the only site-specific part of the script: reproduce whatever
# your JupyterHub kernel already has (unsloth, transformers, torch, scipy, pandas).
# Check with `module avail devel/cuda` which CUDA versions this cluster offers.
module purge
module load devel/cuda/12.4
source "$HOME/venvs/asmr/bin/activate"
# ... or, if the environment is a conda one:
# module load devel/miniforge
# conda activate asmr

# The 32B checkpoint is ~20 GB. $HOME has a small quota on bwUniCluster, so keep the
# Hugging Face cache in a workspace (`ws_allocate asmr 90`, then `ws_find asmr`).
export HF_HOME=${HF_HOME:-$(ws_find asmr 2>/dev/null || echo "$HOME")/hf_cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}

# The per-participant fits run in a multiprocessing.Pool of $SLURM_CPUS_PER_TASK
# workers. Leave BLAS single-threaded or the workers fight each other for cores.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# -------------------------------------------------------------------- run it
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT} on $(hostname)"
echo "started $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python run_asmr_chain.py \
    --domain        "$DOMAIN" \
    --n-runs        "$N_RUNS" \
    --n-iterations  "$N_ITERATIONS" \
    --model         "$LLM" \
    --n-jobs        "$SLURM_CPUS_PER_TASK" \
    --max-seconds   $((11 * 3600))

echo "finished $(date -Is)"
