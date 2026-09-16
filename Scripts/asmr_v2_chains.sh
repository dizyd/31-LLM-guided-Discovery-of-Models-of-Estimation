#!/bin/bash

#SBATCH --job-name=asmr_chains
#SBATCH --partition=gpu_h100,gpu_h100_il,gpu_a100_il
#SBATCH --array=0-19%8
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120000
#SBATCH --output=logs/asmr_chain_%A_%a.out
#SBATCH --error=logs/asmr_chain_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=izydorczyk@uni-mannheim.de

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
module load devel/cuda/12.8
VENV=${VENV:-$HOME/ASMR}
if [[ ! -f "$VENV/bin/activate" ]]; then
    echo "virtualenv not found: $VENV/bin/activate (HOME=$HOME, host=$(hostname))" >&2
    exit 1
fi
source "$VENV/bin/activate"
echo "python: $(command -v python)"
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

if [[ -z "${MAX_SECONDS:-}" && -n "${SLURM_JOB_END_TIME:-}" ]]; then
    MAX_SECONDS=$(( SLURM_JOB_END_TIME - $(date +%s) - 1800 ))
fi
MAX_SECONDS=${MAX_SECONDS:-12600}


echo "job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT} on $(hostname)"
echo "started $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python run_asmr_chain.py \
    --domain        "$DOMAIN" \
    --n-runs        "$N_RUNS" \
    --n-iterations  "$N_ITERATIONS" \
    --model         "$LLM" \
    --n-jobs        "$SLURM_CPUS_PER_TASK" \
    --max-seconds   "$MAX_SECONDS"

echo "finished $(date -Is)"
