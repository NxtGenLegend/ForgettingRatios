#!/bin/bash
# Job array over new model comparison pairs.
# Each task ID maps to one model key via the MODELS / TP_SIZES arrays below.
#
# Submit all 8 models:
#   sbatch run_experiments.sbatch
#
# Submit a subset (e.g. only Qwen 7B pair, indices 0–1):
#   sbatch --array=0-1 run_experiments.sbatch
#
# Submit a single model (e.g. gemma2-27b, index 4):
#   sbatch --array=4 run_experiments.sbatch
#
# NOTE: qwen25-14b-1m (index 3) requests TP_SIZE=4, but may need 8 GPUs if
# KV-cache OOMs at the highest ratios. If that happens, re-submit with:
#   sbatch --array=3 --gres=gpu:8 run_experiments.sbatch
# after temporarily editing TP_SIZES[3] to 8 below.

#SBATCH --job-name=forgetting-exp
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --error=logs/slurm-%A_%a.err
#SBATCH --mail-user=rnene@usc.edu
#SBATCH --mail-type=ALL

# ── Model index table ──────────────────────────────────────────────────────────
MODELS=(
    qwen25-7b-128k    # 0
    qwen25-7b-1m      # 1
    qwen25-14b-128k   # 2
    qwen25-14b-1m     # 3
    gemma2-27b        # 4
    gemma3-27b        # 5
    olmo2-32b         # 6
    olmo3-32b         # 7
)

TP_SIZES=(
    2   # qwen25-7b-128k
    4   # qwen25-7b-1m
    4   # qwen25-14b-128k
    4   # qwen25-14b-1m  (increase to 8 + --gres=gpu:8 if OOM)
    4   # gemma2-27b
    4   # gemma3-27b
    4   # olmo2-32b
    4   # olmo3-32b
)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
export TP_SIZE=${TP_SIZES[$SLURM_ARRAY_TASK_ID]}

# ── Environment ────────────────────────────────────────────────────────────────
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=8

mkdir -p logs

echo "============================================"
echo "Array task : $SLURM_ARRAY_TASK_ID"
echo "Model      : $MODEL"
echo "TP_SIZE    : $TP_SIZE"
echo "Node       : $SLURMD_NODENAME"
echo "============================================"

python big_e2e.py --model "$MODEL" --task all --n 5 --out-prefix e2e
