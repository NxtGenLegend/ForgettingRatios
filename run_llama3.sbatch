#!/bin/bash
#SBATCH --job-name=ctx2-llama3
#SBATCH --account=bggo-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/v2_llama3_%j.out
#SBATCH --error=logs/v2_llama3_%j.err

module load pytorch-conda
conda activate ctx-eval

cd ~/context-eval-v2
python download_filler.py
python run_inference_v2.py --model llama3-8b --task all
