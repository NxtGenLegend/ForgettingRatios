#!/bin/bash
#SBATCH --job-name=ctx2-mistral
#SBATCH --account=bggo-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/v2_mistral_%j.out
#SBATCH --error=logs/v2_mistral_%j.err

module load pytorch-conda
conda activate ctx-eval

cd ~/context-eval-v2
python download_filler.py
python run_inference_v2.py --model mistral-7b --task all
