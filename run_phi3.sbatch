#!/bin/bash
#SBATCH --job-name=ctx2-phi3
#SBATCH --account=bggo-delta-gpu
#SBATCH --partition=gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=logs/v2_phi3_%j.out
#SBATCH --error=logs/v2_phi3_%j.err
#SBATCH --mail-user=adhishch@usc.edu
#SBATCH --mail-type=END,FAIL

module load pytorch-conda
conda activate ctx-eval

cd ~/context-eval-v2
export HF_HUB_OFFLINE=1
export TP_SIZE=4
python download_filler.py
# python run_inference_v2.py --model phi3-mini --task all
python run_inference_v2.py --model phi3-mini --task multi --min-ratio 0.95
python run_inference_v2.py --model phi3-mini --task reasoning
