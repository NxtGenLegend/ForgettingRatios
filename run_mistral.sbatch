#!/bin/bash
#SBATCH --job-name=ctx2-mistral
#SBATCH --account=bggo-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/v2_mistral_%j.out
#SBATCH --error=logs/v2_mistral_%j.err
#SBATCH --mail-user=adhishch@usc.edu
#SBATCH --mail-type=END,FAIL

module load pytorch-conda
conda activate ctx-eval

cd ~/context-eval-v2
python download_filler.py
python run_inference_v2.py --model mistral-7b --task all
