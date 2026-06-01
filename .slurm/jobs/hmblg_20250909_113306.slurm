#!/bin/bash
#SBATCH --job-name=hmblg
#SBATCH --output=.slurm/logs/hmblg_%j.out
#SBATCH --error=.slurm/logs/hmblg_%j.err
#SBATCH --partition=H200-12h
#SBATCH --gres=gpu:2
#SBATCH --mem=140GB
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=achimoa@biu.ac.il
#SBATCH --mail-type=END,FAIL

bash scripts/train_modernbert_large_phase-0.1-pretrain.sh 
