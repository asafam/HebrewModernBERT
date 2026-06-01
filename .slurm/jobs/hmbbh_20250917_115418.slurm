#!/bin/bash
#SBATCH --job-name=hmbbh
#SBATCH --output=.slurm/logs/hmbbh_%j.out
#SBATCH --error=.slurm/logs/hmbbh_%j.err
#SBATCH --partition=H200-12h
#SBATCH --gres=gpu:1
#SBATCH --mem=140GB
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=achimoa@biu.ac.il
#SBATCH --mail-type=END,FAIL

bash scripts/train_modernbert_base_hebrew_phase-0.1-pretrain.sh 
