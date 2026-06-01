#!/bin/bash
#SBATCH --job-name=hmbbh-p02
#SBATCH --output=.slurm/logs/hmbbh-p02_%j.out
#SBATCH --error=.slurm/logs/hmbbh-p02_%j.err
#SBATCH --partition=H200-12h
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=480GB
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --mail-user=achimoa@biu.ac.il
#SBATCH --mail-type=END,FAIL,REQUEUE

# Phase 0.2: Hebrew specialization (100B real tok). Loads phase-0.1 WEIGHTS
# (load_weights_only) and runs a fresh WSD schedule. LAUNCH ONLY AFTER phase 0.1
# has produced checkpoints/base/modern-bert-base-phase-0.1-pretrain/ckpt/latest-rank0.pt.
# autoresume=true => resumes its own checkpoint on each 12h requeue.

bash scripts/train_modernbert_base_hebrew_phase-0.2-pretrain.sh
