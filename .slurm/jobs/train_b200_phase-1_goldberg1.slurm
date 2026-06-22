#!/bin/bash
#SBATCH --job-name=hmbbh-p1-g1
#SBATCH --output=.slurm/logs/hmbbh-p1-g1_%j.out
#SBATCH --error=.slurm/logs/hmbbh-p1-g1_%j.err
#SBATCH --partition=p_b200_goldberg
#SBATCH --account=ug_goldberg
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240GB
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --mail-type=FAIL
# 1-GPU FALLBACK for phase-1 when the 4-GPU p_b200_nlp quota is blocked. Same checkpoints
# (autoresume); global_batch 576 just uses more grad-accum (576/16=36 steps). Slower but
# keeps training instead of idling. Switch back to train_b200_phase-1.slurm when 4 GPUs free.
bash scripts/train_modernbert_base_hebrew_phase-1-contextextension.sh
