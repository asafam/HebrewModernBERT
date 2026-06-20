#!/bin/bash
#SBATCH --job-name=hmbbh-p2-b200
#SBATCH --output=.slurm/logs/hmbbh-p2-b200_%j.out
#SBATCH --error=.slurm/logs/hmbbh-p2-b200_%j.err
#SBATCH --partition=p_b200_nlp
#SBATCH --account=ug_goldberg
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=480GB
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --mail-user=achimoa@biu.ac.il
#SBATCH --mail-type=END,FAIL,REQUEUE

# Phase 2: final anneal at 8192 on 4x B200 (~50B tokens, one_minus_sqrt decay).
# Loads phase-1 weights (load_weights_only). autoresume + spin_dataloaders=false +
# fresh triton cache => clean 4h requeues. LAUNCH AFTER phase-1 completes.

bash scripts/train_modernbert_base_hebrew_phase-2-contextextension.sh
