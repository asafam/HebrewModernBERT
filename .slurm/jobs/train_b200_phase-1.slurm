#!/bin/bash
#SBATCH --job-name=hmbbh-p1-b200
#SBATCH --output=.slurm/logs/hmbbh-p1-b200_%j.out
#SBATCH --error=.slurm/logs/hmbbh-p1-b200_%j.err
#SBATCH --partition=p_b200_nlp
#SBATCH --account=ug_goldberg     # p_b200_nlp allows ug_tsarfaty,ug_dagan,ug_goldberg
#SBATCH --nodes=1
#SBATCH --gres=gpu:4              # NLP entitlement: 4 of the node's 8 B200 GPUs
#SBATCH --ntasks-per-node=1       # one task; Composer spawns the 4 GPU processes (DDP)
#SBATCH --cpus-per-task=32
#SBATCH --mem=480GB
#SBATCH --time=04:00:00           # p_b200_nlp wall is 4h
#SBATCH --requeue
#SBATCH --mail-user=achimoa@biu.ac.il
#SBATCH --mail-type=END,FAIL,REQUEUE

# Phase 1: context extension 1024 -> 8192 on 4x B200 (~30B tokens).
# autoresume=true + spin_dataloaders=false in the YAML => each 4h requeue resumes the
# latest phase-1 checkpoint cleanly. First start loads phase-0.2 weights (load_weights_only).
# Microbatch 144/GPU smoke-validated at 8192 with compile=true.

bash scripts/train_modernbert_base_hebrew_phase-1-contextextension.sh
