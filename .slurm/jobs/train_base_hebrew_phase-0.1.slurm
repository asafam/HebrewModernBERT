#!/bin/bash
#SBATCH --job-name=hmbbh-p01
#SBATCH --output=.slurm/logs/hmbbh-p01_%j.out
#SBATCH --error=.slurm/logs/hmbbh-p01_%j.err
#SBATCH --partition=H200-12h
#SBATCH --nodes=1
#SBATCH --gres=gpu:2          # cluster cap is 2 GPUs/user on the general H200 partitions
#SBATCH --ntasks-per-node=1   # one task; Composer spawns the 2 GPU processes (DDP) itself
#SBATCH --cpus-per-task=16
#SBATCH --mem=480GB
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --mail-user=achimoa@biu.ac.il
#SBATCH --mail-type=END,FAIL,REQUEUE

# Phase 0.1: broad pretraining on the mixed Hebrew+English+code corpus (200B tok).
# autoresume=true in the YAML => each (re)submission resumes from the latest checkpoint
# in checkpoints/base/modern-bert-base-phase-0.1-pretrain/ckpt. With the 12h walltime this
# run spans several requeues; resubmit on timeout (your requeue mechanism) and it continues.
#
# BEFORE launching: run the smoke/throughput probe first
#   (sbatch .slurm/jobs/smoke_base_hebrew_phase-0.1.slurm), then set
#   device_train_microbatch_size in the YAML to the auto-chosen value for best throughput.

bash scripts/train_modernbert_base_hebrew_phase-0.1-pretrain.sh
