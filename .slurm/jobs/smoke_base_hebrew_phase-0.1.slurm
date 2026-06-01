#!/bin/bash
#SBATCH --job-name=hmb-smoke
#SBATCH --output=.slurm/logs/hmb-smoke_%j.out
#SBATCH --error=.slurm/logs/hmb-smoke_%j.err
#SBATCH --partition=H200-4h
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=480GB
#SBATCH --mail-user=achimoa@biu.ac.il
#SBATCH --mail-type=END,FAIL

bash scripts/smoke_test_base_hebrew_phase-0.1.sh
