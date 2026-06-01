#!/bin/bash
#SBATCH --job-name=hmb-smoke
#SBATCH --output=.slurm/logs/hmb-smoke_%j.out
#SBATCH --error=.slurm/logs/hmb-smoke_%j.err
#SBATCH --partition=H200-12h
#SBATCH --gres=gpu:1
#SBATCH --mem=140GB
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=achimoa@biu.ac.il
#SBATCH --mail-type=END,FAIL

bash scripts/smoke_test_base_hebrew_phase-0.1.sh
