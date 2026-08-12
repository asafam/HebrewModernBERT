#!/bin/bash
#SBATCH --job-name=hmb-smk0-b200
#SBATCH --output=.slurm/logs/hmb-smk0-b200_%j.out
#SBATCH --error=.slurm/logs/hmb-smk0-b200_%j.err
#SBATCH --partition=p_b200_goldberg
#SBATCH --account=ug_goldberg  # p_b200_goldberg requires the ug_goldberg account
#SBATCH --nodes=1
#SBATCH --gres=gpu:1           # Goldberg 1-GPU allocation: keeps the 4-GPU p_b200_nlp budget
                               # free for the real run. device_train_microbatch_size is PER-GPU,
                               # so the fit found here is exactly what the 4-GPU run will use
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120GB
#SBATCH --time=01:00:00
#SBATCH --mail-user=achimoa@biu.ac.il
#SBATCH --mail-type=END,FAIL

# Phase-0 microbatch + throughput probe on a SINGLE B200 (1024 ctx, non-packed, unlike
# phase-1/2's 8192 packed context -- expect a much larger microbatch to fit here).
# Schedules far faster than the 4-GPU job (1 free GPU vs 4). The auto-chosen
# device_train_microbatch_size transfers directly to the real 4-GPU run; only
# multi-GPU throughput scaling is not measured here (expect ~4x this rate on 4 GPUs).

bash scripts/smoke_test_base_hebrew_phase-0_b200.sh
