#!/bin/bash
#SBATCH --job-name=hmb-smk0-b200
#SBATCH --output=.slurm/logs/hmb-smk0-b200_%j.out
#SBATCH --error=.slurm/logs/hmb-smk0-b200_%j.err
#SBATCH --partition=p_b200_nlp
#SBATCH --account=ug_goldberg  # p_b200_nlp requires an explicit account (allows ug_tsarfaty,ug_dagan,ug_goldberg)
#SBATCH --nodes=1
#SBATCH --gres=gpu:4           # NLP group entitlement is 4 of the node's 8 B200 GPUs
#SBATCH --ntasks-per-node=1    # one task; Composer spawns the 4 GPU processes (DDP) itself
#SBATCH --cpus-per-task=32
#SBATCH --mem=480GB
#SBATCH --time=01:00:00        # smoke only needs ~30-45min; short walltime backfills sooner
#SBATCH --mail-user=achimoa@biu.ac.il
#SBATCH --mail-type=END,FAIL

# Phase-0 microbatch + throughput probe on B200 (4 GPUs, 1024 ctx, non-packed).
# The node's 8 GPUs are often fully allocated; this job PENDs until 4 free up.
# Reads the auto-chosen device_train_microbatch_size and real tok/s rate into the log --
# needed to size Stage 2's wall-clock for the full 130B-token phase-0 run.

bash scripts/smoke_test_base_hebrew_phase-0_b200.sh
