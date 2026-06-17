#!/bin/bash
#SBATCH --job-name=hmb-smk-b200
#SBATCH --output=.slurm/logs/hmb-smk-b200_%j.out
#SBATCH --error=.slurm/logs/hmb-smk-b200_%j.err
#SBATCH --partition=p_b200_nlp
#SBATCH --nodes=1
#SBATCH --gres=gpu:4           # NLP group entitlement is 4 of the node's 8 B200 GPUs
#SBATCH --ntasks-per-node=1    # one task; Composer spawns the 4 GPU processes (DDP) itself
#SBATCH --cpus-per-task=32
#SBATCH --mem=480GB
#SBATCH --time=01:00:00        # smoke only needs ~30-45min; short walltime backfills sooner
#SBATCH --mail-user=achimoa@biu.ac.il
#SBATCH --mail-type=END,FAIL

# Phase-1 microbatch + throughput probe on B200 (4 GPUs, 8192 context).
# The node's 8 GPUs are often fully allocated; this job PENDs until 4 free up.
# Reads the auto-chosen device_train_microbatch_size into the log for the real run.

bash scripts/smoke_test_base_hebrew_phase-1_b200.sh
