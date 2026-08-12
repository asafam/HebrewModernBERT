#!/bin/bash
#SBATCH --job-name=hmb-p0-nlp4
#SBATCH --output=.slurm/logs/hmb-p0-nlp4_%j.out
#SBATCH --error=.slurm/logs/hmb-p0-nlp4_%j.err
#SBATCH --partition=p_b200_nlp
#SBATCH --account=ug_goldberg  # p_b200_nlp requires an explicit account (allows ug_tsarfaty,ug_dagan,ug_goldberg)
#SBATCH --nodes=1
#SBATCH --gres=gpu:4           # NLP group entitlement is 4 of the node's 8 B200 GPUs
#SBATCH --ntasks-per-node=1    # one task; Composer spawns the 4 GPU processes (DDP) itself
#SBATCH --cpus-per-task=32
#SBATCH --mem=480GB
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --mail-type=FAIL

# Stage 1 arm 1 (20% masking) for phase-0, on p_b200_nlp's 4-GPU entitlement -- runs
# concurrently with the control arm (30% masking) on p_b200_goldberg's 1 GPU. Same effective
# global_train_batch_size (4608) either way (DDP just changes grad-accum, not the optimization
# target), so the two arms remain a fair A/B comparison at matched token counts even though
# wall-clock differs. autoresume:true + save_interval=2000ba (set in the wrapper) let this
# continue across requeues -- shared node is NOT contention-free (see [[b200-blackwell-env]]),
# so grab GPUs opportunistically per the docs/TRAINING.md contention playbook, don't assume
# this stays free.
#
# Submit via: sbatch --export=ALL,RUN_NAME=phase-0-arm1-m20,MLM_PROB=0.2 .slurm/jobs/train_b200_phase-0_nlp4.slurm

bash scripts/train_modernbert_base_hebrew_phase-0-pretrain.sh
