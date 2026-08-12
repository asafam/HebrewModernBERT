#!/bin/bash
#SBATCH --job-name=hmb-p0-g1
#SBATCH --output=.slurm/logs/hmb-p0-g1_%j.out
#SBATCH --error=.slurm/logs/hmb-p0-g1_%j.err
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

# Stage 1 A/B probe arms for phase-0 (unified Hebrew-only pretrain), run on the user's
# exclusive/uncontended 1-GPU allocation (per user direction: prioritize p_b200_goldberg over
# fighting the shared/contended p_b200_nlp queue -- see [[b200-blackwell-env]]). autoresume:true
# + save_interval=2000ba (~9.4B real tok/checkpoint) in the wrapper let this continue across
# many 4h requeues up to checkpoints near the ~10B/~20B token probe targets, and beyond to the
# full 130B if this arm wins Stage 1.
#
# Submit per-arm via --export, e.g.:
#   sbatch --export=ALL,RUN_NAME=phase-0-ctrl-m30,MLM_PROB=0.3 .slurm/jobs/train_b200_phase-0_goldberg1.slurm
#   sbatch --export=ALL,RUN_NAME=phase-0-arm1-m20,MLM_PROB=0.2 .slurm/jobs/train_b200_phase-0_goldberg1.slurm

bash scripts/train_modernbert_base_hebrew_phase-0-pretrain.sh
