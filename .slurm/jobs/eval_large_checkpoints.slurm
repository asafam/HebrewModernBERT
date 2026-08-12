#!/bin/bash
#SBATCH --job-name=hmb-lg-evalchk
#SBATCH --output=.slurm/logs/hmb-lg-evalchk_%j.out
#SBATCH --error=.slurm/logs/hmb-lg-evalchk_%j.err
#SBATCH --partition=p_b200_goldberg
#SBATCH --account=ug_goldberg
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120GB
#SBATCH --time=01:00:00
#SBATCH --mail-type=FAIL

# Apples-to-apples eval of LARGE checkpoints, mirroring .slurm/jobs/eval_phase1_checkpoints.slurm.
#
# WHY THIS EXISTS: the W&B eval curve is only comparable if eval config is held fixed. Changing
# eval_subset_num_batches mid-run once produced a fake "rising loss" trend on the base run and
# nearly triggered a needless multi-day redo (docs/TRAINING.md §7). Re-evaluating checkpoints with
# an IDENTICAL config is the discriminating test. Runs on goldberg, separate from the training job,
# with W&B disabled so it cannot touch the training run.
#
# NOTE: init_from_checkpoint is disabled here. On the phase-0 yaml it would tile the base weights
# in before load_path immediately overwrites them -- pure waste, and it would fail if the base
# checkpoint ever moves. Eval must depend only on the checkpoint under test.
#
# Usage:
#   sbatch --export=ALL,YAML=<phase yaml>,CKDIR=<ckpt dir>,BAS="10000 20000" .slurm/jobs/eval_large_checkpoints.slurm
set -e
mkdir -p .slurm/logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bert-b200
cd /home/nlp/achimoa/workspace/HebrewModernBERT

export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/triton-lg-evalchk-${SLURM_JOB_ID:-$$}"
mkdir -p "$TRITON_CACHE_DIR"

YAML="${YAML:-yamls/main/base_hebrew_large/flex-bert-rope-phase-0-pretrain.yaml}"
CKDIR="${CKDIR:-checkpoints/large/modern-bert-large-phase-0-pretrain/ckpt}"
BAS="${BAS:-}"

if [ -z "$BAS" ]; then
    echo "ERROR: set BAS to the batch numbers to evaluate, e.g. BAS=\"10000 20000\"."
    echo "Available checkpoints in $CKDIR:"; ls -1 "$CKDIR" 2>/dev/null | head -30
    exit 1
fi

for BA in $BAS; do
  echo "############################ EVAL ba${BA} ############################"
  python scripts/eval_checkpoint.py "$YAML" \
    load_path="$CKDIR/ep0-ba${BA}-rank0.pt" \
    load_weights_only=true \
    init_from_checkpoint=null \
    eval_subset_num_batches="${SUBSET:-100}" \
    seed=17 \
    autoresume=false \
    save_folder=null \
    loggers='{}' \
    model.model_config.compile_model=false
done
