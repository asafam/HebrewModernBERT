#!/bin/bash -i
# HebrewModernBERT-LARGE phase-2: 8192 context, sequence-packed (per-document cu_seqlens, so no
# cross-document attention -- the bug that collapsed base retrieval before it was fixed).
# Loads the previous large phase's WEIGHTS only (fresh optimizer/schedule/clock) via the yaml.
#
# Submit with: sbatch .slurm/jobs/train_b200_large_phase-2.slurm

echo "Activating Conda environment: bert-b200"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bert-b200
cd /home/nlp/achimoa/workspace/HebrewModernBERT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Persistent caches: skips the ~10 min inductor rebuild on each 4h requeue.
export TRITON_CACHE_DIR="$HOME/.cache/hmb/triton-large-phase2"
export TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/hmb/inductor-large-phase2"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

RUN_NAME="${RUN_NAME:-modern-bert-large-phase-2-contextextension}"
MLM_PROB="${MLM_PROB:-0.2}"                # yaml default; override to fork an arm
SAVE_INTERVAL="${SAVE_INTERVAL:-100ba}"    # ctx-ext cadence: frequent ckpts for 4h requeues
NGPUS="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
WANDB_ID="${RUN_NAME}-g${NGPUS}"   # see the phase-0 script for why this suffix is required

echo "Starting composer: yamls/main/base_hebrew_large/flex-bert-rope-phase-2-contextextension.yaml"
echo "run_name=$RUN_NAME mlm_probability=$MLM_PROB save_interval=$SAVE_INTERVAL ngpus=$NGPUS wandb_id=$WANDB_ID"
python -m composer main.py yamls/main/base_hebrew_large/flex-bert-rope-phase-2-contextextension.yaml \
    run_name="$RUN_NAME" \
    mlm_probability=$MLM_PROB \
    save_interval="$SAVE_INTERVAL" \
    loggers.wandb.init_kwargs.id="$WANDB_ID" \
    loggers.wandb.init_kwargs.name="$WANDB_ID"
