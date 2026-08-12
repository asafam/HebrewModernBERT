#!/bin/bash -i
# Phase 0: unified Hebrew-only pretrain from scratch, 1024 ctx, non-packed (replaces the old
# phase-0.1 mixed + phase-0.2 Hebrew-specialize split). Runs on the Blackwell env (bert-b200):
# torch 2.7/cu128 + flash-attn 2.7.4.post1 (sm_100) + triton 3.3.1 + composer 0.31.
# B200-validated 2026-07-22: microbatch 576/GPU, compile=true OK (no OOM), 317,567 tok/s/device
# clean/uncontended (see [[b200-blackwell-env]] memory).
#
# Stage 1 A/B arms (masking ratio 30% control vs 20%): set RUN_NAME + MLM_PROB env vars before
# sbatch'ing so each arm gets its own checkpoint dir + a fresh (non-colliding) W&B run id.
#   RUN_NAME=phase-0-ctrl-m30 MLM_PROB=0.3 sbatch --export=ALL .slurm/jobs/train_b200_phase-0_goldberg1.slurm
#   RUN_NAME=phase-0-arm1-m20 MLM_PROB=0.2 sbatch --export=ALL .slurm/jobs/train_b200_phase-0_goldberg1.slurm

echo "Activating Conda environment: bert-b200"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bert-b200
cd /home/nlp/achimoa/workspace/HebrewModernBERT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Fresh-per-arm compile caches (not persistent-per-phase like phase-1/2): different arms may
# compile slightly different shapes/configs; keep them from cross-contaminating each other.
export TRITON_CACHE_DIR="$HOME/.cache/hmb/triton-phase0-${RUN_NAME:-default}"
export TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/hmb/inductor-phase0-${RUN_NAME:-default}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

if [ -z "$RUN_NAME" ]; then
    echo "ERROR: RUN_NAME must be set (distinct per arm, else checkpoints/W&B collide)."
    exit 1
fi
MLM_PROB="${MLM_PROB:-0.3}"

# GPU-count-suffixed W&B id: composer's WandBLogger.log_hyperparameters() calls bare
# wandb.config.update(hyperparameters) with no allow_val_change -- verified in wandb's own
# Config._sanitize (wandb_config.py), it only auto-allows changes in Jupyter, so a stable id
# whose locked num_gpus_per_node differs from a resumed run's actual GPU count hard-crashes
# with a ConfigError no init_kwargs setting can suppress. Since checkpoint resume is keyed on
# run_name (unaffected by this), not the W&B id, giving each GPU count its own W&B id sidesteps
# the crash entirely -- costs a fresh W&B curve per GPU-count change, not lost training progress.
NGPUS="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
WANDB_ID="${RUN_NAME}-g${NGPUS}"

echo "Starting composer on main.py: yamls/main/base_hebrew/flex-bert-rope-phase-0-pretrain.yaml"
echo "run_name=$RUN_NAME mlm_probability=$MLM_PROB ngpus=$NGPUS wandb_id=$WANDB_ID"
python -m composer main.py yamls/main/base_hebrew/flex-bert-rope-phase-0-pretrain.yaml \
    run_name="$RUN_NAME" \
    mlm_probability=$MLM_PROB \
    loggers.wandb.init_kwargs.id="$WANDB_ID" \
    loggers.wandb.init_kwargs.name="$WANDB_ID" \
    save_interval=2000ba
