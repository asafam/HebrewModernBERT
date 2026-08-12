#!/bin/bash -i
# HebrewModernBERT-LARGE phase-0: Hebrew-only pretrain at 1024 ctx, WARM-STARTED from the trained
# base via Phi-style weight tiling (init_from_checkpoint in the yaml). Blackwell env (bert-b200):
# torch 2.7/cu128 + flash-attn 2.7.4.post1 (sm_100) + composer 0.31.
#
# Submit with:
#   sbatch .slurm/jobs/train_b200_large_phase-0_nlp4.slurm       # 4 GPU, p_b200_nlp
#   sbatch .slurm/jobs/train_b200_large_phase-0_goldberg1.slurm  # 1 GPU fallback, never idle
#
# Arm-forking harness, same shape as the base phase-0 script: set RUN_NAME + MLM_PROB before
# sbatch so each arm gets its own checkpoint dir and a non-colliding W&B run id, e.g.
#   sbatch --export=ALL,RUN_NAME=large-arm-m40,MLM_PROB=0.4 .slurm/jobs/train_b200_large_phase-0_nlp4.slurm
# Unset, both fall through to the YAML (run_name=modern-bert-large-phase-0-pretrain, mlm 0.2).

echo "Activating Conda environment: bert-b200"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bert-b200
cd /home/nlp/achimoa/workspace/HebrewModernBERT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# PERSISTENT caches (not per-run): the env/versions are frozen, and this run will requeue ~33 times
# against the 4h wall -- a cold inductor cache costs ~10 min of every one of those restarts.
export TRITON_CACHE_DIR="$HOME/.cache/hmb/triton-large-phase0"
export TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/hmb/inductor-large-phase0"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

RUN_NAME="${RUN_NAME:-modern-bert-large-phase-0-pretrain}"
MLM_PROB="${MLM_PROB:-0.2}"                 # yaml default; override to fork a masking-ratio arm
SAVE_INTERVAL="${SAVE_INTERVAL:-2000ba}"    # ~9.4B real tok/ckpt at the base cadence

# GPU-count-suffixed W&B id: composer's WandBLogger.log_hyperparameters() calls bare
# wandb.config.update() with no allow_val_change, so a stable id whose locked num_gpus_per_node
# differs from a resumed run's actual GPU count hard-crashes with a ConfigError that no init_kwargs
# setting suppresses. Checkpoint resume is keyed on run_name (unaffected), so giving each GPU count
# its own W&B id sidesteps it entirely -- costs a fresh curve per scale-up, not training progress.
# This WILL happen on a large run: p_b200_nlp caps at 4 GPUs / 4h and is shared, so bouncing
# between 1, 2 and 4 GPUs across requeues is the normal case (docs/TRAINING.md §6).
NGPUS="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
WANDB_ID="${RUN_NAME}-g${NGPUS}"

echo "Starting composer: yamls/main/base_hebrew_large/flex-bert-rope-phase-0-pretrain.yaml"
echo "run_name=$RUN_NAME mlm_probability=$MLM_PROB save_interval=$SAVE_INTERVAL ngpus=$NGPUS wandb_id=$WANDB_ID"
python -m composer main.py yamls/main/base_hebrew_large/flex-bert-rope-phase-0-pretrain.yaml \
    run_name="$RUN_NAME" \
    mlm_probability=$MLM_PROB \
    save_interval="$SAVE_INTERVAL" \
    loggers.wandb.init_kwargs.id="$WANDB_ID" \
    loggers.wandb.init_kwargs.name="$WANDB_ID"
