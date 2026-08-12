#!/bin/bash -i
# HebrewModernBERT-LARGE phase-1 (context extension to 8192, sequence-packed) microbatch probe.
#
# The 8192 phases are where the base run hit its six distinct OOM/kernel bugs (docs/TRAINING.md §5).
# Large is ~2.2x the params, so the microbatch must be re-found from scratch: the yaml ships 8
# (base uses 16). Two hard limits apply at once:
#   - memory: dense-8192 activations + the 150K-vocab logits, which scale with hidden size
#   - the flash-attn triton rotary int32 ceiling, which is LOWER for large (16 heads vs 12)
# Validate on 1 GPU (no DDP) FIRST, per docs/TRAINING.md §8.
#
# Run via slurm: sbatch .slurm/jobs/smoke_b200_large_phase-1_1gpu.slurm

# DATALOADER KNOBS (NW / PF). The 1-GPU phase-0 probe measured a hard period-8 stall pattern
# -- one 65-120s block every `num_workers` batches against a 4.0s steady state, a ~5x effective
# penalty. Each worker assembles a full device batch from the 965GB NFS corpus, far slower than
# the GPU consumes them, so the prefetch queue drains. NW/PF let a probe price the fix without
# editing the yaml. These are throughput-only: DataLoader hands whole batches to workers
# round-robin and order is fixed by the sampler + shuffle_seed, so batch composition is
# unchanged -- NOT a recipe divergence from the base run.
#   sbatch --export=ALL,COMPILE=true,MB=576,MAXBA=30,NW=16,PF=6 <this job>
echo "Activating Conda environment: bert-b200"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bert-b200
cd /home/nlp/achimoa/workspace/HebrewModernBERT

export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/triton-cache-${SLURM_JOB_ID:-$$}"
export TORCHINDUCTOR_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/inductor-cache-${SLURM_JOB_ID:-$$}"
rm -rf "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"; mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# MB defaults to 8 rather than `auto`: at 8192 an auto-search can trip the rotary int32 kernel
# limit (an illegal memory access, not a clean OOM) before it finds the memory ceiling.
python -m composer main.py yamls/main/base_hebrew_large/flex-bert-rope-phase-1-contextextension.yaml \
    run_name=smoke-large-phase-1-b200 \
    device_train_microbatch_size=${MB:-8} \
    max_duration=${MAXBA:-40}ba \
    eval_subset_num_batches=2 \
    eval_interval=1000ba \
    save_interval=1000ba \
    save_num_checkpoints_to_keep=1 \
    autoresume=false \
    save_overwrite=true \
    model.model_config.compile_model=${COMPILE:-false} \
    train_loader.num_workers=${NW:-8} \
    train_loader.prefetch_factor=${PF:-2} \
    log_to_console=true \
    console_log_interval=10ba
RC=$?
# Propagate the probe's exit status. Without this the trailing echo becomes the script's last
# command and slurm records COMPLETED 0:0 even when the probe crashed -- a silent green light.
if [ $RC -ne 0 ]; then echo "SMOKE FAILED: composer exited $RC (see .err above)"; exit $RC; fi
echo "Done. If this survives 40 steps, retry once at MB=12 to find the real ceiling;"
echo "if it OOMs or hits an illegal memory access, drop to MB=4."
