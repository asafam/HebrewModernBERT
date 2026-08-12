#!/bin/bash -i
# HebrewModernBERT-LARGE phase-0 smoke: tile-init verification + microbatch/throughput probe.
#
# Two jobs in one, in this order (the first is the cheap gate for the second):
#   1) scripts/verify_tile_init.py -- confirms the base->large Phi-style weight tiling actually
#      reached the token embeddings. This is the silent failure mode: with tie_word_embeddings=True
#      the MLM decoder is re-tied to tok_embeddings, so if tiling skips them, 31% of the model
#      trains from random init while the loss curve still looks plausible. Discriminator is the
#      embedding std: trained base = 0.113, fresh init = ~0.02.
#   2) the REAL phase-0 config with device_train_microbatch_size=auto so Composer binary-searches
#      the largest microbatch that fits, plus a throughput number to size the full 130B run.
#
# After this finishes, read the chosen microbatch from the log ("Setting device_train_microbatch_size
# to N") and set it in yamls/main/base_hebrew_large/flex-bert-rope-phase-0-pretrain.yaml, backed off
# one notch (the probe runs compile off for a fast, reliable search; compile shifts the profile).
#
# Run via slurm: sbatch .slurm/jobs/smoke_b200_large_phase-0_1gpu.slurm   (1 GPU, schedules fast)
#            or: sbatch .slurm/jobs/smoke_b200_large_phase-0.slurm        (4 GPU, real throughput)

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
conda activate bert-b200   # Blackwell env: torch 2.7/cu128 + flash-attn 2.7.4.post1 (sm_100)
cd /home/nlp/achimoa/workspace/HebrewModernBERT

export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Fresh per-job triton cache: the shared ~/.triton cross-contaminates across triton versions and
# poisons flash-attn's rotary kernel with stale cubins -> PY_SSIZE_T_CLEAN at _init_handles.
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/triton-cache-${SLURM_JOB_ID:-$$}"
export TORCHINDUCTOR_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/inductor-cache-${SLURM_JOB_ID:-$$}"
rm -rf "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"; mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

YAML=yamls/main/base_hebrew_large/flex-bert-rope-phase-0-pretrain.yaml

echo "=============================================================="
echo "STEP 1/2: verifying base -> large weight tiling"
echo "=============================================================="
python scripts/verify_tile_init.py "$YAML" || {
    echo "ABORTING: tile-init verification failed. Do not launch the long run."
    exit 1
}

echo
echo "=============================================================="
echo "STEP 2/2: microbatch + throughput probe (MB=${MB:-auto}, COMPILE=${COMPILE:-false})"
echo "=============================================================="
# Override for the compile=true de-risk once a microbatch is known:
#   sbatch --export=ALL,COMPILE=true,MB=256 .slurm/jobs/smoke_b200_large_phase-0_1gpu.slurm
python -m composer main.py "$YAML" \
    run_name=smoke-large-phase-0-b200 \
    device_train_microbatch_size=${MB:-auto} \
    max_duration=${MAXBA:-60}ba \
    scheduler.t_warmup=0ba \
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
echo "Done. Read the auto-chosen device_train_microbatch_size from the log above,"
echo "and note the token throughput (real tokens at 1024 context, non-packed)."
