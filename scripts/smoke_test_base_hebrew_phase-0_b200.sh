#!/bin/bash -i
# Phase-0 (unified Hebrew-only pretrain, 1024 ctx, non-packed) microbatch + throughput
# probe on B200.
#
# Purpose: device_train_microbatch_size=288 in the YAML was tuned on H200 (2x, torch2.4/
# cu124) and never validated on B200 (unlike phase-1/2, which went through this same
# smoke pattern first). This probe runs the REAL phase-0 config with
# device_train_microbatch_size=auto so Composer binary-searches the largest microbatch
# that fits on the B200 (180GB HBM3e), and reports throughput over a few dozen steps.
#
# After this finishes, read the chosen microbatch from the log
#   ("Setting device_train_microbatch_size to N") and set it in
#   yamls/main/base_hebrew/flex-bert-rope-phase-0-pretrain.yaml (back off one notch if
#   the real run uses compile=true; the probe runs compile off for a fast, reliable
#   auto search, and compile can shift the memory profile slightly).
#
# Run via slurm: sbatch .slurm/jobs/smoke_b200_phase-0_1gpu.slurm
#            or: sbatch .slurm/jobs/smoke_b200_phase-0.slurm (4 GPU, real throughput number)

echo "Activating Conda environment: bert-b200"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bert-b200   # Blackwell env: torch 2.7/cu128 + flash-attn 2.7.4.post1 (sm_100)
cd /home/nlp/achimoa/workspace/HebrewModernBERT

export WANDB_MODE=offline
# Fresh per-job triton cache: the shared ~/.triton cross-contaminates across triton
# versions and poisons flash-attn's rotary kernel with stale cubins -> PY_SSIZE_T_CLEAN
# at _init_handles. A clean per-job cache compiles fresh and avoids it.
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/triton-cache-${SLURM_JOB_ID:-$$}"
rm -rf "$TRITON_CACHE_DIR"; mkdir -p "$TRITON_CACHE_DIR"

# MB=auto + COMPILE=false by default (microbatch probe). Override via sbatch --export
# to run the compile=true de-risk: --export=ALL,COMPILE=true,MB=288
python -m composer main.py yamls/main/base_hebrew/flex-bert-rope-phase-0-pretrain.yaml \
    run_name=smoke-phase-0-b200 \
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
    log_to_console=true \
    console_log_interval=10ba
echo "Done. Read the auto-chosen device_train_microbatch_size from the log above,"
echo "and note time/token throughput (real-token rate at 1024 context, non-packed)."
