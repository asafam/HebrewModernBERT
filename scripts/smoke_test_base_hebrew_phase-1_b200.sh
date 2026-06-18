#!/bin/bash -i
# Phase-1 (context extension, 1024 -> 8192) microbatch + throughput probe on B200.
#
# Purpose: phase-1 runs at max_seq_len=8192, where the per-GPU memory cost is ~8x the
# 1024-context phases. device_train_microbatch_size in the YAML is a conservative
# placeholder (4). This probe runs the REAL phase-1 config with
# device_train_microbatch_size=auto so Composer binary-searches the largest microbatch
# that fits on the B200 (180GB HBM3e), and reports throughput over a few dozen steps.
#
# After this finishes, read the chosen microbatch from the log
#   ("Setting device_train_microbatch_size to N") and set it in
#   yamls/main/base_hebrew/flex-bert-rope-phase-1-contextextension.yaml.
#   Back off ONE notch for the real run if it used compile (the probe runs compile off
#   for a fast, reliable auto search; compile can shift the memory profile slightly).
#
# Run via slurm: sbatch .slurm/jobs/smoke_b200_phase-1.slurm

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

python -m composer main.py yamls/main/base_hebrew/flex-bert-rope-phase-1-contextextension.yaml \
    run_name=smoke-phase-1-b200 \
    device_train_microbatch_size=auto \
    max_duration=60ba \
    scheduler.t_warmup=0ba \
    eval_subset_num_batches=2 \
    eval_interval=1000ba \
    save_interval=1000ba \
    save_num_checkpoints_to_keep=1 \
    autoresume=false \
    save_overwrite=true \
    model.model_config.compile_model=false \
    log_to_console=true \
    console_log_interval=10ba
echo "Done. Read the auto-chosen device_train_microbatch_size from the log above,"
echo "and note time/token throughput (real-token rate at 8192 context on 4xB200)."
