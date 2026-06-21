#!/bin/bash -i
# Phase 1: context extension 1024 -> 8192 (loads phase-0.2 WEIGHTS, fresh schedule).
# Runs on the Blackwell env (bert-b200): torch 2.7/cu128 + flash-attn 2.7.4.post1 (sm_100)
# + triton 3.3.1 + composer 0.31. Smoke-validated: microbatch 144/GPU, compile=true OK.

echo "Activating Conda environment: bert-b200"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bert-b200
cd /home/nlp/achimoa/workspace/HebrewModernBERT

# CRITICAL on Blackwell: a per-job triton cache. The shared ~/.triton cross-contaminates
# across triton versions and poisons flash-attn's rotary kernel -> PY_SSIZE_T_CLEAN at
# _init_handles. A fresh cache per job compiles clean.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # eliminate allocator fragmentation (16GB logits alloc failed mid-run otherwise)
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/triton-cache-${SLURM_JOB_ID:-$$}"
rm -rf "$TRITON_CACHE_DIR"; mkdir -p "$TRITON_CACHE_DIR"

echo "Starting composer on main.py: yamls/main/base_hebrew/flex-bert-rope-phase-1-contextextension.yaml"
python -m composer main.py yamls/main/base_hebrew/flex-bert-rope-phase-1-contextextension.yaml
