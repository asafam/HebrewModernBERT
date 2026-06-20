#!/bin/bash -i
# Phase 2: final anneal at 8192 context (one_minus_sqrt decay over 50B tokens).
# Loads phase-1 WEIGHTS (load_weights_only, fresh schedule/clock). Blackwell env.

echo "Activating Conda environment: bert-b200"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bert-b200
cd /home/nlp/achimoa/workspace/HebrewModernBERT

# Fresh per-job triton cache (avoids flash-attn rotary PY_SSIZE_T_CLEAN on Blackwell)
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/triton-cache-${SLURM_JOB_ID:-$$}"
rm -rf "$TRITON_CACHE_DIR"; mkdir -p "$TRITON_CACHE_DIR"

echo "Starting composer on main.py: yamls/main/base_hebrew/flex-bert-rope-phase-2-contextextension.yaml"
python -m composer main.py yamls/main/base_hebrew/flex-bert-rope-phase-2-contextextension.yaml
