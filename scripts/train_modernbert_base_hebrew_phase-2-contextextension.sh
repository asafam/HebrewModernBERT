#!/bin/bash -i
# Phase 2: final anneal at 8192 context (one_minus_sqrt decay over 50B tokens).
# Loads phase-1 WEIGHTS (load_weights_only, fresh schedule/clock). Blackwell env.

echo "Activating Conda environment: bert-b200"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bert-b200
cd /home/nlp/achimoa/workspace/HebrewModernBERT

# Fresh per-job triton cache (avoids flash-attn rotary PY_SSIZE_T_CLEAN on Blackwell)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # eliminate allocator fragmentation (16GB logits alloc failed mid-run otherwise)
# Persistent compile caches across requeues (env+triton version are fixed now, so the
# PY_SSIZE_T_CLEAN cross-version contamination cannot recur) -> each 4h requeue skips the
# ~10min torch.compile/inductor rebuild and reuses cached kernels.
export TRITON_CACHE_DIR="$HOME/.cache/hmb/triton-phase2"
export TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/hmb/inductor-phase2"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

echo "Starting composer on main.py: yamls/main/base_hebrew/flex-bert-rope-phase-2-contextextension.yaml"
python -m composer main.py yamls/main/base_hebrew/flex-bert-rope-phase-2-contextextension.yaml
