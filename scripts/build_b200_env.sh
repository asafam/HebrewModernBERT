#!/bin/bash -i
# Build a Blackwell (B200, sm_100) conda env for HebrewModernBERT training.
#
# Strategy: CLONE the working `bert24` env (reuse all non-core deps — streaming,
# wandb, datasets, omegaconf, typer, safetensors, einops, ...) and upgrade ONLY the
# four Blackwell-critical packages. This avoids rebuilding ~200 deps from scratch.
#
# Target matrix (researched, mid-2026):
#   CUDA 12.8 (cu128) | torch 2.7.0 | triton 3.3 (ships w/ torch) |
#   flash-attn 2.7.4.post1 (FA2 line, runs on B200) | composer 0.31.0 | transformers >=4.48
#
# We do NOT install FlashAttention-3 (flash_attn_interface): it fails on sm_100, and
# src/bert_layers/attention.py guards that import (try/except) -> falls back to FA2.
#
# Run via: sbatch .slurm/jobs/build_b200_env.slurm   (runs on a B200 so the final
# import+forward sanity check executes on the real device).
# NOTE: do NOT use `set -u` — conda's binutils activation hook references an
# unbound $ADDR2LINE and would abort the script at `conda activate`.
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"

ENV=bert-b200

echo "=================================================================="
echo "[1/6] Clone bert24 -> $ENV (skip if already cloned)"
echo "=================================================================="
if conda env list | grep -qE "/${ENV}$"; then
    echo "  $ENV already exists — reusing it (skipping clone)"
else
    conda create --clone bert24 -n $ENV -y
fi
set +e   # conda activate hooks are not -e/-u safe
conda activate $ENV
set -e

echo "=================================================================="
echo "[2/6] nvcc 12.8 toolchain (needed to compile flash-attn for sm_100)"
echo "=================================================================="
conda install -n $ENV -y -c nvidia cuda-nvcc=12.8 cuda-cudart-dev=12.8 cuda-version=12.8

echo "=================================================================="
echo "[3/6] torch 2.7.0 + cu128 (Blackwell kernels + triton 3.3)"
echo "=================================================================="
pip uninstall -y torch torchvision torchaudio flash-attn flash_attn 2>/dev/null || true
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128

echo "=================================================================="
echo "[4/6] Composer 0.31.0 + transformers >=4.48 (also unblocks HF export)"
echo "=================================================================="
pip install "mosaicml-composer==0.31.0"
pip install -U "transformers>=4.48,<5"

echo "=================================================================="
echo "[5/6] Build flash-attn 2.7.4.post1 (FA2) for Blackwell sm_100"
echo "      (compile from source; ~30-60 min. NOT installing FA3.)"
echo "=================================================================="
export TORCH_CUDA_ARCH_LIST="10.0"
export MAX_JOBS=${MAX_JOBS:-32}            # cap parallel nvcc jobs (RAM)
export FLASH_ATTENTION_FORCE_BUILD=TRUE    # force source build for sm_100
pip install flash-attn==2.7.4.post1 --no-build-isolation

echo "=================================================================="
echo "[6/6] Sanity: versions + Blackwell arch + flash-attn import"
echo "=================================================================="
python - <<'PY'
import torch
print("torch:", torch.__version__, "| cuda:", torch.version.cuda)
print("arch_list:", torch.cuda.get_arch_list())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
import flash_attn; print("flash_attn:", flash_attn.__version__)
from flash_attn import flash_attn_varlen_qkvpacked_func          # core FA2
from flash_attn.layers.rotary import RotaryEmbedding             # fused triton ops
from flash_attn.ops.triton.layer_norm import layer_norm_fn, RMSNorm
from flash_attn.ops.triton.rotary import apply_rotary
from flash_attn.losses.cross_entropy import CrossEntropyLoss
print("OK: all flash_attn submodules FlexBERT needs import cleanly")
import composer, transformers
print("composer:", composer.__version__, "| transformers:", transformers.__version__)
PY
echo "DONE. If [6/6] printed OK, run the FlexBERT import+forward smoke next."
