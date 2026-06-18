#!/bin/bash -i
# Build a Blackwell (B200, sm_100) conda env for HebrewModernBERT training.
#
# Strategy: a FRESH env with a clean conda CUDA 12.8 toolkit (cloning bert24 dragged
# ~53 conda cuda-12.4 packages that conflict with 12.8). We install the training stack
# explicitly. Any training dep we miss surfaces as a one-line ImportError in the forward
# smoke -> cheap to add; a corrupt CUDA toolchain is not.
#
# Target matrix (researched, mid-2026):
#   CUDA 12.8 toolkit | torch 2.7.0/cu128 | triton 3.3 (ships w/ torch) |
#   flash-attn 2.7.4.post1 (FA2, runs on B200) | composer/mosaicml 0.31.0 | transformers >=4.48
#
# We do NOT install FlashAttention-3 (flash_attn_interface): it fails on sm_100, and
# src/bert_layers/attention.py guards that import (try/except) -> falls back to FA2.
#
# Run via: sbatch .slurm/jobs/build_b200_env.slurm  (runs on a B200 so the final
# import sanity executes on the real device).
set -e

ENV=bert-b200
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "=================================================================="
echo "[1/5] Fresh env $ENV (python 3.11) + clean CUDA 12.8 toolkit"
echo "=================================================================="
conda env remove -n $ENV -y 2>/dev/null || true
conda create -n $ENV -y python=3.11
conda install -n $ENV -y -c nvidia cuda-toolkit=12.8
set +e; conda activate $ENV; set -e   # activate hooks aren't -e/-u safe
echo "  nvcc: $(nvcc --version 2>/dev/null | grep release || echo 'NOT FOUND')"

echo "=================================================================="
echo "[2/5] torch 2.7.0 + cu128 (Blackwell kernels + triton 3.3)"
echo "=================================================================="
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128

echo "=================================================================="
echo "[3/5] Training stack: composer 0.31 + transformers>=4.48 + repo deps"
echo "=================================================================="
# mosaicml[nlp,wandb]==0.31.0 provides the `composer` module + nlp metrics + wandb;
# the rest mirror environment.yaml / requirements (streaming, omegaconf, typer, etc.).
pip install "mosaicml[nlp,wandb]==0.31.0"
pip install -U "transformers>=4.48,<5"
pip install "mosaicml-streaming" "omegaconf>=2.3" einops typer safetensors \
    datasets "ninja" torch-optimi "ruamel.yaml" zstandard tqdm

echo "=================================================================="
echo "[4/5] Build flash-attn 2.7.4.post1 (FA2) for Blackwell sm_100"
echo "      (compile from source; ~30-60 min. NOT installing FA3.)"
echo "=================================================================="
export CUDA_HOME="$CONDA_PREFIX"           # nvcc 12.8 + headers from the conda toolkit
export TORCH_CUDA_ARCH_LIST="10.0"
export MAX_JOBS=${MAX_JOBS:-32}            # cap parallel nvcc jobs (RAM)
export FLASH_ATTENTION_FORCE_BUILD=TRUE    # force source build for sm_100
echo "  CUDA_HOME=$CUDA_HOME | nvcc=$(which nvcc)"
pip install flash-attn==2.7.4.post1 --no-build-isolation

echo "=================================================================="
echo "[5/5] Sanity: versions + Blackwell arch + flash-attn import"
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
echo "DONE. If [5/5] printed OK, run the FlexBERT import+forward smoke next."
