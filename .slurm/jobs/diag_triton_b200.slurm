#!/bin/bash
#SBATCH --job-name=hmb-diag-triton
#SBATCH --output=.slurm/logs/hmb-diag-triton_%j.out
#SBATCH --error=.slurm/logs/hmb-diag-triton_%j.err
#SBATCH --partition=p_b200_goldberg
#SBATCH --account=ug_goldberg
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=00:15:00        # tiny diagnostic -> backfills fast
#SBATCH --mail-type=FAIL

# Decide the triton-on-Blackwell fork WITHOUT the training stack:
# run a stock triton vector-add kernel (the exact JIT -> _init_handles path that
# crashed with PY_SSIZE_T_CLEAN) on triton 3.3.1, then on 3.4.0, each with a clean
# ~/.triton cache. Result tells us: triton-version lever vs flash-attn-specific.

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bert-b200
cd /home/nlp/achimoa/workspace/HebrewModernBERT

cat > /tmp/triton_min.py <<'PY'
import torch, triton, triton.language as tl
@triton.jit
def add_kernel(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0); offs = pid*BLOCK + tl.arange(0, BLOCK); m = offs < n
    tl.store(o_ptr+offs, tl.load(x_ptr+offs, mask=m) + tl.load(y_ptr+offs, mask=m), mask=m)
x = torch.randn(1024, device="cuda"); y = torch.randn(1024, device="cuda"); o = torch.empty_like(x)
try:
    add_kernel[(1,)](x, y, o, 1024, BLOCK=1024); torch.cuda.synchronize()
    print(f"RESULT triton {triton.__version__}: KERNEL OK (allclose={torch.allclose(o, x+y)})")
except Exception as e:
    print(f"RESULT triton {triton.__version__}: KERNEL FAILED -> {type(e).__name__}: {e}")
PY

echo "############ device ############"; nvidia-smi --query-gpu=name --format=csv,noheader

echo "############ TEST A: triton 3.3.1 (current), clean cache ############"
rm -rf ~/.triton
python /tmp/triton_min.py

echo "############ upgrade to triton 3.4.0 ############"
pip install "triton==3.4.0" -q 2>&1 | tail -2

echo "############ TEST B: triton 3.4.0, clean cache ############"
rm -rf ~/.triton
python /tmp/triton_min.py

echo "############ restore triton 3.3.1 (leave env as-was until we decide) ############"
pip install "triton==3.3.1" -q 2>&1 | tail -1
echo "DONE"
