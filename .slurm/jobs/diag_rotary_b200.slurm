#!/bin/bash
#SBATCH --job-name=hmb-diag-rot
#SBATCH --output=.slurm/logs/hmb-diag-rot_%j.out
#SBATCH --error=.slurm/logs/hmb-diag-rot_%j.err
#SBATCH --partition=p_b200_goldberg
#SBATCH --account=ug_goldberg
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=00:20:00
#SBATCH --mail-type=FAIL

# Isolate the flash-attn triton rotary kernel (the op that faults at 8192 in phase-1):
# test it at short (512) and long (8192) seqlen on triton 3.3.1, then 3.4.0.
# Decides: triton-version fix vs rotary-kernel replacement.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bert-b200
cd /home/nlp/achimoa/workspace/HebrewModernBERT
export CUDA_LAUNCH_BLOCKING=1

cat > /tmp/rot_test.py <<'PY'
import torch, triton, os
os.environ["TRITON_CACHE_DIR"] = f"/tmp/trcache-{os.getpid()}"
from flash_attn.ops.triton.rotary import apply_rotary
def test(sl, bs=4, nh=12, hd=64):
    total = sl * bs
    x = torch.randn(total, nh, hd, device="cuda", dtype=torch.bfloat16)
    cos = torch.randn(sl, hd // 2, device="cuda", dtype=torch.bfloat16)
    sin = torch.randn(sl, hd // 2, device="cuda", dtype=torch.bfloat16)
    cu = (torch.arange(0, bs + 1, device="cuda", dtype=torch.int32) * sl)
    try:
        out = apply_rotary(x, cos, sin, cu_seqlens=cu, max_seqlen=sl)
        torch.cuda.synchronize()
        print(f"  seqlen={sl}: OK {tuple(out.shape)}", flush=True)
        return True
    except Exception as e:
        print(f"  seqlen={sl}: FAIL {type(e).__name__}: {str(e)[:100]}", flush=True)
        return False
print(f"=== triton {triton.__version__} | {torch.cuda.get_device_name(0)} ===", flush=True)
for sl in (512, 2048, 4096, 8192):
    test(sl)
PY

echo "############ TEST on triton 3.3.1 (current) ############"
python /tmp/rot_test.py
echo "############ upgrade triton 3.4.0 ############"
pip install "triton==3.4.0" -q 2>&1 | tail -1
echo "############ TEST on triton 3.4.0 ############"
python /tmp/rot_test.py
echo "############ restore triton 3.3.1 ############"
pip install "triton==3.3.1" -q 2>&1 | tail -1
echo "DONE"
