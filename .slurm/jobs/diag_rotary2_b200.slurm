#!/bin/bash
#SBATCH --job-name=hmb-diag-rot2
#SBATCH --output=.slurm/logs/hmb-diag-rot2_%j.out
#SBATCH --error=.slurm/logs/hmb-diag-rot2_%j.err
#SBATCH --partition=p_b200_goldberg
#SBATCH --account=ug_goldberg
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=00:20:00
#SBATCH --mail-type=FAIL

# FAITHFUL repro of FlexBERT's rotary call (contiguous qkv -> qk=[:,:2].view(total,2*nh,hd),
# inplace) at the REAL total_nnz = microbatch*8192. Bisect microbatch to find where the
# triton rotary kernel faults -> tests the int32-overflow-at-large-total_nnz hypothesis.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bert-b200
cd /home/nlp/achimoa/workspace/HebrewModernBERT
export CUDA_LAUNCH_BLOCKING=1

python - <<'PY'
import torch, triton, os
os.environ["TRITON_CACHE_DIR"] = f"/tmp/trc-{os.getpid()}"
from flash_attn.ops.triton.rotary import apply_rotary
print(f"triton {triton.__version__} | {torch.cuda.get_device_name(0)}", flush=True)
def test(mb, sl=8192, nh=12, hd=64):
    total = mb * sl
    elems = total * 2 * nh * hd
    qkv = torch.randn(total, 3, nh, hd, device="cuda", dtype=torch.bfloat16)
    cos = torch.randn(sl, hd // 2, device="cuda", dtype=torch.bfloat16)
    sin = torch.randn(sl, hd // 2, device="cuda", dtype=torch.bfloat16)
    cu = torch.arange(0, mb + 1, device="cuda", dtype=torch.int32) * sl
    try:
        qk = qkv[:, :2].view(total, -1, hd)   # exact FlexBERT contiguous path
        apply_rotary(qk, cos, sin, cu_seqlens=cu, max_seqlen=sl, inplace=True)
        torch.cuda.synchronize()
        print(f"  mb={mb:3d} total_nnz={total:>9,} qk_elems={elems/1e9:.2f}B (int32max=2.15B): OK", flush=True)
        return True
    except Exception as e:
        print(f"  mb={mb:3d} total_nnz={total:>9,} qk_elems={elems/1e9:.2f}B: FAIL {str(e)[:70]}", flush=True)
        return False
for mb in (8, 16, 32, 48, 64, 96, 144):
    test(mb)
PY
echo "DONE"
