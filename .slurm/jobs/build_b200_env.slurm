#!/bin/bash
#SBATCH --job-name=hmb-build-b200
#SBATCH --output=.slurm/logs/hmb-build-b200_%j.out
#SBATCH --error=.slurm/logs/hmb-build-b200_%j.err
#SBATCH --partition=p_b200_goldberg
#SBATCH --account=ug_goldberg
#SBATCH --nodes=1
#SBATCH --gres=gpu:1           # 1 B200 so the [6/6] sanity import runs on the real device
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32     # parallel nvcc for the flash-attn compile
#SBATCH --mem=240GB
#SBATCH --time=03:30:00        # clone + flash-attn source build can take ~1h+
#SBATCH --mail-user=achimoa@biu.ac.il
#SBATCH --mail-type=END,FAIL

# Builds the bert-b200 conda env (Blackwell) and verifies flash-attn imports on a B200.
# Expect to iterate: CUDA-version mixing (cloned 12.4 cuda libs vs torch cu128) or the
# flash-attn compile are the likely failure points; the log will show exactly where.

bash scripts/build_b200_env.sh
