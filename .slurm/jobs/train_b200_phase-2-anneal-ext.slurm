#!/bin/bash
#SBATCH --job-name=hmbbh-p2ext-b200
#SBATCH --output=.slurm/logs/hmbbh-p2ext-b200_%j.out
#SBATCH --error=.slurm/logs/hmbbh-p2ext-b200_%j.err
#SBATCH --partition=p_b200_goldberg
#SBATCH --account=ug_goldberg
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240GB
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --mail-user=achimoa@biu.ac.il
#SBATCH --mail-type=FAIL

# EXPERIMENT (2026-08-08): extended anneal -- doubled max_duration (15B -> 30B, ~2.5 epochs of
# the ~12B-token rebalanced hebrew_quality_anneal corpus) to test whether a longer curated-heavy
# anneal closes more of the gap to the 0.300 target. Fresh run from phase-1's checkpoint (NOT a
# resume of the completed 15B phase-2 run -- its one_minus_sqrt LR schedule already decayed to
# ~0 at 15B, so extending its max_duration and resuming would recompute a bad LR-spike-then-decay
# curve instead of a smooth single anneal). New run_name avoids colliding with the completed
# phase-2 checkpoint dir. Discriminator eval afterward decides if this is worth keeping.

echo "Activating Conda environment: bert-b200"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bert-b200
cd /home/nlp/achimoa/workspace/HebrewModernBERT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="$HOME/.cache/hmb/triton-phase2-anneal-ext"
export TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/hmb/inductor-phase2-anneal-ext"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

echo "Starting composer on main.py: yamls/main/base_hebrew/flex-bert-rope-phase-2-contextextension.yaml (extended anneal experiment)"
python -m composer main.py yamls/main/base_hebrew/flex-bert-rope-phase-2-contextextension.yaml \
    run_name=modern-bert-base-phase-2-anneal-ext30b \
    max_duration=30_000_000_000tok
