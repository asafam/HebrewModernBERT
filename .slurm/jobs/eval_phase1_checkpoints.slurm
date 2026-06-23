#!/bin/bash
#SBATCH --job-name=hmb-evalchk
#SBATCH --output=.slurm/logs/hmb-evalchk_%j.out
#SBATCH --error=.slurm/logs/hmb-evalchk_%j.err
#SBATCH --partition=p_b200_goldberg
#SBATCH --account=ug_goldberg
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120GB
#SBATCH --time=01:00:00
#SBATCH --mail-type=FAIL

# Confirmation check: eval two phase-1 checkpoints (ba3000~14.2B, ba5000~23.6B) with
# IDENTICAL eval config (subset=100, seed=17) to see if eval CE genuinely rose, or if the
# W&B trend was just the mid-run eval_subset_num_batches 100->50 change. Runs on goldberg
# (separate from the nlp training job); W&B disabled so it can't touch the training run.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bert-b200
cd /home/nlp/achimoa/workspace/HebrewModernBERT

export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/triton-evalchk-${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR"

YAML=yamls/main/base_hebrew/flex-bert-rope-phase-1-contextextension.yaml
CKDIR=checkpoints/base/modern-bert-base-phase-1-contextextension/ckpt
for BA in 3000 5000; do
  echo "############################ EVAL ba${BA} ############################"
  python scripts/eval_checkpoint.py "$YAML" \
    load_path="$CKDIR/ep0-ba${BA}-rank0.pt" \
    load_weights_only=true \
    eval_subset_num_batches=100 \
    seed=17 \
    autoresume=false \
    save_folder=null \
    loggers='{}' \
    model.model_config.compile_model=false \
    run_name="evalchk-ba${BA}"
done
echo "############################ DONE ############################"
