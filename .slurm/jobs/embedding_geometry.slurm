#!/bin/bash
#SBATCH --job-name=emb_geom
#SBATCH --output=.slurm/logs/emb_geom_%j.out
#SBATCH --error=.slurm/logs/emb_geom_%j.err
#SBATCH --partition=p_b200_goldberg
#SBATCH --account=ug_goldberg
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#
# Per-layer embedding-geometry diagnostic: HMB phases vs NeoDictaBERT.
# Measures where MLM representations fail to become retrievable vectors -- see the script docstring.
# see the docstring in scripts/embedding_geometry_vs_neodictabert.py for why that fork matters.
#
# Small job: 4 models x 2 small corpora, all layers in one forward pass.
set -e
mkdir -p .slurm/logs

# B200 is Blackwell/sm_100 -> needs the torch 2.7/cu128 env, not bert24/bert25.
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
source /home/nlp/achimoa/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-bert-b200}"
cd /home/nlp/achimoa/workspace/HebrewModernBERT

# NeoDictaBERT is already in the local HF cache; stay offline so a hub hiccup
# cannot stall the job. Drop this if a model needs fetching.
export HF_HUB_OFFLINE=1

python scripts/embedding_geometry_vs_neodictabert.py \
    --batch_size "${BS:-128}" \
    --max_length "${MAXLEN:-512}" \
    --out "${OUT:-outputs/eval/embedding_geometry.json}"
