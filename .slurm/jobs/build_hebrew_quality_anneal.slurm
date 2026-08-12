#!/bin/bash
#SBATCH --job-name=build-heb-anneal
#SBATCH --output=.slurm/logs/build-heb-anneal_%j.out
#SBATCH --error=.slurm/logs/build-heb-anneal_%j.err
#SBATCH --partition=p_glickman
#SBATCH --account=ug_cs_dsi
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=8:00:00

# Build the quality-heavy ANNEAL corpus (phase-2 only), rebalanced 2026-08 with a
# retrieval-SFT-aware trim (see config file header + docs/RETRIEVAL_EVAL.md for the
# reasoning). Much smaller than the main hebrew_quality corpus (~12B vs ~131B tokens), so
# 8h is generous vs. the main corpus's validated 48h. CPU data-prep on p_glickman (the
# approved data-prep exception).
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bert24
cd /home/nlp/achimoa/workspace/hebrew_text_retrieval
export PYTHONPATH="$PYTHONPATH:$(pwd)/src"

CONFIG=config/data/train/datasets_hebrew_quality_anneal/datasets_train_corpus.yaml
OUT=/home/nlp/achimoa/workspace/HebrewModernBERT/data/hebrewmodernbert/hebrew_quality_anneal

for SPLIT in validation train; do    # validation first (small, fails fast if misconfigured)
  echo "############### building $SPLIT ###############"
  python src/data/datasets/build_datasets.py \
    --config_file "$CONFIG" \
    --output_path "$OUT" \
    --split "$SPLIT" \
    --format mds \
    --shard_size_limit 67108864 \
    --buffer_size 1000000
done
echo "############### DONE: $OUT ###############"
