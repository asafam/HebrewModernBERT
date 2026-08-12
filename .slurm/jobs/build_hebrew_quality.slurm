#!/bin/bash
#SBATCH --job-name=build-heb-quality
#SBATCH --output=.slurm/logs/build-heb-quality_%j.out
#SBATCH --error=.slurm/logs/build-heb-quality_%j.err
#SBATCH --partition=p_glickman
#SBATCH --account=ug_cs_dsi
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00

# Build the quality-re-weighted Hebrew corpus (MAIN, phase-0 + phase-1): MAFAT + FineWeb-2
# net-new (~9.4B tok, deduped vs MAFAT -- see scripts/dedup_fineweb_vs_mafat.py). Rebuilds
# (and replaces via shutil.rmtree) the earlier MAFAT-only validation build at $OUT.
# CPU data-prep on p_glickman (the approved data-prep exception). Uses the tokens_target
# patch (up/down-sampling) in build_datasets.py. Output is per-DOCUMENT text MDS -> phase-0
# tokenizes on-the-fly at 1024; phase-1/2 pack on-the-fly with sequence_packing:true (which
# gives correct per-doc cu_seqlens = the non-damaging ctx-ext fix).
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bert24
cd /home/nlp/achimoa/workspace/hebrew_text_retrieval
export PYTHONPATH="$PYTHONPATH:$(pwd)/src"

CONFIG=config/data/train/datasets_hebrew_quality/datasets_train_corpus.yaml
OUT=/home/nlp/achimoa/workspace/HebrewModernBERT/data/hebrewmodernbert/hebrew_quality

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
