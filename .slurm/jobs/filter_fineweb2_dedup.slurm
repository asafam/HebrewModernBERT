#!/bin/bash
#SBATCH --job-name=filter_fw2
#SBATCH --output=logs/data_prep/filter_fw2_%j.out
#SBATCH --error=logs/data_prep/filter_fw2_%j.err
#SBATCH --partition=cpu512G-48h
#SBATCH --account=ug_cs_dsi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=1:00:00

# Extends dedup_fineweb_vs_mafat.slurm: same corpora/scale (validated at ~34min/~320GiB RSS
# for the hashing-only run), but this time persists the MAFAT hash array and actually emits
# the net-new FineWeb-2 docs as jsonl, which the earlier run never did (stats-only).
#
# Uses --load_hashes to reuse the array already saved by the first (job 17731672) run,
# instead of re-paying the ~50min/~320GiB MAFAT hash cost -- mem/time trimmed accordingly.
# (First run also had a part-filename collision across --fineweb dirs, since fixed in
# dedup_fineweb_vs_mafat.py: chunk lo indices restart at 0 per dir, so validation's lone
# chunk silently overwrote train's first chunk. Filenames now include the source dir.)

set -e
mkdir -p logs/data_prep
cd /home/nlp/achimoa/workspace/HebrewModernBERT
source /home/nlp/achimoa/miniconda3/etc/profile.d/conda.sh
conda activate bert-b200

EMIT_DIR=/home/nlp/achimoa/workspace/hebrew_text_retrieval/data/fineweb2/hebrew
mkdir -p "$EMIT_DIR/_parts"

python3 scripts/dedup_fineweb_vs_mafat.py \
  --fineweb data/hebrewmodernbert/fineweb2_hebrew/train data/hebrewmodernbert/fineweb2_hebrew/validation \
  --workers 64 \
  --load_hashes data/hebrewmodernbert/mafat_hashes.npy \
  --emit_jsonl "$EMIT_DIR/_parts"

# Concatenate parts (ordered by the zero-padded part-{dir}-{lo:09d} names) into the single
# file the datasets_hebrew_quality mix config expects.
cat "$EMIT_DIR"/_parts/part-*.jsonl > "$EMIT_DIR/fineweb2_hebrew_dedup.jsonl"

echo "wrote $(wc -l < "$EMIT_DIR/fineweb2_hebrew_dedup.jsonl") lines -> $EMIT_DIR/fineweb2_hebrew_dedup.jsonl"
