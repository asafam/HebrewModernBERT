#!/bin/bash -i

# Convert a HebrewModernBERT Composer checkpoint to HuggingFace format.
# Defaults to the FINAL model (phase-2, 8192 ctx). Override any of the env vars
# below to convert a different phase/checkpoint, e.g.:
#   YAML=yamls/main/base_hebrew/flex-bert-rope-phase-0.2-pretrain.yaml \
#   CKPT=checkpoints/base/modern-bert-base-phase-0.2-pretrain/ckpt/latest-rank0.pt \
#   OUTPUT_NAME=HebrewModernBERT-base-phase-0.2 MAX_LENGTH=1024 \
#   bash scripts/convert_to_hf_hebmodernbert.sh

# Activate the bert-b200 conda environment (newer transformers that knows modernbert + sm_100)
# Activate bert-b200 only if we are not ALREADY inside it. When this script is called from a slurm
# job that already ran `conda activate bert-b200`, `conda info --base` can resolve wrongly and the
# activation fails noisily (`/usr/etc/profile.d/conda.sh: No such file`) while the job still succeeds
# by inheriting the parent env -- a confusing near-miss that `set -e` in the caller does NOT catch.
if [ "${CONDA_DEFAULT_ENV:-}" = "bert-b200" ]; then
    echo "Already in bert-b200; skipping activation."
else
    echo "Activating Conda environment: bert-b200"
    source "$(conda info --base)/etc/profile.d/conda.sh" || \
      source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate bert-b200 || { echo "ERROR: could not activate bert-b200"; exit 1; }
    echo "Conda environment activated."
fi

# --- configurable inputs (defaults = phase-2 final) ---
YAML="${YAML:-yamls/main/base_hebrew/flex-bert-rope-phase-2-contextextension.yaml}"
CKPT="${CKPT:-checkpoints/base/modern-bert-base-phase-2-contextextension/ckpt/latest-rank0.pt}"
OUTPUT_NAME="${OUTPUT_NAME:-HebrewModernBERT-base-final}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/hf}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
VOCAB_SIZE="${VOCAB_SIZE:-150016}"

# Token IDs are fixed by the 150K tokenizer (cls/bos=2, sep/eos=3, pad=0, mask=4).
echo "Starting conversion to Hugging Face format..."
echo "  yaml=$YAML"
echo "  ckpt=$CKPT"
echo "  out=$OUTPUT_DIR/$OUTPUT_NAME  (max_length=$MAX_LENGTH, vocab=$VOCAB_SIZE)"
python ./src/convert_to_hf.py \
    --yaml-config "$YAML" \
    --output-name "$OUTPUT_NAME" \
    --output-dir "$OUTPUT_DIR" \
    --input-checkpoint "$CKPT" \
    --bos-token-id 2 \
    --eos-token-id 3 \
    --cls-token-id 2 \
    --sep-token-id 3 \
    --pad-token-id 0 \
    --mask-token-id 4 \
    --max-length "$MAX_LENGTH" \
    --vocab-size "$VOCAB_SIZE"

echo "Done. Verify config.json has local_rope_theta=10000, global_rope_theta=160000, mask_token_id=4."
