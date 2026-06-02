#!/bin/bash -i

# Activate the bert25 conda environment
echo "Activating Conda environment: bert25"
source "$(conda info --base)/etc/profile.d/conda.sh"  # Ensure Conda is properly initialized
conda activate bert25
echo "Conda environment activated."

# Run the conversion script with the specified parameters
echo "Starting conversion to Hugging Face format..."
python ./src/convert_to_hf.py \
    --yaml-config yamls/main/base_hebrew/flex-bert-rope-phase-0.2-pretrain.yaml \
    --output-name HebrewModernBERT_base_hebrew_1024_phase-0.2 \
    --output-dir ./outputs/hf \
    --input-checkpoint checkpoints/hebrew/modern-bert-base-phase-0.2-pretrain/ckpt/latest-rank0.pt \
    --bos-token-id 2 \
    --eos-token-id 3 \
    --cls-token-id 2 \
    --sep-token-id 3 \
    --pad-token-id 0 \
    --mask-token-id 4 \
    --max-length 1024 \
    --vocab-size 150016
