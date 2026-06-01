#!/bin/bash -i

# Activate the bert25 conda environment
echo "Activating Conda environment: bert25"
source "$(conda info --base)/etc/profile.d/conda.sh"  # Ensure Conda is properly initialized
conda activate bert25
echo "Conda environment activated."

# Run the conversion script with the specified parameters
echo "Starting conversion to Hugging Face format..."
python ./src/convert_to_hf.py \
    --yaml-config yamls/main/base/flex-bert-rope-phase-0.2-pretrain.yaml \
    --output-name HebrewModernBERT_base_mixed_h50e75c25_1024_0.2 \
    --output-dir ./outputs/hf \
    --input-checkpoint checkpoints/mixed_h50e75c25/modern-bert-base-phase-0.2-pretrain/ckpt/latest-rank0.pt \
    --bos-token-id 2 \
    --eos-token-id 3 \
    --cls-token-id 2 \
    --sep-token-id 3 \
    --pad-token-id 100001 \
    --mask-token-id 100001 \
    --max-length 1024 \
    --vocab-size 100032
