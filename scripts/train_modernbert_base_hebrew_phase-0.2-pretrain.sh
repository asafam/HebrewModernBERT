#!/bin/bash -i
# Phase 0.2: Hebrew-only specialization (continued pretraining from phase 0.1's weights).
echo "Activating Conda environment: bert24"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bert24
cd /home/nlp/achimoa/workspace/HebrewModernBERT
echo "Starting composer on main.py: yamls/main/base_hebrew/flex-bert-rope-phase-0.2-pretrain.yaml"
python -m composer main.py yamls/main/base_hebrew/flex-bert-rope-phase-0.2-pretrain.yaml
