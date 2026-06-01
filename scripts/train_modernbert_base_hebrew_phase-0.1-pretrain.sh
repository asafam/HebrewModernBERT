#!/bin/bash -i

# Activate the bert24 conda environment
echo "Activating Conda environment: bert24"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bert24

# Navigate to the workspace directory
echo "Changing directory to: /home/nlp/achimoa/workspace/HebrewModernBERT"
cd /home/nlp/achimoa/workspace/HebrewModernBERT

# Run the training script with the specified YAML configuration
echo "Starting composer on main.py with configuration: yamls/main/base_hebrew/flex-bert-rope-phase-0.1-pretrain.yaml"
python -m composer main.py yamls/main/base_hebrew/flex-bert-rope-phase-0.1-pretrain.yaml