#!/bin/bash
#SBATCH --job-name=dedup_fw2
#SBATCH --output=logs/data_prep/dedup_fw2_%j.out
#SBATCH --error=logs/data_prep/dedup_fw2_%j.err
#SBATCH --partition=cpu512G-48h
#SBATCH --account=ug_cs_dsi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=300G
#SBATCH --time=8:00:00

set -e
mkdir -p logs/data_prep
cd /home/nlp/achimoa/workspace/HebrewModernBERT
source /home/nlp/achimoa/miniconda3/etc/profile.d/conda.sh
conda activate bert-b200

python3 scripts/dedup_fineweb_vs_mafat.py \
  --mafat data/hebrewmodernbert/hebrew/train data/hebrewmodernbert/hebrew/validation \
  --fineweb data/hebrewmodernbert/fineweb2_hebrew/train data/hebrewmodernbert/fineweb2_hebrew/validation \
  --workers 64
