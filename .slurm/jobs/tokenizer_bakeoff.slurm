#!/bin/bash
#SBATCH --job-name=tok-bakeoff
#SBATCH --output=.slurm/logs/tok-bakeoff_%j.out
#SBATCH --error=.slurm/logs/tok-bakeoff_%j.err
#SBATCH --partition=cpu1T-24h
#SBATCH --cpus-per-task=32
#SBATCH --mem=400GB
#SBATCH --time=24:00:00
#SBATCH --mail-user=achimoa@biu.ac.il
#SBATCH --mail-type=END,FAIL

# Train the 150K tokenizer candidates (CPU; SentencePiece on ~3M docs). No GPU needed.
bash scripts/train_tokenizer_bakeoff.sh
