#!/bin/bash
#SBATCH --job-name=hmb-decomp
#SBATCH --output=.slurm/logs/hmb-decomp_%j.out
#SBATCH --error=.slurm/logs/hmb-decomp_%j.err
#SBATCH --partition=p_b200_goldberg
#SBATCH --account=ug_goldberg
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120GB
#SBATCH --time=03:00:00
#SBATCH --mail-type=FAIL

# Rewrite the zstd-packed MDS as raw .mds so NoStreamingDataset can read it.
# I/O-only (~30-60 min), comfortably inside the 4h goldberg wall.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bert-b200
cd /home/nlp/achimoa/workspace/HebrewModernBERT
python -m src.data_prep.decompress_mds \
    --src data/hebrewmodernbert/hebrew_packed_8192 \
    --out data/hebrewmodernbert/hebrew_packed_8192_raw
echo "Done."
