#!/bin/bash
#SBATCH --job-name=hmb-mine
#SBATCH --output=.slurm/logs/hmb-mine_%j.out
#SBATCH --error=.slurm/logs/hmb-mine_%j.err
#SBATCH --partition=p_b200_goldberg
#SBATCH --account=ug_goldberg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --mail-type=FAIL
#
# Mine asymmetric title/headline <-> body pairs from the raw Hebrew sources. CPU-ONLY (no --gres):
# it reads ~2.6 GB of jsonl, so it must not run on the shared login node.
#
# These are the first stage of mE5's recipe. Unlike the symmetric SimCSE crop/dropout pairs, a
# title is short and query-like while a body is long and document-like -- the actual shape of the
# retrieval task, which is why they should keep producing gradient where the symmetric objective
# saturated (loss 5.17 -> 0.0021 in ~200 of 3,906 steps).
#
# Output contains MAFAT text: keep under data/ (gitignored), never publish.
set -eo pipefail
mkdir -p .slurm/logs data/simcse
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bert-b200
cd /home/nlp/achimoa/workspace/HebrewModernBERT
python scripts/mine_hebrew_weak_pairs.py --out data/simcse/hebrew_weak_pairs.jsonl
wc -l data/simcse/hebrew_weak_pairs.jsonl
