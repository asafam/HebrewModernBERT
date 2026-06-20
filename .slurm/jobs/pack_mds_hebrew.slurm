#!/bin/bash
#SBATCH --job-name=hmb-pack
#SBATCH --output=.slurm/logs/hmb-pack_%j.out
#SBATCH --error=.slurm/logs/hmb-pack_%j.err
#SBATCH --partition=p_glickman
#SBATCH --account=ug_cs_dsi
#SBATCH --nodes=1
#SBATCH --gres=gpu:1            # p_glickman is a GPU partition; the gtx1080ti is unused (CPU-only job)
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH --mail-user=achimoa@biu.ac.il
#SBATCH --mail-type=END,FAIL
# No --time: p_glickman is infinite-wall -> zero risk of a 4h kill corrupting the MDS mid-write.

# Pack the short-doc Hebrew MDS into dense 8192-token sequences for context extension.
# CPU-only (HF fast tokenizer); output -> data/hebrewmodernbert/hebrew_packed_8192/{train,validation}.

echo "Activating Conda environment: bert-b200"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bert-b200
cd /home/nlp/achimoa/workspace/HebrewModernBERT

python -m src.data_prep.pack_mds \
    --src data/hebrewmodernbert/hebrew \
    --out data/hebrewmodernbert/hebrew_packed_8192 \
    --tokenizer tokenizer/v4_bpe_150k \
    --max-length 8192 \
    --train-max-tokens 30e9 \
    --val-max-tokens 1e9
echo "Done."
