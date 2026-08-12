#!/bin/bash
#SBATCH --job-name=mlm_vs_ndb
#SBATCH --output=.slurm/logs/mlm_vs_ndb_%j.out
#SBATCH --error=.slurm/logs/mlm_vs_ndb_%j.err
#SBATCH --partition=p_b200_goldberg
#SBATCH --account=ug_goldberg
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#
# Cross-tokenizer whole-word MLM comparison: every HMB training phase vs NeoDictaBERT.
# Discriminates "weak backbone" from "backbone fine, representations don't transfer" --
# see the docstring in scripts/mlm_compare_vs_neodictabert.py for why that fork matters.
#
# Small job: 7 models x 5 datasets x 150 docs. Minutes, not hours; 1h is slack.
set -e
mkdir -p .slurm/logs

# B200 is Blackwell/sm_100 -> needs the torch 2.7/cu128 env, not bert24/bert25.
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
source /home/nlp/achimoa/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-bert-b200}"
cd /home/nlp/achimoa/workspace/HebrewModernBERT

# NeoDictaBERT is already in the local HF cache; stay offline so a hub hiccup
# cannot stall the job. Drop this if a model needs fetching.
export HF_HUB_OFFLINE=1

# EXTRA/ONLY let the LARGE-run gates score a new checkpoint without editing the MODELS dict:
#   sbatch --export=ALL,EXTRA="HMB large step0=outputs/hf/HebrewModernBERT-large-step0",\
#          ONLY="NeoDictaBERT,phase-0 FINAL,large",OUT=outputs/eval/mlm_large_step0.json \
#          .slurm/jobs/mlm_compare_vs_ndb.slurm
# Unset, behaviour is exactly as before (full MODELS set, default out path).
EXTRA_ARGS=()
[ -n "${EXTRA:-}" ] && EXTRA_ARGS+=(--extra-model "$EXTRA")
[ -n "${EXTRA2:-}" ] && EXTRA_ARGS+=(--extra-model "$EXTRA2")
[ -n "${ONLY:-}" ] && EXTRA_ARGS+=(--only "$ONLY")

python scripts/mlm_compare_vs_neodictabert.py \
    --docs_per_dataset "${DOCS:-150}" \
    --masks_per_doc "${MASKS:-4}" \
    --out "${OUT:-outputs/eval/mlm_vs_neodictabert.json}" \
    "${EXTRA_ARGS[@]}"
