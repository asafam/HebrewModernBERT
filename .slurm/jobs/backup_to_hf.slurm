#!/bin/bash
#SBATCH --job-name=hmb-hf-backup
#SBATCH --output=.slurm/logs/hmb-hf-backup_%j.out
#SBATCH --error=.slurm/logs/hmb-hf-backup_%j.err
#SBATCH --partition=p_b200_goldberg
#SBATCH --account=ug_goldberg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#
# Back up HebrewModernBERT to the HF Hub. CPU-ONLY -- no --gres, because this is a network
# transfer. It runs as a slurm job rather than on the login node because multi-GB uploads must
# not run on the shared login server.
#
# WHAT GOES UP (explicit allowlist -- never a recursive sweep):
#   1. HebrewModernBERT-large   final model + tokenizer   (our model, latest)
#   2. HebrewModernBERT-base    final model + tokenizer   (our model, latest)
#   3. a REPRODUCIBILITY BUNDLE: the scripts+configs that download, dedup, filter and build the
#      corpus, so the data can be rebuilt from raw MAFAT without shipping any MAFAT text.
#
# WHAT DOES NOT GO UP, deliberately:
#   * MAFAT corpus text, or anything derived containing text (data/, *.jsonl, *.mds, *.txt).
#     The recipe is reproducible; the data is private. Raw MAFAT is supplied separately.
#   * NeoDictaBERT. It is dicta-il's model, already public on the Hub -- not ours to
#     redistribute, and a copy would go stale against theirs.
#   * Composer checkpoints (6GB each, 1.3TB total). The HF export is what anyone needs; the
#     Composer ckpt is only for resuming training. Keep those on local/other storage.
#   * logs/ and .env. The sibling repo has leaked HF tokens into logs/slurm/ before (54 files,
#     per its own source comments). Logs are never part of a backup.
#
# REPOS ARE CREATED PRIVATE. Flip to public deliberately, afterwards, once you have looked at
# what is in them. Never the reverse.
#
# Usage:
#   sbatch --export=ALL,HF_ORG=<org> .slurm/jobs/backup_to_hf.slurm            # dry run: lists only
#   sbatch --export=ALL,HF_ORG=<org>,DO_UPLOAD=1 .slurm/jobs/backup_to_hf.slurm # actually upload
set -eo pipefail
mkdir -p .slurm/logs

# `set -u` must NOT be active across `conda activate`: the env's own CUDA hook
# (etc/conda/activate.d/~cuda-nvcc_activate.sh:43) dereferences NVCC_PREPEND_FLAGS while it is
# unset, so -u kills the job at activation before it does any work. Pre-seed the variable AND
# leave -u off (the sibling repo's mlm_compare_vs_ndb.slurm pre-seeds it for the same reason).
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bert-b200           # has huggingface_hub
cd /home/nlp/achimoa/workspace/HebrewModernBERT

HF_ORG="${HF_ORG:-biu-nlp}"
DO_UPLOAD="${DO_UPLOAD:-0}"
SIB=/home/nlp/achimoa/workspace/hebrew_text_retrieval

# Token: huggingface-cli reads ~/.cache/huggingface/token automatically. Assert it exists rather
# than discovering a 401 half way through a multi-GB upload. NEVER echo the value -- this job's
# stdout lands in .slurm/logs/, and that is exactly how 54 plaintext tokens accumulated in the
# sibling repo's logs. Presence only.
TOKFILE="$HOME/.cache/huggingface/token"
[ -s "$TOKFILE" ] || { echo "ERROR: no HF token at $TOKFILE"; exit 1; }
echo "HF token: present"

# TWO MODELS, TWO REPOS. "HebrewModernBERT" was requested as the name, but base and large are
# different models with different results (MLM 44.0 vs 49.0) -- one repo cannot serve both
# without an unlabelled collision, and anyone doing from_pretrained() needs to know which they
# got. So: HebrewModernBERT-large and HebrewModernBERT-base, which also matches how
# dicta-il/NeoDictaBERT and answerdotai/ModernBERT-{base,large} are published.
REPO_LARGE="${REPO_LARGE:-$HF_ORG/HebrewModernBERT-large}"
REPO_BASE="${REPO_BASE:-$HF_ORG/HebrewModernBERT-base}"

LARGE=outputs/hf/HebrewModernBERT-large-p2-ba3180
BASE=outputs/hf/HebrewModernBERT-base-retrain-final
TOK=tokenizer/v4_bpe_150k

for p in "$LARGE" "$BASE" "$TOK"; do
    [ -e "$p" ] || { echo "ERROR: missing $p"; exit 1; }
done

# ---------------------------------------------------------------------------------------------
# Reproducibility bundle: the code+config needed to rebuild the corpus from raw MAFAT.
# Staged in a temp dir so the exact file list is auditable before anything is uploaded.
# ---------------------------------------------------------------------------------------------
STAGE="${SLURM_TMPDIR:-/tmp}/hmb-recipe-${SLURM_JOB_ID:-$$}"
rm -rf "$STAGE"; mkdir -p "$STAGE"/{data_prep,mix_configs,slurm,docs}

# corpus build + dedup + acquisition pipeline (this repo)
cp -r src/data_prep/.            "$STAGE/data_prep/"            2>/dev/null || true
cp scripts/acquire_fineweb2_hebrew.py "$STAGE/data_prep/"        2>/dev/null || true
cp scripts/dedup_fineweb_vs_mafat.py  "$STAGE/data_prep/"        2>/dev/null || true
cp scripts/train_tokenizer_bakeoff.sh "$STAGE/data_prep/"        2>/dev/null || true
cp .slurm/jobs/build_hebrew_quality.slurm \
   .slurm/jobs/build_hebrew_quality_anneal.slurm \
   .slurm/jobs/filter_fineweb2_dedup.slurm \
   .slurm/jobs/dedup_fineweb_vs_mafat.slurm "$STAGE/slurm/"      2>/dev/null || true

# THE MIX CONFIGS LIVE IN THE SIBLING REPO. This is the reproducibility gap: the build jobs are
# here, but the definition of WHAT they build (per-source token targets and epoch multipliers --
# hec4_clean 34.5B, fineweb2_hebrew 30.0B, yifat_long 28.4B, ... = 131.1B) is over there.
# Without these the corpus cannot be rebuilt, so they must travel with the model.
cp -r "$SIB"/config/data/train/datasets_hebrew_quality \
      "$SIB"/config/data/train/datasets_hebrew_quality_anneal "$STAGE/mix_configs/" 2>/dev/null || true

# the training recipe + findings
cp docs/TRAINING.md docs/LARGE_TRAINING.md docs/RETRIEVAL_EVAL.md "$STAGE/docs/" 2>/dev/null || true
cp -r yamls/main/base_hebrew yamls/main/base_hebrew_large "$STAGE/" 2>/dev/null || true

# ---------------------------------------------------------------------------------------------
# PRIVACY GUARD. Refuse to upload if the bundle contains corpus text or a credential.
# A config naming a file is fine; a file CONTAINING Hebrew prose is not.
# ---------------------------------------------------------------------------------------------
echo "=== privacy audit of the reproducibility bundle ==="
BAD=0
while IFS= read -r f; do
    case "$f" in *.jsonl|*.mds|*.txt|*.arrow|*.parquet) echo "  REFUSE (data file): $f"; BAD=1; continue;; esac
    if [ "$(grep -acE '[֐-׿]{40,}' "$f" 2>/dev/null || echo 0)" -gt 0 ]; then
        echo "  REFUSE (contains Hebrew prose): $f"; BAD=1
    fi
    if grep -aqE 'hf_[A-Za-z0-9]{30,}|HF_TOKEN=[^$]' "$f" 2>/dev/null; then
        echo "  REFUSE (possible credential): $f"; BAD=1
    fi
done < <(find "$STAGE" -type f)
[ "$BAD" = "0" ] || { echo "ABORT: bundle failed the privacy audit (see above)."; exit 1; }
echo "  clean: no corpus text, no credentials"
echo
echo "=== bundle contents ($(find "$STAGE" -type f | wc -l) files, $(du -sh "$STAGE" | cut -f1)) ==="
find "$STAGE" -type f | sed "s|$STAGE|  recipe|" | sort

echo
echo "=== models to upload ==="
du -sh "$LARGE" "$BASE" "$TOK" | sed 's/^/  /'

if [ "$DO_UPLOAD" != "1" ]; then
    echo
    echo "DRY RUN. Nothing uploaded. Re-run with DO_UPLOAD=1 to perform the upload."
    exit 0
fi

upload () {  # upload <repo> <local_dir> <path_in_repo>
    echo "--- $1  <=  $2"
    huggingface-cli upload --quiet --repo-type model --private "$1" "$2" "$3"
}

echo
echo "=== uploading (private repos) ==="
upload "$REPO_LARGE" "$LARGE" .
upload "$REPO_LARGE" "$TOK"   .
upload "$REPO_LARGE" "$STAGE" recipe
[ -f docs/MODEL_CARD_large.md ] && \
  huggingface-cli upload --quiet --repo-type model --private \
    "$REPO_LARGE" docs/MODEL_CARD_large.md README.md

upload "$REPO_BASE" "$BASE" .
upload "$REPO_BASE" "$TOK"  .
upload "$REPO_BASE" "$STAGE" recipe

echo
echo "DONE. Both repos are PRIVATE. Review them, then make public deliberately."
echo "  https://huggingface.co/$REPO_LARGE"
echo "  https://huggingface.co/$REPO_BASE"
