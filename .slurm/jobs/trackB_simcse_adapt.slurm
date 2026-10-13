#!/bin/bash
#SBATCH --job-name=hmb-trackB
#SBATCH --output=.slurm/logs/hmb-trackB_%j.out
#SBATCH --error=.slurm/logs/hmb-trackB_%j.err
#SBATCH --partition=p_b200_nlp
#SBATCH --account=ug_goldberg
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240GB
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#
# TRACK B — contrastive (SimCSE) adaptation, then retrieval SFT, then eval.
#
# Runs on p_b200_nlp so it is CONCURRENT with Track A's SFT arms on p_b200_goldberg: the two are
# separate QOS pools (nlp GrpTRES=4 GPU, goldberg GrpTRES=1 GPU), so neither blocks the other.
# 1 GPU only, to leave the rest of the nlp entitlement free.
#
# WHY THIS AND NOT MORE SFT TUNING. HMB-large beats NeoDictaBERT on MLM (49.0 vs 46.9) but gets
# 0.187 avg NDCG@10 vs their 0.330, and recall@100 on arguana is 0.296 vs their 0.974 -- the right
# document is not in our top 100 for 70% of queries. Re-ranking, ranking losses and hard negatives
# all operate on a candidate set that does not contain the answer, so they cannot fix this. The
# embedding space itself is collapsed (anisotropy 0.9997 vs NDB 0.833). MLM trains a token
# predictor, not a metric space; InfoNCE's uniformity term is what spreads a collapsed space.
#
# FIRST RUN IS DELIBERATELY SMALL: ~1M pairs, SimCSE-scale, a few GPU-hours. Measure, then scale
# to 10-20B tokens only on signal. Do not start with the big run.
#
# Usage:
#   sbatch .slurm/jobs/trackB_simcse_adapt.slurm                      # phase-2 weights (default)
#   sbatch --export=ALL,SRC=HebrewModernBERT-large-p0-ba222000 ...    # pre-anneal weights
set -eo pipefail
mkdir -p .slurm/logs

export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"   # conda's cuda hook derefs this unset
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bert-b200
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

HMB=/home/nlp/achimoa/workspace/HebrewModernBERT
SIB=/home/nlp/achimoa/workspace/hebrew_text_retrieval
SRC="${SRC:-HebrewModernBERT-large-p2-ba3180}"
N="${N:-1000000}"
BS="${BS:-256}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-3e-5}"
TAG="${TAG:-simcse-$(echo $SRC | sed 's/HebrewModernBERT-large-//')}"

PAIRS=$HMB/data/simcse/hebrew_simcse_${N}.jsonl
VARIANT=$HMB/outputs/hf/${SRC}-dropout

# ---- step 1: sample SimCSE pairs (CPU) -------------------------------------------------------
cd $HMB
if [ -s "$PAIRS" ]; then
    echo "=== step 1/4: pairs already exist ($(wc -l < "$PAIRS") lines), skipping sampler ==="
else
    echo "=== step 1/4: sampling $N SimCSE pairs from the Hebrew corpus ==="
    python scripts/sample_hebrew_simcse_pairs.py --n "$N" --out "$PAIRS"
fi

# ---- step 2: model variant WITH DROPOUT ------------------------------------------------------
# SimCSE's positive pair is one text encoded twice; dropout is the ONLY thing that makes the two
# views differ. Our exported config ships every dropout at 0.0, which would make the positive an
# exact match and leave only the uniformity term training. Enable it. Weights are symlinked, so
# this costs no disk.
echo "=== step 2/4: building dropout variant of $SRC ==="
mkdir -p "$VARIANT"
for f in model.safetensors pytorch_model.bin special_tokens_map.json spm.model tokenizer_config.json tokenizer.json; do
    [ -e "$HMB/outputs/hf/$SRC/$f" ] && ln -sf "$(realpath $HMB/outputs/hf/$SRC/$f)" "$VARIANT/$f"
done
python - <<PY
import json
d = json.load(open("$HMB/outputs/hf/$SRC/config.json"))
for k in ("attention_dropout", "mlp_dropout", "embedding_dropout"):
    d[k] = 0.1
json.dump(d, open("$VARIANT/config.json", "w"), indent=2)
print("  dropout -> 0.1 on attention/mlp/embedding (was 0.0; SimCSE needs it for the second view)")
PY

#
# OOM FIX (job 34974417 died here at bs 256 / len 512). SimCSE pairs are SYMMETRIC: both sides are
# the SAME full-length document, whereas the BeIR SFT pairs a SHORT query with one long document.
# So at equal batch size this step costs roughly twice the activation memory, and bs 256 that A1
# ran fine for BeIR does not fit here. The fix keeps bs 256 -- the negative pool is the whole point
# of the run -- and buys the memory back from length and recomputation instead:
#   ADAPT_LEN 256 (our --min-chars 200 filter is ~60-70 Hebrew tokens, so most docs fit) and
#   --gradient_checkpointing. Adaptation at 256 then SFT at 512 is a mild length shift, and
#   ModernBERT handles variable length natively via RoPE.
# If it still OOMs, halve BS before lowering ADAPT_LEN further.
# ---- step 3: contrastive adaptation ----------------------------------------------------------
cd $SIB
export PYTHONPATH="$PYTHONPATH:$(pwd)/src"
source "$(pwd)/scripts/model/eval/corpus_paths.sh"
ADAPT=outputs/models/dual_encoder/mean_pooling/hebrew_simcse/$TAG
echo "=== step 3/4: contrastive adaptation ($N pairs, bs $BS, $EPOCHS ep, lr $LR) ==="
python src/model/dual_encoder/train_dual_encoder.py \
    --dataset_name hebrew_simcse \
    --pairs_file "$PAIRS" \
    --query_model_name "$VARIANT" \
    --doc_model_name "$VARIANT" \
    --query_field query --document_field document \
    --max_length "${ADAPT_LEN:-256}" \
    --gradient_checkpointing \
    --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size "$BS" \
    --gradient_accumulation_steps 1 \
    --learning_rate "$LR" \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --pooling mean \
    --temperature 0.05 \
    --seed 17 \
    --bf16 \
    --eval_steps 500 --save_steps 1000 --save_total_limit 1 \
    --output_dir "$ADAPT" \
    --force

# ---- step 4: retrieval SFT on the adapted weights, then eval ---------------------------------
# init_from_checkpoint continues from a fine-tuned dual encoder rather than rebuilding fresh
# encoders from a base model -- exactly the handoff we need here.
LABEL=lg-B1-simcse-$TAG
SFT=outputs/models/dual_encoder/mean_pooling/beir_hebrew/hebmodernbert/$LABEL
echo "=== step 4/4: retrieval SFT from the adapted checkpoint, then BeIR eval ==="
python src/model/dual_encoder/train_dual_encoder.py \
    --dataset_name beir_hebrew \
    --beir_corpora_root "outputs/translation/runs/full_corpus_zeroshot_nocontext_gemini31flashlite_promptv20260801/corpus" \
    --query_model_name "$VARIANT" --doc_model_name "$VARIANT" \
    --init_from_checkpoint "$ADAPT/model" \
    --query_field query --document_field document \
    --max_length 512 --num_train_epochs 10 \
    --per_device_train_batch_size 256 --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 --warmup_ratio 0.05 --lr_scheduler_type cosine \
    --pooling mean --temperature 0.05 --seed 17 --bf16 \
    --output_dir "$SFT" --force

while IFS= read -r CORPUS_DIR; do
    python src/model/eval/eval_beir_retrieval_zeroshot.py \
        --model_name_or_path "$SFT/model" --model_label "$LABEL" \
        --corpus_dir "$CORPUS_DIR" --batch_size 256 --max_length 512 --pooling mean
done < <(printf '%s\n' "${BEIR_CORPUS_DIRS[@]}")

python src/model/eval/collect_beir_results.py --results_dir outputs/eval/beir_zeroshot
echo "DONE: $LABEL   (compare vs A3 mean-pooling baseline 0.1866; NDB 0.3300)"
