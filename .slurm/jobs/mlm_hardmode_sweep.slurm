#!/bin/bash
#SBATCH --job-name=mlm_hardmode
#SBATCH --output=.slurm/logs/mlm_hardmode_%j.out
#SBATCH --error=.slurm/logs/mlm_hardmode_%j.err
#SBATCH --partition=p_b200_goldberg
#SBATCH --account=ug_goldberg
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#
# IS THE HMB/NeoDictaBERT MLM PARITY A CEILING EFFECT?
#
# The single-isolated-word test put HMB retrain-final at 48.6% vs NeoDictaBERT 49.9%. But the
# two models are not the same weight class:
#     HMB  22 layers, FFN 1152, 128-tok local attn w/ global every 3rd layer, 110M non-emb body
#     NDB  28 layers, FFN 3072, FULL attention every layer,                  ~265M non-emb body
# NeoDictaBERT's body is ~2.4x larger. A 2.4x-smaller model matching it should be suspicious:
# recovering ONE masked word from full context is a largely LOCAL task (morphology, agreement,
# collocation), which a base-size model saturates. If that is what happened, the parity says
# nothing about capacity -- and the conclusion drawn from it ("backbone is not the lever, don't
# spend compute on a bigger model") is wrong.
#
# DESIGN: hold the MASK BUDGET CONSTANT at 12 words/doc and vary only CONTIGUITY.
#     span=1 x 12 targets | span=3 x 4 targets | span=6 x 2 targets
# Equal budget matters: raising span_len at fixed k would also raise the total masked fraction,
# confounding "harder because contiguous" with "harder because more is hidden". Masking a
# contiguous run is what removes the local cues -- the middle of a 6-word span cannot be
# recovered from its neighbours, only from sentence-level meaning.
#
# All configs use --rare_pct 95, which drops the top 5% most frequent words. Verified this
# removes function words (כדי, עבור, ידי, הוא, היא, באמצעות) and keeps content words. Note
# rare_pct=50 would threshold at df<=1 -- hapaxes no model can recover, a floor effect that is
# as uninformative as the ceiling it replaces. 95 is the calibrated value, not a guess.
#
# READ THE CURVE, NOT ANY SINGLE NUMBER:
#   gap WIDENS as span grows -> parity was a ceiling artifact, capacity IS the retrieval gap,
#                               and the base_hebrew_large run is the priority.
#   gap STAYS FLAT            -> capacity is not the story; the transfer hypothesis survives.
set -e
mkdir -p .slurm/logs outputs/eval

export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
source /home/nlp/achimoa/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-bert-b200}"
cd /home/nlp/achimoa/workspace/HebrewModernBERT
export HF_HUB_OFFLINE=1

# Keep it to the three models the question needs: the competitor, our best, and the floor.
ONLY="${ONLY:-NeoDictaBERT,retrain FINAL,random-init}"
DOCS="${DOCS:-150}"
RARE="${RARE:-95}"

for CFG in "1 12" "3 4" "6 2"; do
    set -- $CFG; SPAN=$1; K=$2
    echo ""
    echo "############### span_len=$SPAN  targets/doc=$K  (=$((SPAN*K)) words masked/doc) ###############"
    python scripts/mlm_compare_vs_neodictabert.py \
        --docs_per_dataset "$DOCS" \
        --masks_per_doc "$K" \
        --span_len "$SPAN" \
        --rare_pct "$RARE" \
        --only "$ONLY" \
        --out "outputs/eval/mlm_hardmode_span${SPAN}.json"
done

echo ""
echo "=== CURVE: exact whole-span recovery, mean over the 5 datasets ==="
python - <<'PY'
import json, glob, re, os
rows = {}
for f in sorted(glob.glob("outputs/eval/mlm_hardmode_span*.json"),
                key=lambda p: int(re.search(r"span(\d+)", p).group(1))):
    span = int(re.search(r"span(\d+)", f).group(1))
    res = json.load(open(f))
    res = res.get("results", res)
    for label, per_ds in res.items():
        vals = [v["exact_acc"] for v in per_ds.values()]
        rows.setdefault(label, {})[span] = sum(vals) / len(vals)
spans = sorted({s for r in rows.values() for s in r})
print(f"{'model':26} " + " ".join(f"span={s:<7}" for s in spans))
print("-" * (26 + 12 * len(spans)))
for label in sorted(rows, key=lambda k: -rows[k].get(spans[0], 0)):
    print(f"{label:26} " + " ".join(f"{rows[label].get(s, float('nan')):<11.1f}" for s in spans))
ndb = next((k for k in rows if "NeoDicta" in k), None)
hmb = next((k for k in rows if "retrain FINAL" in k), None)
if ndb and hmb:
    print(f"\n{'GAP (NDB - HMB)':26} " +
          " ".join(f"{rows[ndb].get(s,0)-rows[hmb].get(s,0):<+11.1f}" for s in spans))
    print("\nWidening gap => the span=1 parity was a ceiling effect and capacity is the lever.")
PY
