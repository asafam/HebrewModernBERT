"""Sample Hebrew documents from the MDS corpus into SimCSE-style positive pairs.

WHY. HebrewModernBERT-large beats NeoDictaBERT on MLM (49.0 vs 46.9) but reaches only 0.187 avg
NDCG@10 vs their 0.330. The decisive diagnostic is recall@100 on arguana: NDB 0.974, us 0.296 — the
correct document is not in our top 100 for 70% of queries. That is a first-stage retrieval failure,
so re-ranking, ranking losses and hard negatives cannot fix it: they all operate on a candidate set
that does not contain the answer. Our embedding space does not place related Hebrew texts near each
other, which the anisotropy measurement (0.9997 vs NDB's 0.833) says directly.

MLM trains a token predictor, not a metric space. Contrastive training is the fix, and InfoNCE's
uniformity term attacks anisotropy by construction.

WHAT THIS EMITS. `{"query": text, "document": text}` — the SAME text on both sides. That is
unsupervised SimCSE: the positive pair is one text encoded twice, and the only difference between
the two views is dropout. So the consumer MUST enable dropout (our exported config has it at 0.0 —
see scripts/make_simcse_model_variant.sh) or the positive is an exact match and only the uniformity
term trains.

WHY NOT document crops. The corpus is short-document: docs/TRAINING.md §3 measured median ~100
tokens with 92% under 1024. Two "independent crops" of a 100-token document either overlap almost
entirely (a trivial positive) or are a few tokens each. Dropout positives need no length at all.

BALANCED BY SOURCE. The retrieval SFT was 88% nfcorpus and that monoculture cost us measurably
(capping it lifted arguana +322% and fiqa +25%). Do not repeat the mistake at pretraining: this
caps each `_source` to an equal share.

Usage (CPU-only; run via slurm, not on the login node):
    python scripts/sample_hebrew_simcse_pairs.py --n 1000000 --out data/simcse/hebrew_simcse_1M.jsonl
"""

import argparse
import json
import os
import random

import numpy as np
from streaming.base.format import reader_from_json


def load_shards(local: str, split: str):
    obj = json.load(open(os.path.join(local, split, "index.json")))
    shards = []
    for info in obj["shards"]:
        shard = reader_from_json(local, split, info)
        raw = os.path.join(shard.dirname, shard.split, shard.raw_data.basename)
        if os.path.isfile(raw):  # skip shards whose raw file is absent rather than dying
            shards.append(shard)
    return shards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", default="data/hebrewmodernbert/hebrew_quality")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--out", default="data/simcse/hebrew_simcse_1M.jsonl")
    ap.add_argument("--min-chars", type=int, default=200,
                    help="skip very short docs: a 1-2 sentence positive teaches little")
    ap.add_argument("--max-chars", type=int, default=4000,
                    help="truncate; SFT and eval both run at max_length 512 tokens anyway")
    ap.add_argument("--mode", choices=["dropout", "crop"], default="dropout",
                    help="dropout: the SAME text both sides, so the two views differ only by "
                         "dropout -- the model can win by surface matching, and the objective "
                         "SATURATES (measured: InfoNCE loss 5.17 -> 0.0021 by epoch 0.05, i.e. in "
                         "~200 of 3,906 steps). crop: two NON-OVERLAPPING spans of one document, "
                         "which forces semantic alignment instead and keeps the task hard. 90% of "
                         "sampled docs are >=500 chars, so crops are viable here. Crops are also "
                         "shorter, which frees memory for a larger negative pool.")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    random.seed(args.seed)
    shards = load_shards(args.local, args.split)
    print(f"{len(shards)} shards in {args.local}/{args.split}")

    # Visit shards in random order so we are not biased to whatever source was written first.
    order = list(range(len(shards)))
    random.shuffle(order)

    per_source_cap = None  # set after we learn how many sources there are
    counts, kept, seen = {}, 0, 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    with open(args.out, "w") as f:
        for si in order:
            shard = shards[si]
            for i in range(shard.samples):
                if kept >= args.n:
                    break
                sample = shard[i]
                seen += 1
                text = (sample.get("text") or "").strip()
                if len(text) < args.min_chars:
                    continue
                src = sample.get("_source") or "unknown"

                # Learn the source set from the first slice, then cap each equally.
                if per_source_cap is None and seen > 50_000:
                    per_source_cap = max(1, args.n // max(1, len(counts)))
                    print(f"  sources seen: {sorted(counts)} -> cap {per_source_cap:,} each")
                if per_source_cap is not None and counts.get(src, 0) >= per_source_cap:
                    continue

                text = text[: args.max_chars]
                if args.mode == "crop":
                    # Two non-overlapping spans. Split near the middle (jittered so the model
                    # cannot learn a fixed boundary), then take a window from each side.
                    if len(text) < 2 * args.min_chars // 2:
                        continue
                    mid = len(text) // 2
                    jit = random.randint(-len(text) // 10, len(text) // 10)
                    split = max(args.min_chars // 2, min(len(text) - args.min_chars // 2, mid + jit))
                    a, b = text[:split].strip(), text[split:].strip()
                    if len(a) < 100 or len(b) < 100:
                        continue
                    rec = {"query": a, "document": b}
                else:
                    # Same text both sides: dropout supplies the two views.
                    rec = {"query": text, "document": text}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                counts[src] = counts.get(src, 0) + 1
                kept += 1
                if kept % 100_000 == 0:
                    print(f"  {kept:,} pairs written ({seen:,} docs scanned)")
            if kept >= args.n:
                break

    print(f"\nwrote {kept:,} pairs to {args.out}  ({seen:,} docs scanned)")
    print("per-source counts:")
    for s, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {s:28s} {c:>9,}  ({100*c/max(1,kept):.1f}%)")


if __name__ == "__main__":
    main()
