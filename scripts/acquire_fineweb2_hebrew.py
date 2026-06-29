#!/usr/bin/env python3
"""
Acquire FineWeb-2 Hebrew (heb_Hebr) -> MDS, schema-compatible with the existing
Hebrew corpus (columns: _row_number, _source, text). CPU-only, no `src` package
import (avoids the FlexBERT/flash-attn/triton init in src/__init__.py).

Streams from HF (no 90GB in-memory load), writes train/validation MDS shards.
Public data — safe to publish; must still be deduplicated vs MAFAT before training.

Usage:
  python scripts/acquire_fineweb2_hebrew.py --output data/hebrewmodernbert/fineweb2_hebrew [--limit N]
"""
import argparse
import os
import random

from datasets import load_dataset
from streaming import MDSWriter

COLUMNS = {"_row_number": "int", "_source": "str", "text": "str"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, help="MDS output dir (creates train/ and validation/)")
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-2")
    ap.add_argument("--config", default="heb_Hebr")
    ap.add_argument("--source_name", default="fineweb2_hebrew")
    ap.add_argument("--limit", type=int, default=0, help="0 = all docs")
    ap.add_argument("--val_fraction", type=float, default=0.01)
    ap.add_argument("--min_chars", type=int, default=1, help="skip docs shorter than this")
    ap.add_argument("--shard_size_limit", type=int, default=67108864)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_every", type=int, default=50000)
    args = ap.parse_args()

    ds = load_dataset(args.dataset, args.config, split="train", streaming=True)
    rng = random.Random(args.seed)

    train_dir = os.path.join(args.output, "train")
    val_dir = os.path.join(args.output, "validation")
    os.makedirs(args.output, exist_ok=True)

    n_train = n_val = n_skip = 0
    with MDSWriter(out=train_dir, columns=COLUMNS, size_limit=args.shard_size_limit) as wtr, \
         MDSWriter(out=val_dir, columns=COLUMNS, size_limit=args.shard_size_limit) as wval:
        for i, rec in enumerate(ds):
            if args.limit and i >= args.limit:
                break
            text = rec.get("text") or ""
            if len(text) < args.min_chars:
                n_skip += 1
                continue
            row = {"_row_number": i, "_source": args.source_name, "text": text}
            if rng.random() < args.val_fraction:
                wval.write(row); n_val += 1
            else:
                wtr.write(row); n_train += 1
            if (n_train + n_val) % args.log_every == 0:
                print(f"  wrote {n_train:,} train / {n_val:,} val ({n_skip:,} skipped)", flush=True)

    print(f"DONE: {n_train:,} train + {n_val:,} val docs ({n_skip:,} skipped) -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
    # MDS is fully written/closed above. Hard-exit to skip the zstandard/torch
    # interpreter-finalization thread cleanup that cores on this env (harmless,
    # post-write) but would otherwise give a non-zero exit code.
    os._exit(0)
