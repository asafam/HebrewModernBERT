"""Sample a Hebrew-weighted text corpus for tokenizer training (one doc per line).

To beat DictaBERT on Hebrew fertility we want a Hebrew-heavy corpus, while keeping
enough English/code so the tokenizer still serves phase-0.1's mixed data. Samples
random whole shards (the MDS is source-contiguous, so random shards = diverse sources)
from the Hebrew split + the mixed split.

Read-only on the data; writes a plain .txt. Runs locally (I/O-bound).
"""

import argparse
import json
import random
from pathlib import Path

from streaming.base.format import reader_from_json


def sample_to_text(base: str, n_docs: int, out, rng: random.Random, max_chars: int = 20000) -> int:
    shards = json.load(open(f"{base}/index.json"))["shards"]
    order = list(range(len(shards)))
    rng.shuffle(order)
    written = 0
    for si in order:
        if written >= n_docs:
            break
        r = reader_from_json(base, "", shards[si])
        for k in range(shards[si]["samples"]):
            t = r[k].get("text", "").replace("\n", " ").replace("\r", " ").strip()
            if t:
                out.write(t[:max_chars] + "\n")
                written += 1
                if written >= n_docs:
                    break
    return written


def main():
    p = argparse.ArgumentParser(description="Build a Hebrew-weighted tokenizer corpus")
    p.add_argument("--hebrew_docs", type=int, default=2_000_000)
    p.add_argument("--mixed_docs", type=int, default=1_000_000)
    p.add_argument("--hebrew_path", default="data/hebrewmodernbert/hebrew/train")
    p.add_argument("--mixed_path", default="data/hebrewmodernbert/mixed/mixed_h50e75c25/train")
    p.add_argument("--output", default="data/tokenizer_corpus_heb_3M.txt")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    rng = random.Random(a.seed)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    with open(a.output, "w", encoding="utf-8") as out:
        nh = sample_to_text(a.hebrew_path, a.hebrew_docs, out, rng)
        print(f"wrote {nh:,} Hebrew docs")
        nm = sample_to_text(a.mixed_path, a.mixed_docs, out, rng)
        print(f"wrote {nm:,} mixed docs")
    print(f"total {nh + nm:,} docs -> {a.output}")


if __name__ == "__main__":
    main()
