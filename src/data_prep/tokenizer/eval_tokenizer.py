"""Benchmark tokenizer(s) on Hebrew fertility / unknown-rate vs baselines (DictaBERT etc.).

Lower tok/word (fertility) = more efficient. Goal: beat DictaBERT (~1.31 tok/word,
~0.085% UNK). Samples real Hebrew docs from the corpus and scores each candidate plus
the standard baselines on the SAME text.

    python -m src.data_prep.tokenizer.eval_tokenizer --candidates tokenizer/v3_clean tokenizer/cand_unigram
"""

import argparse
import json
import random
import re
from collections import Counter

from streaming.base.format import reader_from_json
from transformers import AutoTokenizer

HEB = re.compile(r"[֐-׿]")
BASELINES = [("DictaBERT", "dicta-il/dictabert"), ("mBERT", "bert-base-multilingual-cased"), ("XLM-R", "xlm-roberta-base")]


class SPMTok:
    """Thin adapter so a raw SentencePiece .model scores like an HF tokenizer."""

    def __init__(self, model_file: str):
        import sentencepiece as spm

        self.sp = spm.SentencePieceProcessor(model_file=model_file)
        self.unk_token_id = self.sp.unk_id()

    def encode(self, text, add_special_tokens=False):
        return self.sp.encode(text)

    def __len__(self):
        return self.sp.get_piece_size()

    def get_vocab(self):
        return {self.sp.id_to_piece(i): i for i in range(self.sp.get_piece_size())}


def load(path: str):
    return SPMTok(path) if path.endswith(".model") else AutoTokenizer.from_pretrained(path)


def sample_docs(base: str, n: int, seed: int = 5) -> list:
    rng = random.Random(seed)
    shards = json.load(open(f"{base}/index.json"))["shards"]
    pick = sorted(rng.sample(range(len(shards)), min(50, len(shards))))
    docs = []
    for si in pick:
        r = reader_from_json(base, "", shards[si])
        for k in range(min(60, shards[si]["samples"])):
            t = r[k].get("text", "").strip()
            if t:
                docs.append(t)
    return docs[:n]


def score(tok, docs, words, chars):
    unk_id = tok.unk_token_id
    total = unk = 0
    for d in docs:
        ids = tok.encode(d, add_special_tokens=False)
        total += len(ids)
        if unk_id is not None:
            unk += ids.count(unk_id)
    return total / words, total / chars, 100 * unk / max(total, 1)


def composition(tok):
    c = Counter()
    for s in tok.get_vocab():
        s = s.replace("▁", "")
        if not s:
            c["space"] += 1
        elif HEB.search(s):
            c["hebrew"] += 1
        elif re.search(r"[A-Za-z]", s):
            c["latin"] += 1
        elif re.search(r"[0-9]", s):
            c["digit"] += 1
        else:
            c["other"] += 1
    return c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", nargs="+", required=True, help="tokenizer dirs to evaluate")
    p.add_argument("--hebrew_path", default="data/hebrewmodernbert/hebrew/train")
    p.add_argument("--n_docs", type=int, default=2500)
    p.add_argument("--no_baselines", action="store_true")
    a = p.parse_args()

    docs = sample_docs(a.hebrew_path, a.n_docs)
    words = sum(len(d.split()) for d in docs)
    chars = sum(len(d) for d in docs)
    print(f"{len(docs)} Hebrew docs | {words:,} words | {chars:,} chars\n")

    rows = [(c, load(c)) for c in a.candidates]
    if not a.no_baselines:
        for name, hub in BASELINES:
            try:
                rows.append((name, AutoTokenizer.from_pretrained(hub)))
            except Exception as ex:
                print(f"  (skip {name}: {type(ex).__name__})")

    print(f"{'tokenizer':<34}{'vocab':>8}{'tok/word':>10}{'tok/char':>10}{'UNK%':>8}")
    for name, tok in rows:
        fw, fc, unk = score(tok, docs, words, chars)
        flag = "  <- beats Dicta" if (fw < 1.313 and unk <= 0.085) else ""
        print(f"{name:<34}{len(tok):>8}{fw:>10.3f}{fc:>10.3f}{unk:>8.3f}{flag}")

    for name, tok in rows:
        if name in a.candidates:
            comp = composition(tok)
            tot = sum(comp.values())
            print(f"\n{name} vocab: " + ", ".join(f"{k} {100*v/tot:.0f}%" for k, v in comp.most_common()))


if __name__ == "__main__":
    main()
