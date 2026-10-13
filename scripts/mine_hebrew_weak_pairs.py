"""Mine ASYMMETRIC weak-supervision pairs (title/headline <-> body) from the raw Hebrew sources.

WHY THIS AND NOT MORE CROP PAIRS. Contrastive adaptation on 1M symmetric SimCSE pairs took
retrieval from 0.187 to 0.295 (89% of NeoDictaBERT) in three hours, but its objective SATURATED --
InfoNCE loss 5.17 -> 0.0021 within ~200 of 3,906 steps. Symmetric long<->long positives are easy:
the model can win by surface matching, and once it can, there is no gradient left.

Title<->body pairs are qualitatively different. A title is SHORT and query-like; a body is LONG and
document-like. That asymmetry is the actual shape of the retrieval task (BeIR pairs a short query
with a long passage), so these teach the thing we are measured on rather than a proxy for it. This
is the first stage of mE5's recipe, which is the only published route to the 0.382 at the top of
our board.

WHAT IS MINABLE, verified 2026-10-14 by reading the raw files:
  wikipedia  text = "TITLE\\n<article>"            -> title <-> body          (1.5 GB)
  news       text = "HEADLINE\\n<article>", 20/20   -> headline <-> body       (1.1 GB)
  hebnli     NOT USABLE. The MAFAT HebNLI dump is flattened web text, not premise/hypothesis
             pairs -- its first record is the boilerplate 'אתר ע"י לינקטק ישראל'. The NLI
             structure did not survive, so do not plan on it.

PRIVACY. These raw files ARE the private MAFAT corpus. Output contains MAFAT text and must stay
under data/ (gitignored) and never be published. Model weights and this script are public; the
pairs are not.

Usage (CPU-only; run via slurm, not on the login node):
    python scripts/mine_hebrew_weak_pairs.py --out data/simcse/hebrew_weak_pairs.jsonl
"""

import argparse
import json
import os
import random
import re

# Hebrew text keeps quotes/gershayim/punctuation that vary between a title and its restatement in
# the body ('דלק חברת הדלק' vs '"דלק" חברת הדלק'), so an exact substring test misses the
# near-duplicate. Normalise those away before comparing.
_NOISE = re.compile(r'["\'\u05f3\u05f4\u2018\u2019\u201c\u201d()\[\],.:;!?\-\u2013\u2014\s]+')


def _norm(s):
    return _NOISE.sub("", s)

# Raw sources live in the sibling repo, which is where the corpus build runs from.
SIB = "/home/nlp/achimoa/workspace/hebrew_text_retrieval"
SOURCES = {
    "wikipedia": "data/mafat/hebrew/sources/AllOfNewHebrewWikipediaWithArticles-Oct29-2023.forgpt.jsonl",
    "news_yisraelhayom": "data/mafat/hebrew/sources/YisraelHayomData-Combined-Deduped.forgpt.jsonl",
}


def mine_title_body(path, min_title=8, max_title=250, min_body=200, max_body=4000, limit=None):
    """Yield (title, body) where the raw text is 'TITLE\\n<body>'.

    utf-8-sig, because the Wikipedia dump carries a BOM that plain utf-8 rejects on line 1.
    """
    n = 0
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            if limit is not None and n >= limit:
                return
            try:
                text = json.loads(line).get("text") or ""
            except json.JSONDecodeError:
                continue
            title, _, body = text.partition("\n")
            title, body = title.strip(), body.strip()
            if not (min_title <= len(title) <= max_title) or len(body) < min_body:
                continue
            # Wikipedia bodies restate the title in their first line(s), often with quotes added
            # ('דלק חברת הדלק' -> '"דלק" חברת הדלק'), so an exact substring test misses it. Compare
            # NORMALISED forms and drop any line that contains, or is contained by, the title. Left
            # in, the positive is solvable by copying the query -- reintroducing exactly the
            # surface-matching shortcut that made the SimCSE objective saturate at loss ~0.
            nt = _norm(title)
            kept = []
            for ln in body.split("\n"):
                nl = _norm(ln)
                if nt and nl and (nt in nl or nl in nt):
                    continue
                kept.append(ln)
            body = "\n".join(kept).strip()
            if len(body) < min_body:
                continue
            yield title, body[:max_body]
            n += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/simcse/hebrew_weak_pairs.jsonl")
    ap.add_argument("--per-source", type=int, default=None,
                    help="cap per source; default = all. Keeps one source from dominating, the "
                         "mistake the retrieval SFT made at 88% nfcorpus.")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()
    random.seed(args.seed)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    counts = {}
    with open(args.out, "w") as out:
        for name, rel in SOURCES.items():
            path = os.path.join(SIB, rel)
            if not os.path.isfile(path):
                print(f"  SKIP {name}: {path} not found")
                continue
            n = 0
            for title, body in mine_title_body(path, limit=args.per_source):
                out.write(json.dumps({"query": title, "document": body}, ensure_ascii=False) + "\n")
                n += 1
                if n % 100_000 == 0:
                    print(f"  {name}: {n:,}")
            counts[name] = n
            print(f"  {name}: {n:,} pairs")

    total = sum(counts.values())
    print(f"\nwrote {total:,} asymmetric pairs to {args.out}")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:22s} {v:>9,}  ({100*v/max(1,total):.1f}%)")
    print("\nREMINDER: this output contains MAFAT text. Keep it under data/ and never publish it.")


if __name__ == "__main__":
    main()
