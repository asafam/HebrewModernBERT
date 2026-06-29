#!/usr/bin/env python3
"""
Measure FineWeb-2 Hebrew net-new vs the existing MAFAT/hebrew corpus.

Cross-corpus dedup by NORMALIZED-text exact match (NFC + whitespace-collapse + strip),
which also catches whitespace/boilerplate-trivial near-dups. Each doc -> 64-bit hash;
MAFAT hashes are stored in a single sorted numpy uint64 array (shared read-only across
fork'd workers via copy-on-write, queried with searchsorted) so memory is ~4GB, not 35GB.
CPU-only; no `src` import (avoids the FlexBERT/triton init).

MAFAT is mostly non-CC (YifatData/HebNLI) with a HeC4 minority, so overlap with FineWeb-2
(CC) is expected low. This reports exact-normalized overlap = an UPPER bound on net-new;
a full minhash near-dup pass would only lower net-new slightly further.
"""
import argparse
import hashlib
import os
import re
import unicodedata
from multiprocessing import Pool

import numpy as np
from streaming import LocalDataset

_WS = re.compile(r"\s+")
CHUNK = 200_000


def norm_u64(text: str) -> int:
    t = unicodedata.normalize("NFC", text or "")
    t = _WS.sub(" ", t).strip()
    return int.from_bytes(hashlib.blake2b(t.encode("utf-8"), digest_size=8).digest(), "little")


def _hash_range(args):
    local, lo, hi = args
    ds = LocalDataset(local=local)
    n = min(hi, len(ds))
    a = np.empty(n - lo, dtype=np.uint64)
    for k, i in enumerate(range(lo, n)):
        a[k] = norm_u64(ds[i]["text"])
    return a


def build_sorted(dirs, workers):
    parts = []
    total = 0
    for local in dirs:
        ds = LocalDataset(local=local); n = len(ds); del ds
        total += n
        ranges = [(local, lo, min(lo + CHUNK, n)) for lo in range(0, n, CHUNK)]
        with Pool(workers) as p:
            for arr in p.imap_unordered(_hash_range, ranges):
                parts.append(arr)
        print(f"   {local}: {n:,} docs", flush=True)
    allh = np.concatenate(parts) if parts else np.empty(0, np.uint64)
    uniq = np.unique(allh)  # sorted + deduped
    return uniq, total


_MAFAT = None  # set in parent before Pool() -> inherited COW by fork'd workers


def _fw_range(args):
    local, lo, hi = args
    ds = LocalDataset(local=local)
    n = min(hi, len(ds))
    nd = nc = dd = dc = 0
    for i in range(lo, n):
        t = ds[i]["text"]; nd += 1; L = len(t); nc += L
        h = np.uint64(norm_u64(t))
        idx = np.searchsorted(_MAFAT, h)
        if idx < len(_MAFAT) and _MAFAT[idx] == h:
            dd += 1; dc += L
    return nd, nc, dd, dc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mafat", nargs="+", required=True)
    ap.add_argument("--fineweb", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 1))
    ap.add_argument("--chars_per_token", type=float, default=4.40)
    args = ap.parse_args()

    print(f"[1/2] hashing MAFAT {args.mafat} ({args.workers} workers)...", flush=True)
    global _MAFAT
    _MAFAT, mafat_n = build_sorted(args.mafat, args.workers)
    print(f"   MAFAT: {mafat_n:,} docs -> {len(_MAFAT):,} unique normalized hashes "
          f"(~{_MAFAT.nbytes/1e9:.1f}GB)", flush=True)

    print(f"[2/2] checking FineWeb {args.fineweb} vs MAFAT...", flush=True)
    nd = nc = dd = dc = 0
    for local in args.fineweb:
        ds = LocalDataset(local=local); n = len(ds); del ds
        ranges = [(local, lo, min(lo + CHUNK, n)) for lo in range(0, n, CHUNK)]
        with Pool(args.workers) as p:  # fork AFTER _MAFAT is set -> COW shared
            for a, b, c, e in p.imap_unordered(_fw_range, ranges):
                nd += a; nc += b; dd += c; dc += e

    net_docs, net_chars = nd - dd, nc - dc
    net_tok = net_chars / args.chars_per_token
    dup_tok = dc / args.chars_per_token
    print("\n================ DEDUP RESULT ================", flush=True)
    print(f"FineWeb docs:          {nd:,}", flush=True)
    print(f"  dup of MAFAT:        {dd:,} ({(dd/nd*100 if nd else 0):.1f}%)  ~{dup_tok/1e9:.2f}B tok", flush=True)
    print(f"  NET-NEW:             {net_docs:,} ({(net_docs/nd*100 if nd else 0):.1f}%)  ~{net_tok/1e9:.2f}B tok", flush=True)
    print(f"\n=> MAFAT ~30B + net-new ~{net_tok/1e9:.1f}B = ~{(30e9+net_tok)/1e9:.0f}B unique Hebrew", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
