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


def norm_u64(text: str) -> int:
    # Cap before normalize/regex: a few pathologically huge docs (hundreds of MB) make
    # unicodedata.normalize + the \s+ sub hang for minutes (this wedged two prior runs).
    # The first 100K chars (~22K tokens) fingerprint any real doc uniquely for dedup.
    t = (text or "")[:100_000]
    t = unicodedata.normalize("NFC", t)
    t = _WS.sub(" ", t).strip()
    return int.from_bytes(hashlib.blake2b(t.encode("utf-8"), digest_size=8).digest(), "little")


_DS = {}  # per-worker LocalDataset cache: open once per dir per worker (NOT per chunk)


def _get_ds(local):
    ds = _DS.get(local)
    if ds is None:
        ds = LocalDataset(local=local)
        _DS[local] = ds
    return ds


def _chunks(local, chunk_size):
    """Many small contiguous chunks -> imap_unordered load-balances across workers (no
    stragglers); cached LocalDataset means no per-chunk re-open. Sequential read within a chunk."""
    ds = LocalDataset(local=local); n = len(ds); del ds
    return [(local, lo, min(lo + chunk_size, n)) for lo in range(0, n, chunk_size)], n


def _hash_range(args):
    local, lo, hi = args
    ds = _get_ds(local)
    a = np.empty(hi - lo, dtype=np.uint64)
    for k, i in enumerate(range(lo, hi)):
        a[k] = norm_u64(ds[i]["text"])
    return a


def build_sorted(dirs, workers, chunk_size):
    parts = []
    total = 0
    for local in dirs:
        chunks, n = _chunks(local, chunk_size); total += n
        done = 0
        with Pool(workers, maxtasksperchild=8) as p:
            for arr in p.imap_unordered(_hash_range, chunks):
                parts.append(arr); done += 1
                if done % 25 == 0 or done == len(chunks):
                    print(f"   {local}: {done}/{len(chunks)} chunks ({done*chunk_size:,}/{n:,} docs)", flush=True)
        print(f"   {local}: {n:,} docs hashed", flush=True)
    allh = np.concatenate(parts) if parts else np.empty(0, np.uint64)
    print(f"   merging {len(allh):,} hashes (np.unique)...", flush=True)
    uniq = np.unique(allh)
    return uniq, total


_MAFAT = None  # set in parent before Pool() -> inherited COW by fork'd workers


def _fw_range(args):
    local, lo, hi = args
    ds = _get_ds(local)
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
    ap.add_argument("--chunk_size", type=int, default=250_000, help="docs per chunk (load-balancing granularity)")
    ap.add_argument("--chars_per_token", type=float, default=4.40)
    args = ap.parse_args()

    print(f"[1/2] hashing MAFAT {args.mafat} ({args.workers} workers, chunk {args.chunk_size:,})...", flush=True)
    global _MAFAT
    _MAFAT, mafat_n = build_sorted(args.mafat, args.workers, args.chunk_size)
    print(f"   MAFAT: {mafat_n:,} docs -> {len(_MAFAT):,} unique normalized hashes "
          f"(~{_MAFAT.nbytes/1e9:.1f}GB)", flush=True)

    print(f"[2/2] checking FineWeb {args.fineweb} vs MAFAT...", flush=True)
    nd = nc = dd = dc = 0
    for local in args.fineweb:
        chunks, n = _chunks(local, args.chunk_size)
        with Pool(args.workers, maxtasksperchild=8) as p:  # fork AFTER _MAFAT set -> COW; recycle to free shard cache
            for a, b, c, e in p.imap_unordered(_fw_range, chunks):
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
