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
import json
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


_EMIT_DIR = None  # set in parent before Pool() -> inherited COW by fork'd workers


def _fw_filter_range(args):
    # Same scan as _fw_range, but non-dup docs are written straight to a per-chunk part
    # file (never returned through the Pool IPC channel -> avoids piping ~41GB of text).
    local, lo, hi = args
    ds = _get_ds(local)
    n = min(hi, len(ds))
    nd = nc = dd = dc = 0
    # Chunk (lo) indices restart at 0 for EACH `local` dir independently (see _chunks) -> a
    # part filename keyed on lo alone collides across dirs (e.g. a small validation split's
    # only chunk vs. train's first chunk both have lo=0, silently overwriting one another).
    # Key the filename on the source dir too so every (local, lo) pair is unique.
    safe_local = local.strip("/").replace("/", "_")
    out_path = os.path.join(_EMIT_DIR, f"part-{safe_local}-{lo:09d}.jsonl")
    with open(out_path, "w", encoding="utf-8") as out:
        for i in range(lo, n):
            t = ds[i]["text"]; nd += 1; L = len(t); nc += L
            h = np.uint64(norm_u64(t))
            idx = np.searchsorted(_MAFAT, h)
            if idx < len(_MAFAT) and _MAFAT[idx] == h:
                dd += 1; dc += L
            else:
                out.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    return nd, nc, dd, dc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mafat", nargs="+", default=None, help="required unless --load_hashes is given")
    ap.add_argument("--fineweb", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 1))
    ap.add_argument("--chunk_size", type=int, default=250_000, help="docs per chunk (load-balancing granularity)")
    ap.add_argument("--chars_per_token", type=float, default=4.40)
    ap.add_argument("--save_hashes", default=None, help="path to persist sorted MAFAT uint64 hash array (.npy)")
    ap.add_argument("--load_hashes", default=None,
                    help="path to a previously --save_hashes'd .npy; skips re-hashing --mafat entirely")
    ap.add_argument("--emit_jsonl", default=None,
                    help="if set, write non-dup FineWeb docs as {'text':...} jsonl parts into this dir")
    ap.add_argument("--emit_limit", type=int, default=0,
                    help="cap total FineWeb docs scanned (across all --fineweb dirs), for smoke tests")
    args = ap.parse_args()

    global _MAFAT
    if args.load_hashes:
        print(f"[1/2] loading MAFAT hashes from {args.load_hashes} (skipping re-hash)...", flush=True)
        _MAFAT = np.load(args.load_hashes)
        print(f"   loaded {len(_MAFAT):,} unique normalized hashes (~{_MAFAT.nbytes/1e9:.1f}GB)", flush=True)
    else:
        if not args.mafat:
            ap.error("--mafat is required unless --load_hashes is given")
        print(f"[1/2] hashing MAFAT {args.mafat} ({args.workers} workers, chunk {args.chunk_size:,})...", flush=True)
        _MAFAT, mafat_n = build_sorted(args.mafat, args.workers, args.chunk_size)
        print(f"   MAFAT: {mafat_n:,} docs -> {len(_MAFAT):,} unique normalized hashes "
              f"(~{_MAFAT.nbytes/1e9:.1f}GB)", flush=True)

        if args.save_hashes:
            np.save(args.save_hashes, _MAFAT)
            print(f"   saved MAFAT hashes -> {args.save_hashes} ({_MAFAT.nbytes/1e9:.1f}GB)", flush=True)

    fw_fn = _fw_range
    if args.emit_jsonl:
        os.makedirs(args.emit_jsonl, exist_ok=True)
        global _EMIT_DIR
        _EMIT_DIR = args.emit_jsonl
        fw_fn = _fw_filter_range

    print(f"[2/2] checking FineWeb {args.fineweb} vs MAFAT...", flush=True)
    nd = nc = dd = dc = 0
    remaining = args.emit_limit if args.emit_limit > 0 else None
    for local in args.fineweb:
        chunks, n = _chunks(local, args.chunk_size)
        if remaining is not None:
            if remaining <= 0:
                break
            capped = []
            for (loc, lo, hi) in chunks:
                if remaining <= 0:
                    break
                hi2 = min(hi, lo + remaining)
                capped.append((loc, lo, hi2))
                remaining -= (hi2 - lo)
            chunks = capped
        with Pool(args.workers, maxtasksperchild=8) as p:  # fork AFTER _MAFAT set -> COW; recycle to free shard cache
            for a, b, c, e in p.imap_unordered(fw_fn, chunks):
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
