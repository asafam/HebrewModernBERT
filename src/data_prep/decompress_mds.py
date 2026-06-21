"""Rewrite a compressed (zstd) MDS dataset as raw uncompressed .mds shards.

NoStreamingDataset (streaming: false) — the robust multi-GPU path used by phases
0.1/0.2 — only reads raw .mds shards; it has no decompression. The packed dataset
was written with compression=zstd. This reads it back (StreamingDataset decompresses)
and rewrites it with compression=None, reusing all the tokenization work (I/O only,
no re-tokenize). Columns preserved: input_ids (bytes), len (int).
"""
import argparse
import os
from streaming import StreamingDataset, MDSWriter
from torch.utils.data import DataLoader


def _collate(b):
    return [(bytes(x["input_ids"]), int(x["len"])) for x in b]


def go(src, split, out, workers, read_batch):
    if os.path.exists(os.path.join(out, "index.json")):
        print(f"  [{split}] already uncompressed (index.json exists) -> skip", flush=True)
        return
    os.makedirs(out, exist_ok=True)
    ds = StreamingDataset(local=src, split=split, shuffle=False, batch_size=read_batch)
    dl = DataLoader(ds, batch_size=read_batch, num_workers=workers, collate_fn=_collate,
                    prefetch_factor=4 if workers else None)
    n = 0
    with MDSWriter(out=out, columns={"input_ids": "bytes", "len": "int"}, compression=None) as w:
        for batch in dl:
            for ids, ln in batch:
                w.write({"input_ids": ids, "len": ln})
                n += 1
            if n % (read_batch * 200) < read_batch:
                print(f"  [{split}] {n:,} samples", flush=True)
    print(f"  [{split}] DONE {n:,} samples -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/hebrewmodernbert/hebrew_packed_8192")
    ap.add_argument("--out", default="data/hebrewmodernbert/hebrew_packed_8192_raw")
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--read-batch", type=int, default=512)
    args = ap.parse_args()
    go(args.src, "validation", os.path.join(args.out, "validation"), args.num_workers, args.read_batch)
    go(args.src, "train", os.path.join(args.out, "train"), args.num_workers, args.read_batch)
    print("DECOMPRESS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
