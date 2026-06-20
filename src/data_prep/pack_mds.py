"""Pack a raw-text MDS dataset into fixed-length tokenized sequences for context extension.

The Hebrew corpus is short-document (median ~100 tok, 92% < 1024), so training at
max_seq_len=8192 one-doc-per-sample never exercises long context. This concatenates
tokenized documents into dense `max_length`-token sequences (the ModernBERT / Fu et al.
approach) so phase-1/2 actually learn long-range attention.

Input  MDS columns:  text (str)            -> consumed via text_data._tokenize path
Output MDS columns:  input_ids (bytes,int64), len (int)
                     -> consumed via text_data._read_binary_tokenized_sample path

Each doc is tokenized WITH special tokens ([CLS] doc [SEP] for DebertaV2), the token
stream is concatenated, then sliced into exact `max_length` chunks. Doc boundaries stay
visible to the model as [SEP][CLS].

Perf: reads via a DataLoader with num_workers (StreamingDataset's per-doc zstd decode is
the bottleneck, not the tokenizer) and uses a numpy chunk buffer (Python-list slicing was
O(n) per chunk). Skips a split whose index.json already exists.
"""
import argparse
import os
import numpy as np
from streaming import StreamingDataset, MDSWriter
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def _collate(batch):
    return [x["text"] for x in batch if x.get("text")]


def pack_split(src_local, split, out_dir, tokenizer, max_length, max_tokens, compression, read_batch, num_workers):
    if os.path.exists(os.path.join(out_dir, "index.json")):
        print(f"  [{split}] already packed (index.json exists) -> skipping", flush=True)
        return
    os.makedirs(out_dir, exist_ok=True)
    ds = StreamingDataset(local=src_local, split=split, shuffle=False, batch_size=read_batch)
    loader = DataLoader(ds, batch_size=read_batch, num_workers=num_workers, collate_fn=_collate, prefetch_factor=4 if num_workers else None)

    leftover = np.empty(0, dtype=np.int64)
    emitted = 0
    n_seq = 0
    n_docs = 0
    with MDSWriter(out=out_dir, columns={"input_ids": "bytes", "len": "int"}, compression=compression) as w:
        stop = False
        for texts in loader:
            if not texts:
                continue
            n_docs += len(texts)
            enc = tokenizer(texts, truncation=False, padding=False, add_special_tokens=True)["input_ids"]
            flat = np.concatenate([leftover] + [np.asarray(ids, dtype=np.int64) for ids in enc])
            n_full = len(flat) // max_length
            if n_full:
                chunks = flat[: n_full * max_length].reshape(n_full, max_length)
                for row in chunks:
                    w.write({"input_ids": row.tobytes(), "len": int(max_length)})
                n_seq += n_full
                emitted += n_full * max_length
                leftover = flat[n_full * max_length:].copy()
            else:
                leftover = flat
            if n_docs % (read_batch * 100) < read_batch:
                print(f"  [{split}] docs={n_docs:,} seqs={n_seq:,} tokens={emitted/1e9:.2f}B", flush=True)
            if max_tokens and emitted >= max_tokens:
                stop = True
                break
        del stop
    print(f"  [{split}] DONE docs={n_docs:,} seqs={n_seq:,} tokens={emitted/1e9:.3f}B -> {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/hebrewmodernbert/hebrew")
    ap.add_argument("--out", default="data/hebrewmodernbert/hebrew_packed_8192")
    ap.add_argument("--tokenizer", default="tokenizer/v4_bpe_150k")
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--train-max-tokens", type=float, default=30e9)
    ap.add_argument("--val-max-tokens", type=float, default=1e9)
    ap.add_argument("--compression", default="zstd")
    ap.add_argument("--read-batch", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=12)
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"tokenizer: {args.tokenizer} (mask={tok.mask_token_id} pad={tok.pad_token_id}) max_length={args.max_length} workers={args.num_workers}", flush=True)

    pack_split(args.src, "validation", os.path.join(args.out, "validation"), tok, args.max_length, int(args.val_max_tokens), args.compression, args.read_batch, args.num_workers)
    pack_split(args.src, "train", os.path.join(args.out, "train"), tok, args.max_length, int(args.train_max_tokens), args.compression, args.read_batch, args.num_workers)
    print("PACKING COMPLETE", flush=True)


if __name__ == "__main__":
    main()
