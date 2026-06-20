"""Pack a raw-text MDS dataset into fixed-length tokenized sequences for context extension.

The Hebrew corpus is short-document (median ~100 tok, 92% < 1024), so training at
max_seq_len=8192 one-doc-per-sample never exercises long context. This concatenates
tokenized documents into dense `max_length`-token sequences (the ModernBERT / Fu et al.
approach) so phase-1/2 actually learn long-range attention.

Input  MDS columns:  text (str)            -> consumed via text_data._tokenize path
Output MDS columns:  input_ids (bytes,int64), len (int)
                     -> consumed via text_data._read_binary_tokenized_sample path

Each doc is tokenized WITH special tokens ([CLS] doc [SEP] for DebertaV2), the token
stream is concatenated, then sliced into exact `max_length` chunks (remainder wraps to
the next chunk). Doc boundaries remain visible to the model as [SEP][CLS].
"""
import argparse
import os
import numpy as np
from streaming import StreamingDataset, MDSWriter
from transformers import AutoTokenizer


def pack_split(src_local, split, out_dir, tokenizer, max_length, max_tokens, compression, batch_docs):
    os.makedirs(out_dir, exist_ok=True)
    ds = StreamingDataset(local=src_local, split=split, shuffle=False, batch_size=1)
    buf: list[int] = []
    emitted_tokens = 0
    n_seq = 0
    n_docs = 0
    pending: list[str] = []

    def flush_pending(writer):
        nonlocal buf, emitted_tokens, n_seq
        if not pending:
            return False
        enc = tokenizer(pending, truncation=False, padding=False)["input_ids"]
        for ids in enc:
            buf.extend(ids)
        pending.clear()
        while len(buf) >= max_length:
            chunk = np.asarray(buf[:max_length], dtype=np.int64)
            del buf[:max_length]
            writer.write({"input_ids": chunk.tobytes(), "len": int(max_length)})
            n_seq += 1
            emitted_tokens += max_length
            if max_tokens and emitted_tokens >= max_tokens:
                return True
        return False

    with MDSWriter(out=out_dir, columns={"input_ids": "bytes", "len": "int"}, compression=compression) as w:
        for sample in ds:
            text = sample.get("text")
            if not text:
                continue
            pending.append(text)
            n_docs += 1
            if len(pending) >= batch_docs:
                if flush_pending(w):
                    break
                if n_docs % (batch_docs * 50) == 0:
                    print(f"  [{split}] docs={n_docs:,} seqs={n_seq:,} tokens={emitted_tokens/1e9:.2f}B", flush=True)
        else:
            flush_pending(w)  # drain remaining pending (loop finished without hitting budget)
    print(f"  [{split}] DONE docs={n_docs:,} seqs={n_seq:,} tokens={emitted_tokens/1e9:.3f}B -> {out_dir}", flush=True)
    return n_seq, emitted_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/hebrewmodernbert/hebrew", help="source MDS root (has train/ validation/)")
    ap.add_argument("--out", default="data/hebrewmodernbert/hebrew_packed_8192", help="output MDS root")
    ap.add_argument("--tokenizer", default="tokenizer/v4_bpe_150k")
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--train-max-tokens", type=float, default=90e9, help="cap packed train tokens (covers phase1 30B + phase2 50B + margin; bounds runtime to fit the 4h wall)")
    ap.add_argument("--val-max-tokens", type=float, default=1e9)
    ap.add_argument("--compression", default="zstd")
    ap.add_argument("--batch-docs", type=int, default=2000, help="docs per batched tokenizer call")
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"tokenizer: {args.tokenizer} (mask={tok.mask_token_id} pad={tok.pad_token_id}) | max_length={args.max_length}", flush=True)

    pack_split(args.src, "validation", os.path.join(args.out, "validation"), tok, args.max_length, int(args.val_max_tokens), args.compression, args.batch_docs)
    pack_split(args.src, "train", os.path.join(args.out, "train"), tok, args.max_length, int(args.train_max_tokens), args.compression, args.batch_docs)
    print("PACKING COMPLETE", flush=True)


if __name__ == "__main__":
    main()
