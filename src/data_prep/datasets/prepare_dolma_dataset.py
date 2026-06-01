"""Sample raw documents from Dolma up to a token budget (for the English/code mix parts).

The token budget is measured with the project tokenizer. The original implementation
tokenized **one document at a time, single-threaded**, across tens of billions of
tokens, purely to count — then stored the raw text and discarded the tokens, and
additionally wrapped the generator in ``IterableDataset.from_generator``. That made a
budget pass take many hours/days.

This version:
  * tokenizes in **batches** (HF fast tokenizers run multi-threaded in Rust; spm also
    encodes a list at once), which is the dominant speedup;
  * writes raw text straight to JSONL with no ``IterableDataset`` round-trip;
  * takes the tokenizer path as an argument (no hard-coded absolute path).

Run from the repo root, e.g.:
    python -m src.data_prep.datasets.prepare_dolma_dataset \
        --output_file data/dolma/corpus_sampled_eng_75B.jsonl \
        --token_budget 75_000_000_000 --data_path data/dolma \
        --tokenizer_path tokenizer/v3_clean --exclude_source starcoder
"""

import argparse
import glob
import json
import os
from typing import Callable, List

from datasets import load_dataset
from tqdm import tqdm


def make_batch_counter(tokenizer_path: str, max_length: int = 8192) -> Callable[[List[str]], List[int]]:
    """Return a function mapping a list of texts -> list of token counts (batched)."""
    if tokenizer_path.endswith(".model"):
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor(model_file=tokenizer_path)

        def count(texts: List[str]) -> List[int]:
            return [len(ids) for ids in sp.encode(texts)]

    else:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(tokenizer_path)

        def count(texts: List[str]) -> List[int]:
            enc = tok(texts, add_special_tokens=True, truncation=True, max_length=max_length,
                      return_attention_mask=False, return_token_type_ids=False)
            return [len(ids) for ids in enc["input_ids"]]

    return count


def sample_to_budget(files, token_budget, counter, output_file, text_field="text",
                     shuffle_buffer=1_000_000, seed=42, batch_size=1000):
    ds = load_dataset("json", data_files=files, split="train", streaming=True).shuffle(
        buffer_size=shuffle_buffer, seed=seed
    )
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    tokens = 0
    written = 0
    batch_texts: List[str] = []
    batch_recs: List[dict] = []
    pbar = tqdm(desc="sampling", unit=" tok", unit_scale=True, total=token_budget)

    with open(output_file, "w", encoding="utf-8") as out:
        def flush() -> bool:
            nonlocal tokens, written
            for length, rec in zip(counter(batch_texts), batch_recs):
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
                tokens += length
                pbar.update(length)
                if tokens >= token_budget:
                    return True
            return False

        done = False
        for ex in ds:
            batch_texts.append(ex[text_field])
            batch_recs.append({"id": ex.get("id"), "text": ex[text_field], "source": ex.get("source")})
            if len(batch_texts) >= batch_size:
                done = flush()
                batch_texts.clear()
                batch_recs.clear()
                if done:
                    break
        if not done and batch_texts:
            flush()
    pbar.close()
    print(f"Wrote {written:,} docs / {tokens:,} tokens (budget {token_budget:,}) -> {output_file}")
    return written, tokens


def main():
    p = argparse.ArgumentParser(description="Sample Dolma to a token budget")
    p.add_argument("--output_file", required=True)
    p.add_argument("--token_budget", type=int, default=25_000_000_000)
    p.add_argument("--data_path", default="data/dolma")
    p.add_argument("--tokenizer_path", default="tokenizer/v3_clean",
                   help="HF tokenizer dir or a SentencePiece .model file")
    p.add_argument("--exclude_source", nargs="*", default=[])
    p.add_argument("--include_source", nargs="*", default=[])
    p.add_argument("--shuffle_buffer", type=int, default=1_000_000)
    p.add_argument("--batch_size", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    files = glob.glob(os.path.join(a.data_path, "**/*.json.gz"), recursive=True)
    assert files, f"No .json.gz files found under {a.data_path}"
    if a.include_source:
        files = [f for f in files if any(s in f for s in a.include_source)]
    if a.exclude_source:
        files = [f for f in files if not any(s in f for s in a.exclude_source)]
    assert files, "No files left after include/exclude filtering"
    print(f"{len(files):,} files; budget {a.token_budget:,} tokens; tokenizer {a.tokenizer_path}")

    counter = make_batch_counter(a.tokenizer_path)
    sample_to_budget(files, a.token_budget, counter, a.output_file,
                     shuffle_buffer=a.shuffle_buffer, seed=a.seed, batch_size=a.batch_size)


if __name__ == "__main__":
    main()
