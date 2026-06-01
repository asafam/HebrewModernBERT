"""Train the SentencePiece BPE model for HebrewModernBERT.

Reproduces the original recipe (BPE, 100K vocab, character_coverage 0.9995, pad/unk at
0/1, [CLS]/[SEP]/[MASK] as user symbols) — the only change is using all CPU cores
instead of a hard-coded 4 threads. The produced ``<model_prefix>.model`` is then turned
into the HF fast tokenizer by ``build_hf_tokenizer.py`` (which renames <pad>/<unk> ->
[PAD]/[UNK]).

This is a *training* step — run it as a slurm job, not on the login node.
Build the corpus first with build_datasets.py (--format txt).
"""

import argparse
import os
from pathlib import Path

import sentencepiece as spm


def train(corpus: str, vocab_size: int, output_prefix: str, character_coverage: float = 0.9995) -> None:
    if not os.path.exists(corpus):
        raise FileNotFoundError(corpus)
    Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=corpus,
        model_prefix=output_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=character_coverage,
        num_threads=os.cpu_count() or 4,
        pad_id=0,
        unk_id=1,
        bos_id=-1,
        eos_id=-1,
        user_defined_symbols=["[CLS]", "[SEP]", "[MASK]"],
    )


def main():
    p = argparse.ArgumentParser(description="Train a SentencePiece BPE tokenizer")
    p.add_argument("--corpus", required=True, help="one-document-per-line .txt corpus")
    p.add_argument("--vocab_size", type=int, default=100000)
    p.add_argument("--output_prefix", default="tokenizer/spiece", help="model_prefix (writes <prefix>.model/.vocab)")
    p.add_argument("--character_coverage", type=float, default=0.9995)
    a = p.parse_args()
    train(a.corpus, a.vocab_size, a.output_prefix, a.character_coverage)


if __name__ == "__main__":
    main()
