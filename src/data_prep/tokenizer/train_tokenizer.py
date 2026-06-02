"""Train a SentencePiece tokenizer for HebrewModernBERT.

Parameterized for the tokenizer bake-off: model_type (unigram|bpe), vocab_size,
character_coverage, byte_fallback. Special tokens kept at the canonical ids
pad=0/unk=1/[CLS]=2/[SEP]=3/[MASK]=4; build_hf_tokenizer.py then renames <pad>/<unk>
-> [PAD]/[UNK] and wraps as an HF fast tokenizer.

byte_fallback=True makes OOV characters encode as bytes (never [UNK]) — drives the
unknown rate to ~0. This is a *training* step — run as a slurm job (CPU; 150K vocab
wants lots of RAM, e.g. cpu1T-24h). Build the corpus first (build_tokenizer_corpus.py).
"""

import argparse
import os
from pathlib import Path

import sentencepiece as spm


def train(
    corpus: str,
    vocab_size: int,
    output_prefix: str,
    model_type: str = "unigram",
    character_coverage: float = 0.9999,
    byte_fallback: bool = True,
    input_sentence_size: int = 0,
) -> None:
    if not os.path.exists(corpus):
        raise FileNotFoundError(corpus)
    Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        input=corpus,
        model_prefix=output_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        byte_fallback=byte_fallback,
        num_threads=os.cpu_count() or 4,
        pad_id=0,
        unk_id=1,
        bos_id=-1,
        eos_id=-1,
        user_defined_symbols=["[CLS]", "[SEP]", "[MASK]"],
        train_extremely_large_corpus=True,
    )
    if input_sentence_size > 0:
        kwargs["input_sentence_size"] = input_sentence_size
        kwargs["shuffle_input_sentence"] = True
    spm.SentencePieceTrainer.train(**kwargs)


def main():
    p = argparse.ArgumentParser(description="Train a SentencePiece tokenizer")
    p.add_argument("--corpus", required=True, help="one-document-per-line .txt corpus")
    p.add_argument("--vocab_size", type=int, default=150016, help="multiple of 128 for tensor-core efficiency")
    p.add_argument("--output_prefix", default="tokenizer/spiece", help="writes <prefix>.model/.vocab")
    p.add_argument("--model_type", default="unigram", choices=["unigram", "bpe"])
    p.add_argument("--character_coverage", type=float, default=0.9999)
    p.add_argument("--no_byte_fallback", action="store_true", help="disable byte fallback (allows [UNK])")
    p.add_argument("--input_sentence_size", type=int, default=0, help="cap sentences (0 = all)")
    a = p.parse_args()
    train(a.corpus, a.vocab_size, a.output_prefix, a.model_type, a.character_coverage,
          byte_fallback=not a.no_byte_fallback, input_sentence_size=a.input_sentence_size)


if __name__ == "__main__":
    main()
