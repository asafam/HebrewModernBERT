"""Build the HebrewModernBERT HuggingFace **fast** tokenizer from a trained
SentencePiece model — reproducibly, and without the defects of the original
notebook-cell wrap.

Background / what this fixes
----------------------------
The original tokenizer was produced by an ad-hoc notebook cell that wrapped
``spiece.model`` in a slow ``AlbertTokenizer``. That introduced two defects:

1. **Duplicate special tokens.** Albert re-added ``[UNK]``/``[PAD]`` as *new*
   tokens at ids 100000/100001, on top of the spm model's native ``<unk>``@1 /
   ``<pad>``@0 — leaving two dead duplicate slots (vocab 100002 instead of 100000).
2. **Bad normalization defaults.** Albert applies ``do_lower_case=True`` and
   ``keep_accents=False`` on top of the spm model's own ``nmt_nfkc`` normalizer —
   lowercasing Latin and stripping combining marks the spm would otherwise keep.

This script instead:
  * renames the spm model's native pieces ``<pad>``@0 -> ``[PAD]`` and
    ``<unk>``@1 -> ``[UNK]`` (proto edit, no retraining) so the special tokens are
    a clean, BERT-style ``[PAD]@0 [UNK]@1 [CLS]@2 [SEP]@3 [MASK]@4`` with **no
    duplicates** (final vocab = 100000);
  * converts the (BPE) spm model to a HF **fast** tokenizer that reproduces the
    spm token ids exactly, carrying only the spm's own ``nmt_nfkc`` normalization
    (no extra lowercasing / accent stripping);
  * verifies fidelity (fast ids == raw spm ids), special-token ids, and round-trips.

Note on niqqud: the spm model cannot represent Hebrew niqqud (vowel marks fall
below its character-coverage cutoff -> ``<unk>``). The training corpus is
unvocalized modern Hebrew (measured: 0% of docs carry niqqud), so this is
expected and intentionally left as-is. Supporting niqqud would require retraining
the spm model (a separate, slurm job).

The model embedding is padded to a multiple of 64 (100000 -> 100032) at model
build time by ``src/flex_bert.py`` (it appends ``<DUMMY_i>`` rows); the tokenizer
itself stays at 100000 real pieces.

Runs locally (lightweight — no training). Env: ``bert25`` (newer transformers).
"""

import argparse
import json
from pathlib import Path

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as sp_pb2
from tokenizers import Tokenizer
from transformers import DebertaV2Tokenizer, PreTrainedTokenizerFast
from transformers.convert_slow_tokenizer import convert_slow_tokenizer

# Canonical special tokens and their (post-rename) spm ids.
SPECIALS = {"pad": ("[PAD]", 0), "unk": ("[UNK]", 1), "cls": ("[CLS]", 2), "sep": ("[SEP]", 3), "mask": ("[MASK]", 4)}

# Sample texts used to assert the fast tokenizer reproduces the spm exactly.
_CHECK_TEXTS = [
    "שלום עולם",
    "הילד הלך לבית הספר עם חבריו בבוקר",
    "המשרד לביטחון פנים פרסם הודעה רשמית",
    "Mixed: def add(a, b): return a + b  # 123 ₪",
]


def rename_spm_specials(input_spm: Path, output_spm: Path) -> None:
    """Rename the spm model's native ``<pad>``@0 -> ``[PAD]`` and ``<unk>``@1 -> ``[UNK]``.

    Only the surface strings change; the piece *types* (CONTROL for pad, UNKNOWN
    for unk) and ids are preserved, so encoding behaviour is unchanged.
    """
    proto = sp_pb2.ModelProto()
    proto.ParseFromString(input_spm.read_bytes())
    assert proto.pieces[0].piece == "<pad>", f"expected <pad>@0, got {proto.pieces[0].piece!r}"
    assert proto.pieces[1].piece == "<unk>", f"expected <unk>@1, got {proto.pieces[1].piece!r}"
    proto.pieces[0].piece = "[PAD]"
    proto.pieces[1].piece = "[UNK]"
    output_spm.parent.mkdir(parents=True, exist_ok=True)
    output_spm.write_bytes(proto.SerializeToString())


def _rename_specials_in_fast(tokenizer_obj: Tokenizer) -> Tokenizer:
    """Rename ``<pad>``/``<unk>`` -> ``[PAD]``/``[UNK]`` in a converted fast BPE tokenizer.

    Done on the converted JSON (not the source spm) because the slow->fast
    converter resolves the unk piece by its original name during conversion;
    renaming vocab keys, the model ``unk_token`` field, and the added-tokens
    entries together keeps everything consistent. The two pieces never appear in
    BPE merges (they are control/unk), so merges need no edits.
    """
    d = json.loads(tokenizer_obj.to_str())
    vocab = d["model"]["vocab"]
    for old, new in (("<pad>", "[PAD]"), ("<unk>", "[UNK]")):
        assert old in vocab, f"{old} not in converted vocab"
        vocab[new] = vocab.pop(old)
    if d["model"].get("unk_token") == "<unk>":
        d["model"]["unk_token"] = "[UNK]"
    for at in d.get("added_tokens", []):
        if at["content"] == "<pad>":
            at["content"] = "[PAD]"
        elif at["content"] == "<unk>":
            at["content"] = "[UNK]"
    return Tokenizer.from_str(json.dumps(d))


def build(input_spm: Path, output_dir: Path) -> PreTrainedTokenizerFast:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Slow spm tokenizer used purely as the conversion vehicle. do_lower_case=False
    # and split_by_punct=False keep it a faithful pass-through to the spm model
    # (only the spm's own nmt_nfkc normalization is applied — no lowercasing or
    # accent stripping).
    slow = DebertaV2Tokenizer(vocab_file=str(input_spm), do_lower_case=False, split_by_punct=False)
    fast_obj = _rename_specials_in_fast(convert_slow_tokenizer(slow))

    fast = PreTrainedTokenizerFast(
        tokenizer_object=fast_obj,
        bos_token=SPECIALS["cls"][0],
        eos_token=SPECIALS["sep"][0],
        unk_token=SPECIALS["unk"][0],
        sep_token=SPECIALS["sep"][0],
        pad_token=SPECIALS["pad"][0],
        cls_token=SPECIALS["cls"][0],
        mask_token=SPECIALS["mask"][0],
        model_max_length=1024,
    )
    # Also save a special-renamed spiece.model alongside, for provenance / slow fallback.
    rename_spm_specials(input_spm, output_dir / "spiece.model")
    fast.save_pretrained(str(output_dir))
    return fast


def verify(input_spm: Path, output_dir: Path) -> None:
    """Assert the built tokenizer is faithful and correctly configured."""
    sp = spm.SentencePieceProcessor(model_file=str(input_spm))
    fast = PreTrainedTokenizerFast.from_pretrained(str(output_dir))

    # 1. special-token ids match the canonical clean scheme
    for name, (tok, tid) in SPECIALS.items():
        got = fast.convert_tokens_to_ids(tok)
        assert got == tid, f"{tok} expected id {tid}, got {got}"
    assert (fast.pad_token_id, fast.unk_token_id, fast.cls_token_id, fast.sep_token_id, fast.mask_token_id) == (
        0, 1, 2, 3, 4,
    ), "special-token id attributes mismatch"

    # 2. vocab size is the clean 100000 (no duplicate unk/pad)
    assert len(fast) == sp.get_piece_size() == 100000, f"vocab size {len(fast)} != 100000"

    # 3. fast ids reproduce the raw spm ids exactly (core text, no specials)
    for t in _CHECK_TEXTS:
        assert fast.encode(t, add_special_tokens=False) == sp.encode(t), f"id mismatch on: {t!r}"

    # 4. specials are added as [CLS] ... [SEP] and decode round-trips the letters
    for t in _CHECK_TEXTS:
        ids = fast.encode(t, add_special_tokens=True)
        assert ids[0] == 2 and ids[-1] == 3, f"missing [CLS]/[SEP] wrap on: {t!r}"
        dec = fast.decode(fast.encode(t, add_special_tokens=False), skip_special_tokens=True)
        assert dec.replace(" ", "") in t.replace(" ", "") or t.replace(" ", "") in dec.replace(" ", ""), (
            f"round-trip drift on: {t!r} -> {dec!r}"
        )

    print("VERIFY PASS")
    print(f"  vocab (real pieces) : {len(fast)}  (model pads to 100032 at build via <DUMMY_i>)")
    print(f"  special ids         : [PAD]=0 [UNK]=1 [CLS]=2 [SEP]=3 [MASK]=4")
    print(f"  fidelity            : fast ids == raw spm ids on {len(_CHECK_TEXTS)} samples")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-spm", type=Path, default=Path("tokenizer/spiece.model"),
                    help="trained SentencePiece model (default: tokenizer/spiece.model)")
    ap.add_argument("--output-dir", type=Path, default=Path("tokenizer/v3_clean"),
                    help="output dir for the HF fast tokenizer (default: tokenizer/v3_clean)")
    args = ap.parse_args()

    print(f"Building HF fast tokenizer from {args.input_spm} -> {args.output_dir}")
    build(args.input_spm, args.output_dir)
    # verify against the renamed spm in the output dir (encoding is identical to the source)
    verify(args.output_dir / "spiece.model", args.output_dir)


if __name__ == "__main__":
    main()
