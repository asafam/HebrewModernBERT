"""Cross-tokenizer MLM comparison: HebrewModernBERT (every training phase) vs NeoDictaBERT.

WHY THIS EXISTS
---------------
Our BeIR retrieval score is ~0.185 vs NeoDictaBERT's 0.332 through the *identical* SFT+eval
pipeline. Two very different causes produce that, and they imply opposite next moves:

  (a) our backbone genuinely models Hebrew worse  -> fix with data/recipe/more pretraining
  (b) our backbone models Hebrew comparably well, but its representations don't transfer to
      retrieval                                   -> fix with architecture/objective/pooling;
                                                     more pretraining would be wasted compute

In-training MaskedAccuracy cannot settle this: each run measured a different validation split,
its own tokenizer and its own masking rate, so 65.98% (phase-0) vs NeoDictaBERT's published
numbers are not comparable quantities.

This script makes them comparable by scoring WHOLE-WORD recovery on identical text: mask a
whole word, let each model fill every token of it under its own tokenization, and check whether
the decoded string equals the original word. The unit of measurement is a Hebrew word, which is
tokenizer-independent -- so a 150K and a 128K vocab can be compared directly.

Fairness note: measured 2026-08-11, the two tokenizers fragment this corpus near-identically
(1.414 vs 1.408 tokens/word, ratio 1.00x), so per-token numbers are also roughly fair here and
are reported as a secondary metric. That is a property of this corpus pair, not a general one.

Eval text is the translated Hebrew BeIR corpora -- the actual retrieval target domain, and
out-of-domain for both models (neither trained on translated text), so neither is favored.
"""
import argparse, glob, json, os, random, re, sys
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

BEIR_ROOT = ("/home/nlp/achimoa/workspace/hebrew_text_retrieval/outputs/translation/runs/"
             "full_corpus_zeroshot_nocontext_gemini31flashlite_promptv20260531/corpus")
HF = "/home/nlp/achimoa/workspace/HebrewModernBERT/outputs/hf"

# label -> path. Ordered as the training story: floor, the Stage-1 A/B, then the phase chain.
MODELS = {
    "NeoDictaBERT":            "dicta-il/NeoDictaBERT",
    "HMB random-init":         f"{HF}/HebrewModernBERT-base-random",
    "HMB ctrl-m30 @ba76000":   f"{HF}/HebrewModernBERT-base-phase0-ctrl-m30-ba76000",
    "HMB arm1-m20 @ba76000":   f"{HF}/HebrewModernBERT-base-phase0-arm1-m20-ba76000",
    "HMB phase-0 FINAL":       f"{HF}/HebrewModernBERT-base-phase0-arm1-m20-final",
    "HMB retrain FINAL (p2)":  f"{HF}/HebrewModernBERT-base-retrain-final",
    "HMB OLD shipped":         f"{HF}/HebrewModernBERT-base-final",
}

HEB = re.compile(r"[֐-׿]")

# Surface-form noise that differs BETWEEN tokenizers and must not count as a model error:
# WordPiece continuation marks (NeoDictaBERT: 'בייל ##וד'), the SentencePiece word marker
# (HMB: '▁'), and whitespace decode inserts around punctuation ('מיקרו - מבנית').
# Comparing normalized forms is what makes a 128K WordPiece and a 150K SentencePiece model
# answerable on the same question.
_NOISE = re.compile(r"\s+|##|▁")


def norm(s):
    return _NOISE.sub("", s)


def load_docs(n_per_ds, seed=17):
    """Sample docs per BeIR dataset. Returns {dataset: [text, ...]}."""
    out = {}
    for path in sorted(glob.glob(f"{BEIR_ROOT}/BeIR_*/beir/corpus.jsonl")):
        ds = path.split("/BeIR_")[1].split("/")[0]
        docs = []
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                txt = ((d.get("title") or "") + " " + (d.get("text") or "")).strip()
                # need enough words to mask one and leave real context
                if txt and len(txt.split()) >= 25:
                    docs.append(txt)
                if len(docs) >= n_per_ds * 3:
                    break
        random.Random(seed).shuffle(docs)
        out[ds] = docs[:n_per_ds]
    return out


def build_doc_freq(docs_by_ds):
    """Document frequency per whitespace word, over the SAME text both models see.

    Tokenizer-independent by construction, so the "is this word rare" decision is identical
    for every model -- the candidate set must not depend on whose vocabulary is being tested.
    """
    from collections import Counter
    df = Counter()
    ndocs = 0
    for docs in docs_by_ds.values():
        for t in docs:
            ndocs += 1
            df.update(set(t.split()))
    return df, ndocs


def pick_words(text, k, rng, span_len=1, df=None, rare_max_df=None):
    """Choose up to k NON-OVERLAPPING targets, each covering `span_len` consecutive words.

    span_len > 1 is the hard mode: masking a contiguous run removes the local context
    (morphology, agreement, collocation) that lets a small model recover an isolated word,
    so the middle of the run can only be filled from sentence-level meaning.

    rare_max_df restricts targets to words at or below that document frequency -- i.e. drops
    function words and boilerplate, keeping content words that actually carry topic.
    """
    toks = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]

    def ok(i):
        w = text[toks[i][0]:toks[i][1]]
        if len(w) < 3 or len(HEB.findall(w)) < 3:
            return False
        if rare_max_df is not None and df is not None and df.get(w, 0) > rare_max_df:
            return False
        return True

    starts = [i for i in range(len(toks) - span_len + 1)
              if all(ok(j) for j in range(i, i + span_len))]
    if not starts:
        return []
    rng.shuffle(starts)
    chosen, used = [], set()
    for i in starts:
        idxs = range(i, i + span_len)
        if any(j in used for j in idxs):
            continue
        chosen.append((toks[i][0], toks[i + span_len - 1][1]))
        used.update(idxs)
        if len(chosen) >= k:
            break
    return chosen


@torch.no_grad()
def score_model(label, path, docs_by_ds, masks_per_doc, device, max_len=256,
                span_len=1, df=None, rare_max_df=None):
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(path, trust_remote_code=True).to(device).eval()
    if tok.mask_token_id is None:
        raise RuntimeError(f"{label}: tokenizer has no mask token")

    per_ds = {}
    for ds, docs in docs_by_ds.items():
        word_hit = word_tot = tok_hit = tok_tot = 0
        exact_hit = 0          # all tokens of the word recovered -- needs no decoding at all
        span_toks = 0          # tokens per masked word, to prove difficulty is matched
        dropped = 0            # words this tokenizer cannot round-trip -> excluded, not failed
        for di, text in enumerate(docs):
            rng = random.Random(1000 + di)
            enc = tok(text, return_offsets_mapping=True, truncation=True,
                      max_length=max_len, return_tensors="pt")
            offsets = enc.pop("offset_mapping")[0].tolist()
            ids = enc["input_ids"][0].tolist()
            # char span covered by the (possibly truncated) encoding
            covered = max((e for (s, e) in offsets), default=0)
            targets = [(s, e) for (s, e) in
                       pick_words(text, masks_per_doc, rng, span_len=span_len,
                                  df=df, rare_max_df=rare_max_df)
                       if e <= covered]
            if not targets:
                continue

            # Map each target word -> the token positions covering it.
            # OVERLAP, not containment: SentencePiece emits the leading space as part of the
            # token, so the token's start offset sits one char BEFORE the word. A containment
            # test (s >= ws) silently drops most words for that tokenizer and biases the whole
            # comparison. Then require the span to round-trip to the gold word, otherwise this
            # word is not measurable in this tokenization and is EXCLUDED (not scored as a
            # miss) -- scoring it as a miss is what makes one tokenizer look catastrophic.
            groups = []
            for (ws, we) in targets:
                pos = [i for i, (s, e) in enumerate(offsets)
                       if e > s and max(s, ws) < min(e, we)]
                if not pos:
                    dropped += 1
                    continue
                if norm(tok.decode([ids[p] for p in pos])) != norm(text[ws:we]):
                    dropped += 1
                    continue
                groups.append(((ws, we), pos))
            if not groups:
                continue

            masked = list(ids)
            for _, pos in groups:
                for p in pos:
                    masked[p] = tok.mask_token_id
            inp = {"input_ids": torch.tensor([masked], device=device),
                   "attention_mask": torch.ones(1, len(masked), dtype=torch.long, device=device)}
            pred = model(**inp).logits[0].argmax(-1).tolist()

            for (ws, we), pos in groups:
                word_tot += 1
                span_toks += len(pos)
                # (1) surface metric: decoded string matches, modulo tokenizer noise
                if norm(tok.decode([pred[p] for p in pos])) == norm(text[ws:we]):
                    word_hit += 1
                # (2) exact metric: every token id recovered -- zero decoding involved,
                #     so no tokenizer surface convention can distort it
                if all(pred[p] == ids[p] for p in pos):
                    exact_hit += 1
                for p in pos:
                    tok_tot += 1
                    if pred[p] == ids[p]:
                        tok_hit += 1

        per_ds[ds] = {
            "word_acc": 100.0 * word_hit / max(word_tot, 1),
            "exact_acc": 100.0 * exact_hit / max(word_tot, 1),
            "tok_acc": 100.0 * tok_hit / max(tok_tot, 1),
            "n_words": word_tot,
            "toks_per_word": span_toks / max(word_tot, 1),
            "dropped_pct": 100.0 * dropped / max(dropped + word_tot, 1),
        }
        d = per_ds[ds]
        print(f"    {label:24} {ds:9} word={d['word_acc']:5.1f}%  exact={d['exact_acc']:5.1f}%  "
              f"tok={d['tok_acc']:5.1f}%  (n={word_tot}, {d['toks_per_word']:.2f} tok/word, "
              f"{d['dropped_pct']:.0f}% dropped)", flush=True)

    del model
    torch.cuda.empty_cache()
    return per_ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs_per_dataset", type=int, default=150)
    ap.add_argument("--masks_per_doc", type=int, default=4)
    ap.add_argument("--out", default="outputs/eval/mlm_vs_neodictabert.json")
    # Added for the LARGE run: score extra checkpoints (e.g. the tile-init step-0 gate, or a
    # mid-run large checkpoint) without editing the MODELS dict above for every gate.
    ap.add_argument("--extra-model", action="append", default=[], metavar="LABEL=PATH",
                    help="Additional model to score, repeatable. e.g. "
                         "--extra-model 'HMB large step0=outputs/hf/HebrewModernBERT-large-step0'")
    ap.add_argument("--only", default=None, metavar="LABEL[,LABEL]",
                    help="Score only these labels (substring match) -- keeps a quick gate cheap.")
    # HARD MODE (2026-08-11). The default single-isolated-word test showed HMB (110M
    # non-embedding body) at parity with NeoDictaBERT (~265M, 2.4x larger, 28 layers vs 22,
    # FFN 3072 vs 1152, full attention vs our 128-token local window on 2/3 of layers). That
    # parity is suspected to be a CEILING EFFECT: recovering one masked word from full context
    # is a local task -- morphology, agreement and collocation pin it down -- and a base-size
    # model saturates it. These two flags remove the local shortcuts. If the larger model pulls
    # ahead here, the parity was an artifact and capacity is the real retrieval gap; if parity
    # survives a genuinely hard task, capacity is not the story.
    ap.add_argument("--span_len", type=int, default=1,
                    help="Mask this many CONSECUTIVE words per target (1 = original test). "
                         "5 makes the middle of the span unrecoverable from local cues.")
    ap.add_argument("--rare_pct", type=float, default=100.0,
                    help="Only mask words in the rarest N%% by document frequency over the eval "
                         "text (100 = no filter). 50 drops function words and boilerplate, "
                         "keeping content words. Computed tokenizer-independently, so every "
                         "model gets the identical candidate set.")
    args = ap.parse_args()

    models = dict(MODELS)
    for spec in args.extra_model:
        if "=" not in spec:
            ap.error(f"--extra-model expects LABEL=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        models[label.strip()] = path.strip()
    if args.only:
        wanted = [w.strip() for w in args.only.split(",") if w.strip()]
        models = {k: v for k, v in models.items() if any(w in k for w in wanted)}
        if not models:
            ap.error(f"--only {args.only!r} matched no labels; available: {list(MODELS)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hard = args.span_len > 1 or args.rare_pct < 100.0
    print(f"device={device}  docs/ds={args.docs_per_dataset}  masks/doc={args.masks_per_doc}  "
          f"span_len={args.span_len}  rare_pct={args.rare_pct}  "
          f"MODE={'HARD' if hard else 'original'}\n")

    docs_by_ds = load_docs(args.docs_per_dataset)
    for ds, d in docs_by_ds.items():
        print(f"  {ds:10} {len(d)} docs")
    print()

    # Rarity threshold: computed once, from the same text, and reused for every model.
    df, ndocs = build_doc_freq(docs_by_ds)
    rare_max_df = None
    if args.rare_pct < 100.0:
        import statistics
        elig = [w for w in df if len(w) >= 3 and len(HEB.findall(w)) >= 3]
        counts = sorted(df[w] for w in elig)
        idx = max(0, min(len(counts) - 1, int(len(counts) * args.rare_pct / 100.0) - 1))
        rare_max_df = counts[idx]
        keep = sum(1 for w in elig if df[w] <= rare_max_df)
        print(f"  rarity filter: df<={rare_max_df} over {ndocs} docs -> "
              f"{keep}/{len(elig)} eligible word types kept ({100.0*keep/max(len(elig),1):.0f}%)\n")

    results = {}
    for label, path in models.items():
        if not path.startswith("dicta-il") and not os.path.isdir(path):
            print(f"  SKIP {label}: {path} not found")
            continue
        print(f"  scoring {label} ...", flush=True)
        try:
            results[label] = score_model(label, path, docs_by_ds,
                                         args.masks_per_doc, device,
                                         span_len=args.span_len, df=df,
                                         rare_max_df=rare_max_df)
        except Exception as exc:  # one bad checkpoint must not kill the sweep
            print(f"  FAILED {label}: {type(exc).__name__}: {exc}")

    datasets = sorted(next(iter(results.values())).keys()) if results else []
    hdr = f"{'model':24} " + " ".join(f"{d:>9}" for d in datasets) + f" {'MEAN':>9}"

    def table(title, key):
        print(f"\n=== {title} ===")
        print(hdr); print("-" * len(hdr))
        for label in models:
            if label not in results:
                continue
            r = results[label]
            mean = sum(r[d][key] for d in datasets) / len(datasets)
            print(f"{label:24} " + " ".join(f"{r[d][key]:9.1f}" for d in datasets) + f" {mean:9.1f}")

    table("EXACT whole-word recovery (%) — PRIMARY, no decoding, tokenizer-surface-proof", "exact_acc")
    table("whole-word recovery by decoded string (%) — normalized, should track EXACT closely", "word_acc")
    table("per-TOKEN masked accuracy (%) — secondary", "tok_acc")

    # Sanity columns. A previous version of this script silently dropped most words for the
    # SentencePiece tokenizer (containment-vs-overlap offset bug) and scored the survivors,
    # producing 78% token accuracy alongside 2% word accuracy. Any large asymmetry in
    # 'dropped' or 'tok/word' between models means the comparison is NOT apples-to-apples --
    # check these before believing the tables above.
    print("\n=== SANITY (must be comparable across models, else the comparison is invalid) ===")
    print(f"{'model':24} {'words scored':>13} {'tok/word':>10} {'dropped %':>10}")
    print("-" * 60)
    for label in models:
        if label not in results:
            continue
        r = results[label]
        n = sum(r[d]["n_words"] for d in datasets)
        tpw = sum(r[d]["toks_per_word"] for d in datasets) / len(datasets)
        dp = sum(r[d]["dropped_pct"] for d in datasets) / len(datasets)
        flag = "   <-- OUTLIER" if dp > 40 else ""
        print(f"{label:24} {n:13} {tpw:10.2f} {dp:9.0f}%{flag}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {args.out}")

    print("""
HOW TO READ THIS
  If HMB retrain-FINAL is CLOSE to NeoDictaBERT on word accuracy, the backbone models Hebrew
  fine and the 0.185-vs-0.332 retrieval gap is a representation/transfer problem -> more
  pretraining is the wrong lever; change architecture/objective/pooling instead.
  If HMB is well BELOW NeoDictaBERT, the backbone is genuinely weaker -> data/recipe/pretraining
  remain the lever, and the retrain-v3 case is much stronger.""")


if __name__ == "__main__":
    sys.exit(main())
