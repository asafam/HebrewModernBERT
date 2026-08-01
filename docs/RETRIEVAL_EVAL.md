# HebrewModernBERT — Retrieval Evaluation (BeIR Hebrew, NDCG@10)

Status: **weak-backbone diagnosis (2026-06-29) → retrain in progress; Stage 1 masking-ratio A/B decided 20% (2026-07-30)**, see "Retrain plan" section below.
All models below go through the **identical** dual-encoder SFT + BeIR eval pipeline
(`../hebrew_text_retrieval`), so numbers are apples-to-apples. `vsRand` = lift over a
randomly-initialized backbone given the same SFT — the cleanest measure of backbone value.

## Progress tracker (update at each stage gate)

| | avg NDCG@10 | vs. current baseline (B′=0.1601) |
|---|---|---|
| mE5-large (baseline) | 0.382 | +139% |
| NeoDictaBERT (baseline) | 0.336 | +110% |
| mE5-base (baseline) | 0.333 | +108% |
| **Target (1.5×B′, ≥50% relative gain)** | **≥0.240** | **+50%** |
| **HMB — currently shipped** (`base-final`, locked plain-positive protocol) | **0.1601 (B′)** | baseline |
| HMB retrain — Stage 1 checkpoint (arm1/20%-masking, ba76000, ~44% through phase-0 only) | 0.1268 | -21% (expected — unfinished checkpoint, not a regression) |
| Random-init control | 0.089 | -44% |

Last updated 2026-08-01. The retrain row is a **mid-training snapshot**, not the retrain's
result — phase-0 is ~72% done as of this update, phases 1-2 haven't started. Re-measure at
each stage gate (post-phase-0, post-phase-1, final) and update this table each time so
progress is tracked over the life of the retrain, not just at the end.

| Model | arguana | fiqa | nfcorpus | scidocs | scifact | AVG | vsRand |
|---|---|---|---|---|---|---|---|
| mE5-large (baseline) | 0.327 | 0.382 | 0.385 | 0.232 | 0.584 | **0.382** | +0.293 |
| NeoDictaBERT (baseline, HN-SFT) | 0.350 | 0.324 | 0.350 | 0.157 | 0.501 | **0.336** | +0.247 |
| mE5-base (baseline) | 0.272 | 0.288 | 0.335 | 0.217 | 0.553 | **0.333** | +0.244 |
| NeoDictaBERT mean-pool | 0.346 | 0.325 | 0.345 | 0.154 | 0.485 | **0.331** | +0.242 |
| OLD HMB 100K (HN-cls) | 0.087 | 0.082 | 0.271 | 0.066 | 0.341 | **0.169** | +0.080 |
| OLD HMB 100K (mean-pool) | 0.091 | 0.084 | 0.267 | 0.063 | 0.297 | **0.160** | +0.072 |
| **NEW HMB 150K (mean)** | 0.022 | 0.085 | 0.277 | 0.058 | 0.313 | **0.151** | +0.062 |
| **NEW HMB 150K (cls)** | 0.022 | 0.074 | 0.266 | 0.061 | 0.313 | **0.147** | +0.058 |
| NEW HMB ba76000 (mean, intermediate) | 0.011 | 0.083 | 0.267 | 0.048 | 0.297 | 0.141 | +0.052 |
| NEW HMB ba48000 (HN, intermediate) | 0.004 | 0.036 | 0.258 | 0.050 | 0.260 | 0.122 | +0.033 |
| RANDOM init control | 0.001 | 0.007 | 0.205 | 0.006 | 0.226 | 0.089 | +0.000 |

## Findings

1. **The conversion is NOT the problem.** Re-converting phase-2 gave 137/137 bit-identical
   tensors; fill-mask produces sensible Hebrew. (Two minor config bugs — `local_rope_theta`,
   `mask_token_id` — found and fixed in `src/convert_to_hf.py`; measured impact small.)
2. **The backbone is weak.** A *random* backbone reaches 0.089 through this SFT (lexical
   fitting on nfcorpus/scifact). The new HMB lifts only **+0.058–0.062** over random, vs
   **+0.24–0.29** for every usable Hebrew/multilingual baseline — HMB captures ~20–24% of the
   backbone value a usable model delivers.
3. **The new 150K model regressed vs the old 100K model** (0.151 vs 0.169).
4. **Pooling is not the lever** (mean 0.151 ≈ cls 0.147). Neither is the SFT recipe
   (NeoDictaBERT scores 0.336 through the *same* pipeline → SFT is adequate).
5. **Data-constrained, not compute-constrained.** ~30B unique Hebrew, already seen ~6× —
   past the ~4-epoch data-scaling knee. More tokens/epochs over the same data won't help.
   See the retrain-decision plan; the discriminating de-risk is the phase-0.2 (pre-anneal)
   checkpoint eval.

## Eval pipeline fixes made (sibling repo `hebrew_text_retrieval`, branch `fix/beir-jsonl-eval-pipeline`)

- **JSONL qrels**: corpora ship `qrels/*.jsonl` (`{query-id, corpus-id, score}`); loaders
  globbed/parsed only `.tsv` → eval crashed + SFT saw empty dataset. Fixed in
  `train_dual_encoder.py` and `eval_beir_retrieval_zeroshot.py`.
- **`bert-b200` deps**: added `python-dotenv`, `faiss-gpu`, `scikit-learn`,
  `sentence-transformers`.
- **Stale embedding cache**: the eval cached query/doc embeddings under the model-label dir
  and reused them by label only → a mean-pool eval silently returned the CLS model's 6h-old
  embeddings. Added automatic invalidation (`_model_mtime`): re-encode when the model is
  newer than the cache.

## Retrain plan (2026-07 overhaul): status and Stage 1 A/B masking-ratio test

Following the weak-backbone diagnosis above, a full retrain was scoped around two identified
culprits: (1) not enough quality Hebrew data, (2) the phase-1 context-extension bug (dense
packing let tokens attend across document boundaries, collapsing retrieval — the pre-damage
phase-0.2 checkpoint scored 0.165, *above* the final 0.147/0.151). Fixes applied before any
new training:

- **FineWeb-2 folded in**: ~9.4B net-new tokens (deduped vs MAFAT, 98.8% net-new) added to a
  rebuilt, quality-reweighted corpus (`data/hebrewmodernbert/hebrew_quality`, ~131.1B tokens
  total across 10 sources, each upsampled to its own designed epoch target — curated ~5×,
  clean web ~3-4×, NLI only ~0.7×). Unique Hebrew now ~39B (was ~30B).
- **Sequence packing fixed** for phase-1/2 (`train_loader.sequence_packing: true` — proper
  per-document `cu_seqlens`, no cross-doc attention).
- **Phase-0 rebuilt as unified Hebrew-only pretrain** (drops the old mixed-then-Hebrew split
  entirely) — validated against the *exact* checkpoint this doc benchmarks against:
  `dicta-il/NeoDictaBERT`'s own model card confirms it was "pretrained only on Hebrew," no
  mixed-language phase either. The bilingual variant (`dicta-il/neodictabert-bilingual`,
  612B tok, 60/40 En-He) is a different model, not what's in the table above.
- **SFT protocol relocked (Stage 0b, 2026-07-22)**: plain positives beat hard negatives on the
  shipped model, same pattern as NeoDictaBERT's own hard-neg regression (0.336→0.294, row
  above). `hebrewmodernbert-base-final-plain-cls` scores **0.1601** avg NDCG@10 vs **0.1470**
  hard-neg — plain positives (`--dataset_name beir_hebrew`) is now the fixed protocol for
  every subsequent eval in this plan, recalibrating the retrain's target to **1.5×0.1601 ≈
  0.240** (not the nominal 0.227 off the old hard-neg baseline).

### Stage 1: masking ratio A/B (2026-07-30)

NeoDictaBERT uses 20% masking vs. HMB's historical 30% (Wettig et al.: 20% optimal for
base-size models, 40% for large — NeoDictaBERT's own ablation found 20% cost them -0.7% GLUE
at their test scale, a bet that paid off at larger scale/data). Tested empirically rather than
assumed: two phase-0 arms (`phase-0-ctrl-m30`, `phase-0-arm1-m20`), same seed (17) → identical
documents/token order, differing only in `mlm_probability`. Compared at the matched checkpoint
`ba76000` via the locked plain-positive discriminator:

| Arm | arguana | fiqa | nfcorpus | scidocs | scifact | **avg NDCG@10** |
|---|---|---|---|---|---|---|
| Control (30% masking) | 0.006 | 0.045 | 0.219 | 0.016 | 0.169 | **0.0910** |
| **Arm 1 (20% masking)** | 0.051 | 0.083 | 0.251 | 0.035 | 0.214 | **0.1268** |

**+0.036 gap (+39% relative)** — well above the ~0.01-0.02 SFT-eval noise band, a decisive
result. **20% masking wins.** Control was stopped; `mlm_probability: 0.2` is now set across
phase-0/1/2 YAMLs (was 0.3). Arm 1 continues alone toward the full 130B-token phase-0, then
phase-1 (40B, packed) → phase-2 (anneal), each against the same locked discriminator, before
final verification against the recalibrated ≥0.240 target.

## Reproduce the table

`python3 /tmp/build_beir_table.py` (reads each model's `results.json` under
`outputs/eval/beir_zeroshot/<label>/BeIR_*/`). Open gap: **arguana + scidocs have no
`hard_negatives_train.jsonl`**, so they are zero-shot for *all* models (not an HMB-specific
penalty) — mining hard negatives for them would complete the average.
