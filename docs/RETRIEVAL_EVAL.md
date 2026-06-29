# HebrewModernBERT — Retrieval Evaluation (BeIR Hebrew, NDCG@10)

Status: **base model evaluated; diagnosis = weak backbone** (2026-06-29).
All models below go through the **identical** dual-encoder SFT + BeIR eval pipeline
(`../hebrew_text_retrieval`), so numbers are apples-to-apples. `vsRand` = lift over a
randomly-initialized backbone given the same SFT — the cleanest measure of backbone value.

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

## Reproduce the table

`python3 /tmp/build_beir_table.py` (reads each model's `results.json` under
`outputs/eval/beir_zeroshot/<label>/BeIR_*/`). Open gap: **arguana + scidocs have no
`hard_negatives_train.jsonl`**, so they are zero-shot for *all* models (not an HMB-specific
penalty) — mining hard negatives for them would complete the average.
