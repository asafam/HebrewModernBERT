# HebrewModernBERT — Retrieval Evaluation (BeIR Hebrew, NDCG@10)

Status: **RETRAIN COMPLETE (2026-08-08) — real improvement (+28.6% vs. shipped), but well short
of the actual target of 0.300 (user-clarified 2026-08-08 — supersedes the earlier 1.5×B′≈0.240
framing, which was this doc's own derived interpretation of "50% more in points," not the
user's real bar).**
See "Retrain plan" section below for the full history (diagnosis → fixes → Stage 0/1 → final result).
All models below go through the **identical** dual-encoder SFT + BeIR eval pipeline
(`../hebrew_text_retrieval`), so numbers are apples-to-apples. `vsRand` = lift over a
randomly-initialized backbone given the same SFT — the cleanest measure of backbone value.

> **Correction (2026-08-10/11):** every row evaluated **before** the 2026-07-30 eval bugfix
> (IDCG truncation + ArguAna self-retrieval) was previously read from each `results.json`'s
> stale `metrics_pre_fix` field instead of the corrected `metrics` field. That affected **both**
> the baselines (mE5-large/base, NeoDictaBERT ×2, random-init) **and** every pre-2026-07-30 HMB
> row (old 100K, new 150K cls/mean, ba76000/ba48000 intermediates) — while the post-fix HMB
> retrain rows were already correct, making the whole table apples-to-oranges. All rows are now
> read from `metrics` and match `../hebrew_text_retrieval/docs/benchmark/results.md`; every
> derived percentage, gap, and `vsRand` has been recomputed. Directionally nothing flips (the
> 150K-vs-100K regression still holds, the backbone is still weak), but the magnitudes move —
> the ArguAna fix in particular shifts HMB arguana scores materially.

## Progress tracker (final)

| | avg NDCG@10 | vs. current baseline (B′=0.144) | vs. TARGET (0.300) |
|---|---|---|---|
| mE5-large (baseline) | 0.358 | +148.6% | past target |
| NeoDictaBERT (baseline) | 0.332 | +130.6% | past target |
| mE5-base (baseline) | 0.305 | +111.8% | past target |
| **TARGET (user-specified, 2026-08-08)** | **0.300** | +108.3% | — |
| ~~Derived target (1.5×B′, ≥50% relative gain)~~ superseded | ~~≥0.240~~ | ~~+50%~~ | — |
| ~~HMB retrain-FINAL + MEAN pooling~~ **RETRACTED — corpus-confounded** (mean@v20260801 vs cls@v20260531; see the retraction box below) | ~~0.1965~~ | — | — |
| **HMB retrain — FINAL** (`HebrewModernBERT-base-retrain-final`, phase-0+1+2 complete, cls, mean of 2 replicated runs) | **~0.185** (0.1843 / 0.1862) | **+28.6%** | **-38.3% (gap: 0.115)** |
| HMB — currently shipped (`base-final`, locked plain-positive protocol) | 0.144 (B′) | baseline | -52.0% |
| HMB — old 100K-tokenizer model | 0.158 (HN) / 0.149 (mean) | +9.7% (HN) | -47.4% (HN) |
| HMB retrain — phase-0 only (full 130B tokens, before ctx-ext/anneal) | 0.1379 | -4.2% | -54.0% |
| HMB retrain — Stage 1 checkpoint (ba76000, ~44% through phase-0) | 0.1268 | -11.9% | -57.7% |
| Random-init control | 0.084 | -41.7% | -72.0% |

**Result, last updated 2026-08-08:** the finished retrain (all 3 phases) scores **~0.185** avg
NDCG@10, replicated across 2 independent SFT+eval runs (0.1843 / 0.1862 — spread ~0.002, a
stable result, not noise). That's a **real +28.6% improvement** over the currently shipped
model, and it also beats the old 100K-tokenizer model (0.158), fixing the regression that
partly motivated this retrain. **Against the actual target of 0.300, though, this leaves a
substantial gap (0.115 absolute, ~62% relative improvement still needed from here)** — closing
it would require getting most of the way to NeoDictaBERT/mE5-base territory (0.305-0.332), a
materially bigger ask than the originally-tracked "50% relative" framing implied. Progression through the
curriculum: phase-0 alone 0.1379 → full retrain 0.185, so phase-1 (ctx-ext, now non-damaging)
+ phase-2 (rebalanced anneal) together added +0.047 (+34% relative) on top of phase-0 alone —
consistent with the historical precedent that these phases should help once the packing bug
is fixed, not hurt. Candidate follow-ups if pursuing the remaining gap: RMSNorm (accepting its
HF-export cost), the masking-decay-schedule stretch arm (Stage 1.5, never built), further data
growth beyond FineWeb-2, or architecture changes beyond this round's scope.

All values below are post-fix (`metrics`). `vsRand` is computed against the post-fix
random-init control (0.084).

| Model | arguana | fiqa | nfcorpus | scidocs | scifact | AVG | vsRand |
|---|---|---|---|---|---|---|---|
| mE5-large (baseline) | 0.440 | 0.335 | 0.294 | 0.139 | 0.581 | **0.358** | +0.274 |
| NeoDictaBERT (baseline, HN-SFT) | 0.451 | 0.288 | 0.329 | 0.093 | 0.501 | **0.332** | +0.248 |
| NeoDictaBERT mean-pool | 0.456 | 0.285 | 0.327 | 0.092 | 0.484 | **0.329** | +0.245 |
| mE5-base (baseline) | 0.361 | 0.241 | 0.248 | 0.125 | 0.549 | **0.305** | +0.221 |
| **HMB RETRAIN-FINAL** (mean of run1/run2) | 0.229 | 0.115 | 0.275 | 0.043 | 0.265 | **0.185** | +0.101 |
| OLD HMB 100K (HN-cls) | 0.101 | 0.061 | 0.253 | 0.035 | 0.340 | **0.158** | +0.074 |
| OLD HMB 100K (mean-pool) | 0.105 | 0.063 | 0.249 | 0.035 | 0.295 | **0.149** | +0.065 |
| HMB shipped `base-final-plain-cls` (B′) | 0.063 | 0.114 | 0.262 | 0.044 | 0.238 | **0.144** | +0.061 |
| HMB retrain phase-0 only (130B tok) | 0.074 | 0.117 | 0.231 | 0.043 | 0.224 | **0.138** | +0.054 |
| **NEW HMB 150K (mean)** | 0.023 | 0.060 | 0.254 | 0.029 | 0.312 | **0.136** | +0.052 |
| **NEW HMB 150K (cls)** | 0.023 | 0.055 | 0.247 | 0.031 | 0.312 | **0.134** | +0.050 |
| NEW HMB ba76000 (mean, intermediate) | 0.012 | 0.062 | 0.244 | 0.025 | 0.295 | 0.128 | +0.044 |
| HMB retrain Stage-1 arm1 @ba76000 | 0.051 | 0.083 | 0.251 | 0.035 | 0.214 | **0.127** | +0.043 |
| NEW HMB ba48000 (HN, intermediate) | 0.005 | 0.024 | 0.237 | 0.023 | 0.258 | 0.109 | +0.026 |
| RANDOM init control | 0.001 | 0.004 | 0.190 | 0.002 | 0.223 | 0.084 | +0.000 |

## Findings

1. **The conversion is NOT the problem.** Re-converting phase-2 gave 137/137 bit-identical
   tensors; fill-mask produces sensible Hebrew. (Two minor config bugs — `local_rope_theta`,
   `mask_token_id` — found and fixed in `src/convert_to_hf.py`; measured impact small.)
2. **The backbone is weak.** A *random* backbone reaches 0.084 through this SFT (lexical
   fitting on nfcorpus/scifact). The new HMB lifted only **+0.050–0.052** over random, vs
   **+0.22–0.27** for every usable Hebrew/multilingual baseline — HMB captured ~18–23% of the
   backbone value a usable model delivers. The finished retrain improves this to +0.101
   (~37–46% of baseline backbone value) — better, still not competitive.
3. **The new 150K model regressed vs the old 100K model** (0.136 vs 0.158 post-fix; the
   regression holds under corrected metrics, and was the motivation for the retrain).
4. **Pooling is not the lever** (mean 0.136 ≈ cls 0.134) — measured on the OLD pre-packing-fix
   model, so it does not necessarily transfer to retrain-final. An attempt to re-test this on
   2026-08-11 produced a corpus-confounded result and was retracted; **the question is currently
   OPEN**, pending corpus-matched numbers (job 21316622). Also still true: the SFT recipe is not
   the lever (NeoDictaBERT scores 0.332 through the *same* pipeline → SFT is adequate).
5. ***(Superseded — this finding described the pre-retrain state. FineWeb-2 has since taken
   unique Hebrew to ~39B and the retrain re-ran the whole curriculum on it; see the retrain-plan
   section below. Kept for the record.)*** **Data-constrained, not compute-constrained.**
   ~30B unique Hebrew, already seen ~6× —
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
phase-0.2 checkpoint scored 0.165, *above* the final 0.147/0.151 — all three pre-fix figures,
kept as the historical record of the decision). Fixes applied before any
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
  *(Note, 2026-08-10: 0.1601/0.1470 here are pre-metric-fix values from before the 2026-07-30
  eval bugfix; the corrected shipped-model score is 0.144 — see the correction note above. This
  bullet is left as the historical record of the protocol decision; the plain-vs-hard-neg
  comparison itself was not re-run post-fix.)*

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

~~**+0.036 gap (+39% relative)** — well above the ~0.01-0.02 SFT-eval noise band, a decisive
result. **20% masking wins.**~~
**⚠️ DOWNGRADED 2026-08-11:** the true seed-to-seed SFT noise band is ~0.025 (sigma ~0.0245),
not 0.01-0.02 — see the noise-band section below. This +0.0358 gap is **1.5 sigma from a single
run per arm**, needing ~4 runs/arm to call at 2 sigma. **It is suggestive, not decisive.** 20%
masking may still be correct (it matches NeoDictaBERT and Wettig et al.), but this experiment
did not establish it — and the whole retrain was built on it. Control was stopped; `mlm_probability: 0.2` is now set across
phase-0/1/2 YAMLs (was 0.3). Arm 1 continues alone toward the full 130B-token phase-0, then
phase-1 (40B, packed) → phase-2 (anneal), each against the same locked discriminator, before
final verification against the recalibrated ≥0.240 target.

## Where the Stage-1 masking win went (analysis, 2026-08-11)

The Stage-1 A/B looked like a ~2× result but the final model only reached 0.185. Tracing it
per-dataset (all post-fix, both arms evaluated after the fix — this comparison is *not*
contaminated):

**The "2×" was real, but never on the average.** At the matched checkpoint `ba76000`:

| | arguana | fiqa | nfcorpus | scidocs | scifact | AVG |
|---|---|---|---|---|---|---|
| ctrl (30% mask) | 0.0056 | 0.0448 | 0.2193 | 0.0161 | 0.1693 | 0.0910 |
| arm1 (20% mask) | 0.0514 | 0.0833 | 0.2512 | 0.0345 | 0.2135 | 0.1268 |
| **arm1/ctrl** | **9.24×** | **1.86×** | 1.15× | **2.15×** | 1.26× | **1.39×** |
| random-init floor | 0.0007 | 0.0039 | 0.1898 | 0.0021 | 0.2228 | 0.0839 |

The ~2× is concentrated on the three **low-floor** datasets (arguana/fiqa/scidocs, random ≈0).
nfcorpus and scifact carry the largest absolute NDCG but sit on an enormous lexical floor
(random-init alone scores 0.190 and 0.223), so they compress any backbone gain into a ~1.2×
ratio and drag the mean to 1.39×. **The average is a floor-dominated metric.** Worse: at
ba76000 *both* arms scored **below** the random floor on scifact (0.169 and 0.214 vs 0.223) —
at that stage the trained backbone was actively worse than noise there, capping how much the
A/B average could ever have shown.

**Fade 1 — the back half of phase-0 bought almost nothing.** `ba76000` (~44% of phase-0) →
phase-0 final (130B tok) gained only **+8.8%** (0.1268 → 0.1379), and nfcorpus *regressed*
−8.2%. (Verified these are genuinely different checkpoints, not an SFT re-run: distinct
`model_path` — `phase0-arm1-m20-ba76000` vs `phase0-arm1-m20-final-ba76000` — and different
`num_self_excluded`, 666 vs 430.)

**Fade 2 — phase-1+2's gain is one dataset.** phase-0 final → retrain final gained +34.3%
(0.1379 → 0.1852), but **65% of that is arguana alone** (0.074 → 0.229, +207%) — consistent
with phase-1 being context extension and arguana having long argument passages. Meanwhile
**fiqa and scidocs are frozen**: 0.1175 → 0.1147 (−2.4%) and 0.0432 → 0.0434 (+0.4%).

**So nothing "faded" — the gains landed on the floor-dominated and length-sensitive datasets,
and the semantic core never moved.** Against mE5-large we now capture: nfcorpus 93.5%
(essentially maxed — nothing left there), arguana 51.9%, scifact 45.5%, fiqa 34.2%,
scidocs 31.3%.

**What 0.300 actually requires.** Need avg 0.300 = sum 1.500; we have sum 0.926 → **+0.574**
needed. Closing scifact (+0.316) and fiqa (+0.220) to mE5-large parity alone is +0.536 —
nearly the whole requirement. **Any intervention that lifts everything a little (RMSNorm, a
masking schedule) is arithmetically insufficient.** Only something that specifically moves
semantic matching on scifact + fiqa reaches the target.

*Caveat:* the control arm was stopped at ba76000, so there is no 30%-masking checkpoint at
full phase-0. "20% masking wins" rests on that single early checkpoint and was never
re-verified at scale.

## MLM parity with NeoDictaBERT — the gap is TRANSFER, not the backbone (2026-08-11)

Decisive diagnostic. `scripts/mlm_compare_vs_neodictabert.py` (job via
`.slurm/jobs/mlm_compare_vs_ndb.slurm`) scores **whole-word recovery** on the translated BeIR
Hebrew text: mask a whole Hebrew word, let each model fill every token of it under its own
tokenization, count a hit only if every token id is recovered. The unit is a word, so a 150K
SentencePiece and a 128K WordPiece vocab are directly comparable — unlike in-training
MaskedAccuracy, which used a different split/tokenizer/mask-rate per run and was never
comparable to NeoDictaBERT at all.

| Model | arguana | fiqa | nfcorpus | scidocs | scifact | **MEAN** |
|---|---|---|---|---|---|---|
| **NeoDictaBERT** | 47.7 | 47.7 | 54.8 | 51.0 | 48.2 | **49.9** |
| **HMB retrain FINAL (phase-2)** | 47.3 | 46.6 | 52.6 | 49.6 | 47.0 | **48.6** |
| HMB phase-0 FINAL | 44.2 | 47.1 | 49.0 | 48.3 | 42.9 | 46.3 |
| HMB arm1-m20 @ba76000 | 42.2 | 43.8 | 49.0 | 47.2 | 42.9 | 45.0 |
| HMB ctrl-m30 @ba76000 | 43.7 | 42.6 | 48.6 | 44.0 | 43.5 | 44.5 |
| HMB OLD shipped | 29.9 | 32.5 | 37.8 | 33.9 | 32.2 | 33.3 |
| HMB random-init | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

Validity: all models scored 2707–2725 words at 1.27–1.32 tokens/word with **0% dropped**, and
the decoded-string and per-token metrics track the exact metric closely. Our slightly higher
tok/word (1.32 vs 1.27) makes exact-match marginally *harder* for us, so the comparison is
conservative in NeoDictaBERT's favor.

**The finding: on masked language modelling we are at parity with NeoDictaBERT — 48.6 vs 49.9,
a 1.3-point (2.6% relative) gap. On retrieval through the identical SFT pipeline we score
0.185 vs their 0.332 — 56% of their number, a 44% shortfall.** Both models are Hebrew-only
MLM encoders. Equivalent language modelling, wildly different retrieval.

**Therefore the retrieval gap is not backbone language knowledge. It is a representation-transfer
problem** — how the model's MLM representations turn into a single retrievable vector.
Consequences:

- **Retrain-v3 for backbone quality is largely wasted compute.** More data, more tokens, or a
  better pretraining recipe all target MLM ability, which is already at parity. This is
  independent evidence for the same conclusion the 0.300 decomposition reached from the other
  direction (broad small lifts can't close scifact+fiqa).
- Supporting evidence in the same table: **MLM improved far more than retrieval did.** Old
  shipped → retrain-final is +46% relative on MLM (33.3 → 48.6) but only +28.6% on retrieval
  (0.144 → 0.185). Backbone gains are not converting.
- The masking A/B barely registers here: ctrl-m30 44.5 vs arm1-m20 45.0, +0.5pt, against a
  +39% retrieval gap at the same checkpoint. Masking ratio moved retrieval on some axis other
  than language-modelling quality.
- **Resolves an open question:** in-training MaskedAccuracy *fell* at phase-2 (68.78% → 64.01%)
  and the doc flagged it as probably a validation-set artifact. Confirmed — on fixed common
  text phase-2 is the *best* HMB checkpoint (46.3 → 48.6). Phase-2 did not damage the model.

### Follow-up: pre-SFT embedding geometry — NEGATIVE result (2026-08-11)

`scripts/embedding_geometry_vs_neodictabert.py` (job `.slurm/jobs/embedding_geometry.slurm`)
measured per-layer zero-shot NDCG@10, anisotropy (mean pairwise cosine between random docs),
and query/positive alignment, for both poolings, on scifact + nfcorpus.

**Pre-SFT geometry does not predict post-SFT retrieval — do not invest further here.** Both
HMB and NeoDictaBERT have collapsed last-layer spaces (anisotropy ~0.99) and both are ~0
zero-shot (NDB 0.016, HMB 0.0014 on scifact/mean). A 10× zero-shot ratio against a 1.8×
post-SFT ratio, with both values in the noise.

The headline oddity is an artifact, not a bug: **HMB random-init scores the *highest* zero-shot
NDCG@10** (0.335 scifact / 0.116 nfcorpus, anisotropy 0.34). A random projection preserves
lexical overlap, so an untrained model acts as a fuzzy BM25 — consistent with random-init
scoring 0.223 on scifact *after* SFT. Trained models collapse that spread into a narrow cone,
which destroys raw cosine retrieval while being irrelevant to what SFT later recovers.

Two real observations did come out of it, both about our model specifically:

- **HMB's CLS space is far more collapsed than NeoDictaBERT's** (anisotropy 0.9997 vs 0.833) —
  and our locked SFT protocol trains on exactly that vector.
- **phase-1/2 flipped query/document alignment negative**: +0.914 at phase-0 → **−0.269** at
  phase-2 (anisotropy still 0.987), i.e. queries sit anti-parallel to the document cluster.
  This appeared during ctx-ext/anneal and has no counterpart in NeoDictaBERT.

Neither is actionable on its own, but together they invalidated the evidence behind Finding #4
("pooling is not the lever") *for the current checkpoint*. That prompted the test below — and
the negative geometry result still paid for itself, because the CLS-collapse observation is
what motivated it.

### ~~Mean pooling beats CLS on retrain-final — +0.0112~~ RETRACTED, corpus-confounded (2026-08-11)

Job 21316570, `scripts/model/dual_encoder/heq/train/train_eval_hmb_retrain_final_meanpool.sh`
in the sibling repo. Locked recipe verbatim (plain positives, lr 2e-5, bs 32x4, cosine,
warmup 0.05, 10 epochs, max_len 512); **only `--pooling` changed cls → mean**. Completed
02:05:13.

| | arguana | fiqa | nfcorpus | scidocs | scifact | **AVG** |
|---|---|---|---|---|---|---|
| CLS run1 (baseline) | 0.2337 | 0.1155 | 0.2683 | 0.0436 | 0.2604 | 0.1843 |
| CLS run2 (baseline) | 0.2238 | 0.1139 | 0.2813 | 0.0431 | 0.2688 | 0.1862 |
| **MEAN (new)** | **0.2713** | **0.1339** | 0.2667 | **0.0513** | 0.2592 | **0.1965** |
| **delta vs CLS mean-of-2** | **+0.0426** | **+0.0192** | −0.0081 | +0.0079 | −0.0055 | **+0.0112** |

> ## ⚠️ RETRACTED (2026-08-11, same day) — THIS COMPARISON IS CONFOUNDED
>
> **The +0.0112 below compares two different corpora, not two poolings. Do not cite it.**
>
> The cls baseline was evaluated when only translation run **v20260531** existed. The
> mean-pooling job ran after **v20260801** was exported, and the eval loop's unanchored
> `find outputs/translation/runs -name corpus.jsonl -path "*/beir/corpus.jsonl" | sort`
> matched **both**. Both passes wrote to the same results dir under the same label, and
> `0531` sorts before `0801`, so **0801 silently overwrote 0531**. Verified from the job
> log's eval order (positions 1-5 = 0531, 6-10 = 0801) and the stored file mtimes.
>
> So the table reads **mean@v20260801 vs cls@v20260531**. Worse, the bias runs in the
> direction that fakes a win: v20260801 unified the query/document translation prompt
> (v20260531 could render the same English term differently on each side — ECMO → אקמו in a
> query but ECMO in a document — breaking the lexical overlap retrieval depends on), so it
> should raise scores for *any* model. The per-dataset pattern I read as "mean pooling helps
> exactly where discrimination matters" is equally consistent with "the corpus fix helps
> exactly where lexical overlap was broken."
>
> Note the query counts and corpus sizes are **identical** across the two translation runs
> (1401/648/323/1000/300 queries; 8674/57600/3633/25313/5183 docs), so nothing in the stored
> `results.json` reveals which corpus produced it. Only the job log does. That is why this
> was not visible in the results table.
>
> Corpus-matched numbers pending: job 21316622 re-scores both cls checkpoints on the pinned
> v20260801 (eval only, no retraining), and the seed-1234 mean run already uses the pinned
> path. `scripts/model/eval/corpus_paths.sh` now pins the corpus for all callers, which is
> what stops this recurring.

**+0.0112 against a measured run-to-run spread of 0.0019 — roughly 6x the noise band.**
*(Superseded by the retraction above — the delta is not attributable to pooling.)*

The per-dataset shape matches the geometry diagnosis exactly: mean pooling gains on the three
**low-floor** datasets (arguana +0.043, fiqa +0.019, scidocs +0.008) and is flat-to-slightly-down
on the two **floor-dominated** ones (nfcorpus −0.008, scifact −0.006). CLS was costing us
precisely where discrimination, not lexical overlap, does the work — consistent with HMB's CLS
anisotropy of 0.9997 vs NeoDictaBERT's 0.833.

**Notable: fiqa moved.** It had been frozen across the entire retrain (0.1175 phase-0 →
0.1147 phase-2, −2.4%). Mean pooling puts it at 0.1339, the first movement on one of the two
datasets that must move to reach 0.300.

Caveat: this is a **single run** against a 2-run CLS baseline. The margin is 6x the noise band
so replication is unlikely to overturn it, but a second seed would make it airtight, and
`--pooling mean` should become the locked protocol only after that.

~~New standing: 0.1965 vs the 0.300 target — gap 0.104.~~ **Retracted with the rest of this
section.** The standing number remains **0.185** (cls, corpus v20260531) until the
corpus-matched re-eval lands, and the summed-score requirement stays **+0.574**, with scifact
(+0.322) and fiqa (+0.201) nearly the whole of it. Note the corpus-matched numbers may move
*every* model up, target included in difficulty terms — a rising tide here is not progress.

## CANONICAL TABLE — everything on the pinned corpus v20260801 (2026-08-11)

Job 21361696 re-scored every baseline on the pinned corpus (eval only; all checkpoints already
existed; each model's pooling/prefix settings copied verbatim from its own stored config, so
the corpus is the only thing that changed). **This table supersedes the v20260531 table at the
top of this doc for all forward-looking comparisons.**

| model | arguana | fiqa | nfcorpus | scidocs | scifact | **AVG** | vsRand |
|---|---|---|---|---|---|---|---|
| mE5-large | 0.559 | 0.335 | 0.288 | 0.141 | 0.589 | **0.3824** | +0.333 |
| NeoDictaBERT (HN-SFT) | 0.446 | 0.290 | 0.340 | 0.091 | 0.483 | **0.3300** | +0.281 |
| NeoDictaBERT mean-pool | 0.472 | 0.275 | 0.339 | 0.093 | 0.453 | **0.3264** | +0.277 |
| mE5-base | 0.447 | 0.244 | 0.244 | 0.118 | 0.553 | **0.3211** | +0.272 |
| HMB retrain-final, MEAN seed 42 | 0.271 | 0.134 | 0.267 | 0.051 | 0.259 | 0.1965 | +0.147 |
| HMB retrain-final, MEAN seed 1234 | 0.249 | 0.122 | 0.246 | 0.045 | 0.182 | 0.1689 | +0.119 |
| HMB retrain-final, CLS run1 | 0.223 | 0.117 | 0.265 | 0.041 | 0.178 | 0.1649 | +0.115 |
| HMB retrain-final, CLS run2 | 0.218 | 0.117 | 0.268 | 0.039 | 0.187 | 0.1656 | +0.116 |
| RANDOM-init control | 0.001 | 0.005 | 0.160 | 0.001 | 0.080 | **0.0494** | +0.000 |

### The corpus change is model-dependent, and that is the interesting part

| model | v20260531 | v20260801 | delta |
|---|---|---|---|
| mE5-large | 0.3578 | 0.3824 | **+0.0246** |
| mE5-base | 0.3048 | 0.3211 | **+0.0164** |
| NeoDictaBERT (HN-SFT) | 0.3324 | 0.3300 | −0.0024 |
| NeoDictaBERT mean-pool | 0.3288 | 0.3264 | −0.0024 |
| **HMB retrain-final (cls)** | 0.1852 | 0.1653 | **−0.0199** |
| **RANDOM-init control** | 0.0839 | 0.0494 | **−0.0345** |

Not a uniform shift — it **sorts models by how much they lean on lexical overlap.** The strong
multilingual semantic retrievers *gain*; NeoDictaBERT is flat; the random-init lexical baseline
loses the most (−0.0345), and HMB loses the second most (−0.0199). v20260801 translated with a
unified query/document prompt, temperature 0, and a shared cache — a more faithful, less
surface-leaky rendering. **It rewards semantic matching and punishes lexical shortcuts, and our
model patterns with the lexical baseline.** That is itself a finding about HMB: a meaningful
share of its retrieval score was riding on surface overlap that a cleaner benchmark removes.

This also corrects my earlier guess in the retraction box that the new corpus "should raise
scores for any model" and then that it was simply "harder" — both wrong. It is harder *for
lexical matchers* and easier *for semantic ones*.

### Revised standing

- The **lexical floor drops from 0.0839 to 0.0494**, so v20260801 is a cleaner benchmark and
  `vsRand` is now a more meaningful column than it was.
- **HMB is 50% of NeoDictaBERT (was 56%) and 43% of mE5-large (was 52%).** We look *worse* on
  the better benchmark.
- Backbone value captured vs NeoDictaBERT (vsRand ratio): **41%**.
- **Gap to the 0.300 target grows from 0.115 to 0.135** (from cls 0.1653). Note 0.300 still sits
  right at NeoDictaBERT/mE5-base territory (0.330/0.321) on this corpus, so the target's meaning
  is unchanged even though the distance grew.

## Hard-mode MLM: the capacity effect is REAL but ~8x too small to explain retrieval (2026-08-11)

Job 21316632, `.slurm/jobs/mlm_hardmode_sweep.slurm`. Tests whether the earlier MLM parity was
a ceiling effect hiding NeoDictaBERT's 2.4x-larger body (~265M non-embedding vs our 110M;
28 layers/FFN 3072/full attention vs 22/1152/128-token local on 2/3 of layers). Mask budget
held constant at 12 words/doc, varying only contiguity; top 5% most frequent words dropped.
Deterministic inference — none of the ~0.025 SFT seed variance applies here.

**Per-token accuracy inside the masked span** (the exact-whole-span metric floors out — see below):

| span | NeoDictaBERT | HMB retrain-final | gap | z | **relative gap** |
|---|---|---|---|---|---|
| 1 | 39.2% | 36.5% | +2.7pp | 4.3 | **6.8%** |
| 3 | 19.4% | 17.3% | +2.0pp | 3.9 | **10.5%** |
| 6 | 13.5% | 12.0% | +1.5pp | 2.2 | **11.2%** |

**Result: the ceiling hypothesis is directionally right but quantitatively irrelevant.** The
relative gap does widen as local cues are removed (6.8% → 11.2%), and every gap is statistically
real (z = 2.2–4.3, n = 4.4k–11.8k tokens). So capacity *does* buy NeoDictaBERT something that
isolated-word MLM understates. But:

- **NeoDictaBERT's MLM edge is 7–11% relative, even at maximum hardness.**
- **NeoDictaBERT's retrieval edge is ~79% relative** (0.332 vs ~0.185).

An order of magnitude apart. A 2.4x-larger model buys ~11% on hard masked prediction and ~79%
on retrieval — so **whatever drives the retrieval gap is not masked-token prediction ability**,
at any difficulty we can construct. MLM is not a proxy for retrieval here, and that is now
measured rather than assumed.

This does **not** clear capacity as a cause of the retrieval gap; it only shows MLM cannot
detect it. Capacity could still dominate retrieval through a mechanism MLM never exercises —
compressing a whole passage into one discriminative vector, where depth, FFN width and
full-vs-local attention plausibly matter far more than they do for filling in a masked token.
That remains the leading hypothesis and is still unproven.

**Direct consequence for the large run:** [[large-model-run-decisions]] records HMB-large as
"judged on MLM not BeIR". On this evidence that gate is close to uninformative — it will likely
show a few points of MLM improvement and say nothing about whether retrieval moved. If the
large run is meant to close the retrieval gap, it has to be judged on retrieval.

*Method note:* the sweep's primary metric (exact whole-span recovery — every token of the span
correct) was a **design error**: it compounds multiplicatively, so it collapsed to 2.1%/0.2% at
span 3/6 for NeoDictaBERT and 1.1%/0.0% for HMB — a floor as uninformative as the ceiling it was
built to escape. The per-token metric above was collected in the same run and is the valid one.
The job's own "Widening gap =>" footer is static legend text, not a computed verdict; on the
exact metric the absolute gap *narrows* purely because both models approach zero.

## Corpus-matched pooling result: UNDECIDED at 0.71 sigma (2026-08-11)

All four runs on the pinned corpus **v20260801**, same checkpoint, locked recipe:

| run | seed | arguana | fiqa | nfcorpus | scidocs | scifact | **AVG** |
|---|---|---|---|---|---|---|---|
| CLS run1 | 42 | 0.2231 | 0.1171 | 0.2648 | 0.0409 | 0.1785 | 0.1649 |
| CLS run2 | 42 | 0.2175 | 0.1169 | 0.2678 | 0.0390 | 0.1869 | 0.1656 |
| MEAN | 42 | 0.2713 | 0.1339 | 0.2667 | 0.0513 | 0.2592 | 0.1965 |
| MEAN | 1234 | 0.2490 | 0.1223 | 0.2457 | 0.0451 | 0.1823 | 0.1689 |
| | | | | | | **CLS** | **0.1653** |
| | | | | | | **MEAN** | **0.1827** |

**delta = +0.0174 = 0.71 sigma → indistinguishable. Pooling remains UNDECIDED**, and settling
it needs ~3 seeds per arm (~12h). Finding #4 stands by default, not by evidence.

This table also *confirms the noise diagnosis independently*: the two CLS runs (**same** seed 42)
differ by **0.0007**, while the two MEAN runs (**different** seeds) differ by **0.0276**. Same-seed
replication measures almost nothing; the seed is where the variance lives.

**Two corrections to the retraction box above:**

1. **v20260801 is HARDER, not easier.** Same model, same pooling, same seed: CLS scores
   **0.1852 → 0.1653 (−0.0200)** moving from v20260531 to v20260801. My stated reason for
   retracting — "the newer corpus should raise scores, which would fake a pooling win" — had the
   direction backwards. The confound was real and the retraction was correct, but it was working
   *against* mean pooling, not for it: the corpus-matched delta (+0.0174) is *larger* than the
   cross-corpus one (+0.0112). It is still not significant.
2. **Every baseline in this doc is on v20260531.** NeoDictaBERT 0.332, mE5-large 0.358,
   mE5-base 0.305, random-init 0.084 were all measured in June/July, before v20260801 existed,
   and `results.json` does not record which corpus produced it. So **our new v20260801 numbers
   cannot be compared against those baselines or against the 0.300 target.** If v20260801 costs
   every model ~0.02 the way it cost ours, the effective target on the pinned corpus is nearer
   0.28 — but that is an assumption, not a measurement. **Re-running the four baselines on
   v20260801 is now a prerequisite for any further target-gap claim** (eval only, no training,
   ~20 min total).

Standing number on the pinned corpus: **CLS 0.1653 / MEAN 0.1827**, both un-comparable to the
0531-era baseline table above until those baselines are re-scored.

## ⚠️ THE SFT NOISE BAND IS ~0.025, NOT ~0.002 — most A/B calls in this doc are underpowered (2026-08-11)

Two mean-pooling runs, **identical in every respect except the random seed**, on the same
pinned corpus v20260801:

| run | arguana | fiqa | nfcorpus | scidocs | scifact | **AVG** |
|---|---|---|---|---|---|---|
| mean, seed 42 | 0.2713 | 0.1339 | 0.2667 | 0.0513 | 0.2592 | **0.1965** |
| mean, seed 1234 | 0.2490 | 0.1223 | 0.2457 | 0.0451 | 0.1823 | **0.1689** |
| | | | | | **spread** | **0.0276** |

**The seed-to-seed spread is 0.0276 — about 15x the 0.0019 "noise band" this project has been
using.** That 0.0019 came from jobs 20513043/45, which were described as replicates but
**both ran at seed 42** (the trainer had no `--seed` flag until 2026-08-11; `TrainingArguments`
never set one and the loaders defaulted to 42). They therefore measured GPU nondeterminism
only — not the train/val split, shuffling, or dropout. Implied per-run sigma ~0.0245.

Consequences, in order of how much they matter:

1. **The retracted pooling result was doubly dead.** Besides the corpus confound, its +0.0112
   is **0.5 sigma** — about 39 runs per arm would be needed to call it. It was never
   detectable with one run per arm, regardless of corpus.
2. **The Stage-1 masking A/B is much weaker than recorded.** Its +0.0358 gap (ctrl-m30 0.0910
   vs arm1-m20 0.1268) is **1.5 sigma**, needing ~4 runs per arm; it had **one**. The doc calls
   it "well above the ~0.01-0.02 SFT-eval noise band, a decisive result, not noise" — that
   claim does not survive. 20% masking may well be right (it matches NeoDictaBERT and Wettig
   et al.), but **this experiment did not establish it**, and it is the decision the entire
   retrain was built on.
3. **Any single-run SFT comparison in this doc separated by less than ~0.05 should be read as
   "no difference measured."** That includes the phase-0-only (0.1379) vs Stage-1 (0.1268)
   step, and the shipped-vs-old-100K comparison.
4. **Retrain-final's headline ~0.185 is a 2-run same-seed mean**, so its true confidence
   interval is wider than the quoted 0.1843/0.1862 suggests. It is not wrong, just less precise
   than it reads.

Protocol going forward: **≥3 seeds per arm for any SFT-based A/B**, report mean ± spread, and
treat anything under ~2 sigma as undecided. `--seed` now threads into both the train/val split
and `TrainingArguments(seed, data_seed)`; the job script auto-suffixes output dirs and labels
per seed. Cost is ~2h/run, so a 3-seed A/B is ~12h across both arms — which is the real price
of a trustworthy answer here, and cheaper than another misdirected retrain.

## Reproduce the table

`python3 /tmp/build_beir_table.py` (reads each model's `results.json` under
`outputs/eval/beir_zeroshot/<label>/BeIR_*/`). Open gap: **arguana + scidocs have no
`hard_negatives_train.jsonl`**, so they are zero-shot for *all* models (not an HMB-specific
penalty) — mining hard negatives for them would complete the average.

**Caution:** each `results.json` may contain both `metrics` (current, correct) and
`metrics_pre_fix` (superseded pre-2026-07-30 values, kept for audit) — always read `metrics`,
never `metrics_pre_fix`. The stale baseline numbers corrected on 2026-08-10 (see note near the
top of this doc) came from reading the wrong field.
