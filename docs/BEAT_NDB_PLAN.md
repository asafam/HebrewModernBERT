# Beating NeoDictaBERT on retrieval — what it actually takes

Goal (user, 2026-10-13): **exceed** NeoDictaBERT on BeIR Hebrew avg NDCG@10, not match it.
NDB = **0.330**. Our best measured = **0.186** (per-dataset maxima across sweep arms = 0.2045).
mE5-large = 0.382 (proof the bar is clearable).

## The gap is one dataset

| dataset | ours (best) | NDB | % of NDB | gap | **share of total gap** |
|---|---|---|---|---|---|
| **arguana** | 0.0637 | 0.4461 | **14%** | 0.3824 | **61%** |
| scifact | 0.3694 | 0.4831 | 76% | 0.1137 | 18% |
| fiqa | 0.2048 | 0.2896 | 71% | 0.0848 | 14% |
| nfcorpus | 0.3144 | 0.3403 | 92% | 0.0259 | 4% |
| scidocs | 0.0701 | 0.0908 | 77% | 0.0207 | 3% |

Two things follow, and they set the whole strategy:

1. **On four of five datasets we are already at 71-92% of NDB.** We even beat **mE5-large** on
   nfcorpus (0.3144 vs 0.2878). Those four are a config-tuning problem.
2. **arguana is 61% of the gap and we score 14% of NDB there.** ArguAna is the only dataset with
   **no training pairs** — its score is pure pre-SFT embedding geometry. So the dominant term in
   the gap is not our SFT recipe and not our language modelling. It is that **our embedding space
   is bad, and our pretraining made it worse**: base scores 0.2201 on arguana, large scores 0.0637.

Corroborating measurements: CLS anisotropy **0.9997** vs NDB's **0.833**; with CLS the model fails
to retrieve a document's own identical text 93% of the time (98/1401) vs 60% with mean pooling;
base's query/doc alignment flipped **+0.914 -> -0.269** across phase-1/2.

## Why config tuning cannot get there

The three sweep arms gave +2.9% to +6.8%, and base's own seed spread is 0.0084 — so most of that
is noise. Summing every per-dataset maximum gives 0.2045, and that is an *illusory ceiling* (it
mixes incompatible configs). Realistic combined: **~0.19-0.21**. We need **0.33+**. Config work is
worth doing — it is nearly free — but it is **~15% of the required distance**.

## What has the right magnitude: retrieval-aware pretraining

Look at the ladder:

| model | avg | how it was trained |
|---|---|---|
| mE5-large | 0.382 | **contrastively pretrained** on paired data |
| NeoDictaBERT | 0.330 | MLM-only, but **28/28 global attention**, RMSNorm, SwiGLU, untied |
| **HMB-large** | 0.186 | MLM-only, **10/28 global**, LayerNorm, GeGLU, tied |

MLM trains a *token predictor*. Retrieval needs a *metric space*. Everything in our stack so far
has been reading a space that was never trained to be one — which is exactly why the zero-shot
dataset (arguana) is where we collapse, and why more/better MLM pretraining (185B tokens, +5.0 MLM
over base) made arguana **worse**.

**Our genuine asset:** we already have the best Hebrew MLM backbone measured (49.0 vs NDB 46.9).
Contrastive pretraining on a stronger backbone should beat contrastive pretraining on a weaker one.
Nobody else has a Hebrew-specialised contrastively-pretrained encoder.

## Staged plan, with kill criteria

### Stage 0 — free diagnostics (hours, no training)
Run before anything expensive; any one of these could redirect the plan.
1. **SFT from phase-0 / phase-1 instead of phase-2.** The anneal is the prime suspect for the
   geometry damage (alignment flip above). `ba222000` phase-0 export already exists.
2. **Disable local attention at load** — `global_attn_every_n_layers=1` makes all 28 layers global
   on existing weights (verified `modeling_modernbert.py:464`). Free test of the architecture
   hypothesis. It is a train/test shift, so degradation is also informative.
3. **Mean pooling** everywhere (adopted — A3 was the best arm and the only one to move arguana).
4. **Whitening / isotropy correction** on the output embeddings — standard cheap fix for
   anisotropy 0.9997.

*Kill criterion:* if (1) recovers arguana substantially, the anneal is the culprit and Stage 1
should start from an earlier checkpoint — big saving.

### Stage 1 — harvest the cheap SFT wins (~1 day)
Combined arm: **mean pooling + GradCache negatives (1k-8k) + rebalanced mix + LR/epoch retune +
trained projection head**. Expected **0.19-0.22**.

*Kill criterion:* if this lands below ~0.19, the SFT recipe is not the binding constraint and we
should go straight to Stage 2.

### Stage 2 — contrastive pretraining (the actual answer)
Continue pretraining the existing large weights with a **contrastive objective alongside MLM**:

- **Positives from unlabelled text — but the corpus is SHORT-document, so pick the scheme first.**
  `TRAINING.md` §3: median ~100 tokens, 92% under 1024. Two "independent crops" of a 100-token
  document either overlap almost entirely (a trivial positive) or are a few tokens each. Options,
  decide before writing the sampler: (a) filter `_source` to the long-doc sources (`yifat_long`
  ~305 tok/doc, `books`, `wikipedia`) and crop those; (b) SimCSE dropout-positives — the same text
  twice through different dropout masks, which needs no length at all; (c) title/body pairs from
  Wikipedia and news. (b) is the cheapest to implement and is the standard unsupervised baseline.
- **InfoNCE with a large negative pool** via GradCache (not the 32 we were using).
- **Keep MLM as an auxiliary loss** so we do not destroy the 49.0 MLM win — multi-task, not replace.
- **Mean pooling** during pretraining, matching downstream use.
- **All-global attention** (`global_attn_every_n_layers: 1`) — only **+9%** at 1024 context.
- **Start from phase-0 or phase-1, not phase-2** (pending Stage 0 item 1).
- **FIRST RUN IS SMALL, not 10-20B tokens.** Unsupervised SimCSE-scale adaptation (~1-5M pairs,
  a few GPU-hours) is enough to tell us whether contrastive adaptation moves the metric at all.
  Scale to 10-20B tokens (~30-40 GPU-hours) only if the small run shows signal. Measure first.
- **MLM auxiliary loss: deferred for the first run.** The sibling repo's `InfoNCEDualEncoder` uses
  `AutoModel`, so there is no MLM head and no aux loss. For a short run the forgetting is bounded
  by step count, so the first experiment is **pure contrastive** — but re-run
  `mlm_compare_vs_neodictabert.py` afterwards to measure what the 49.0 MLM win cost. Add the
  multi-task loss only if the trade is bad and the retrieval gain is worth keeping.

Why this targets the actual gap: contrastive losses have an explicit uniformity term, so they fix
anisotropy *by construction* — which is the arguana failure, i.e. 61% of the distance to NDB.

*New engineering required:* this repo has no contrastive pretraining path. Needs a dataset that
emits crop pairs, an InfoNCE+MLM multi-task loss, and wiring in `main.py` — a few hundred lines,
not a config change. This is the one item that is real implementation work.

### Stage 3 — only if Stages 0-2 fall short
Full retrain with NDB's architecture: all-global attention (**+47% total compute, +60 GPU-hours**,
almost all of it in the 8192 phases), RMSNorm, SwiGLU, untied embeddings.

**Note on the 8192 phases:** retrieval SFT and eval both run at `max_length: 512`. We spent ~99h
and 55B tokens extending context to 8192, and the retrieval metric truncates at 512 — so phases 1-2
contribute nothing directly to this goal, and phase-2 may have actively damaged it. A
retrieval-first retrain should do all-global at 1024 and skip or defer the 8192 extension.

## Honest expected outcome

Stage 0+1 alone: **~0.20**. Not enough.
Stage 0+1+2: plausibly **0.28-0.32** — NDB parity, possibly just past it.
Beating NDB **clearly** likely also needs Stage 3, or hard-negative mining, or the full-BeIR
translation.

**Fairness caveat on the full-BeIR translation:** arguana currently has no training pairs *for
anyone*, so NDB's 0.4461 is a zero-shot score. If we add arguana training data, an honest
comparison requires re-running NDB with the same data — otherwise "we beat NDB" measures our better
training set, not our better model. Both are legitimate goals ("best Hebrew retriever" vs "best
Hebrew backbone") but they are different claims and should not be conflated.
