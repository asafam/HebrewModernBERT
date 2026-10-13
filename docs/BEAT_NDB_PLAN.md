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

---

# RESULT: contrastive adaptation works. 0.187 -> 0.295 (+58%), 89% of NeoDictaBERT

Stage 2 (Track B) ran 2026-10-13, job 34976281. **~1M SimCSE pairs, 1 epoch, 3 hours of 1 GPU.**

| run | arguana | fiqa | nfcorpus | scidocs | scifact | **ALL 5** | ex-arguana |
|---|---|---|---|---|---|---|---|
| baseline (cls, phase-2) | 0.0146 | 0.1637 | 0.3144 | 0.0628 | 0.3180 | 0.1747 | 0.2147 |
| A3 (mean, phase-2) | 0.0637 | 0.1661 | 0.2939 | 0.0678 | 0.3415 | 0.1866 | 0.2173 |
| S0a (mean, **phase-0**) | 0.0913 | 0.1569 | 0.2752 | 0.0693 | 0.3047 | 0.1795 | 0.2015 |
| **B1 (SimCSE adapt)** | **0.4297** | **0.2408** | **0.3260** | **0.0787** | **0.4013** | **0.2953** | **0.2617** |
| NeoDictaBERT | 0.4461 | 0.2896 | 0.3403 | 0.0908 | 0.4831 | 0.3300 | 0.3009 |

**Every single dataset improved.** +575% arguana, +45% fiqa, +18% scifact, +16% scidocs, +11%
nfcorpus. We went from **57% of NDB to 89%** (87% excluding arguana).

### The predicted mechanism is confirmed, not just the outcome

The diagnosis was that first-stage recall was broken because the embedding space was collapsed.
Both diagnostics moved to NDB's level:

| | before (A3) | **after (B1)** | NeoDictaBERT |
|---|---|---|---|
| arguana recall@100 | 0.296 | **0.946** | 0.974 |
| arguana self-retrieval /1401 | 845 | **1293** | 1294 |

Self-retrieval going 845 -> 1293 against NDB's 1294 is the cleanest possible confirmation: the
space now behaves like theirs on the easiest retrieval task there is.

**This cannot be training-data leakage.** ArguAna has **zero** training pairs, and it is the dataset
that improved most (6.7x). Leakage cannot produce that; fixing geometry predicts exactly it.

### Stage 0a: the anneal-damage hypothesis is REFUTED

SFT from the pre-anneal phase-0 checkpoint scored **0.1795 vs phase-2's 0.1866** — slightly worse
overall and worse ex-arguana (0.2015 vs 0.2173), though better on arguana alone (0.0913 vs 0.0637).
So phase-2's anneal did **not** damage the space in a way that costs retrieval, and contrastive
adaptation should keep starting from **phase-2**, which is what B1 did. The earlier
query/doc-alignment flip (+0.914 -> -0.269) was real but is not what was costing us points.

### The remaining gap is now the TRAINED datasets

| dataset | B1 | NDB | gap | share of remaining gap |
|---|---|---|---|---|
| **scifact** | 0.4013 | 0.4831 | 0.0818 | **47%** |
| **fiqa** | 0.2408 | 0.2896 | 0.0488 | **28%** |
| arguana | 0.4297 | 0.4461 | 0.0164 | 9% |
| nfcorpus | 0.3260 | 0.3403 | 0.0143 | 8% |
| scidocs | 0.0787 | 0.0908 | 0.0121 | 7% |

**scifact + fiqa are 76% of what is left, and both have training pairs** — so the binding
constraint has moved from geometry back to the SFT recipe. That is precisely where Track A's arms
already showed signal: A1 (negatives 32->256) gave scifact +16%, A2 (rebalanced mix) gave fiqa +25%.
Those were measured on the un-adapted model and should compose with B1.

### Next, in order

1. **B1 + Track A combined** — contrastive adaptation, then SFT with 256+ negatives and the
   rebalanced mix, two seeds. Cheap, and it targets the 76%.
2. **Do NOT just scale the adaptation — the task SATURATES.** The InfoNCE loss went 5.17 (chance
   for 256 candidates is ln(256) = 5.545) to **0.0021 by epoch 0.05**, i.e. within ~200 of the
   3,906 steps, then sat at ~0.0001 for the remaining 95% of the run. So ~10 minutes of the 3
   hours did the work and more pairs or epochs adds no gradient signal. **The next lever is a
   HARDER task, not a longer one:**
   - **more negatives** (GradCache to 4k-32k) — the discrimination is currently 1-of-256;
   - **harder positives** — dropout-only views are nearly identical text, so the model can win by
     surface matching. Document crops (two different spans of one document), span deletion or
     back-translation force actual semantic alignment;
   - **lower temperature** to sharpen the objective.

   That even a saturating, trivially-easy objective bought +58% says the collapse was the whole
   problem and very little uniformity pressure was needed to break it. Getting the *rest* of the
   way needs the task to stay hard enough to keep producing gradient.
3. **GradCache + gradient-checkpointing support** on `InfoNCEDualEncoder` to push the negative pool
   past 256 (it currently refuses checkpointing; see the Track B job comments).
4. **All-global attention** — Stage 0b still pending.
5. Full-BeIR translation for arguana/scidocs training pairs.

**Beating NDB now needs +12% from 0.2953, against the +58% just achieved from one 3-hour run.**

## Stage 0b: all-global attention is NOT worth retraining for — hypothesis refuted

Clean single-variable test on the existing weights (`global_attn_every_n_layers` 3 -> 1, i.e.
10/28 global layers -> 28/28; same checkpoint, same mean pooling, same seed, locked otherwise):

| dataset | 10/28 global | 28/28 global | delta |
|---|---|---|---|
| arguana | 0.0637 | 0.0682 | +0.0045 |
| fiqa | 0.1661 | 0.1705 | +0.0044 |
| nfcorpus | 0.2939 | 0.3040 | +0.0101 |
| scifact | 0.3415 | 0.3338 | -0.0077 |
| **AVG(4)** | **0.2163** | **0.2191** | **+0.0028 (+1.3%)** |

**+0.0028 against a 0.0084 seed-noise floor — inside noise.** The retrain would have cost +60
GPU-hours (+47%, almost all of it in the 8192 phases) and weeks of calendar time. **Removed from
the plan.**

The hypothesis was reasonable: it explained both halves of the paradox (local attention is fine for
MLM, poor for a pooled document vector) and NeoDictaBERT genuinely is 28/28 global with RMSNorm,
SwiGLU and untied embeddings. But it was inference from correlation, and the decisive evidence is
B1, not 0b: **contrastive adaptation reached 89% of NDB with zero architecture change.** NDB's
global attention and NDB's good geometry co-occur in their model; the retrieval strength was
attributed to the wrong one.

*Caveat:* this runs global attention on weights trained with local attention, a train/test
mismatch, so it is not a perfect proxy for pretraining that way. The logic still holds — if local
attention were starving the pooled vector, granting full global range should have helped even under
mismatch, since strictly more information is available. It did not move.

**Consequence for the remaining 12%:** it is not pretraining and not architecture. scifact (47% of
the gap) and fiqa (28%) are both TRAINED datasets, so the levers are the SFT recipe (A1's negatives
gave scifact +16%, A2's rebalance gave fiqa +25%) and a HARDER contrastive objective — more
negatives via GradCache, crop-based positives instead of dropout, lower temperature. Not more of
the same.

---

# Chasing mE5-large (0.382), not just NDB (0.330)

After Track B took us to 0.2953, mE5 is a credible target. The case, with the measurements:

## What we already hold over mE5

**1. Tokenizer — measured on 80,456 words of our own Hebrew corpus:**

| tokenizer | tok/word |
|---|---|
| **HMB (BPE 150K)** | **1.256** |
| NeoDictaBERT (WordPiece 128K) | 1.248 |
| mE5-large (XLM-R 250K) | **1.889** |

mE5 needs **50% more tokens** for the same Hebrew. At `max_length 512` it fits ~271 Hebrew words
against our ~408 — we see half again as much document per query, and its Hebrew subword coverage is
correspondingly coarser.

**2. MLM: 49.0 vs NDB's 46.9**, measured. mE5 is multilingual, so Hebrew is one slice of its
pretraining; ours is 185B Hebrew tokens.

**3. Hebrew-specialised throughout** — tokenizer, corpus, and now embedding geometry.

## mE5's single advantage, and why it is beatable

mE5's recipe is **weakly-supervised contrastive pretraining on ~1B mined text pairs**, then
supervised fine-tuning. We have done **1M self-supervised pairs, once, for three hours**.

But mE5 spread that ~1B across ~100 languages. Hebrew's share is plausibly ~1%, so **~10M Hebrew
pairs**. Ten to fifty million *Hebrew* pairs is therefore comparable to or better than what mE5
ever saw for Hebrew — and that is reachable from the corpus we already own.

## Where 10-50M Hebrew weak pairs come from (all already on disk)

| source | in-corpus tokens | pair construction |
|---|---|---|
| wikipedia | 1.1B | title<->body, section<->section, anchor<->target article |
| news (YisraelHayom, Geektime) | 0.85B | headline<->article |
| HeQ | — | question<->context (**F1 is testing this now**) |
| HebNLI | 6.6B | premise<->hypothesis (flattened in the MDS; needs the raw jsonl) |
| hec4_clean + fineweb2 | 64B | adjacent-paragraph and crop pairs — effectively unlimited |

The crop sampler already generates unlimited pairs from 39B unique tokens, so **pair volume is not
the constraint** — pair *quality and variety* is.

## Cost

Measured rate: 1M pairs at bs 256 took ~3h on one GPU. **10M pairs ~= 15-30 GPU-hours**, comparable
to a single training phase and far below the 128h the MLM pretraining cost. Affordable.

## The honest blocker

mE5's *supervised* stage uses large labelled retrieval sets (MS MARCO, NQ, ...). We have only BeIR
Hebrew: **125,625 pairs across 3 datasets**, and 88% of it is one dataset. That is the thinnest
part of our pipeline and is exactly what the full-BeIR translation would widen.

## Sequencing

**Do not scale B1 blindly.** The 7 arms in flight are determining *which* contrastive configuration
is worth scaling (harder positives? more negatives? sharper temperature? real pairs vs
self-supervised views?). Scale the winner, after it replicates at a second seed. Scaling the wrong
configuration 10x would spend 30 GPU-hours confirming a noise artifact.

---

# PARITY WITH NEODICTABERT REACHED: 0.3290 vs 0.3300

Arm **D** (contrastive adaptation + rebalanced SFT mix), job 35085819, seed 17.

| dataset | **D (ours)** | NeoDictaBERT | delta | |
|---|---|---|---|---|
| arguana | **0.4701** | 0.4461 | **+0.0240** | **we win** |
| scifact | **0.5360** | 0.4831 | **+0.0529** | **we win** |
| scidocs | **0.0991** | 0.0908 | **+0.0083** | **we win** |
| fiqa | 0.2659 | 0.2896 | -0.0237 | they win |
| nfcorpus | 0.2737 | 0.3403 | **-0.0666** | they win |
| **AVG** | **0.3290** | **0.3300** | **-0.0010** | **dead heat** |

**-0.0010 against a 0.0084 noise floor: statistically tied, and we win 3 of 5 datasets.**
Verified apples-to-apples: identical query counts (1401/300/323), identical mean pooling, identical
corpus run (20260801).

**scifact 0.5360 is the headline** — it beats NDB by 11% and is closing on mE5-large's 0.5894.

## The whole remaining deficit is self-inflicted

nfcorpus (-0.0666) is ~2.8x the next-largest gap, and we caused it: the rebalance caps nfcorpus
training pairs from 110,575 to an **arbitrary 15,000**. The trade-off is visible across arms:

| nfcorpus cap | nfcorpus | scifact | AVG |
|---|---|---|---|
| 110,575 (uncapped, B1) | **0.3260** | 0.4013 | 0.2953 |
| 15,000 (D) | 0.2737 | **0.5360** | **0.3290** |

Capping cost 0.052 on nfcorpus and bought 0.135 on scifact — net strongly positive, but 15,000 was
never tuned. An intermediate cap should hold most of the scifact gain while recovering nfcorpus.
Arithmetic: nfcorpus back to ~0.32 with scifact at ~0.50 gives **~0.331, past NDB.**

Launched: H cap=50000 (35139003), H cap=30000 (35139004), and D at seed 43 (35139005) to replicate
the leader before trusting it.

## Progression

| | AVG | note |
|---|---|---|
| shipped base | 0.1440 | |
| large, cls, locked SFT | 0.1747 | where this effort started |
| + mean pooling (A3) | 0.1866 | |
| + contrastive adaptation (B1) | 0.2953 | the breakthrough, 3 GPU-hours |
| **+ rebalanced mix (D)** | **0.3290** | **NDB parity** |
| NeoDictaBERT | 0.3300 | |
| mE5-large | 0.3823 | next target |

**0.1747 -> 0.3290 is +88%, and none of it required retraining the backbone.**
