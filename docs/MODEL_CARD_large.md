---
language: he
license: apache-2.0
library_name: transformers
pipeline_tag: fill-mask
tags:
  - modernbert
  - hebrew
  - encoder
---

# HebrewModernBERT-large

A Hebrew ModernBERT encoder: **28 layers, 1024 hidden, 16 heads, 2624 FFN, 498M parameters,
8192-token context**, vocab 150,016 (BPE).

Trained on **185B tokens** of Hebrew: 130B pretrain @1024 → 40B context-extension @8192 →
15B curated anneal. Warm-started from HebrewModernBERT-base by Phi-style weight tiling
(ModernBERT-large's own recipe), not from scratch.

## Results

**Masked language modelling — beats the previous Hebrew SOTA.** Whole-word recovery on
translated BeIR Hebrew text: a whole Hebrew word is masked and each model fills every token of
it under its own tokenization, counting a hit only if every token id is recovered. The unit is a
word, so a 150K SentencePiece and a 128K WordPiece vocab are directly comparable.

| model | arguana | fiqa | nfcorpus | scidocs | scifact | **mean** |
|---|---|---|---|---|---|---|
| **HebrewModernBERT-large** | **46.7** | **48.5** | **54.0** | 52.1 | **43.7** | **49.0** |
| NeoDictaBERT (previous SOTA) | 43.7 | 45.3 | 51.8 | **52.7** | 41.2 | 46.9 |
| HebrewModernBERT-base | 41.3 | 45.4 | 49.1 | 45.9 | 38.4 | 44.0 |

Wins 4 of 5 datasets. All models scored in the same run on identical words and masks; 0% of
words dropped; our 1.38 tokens/word vs NeoDictaBERT's 1.30 makes exact match marginally harder
for us, so the comparison is conservative in their favour.

**Retrieval — the model's main weakness.** avg NDCG@10 on BeIR Hebrew after dual-encoder SFT
(**use mean pooling**, see Usage):

| | arguana | fiqa | nfcorpus | scidocs | scifact | all 5 | **excl. arguana** |
|---|---|---|---|---|---|---|---|
| **HebrewModernBERT-large** | 0.064 | 0.166 | 0.294 | 0.068 | 0.342 | **0.187** | **0.217** |
| HebrewModernBERT-base | 0.223 | 0.117 | 0.265 | 0.041 | 0.179 | 0.165 | 0.150 |
| NeoDictaBERT | 0.446 | 0.290 | 0.340 | 0.091 | 0.483 | 0.330 | 0.301 |
| mE5-large | 0.559 | 0.335 | 0.288 | 0.141 | 0.589 | 0.382 | 0.338 |

Large improves on base by **+45% on the four non-arguana datasets** (0.217 vs 0.150) but only +13%
overall, because **arguana collapses**: 0.064 against base's 0.223.

**This is a representation problem, and it is the honest limitation of this release.** The diagnostic
number is recall@100 on arguana: **NeoDictaBERT 0.974, this model 0.296** — NDB has the correct
document in its top 100 for 97% of queries, we do for 30%. The right document is not in our
candidate set at all, so it is first-stage retrieval that fails, not ranking. Supporting
measurements: CLS-space anisotropy **0.9997** vs NDB's 0.833, and with CLS pooling the model fails
to retrieve a document's own identical text 93% of the time (mean pooling: 40%).

Root cause, as best we understand it: the model is trained **only** on masked language modelling,
which produces a token predictor rather than a metric space. More MLM pretraining made this axis
*worse* — large beats base by 5.0 on MLM and is 3.5x worse on arguana. Models that do well at
retrieval are pretrained for it (mE5) or use full global attention (NDB uses 28/28 global; this
model uses 10/28 with a 128-token sliding window). Contrastive pretraining is the planned fix; see
`recipe/docs/BEAT_NDB_PLAN.md`.

**Use this model for** MLM, classification, token classification, NER, and long-context Hebrew
understanding. **For retrieval**, NeoDictaBERT or mE5 currently perform better out of the box.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
tok = AutoTokenizer.from_pretrained("<org>/HebrewModernBERT-large", use_fast=True)
model = AutoModelForMaskedLM.from_pretrained("<org>/HebrewModernBERT-large")
```

**If you use this for embeddings, use mean pooling, not CLS.** The CLS vector is near-degenerate
(anisotropy 0.9997); mean pooling measured 4.4x better on zero-shot retrieval.

`use_fast=True` is required — the fast `tokenizer.json` is the source of truth. Special ids:
`[PAD]=0 [UNK]=1 [CLS]=2 [SEP]=3 [MASK]=4`.

## Training data

~39B unique Hebrew tokens, upsampled per-source to 131.1B: MAFAT Hebrew collections (HeC4,
Yifat, OSCAR, HebNLI, Tzenzura, books, Wikipedia, news) plus FineWeb-2 Hebrew (~9.4B net-new
after dedup against MAFAT). Hebrew-only — no English or code.

**The MAFAT corpus is private and is not distributed with this model.** The `recipe/` directory
in this repo contains everything needed to rebuild the corpus from raw MAFAT: the acquisition,
dedup and filtering scripts, the per-source mix configs with their exact token targets and epoch
multipliers, the build jobs, the training YAMLs for every phase, and the process documentation.
Given the raw MAFAT files, the corpus and the training run are reproducible.

## Recipe notes

- **20% masking** — chosen by A/B against 30% on this corpus, and matching NeoDictaBERT.
- Hebrew-only pretraining; the earlier mixed English/code phase was dropped.
- Sequence packing with per-document `cu_seqlens` during context extension. Without it,
  tokens attend across document boundaries and retrieval collapses — this was a real regression
  in an earlier run.
- Architecture is ModernBERT throughout (LayerNorm, GeGLU, RoPE 160000/10000, sliding window
  128 with global attention every 3 layers). NeoDictaBERT's RMSNorm / SwiGLU / untied
  embeddings / full global attention were **not** adopted; that remains an untested axis.

Full training log, including the measured throughput, the gates at each stage, and the bugs
found along the way: `recipe/docs/LARGE_TRAINING.md`.

## Limitations

- Retrieval is well below NeoDictaBERT and mE5 (see above).
- arguana retrieval (0.064) is far below even our own base model (0.223) — see above.
- recall@100 is 0.296 where NeoDictaBERT reaches 0.974: first-stage retrieval, not ranking, is the failure.
- No Hebrew NLU/GLUE benchmark has been run; MLM whole-word recovery is the only
  apples-to-apples evaluation performed.
- Trained and evaluated on Hebrew only.
