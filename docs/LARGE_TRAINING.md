# HebrewModernBERT-**large** — Training Log & Findings

Live record of the large-model run: what was decided, what was measured, what broke, and what the
numbers are. Companion to `docs/TRAINING.md` (the base run's process doc, whose §8 holds the
runbook) and `docs/RETRIEVAL_EVAL.md` (evaluation history).

**Status: phase-0 training, launched 2026-08-11.** Job `21342052`, 4x B200 on `p_b200_nlp`.
W&B: `HebModernBERT-large-phase0` / `modern-bert-large-phase-0-pretrain-g4`.

---

## 1. What this model is, and why

| | base | **large** | NeoDictaBERT (the target) |
|---|---|---|---|
| layers x hidden x heads x ffn | 22 x 768 x 12 x 1152 | **28 x 1024 x 16 x 2624** | 28 x 768 x 12 x 3072 |
| total params | 226M | **498M** | 461M |
| **non-embedding params** | 110M | **343M** | 264M |
| norm / activation | LayerNorm / GeGLU | LayerNorm / GeGLU | RMSNorm / SwiGLU |
| attention | sliding 128 + global every 3 | same | **full global every layer** |
| vocab | 150,016 BPE | same | 128,000 WordPiece |

**Why large.** `docs/RETRIEVAL_EVAL.md` (2026-08-11) established that HMB-base is at *MLM parity*
with NeoDictaBERT (48.6 vs 49.9 whole-word recovery) while scoring 0.185 vs 0.332 on BeIR — so the
retrieval gap is a representation-*transfer* problem, not a backbone problem. Large is therefore
justified as **a stronger Hebrew encoder and the second member of a released family — NOT as the
lever that closes BeIR.**

A number that reframes the comparison: **base reached MLM parity with 42% of NeoDictaBERT's
non-embedding capacity** (110M vs 264M). Large is the first configuration that exceeds them (343M,
1.30x). Separately: the base run took *20% masking, Hebrew-only pretraining, and the plain-positive
SFT protocol* from NeoDictaBERT, but its **architecture is 100% ModernBERT** — RMSNorm, SwiGLU,
untied embeddings and full global attention were never adopted. That remains the open axis.

**Known limitation:** this repo has **no Hebrew NLU/GLUE eval wired up**, so
`scripts/mlm_compare_vs_neodictabert.py` is currently the only apples-to-apples quality signal.

## 2. Locked decisions (user, 2026-08-11)

- **Warm start by Phi-style weight tiling** from `checkpoints/base/phase-0-arm1-m20/ckpt` (the
  130B-token, 20%-masking phase-0 endpoint). Deliberately **not** the annealed phase-2 final, whose
  query/document alignment flipped negative (+0.914 -> -0.269).
- **Masking 20%**, matching base's A/B winner and NeoDictaBERT.
- **Corpus FROZEN**: same `hebrew_quality`, same 130B / 40B / 15B phase budgets as the validated
  base run. No rebuild, no re-weighting, no new sources.
- **Budget: whatever it takes.**

### Why the corpus stayed frozen
Large re-consumes a corpus base already saw, so "are we past the repetition knee?" was raised. It
was closed on evidence: NeoDictaBERT itself ran **6.0 epochs** over ~46B unique; the base retrain ran
**4.7** — the most conservative of the three. Base's MLM rose monotonically across all 4.7 epochs
(45.0 @ba76000 -> 46.3 @phase-0 end -> 48.6 @phase-2 end). The binding constraint is **token
intensity** (~370 tok/param for large vs ModernBERT's ~19,100), not over-repetition. Re-weighting
the mix was rejected because it would push `hec4_clean` alone to ~8 epochs over 8.63B unique;
acquiring more unique Hebrew (Knesset/Ben-Yehuda/Sefaria) is the real lever but **not feasible at
this stage**.

One deliberate, non-corpus change: `train_loader.shuffle_seed: 71`. Large inherits base's weights,
so replaying base's exact document order, packing and mask positions would be maximally redundant.
Set on the **train** loader only — the eval loader and `seed: 17` are untouched, so the eval curve
stays comparable to base's.

## 3. Bugs found — the `init_from_checkpoint` path had NEVER run in this repo

Base chained its phases via `load_path` (Composer's own loader), so the tiling path was executed for
the first time by this run. Four defects surfaced, each caught in seconds by gate ordering rather
than GPU-hours:

1. **`tile_embedding(...tok_embeddings...)` was commented out** in `init_model_from_pretrained`
   (`src/bert_layers/model.py`). With `tie_word_embeddings=True` (HF default, unset in our YAMLs)
   the decoder is re-tied to `tok_embeddings`, so this left the **150016 x 1024 embedding matrix —
   31% of the model — at random init**, silently, while the loss curve looked plausible.
2. **`checkpoint_cfg: ${model}`** — the form used by every commented-out example in the repo —
   resolves to the enclosing file's *own* model block, so it builds a *large* model and loads a
   *base* state dict into it. Fixed with a literal `base_model:` block.
3. **`torch.load` needs `weights_only=False`** on torch >= 2.6; Composer checkpoints carry
   `datetime.timedelta`, which the restricted unpickler rejects. Fixed in `main.py` and the six
   sibling calls in `model.py`.
4. **`base_model` must mirror the large block** on `bert_layer` / `padding` / `embedding_layer` /
   `compile_model` — these select layer *classes* and `init_model_from_pretrained` asserts type
   equality. The smoke's `compile_model=false` override desynced them. Now **interpolated** from the
   model block so no CLI override can break it, plus a preflight check that catches it.

**Verification that the warm start took** (`scripts/verify_tile_init.py`, run by the smoke and
gating it): `tok_embeddings` std **0.0176 -> 0.1129** against the base checkpoint's 0.113, with
`decoder tied: True`. A fresh `init_std: 0.02` init reads ~0.018.

### Two settings that look active but are not
- **`loss_kwargs.inplace_backward`** is unconditionally reset to False by `FlexBertConfig.__init__`
  (`self.loss_kwargs` is the same dict `get_loss_fn` reads). It never took effect in the base run
  either. What actually carries the 150K-vocab backward is the small microbatch +
  `masked_prediction: true` + `expandable_segments:True`.
- **`batch_size_warmup_*` in phase-0** is read only inside the `sequence_packing` branch of
  `src/text_data.py`, which phase-0 does not use. **No batch-size ramp has ever run in phase-0.**

Both kept for parity with base and annotated in the configs.

## 4. Measured performance

| | value |
|---|---|
| `device_train_microbatch_size` | **576** (auto-search 4608->2304->1152->576; `compile=true` validated) |
| real tokens per batch | **549,523** (= 130B / 236,569 ba from base; confirmed independently at 119.3 tok/sample) |
| throughput, 4x B200 | **~557-597K real tok/s**, 0.99 s/ba |
| large vs base compute | only **2.31x** slower despite a **3.10x** non-embedding param ratio |
| checkpoint size | **6.0 GB** (`save_interval: 2000ba`, `keep: 20`) |
| **phase-0 (130B) ETA** | **~2.0-2.6 days** of 4-GPU time |

576 is also correct by arithmetic for 4 GPUs: 4608/4 = 1152 per device, so 576 gives exactly 2
gradient-accumulation steps; the only larger divisor is 1152, which the search proved OOMs.

### The throughput saga — and the measurement rule it produced
Phase-0 initially sustained only **~180K tok/s (3.05 s/ba)**, an 8.2-day ETA, against a GPU-bound
ceiling of ~550K. Diagnosis from `sstat`: only **~7 of 32 allocated CPUs** busy => workers **blocked
on I/O**, not computing. Fix: `train_loader.num_workers` 8 -> **24**, `prefetch_factor` 2 -> **4**
(eval loader left at 8), applied by editing the YAML and letting the **4h-wall requeue** pick it up
— no cancel, no lost progress, no surrendered GPUs.

| | throughput | s/ba | ETA |
|---|---|---|---|
| before | ~180K tok/s | 3.05 | ~8.2 d |
| **after** | **~557-597K tok/s** | **0.99** | **~2.0-2.6 d** |

**No extra CPUs were needed** — raising `--cpus-per-task` would have done nothing, since the cores
were idle rather than saturated. RSS rose 374GB -> ~969GB (mmap'd page cache the job previously
could not hold), so `--mem` 480GB -> 1200GB is staged in the job file as a further improvement; it
only applies on a fresh `sbatch`, not an automatic requeue.

> **RULE — short probes measure BURST, not sustained, throughput.** Two 30-batch probes reported
> ~550K and wrongly concluded the dataloader was fine. They ran after an 80-200s `torch.compile`
> during which the workers filled the prefetch queue, so every batch was served from that buffer at
> GPU speed. A dataloader probe must run long enough to **drain `num_workers x prefetch_factor`
> batches AND dwarf the compile prefill**. Judge sustained rate from **5-minute windows of a live
> run** instead.

## 5. Quality gates

`sbatch --export=ALL,BA=<n> .slurm/jobs/gate_large_phase0.slurm` converts a checkpoint to HF and
scores whole-word MLM recovery against NeoDictaBERT and base **in the same job**.

> **RULE — compare within a single job's output only.** The same NeoDictaBERT and base checkpoints
> scored 49.9/46.3 on record and 48.6/44.8 on 2026-08-11, with identical `n_words` but different
> `toks_per_word`: the sampled BeIR text changed underneath (the sibling repo's corpora are
> regenerated by other work). Both references moved by a similar margin, so within-run comparisons
> stay valid while cross-run ones do not.

### ba6000 — 3.3B tokens (2.5%), 2026-08-11 — pipeline validation

| model | arguana | fiqa | nfcorpus | scidocs | scifact | **MEAN** |
|---|---|---|---|---|---|---|
| NeoDictaBERT | 47.5 | 49.0 | 51.2 | 53.8 | 41.5 | **48.6** |
| HMB phase-0 FINAL *(tile source)* | 45.1 | 47.8 | 46.3 | 46.7 | 38.3 | **44.8** |
| **HMB large @ba6000** | 45.0 | 46.8 | **47.0** | 45.5 | 37.6 | **44.4** |

Large **recovered to parity with its tile source** after 2.5% of training (tiling is not
function-preserving, so it must climb back from the perturbation) and already edges base on
nfcorpus. Not yet compounding — expected this early. Sanity: 2,695-2,699 words, **0% dropped**,
HMB 1.38 tok/word vs NeoDictaBERT 1.30 (conservative in their favour).

This gate also validated the **export path for large** — `convert_to_hf` produces a correct
`modernbert` config (28L/1024/16/2624, vocab 150016, rope 160000/10000, `mask_token_id: 4`),
closing the last unticked item of `docs/TRAINING.md` §8.

### ba36000 — 19.8B tokens — the real decision gate
Submitted as job `21736912`. **Reading:** base's own curve reached 45.0 at 41.8B tokens, so large
beating ~45 at half that token count is the strong signal that the warm start is compounding.

## 6. Operational rules learned

- **Never edit a wrapper script while a job is executing it** — bash reads scripts incrementally and
  the job dies mid-parse (this killed smoke `21316657` after a clean 60/60 run). All large slurm
  jobs now snapshot their wrapper to `$SLURM_TMPDIR` first.
- **Propagate exit codes.** A trailing `echo` made a crashed probe report `COMPLETED 0:0` — a silent
  green light. Both smoke scripts now exit non-zero on failure.
- **Partitions: ONLY `p_b200_nlp` and `p_b200_goldberg`**, account `ug_goldberg`. Not
  `p_b200_tsarfaty`, and not the general `B200-4h`/`B200-8h`/`H200` lanes **even when they have free
  GPUs and the allowed lanes are saturated** — wait, or run at fewer GPUs.
- **Run gates on the 1-GPU Goldberg lane** so they never compete with phase-0's 4-GPU allocation.
- **Preflight before every launch**: `python scripts/preflight_check_configs.py yamls/main/base_hebrew_large/*.yaml`
  replicates the `FlexBertConfig` validators without a GPU (they otherwise fire only after a queue
  wait). Run it under `bert-b200` for the strict, interpolation-resolved version.
- The **1-GPU fallback is genuinely dataloader-bound** (~3.6x penalty at 4608 device batch) — a
  stopgap when the 4-GPU lane is blocked, not a comfortable steady state.

## 7. Open items

- [ ] **ba36000 gate** (job `21736912`) — the real 20B decision point.
- [ ] Phase-1 (40B @ 8192, packed) — **re-probe the microbatch**; it is a different throughput
      regime, do not extrapolate phase-0's numbers. Start well below base's 16.
- [ ] Phase-2 (15B curated anneal @ 8192).
- [ ] `--mem` 480GB -> 1200GB on a future fresh `sbatch` (staged in the job file).
- [ ] **No Hebrew NLU/GLUE eval exists** — decide whether to wire one up while phase-0 runs.
- [ ] Track B (representation transfer) is scoped separately; large is not expected to close BeIR.

**Release policy:** weights + tokenizer + code + recipe are public; the **MAFAT training corpus is
private and must never be published.**
