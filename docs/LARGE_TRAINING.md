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

> **RULE — compare within a single job's output only, and never mix the metric tables.**
> The script emits three tables (whole-word EXACT, whole-word by decoded string, and per-TOKEN);
> per-token reads several points higher, so quoting one gate's token table against another's word
> table invents a difference. Cross-*run* drift is also real but has a specific cause: the recorded
> v2 numbers (NeoDictaBERT 49.9 / base phase-0 46.3) predate a change to
> `scripts/mlm_compare_vs_neodictabert.py` that added `span_len`/`rare_pct`, so they are not
> comparable to post-change gates (46.9 / 44.0). Gates run after that change ARE mutually comparable
> — the references reproduce **exactly** — which is why every gate scores the references alongside
> the model under test.

### Progress vs baselines — whole-word MLM recovery (all post-change, mutually comparable)

**HEADLINE (2026-08-25): at 94% of phase-0, large is at MLM PARITY with NeoDictaBERT** — before
context extension and annealing have contributed anything.

| model | tokens | arguana | fiqa | nfcorpus | scidocs | scifact | **MEAN** |
|---|---|---|---|---|---|---|---|
| NeoDictaBERT *(baseline)* | — | 43.7 | 45.3 | **51.8** | **52.7** | 41.2 | **46.9** |
| **HMB large @ba222000** | **122B (94%)** | **45.0** | **45.6** | **51.8** | 48.9 | **42.1** | **46.7** |
| HMB large @ba36000 | 19.8B (15%) | 43.7 | 45.9 | 48.7 | 47.2 | 39.8 | 45.0 |
| HMB base phase-0 FINAL *(tile source)* | 130B (100%) | 41.3 | 45.4 | 49.1 | 45.9 | 38.4 | 44.0 |
| HMB large @ba6000 | 3.3B (2.5%) | 40.6 | 43.8 | 49.3 | 44.7 | 37.2 | 43.1 |

Monotonic closure across three gates:

| gate | vs base phase-0 FINAL | vs NeoDictaBERT |
|---|---|---|
| ba6000 (2.5%) | -0.9 | -3.8 |
| ba36000 (15%) | +1.0 | -1.9 |
| **ba222000 (94%)** | **+2.7** | **-0.3** |

Large **wins or ties 4 of 5 datasets** (arguana +1.3, scifact +0.9, fiqa +0.3, nfcorpus tied). The
whole remaining gap is **scidocs (-3.8)** — one concentrated weakness, not a broad deficit. Worth
investigating: scidocs is also where base was weakest relative to NeoDictaBERT.

*Calibration:* ~2,700 scored words => ~1.0 point independent SE, so -0.3 is inside noise. Call this
**parity, not a win.** The comparison is paired (identical words and masks per model) and the
references reproduce exactly across all three gates. Sanity: 2,695-2,699 words, 0% dropped, HMB 1.38
tok/word vs NeoDictaBERT 1.30 (conservative in their favour).

*Headroom:* for base, phase-1 + phase-2 added **+2.3** on this metric (46.3 -> 48.6, pre-script-change
calibration). If large gains comparably, it ends **clearly ahead** of NeoDictaBERT rather than level.

**Caveat that has not changed:** this is MLM. Per `docs/RETRIEVAL_EVAL.md`, base was *already* at MLM
parity with NeoDictaBERT while scoring 0.185 vs their 0.332 on BeIR — the retrieval gap is a
representation-transfer problem. **No BeIR evaluation of large exists**, and MLM parity does not
predict retrieval.


### Trajectory vs NeoDictaBERT (all gates mutually comparable)

| checkpoint | tokens | **MEAN** | vs NeoDictaBERT (46.9) | vs base phase-0 FINAL (44.0) |
|---|---|---|---|---|
| phase-0 @ba6000 | 3.3B | 43.1 | -3.8 | -0.9 |
| phase-0 @ba36000 | 19.8B | 45.0 | -1.9 | +1.0 |
| phase-0 @ba222000 | 122B | 46.7 | -0.3 | +2.7 |
| **phase-1 @ba5000** | **+2.7B @8192** | **47.0** | **+0.0** | **+3.0** |

**Context extension is additive, not destructive.** This was the key risk: base's phase-1 damaged
retrieval because dense packing let tokens attend across document boundaries (0.165 -> 0.142).
Large's `sequence_packing: true` (per-document `cu_seqlens`) holds — phase-1 *improved* MLM, and
notably lifted **scidocs 48.9 -> 49.6**, which had been the entire remaining deficit vs NeoDictaBERT.

At ~1.0 point SE, +0.0 is parity rather than a lead. Phase-1 was only ~13% through its 40B at this
checkpoint, and the 15B anneal follows; for base, phases 1+2 together added +2.3 on this metric.

**Unchanged caveat:** this is MLM. Base was *already* at MLM parity with NeoDictaBERT while scoring
0.185 vs their 0.332 on BeIR. MLM does not predict retrieval — no BeIR eval of large exists, and it
requires the finished model plus dual-encoder SFT in the sibling repo.

### ba36000 — the decision gate: **PASS** (table above)

| model | arguana | fiqa | nfcorpus | scidocs | scifact | **MEAN** |
|---|---|---|---|---|---|---|
| NeoDictaBERT | 43.7 | 45.3 | 51.6 | 52.5 | 41.2 | **46.9** |
| HMB phase-0 FINAL *(tile source, full 130B)* | 41.3 | 45.2 | 48.9 | 45.9 | 38.2 | **43.9** |
| **HMB large @ba36000** | **43.7** | **45.9** | 48.4 | 47.2 | 39.6 | **45.0** |

**Large has overtaken the model it was tiled from, at 15% of the token budget** — base phase-0 FINAL
is the complete 130B-token model. It also matches NeoDictaBERT on arguana and beats it on fiqa.

Absolutes shift between gates (NeoDictaBERT read 49.9 / 48.6 / 46.9 across three runs), so the
calibration-independent measure is the **delta against the in-run references**:

| gate | vs base phase-0 | vs NeoDictaBERT |
|---|---|---|
| ba6000 (3.3B, 2.5%) | -0.4 | -4.2 |
| **ba36000 (19.8B, 15%)** | **+1.1** | **-1.9** |
| movement | **+1.5** | **+2.3** |

The gap to NeoDictaBERT **halved** between the two gates.

*Calibration caveat:* at ~2,700 scored words the independent SE on a ~44% rate is ~1.0 point, so
+1.1 alone is ~1 SE. The comparison is paired (identical words and masks across models), which
tightens it, and the trend moving the same direction on **both** references across two independent
gates is more persuasive than either point alone. Treat as a solid positive signal, not a proven
margin. Sanity: 2,695-2,699 words, 0% dropped, HMB 1.38 tok/word vs NeoDictaBERT 1.30.

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


## 8. Autonomous completion chain (set up 2026-08-25)

Phase-0 took **14 calendar days for ~29 h of compute** — it held GPUs only **~20% of the time**.
The training never failed (20 requeues, all clean); the loss was purely GPU contention on the shared
`p_b200_nlp` QOS, and **nobody worked the §6 contention playbook** because nothing was running
between sessions. The 1-GPU fallback job existed and was never submitted, not once.

**Fix: a pure slurm dependency chain.** Nothing depends on an agent or a login-node process — the
same `--requeue` + `autoresume` machinery that carried phase-0 through 20 restarts now carries the
whole curriculum:

```
21342052  phase-0  (94% done)
24875811  phase-1  --dependency=afterok:21342052
24875812  phase-2  --dependency=afterok:24875811
```

`afterok` (not `afterany`) is deliberate: if a phase fails, the chain **stops** rather than training
the next phase on a broken checkpoint.

**Microbatch lowered 8 -> 4 for phases 1/2.** Base validated mb=16 at 22L/768 at 8192; large's
activation cost scales ~1.70x, so the equivalent is ~9.4 — mb=8 leaves only 1.18x headroom, mb=4
leaves 2.36x. With the chain running unattended, an OOM would break it and cost days of queue time,
while mb=4 costs only extra gradient-accumulation steps. Raise to 8-12 only after a smoke validates
it; the change is picked up automatically at the next requeue (the same mechanism that delivered
the num_workers fix).


### Wall-safe self-requeue (added 2026-08-25, after the chain broke)

**Do not trust slurm's requeue-on-TIMEOUT.** Job `24881738` hit the 4h wall and ended `TIMEOUT`
rather than `REQUEUED`, under the same `#SBATCH --requeue` and the same job file whose predecessor
(`21342052`) had requeued **20 consecutive times**. `afterok` treats TIMEOUT as failure, so phase-1
went to `DependencyNeverSatisfied` and the whole chain died silently. It only surfaced because a
monitor was watching.

All six large train jobs now requeue themselves before the wall instead:

```bash
#SBATCH --signal=B:USR1@180     # USR1 to the BATCH SCRIPT 3 min before the wall
trap _wall_requeue USR1         # _wall_requeue -> scontrol requeue $SLURM_JOB_ID
bash "$SNAP" & CHILD=$!; wait $CHILD   # child MUST be backgrounded or the trap cannot fire
```

The job therefore never reaches TIMEOUT. When training genuinely completes, composer exits 0, the
trap does not fire, the job goes `COMPLETED`, and `afterok` releases the next phase.

Two things to remember:
- The `B:` prefix matters — without it the signal goes to the job steps, not the batch script.
- `wait` on a backgrounded child is required; a foreground `bash "$SNAP"` blocks trap delivery.
- **This path is unvalidated until a phase actually hits a wall.** Phase-0's remaining 3.6B fits in
  one block, so the first real test is phase-1's first wall. Verify it fires; do not assume.

### What still cannot be automated
- **Contention.** A 4-GPU job cannot run on the 1-GPU Goldberg lane, and slurm here rejects a
  flexible `--gpus=1-4` request ("Requested node configuration is not available"). Multi-partition
  submission (`--partition=p_b200_nlp,p_b200_goldberg`) works only for requests that fit both lanes.
- **`CronCreate` jobs do not survive the session**, so a scheduled caretaker agent is not an option
  across the weeks this run spans.
- **The real lever is a SLURM reservation** (§6 flags it) — it needs an admin, not code. At a ~20%
  duty cycle the remaining 55B tokens of phases 1-2 imply **weeks** of calendar time; with a
  reservation it would be days.

### Reporting rule for this run
Quote **calendar** ETAs, not compute-time ones. Phase-0's "~2.4 days" figures were 4-GPU compute
time and read as wall-clock; the honest number at a 20% duty cycle was ~2 weeks.

## 7. Open items

- [x] **ba36000 gate** — PASS (see §5): large +1.1 over its tile source at 15% of the budget.
- [x] Phase-0 reached 94% (ba222000, 122.3B) with loss ~1.86-1.91.
- [x] Gated at ba222000 (94%): MLM parity with NeoDictaBERT (46.7 vs 46.9), +2.7 over base.
- [ ] Gate the true phase-0 FINAL, then phase-1 and phase-2 outputs.
- [ ] Investigate the scidocs gap (-3.8) — the entire remaining deficit vs NeoDictaBERT.
- [x] Phase-1 chained (job 24875811) at a deliberately safe mb=4; optimise to 8-12 only after a smoke.
- [x] Phase-2 chained (job 24875812).
- [ ] **Ask for a SLURM reservation** — the only real fix for the ~20% duty cycle.
- [ ] Prune checkpoints: 111 files / 621GB because Composer's keep-20 tracker resets each requeue.
- [ ] `--mem` 480GB -> 1200GB on a future fresh `sbatch` (staged in the job file).
- [ ] **No Hebrew NLU/GLUE eval exists** — decide whether to wire one up while phase-0 runs.
- [ ] Track B (representation transfer) is scoped separately; large is not expected to close BeIR.

**Release policy:** weights + tokenizer + code + recipe are public; the **MAFAT training corpus is
private and must never be published.**
