# Training HebrewModernBERT — Process & Lessons Learned

This document records how the **base** Hebrew ModernBERT was trained end-to-end on B200
(Blackwell) GPUs, and the (many) non-obvious lessons learned. It is the canonical
reference for re-running training — in particular the **large** model, which is the next
effort and should not have to repeat the gauntlet below.

Status: **base model fully trained, exported to HF, validated** (2026-06).
Final checkpoint: `checkpoints/base/modern-bert-base-phase-2-contextextension/ckpt/latest-rank0.pt`
HF export: `outputs/hf/HebrewModernBERT-base-final` (modernbert, 8192 ctx, rope 160000, vocab 150016).

---

## 1. The curriculum (4 phases)

Each phase is a separate Composer run fed the previous phase's **weights** (not full
state) via `load_path` + `load_weights_only: true` (fresh optimizer/schedule/clock).

| Phase | YAML (`yamls/main/base_hebrew/`) | Data | Seq len | Tokens | Notes |
|---|---|---|---|---|---|
| 0.1 pretrain | `flex-bert-rope-phase-0.1-pretrain.yaml` | mixed (H/E/code) | 1024 | ~100B | broad MLM, 30% mask |
| 0.2 pretrain | `flex-bert-rope-phase-0.2-pretrain.yaml` | hebrew | 1024 | ~100B | Hebrew specialize, WSD |
| 1 context-ext | `flex-bert-rope-phase-1-contextextension.yaml` | **packed** hebrew | **8192** | 30B | RoPE θ 160000/10000, `constant_with_warmup` |
| 2 anneal | `flex-bert-rope-phase-2-contextextension.yaml` | **packed** hebrew | **8192** | 50B | `one_minus_sqrt` LR→0 |

Final eval (apples-to-apples): LCE 0.921 → **0.883**, MaskedAccuracy **81.7%** — monotonic
decline then plateau across phase-2's anneal.

---

## 2. Environment — `bert-b200` (Blackwell, sm_100)

The original `bert24` env (torch 2.4/cu124) has **no Blackwell kernels** → crashes on B200
("no kernel image available"). Built `bert-b200` via `scripts/build_b200_env.sh`
(`.slurm/jobs/build_b200_env.slurm`, run on `p_b200_goldberg`). Validated matrix:

- CUDA 12.8 toolkit (conda, via **micromamba** — conda's classic solver zombies for hours on the nvidia channel)
- torch 2.7.0+cu128, **triton 3.3.1** (pinned **last** — peft/accelerate re-resolve torch's 3.3.0 pin)
- flash-attn 2.7.4.post1 built from source for `TORCH_CUDA_ARCH_LIST=10.0` (**not** FA3 — fails on sm_100; FlexBERT guards that import)
- transformers 4.57, mosaicml/composer 0.31.0, peft + accelerate (convert needs peft)
- extra deps a fresh env needs: numba, sentencepiece, protobuf

### THE Blackwell gotcha — fresh triton cache per job
flash-attn's triton rotary kernel crashes with `SystemError: PY_SSIZE_T_CLEAN macro must be
defined for '#' formats` at triton `_init_handles` **if** the shared `~/.triton` cache is
reused across triton versions (stale cubin). It was never a Blackwell or version problem —
just cache cross-contamination. A stock triton kernel works fine with a clean cache.
**Fix:** run scripts set `TRITON_CACHE_DIR` (and `TORCHINDUCTOR_CACHE_DIR`) to per-run
paths. Once the env/version is frozen, a **persistent** cache is safe and skips the ~10min
recompile on each requeue.

Code patches the new libs required (committed): `tokenizer._pad_token` → `tokenizer.pad_token`
(transformers ≥4.48 removed the private attr).

---

## 3. Data — packing for context extension (critical)

The raw Hebrew corpus is **short-document** (median ~100 tok, 92% < 1024). At
`max_seq_len: 8192` with the default **one-doc-per-sample** dataloader, the model sees
~337-token sequences on average and **never exercises long context** — context extension
would be a no-op (and wasteful: ~234K tok/s, mostly padding).

**Fix:** pack documents into dense 8192-token sequences (the ModernBERT / Fu et al.
approach), then point context-ext phases at the packed data.

```bash
# 1) pack raw text MDS -> dense 8192 input_ids (zstd) on an infinite-wall partition
sbatch .slurm/jobs/pack_mds_hebrew.slurm          # src/data_prep/pack_mds.py
# 2) decompress zstd -> raw .mds (NoStreamingDataset cannot read zstd)
sbatch .slurm/jobs/decompress_mds.slurm           # src/data_prep/decompress_mds.py
# result: data/hebrewmodernbert/hebrew_packed_8192_raw/{train,validation}
```

Packing throughput was read-bound (StreamingDataset per-doc zstd decode), not tokenizer-
bound — `pack_mds.py` reads via a `DataLoader(num_workers=...)` and uses a numpy chunk
buffer (Python-list slicing was O(n)/chunk).

---

## 4. Running — Slurm + the partition constraint

Training **requires** B200 GPUs and may use **ONLY these two partitions** (user-confirmed
2026-08-11): **`p_b200_nlp`** (4-GPU shared QOS) and **`p_b200_goldberg`** (1 GPU, Yoav Goldberg's
lane) — both with `--account=ug_goldberg`. **NOT** `p_b200_tsarfaty` (previously listed here in
error), and **NOT** the general `B200-4h`/`B200-8h`/`H200` partitions, even when they have free GPUs. CPU data-prep that needs an
infinite wall has used `p_glickman` (account `ug_cs_dsi`) as an approved exception.

Train scripts (`scripts/train_modernbert_base_hebrew_phase-*.sh`) activate `bert-b200`, set
the caches + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, and run
`python -m composer main.py <yaml>`. Slurm jobs in `.slurm/jobs/train_b200_phase-{1,2}.slurm`
(4h wall + `--requeue`; YAML has `autoresume: true` + `spin_dataloaders: false`).
Override GPU count on the CLI: `sbatch --gres=gpu:N <slurm>`.

Eval-only on a checkpoint: `scripts/eval_checkpoint.py` (reuses `main(do_train=False)` +
`trainer.eval()`), e.g. `.slurm/jobs/eval_phase1_checkpoints.slurm`.

---

## 5. The context-extension fix stack (6 distinct bugs)

Phase-1 at dense 8192 on multi-GPU needed **six** fixes beyond the base env, each surfacing
only at full scale. **All of these must be present in any context-ext config** (the base_hebrew
phase-1/2 have them; verify before the large run — see §8):

1. **Packed data** (§3) — else no long context is seen.
2. **`data_local: .../hebrew_packed_8192_raw`** + **raw (uncompressed)** MDS — `NoStreamingDataset` has no zstd decode.
3. **`streaming: false`** (train+eval loaders) — `StreamingTextDataset`'s multi-GPU `get_shm_prefix`+`dist.barrier` NCCL-cascades at init on >1 GPU.
4. **Patched `NoStreamingDataset.__getitem__`** (`src/text_data.py`) — its `input_ids` branch assumed ndarray encoding and `del`'d our **bytes**-encoded `input_ids` → `KeyError`. Now `np.frombuffer` for bytes.
5. **`device_train_microbatch_size: 24`** (base) — dense 8192 forward activations + the 150K-vocab logits OOM above this. Also stays under the flash-attn **triton rotary int32 limit** (~96; `mb=144` → 1.81B elems → illegal memory access). For sustained 4-GPU runs **16** is safer (masking-variance spikes + fragmentation; combine with `expandable_segments`).
6. **`loss_kwargs.inplace_backward: true`** — ⚠️ **CORRECTION (2026-08-11): this setting is a
   NO-OP**; `FlexBertConfig.__init__` unconditionally resets it to False (see §8). The
   reasoning below is why it was *added*, not what actually happened. Flash `CrossEntropyLoss` over the 150K vocab allocates a ~16GB `dlogits` copy in backward; DDP buckets+NCCL ate the last GB → OOM. inplace reuses the logits buffer.

Plus `expandable_segments:True` (env) to kill allocator fragmentation that OOM'd a 16GB
logits alloc mid-run (~step 61), and `load_weights_only: true` (else inherits the previous
phase's clock → "max_duration <= elapsed").

**On warmup:** the context-ext phases reset the optimizer (`load_weights_only`) but use
`t_warmup: 0` (phase-1 `constant_with_warmup`) / no warmup (phase-2 `one_minus_sqrt`). This
is theoretically wrong (a reset optimizer should warm up), and the project plan flagged it.
**But it caused no measurable harm** — phase-1/2 eval improved monotonically when measured
apples-to-apples. Verified by re-evaluating checkpoints, not the W&B curve (see §7). LR is a
*drop* (8e-4→3e-4) and phase-2's LR only decreases, so no-warmup is low-risk. Don't add it
on speculation; if a future run's eval *doesn't* anneal down, revisit. `one_minus_sqrt` has
no `t_warmup` — would need `WarmupStableDecayScheduler`/`CosineInverseSqrtScheduler`.

---

## 6. GPU contention strategy (shared `p_b200_nlp` QOS)

`p_b200_nlp` is a **shared QOS capped at 4 GPUs**, `PreemptMode=OFF`, and our fair-share
priority is **lower** than the heavy users on it. Consequences and the playbook:

- **Holding a running GPU is the only real leverage** (no preemption → can't be evicted). **Never cancel a running job speculatively** — on a full node the released GPU is lost to a higher-priority renewal. (Cost us ~26h idle once.)
- **Don't idle** — if blocked on 4 GPUs, grab whatever's free (even 1) and train; the model is identical regardless of GPU count (global batch held constant via grad accum; 1 GPU is just slower, not worse).
- **Greedy scale-up** (1→2→3→4) via cancel+resubmit at a higher `--gres`, but **only when ≥2 GPUs are genuinely free** (margin against the release-gap race) — and **fall back** to the largest grabbable count if the bigger request PDs. Aggressive (1-free) scale-ups repeatedly lost the release-gap.
- **Requeue is the danger point**: at the 4h wall the job releases all GPUs and competitors refill the node; reclaim is uncontested only if you're the sole QOS user. Adjust the GPU request down at the requeue if competitors returned.
- The clean fix to all of this is a **SLURM reservation** (overrides priority, holds GPUs through requeues) — pursue it for long runs if possible.

The base run climbed 1→4 and finished; 50B anneal took ~3.5h once stably on 4 GPUs
(~970K tok/s) vs ~57h on 1 GPU (~244K tok/s).

---

## 7. Monitoring & loss tracking

- Context-ext runs set `disable_train_metrics: true`, so the **tqdm train-loss is a frozen
  artifact** — ignore it. The real signal is **eval** (in W&B), pulled via
  `python scripts/track_loss.py <project> [run_id]`.
- **Keep `eval_subset_num_batches` constant within a phase.** Changing it mid-run (100→50
  for speed) makes the logged eval points **non-comparable** and produces a fake "rising
  loss" trend. This caused a false alarm that nearly triggered a needless multi-day redo —
  the discriminating test was re-evaluating two checkpoints with **identical** config
  (`scripts/eval_checkpoint.py`), which showed the loss was actually *falling*.
- **Measure before you commit.** Don't act on a noisy/non-comparable trend; a 15-min eval
  beats a 3-day redo.
- **W&B run id + GPU-count scale-ups don't mix:** a stable `init_kwargs.id` + `resume`
  makes Composer resume the run, but W&B locks `num_gpus_per_node` — changing GPU count
  (scale-up) → `ConfigError` crash (`allow_val_change` does **not** fix it). With scaling,
  use **fresh run per launch** (no stable id) and sweep clutter periodically. Re-add a
  stable id only once the GPU count is fixed (steady-state requeues at the same count
  resume cleanly).

---

## 8. The LARGE model — prepared recipe and runbook

**Status (2026-08-11): configs, scripts and slurm jobs are written and preflight-clean. Nothing
has been launched.** The model is 28L / 1024 hidden / 16 heads / 2624 intermediate =
**~498M params** (base: 22L/768/12/1152, ~226M) — 2.20x the parameters but **3.10x the
non-embedding** parameters, which is what sets the slowdown.

### What was built

| Path | What |
|---|---|
| `yamls/main/base_hebrew_large/flex-bert-rope-phase-{0-pretrain,1-contextextension,2-contextextension}.yaml` | The recipe. Derived from `base_hebrew/`, **not** from the stale `yamls/main/large/`. |
| `scripts/verify_tile_init.py` | Gate: proves the base→large warm start actually took. |
| `scripts/preflight_check_configs.py` | GPU-free replication of the `FlexBertConfig` validators. |
| `scripts/smoke_test_large_phase-{0,1}_b200.sh` | Tile-init check + microbatch/throughput probes. |
| `scripts/train_modernbert_large_phase-{0-pretrain,1,2}*.sh` | Launchers (persistent caches, GPU-count-suffixed W&B id, `RUN_NAME`/`MLM_PROB`/`SAVE_INTERVAL` arm-forking). |
| `.slurm/jobs/{smoke,train}_b200_large_*.slurm` | 1-GPU and 4-GPU jobs, 4h + `--requeue`, with a 1-GPU fallback for every phase. |
| `.slurm/jobs/eval_large_checkpoints.slurm` | Fixed-config checkpoint re-eval (the §7 discriminator). |

**Script conventions are aligned with the base run's most-evolved script**
(`train_modernbert_base_hebrew_phase-0-pretrain.sh`, 2026-07-30), forward-ported to all three
large phases — the base phase-1/2 launchers date from 2026-06-22 and predate both fixes:

- **`RUN_NAME` / `MLM_PROB` / `SAVE_INTERVAL` env overrides** so an arm can be forked without
  editing YAML: `sbatch --export=ALL,RUN_NAME=large-arm-m40,MLM_PROB=0.4 ...`. Unset, all three
  fall through to the YAML.
- **GPU-count-suffixed W&B id** (`${RUN_NAME}-g${NGPUS}`) on *every* phase. This supersedes base
  phase-1/2's "no stable id, fresh run per launch" workaround: suffixing keeps one continuous
  curve per GPU count while still sidestepping the `ConfigError`, and checkpoint resume is keyed
  on `run_name`, not the W&B id, so it costs no progress. Expect to bounce between 1/2/4 GPUs.
- **Persistent triton/inductor caches** for the long phases; fresh per-job caches for smokes.
- `--mail-type=END,FAIL,REQUEUE` on the requeueing train jobs, matching base phase-1/2.

`scripts/mlm_compare_vs_neodictabert.py` now takes `--extra-model LABEL=PATH` and `--only`, so the
large gates can score a new checkpoint without editing its hardcoded `MODELS` dict; the slurm job
passes them through as `EXTRA=` / `ONLY=`.

**`yamls/main/large/` is DEPRECATED** — a Jun-18 draft predating the overhaul. Do not launch it;
`preflight_check_configs.py` fails it on the missing `sequence_packing` (the exact defect that
collapsed base retrieval).

### The warm start (the key difference from how base was trained)

Large is **initialized from the trained base** by Phi-style weight tiling — ModernBERT-large's own
recipe — rather than from scratch. Source: `checkpoints/base/phase-0-arm1-m20/ckpt/latest-rank0.pt`
(the 130B-token, 20%-masking phase-0 endpoint). Deliberately **not** the phase-2 final, which is
annealed to LR≈0 and whose query/document alignment flipped negative (+0.914 → −0.269).

Two traps, both already handled in the config, both worth understanding before editing it:

1. **`checkpoint_cfg: ${model}` is wrong in every commented-out example in this repo.** It
   resolves to the enclosing file's *own* model block, so in a large YAML it builds a large model
   and then loads a base state dict into it. The large phase-0 YAML therefore carries a separate
   literal `base_model:` block (a full model block — `init_from_checkpoint` reads
   `.pretrained_model_name`), verified to match `base_hebrew` phase-0 exactly.
2. **`tile_embedding(...tok_embeddings...)` was commented out** at
   `src/bert_layers/model.py` in `init_model_from_pretrained`. With `tie_word_embeddings=True`
   (the HF default, unset in our YAMLs) the decoder is re-tied to `tok_embeddings`, so leaving it
   commented left the **150016x1024 embedding matrix — 31% of the model — at random init** while
   the loss curve still looked plausible. Now uncommented, with a comment saying why.

Verify it took: `python scripts/verify_tile_init.py <yaml>` — trained base embeddings have
**std 0.113**; a fresh `init_std: 0.02` init is ~0.02. The smoke script runs this and aborts on
failure before spending GPU time.

### Two settings that look active but are not

Both were found while bringing up the large run, both are kept for parity with the base run (which
produced its numbers under the same no-ops), and both are annotated in the configs:

1. **`loss_kwargs.inplace_backward`** — `FlexBertConfig.__init__`
   (`src/bert_layers/configuration_bert.py:227-229`) unconditionally resets it to False, and
   `self.loss_kwargs` is the same dict object `get_loss_fn` later reads. So the 8192 phases never
   had it. What actually carries the 150K-vocab backward is the small microbatch +
   `masked_prediction: true` + `expandable_segments:True`. Do not budget memory assuming it is on.
   If large OOMs at 8192, making that guard conditional on `disable_train_metrics: true` (already
   set in phases 1/2, so "incorrect metrics" is moot) is an available lever — measure first.
2. **`batch_size_warmup_min_size` / `batch_size_warmup_tokens` in phase-0** — `src/text_data.py`
   reads these only inside `if not streaming and sequence_packing:`, and phase-0 sets no
   `sequence_packing`. **No batch-size ramp has ever run in phase-0.** The ramp is live in
   phases 1/2, which do set `sequence_packing`.

### Runbook

```bash
# 0) preflight (seconds, no GPU)
python scripts/preflight_check_configs.py yamls/main/base_hebrew_large/*.yaml

# 1) smoke: tile-init gate + microbatch/throughput probe (1 GPU, private lane, ~30 min)
sbatch .slurm/jobs/smoke_b200_large_phase-0_1gpu.slurm
#    then pin the reported microbatch in the phase-0 yaml (back off one notch for compile=true)
#    and re-run once with --export=ALL,COMPILE=true,MB=<N> to de-risk

# 2) phase-0: 130B tokens @1024, warm-started
sbatch .slurm/jobs/train_b200_large_phase-0_nlp4.slurm        # 4 GPU
sbatch .slurm/jobs/train_b200_large_phase-0_goldberg1.slurm   # 1 GPU, never-idle fallback

# 3) phase-1: re-find the microbatch at 8192 on 1 GPU FIRST, then 40B tokens
sbatch .slurm/jobs/smoke_b200_large_phase-1_1gpu.slurm
sbatch .slurm/jobs/train_b200_large_phase-1.slurm

# 4) phase-2: 15B curated anneal
sbatch .slurm/jobs/train_b200_large_phase-2.slurm
#    1-GPU fallback for any phase when the 4-GPU quota is blocked -- never idle (§6):
#    sbatch .slurm/jobs/train_b200_large_phase-{0_goldberg1,1_goldberg1,2_goldberg1}.slurm

# 4b) sanity-check a suspicious eval trend with FIXED config (never trust a curve whose
#     eval_subset_num_batches changed mid-run -- see §7)
sbatch --export=ALL,CKDIR=checkpoints/large/modern-bert-large-phase-0-pretrain/ckpt,BAS="10000 20000" \
  .slurm/jobs/eval_large_checkpoints.slurm

# 5) export (already parameterized -- no code change needed for large)
YAML=yamls/main/base_hebrew_large/flex-bert-rope-phase-2-contextextension.yaml \
CKPT=checkpoints/large/modern-bert-large-phase-2-contextextension/ckpt/latest-rank0.pt \
OUTPUT_NAME=HebrewModernBERT-large-final \
  bash scripts/convert_to_hf_hebmodernbert.sh
```

### Quality gates (don't wait until the end)

- **After tile-init, before training**: convert the step-0 model and run
  `scripts/mlm_compare_vs_neodictabert.py`. A correctly tiled large model should already score in
  the base model's range. If it scores ~0, the warm start failed silently — stop.
- **At ~15-20B tokens**: convert + MLM-compare against base's curve at the matched token count.
- Reference points, whole-word recovery: base phase-0 **46.3**, base final **48.6**,
  NeoDictaBERT **49.9**.

### Budget — MEASURED 2026-08-11 (jobs 21316657 / 21316665 / 21316666)

| | |
|---|---|
| `device_train_microbatch_size` | **576** (auto-search 4608→2304→1152→576; `compile=true` re-validated, no OOM) |
| real tokens per batch | **549,523** (= 130B / 236,569 ba from base; independently confirmed by speed_monitor at 119.3 tok/sample) |
| throughput, 4x B200, compile=true | **~550K real tok/s total (~137K/GPU)**, 1.0 s/ba |
| scaling | linear: 1-GPU steady state was also ~137K tok/s |
| **phase-0 (130B tokens)** | **~66 h of 4-GPU compute (~2.7 days)** |

Phases 1-2 run at 8192 with sequence packing, a different throughput regime — do not extrapolate the
above to them; re-probe with the phase-1 smoke.

Add requeue overhead (~17 requeues against the 4h wall; small with the persistent inductor cache)
and shared-QOS contention, which is the dominant real-world variable — see §6.

**RESOLVED (2026-08-11).** Phase-0 initially sustained only **~180K tok/s (3.05 s/ba)** on 4 GPUs
against a GPU-bound ceiling of ~550K. Cause: the dataloader was **I/O-latency bound** — `sstat`
showed only **~7 of 32 allocated CPUs** busy, i.e. workers were blocked on NFS reads, not computing.
Fix: `train_loader.num_workers` 8 -> **24**, `prefetch_factor` 2 -> **4** (eval loader left at 8).
Applied by editing the YAML and letting the **4h-wall requeue** pick it up — no cancel, no lost
progress, no surrendered GPUs.

| | throughput | s/ba | phase-0 ETA |
|---|---|---|---|
| before | ~180K tok/s | 3.05 | ~8.2 d |
| **after** | **~557K tok/s** (15-min peak 596K) | **0.99** | **~2.6 d** |

It needed **no extra CPUs** (still 32). Raising `--cpus-per-task` would have done nothing — the
cores were idle, not saturated. RSS rose from 374GB to ~969GB (mmap'd page cache the job previously
could not hold), so `--mem` 480GB -> 1200GB is staged in the job file as a further improvement; it
only applies on a fresh `sbatch`, not an automatic requeue.

**Method note — the probes that misled us.** Two 30-batch probes reported ~550K and concluded "the
dataloader is fine". They ran after an 80-200s `torch.compile` during which the workers filled the
prefetch queue, so every batch was served from that buffer at GPU speed. **A dataloader probe must
run long enough to drain `num_workers x prefetch_factor` batches AND dwarf the compile prefill**, or
it measures burst, not sustained, throughput. Judge sustained rate from 5-minute windows of a live
run instead. The earlier 1-GPU stall observation was directionally correct and was wrongly dismissed.

The superseded A/B (30 batches, burst-limited, kept as the record of the mistake):

| | num_workers / prefetch | cpus / mem | burst s/ba |
|---|---|---|---|
| A | 8 / 2 | 32 / 480G | 1.0 |
| B | 16 / 6 | 64 / 960G | 1.0 |

Both were prefetch-buffer-limited, so the comparison was uninformative in either direction.

### Data budget — DECIDED: the corpus is frozen (2026-08-11)

The large run uses the **same `hebrew_quality` corpus, the same 130B/40B/15B budgets, and the same
20% masking as the validated base run**. No rebuild, no re-weighting, no new sources.

The question that was worked through and closed: large is tile-initialized from a base that already
consumed this corpus, so is the repeated exposure past the data-constrained repetition knee? **No —
and the risk runs the other way.**

- NeoDictaBERT ran **6.0 epochs** over ~46B unique (278B total). The base retrain ran **4.7** — the
  most conservative of the models being compared.
- Base's own MLM improved monotonically across all 4.7 epochs (45.0 @ba76000 → 46.3 @phase-0 end →
  48.6 @phase-2 end). No knee was observed, and large has **3.1x base's non-embedding capacity**
  (343M vs 110M), so it can extract more from the same tokens than base could.
- Token intensity, not repetition, is where this model is starved: ModernBERT-base ~19,100 tok/param
  (per the upstream configs in `yamls/main/base/`), NeoDictaBERT ~600, base retrain ~820,
  **large as configured ~370**.

Two alternatives were considered and rejected **for this run**:

- *Re-weight the mix for a longer phase-0* (curated pinned at 5x, clean web pushed to 6-8x — same
  ten Hebrew files, no language change). Rejected: it would put `hec4_clean` individually at ~8
  epochs over 8.63B unique, trading an audited non-uniform per-source schedule for an unmeasured
  one, to chase a proxy metric.
- *Acquire more unique Hebrew* (Knesset / Ben-Yehuda / Sefaria, which the mix config itself names as
  the only way past the ~0.8B curated floor). This is the real long-term lever but is **not feasible
  at this stage** — recorded here so it is not re-proposed as a quick fix.

**The mid-run gate is what would reopen this.** A data-starved model plateaus; a capacity-starved
one keeps climbing. If large's MLM curve is still rising at the phase-0 end, the budget was right.

One deliberate deviation, which is NOT a corpus change: the large train loaders set
`shuffle_seed: 71`. Large inherits base's weights, so replaying the identical document order,
packing and mask positions would be the most redundant possible second pass. Set on the **train**
loader only — the eval loader and `seed: 17` are untouched, so the eval curve stays directly
comparable to base's.

### Scope note

Per `docs/RETRIEVAL_EVAL.md` (2026-08-11), HMB-base is already at **MLM parity with
NeoDictaBERT** (48.6 vs 49.9) while scoring 0.185 vs 0.332 on BeIR — the retrieval gap is a
representation-*transfer* problem. Large is therefore justified as a stronger encoder and as the
second member of a released family, **not** as the lever that closes BeIR. Note also that this
repo has **no Hebrew NLU/GLUE eval wired up**, so `mlm_compare_vs_neodictabert.py` is currently
the only apples-to-apples quality signal for large.

## 9. Export / release

```bash
# CPU-only; runs in bert-b200 (needs peft); 8192 ctx, packed-tokenizer ids
python ./src/convert_to_hf.py \
  --yaml-config yamls/main/base_hebrew/flex-bert-rope-phase-2-contextextension.yaml \
  --input-checkpoint checkpoints/base/modern-bert-base-phase-2-contextextension/ckpt/latest-rank0.pt \
  --output-name HebrewModernBERT-base-final --output-dir ./outputs/hf \
  --bos-token-id 2 --eos-token-id 3 --cls-token-id 2 --sep-token-id 3 \
  --pad-token-id 0 --mask-token-id 4 --max-length 8192 --vocab-size 150016
```

Validated: `config.json` (`model_type=modernbert`, 8192 ctx, rope 160000) + tokenizer
(`[MASK]`=4) load and produce sane Hebrew fill-mask. Known cosmetic issue:
`config.mask_token_id` writes as `null` (the tokenizer carries `[MASK]`=4, so fill-mask
works) — a future `convert_to_hf` patch could write it.

**Release policy:** model weights + tokenizer + code + recipe are public; the **MAFAT
training corpus is private and must never be published.**
