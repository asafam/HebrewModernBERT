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

Training **requires** B200 GPUs and may use **only** these partitions (institutional):
`p_b200_nlp` (4-GPU shared QOS, account `ug_goldberg`), `p_b200_goldberg` (1 GPU),
`p_b200_tsarfaty`. **Not** the general B200-4h/H200 partitions. CPU data-prep that needs an
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
6. **`loss_kwargs.inplace_backward: true`** — flash `CrossEntropyLoss` over the 150K vocab allocates a ~16GB `dlogits` copy in backward; DDP buckets+NCCL ate the last GB → OOM. inplace reuses the logits buffer.

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

## 8. Next: the LARGE model — checklist

The `yamls/main/large/` configs exist (28L / 1024 hidden / 16 heads, vs base 22L/768) but
the context-ext ones currently have only **part** of the §5 fix stack. **Before launching
large phase-1/2, port every §5 fix and re-tune memory** (the large model is ~2× the
params, so OOM thresholds are tighter):

- [ ] Build/verify a **large packed dataset** at 8192 (reuse `pack_mds.py`/`decompress_mds.py`).
- [ ] `data_local` → the raw packed dir; **`streaming: false`** on train+eval loaders.
- [ ] Confirm `src/text_data.py` `NoStreamingDataset` bytes-`input_ids` patch is present (it is, repo-wide).
- [ ] **`loss_kwargs.inplace_backward: true`** + `load_weights_only: true`.
- [ ] `expandable_segments:True` + per-run/persistent triton & inductor caches in the large train scripts.
- [ ] **Re-find the microbatch** for the large model at 8192 — start *well below* base's 24 (try 8–12) and validate on 1 GPU (no DDP) first; the 150K-vocab logits dominate and scale with the larger hidden size. Stay under the rotary int32 limit regardless.
- [ ] Leave warmup off to match base (proven harmless), but **watch the eval anneal** on the first ~10B tokens; if it doesn't decline, add a short warmup via a warmup-capable scheduler.
- [ ] Validate the export path on the large model (`convert_to_hf` → HF ModernBert → Hebrew fill-mask) before declaring done.
- [ ] Gate the large run on the **base model's downstream eval** going smoothly (per project direction).

Throughput will be lower than base; budget GPU time accordingly and use the §6 contention
playbook (greedy-but-safe scaling, never idle, requeue-aware).

---

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
