# Retrain Request — HebrewModernBERT-base (handoff to a fresh Claude Code session)

> Working dir: `/home/nlp/achimoa/workspace/HebrewModernBERT`
> Branch: `overhaul/tokenizer-config-training`
> Read first: `CLAUDE.md`, the overhaul plan `/home/nlp/achimoa/.claude/plans/let-s-get-to-the-adaptive-clover.md`, and the project memory in `.claude/projects/.../memory/`.

## Motivation (why we are doing this)

A full audit (prior session, 2026-06-29) established two things with high confidence:

1. **The Composer→HF conversion is CORRECT — not the problem.** Re-converting from the phase-2 checkpoint produced 137/137 bit-identical tensors and a byte-identical config; all weights (incl. the MLM head) are present and trained; the run is silent with exit 0. The long-standing worry that "weights get randomized during conversion" is **ruled out**.
2. **The pretrained model is genuinely UNDER-TRAINED — that is the real problem.** Fill-mask smoke tests: it knows easy facts (`בירת ישראל היא [MASK]` → `ירושלים` rank 0) but fails trivial cloze (`השמש זורחת ב[MASK]` → "east" ranks 585; predicts punctuation). BeIR retrieval after dual-encoder SFT is 0.147 avg NDCG@10 — below the older 100K-vocab model (0.161) and far below baselines (mE5-large 0.382, NeoDictaBERT 0.336).

Two minor conversion config bugs were found AND already fixed in `src/convert_to_hf.py` this session (verified):
- `local_rope_theta` now correctly resolves to the trained local base (10000), not the global 160000.
- `mask_token_id` is now written into `config.json` (was missing → `None`).
(Impact of these on the old model was measured as small: fill-mask unchanged; long-doc CLS embedding cosine shift ~0.96. They are correctness fixes, not the cause of low scores.)

**Conclusion:** do not re-debug conversion. Retrain a stronger base model using the approved overhaul recipe, then convert with the (now-fixed) script and re-evaluate.

## Goal

Produce a stronger HebrewModernBERT-base by retraining the 4-phase pipeline **from scratch** with the overhaul recipe improvements, then export to HF and confirm the gain on BeIR Hebrew retrieval.

Success = (a) MLM smoke test predicts sensible Hebrew on trivial cloze (east/west, school) in top-8; (b) BeIR avg NDCG@10 after the same dual-encoder SFT clearly exceeds 0.147 (target: beat the old 0.161, ideally approach NeoDictaBERT range).

## Pre-flight (already done — verify, don't redo)

- Phase chain `load_path` is correctly wired (verified): 0.1 = from scratch; 0.2 ← 0.1 latest; 1 ← 0.2 latest; 2 ← 1 latest. Files: `yamls/main/base_hebrew/flex-bert-rope-phase-{0.1,0.2,1,2}*.yaml`.
- `src/convert_to_hf.py` rope/mask bugs fixed.
- Tokenizer is the 150K fast tokenizer (`tokenizer.json`); always load with `use_fast=True`. The `spm.model` in HF dirs is the OLD 100K tokenizer — do not use it.

## Recipe decisions to make BEFORE launching (see overhaul plan §"Open decisions")

Confirm with the user, or apply these defaults across **all four** `base_hebrew` YAMLs identically:
- **Masking schedule**: default = keep flat 30%. (Optional A/B: 40%→15% decay — needs new `src/algorithms/masking_schedule.py`, plan Step 4.)
- **Normalization**: default = keep `layernorm`. (Optional: `rmsnorm` — config flip, plan Step 3; re-verify `convert_to_hf.py` exports a norm field HF understands.)
- **Phase-1 warmup**: plan recommends a small re-warmup (`t_warmup ≈ 0.01dur`) — currently `0tok`. Decide whether to apply.
- **hidden_act**: default = keep `gelu` (GeGLU). (Optional: `silu`/SwiGLU.)

If the user wants the *minimal* retrain (just a clean from-scratch run with the already-fixed wiring), skip all optional knobs and run the pipeline as-is.

## Steps to run

All training is **slurm-only** (local HW cannot train). Env `bert24` for training; `bert-b200` for B200 phases and for HF conversion/inference. Entry point is always `python -m composer main.py <yaml>`.

### 0. Smoke test (~5 min, do this first)
```bash
sbatch .slurm/jobs/smoke_base_hebrew_phase-0.1.slurm
```
Confirm in the log: loss decreasing + MaskedAccuracy increasing. If it fails, STOP and fix before the long runs.

### 1. Phase 0.1 — main MLM pretrain (1024 ctx, ~100B tokens, from scratch)
```bash
sbatch .slurm/jobs/train_base_hebrew_phase-0.1.slurm   # partition H200-12h, 2 GPU, 12h wall, autoresumes/requeues
```
This is the longest phase (many requeues). W&B uses a stable run id with `resume: allow` across requeues. Wait for `checkpoints/base/modern-bert-base-phase-0.1-pretrain/ckpt/latest-rank0.pt` to reach the target token budget before proceeding.

### 2. Phase 0.2 — continued pretrain (1024 ctx, ~100B Hebrew)
```bash
sbatch .slurm/jobs/train_base_hebrew_phase-0.2.slurm   # auto-loads phase-0.1 latest via load_path
```

### 3. Phase 1 — context extension to 8192 (~30B tokens)
```bash
sbatch .slurm/jobs/train_b200_phase-1.slurm            # partition p_b200_nlp, account ug_goldberg, 4 GPU, 4h wall (requeues)
```

### 4. Phase 2 — anneal at 8192 (~50B tokens)
```bash
sbatch .slurm/jobs/train_b200_phase-2.slurm            # p_b200_nlp, 4 GPU, 4h wall (requeues)
```
Final checkpoint: `checkpoints/base/modern-bert-base-phase-2-contextextension/ckpt/latest-rank0.pt`.

### 5. Convert to HF (fixed script, run in bert-b200)
```bash
conda activate bert-b200
python ./src/convert_to_hf.py \
  --yaml-config yamls/main/base_hebrew/flex-bert-rope-phase-2-contextextension.yaml \
  --output-name HebrewModernBERT-base-final-v2 --output-dir ./outputs/hf \
  --input-checkpoint checkpoints/base/modern-bert-base-phase-2-contextextension/ckpt/latest-rank0.pt \
  --bos-token-id 2 --eos-token-id 3 --cls-token-id 2 --sep-token-id 3 \
  --pad-token-id 0 --mask-token-id 4 --max-length 8192 --vocab-size 150016
```
Verify `config.json`: `local_rope_theta=10000`, `global_rope_theta=160000`, `mask_token_id=4`, `vocab_size=150016`.

### 6. MLM smoke test (the fast quality gate, CPU ok, bert-b200)
Load `outputs/hf/HebrewModernBERT-base-final-v2` with `AutoTokenizer(..., use_fast=True)` + `AutoModelForMaskedLM`, run fill-mask on:
`בירת ישראל היא [MASK].` / `השמש זורחת ב[MASK] ושוקעת במערב.` / `הילד הלך לבית ה[MASK] ללמוד.`
A healthy model should put `ירושלים`/`מזרח`/`ספר` in the top few. (Reference script from the audit: `/tmp/rope_mask_experiment.py`.)

### 7. Retrieval eval (sibling repo, B200)
In `/home/nlp/achimoa/workspace/hebrew_text_retrieval`, reuse the dual-encoder SFT + BeIR eval scripts (point MODEL at the new HF dir):
- Train: `scripts/model/dual_encoder/heq/train/train_dual_encoder_beir_hn_hmb_final.sh` (set MODEL to `...-v2`, new OUTPUT_DIR)
- Eval: `scripts/model/eval/eval_beir_hmb_final_dualenc.sh`
- Submit across both partitions to parallelize: `sbatch --partition=p_b200_goldberg,p_b200_nlp --account=ug_goldberg ...`, eval with `--dependency=afterok:<trainjob>`.
- Compare avg NDCG@10 vs 0.147 (this run) and 0.161 (old model).

## Constraints / gotchas (do not violate)
- **MAFAT data is PRIVATE** — never publish/leak the Hebrew training corpus. Model+code+recipe public, data private.
- Slurm only; use `p_` partitions for B200 (`p_b200_nlp` 4-GPU / `p_b200_goldberg` 1-GPU, account `ug_goldberg`); phases 0.1/0.2 use `H200-12h` (2-GPU) as wired — confirm availability.
- B200 needs env `bert-b200` (torch2.7/cu128, flash-attn 2.7.4 sm_100); fresh `TRITON_CACHE_DIR` gotcha (see memory `b200-blackwell-env`).
- Jobs auto-requeue at wall-clock; W&B resume is configured — don't fragment runs by changing `run_name`.
- Tokenizer: `use_fast=True` always; `bert25` env can't load `tokenizer.json`.
```
