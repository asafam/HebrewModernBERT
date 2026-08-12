# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

This is a fork of Answer.AI's [ModernBERT](https://github.com/AnswerDotAI/ModernBERT) research training repository, adapted to pre-train **HebrewModernBERT** — a Hebrew-language ModernBERT encoder. The upstream code (FlexBERT modular encoder, MosaicML Composer training harness, GLUE/retrieval evals) is mostly unchanged; the Hebrew-specific work lives in a custom tokenizer, the `yamls/main/base_hebrew/` configs, the `scripts/` wrappers, and `src/convert_to_hf.py`.

When reasoning about training internals, prefer the upstream `README.md` and `RunEvals.md` — they document the FlexBERT/Composer machinery. This file covers what is local and non-obvious.

## Environments

Two conda environments are used, and the scripts switch between them:
- **`bert24`** — training and evals (created from `environment.yaml`; `conda env create -f environment.yaml`). Requires a GPU + `flash_attn==2.6.3` (see `README.md` Setup).
- **`bert25`** — only for `src/convert_to_hf.py` (HF checkpoint export). Uses a newer `transformers` that knows the `modernbert` model type.

`.env` sets `PYTORCH_CUDA_ALLOC_CONF`; it is loaded by training scripts.

## Common commands

Training is driven by MosaicML Composer over a single YAML config:
```bash
python -m composer main.py yamls/main/base_hebrew/flex-bert-rope-phase-0.1-pretrain.yaml
```
The `scripts/train_modernbert_*_phase-*.sh` wrappers just `conda activate bert24`, `cd` here, and run the above with the matching YAML. Use them as the canonical entry points.

Convert a trained Composer checkpoint to HuggingFace format (run in `bert25`):
```bash
bash scripts/convert_to_hf_hebmodernbert.sh   # wraps src/convert_to_hf.py with the right token IDs / vocab size
```

Lint: `ruff check .` (config in `ruff.toml`: line-length 120, py311).

Tests (pytest, in `tests/`): `pytest tests/test_main.py`, single test e.g. `pytest tests/test_rotary.py::<name>`. Tests run against the small `tests/smoketest_config_*.yaml` configs, not full training.

Evals: GLUE for a ModernBERT checkpoint via `run_evals.py` (give it a checkpoint + training config); non-ModernBERT models via `glue.py`. See `RunEvals.md`.

## Multi-phase pre-training

Training is staged; each phase is a separate YAML run, fed the previous phase's weights via `load_path` + `load_weights_only: true` (fresh optimizer/schedule/clock):
- **phase-0-pretrain** — unified Hebrew-only MLM pre-training at `max_seq_len: 1024`, **20% masking**, constant LR (no anneal), ~130B tokens on `hebrew_quality`.
- **phase-1-contextextension** — extend to `max_seq_len: 8192` with `sequence_packing: true`, ~40B tokens.
- **phase-2-contextextension** — curated-heavy anneal at 8192 (`one_minus_sqrt` LR→0), ~15B tokens on `hebrew_quality_anneal`.

The older **phase-0.1 (mixed En/code) + phase-0.2 (Hebrew specialize)** split is superseded — those YAMLs are kept only as history.

Active config trees: **`yamls/main/base_hebrew/`** (the trained base) and **`yamls/main/base_hebrew_large/`** (the large model; warm-started from base by weight tiling — see `docs/TRAINING.md` §8). **`yamls/main/base/` and `yamls/main/large/` are deprecated** — `base/` still carries the old 100K tokenizer and `large/` predates the overhaul. Do not use either as a template. When editing a config, check whether the same change is needed in the other phases of the same tree; `scripts/preflight_check_configs.py` catches the common mistakes without needing a GPU.

Key Hebrew-specific knobs in these YAMLs (they must stay consistent with the tokenizer and `convert_to_hf` args):
- `tokenizer_name: tokenizer/v4_bpe_150k` — a local path, **not** a HF hub name.
- `vocab_size: 150016`, `pad_token_id: 0`, `bos/cls_token_id: 2`, `eos/sep_token_id: 3`, `mask_token_id: 4`.
- `save_folder` / `load_path` are templated off `${checkpoint_dir}/${run_group}/${run_name}`; `run_group` is `base` or `large`.

## Tokenizer

The current tokenizer is **`tokenizer/v4_bpe_150k`** — a 150,016-vocab BPE fast tokenizer (`tokenizer.json`), trained in a Unigram-vs-BPE bake-off (`scripts/train_tokenizer_bakeoff.sh`) on a Hebrew-heavy corpus. 1.302 tokens/word with 0% UNK, beating DictaBERT's ~1.31. Special IDs: `[PAD]=0 [UNK]=1 [CLS]=2 [SEP]=3 [MASK]=4`; 150016 = 1172x128, tensor-core friendly.

**Always load it with `use_fast=True`** — the `bert25` env cannot read `tokenizer.json`; use `bert-b200`.

Superseded and NOT to be used: the older ~100K SentencePiece model at `tokenizer/` root (`spiece.model`, `HebrewModernBERT_mixed_1M_100K.vocab`) and `tokenizer/tokenizer_v{1,2}`, plus any `spm.model`/`spiece.model` found inside older HF export dirs — those are the old 100K vocab. The `tokenizer-save-dir-*/` directories are scratch output.

Token IDs are hardcoded across the training YAMLs and `scripts/convert_to_hf_hebmodernbert.sh` — if you change the tokenizer, update all three: special-token IDs, `vocab_size`, and the model embedding size.

## Data

Datasets are MosaicML MDS-format streaming datasets under `data/hebrewmodernbert/<corpus>/` (split into `train`/`validation`). The pipeline to build them lives in `src/data/` (`hf_to_mds.py` to convert a HF dataset to MDS, `sample_dataset_from_config.py` to sample/mix sources); see `src/data/README.md`. Two dataset classes (`StreamingTextDataset` vs `NoStreamingDataset`) are selected per-loader via `streaming: true|false` in the YAML — local training uses `streaming: false` for throughput (see `README.md` Data section).

## Architecture orientation

- `main.py` — the training entry point. Reads the YAML, builds the FlexBERT model, Composer `Trainer`, dataloaders, optimizer/scheduler, callbacks, and W&B logging.
- `src/flex_bert.py` + `src/bert_layers/` — the FlexBERT model. `bert_layers/` holds the modular building blocks (attention/rotary, mlp/glu, normalization, embeddings, loss) that the YAML's `model_config` selects by name (e.g. `attention_layer: rope`, `mlp_layer: glu`, `bert_layer: prenorm`). Architecture is configured, not coded — to change the model, edit `model_config`, not Python.
- `src/text_data.py` / `src/text_data_tokenize.py` — dataset/dataloader and on-the-fly MLM masking + sequence packing.
- `src/convert_to_hf.py` — Typer CLI that maps a Composer FlexBERT checkpoint + its training YAML into a HF `ModernBertForMaskedLM` (`config.json` + safetensors). It translates FlexBERT config field names to HF ModernBERT field names; token IDs and `vocab_size` are passed explicitly on the command line, not inferred.
- `src/scheduler.py`, `src/optimizer.py`, `src/algorithms/rope_schedule.py`, `src/callbacks/` — Composer-side schedulers, optimizers, and callbacks referenced by name in the YAML.
