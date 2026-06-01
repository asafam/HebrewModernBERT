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

Training is staged; each phase is a separate YAML run, fed the previous phase's checkpoint via `load_path` / `pretrained_checkpoint`:
- **phase-0.1-pretrain** — main MLM pre-training at `max_seq_len: 1024`, 30% masking.
- **phase-0.2-pretrain** — continued pre-training.
- **phase-1-contextextension** / **phase-2-contextextension** — extend context length (raise `max_seq_len` and `rotary_emb_base`).

Parallel config trees exist for three model sizes: `yamls/main/base/`, `yamls/main/base_hebrew/`, `yamls/main/large/`. The `base_hebrew` tree is the active Hebrew run; `base`/`large` are the upstream/English-style references. When editing a Hebrew config, check whether the same change is needed in the other phases of the same tree, since they share architecture and only differ in seq-len/checkpoint wiring.

Key Hebrew-specific knobs in these YAMLs (they must stay consistent with the tokenizer and `convert_to_hf` args):
- `tokenizer_name: tokenizer` — points at the local `tokenizer/` dir (SentencePiece, see below), **not** a HF hub name.
- `vocab_size: ~100002`, `pad_token_id: 100001`, `bos/cls_token_id: 2`, `eos/sep_token_id: 3`.
- `save_folder` / `load_path` are templated off `${checkpoint_dir}/${data_corpus}/${run_name}`; `data_corpus: hebrew`.

## Tokenizer

The Hebrew tokenizer is a custom SentencePiece model checked into `tokenizer/` (`spiece.model`, `HebrewModernBERT_mixed_1M_100K.vocab`, ~100K vocab) with versioned variants in `tokenizer/tokenizer_v1` and `tokenizer/tokenizer_v2`. Configs reference it by the local path `tokenizer`. Token IDs are hardcoded across the training YAMLs and `scripts/convert_to_hf_hebmodernbert.sh` — if you change the tokenizer, update all three: special-token IDs, `vocab_size`, and the model embedding size. The `tokenizer-save-dir-*/` directories in the repo root are scratch output and not the source of truth.

## Data

Datasets are MosaicML MDS-format streaming datasets under `data/hebrewmodernbert/<corpus>/` (split into `train`/`validation`). The pipeline to build them lives in `src/data/` (`hf_to_mds.py` to convert a HF dataset to MDS, `sample_dataset_from_config.py` to sample/mix sources); see `src/data/README.md`. Two dataset classes (`StreamingTextDataset` vs `NoStreamingDataset`) are selected per-loader via `streaming: true|false` in the YAML — local training uses `streaming: false` for throughput (see `README.md` Data section).

## Architecture orientation

- `main.py` — the training entry point. Reads the YAML, builds the FlexBERT model, Composer `Trainer`, dataloaders, optimizer/scheduler, callbacks, and W&B logging.
- `src/flex_bert.py` + `src/bert_layers/` — the FlexBERT model. `bert_layers/` holds the modular building blocks (attention/rotary, mlp/glu, normalization, embeddings, loss) that the YAML's `model_config` selects by name (e.g. `attention_layer: rope`, `mlp_layer: glu`, `bert_layer: prenorm`). Architecture is configured, not coded — to change the model, edit `model_config`, not Python.
- `src/text_data.py` / `src/text_data_tokenize.py` — dataset/dataloader and on-the-fly MLM masking + sequence packing.
- `src/convert_to_hf.py` — Typer CLI that maps a Composer FlexBERT checkpoint + its training YAML into a HF `ModernBertForMaskedLM` (`config.json` + safetensors). It translates FlexBERT config field names to HF ModernBERT field names; token IDs and `vocab_size` are passed explicitly on the command line, not inferred.
- `src/scheduler.py`, `src/optimizer.py`, `src/algorithms/rope_schedule.py`, `src/callbacks/` — Composer-side schedulers, optimizers, and callbacks referenced by name in the YAML.
