# `src/data_prep` — tokenizer & pretraining-data pipeline

Self-contained, reproducible pipeline to build the HebrewModernBERT **tokenizer** and
**pretraining data**. Consolidated here from the sibling `hebrew_text_retrieval` repo so
the model is reproducible from this repo alone.

## ⚠️ Data privacy (MAFAT)

The Hebrew corpus (**MAFAT**, `data/mafat/hebrew/sources`) is **private and must never be
published or leaked**. The model (weights + code + this recipe) is public; the data is
not. Nothing under `data/` is committed (it is gitignored). These configs only *name* the
MAFAT sources — they contain none of its content. The English/code half is public
(Dolma), so that part is fully reproducible; the Hebrew half is reproducible only with
authorized MAFAT access.

## Layout

```
src/data_prep/
  datasets/
    build_datasets.py        # assemble sources -> MDS/JSONL/TXT (single-pass train+val)
    prepare_dolma_dataset.py # sample Dolma to a token budget (batched tokenization)
    dataset_formats/         # jsonl + hf source readers
    utils.py                 # MDS/JSONL/TXT writers, hashing
  tokenizer/
    train_tokenizer.py       # SentencePiece BPE 100K (run as a slurm job)
    build_hf_tokenizer.py    # spm.model -> clean HF *fast* tokenizer (runs locally)
  config/train/              # mix + tokenizer-corpus configs
```

## Recipe (end to end)

Heavy steps (Dolma sampling, tokenizer training, data-mix build over the full corpus,
and of course pretraining) are **slurm jobs** — the login node can't run them.

1. **Sample Dolma** to token budgets (English excludes `starcoder`; code is `starcoder`):
   ```
   python -m src.data_prep.datasets.prepare_dolma_dataset \
     --output_file data/dolma/corpus_sampled_eng_75B.jsonl --token_budget 75_000_000_000 \
     --data_path data/dolma --tokenizer_path tokenizer/v3_clean --exclude_source starcoder
   ```
2. **Tokenizer**: build the 1M-doc mixed corpus, train spm, convert to HF fast tokenizer:
   ```
   python -m src.data_prep.datasets.build_datasets \
     --config_file src/data_prep/config/train/datasets_tokenizer_corpus.yaml \
     --output_path data/tokenizer_corpus --format txt
   python -m src.data_prep.tokenizer.train_tokenizer \
     --corpus data/tokenizer_corpus/train.txt --vocab_size 100000 --output_prefix tokenizer/spiece
   python src/data_prep/tokenizer/build_hf_tokenizer.py \
     --input-spm tokenizer/spiece.model --output-dir tokenizer/v3_clean
   ```
   The HF builder renames the spm natives `<pad>`/`<unk>` -> `[PAD]`/`[UNK]` and yields a
   clean fast tokenizer (`[PAD]=0 [UNK]=1 [CLS]=2 [SEP]=3 [MASK]=4`, vocab 100000;
   niqqud unsupported by design — the corpus is unvocalized).
3. **Build the pretraining mix** (MDS) used by training:
   ```
   python -m src.data_prep.datasets.build_datasets \
     --config_file src/data_prep/config/train/datasets_h50_e75_c25.yaml \
     --output_path data/hebrewmodernbert/mixed/mixed_h50e75c25 --format mds
   ```
4. **Pretrain** (`yamls/main/base_hebrew/...`) then **export** to HF
   (`scripts/convert_to_hf_hebmodernbert.sh`).

## Efficiency notes (vs. the original)

- `prepare_dolma_dataset.py` tokenizes in **batches** (was: one doc at a time,
  single-threaded, over tens of B tokens just to count) — the dominant speedup.
- `build_datasets.py` writes train+validation in a **single pass** (was: one full read
  per split) and computes the dedup hash only when `--remove_duplicates` is set.
- `train_tokenizer.py` uses all CPU cores (was: 4 threads).
