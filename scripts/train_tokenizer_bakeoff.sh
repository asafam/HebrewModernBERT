#!/bin/bash -i
# Tokenizer bake-off: train candidate SentencePiece models to beat DictaBERT on Hebrew.
# 150,016 vocab (= 1172x128, tensor-core friendly), byte_fallback on (UNK -> 0).
# A: Unigram (usually best fertility for morphologically rich langs)  B: BPE (control).
# CPU-only + RAM-heavy -> run on cpu1T-24h. Benchmark afterwards with eval_tokenizer.py.

echo "Activating Conda environment: bert24"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bert24
cd /home/nlp/achimoa/workspace/HebrewModernBERT

CORPUS=data/tokenizer_corpus_heb_3M.txt
[ -f "$CORPUS" ] || { echo "Corpus $CORPUS missing — build it first (build_tokenizer_corpus.py)"; exit 1; }

echo "=== Candidate A: Unigram 150016 + byte_fallback ==="
python -m src.data_prep.tokenizer.train_tokenizer \
    --corpus "$CORPUS" --vocab_size 150016 --model_type unigram \
    --output_prefix tokenizer/cand_unigram/spiece

echo "=== Candidate B: BPE 150016 + byte_fallback ==="
python -m src.data_prep.tokenizer.train_tokenizer \
    --corpus "$CORPUS" --vocab_size 150016 --model_type bpe \
    --output_prefix tokenizer/cand_bpe/spiece

echo "Done. Benchmark (run on a node with internet for the Dicta baseline):"
echo "  python -m src.data_prep.tokenizer.eval_tokenizer --candidates tokenizer/cand_unigram/spiece.model tokenizer/cand_bpe/spiece.model"
