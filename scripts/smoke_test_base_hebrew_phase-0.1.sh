#!/bin/bash -i
# Packing probe for phase-0.1: measure REAL-token throughput with sequence packing on,
# and confirm packing trains correctly (loss down + MaskedAccuracy up) before the long run.
#
# Why: without packing the loader pads each ~392-token doc to 1024 (~60% padding) and the
# budget counts it -> the model would see far fewer real tokens than intended. Packing fills
# each 1024 sequence with real tokens from multiple docs (ModernBERT's approach). The data is
# uncompressed so NoStreamingDataset + packing just turn on.
#
# microbatch=auto: each packed sequence is now ~1024 real tokens (~2.6x denser), so the old
# 288 will likely OOM — let Composer find the fit. compile OFF so auto doesn't recompile per
# trial (that was the earlier "hang") and starts in seconds.
#
# Judge it on: (a) loss decreases + MaskedAccuracy climbs (packing masking is correct),
#              (b) throughput/tokens_per_sec (now ~all real) vs the ~174K real-tok/s baseline.
# Run via slurm: sbatch .slurm/jobs/smoke_base_hebrew_phase-0.1.slurm

echo "Activating Conda environment: bert24"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bert24
cd /home/nlp/achimoa/workspace/HebrewModernBERT

export WANDB_MODE=offline

# Packing needs a CONCRETE microbatch (the packer computes packed shapes from it, so
# 'auto' can't work), and packed sequences are ~2.6x denser than unpacked, so start
# conservative at 96 (unpacked auto-fit was 288). batch_size_warmup_min_size=null disables
# the batch-size warmup (which otherwise crashes on the string microbatch AND would run the
# probe at a tiny non-representative batch).
python -m composer main.py yamls/main/base_hebrew/flex-bert-rope-phase-0.1-pretrain.yaml \
    run_name=smoke-packing-phase-0.1 \
    max_duration=60ba \
    train_loader.dataset.streaming=false \
    train_loader.sequence_packing=true \
    train_loader.batch_size_warmup_min_size=null \
    train_loader.packing_buffer_size=2000 \
    train_loader.num_workers=4 \
    device_train_microbatch_size=96 \
    model.model_config.compile_model=false \
    eval_interval=1000ba \
    save_interval=1000ba \
    save_num_checkpoints_to_keep=1 \
    autoresume=false \
    log_to_console=true \
    console_log_interval=10ba
echo "Packing probe done. Check: loss decreasing + MaskedAccuracy rising, and throughput/tokens_per_sec (now ~all real) vs ~174K baseline."
