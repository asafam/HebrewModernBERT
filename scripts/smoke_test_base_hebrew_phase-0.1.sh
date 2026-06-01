#!/bin/bash -i
# Smoke test + THROUGHPUT PROBE for the phase-0.1 (mixed-corpus) setup, before the full run.
#
# Validates end-to-end that the model builds with the rebuilt tokenizer + curriculum
# (vocab 100000->100032, pad=0), trains, evals, and checkpoints. AND measures the REAL
# tokens/sec on the H200s at realistic settings — `device_train_microbatch_size=auto`
# lets Composer pick the largest microbatch that fits (a 150M model badly under-uses an
# H200 at mb=128, so this is also our main throughput lever). torch.compile stays on so
# the rate is representative. ~60 steps; a few minutes after compile/warmup.
#
# After it runs, read the throughput from the log (or the offline W&B run):
#   grep -iE "tokens_per_sec|throughput|MicrobatchSize|device_train_microbatch" .slurm/logs/hmb-smoke_*.out
# Use that measured tokens/sec (x the #GPUs) to size the real run.
#
# Run via slurm: sbatch .slurm/jobs/smoke_base_hebrew_phase-0.1.slurm

echo "Activating Conda environment: bert24"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bert24
cd /home/nlp/achimoa/workspace/HebrewModernBERT

export WANDB_MODE=offline   # keep the probe out of the W&B project / no login needed

python -m composer main.py yamls/main/base_hebrew/flex-bert-rope-phase-0.1-pretrain.yaml \
    run_name=smoke-test-phase-0.1 \
    max_duration=60ba \
    device_train_microbatch_size=auto \
    eval_interval=1000ba \
    save_interval=1000ba \
    save_num_checkpoints_to_keep=1 \
    autoresume=false \
    log_to_console=true \
    console_log_interval=10ba
echo "Probe done. Check the log for tokens_per_sec and the auto-chosen microbatch size."
