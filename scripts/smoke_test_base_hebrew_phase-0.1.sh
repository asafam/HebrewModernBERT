#!/bin/bash -i
# Quick end-to-end smoke test of the overhauled phase-0.1 setup BEFORE a full run:
# validates that the model builds with the rebuilt tokenizer (vocab 100000, pad=0,
# embeddings padded to 100032), trains a few steps with decreasing loss, runs the MLM
# eval, and writes a checkpoint to the right path. ~200 tiny batches; minutes, not days.
#
# Reuses the real phase-0.1 YAML with CLI overrides (main.py merges om.from_cli).
# Run via slurm: sbatch .slurm/jobs/smoke_base_hebrew_phase-0.1.slurm

echo "Activating Conda environment: bert24"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bert24
cd /home/nlp/achimoa/workspace/HebrewModernBERT

# keep W&B local-only for the smoke (no project clutter / no login needed)
export WANDB_MODE=offline

python -m composer main.py yamls/main/base_hebrew/flex-bert-rope-phase-0.1-pretrain.yaml \
    run_name=smoke-test-phase-0.1 \
    max_duration=200ba \
    global_train_batch_size=64 \
    device_train_microbatch_size=64 \
    eval_interval=100ba \
    eval_subset_num_batches=10 \
    save_interval=150ba \
    save_num_checkpoints_to_keep=1 \
    model.model_config.compile_model=false
echo "Smoke test finished. Checkpoint -> checkpoints/hebrew/smoke-test-phase-0.1/ckpt"
