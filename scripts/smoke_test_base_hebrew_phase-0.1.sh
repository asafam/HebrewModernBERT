#!/bin/bash -i
# Optional quick validation of the phase-0.1 setup (the real config, no packing).
# We already validated on 2xH200 (job 16472430): model builds with the rebuilt tokenizer +
# mixed curriculum, loss decreases, ~174K REAL tokens/sec. With count_padding_tokens=false
# the budget now counts real tokens, so this just confirms the run is healthy before the
# long job: loss down + MaskedAccuracy up, and time/token advancing at the ~real rate.
#
# NOTE: sequence packing is NOT used. The model already unpads (padding: unpadded), so the
# heavy compute runs on real tokens regardless; packing's upside here is small and the
# runtime packer was a bottleneck. Offline-concatenated MDS is parked as a v2 throughput idea.
#
# Run via slurm: sbatch .slurm/jobs/smoke_base_hebrew_phase-0.1.slurm

echo "Activating Conda environment: bert24"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bert24
cd /home/nlp/achimoa/workspace/HebrewModernBERT

export WANDB_MODE=offline

python -m composer main.py yamls/main/base_hebrew/flex-bert-rope-phase-0.1-pretrain.yaml \
    run_name=smoke-test-phase-0.1 \
    max_duration=40ba \
    eval_interval=1000ba \
    save_interval=1000ba \
    save_num_checkpoints_to_keep=1 \
    autoresume=false \
    model.model_config.compile_model=false \
    log_to_console=true \
    console_log_interval=10ba
echo "Done. Confirm: loss down + MaskedAccuracy up, and time/token advancing (real-token budget)."
