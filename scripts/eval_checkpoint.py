"""Eval-only: load a checkpoint and evaluate on the packed validation set with a FIXED
config. Used to confirm whether eval cross-entropy genuinely rises across phase-1
checkpoints, or whether the W&B trend was an artifact of changing eval_subset_num_batches
mid-run (which made the logged points non-comparable).

Usage:
  python scripts/eval_checkpoint.py <yaml> load_path=<ckpt> load_weights_only=true \
      eval_subset_num_batches=100 seed=17 ...overrides
Reuses main()'s build path with do_train=False, then runs trainer.eval().
"""
import sys
from typing import cast

from omegaconf import OmegaConf as om, DictConfig

from main import main

yaml_path, args_list = sys.argv[1], sys.argv[2:]
with open("yamls/defaults.yaml") as f:
    default_cfg = om.load(f)
with open(yaml_path) as f:
    yaml_cfg = om.load(f)
cli_cfg = om.from_cli(args_list)
cfg = cast(DictConfig, om.merge(default_cfg, yaml_cfg, cli_cfg))

trainer = main(cfg, return_trainer=True, do_train=False)
trainer.eval()

print("==== EVAL RESULT ====")
print(f"ckpt: {cfg.get('load_path')}")
print(f"eval_subset_num_batches: {cfg.get('eval_subset_num_batches')} | seed: {cfg.get('seed')}")
for ev_name, metrics in trainer.state.eval_metrics.items():
    for m_name, metric in metrics.items():
        try:
            print(f"  {ev_name}/{m_name}: {metric.compute().item():.4f}")
        except Exception as e:
            print(f"  {ev_name}/{m_name}: <{e}>")
