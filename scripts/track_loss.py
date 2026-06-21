#!/usr/bin/env python
"""Pull eval loss/accuracy from W&B for run-health monitoring.

Context-ext runs set disable_train_metrics=true, so the tqdm train-loss is a frozen
artifact; the real health signal is the EVAL loss, which lives in W&B cloud (the local
slurm log truncates the tqdm postfix). This queries the W&B API.

Usage:  python scripts/track_loss.py [project] [run_id]
  default project: asafam/HebModernBERT-phase1  (use .../HebModernBERT-phase2 for phase 2)
  run_id optional; if omitted, picks the most recent running/finished run.
"""
import sys
import wandb

project = sys.argv[1] if len(sys.argv) > 1 else "asafam/HebModernBERT-phase1"
run_id = sys.argv[2] if len(sys.argv) > 2 else None
api = wandb.Api()

KEYS = ["time/token", "metrics/eval/LanguageCrossEntropy", "metrics/eval/MaskedAccuracy"]


def show(r):
    h = r.history(keys=KEYS, pandas=False)
    evals = [(row.get("time/token"), row.get("metrics/eval/LanguageCrossEntropy"), row.get("metrics/eval/MaskedAccuracy"))
             for row in h if row.get("metrics/eval/LanguageCrossEntropy") is not None]
    print(f"run {r.id} ({r.state}) — {len(evals)} evals")
    for tok, lce, acc in evals:
        print(f"  {(tok or 0)/1e9:7.2f}B tok | eval LCE {lce:.4f} | MaskedAcc {acc:.4f}")
    if len(evals) >= 2:
        d = evals[-1][1] - evals[0][1]
        print(f"  trend: LCE {evals[0][1]:.4f} -> {evals[-1][1]:.4f} ({'DOWN, healthy' if d < 0 else 'UP/flat — check'})")


if run_id:
    show(api.run(f"{project}/{run_id}"))
else:
    for r in api.runs(project, order="-created_at"):
        if r.state in ("running", "finished"):
            show(r)
            break
