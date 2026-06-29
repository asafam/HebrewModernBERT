#!/usr/bin/env python3
"""
Create a randomly initialized HMB model with the same architecture as an existing HF checkpoint.
Used to establish a random-weights baseline for retrieval evaluation.
"""
import argparse
import torch
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="outputs/hf/HebrewModernBERT-base-final",
                        help="Path to HF model to copy architecture from")
    parser.add_argument("--output", default="outputs/hf/HebrewModernBERT-base-random",
                        help="Where to save the randomly initialized model")
    args = parser.parse_args()

    print(f"Loading config from {args.source}...")
    config = AutoConfig.from_pretrained(args.source, trust_remote_code=True)

    print(f"Creating randomly initialized model (same architecture, weights NOT loaded)...")
    model = AutoModelForMaskedLM.from_config(config)

    n_params = sum(p.numel() for p in model.parameters())
    emb_std = model.model.embeddings.tok_embeddings.weight.std().item()
    print(f"Parameters: {n_params:,}")
    print(f"Embedding weight std (should be ~0.02 for random init): {emb_std:.4f}")

    print(f"Saving to {args.output}...")
    model.save_pretrained(args.output)

    tok = AutoTokenizer.from_pretrained(args.source, trust_remote_code=True)
    tok.save_pretrained(args.output)

    print(f"Done. Random baseline model saved to {args.output}")
    print(f"Use this for SFT + eval to compare against the trained model.")


if __name__ == "__main__":
    main()
