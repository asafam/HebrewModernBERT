"""Verify that base -> large Phi-style weight tiling actually took.

The failure mode this guards against is silent: with `tie_word_embeddings=True` (the HF default,
unset in our YAMLs) the MLM decoder weight is re-tied to `tok_embeddings`, so if
`tile_embedding(...tok_embeddings...)` in `init_model_from_pretrained` is not called, the entire
150016 x hidden token-embedding matrix -- 31% of HebrewModernBERT-large -- stays at random init
while training starts happily and loss looks plausible.

Discriminator: a TRAINED embedding matrix has a much larger std than a fresh `init_std: 0.02` one.
Measured on the base phase-0 endpoint (checkpoints/base/phase-0-arm1-m20): std = 0.113.

Usage (from repo root, in the bert-b200 env):
    python scripts/verify_tile_init.py yamls/main/base_hebrew_large/flex-bert-rope-phase-0-pretrain.yaml
Exits non-zero if the warm start did not take.
"""

import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf as om

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import build_model, init_from_checkpoint  # noqa: E402

# A freshly initialized matrix sits at ~init_std (0.02); the trained base embedding is 0.113.
# Anything below this threshold means the tiling did not reach the token embeddings.
RANDOM_INIT_CEILING = 0.05


def main(yaml_path: str) -> int:
    cfg = om.merge(om.load("yamls/defaults.yaml"), om.load(yaml_path))
    cfg = om.create(om.to_container(cfg, resolve=True))

    if cfg.get("init_from_checkpoint", None) is None:
        print(f"FAIL: {yaml_path} has no `init_from_checkpoint` block -- nothing to verify.")
        return 1

    print("Building the large model (random init)...")
    model = build_model(cfg.model)
    emb = model.model.bert.embeddings.tok_embeddings.weight
    # NOTE: tile_embedding writes IN PLACE under torch.no_grad() and does not rebind the
    # Parameter, so `emb` is the same tensor object before and after. Snapshot the value now or
    # the "before" reading silently reports the post-tiling number.
    before = float(emb.detach().clone().float().std())
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params={n_params:.4e}  tok_embeddings={tuple(emb.shape)}  std(before)={before:.4f}")

    print("\nTiling weights from the base checkpoint...")
    init_from_checkpoint(cfg.init_from_checkpoint, model)

    after = float(emb.detach().float().std())
    dec = model.model.decoder.weight
    tied = dec.data_ptr() == emb.data_ptr()

    print("\n--- results ---")
    print(f"  tok_embeddings std : {before:.4f} (before) -> {after:.4f} (after)")
    print(f"  decoder tied to tok_embeddings: {tied}")
    print(f"  reference: trained base = 0.113, fresh init_std=0.02 init ~= 0.02")

    ok = after > RANDOM_INIT_CEILING and after != before
    if after == before:
        print("\nFAIL: std is unchanged -- tiling did not touch the token embeddings.")
        return 1
    if not ok:
        print(
            f"\nFAIL: tok_embeddings std {after:.4f} <= {RANDOM_INIT_CEILING} -- the token embeddings\n"
            "are still at random init. Check that the `tile_embedding(...tok_embeddings...)` call in\n"
            "src/bert_layers/model.py `init_model_from_pretrained` is NOT commented out."
        )
        return 1

    # Spot-check that encoder weights moved too (Gopher layer mapping round(i * 22/28)).
    w = model.model.bert.encoder.layers[0].attn.Wqkv.weight
    print(f"  layer0 Wqkv std    : {float(w.float().std()):.4f}  shape={tuple(w.shape)}")
    print("\nPASS: warm start took -- token embeddings carry trained statistics.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    torch.manual_seed(17)
    sys.exit(main(sys.argv[1]))
