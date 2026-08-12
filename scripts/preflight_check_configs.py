"""GPU-free preflight for the HebrewModernBERT training YAMLs.

`FlexBertConfig` cannot be imported on a login node (flash-attn's triton layer_norm needs an
active CUDA driver at import time), so its constructor validators only fire once a job is already
on a GPU -- after a queue wait. This replicates those checks in pure Python against the YAML, plus
a few cross-file invariants, so a typo costs seconds instead of a scheduling round-trip.

Mirrors src/bert_layers/configuration_bert.py:220-260 and src/flex_bert.py:256-264.

Usage:  python scripts/preflight_check_configs.py yamls/main/base_hebrew_large/*.yaml
"""

import sys

import yaml

# Registry values, from src/bert_layers/options.py and the *2CLS dicts it re-exports.
NORMS = {"layernorm", "triton_layernorm", "rmsnorm", "triton_rmsnorm"}
MLPS = {"mlp", "glu", "parallel_glu"}
PADDINGS = {"unpadded", "padded"}
EMBEDDINGS = {"absolute_pos", "sans_pos"}
ATTN_STEMS = {"base", "parallel", "rope", "rope_parallel"}
LAYER_STEMS = {"prenorm", "compile_prenorm", "parallel_prenorm", "postnorm"}
# Every key FlexBertConfig actually accepts is not enumerable here (it ends in **kwargs), so we
# only flag keys we KNOW are dead -- these are already present in the base YAMLs.
KNOWN_DEAD_KEYS = {"sparse_prediction", "activation_function"}


def _load(path):
    """Load a training YAML with OmegaConf interpolations resolved when possible.

    The large phase-0 config deliberately interpolates its `base_model` class-selecting fields from
    the `model` block (`${model.model_config.bert_layer}` etc.), so a raw yaml.safe_load sees the
    literal "${...}" string and every registry check misfires. OmegaConf is not installed in the
    login-node python, so fall back to stripping unresolved interpolations: their real values live
    in the block being referenced, which is validated on its own.
    """
    try:
        from omegaconf import OmegaConf  # available inside the training envs

        cfg = OmegaConf.load(path)
        return OmegaConf.to_container(cfg, resolve=True), True
    except ImportError:
        return yaml.safe_load(open(path)), False


def _strip_interp(node):
    """Drop values that are unresolved ${...} references so checks skip rather than false-alarm."""
    if isinstance(node, dict):
        return {k: _strip_interp(v) for k, v in node.items()
                if not (isinstance(v, str) and v.startswith("${"))}
    return node


def check(path):
    errs, warns = [], []
    d, resolved = _load(path)
    if not resolved:
        d = {k: (_strip_interp(v) if isinstance(v, dict) else v) for k, v in d.items()}
        d = {k: ({kk: _strip_interp(vv) for kk, vv in v.items()} if isinstance(v, dict) else v)
             for k, v in d.items()}

    for block in ("model", "base_model"):
        if block not in d:
            continue
        c = d[block]["model_config"]
        tag = f"{block}.model_config"

        # --- configuration_bert.py:231 ---
        gaenl = c.get("global_attn_every_n_layers", -1)
        nl = c["num_hidden_layers"]
        if gaenl > 0 and (nl - 1) % gaenl != 0:
            errs.append(f"{tag}: global_attn_every_n_layers={gaenl} must divide num_hidden_layers-1={nl - 1}")

        # --- configuration_bert.py:236-249 ---
        sw = c.get("sliding_window", -1)
        if sw != -1:
            if not c.get("use_fa2", True):
                errs.append(f"{tag}: sliding_window requires use_fa2=True")
            if sw % 2 != 0 and sw % 64 != 0:
                errs.append(f"{tag}: sliding_window={sw} must be even or divisible by 64")
        else:
            if c.get("local_attn_rotary_emb_base", -1) != -1:
                errs.append(f"{tag}: local_attn_rotary_emb_base must be -1 when sliding_window is disabled")
            if c.get("local_attn_rotary_emb_dim") is not None:
                errs.append(f"{tag}: local_attn_rotary_emb_dim must be None when sliding_window is disabled")

        # --- configuration_bert.py:251-259 ---
        unpad = c.get("unpad_embeddings", False)
        if unpad and c.get("padding", "unpadded") != "unpadded":
            warns.append(f"{tag}: unpad_embeddings=True will silently force padding='unpadded'")
        if c.get("pad_logits", False) and not unpad:
            errs.append(f"{tag}: pad_logits=True requires unpad_embeddings=True")
        if unpad and c.get("embedding_layer") == "absolute_pos":
            errs.append(f"{tag}: unpad_embeddings=True is incompatible with embedding_layer='absolute_pos'")

        # --- flex_bert.py:256-260 ---
        bl = c.get("bert_layer", "prenorm")
        if "prenorm" in bl and not c.get("final_norm", False):
            errs.append(f"{tag}: final_norm must be True with a prenorm bert_layer")
        if "prenorm" not in bl and "postnorm" not in bl:
            errs.append(f"{tag}: bert_layer must contain 'prenorm' or 'postnorm', got {bl!r}")
        if "postnorm" in bl and c.get("final_norm", False):
            errs.append(f"{tag}: final_norm must be False with a postnorm bert_layer")

        # --- model.py:1094 / embeddings.py: compile requires sans_pos ---
        if c.get("compile_model", False) and c.get("embedding_layer") != "sans_pos":
            errs.append(f"{tag}: compile_model=True requires embedding_layer='sans_pos'")

        # --- registry membership ---
        for key, valid in (("normalization", NORMS), ("mlp_layer", MLPS),
                           ("padding", PADDINGS), ("embedding_layer", EMBEDDINGS)):
            if key in c and c[key] not in valid:
                errs.append(f"{tag}: {key}={c[key]!r} not in {sorted(valid)}")
        for key, stems in (("attention_layer", ATTN_STEMS), ("bert_layer", LAYER_STEMS)):
            v = c.get(key, "")
            if v and v.removeprefix("unpadded_").removeprefix("padded_") not in stems:
                errs.append(f"{tag}: {key}={v!r} not a known layer type")

        # --- head dim / vocab padding ---
        if c["hidden_size"] % c["num_attention_heads"] != 0:
            errs.append(f"{tag}: hidden_size {c['hidden_size']} not divisible by {c['num_attention_heads']} heads")
        if c["vocab_size"] % 64 != 0:
            warns.append(f"{tag}: vocab_size {c['vocab_size']} will be auto-padded up to a multiple of 64")

        # --- known no-ops / dead keys ---
        if c.get("loss_kwargs", {}).get("inplace_backward"):
            warns.append(f"{tag}: inplace_backward=True is a NO-OP (config resets it to False)")
        for k in sorted(KNOWN_DEAD_KEYS & set(c)):
            warns.append(f"{tag}: {k!r} is not a FlexBertConfig parameter (silently ignored)")

    # --- cross-file / dataloader invariants ---
    for loader in ("train_loader", "eval_loader"):
        ld = d.get(loader)
        if ld and ld["dataset"].get("streaming", True):
            errs.append(f"{loader}: streaming must be false (StreamingTextDataset NCCL-hangs at init on >1 GPU)")
    if d.get("max_seq_len", 0) >= 8192:
        for loader in ("train_loader", "eval_loader"):
            if d.get(loader) and not d[loader].get("sequence_packing", False):
                errs.append(f"{loader}: sequence_packing must be true at {d['max_seq_len']} (else cross-document attention)")
        if not d.get("load_weights_only", False) and d.get("load_path"):
            errs.append("load_weights_only must be true when chaining phases (else the clock is inherited)")
    if d.get("autoresume") and d.get("spin_dataloaders", True):
        errs.append("spin_dataloaders must be false with autoresume (resume re-iterates 200M+ docs and hangs)")

    # base_model vs model: the fields that select layer CLASSES must agree, or
    # init_model_from_pretrained's isinstance asserts fire at tiling time. Checked against the RAW
    # yaml so an explicit ${model.model_config.X} interpolation counts as agreement by construction.
    raw = yaml.safe_load(open(path))
    if "base_model" in raw and "model" in raw:
        mc, bc = raw["model"]["model_config"], raw["base_model"]["model_config"]
        for key in ("bert_layer", "padding", "embedding_layer", "compile_model"):
            want, got = mc.get(key), bc.get(key)
            if got == "${" + f"model.model_config.{key}" + "}":
                continue  # tracks the model block; safe under CLI overrides
            if want != got:
                errs.append(f"base_model.model_config.{key}={got!r} != model.model_config.{key}={want!r} "
                            f"-- these select layer classes; tiling will fail its isinstance assert")
            else:
                warns.append(f"base_model.model_config.{key} is hardcoded to {got!r}; prefer "
                             f"${{model.model_config.{key}}} so a CLI override cannot desync the two blocks")

    # tile-init block, if present
    ifc = d.get("init_from_checkpoint")
    if ifc:
        if ifc.get("checkpoint_cfg") == "${model}":
            errs.append("init_from_checkpoint.checkpoint_cfg=${model} resolves to THIS file's own (larger) "
                        "model block -- it must point at the PRETRAINED model's config block")
        if "base_model" not in d and ifc.get("checkpoint_cfg", "").startswith("${"):
            warns.append(f"init_from_checkpoint.checkpoint_cfg={ifc.get('checkpoint_cfg')} has no matching block")

    return errs, warns


if __name__ == "__main__":
    paths = sys.argv[1:]
    if not paths:
        print(__doc__)
        sys.exit(2)
    failed = False
    for p in paths:
        errs, warns = check(p)
        status = "FAIL" if errs else "PASS"
        failed |= bool(errs)
        print(f"[{status}] {p}")
        for e in errs:
            print(f"   ERROR {e}")
        for w in warns:
            print(f"   warn  {w}")
    sys.exit(1 if failed else 0)
