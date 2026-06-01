import json
import re
from enum import Enum
from pathlib import Path
from typing import Annotated, cast

import torch
from omegaconf import DictConfig, OmegaConf
import typer
from composer.models import write_huggingface_pretrained_from_composer_checkpoint
from typer import Option
from safetensors.torch import save_file as safetensors_save_file, save_model
from transformers import AutoConfig, AutoModelForMaskedLM  # Change as appropriate

app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]}, pretty_exceptions_show_locals=False)


class TorchDtype(str, Enum):
    float32 = "float32"
    float16 = "float16"
    bfloat16 = "bfloat16"


def update_config(
    source_config: dict,
    bos_token_id: int,
    eos_token_id: int,
    cls_token_id: int,
    pad_token_id: int,
    sep_token_id: int,
    max_length: int,
    vocab_size: int,
    torch_dtype: TorchDtype,
) -> dict:
    target_config = {
        "_name_or_path": "ModernBERT-base",
        "architectures": ["ModernBertForMaskedLM"],
        "attention_bias": source_config["attn_out_bias"],
        "attention_dropout": source_config["attention_probs_dropout_prob"],
        "bos_token_id": bos_token_id,
        "classifier_activation": source_config.get("head_class_act", source_config["hidden_act"]),
        "classifier_bias": source_config.get("head_class_bias", False),
        "classifier_dropout": source_config.get("head_class_dropout", 0.0),
        "classifier_pooling": source_config.get("classifier_pooling", "mean"),
        "cls_token_id": cls_token_id,
        "decoder_bias": source_config.get("decoder_bias", True),
        "deterministic_flash_attn": source_config.get("deterministic_fa2", False),
        "embedding_dropout": source_config["embed_dropout_prob"],
        "eos_token_id": eos_token_id,
        "global_attn_every_n_layers": source_config["global_attn_every_n_layers"],
        "global_rope_theta": source_config["rotary_emb_base"],
        "gradient_checkpointing": source_config.get("gradient_checkpointing", False),
        "hidden_activation": source_config["hidden_act"],
        "hidden_size": source_config["hidden_size"],
        "initializer_cutoff_factor": source_config.get("init_cutoff_factor", 2.0),
        "initializer_range": source_config.get("initializer_range", 0.02),
        "intermediate_size": source_config["intermediate_size"],
        "layer_norm_eps": source_config["norm_kwargs"]["eps"],
        "local_attention": source_config["sliding_window"],
        "local_rope_theta": source_config.get("local_attn_rotary_emb_base", source_config["rotary_emb_base"]),
        "max_position_embeddings": max_length,  # Override with first config value
        "mlp_bias": source_config["mlp_in_bias"],
        "mlp_dropout": source_config["mlp_dropout_prob"],
        "model_type": "modernbert",
        "norm_bias": source_config["norm_kwargs"]["bias"],
        "norm_eps": source_config["norm_kwargs"]["eps"],
        "num_attention_heads": source_config["num_attention_heads"],
        "num_hidden_layers": source_config["num_hidden_layers"],
        "pad_token_id": pad_token_id,
        "position_embedding_type": source_config.get("position_embedding_type", "absolute"),
        "sep_token_id": sep_token_id,
        "tie_word_embeddings": source_config.get("tie_word_embeddings", True),
        "torch_dtype": torch_dtype.value,
        "transformers_version": "4.48.0",
        "vocab_size": vocab_size,
    }
    target_config2 = {
        "_name_or_path": "ModernBERT-base",
        "architectures": ["ModernBertForMaskedLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 50281,
        "classifier_activation": "gelu",
        "classifier_bias": False,
        "classifier_dropout": 0.0,
        "classifier_pooling": "mean",
        "cls_token_id": 50281,
        "decoder_bias": True,
        "deterministic_flash_attn": False,
        "embedding_dropout": 0.0,
        "eos_token_id": 50282,
        "global_attn_every_n_layers": 3,
        "global_rope_theta": 160000.0,
        "gradient_checkpointing": False,
        "hidden_activation": "gelu",
        "hidden_size": 768,
        "initializer_cutoff_factor": 2.0,
        "initializer_range": 0.02,
        "intermediate_size": 1152,
        "layer_norm_eps": 1e-05,
        "local_attention": 128,
        "local_rope_theta": 10000.0,
        "max_position_embeddings": 8192,
        "mlp_bias": False,
        "mlp_dropout": 0.0,
        "model_type": "modernbert",
        "norm_bias": False,
        "norm_eps": 1e-05,
        "num_attention_heads": 12,
        "num_hidden_layers": 22,
        "pad_token_id": 50283,
        "position_embedding_type": "absolute",
        "sep_token_id": 50282,
        "tie_word_embeddings": True,
        "torch_dtype": "float32",
        "transformers_version": "4.48.0",
        "vocab_size": 50368
    }
    return target_config


@app.command(help="Convert a ModernBERT Composer checkpoint to HuggingFace pretrained format.")
def main(
    yaml_config: Annotated[Path, Option(help="Path to the YAML configuration file", show_default=False)],
    output_name: Annotated[str, Option(help="Name of the output model", show_default=False)],
    output_dir: Annotated[Path, Option(help="Path to the output directory", show_default=False)],
    input_checkpoint: Annotated[Path, Option(help="Path to the ModernBERT Composer checkpoint file", show_default=False)],
    bos_token_id: Annotated[int, Option(help="ID of the BOS token. Defaults to the ModernBERT BOS token.")] = 50281,
    eos_token_id: Annotated[int, Option(help="ID of the EOS token. Defaults to the ModernBERT EOS token.")] = 50282,
    cls_token_id: Annotated[int, Option(help="ID of the CLS token. Defaults to the ModernBERT CLS token.")] = 50281,
    sep_token_id: Annotated[int, Option(help="ID of the SEP token. Defaults to the ModernBERT SEP token.")] = 50282,
    pad_token_id: Annotated[int, Option(help="ID of the PAD token. Defaults to the ModernBERT PAD token.")] = 50283,
    mask_token_id: Annotated[int, Option(help="ID of the MASK token. Defaults to the ModernBERT MASK token.")] = 50284,
    max_length: Annotated[int, Option(help="Maximum length of the input sequence. Defaults to the final ModernBERT sequence length.")] = 8192,
    vocab_size: Annotated[int, Option(help="Vocabulary size. Defaults to the ModernBERT vocabulary size.")] = 50368,
    torch_dtype: Annotated[TorchDtype, Option(help="Torch dtype to use for the model.")] = TorchDtype.float32,
    pytorch_bin: Annotated[bool, Option(help="Save weights as a pytorch_model.bin file.")] = True,
    safetensors: Annotated[bool, Option(help="Save weights as a model.safetensors file.")] = True,
    drop_tied_decoder_weights: Annotated[bool, Option(help="Don't save the wieght tied decoder weights.")] = True,
):  # fmt: skip
    """
    Convert a ModernBERT Composer checkpoint to HuggingFace pretrained format.
    """
    target_path = f"{output_dir}/{output_name}"
    write_huggingface_pretrained_from_composer_checkpoint(input_checkpoint, target_path)

    # Process pytorch_model.bin
    state_dict_path = f"{target_path}/pytorch_model.bin"
    state_dict = torch.load(state_dict_path, map_location=torch.device("cpu"))
    var_map = (
        (re.compile(r"encoder\.layers\.(.*)"), r"layers.\1"),
        (re.compile(r"^bert\.(.*)"), r"model.\1"),  # Replaces 'bert.' with '' at the start of keys
    )
    for pattern, replacement in var_map:
        state_dict = {re.sub(pattern, replacement, name): tensor for name, tensor in state_dict.items()}

    # Update config.json
    with open(yaml_config) as f:
        config = OmegaConf.load(f)
        model_config = OmegaConf.to_container(config.model.model_config, resolve=True)
        config_dict = cast(DictConfig, model_config)
        config_dict = update_config(
            config_dict, bos_token_id, eos_token_id, cls_token_id, pad_token_id, sep_token_id, max_length, vocab_size, torch_dtype
        )

    config_json_path = f"{target_path}/config.json"
    with open(config_json_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    if config_dict.get("tie_word_embeddings", False) and drop_tied_decoder_weights:
        if "decoder.weight" in state_dict:
            del state_dict["decoder.weight"]

    # Export to pytorch_model.bin
    if pytorch_bin:
        torch.save(state_dict, state_dict_path)

    # Export to safetensors
    if safetensors:
        safetensors_path = f"{target_path}/model.safetensors"
        safetensors_save_file(state_dict, safetensors_path)

    # Update tokenizer_config.json
    tokenizer_config_path = f"{target_path}/tokenizer_config.json"
    with open(tokenizer_config_path, "r") as f:
        config_dict = json.load(f)
    config_dict["model_max_length"] = max_length
    config_dict["added_tokens_decoder"][str(mask_token_id)]["lstrip"] = True
    config_dict["model_input_names"] = ["input_ids", "attention_mask"]
    config_dict["tokenizer_class"] = "PreTrainedTokenizerFast"

    if "extra_special_tokens" in config_dict:
        del config_dict["extra_special_tokens"]
    with open(tokenizer_config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    # Update special_tokens_map.json
    special_tokens_path = f"{target_path}/special_tokens_map.json"
    with open(special_tokens_path, "r") as f:
        config_dict = json.load(f)
    config_dict["mask_token"]["lstrip"] = True
    with open(special_tokens_path, "w") as f:
        json.dump(config_dict, f, indent=2)


if __name__ == "__main__":
    app()
