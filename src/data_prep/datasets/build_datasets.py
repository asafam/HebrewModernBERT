"""Build a pretraining/tokenizer corpus from multiple sources into MDS (or JSONL/TXT).

Reads a YAML config listing sources (each ``type: jsonl`` or ``type: hf``) and writes
a single dataset with ``text / _source / _row_number`` columns.

Efficiency vs. the original (in ../hebrew_text_retrieval):
  * **single pass**: train and validation are written together in one read of the
    corpus (the original ran once per split, reading the whole corpus twice);
  * split is decided per-source with a seeded RNG (reproducible) from each source's
    ``split_ratio.validation``;
  * dedup (optional) shares one global ``seen`` set across sources and only then is
    the ``guid`` hash computed (off by default — see dataset_format_jsonl.py).

Run from the repo root, e.g.:
    python -m src.data_prep.datasets.build_datasets \
        --config_file src/data_prep/config/train/datasets_h50_e75_c25.yaml \
        --output_path data/hebrewmodernbert/mixed/mixed_h50e75c25 --format mds
"""

import argparse
import contextlib
import json
import os
import random
from itertools import chain
from typing import Iterator, Tuple

import yaml

from .dataset_formats.dataset_format_hf import DatasetFormatHF
from .dataset_formats.dataset_format_jsonl import DatasetFormatJSONL
from .utils import MDSWriter

SPLITS = ("train", "validation")
COLUMNS = {"text": "str", "_source": "str", "_row_number": "int"}


def _source_stream(cfg: dict, compute_guid: bool) -> Iterator[dict]:
    """Build the record generator for one source entry."""
    if cfg["type"] == "jsonl":
        files = cfg.get("files") or [cfg["dir"]]
        return DatasetFormatJSONL(cfg["name"]).stream(
            jsonl_files=files,
            exclude_jsonl_files=cfg.get("exclude_files", []),
            text_field=cfg.get("text_field", "text"),
            guid_field=cfg.get("guid_field"),
            compute_guid=compute_guid,
            limit=cfg.get("limit", 0),
        )
    if cfg["type"] == "hf":
        return DatasetFormatHF(cfg["name"]).stream(
            hf_dataset_args=cfg.get("args", {}),
            text_field=cfg.get("text_field", "text"),
            filter_criteria=cfg.get("filter_criteria"),
            limit=cfg.get("limit", 0),
            compute_guid=compute_guid,
        )
    raise ValueError(f"Unknown source type: {cfg['type']!r}")


def _dedup(records: Iterator[dict], seen: set) -> Iterator[dict]:
    for r in records:
        g = r["guid"]
        if g in seen:
            continue
        seen.add(g)
        yield r


def _tag_split(records: Iterator[dict], val_fraction: float, seed: int) -> Iterator[Tuple[str, dict]]:
    rng = random.Random(seed)
    for r in records:
        yield ("validation" if rng.random() < val_fraction else "train"), r


def build(
    config_file: str,
    output_path: str,
    fmt: str = "mds",
    shard_size_limit: int = 67108864,
    default_validation_fraction: float = 0.01,
    remove_duplicates: bool = False,
    compression: str = None,
    random_state: int = 42,
) -> None:
    with open(config_file) as f:
        sources = yaml.safe_load(f)

    seen = set() if remove_duplicates else None

    def all_tagged() -> Iterator[Tuple[str, dict]]:
        for i, src in enumerate(sources):
            print(f"Source: {src['name']} ({src['type']})")
            stream = _source_stream(src, compute_guid=remove_duplicates)
            if remove_duplicates:
                stream = _dedup(stream, seen)
            val_frac = src.get("split_ratio", {}).get("validation", default_validation_fraction)
            yield from _tag_split(stream, val_frac, seed=random_state + i)

    fmt = fmt.lower()
    counts = {s: 0 for s in SPLITS}
    with contextlib.ExitStack() as stack:
        if fmt == "mds":
            if MDSWriter is None:
                raise ImportError("mosaicml-streaming is required for --format mds")
            writers = {
                s: stack.enter_context(
                    MDSWriter(out=os.path.join(output_path, s), columns=COLUMNS,
                              size_limit=shard_size_limit, compression=compression)
                )
                for s in SPLITS
            }

            def write(split, r):
                writers[split].write({k: r[k] for k in COLUMNS})
        elif fmt in ("jsonl", "txt"):
            os.makedirs(output_path, exist_ok=True)
            files = {s: stack.enter_context(open(os.path.join(output_path, f"{s}.{fmt}"), "w", encoding="utf-8"))
                     for s in SPLITS}
            if fmt == "jsonl":
                def write(split, r):
                    files[split].write(json.dumps({k: r[k] for k in COLUMNS}, ensure_ascii=False) + "\n")
            else:
                def write(split, r):
                    files[split].write(r["text"].replace("\n", " ") + "\n")
        else:
            raise ValueError(f"Unknown format: {fmt!r} (use mds/jsonl/txt)")

        for split, r in all_tagged():
            write(split, r)
            counts[split] += 1

    print(f"Done. Wrote {counts['train']:,} train + {counts['validation']:,} validation records to {output_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Build a multi-source corpus into MDS/JSONL/TXT")
    p.add_argument("--config_file", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--format", default="mds", choices=["mds", "jsonl", "txt"])
    p.add_argument("--shard_size_limit", type=int, default=67108864)
    p.add_argument("--validation_fraction", type=float, default=0.01,
                   help="fallback val fraction for sources without split_ratio.validation")
    p.add_argument("--remove_duplicates", action="store_true")
    p.add_argument("--compression", default=None)
    p.add_argument("--random_state", type=int, default=42)
    a = p.parse_args()
    build(
        config_file=a.config_file,
        output_path=a.output_path,
        fmt=a.format,
        shard_size_limit=a.shard_size_limit,
        default_validation_fraction=a.validation_fraction,
        remove_duplicates=a.remove_duplicates,
        compression=a.compression,
        random_state=a.random_state,
    )


if __name__ == "__main__":
    main()
