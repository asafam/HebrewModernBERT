"""Shared helpers for the data-prep pipeline (MDS/JSONL/TXT writers, hashing)."""

from typing import Dict, Iterable, Optional
import hashlib
import json
from pathlib import Path

from tqdm import tqdm

try:
    from streaming import MDSWriter
except ImportError:  # streaming is only needed for MDS output
    MDSWriter = None


def hash_text(text: str) -> str:
    """SHA-256 hex digest of ``text`` (used for cross-source dedup)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save_as_mds(
    records: Iterable[dict],
    columns: Dict[str, str],
    output_dir: str,
    shard_size_limit: int,
    compression: Optional[str] = None,
) -> int:
    """Stream ``records`` into an MDS dataset directory. Returns the record count."""
    if MDSWriter is None:
        raise ImportError("mosaicml-streaming is required for MDS output (`pip install mosaicml-streaming`).")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    n = 0
    with MDSWriter(out=output_dir, columns=columns, size_limit=shard_size_limit, compression=compression) as writer:
        for record in tqdm(records, desc=f"MDS -> {output_dir}"):
            writer.write({k: record[k] for k in columns})  # write only declared columns
            n += 1
    return n


def save_as_jsonl(records: Iterable[dict], output_file: str, columns: Optional[Iterable[str]] = None) -> int:
    """Stream ``records`` to a JSONL file. If ``columns`` is given, keep only those keys."""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    cols = list(columns) if columns else None
    n = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for record in tqdm(records, desc=f"JSONL -> {output_file}"):
            out = {k: record[k] for k in cols} if cols else record
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
    return n


def save_as_txt(records: Iterable[dict], output_file: str, column: str = "text") -> int:
    """Stream the ``column`` field of ``records`` to a plain-text file (one doc per line)."""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for record in tqdm(records, desc=f"TXT -> {output_file}"):
            f.write(record[column].replace("\n", " ") + "\n")
            n += 1
    return n
