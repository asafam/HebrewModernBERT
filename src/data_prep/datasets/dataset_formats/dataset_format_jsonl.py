"""Stream records out of JSONL files (or directories of .jsonl files).

Efficiency notes vs. the original (in ../hebrew_text_retrieval):
  * yields ALL rows (the train/validation split is decided once, downstream in
    build_datasets, so the corpus is read a single time instead of once per split);
  * the SHA-256 ``guid`` is computed lazily (only when ``compute_guid`` is set for
    dedup) instead of hashing every document's full text on every run;
  * drops the per-record "lowercase every key" dict-comprehension whose output was
    discarded by the MDS writer anyway.
"""

from typing import Iterator, List, Optional
import json
import os
from pathlib import Path

from tqdm import tqdm

from ..utils import hash_text


class DatasetFormatJSONL:
    def __init__(self, name: str):
        self.name = name

    def _resolve_files(self, jsonl_files, exclude_jsonl_files) -> List[str]:
        files: List[str] = []
        for entry in jsonl_files:
            if os.path.isdir(entry):
                files += sorted(str(f) for f in Path(entry).glob("*.jsonl"))
            else:
                files.append(entry)
        exclude = set(os.path.basename(e) for e in (exclude_jsonl_files or []))
        return [f for f in files if os.path.basename(f) not in exclude]

    def stream(
        self,
        jsonl_files: List[str],
        exclude_jsonl_files: Optional[List[str]] = None,
        text_field: str = "text",
        guid_field: Optional[str] = None,
        compute_guid: bool = False,
        limit: int = 0,
        encoding: str = "utf-8-sig",
    ) -> Iterator[dict]:
        """Yield ``{"text", "_source", "_row_number"[, "guid"]}`` for every row."""
        files = self._resolve_files(list(jsonl_files), exclude_jsonl_files)
        yielded = 0
        for file_path in tqdm(files, desc=f"{self.name}: reading JSONL"):
            source = f"{self.name}_{os.path.basename(file_path)}"
            with open(file_path, "r", encoding=encoding) as f:
                for idx, line in enumerate(f):
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = data[text_field]
                    record = {
                        "text": text,
                        "_source": data.get("_source", source),
                        "_row_number": idx,
                    }
                    if compute_guid:
                        record["guid"] = data[guid_field] if guid_field else hash_text(text)
                    yield record
                    yielded += 1
                    if limit and yielded >= limit:
                        return
