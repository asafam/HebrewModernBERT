"""Stream records out of a HuggingFace dataset.

Mainly used for the tokenizer-corpus build (the pretraining mixes feed pre-sampled
JSONL instead — see prepare_dolma_dataset.py). Like the JSONL handler, this yields
ALL rows (split routing happens once, downstream) and computes ``guid`` lazily.

Note: the original implementation iterated the whole dataset several times just to
count it, hard-coded a broken ``tokenize(text)`` call for token-budgeting, and did
its own index-based split. Those are removed here; for token-budgeted HF sampling
use prepare_dolma_dataset.py (batched tokenization).
"""

from typing import Iterator, List, Optional

from datasets import concatenate_datasets, load_dataset

from ..utils import hash_text


class DatasetFormatHF:
    def __init__(self, dataset_name: str):
        self.name = dataset_name

    def stream(
        self,
        hf_dataset_args: dict,
        text_field: str = "text",
        filter_criteria: Optional[List[dict]] = None,
        limit: int = 0,
        shuffle: bool = True,
        compute_guid: bool = False,
        random_state: int = 42,
    ) -> Iterator[dict]:
        dataset = load_dataset(**hf_dataset_args)

        if filter_criteria:
            parts = [
                dataset.filter(lambda ex, c=criteria: all(ex.get(f) == v for f, v in c.items()))
                for criteria in filter_criteria
            ]
            dataset = concatenate_datasets(parts)

        if shuffle:
            dataset = dataset.shuffle(seed=random_state)

        for idx, record in enumerate(dataset):
            if limit and idx >= limit:
                return
            text = record[text_field]
            out = {"text": text, "_source": self.name, "_row_number": idx}
            if compute_guid:
                out["guid"] = hash_text(text)
            yield out
