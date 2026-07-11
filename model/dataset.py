"""
Dataset for training. Groups derived events by company, sorts
chronologically (by payload.date — the observed/filed date, NOT
action_date, per §9's no-leakage principle), truncates/pads to
max_seq_len, encodes via EventVocab.

One sequence = one company's filing history. This is why EntityEmbedding
is sequence-level (§ reconciliation with model.py) — company_id is
constant across the whole sequence by construction.
"""
import json
from collections import defaultdict
from datetime import datetime

import torch
from torch.utils.data import Dataset

from model.vocab import EventVocab


def _parse_date(date_str: str):
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def load_company_sequences(derived_event_paths: list[str]) -> dict[str, list[dict]]:
    """
    Reads derived_events JSONL file(s), groups by company_number, sorts
    each company's events chronologically by date. Events with no parseable
    date are dropped (can't be ordered).
    """
    by_company: dict[str, list[dict]] = defaultdict(list)

    for path in derived_event_paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                event = json.loads(line)
                if _parse_date(event.get("date")) is not None:
                    by_company[event["company_number"]].append(event)

    for company, events in by_company.items():
        events.sort(key=lambda e: _parse_date(e["date"]))

    return dict(by_company)


class CompanySequenceDataset(Dataset):
    def __init__(
        self,
        company_sequences: dict[str, list[dict]],
        vocab: EventVocab,
        max_seq_len: int = 64,
        min_seq_len: int = 2,  # sequences shorter than this have nothing to predict causally
    ):
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        # Drop companies with too few events — nothing meaningful to learn
        # from a 1-event sequence under causal masking.
        self.sequences = [
            (company, events) for company, events in company_sequences.items()
            if len(events) >= min_seq_len
        ]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        company_number, events = self.sequences[idx]
        events = events[-self.max_seq_len:]  # keep most recent if too long
        L = len(events)

        category_ids = torch.tensor([self.vocab.encode_category(e["category"]) for e in events])
        type_ids = torch.tensor([self.vocab.encode_type(e["type"]) for e in events])
        subtype_ids = torch.tensor([self.vocab.encode_subtype(e["subtype"]) for e in events])
        company_id = torch.tensor(self.vocab.encode_company(company_number))
        positions = torch.arange(L)

        return {
            "category_ids": category_ids,
            "type_ids": type_ids,
            "subtype_ids": subtype_ids,
            "company_id": company_id,
            "positions": positions,
            "length": L,
        }


def collate_fn(batch: list[dict], pad_id: int = 0) -> dict:
    max_len = max(item["length"] for item in batch)
    B = len(batch)

    category_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    type_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    subtype_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    positions = torch.zeros((B, max_len), dtype=torch.long)
    attention_mask = torch.zeros((B, max_len), dtype=torch.bool)
    company_ids = torch.zeros((B,), dtype=torch.long)

    for i, item in enumerate(batch):
        L = item["length"]
        category_ids[i, :L] = item["category_ids"]
        type_ids[i, :L] = item["type_ids"]
        subtype_ids[i, :L] = item["subtype_ids"]
        positions[i, :L] = item["positions"]
        attention_mask[i, :L] = True
        company_ids[i] = item["company_id"]

    return {
        "category_ids": category_ids,
        "type_ids": type_ids,
        "subtype_ids": subtype_ids,
        "company_ids": company_ids,
        "positions": positions,
        "attention_mask": attention_mask,
    }
