"""
Vocabulary builder for the 3-level event hierarchy (category → type → subtype).

Unlike GDELT's fixed CAMEO vocabulary, Companies House's vocab isn't known
in advance — §6/§9/§11 all came from inspecting real data. This builds the
category/type/subtype -> int mappings FROM the derived event stream (which
already has §11's correction applied), rather than hardcoding anything.

Reserved index 0 in every vocab is PAD, matching the padding_idx=0
convention (padding_idx zeroes gradient + output for that row, standard
practice for variable-length sequences).

Usage:
    python3 -m model.vocab --input ./output/derived_events.period_1.jsonl \
        ./output/derived_events.period_2.jsonl \
        --output ./output/vocab.json
"""
import argparse
import json
import logging
from collections import Counter

logger = logging.getLogger("ch_pipeline.model.vocab")

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"  # for subtype/type values not seen during vocab building (test-time novelty)
MASK_TOKEN = "<MASK>"  # for MLM-style masked prediction training


class EventVocab:
    """
    Three independent vocabularies (category, type, subtype), each with
    PAD=0 and UNK=1 reserved. UNK matters more for `type` and `subtype`
    than `category` — §6 showed a long tail of rare subtypes, some of
    which won't appear in period_1 (train) but could appear in period_2
    (test) or at inference time.
    """

    def __init__(self, category_to_id: dict, type_to_id: dict, subtype_to_id: dict, company_to_id: dict):
        self.category_to_id = category_to_id
        self.type_to_id = type_to_id
        self.subtype_to_id = subtype_to_id
        self.company_to_id = company_to_id

    @property
    def n_categories(self) -> int:
        return len(self.category_to_id)

    @property
    def n_types(self) -> int:
        return len(self.type_to_id)

    @property
    def n_subtypes(self) -> int:
        return len(self.subtype_to_id)

    @property
    def n_companies(self) -> int:
        return len(self.company_to_id)

    def encode_company(self, value: str) -> int:
        return self.company_to_id.get(value, self.company_to_id[UNK_TOKEN])

    @property
    def category_mask_id(self) -> int:
        return self.category_to_id[MASK_TOKEN]

    @property
    def type_mask_id(self) -> int:
        return self.type_to_id[MASK_TOKEN]

    @property
    def subtype_mask_id(self) -> int:
        return self.subtype_to_id[MASK_TOKEN]

    def encode_category(self, value: str) -> int:
        return self.category_to_id.get(value, self.category_to_id[UNK_TOKEN])

    def encode_type(self, value: str) -> int:
        return self.type_to_id.get(value, self.type_to_id[UNK_TOKEN])

    def encode_subtype(self, value: str) -> int:
        return self.subtype_to_id.get(value, self.subtype_to_id[UNK_TOKEN])

    def to_dict(self) -> dict:
        return {
            "category_to_id": self.category_to_id,
            "type_to_id": self.type_to_id,
            "subtype_to_id": self.subtype_to_id,
            "company_to_id": self.company_to_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EventVocab":
        return cls(d["category_to_id"], d["type_to_id"], d["subtype_to_id"], d["company_to_id"])

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "EventVocab":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def _build_single_vocab(values: list[str], include_mask: bool = False) -> dict:
    counter = Counter(values)
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    if include_mask:
        vocab[MASK_TOKEN] = 2
    # Sort by frequency descending for readability/debuggability — doesn't
    # affect model behaviour, embedding lookup doesn't care about id order.
    for value, _ in counter.most_common():
        if value not in vocab:
            vocab[value] = len(vocab)
    return vocab


def build_vocab_from_derived_events(derived_event_paths: list[str]) -> EventVocab:
    """
    Builds vocab from the given files. Should be called on period_1
    (training) data ONLY — including period_2 here would leak test-set
    categories/types/subtypes into the vocab, undermining the whole point
    of UNK handling and the period_1/period_2 split. This function warns
    (doesn't block) if a period_2 file looks like it's been passed in,
    since file naming isn't guaranteed to follow convention.
    """
    for path in derived_event_paths:
        if "period_2" in path or "period2" in path:
            logger.warning(
                "Input path '%s' looks like test-split (period_2) data. "
                "Building vocab from test data leaks test-set categories "
                "into the model and defeats UNK handling — confirm this is "
                "intentional.", path,
            )

    categories, types, subtypes, companies = [], [], [], []

    for path in derived_event_paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                event = json.loads(line)
                if event.get("category"):
                    categories.append(event["category"])
                if event.get("type"):
                    types.append(event["type"])
                if event.get("subtype"):
                    subtypes.append(event["subtype"])
                if event.get("company_number"):
                    companies.append(event["company_number"])

    logger.info(
        "Building vocab from %d category, %d type, %d subtype, %d company observations",
        len(categories), len(types), len(subtypes), len(companies),
    )

    vocab = EventVocab(
        category_to_id=_build_single_vocab(categories, include_mask=True),
        type_to_id=_build_single_vocab(types, include_mask=True),
        subtype_to_id=_build_single_vocab(subtypes, include_mask=True),
        company_to_id=_build_single_vocab(companies, include_mask=False),
    )
    logger.info(
        "Vocab sizes: category=%d type=%d subtype=%d company=%d",
        vocab.n_categories, vocab.n_types, vocab.n_subtypes, vocab.n_companies,
    )
    return vocab


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Build category/type/subtype vocab from derived events")
    parser.add_argument("--input", nargs="+", required=True, help="One or more derived_events JSONL files")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    vocab = build_vocab_from_derived_events(args.input)
    vocab.save(args.output)
    logger.info("Vocab written to %s", args.output)


if __name__ == "__main__":
    main()
