"""
Derived event construction.

Takes a raw envelope event (as produced by the receptor — untouched CH
payload) and produces a derived event carrying:
  - the corrected category (§11 rules applied)
  - the 3-level hierarchy fields ready for EventEmbedding (§9): category,
    type (form code), subtype (description slug)
  - the original raw payload, preserved, for audit — the correction is
    additive, not destructive

This is the boundary the working doc keeps drawing: receptors are pure
ingestion, derived layers are where business logic (like category
correction) belongs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from derived.category_rules import correct_category


@dataclass(frozen=True)
class DerivedEvent:
    event_id: str
    company_number: str
    run_mode: str
    period: Optional[str]

    # 3-level hierarchy, post-correction (§7, §9, §11)
    category: str
    category_original: str
    category_rule_applied: Optional[str]
    type: Optional[str]
    subtype: Optional[str]

    # temporal anchor — payload.date, NOT action_date (§9)
    date: Optional[str]
    action_date: Optional[str]

    description_values: dict[str, Any]
    raw_payload: dict[str, Any]

    @classmethod
    def from_raw_event(cls, raw_event: dict[str, Any]) -> "DerivedEvent":
        payload = raw_event.get("payload", {})
        corrected_category, rule_applied = correct_category(payload)

        return cls(
            event_id=raw_event["event_id"],
            company_number=raw_event["company_number"],
            run_mode=raw_event.get("run_mode", "inference"),
            period=raw_event.get("period"),
            category=corrected_category,
            category_original=payload.get("category"),
            category_rule_applied=rule_applied,
            type=payload.get("type"),
            subtype=payload.get("description"),
            date=payload.get("date"),
            action_date=payload.get("action_date"),
            description_values=payload.get("description_values", {}),
            raw_payload=payload,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "company_number": self.company_number,
            "run_mode": self.run_mode,
            "period": self.period,
            "category": self.category,
            "category_original": self.category_original,
            "category_rule_applied": self.category_rule_applied,
            "type": self.type,
            "subtype": self.subtype,
            "date": self.date,
            "action_date": self.action_date,
            "description_values": self.description_values,
            "raw_payload": self.raw_payload,
        }
