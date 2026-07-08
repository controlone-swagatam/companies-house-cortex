"""
Raw event envelope for the receptor layer.

Per the architecture's locked separation: source receptors are pure ingestion.
This envelope wraps a CH filingHistoryItem verbatim in a thin transport
wrapper (source, company_number, ingestion timestamp, raw payload). It does
NOT extract category/subtype into NTST's event_type/event_subtype fields —
that mapping is business logic and belongs to a separate derived (rules)
layer reading off the Kafka spine, not to this receptor.

`run_mode` and `period` are ingestion-run metadata, not filing-type
interpretation — they record which pipeline run and (for training) which
date-bounded split an event was ingested under, so downstream consumers can
route/filter without re-deriving it. This is transport bookkeeping, not
business logic: it says nothing about what the filing *is*, only which
batch it arrived in.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass(frozen=True)
class RawFilingEvent:
    event_id: str
    source_type: str
    company_number: str
    ingested_at: str
    payload: dict[str, Any]
    run_mode: str = "inference"          # "training" | "inference"
    period: Optional[str] = None          # "period_1" | "period_2" | None (inference)

    @classmethod
    def from_filing_item(
        cls,
        company_number: str,
        item: dict[str, Any],
        run_mode: str = "inference",
        period: Optional[str] = None,
    ) -> "RawFilingEvent":
        return cls(
            event_id=str(uuid.uuid4()),
            source_type="companies_house.filing_history",
            company_number=company_number,
            ingested_at=datetime.now(timezone.utc).isoformat(),
            payload=item,
            run_mode=run_mode,
            period=period,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "source_type": self.source_type,
            "company_number": self.company_number,
            "ingested_at": self.ingested_at,
            "payload": self.payload,
            "run_mode": self.run_mode,
            "period": self.period,
        }
