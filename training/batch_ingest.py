"""
Training batch ingestion.

For each company: fetch the FULL filing history (the CH filing-history
endpoint has no server-side date filter), then split events locally by
`payload.date` against two runtime-supplied date ranges:

    period_1 = training split
    period_2 = held-out test split

This is still pure ingestion, not feature engineering — the split is a
date-boundary partition of raw events into two physically separate streams,
not an interpretation of what any filing means. Events outside both ranges
are dropped (logged, not silently discarded).

Usage:
    export CH_API_KEY=...
    python -m training.batch_ingest \\
        --company-numbers-file companies.txt \\
        --period-1-start 2019-01-01 --period-1-end 2022-12-31 \\
        --period-2-start 2023-01-01 --period-2-end 2024-12-31 \\
        --sink jsonl
"""
import argparse
import logging
from datetime import date, datetime

from common.ch_client import CompaniesHouseClient
from common.config import base_config_from_env, with_overrides
from common.envelope import RawFilingEvent
from common.sinks import build_sink

logger = logging.getLogger("ch_pipeline.training.batch_ingest")


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def classify_period(
    filing_date_str: str,
    p1_start: date, p1_end: date,
    p2_start: date, p2_end: date,
) -> str | None:
    """Returns 'period_1', 'period_2', or None if outside both ranges."""
    try:
        d = _parse_date(filing_date_str)
    except (ValueError, TypeError):
        return None

    if p1_start <= d <= p1_end:
        return "period_1"
    if p2_start <= d <= p2_end:
        return "period_2"
    return None


def read_company_numbers(args: argparse.Namespace) -> list[str]:
    numbers: list[str] = list(args.company_numbers or [])
    if args.company_numbers_file:
        with open(args.company_numbers_file, encoding="utf-8") as fh:
            numbers.extend(line.strip() for line in fh if line.strip())
    if not numbers:
        raise SystemExit("No company numbers provided (--company-numbers or --company-numbers-file)")
    return numbers


def run(
    company_numbers: list[str],
    p1_start: date, p1_end: date,
    p2_start: date, p2_end: date,
    sink_override: str | None = None,
) -> None:
    config = base_config_from_env()
    overrides = {"route_by_period": True}
    if sink_override:
        overrides["sink"] = sink_override
    config = with_overrides(config, **overrides)

    client = CompaniesHouseClient(config)
    sink = build_sink(config)

    counts = {"period_1": 0, "period_2": 0, "dropped_out_of_range": 0}
    try:
        for company_number in company_numbers:
            logger.info("Fetching full filing history for %s", company_number)
            for raw_item in client.iter_filing_history(company_number):
                period = classify_period(
                    raw_item.get("date"), p1_start, p1_end, p2_start, p2_end
                )
                if period is None:
                    counts["dropped_out_of_range"] += 1
                    continue

                event = RawFilingEvent.from_filing_item(
                    company_number, raw_item, run_mode="training", period=period
                )
                sink.send(event.to_dict())
                counts[period] += 1

            logger.info("  done: %s", company_number)
    finally:
        sink.close()

    logger.info(
        "Training batch ingest complete. period_1=%d period_2=%d dropped(out_of_range)=%d",
        counts["period_1"], counts["period_2"], counts["dropped_out_of_range"],
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Companies House training batch ingester")
    parser.add_argument("--company-numbers", nargs="*")
    parser.add_argument("--company-numbers-file")
    parser.add_argument("--period-1-start", required=True, type=_parse_date)
    parser.add_argument("--period-1-end", required=True, type=_parse_date)
    parser.add_argument("--period-2-start", required=True, type=_parse_date)
    parser.add_argument("--period-2-end", required=True, type=_parse_date)
    parser.add_argument("--sink", choices=["kafka", "stdout", "jsonl"])
    args = parser.parse_args()

    if args.period_1_end >= args.period_2_start:
        logger.warning(
            "period_1_end (%s) is not before period_2_start (%s) — "
            "ranges overlap or are out of order. Proceeding, but check this "
            "is intentional (a leaked overlap would let training data leak into test).",
            args.period_1_end, args.period_2_start,
        )

    company_numbers = read_company_numbers(args)
    run(
        company_numbers,
        args.period_1_start, args.period_1_end,
        args.period_2_start, args.period_2_end,
        sink_override=args.sink,
    )


if __name__ == "__main__":
    main()
