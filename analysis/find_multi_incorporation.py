"""
Diagnostic: find companies whose filing history contains more than one
`category == "incorporation"` item. A company should only ever have one
incorporation event — more than one signals either a genuine data anomaly
worth escalating to CH, or a company that's been dissolved/restored/
re-incorporated in a way that produces a second incorporation-category
record (e.g. after a restoration order).

Reuses the same advanced-search sampling as sample_taxonomy.py so results
are directly comparable to a prior run with the same filters.

Usage:
    export CH_API_KEY=...
    python3 -m analysis.find_multi_incorporation \
        --sample-size 200 --company-status active \
        --incorporated-from 2015-01-01 --incorporated-to 2024-12-31
"""
import argparse
import logging

from common.ch_client import CompaniesHouseClient
from common.config import base_config_from_env

logger = logging.getLogger("ch_pipeline.analysis.find_multi_incorporation")


def run(sample_size: int, search_filters: dict) -> dict:
    config = base_config_from_env()
    client = CompaniesHouseClient(config)

    logger.info("Sampling %d companies (filters=%s)", sample_size, search_filters)
    company_numbers = [
        item.get("company_number")
        for item in client.iter_advanced_search(sample_size, **search_filters)
        if item.get("company_number")
    ]
    logger.info("Checking incorporation-event counts for %d companies", len(company_numbers))

    offenders: dict[str, list[dict]] = {}
    for i, company_number in enumerate(company_numbers, 1):
        incorporation_events = [
            {"type": item.get("type"), "date": item.get("date"), "description": item.get("description")}
            for item in client.iter_filing_history(company_number)
            if item.get("category") == "incorporation"
        ]
        if len(incorporation_events) > 1:
            offenders[company_number] = incorporation_events

        if i % 25 == 0:
            logger.info("  checked %d/%d", i, len(company_numbers))

    return offenders


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Find companies with >1 incorporation-category filing")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--company-status", default="active")
    parser.add_argument("--incorporated-from")
    parser.add_argument("--incorporated-to")
    args = parser.parse_args()

    filters = {"company_status": args.company_status}
    if args.incorporated_from:
        filters["incorporated_from"] = args.incorporated_from
    if args.incorporated_to:
        filters["incorporated_to"] = args.incorporated_to

    offenders = run(args.sample_size, filters)

    print(f"\nCompanies with >1 incorporation-category filing: {len(offenders)}")
    for company_number, events in offenders.items():
        print(f"\n{company_number}:")
        for e in events:
            print(f"  {e['date']} | {e['type']} | {e['description']}")


if __name__ == "__main__":
    main()
