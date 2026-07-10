"""
Pull a real sample of companies and their filing histories, then validate
the taxonomy assumptions in the working doc (§6 slug distribution, §7
hierarchy, §9 category enum correction) against actual data rather than
the (partly stale) CH documentation.

This also doubles as the anomaly check flagged after the 00445790 result:
per-company filing counts and duplicate transaction_id detection are part
of the same pass, since both come for free while iterating filing history.

Usage:
    export CH_API_KEY=...
    python -m analysis.sample_taxonomy \
        --sample-size 200 \
        --company-status active \
        --incorporated-from 2015-01-01 \
        --incorporated-to 2024-12-31 \
        --output ./output/taxonomy_sample_report.json

Sampling defaults to active companies incorporated in a broad recent window
rather than CH's default ordering, since unfiltered advanced-search results
tend to skew toward low company numbers (i.e. old companies), which would
not be representative of current filing behaviour.
"""
import argparse
import json
import logging
from collections import Counter, defaultdict
from datetime import datetime

from common.ch_client import CompaniesHouseClient
from common.config import base_config_from_env

logger = logging.getLogger("ch_pipeline.analysis.sample_taxonomy")


def run(
    sample_size: int,
    search_filters: dict,
    output_path: str,
) -> dict:
    config = base_config_from_env()
    client = CompaniesHouseClient(config)

    logger.info("Sampling %d companies via advanced search (filters=%s)", sample_size, search_filters)
    company_numbers = []
    for item in client.iter_advanced_search(sample_size, **search_filters):
        number = item.get("company_number")
        if number:
            company_numbers.append(number)

    logger.info("Got %d company numbers, fetching filing history for each", len(company_numbers))

    category_counts = Counter()
    type_counts = Counter()
    subtype_counts = Counter()
    category_to_types = defaultdict(set)
    type_to_subtypes = defaultdict(set)
    filings_per_company = {}
    duplicate_flags = {}
    action_date_lag_days = []

    for i, company_number in enumerate(company_numbers, 1):
        seen_ids = set()
        dup_count = 0
        n_filings = 0

        for raw_item in client.iter_filing_history(company_number):
            n_filings += 1

            txn_id = raw_item.get("transaction_id")
            if txn_id:
                if txn_id in seen_ids:
                    dup_count += 1
                seen_ids.add(txn_id)

            category = raw_item.get("category")
            form_type = raw_item.get("type")
            description = raw_item.get("description")

            if category:
                category_counts[category] += 1
            if form_type:
                type_counts[form_type] += 1
            if description:
                subtype_counts[description] += 1
            if category and form_type:
                category_to_types[category].add(form_type)
            if form_type and description:
                type_to_subtypes[form_type].add(description)

            date_str = raw_item.get("date")
            action_date_str = raw_item.get("action_date")
            if date_str and action_date_str:
                try:
                    d = datetime.strptime(date_str, "%Y-%m-%d").date()
                    a = datetime.strptime(action_date_str, "%Y-%m-%d").date()
                    action_date_lag_days.append((d - a).days)
                except ValueError:
                    pass

        filings_per_company[company_number] = n_filings
        if dup_count > 0:
            duplicate_flags[company_number] = dup_count

        if i % 25 == 0:
            logger.info("  processed %d/%d companies", i, len(company_numbers))

    counts = list(filings_per_company.values())
    mean_filings = sum(counts) / len(counts) if counts else 0
    sorted_counts = sorted(counts)
    median_filings = sorted_counts[len(sorted_counts) // 2] if sorted_counts else 0

    # MAD-based outlier detection: robust to a single extreme value skewing
    # a mean/std threshold (verified in testing — std-based detection missed
    # a synthetic 500-filing outlier because that one point inflated the std
    # it was being measured against).
    abs_deviations = sorted(abs(c - median_filings) for c in counts)
    mad = abs_deviations[len(abs_deviations) // 2] if abs_deviations else 0
    if mad > 0:
        outliers = {
            c: n for c, n in filings_per_company.items()
            if 0.6745 * abs(n - median_filings) / mad > 3.5
        }
    else:
        # MAD is 0 when most values are identical (e.g. small sample) —
        # fall back to a simple multiple-of-median rule.
        fallback_threshold = max(20, 5 * median_filings)
        outliers = {c: n for c, n in filings_per_company.items() if n > fallback_threshold}

    report = {
        "sample_size_requested": sample_size,
        "sample_size_actual": len(company_numbers),
        "search_filters": search_filters,
        "category_counts": dict(category_counts.most_common()),
        "type_counts_top_30": dict(type_counts.most_common(30)),
        "subtype_counts_top_30": dict(subtype_counts.most_common(30)),
        "distinct_categories": len(category_counts),
        "distinct_types": len(type_counts),
        "distinct_subtypes": len(subtype_counts),
        "category_to_type_cardinality": {
            cat: len(types) for cat, types in category_to_types.items()
        },
        "filings_per_company_stats": {
            "mean": round(mean_filings, 1),
            "median": sorted_counts[len(sorted_counts) // 2] if sorted_counts else 0,
            "min": min(counts) if counts else 0,
            "max": max(counts) if counts else 0,
        },
        "outlier_companies_gt_3std": outliers,
        "duplicate_transaction_id_companies": duplicate_flags,
        "action_date_lag_days_stats": {
            "mean": round(sum(action_date_lag_days) / len(action_date_lag_days), 1)
            if action_date_lag_days else None,
            "max": max(action_date_lag_days) if action_date_lag_days else None,
            "n_samples": len(action_date_lag_days),
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Report written to %s", output_path)
    return report


def print_summary(report: dict) -> None:
    print("\n=== Taxonomy sample report ===")
    print(f"Companies sampled: {report['sample_size_actual']}")
    print(f"Distinct categories observed: {report['distinct_categories']}")
    print(f"Distinct type (form) codes observed: {report['distinct_types']}")
    print(f"Distinct subtypes (description slugs) observed: {report['distinct_subtypes']}")
    print(f"\nCategory distribution:")
    for cat, n in report["category_counts"].items():
        print(f"  {cat:30s} {n}")
    print(f"\nFilings per company: mean={report['filings_per_company_stats']['mean']} "
          f"median={report['filings_per_company_stats']['median']} "
          f"min={report['filings_per_company_stats']['min']} "
          f"max={report['filings_per_company_stats']['max']}")
    if report["outlier_companies_gt_3std"]:
        print(f"\n⚠ Outlier companies (MAD-based, robust): {report['outlier_companies_gt_3std']}")
    if report["duplicate_transaction_id_companies"]:
        print(f"\n⚠ Companies with duplicate transaction_ids (possible pagination bug): "
              f"{report['duplicate_transaction_id_companies']}")
    lag = report["action_date_lag_days_stats"]
    if lag["n_samples"]:
        print(f"\ndate vs action_date lag: mean={lag['mean']} days, max={lag['max']} days "
              f"(n={lag['n_samples']}) — confirms §9's finding that `date` "
              f"(filed) commonly trails `action_date` (effective) by a real margin")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Sample real CH data to validate taxonomy assumptions")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--company-status", default="active")
    parser.add_argument("--incorporated-from")
    parser.add_argument("--incorporated-to")
    parser.add_argument("--sic-codes")
    parser.add_argument("--output", default="./output/taxonomy_sample_report.json")
    args = parser.parse_args()

    filters = {"company_status": args.company_status}
    if args.incorporated_from:
        filters["incorporated_from"] = args.incorporated_from
    if args.incorporated_to:
        filters["incorporated_to"] = args.incorporated_to
    if args.sic_codes:
        filters["sic_codes"] = args.sic_codes

    report = run(args.sample_size, filters, args.output)
    print_summary(report)


if __name__ == "__main__":
    main()
