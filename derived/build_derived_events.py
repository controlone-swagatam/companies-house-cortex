"""
Build derived events from the raw spine.

Reads raw envelope events (JSONL file for dev, or Kafka topic in
production) and writes derived events — same events, plus corrected
category and the 3-level hierarchy fields, ready for the sequence builder.

Usage (dev, JSONL in and out):
    python3 -m derived.build_derived_events \
        --source jsonl --input ./output/companies_house_raw_events.period_1.jsonl \
        --sink jsonl --output ./output/derived_events.period_1.jsonl

Usage (production, Kafka in and out):
    python3 -m derived.build_derived_events \
        --source kafka --kafka-input-topic companies-house.filing-history.raw.period_1 \
        --sink kafka --kafka-output-topic companies-house.filing-history.derived.period_1
"""
import argparse
import logging
from collections import Counter

from common.config import base_config_from_env, with_overrides
from common.sinks import build_sink
from derived.derived_event import DerivedEvent
from derived.sources import read_jsonl, read_kafka

logger = logging.getLogger("ch_pipeline.derived.build_derived_events")


def run(source_iter, sink) -> dict:
    rule_application_counts = Counter()
    total = 0

    try:
        for raw_event in source_iter:
            derived = DerivedEvent.from_raw_event(raw_event)
            sink.send(derived.to_dict())
            total += 1
            if derived.category_rule_applied:
                rule_application_counts[derived.category_rule_applied] += 1
    finally:
        sink.close()

    logger.info(
        "Derived %d events. Rule applications: %s",
        total, dict(rule_application_counts) or "none",
    )
    return {"total": total, "rule_applications": dict(rule_application_counts)}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Build derived events from raw CH filing events")
    parser.add_argument("--source", choices=["jsonl", "kafka"], required=True)
    parser.add_argument("--input", help="Path to raw JSONL file (required if --source jsonl)")
    parser.add_argument("--kafka-input-topic", help="Kafka topic to consume raw events from")
    parser.add_argument("--kafka-consumer-group", default="ch-pipeline-derived-layer")
    parser.add_argument("--sink", choices=["jsonl", "kafka", "stdout"], required=True)
    parser.add_argument("--output", help="Path to write derived JSONL (required if --sink jsonl)")
    args = parser.parse_args()

    config = base_config_from_env()

    if args.source == "jsonl":
        if not args.input:
            raise SystemExit("--input is required when --source jsonl")
        source_iter = read_jsonl(args.input)
    else:
        if not args.kafka_input_topic:
            raise SystemExit("--kafka-input-topic is required when --source kafka")
        source_iter = read_kafka(
            config.kafka_bootstrap_servers, args.kafka_input_topic, args.kafka_consumer_group
        )

    sink_overrides = {"sink": args.sink, "route_by_period": True}
    if args.sink == "jsonl":
        if not args.output:
            raise SystemExit("--output is required when --sink jsonl")
        sink_overrides["jsonl_output_path"] = args.output
    config = with_overrides(config, **sink_overrides)
    sink = build_sink(config)

    run(source_iter, sink)


if __name__ == "__main__":
    main()
