"""
Shared configuration for the Companies House pipeline.

Training and inference both build on this; each adds its own CLI surface
(see training/config_training.py and inference/config_inference.py).
"""
import os
from dataclasses import dataclass, replace
from typing import Optional


@dataclass(frozen=True)
class ReceptorConfig:
    # Companies House REST API
    ch_api_key: str
    ch_base_url: str = "https://api.companieshouse.gov.uk"

    # Rate limiting — CH allows 600 requests / 5 min (2/sec) per key,
    # shared across all endpoints on that key.
    ch_rate_limit_requests: int = 600
    ch_rate_limit_window_seconds: int = 300

    # Kafka spine
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic: str = "companies-house.filing-history.raw"

    # Sink selection: "kafka" | "stdout" | "jsonl"
    sink: str = "stdout"
    jsonl_output_path: str = "./output/companies_house_raw_events.jsonl"

    # When True, events are routed to sub-topics/files keyed by period
    # (period_1 / period_2 / live) instead of one shared stream.
    route_by_period: bool = False

    # HTTP behaviour
    request_timeout_seconds: float = 15.0
    max_retries: int = 5
    items_per_page: int = 100  # CH max page size for filing-history list


def base_config_from_env() -> ReceptorConfig:
    api_key = os.environ.get("CH_API_KEY")
    if not api_key:
        raise RuntimeError(
            "CH_API_KEY environment variable is required. "
            "Register an API key at https://developer.company-information.service.gov.uk "
            "and export it before running the pipeline."
        )

    return ReceptorConfig(
        ch_api_key=api_key,
        kafka_bootstrap_servers=os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        kafka_topic=os.environ.get("KAFKA_TOPIC", "companies-house.filing-history.raw"),
        sink=os.environ.get("RECEPTOR_SINK", "stdout"),
        jsonl_output_path=os.environ.get(
            "JSONL_OUTPUT_PATH", "./output/companies_house_raw_events.jsonl"
        ),
    )


def with_overrides(config: ReceptorConfig, **overrides) -> ReceptorConfig:
    return replace(config, **overrides)
