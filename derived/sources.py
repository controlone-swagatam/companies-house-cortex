"""
Reading raw envelope events off the spine (or a JSONL dev file) to feed the
derived layer. Kept separate from common/sinks.py since that module is
about writing to the spine, not reading from it — the receptor never reads
its own output, only the derived layer does.
"""
import json
import logging
from typing import Iterator

logger = logging.getLogger("ch_pipeline.derived.sources")


def read_jsonl(path: str) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_kafka(bootstrap_servers: str, topic: str, group_id: str) -> Iterator[dict]:
    """
    Consume raw envelope events from Kafka. Requires kafka-python
    (pip install kafka-python). Runs until manually stopped — intended for
    a long-lived derived-layer consumer process, not a one-shot batch job.
    """
    try:
        from kafka import KafkaConsumer
    except ImportError as exc:
        raise RuntimeError(
            "kafka-python is required for read_kafka: pip install kafka-python"
        ) from exc

    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
    )
    for message in consumer:
        yield message.value
