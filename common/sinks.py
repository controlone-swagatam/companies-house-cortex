"""
Output sinks for raw filing events.

KafkaSink pushes onto the Kafka spine (the real target). StdoutSink and
JSONLFileSink exist for local development and testing without a broker —
useful given the sandbox this was built in has no route to a live Kafka
cluster or the Companies House API itself.
"""
import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger("ch_receptor.sinks")


class EventSink(ABC):
    @abstractmethod
    def send(self, event: dict[str, Any]) -> None:
        ...

    def close(self) -> None:
        pass


class StdoutSink(EventSink):
    def send(self, event: dict[str, Any]) -> None:
        print(json.dumps(event, default=str))


class JSONLFileSink(EventSink):
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")

    def send(self, event: dict[str, Any]) -> None:
        self._fh.write(json.dumps(event, default=str) + "\n")

    def close(self) -> None:
        self._fh.close()


class KafkaSink(EventSink):
    """
    Requires `kafka-python` (pip install kafka-python) and a reachable
    broker. Kept as a thin wrapper — the receptor's job ends at handing the
    event to the spine.
    """

    def __init__(self, bootstrap_servers: str, topic: str):
        try:
            from kafka import KafkaProducer
        except ImportError as exc:
            raise RuntimeError(
                "kafka-python is required for KafkaSink: pip install kafka-python"
            ) from exc

        self.topic = topic
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
        )

    def send(self, event: dict[str, Any]) -> None:
        # Keyed by company_number so all events for one company land on the
        # same partition, preserving per-entity ordering for the sequence builder.
        self.producer.send(self.topic, key=event.get("company_number"), value=event)

    def close(self) -> None:
        self.producer.flush()
        self.producer.close()


class MultiplexSink(EventSink):
    """
    Routes each event to a sub-sink keyed by event["period"] (falling back to
    "live" when period is None, i.e. inference events). Lazily creates
    sub-sinks via `sink_factory(period_key) -> EventSink` on first use.

    This is how training's period_1/period_2 split becomes physically
    separate Kafka topics or files, without the receptor knowing anything
    about what those periods mean semantically — it only routes on the
    ingestion-run metadata already in the envelope.
    """

    def __init__(self, sink_factory):
        self._sink_factory = sink_factory
        self._sinks: dict[str, EventSink] = {}

    def send(self, event: dict[str, Any]) -> None:
        key = event.get("period") or "live"
        if key not in self._sinks:
            self._sinks[key] = self._sink_factory(key)
        self._sinks[key].send(event)

    def close(self) -> None:
        for sink in self._sinks.values():
            sink.close()


def build_sink(config) -> EventSink:
    if config.sink == "kafka":
        if getattr(config, "route_by_period", False):
            return MultiplexSink(
                lambda period_key: KafkaSink(
                    config.kafka_bootstrap_servers, f"{config.kafka_topic}.{period_key}"
                )
            )
        return KafkaSink(config.kafka_bootstrap_servers, config.kafka_topic)

    if config.sink == "jsonl":
        if getattr(config, "route_by_period", False):
            base, ext = os.path.splitext(config.jsonl_output_path)
            return MultiplexSink(
                lambda period_key: JSONLFileSink(f"{base}.{period_key}{ext}")
            )
        return JSONLFileSink(config.jsonl_output_path)

    if config.sink == "stdout":
        return StdoutSink()

    raise ValueError(f"Unknown sink type: {config.sink!r} (expected kafka|jsonl|stdout)")
