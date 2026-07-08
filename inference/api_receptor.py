"""
Inference-time receptor.

Unlike training's batch_ingest (many companies, full history, split by
period), inference is on-demand: given a company number at prediction time,
fetch its current filing history via the same CH REST client and hand it to
the sink immediately. No period tagging — these are "live" events by
definition (envelope.period stays None, MultiplexSink routes them to the
"live" stream/topic).

Designed to be imported and called per-request by whatever serves
predictions (e.g. `fetch_for_inference(company_number)` returns the raw
items so a caller can feed them straight into the sequence builder), with a
CLI wrapper for manual/ad-hoc use.

Usage (library):
    from inference.api_receptor import InferenceReceptor
    receptor = InferenceReceptor()
    events = receptor.fetch(company_number="00445790")  # -> list[dict]

Usage (CLI, ad-hoc):
    export CH_API_KEY=...
    python -m inference.api_receptor --company-number 00445790 --sink stdout
"""
import argparse
import logging

from common.ch_client import CompaniesHouseClient
from common.config import base_config_from_env, with_overrides
from common.envelope import RawFilingEvent
from common.sinks import build_sink, EventSink

logger = logging.getLogger("ch_pipeline.inference.api_receptor")


class InferenceReceptor:
    """
    Thin on-demand wrapper. One instance can be reused across many
    fetch() calls — the underlying client keeps its own rate limiter state,
    which matters if this is called frequently from a live service.
    """

    def __init__(self, sink_override: str | None = None):
        config = base_config_from_env()
        if sink_override:
            config = with_overrides(config, sink=sink_override)
        self.config = config
        self.client = CompaniesHouseClient(config)
        self._sink: EventSink | None = None

    def _get_sink(self) -> EventSink:
        if self._sink is None:
            self._sink = build_sink(self.config)
        return self._sink

    def fetch(self, company_number: str, emit_to_sink: bool = True) -> list[dict]:
        """
        Fetch current filing history for one company. Returns the raw
        envelope dicts. If emit_to_sink is True (default), also pushes each
        event onto the configured sink (spine) as it's ingested — matters
        for keeping the Kafka spine as the single source of truth even for
        on-demand inference lookups, so training-side consumers and
        inference-side consumers see the same event shape.
        """
        events = []
        sink = self._get_sink() if emit_to_sink else None

        for raw_item in self.client.iter_filing_history(company_number):
            event = RawFilingEvent.from_filing_item(
                company_number, raw_item, run_mode="inference", period=None
            )
            d = event.to_dict()
            events.append(d)
            if sink is not None:
                sink.send(d)

        logger.info("Inference fetch: %s -> %d filing events", company_number, len(events))
        return events

    def close(self) -> None:
        if self._sink is not None:
            self._sink.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Companies House inference-time receptor (ad-hoc CLI)")
    parser.add_argument("--company-number", required=True)
    parser.add_argument("--sink", choices=["kafka", "stdout", "jsonl"])
    args = parser.parse_args()

    receptor = InferenceReceptor(sink_override=args.sink)
    try:
        receptor.fetch(args.company_number)
    finally:
        receptor.close()


if __name__ == "__main__":
    main()
