# Companies House Pipeline — Training / Inference Split

Restructured from the single-script receptor into separate training and
inference paths that share one REST client.

```
ch_pipeline/
  common/
    ch_client.py    HTTP concerns only: auth, rate limiting, retries, pagination
    envelope.py     RawFilingEvent — adds run_mode + period as ingestion metadata
    sinks.py        Kafka / stdout / JSONL, + MultiplexSink for period routing
    config.py       Shared config, env-based
  training/
    batch_ingest.py Batch: many companies, FULL history per company,
                     split locally by date into period_1 (train) / period_2 (test)
  inference/
    api_receptor.py On-demand: one company at a time, real-time REST call,
                     no period tagging (routes to "live")
```

## Why the split

Training and inference hit the same CH REST API but with different access
patterns:

- **Training** is a bulk, offline job: iterate a company list, pull each
  company's entire filing history (CH's filing-history endpoint has no
  server-side date filter), then partition locally by `payload.date` into
  two physically separate streams. Period_1 and period_2 land in different
  Kafka topics (`<topic>.period_1`, `<topic>.period_2`) or files, so the
  sequence builder / trainer never has to re-derive the split or risk
  leaking test data into training.
- **Inference** is on-demand: given a company number at prediction time,
  fetch current history and push it onto a `.live` stream immediately. No
  batching, no period logic — `InferenceReceptor` is designed to be
  imported and called per-request by whatever serves predictions, not run
  as a standalone batch job.

Both still obey the pure-ingestion principle: neither module extracts
category/subtype into NTST's schema. `run_mode` and `period` are ingestion
bookkeeping (which run, which split) — not interpretation of what a filing
means.

## Training usage

```bash
export CH_API_KEY=your_api_key_here

python -m training.batch_ingest \
  --company-numbers-file companies.txt \
  --period-1-start 2019-01-01 --period-1-end 2022-12-31 \
  --period-2-start 2023-01-01 --period-2-end 2024-12-31 \
  --sink kafka
```

Dates are runtime arguments, deliberately not hardcoded — set them per
training run. Events outside both ranges are counted and dropped (logged),
not silently discarded. If `period_1_end >= period_2_start` (overlapping or
misordered ranges) the script warns but proceeds, since that's sometimes
intentional and sometimes a leak — worth a second look either way.

With `--sink kafka`, output topics are `<KAFKA_TOPIC>.period_1` and
`<KAFKA_TOPIC>.period_2`. With `--sink jsonl`, output files are
`<path>.period_1.jsonl` and `<path>.period_2.jsonl`.

## Inference usage

As a library (typical usage — called by the prediction service):

```python
from inference.api_receptor import InferenceReceptor

receptor = InferenceReceptor()  # reads CH_API_KEY, sink config from env
events = receptor.fetch(company_number="00445790")
# events -> list of envelope dicts, also pushed to the "live" sink/topic
```

Ad-hoc CLI:

```bash
python -m inference.api_receptor --company-number 00445790 --sink stdout
```

## What's tested vs. not

Same caveat as before: this sandbox has no network route to
`api.companieshouse.gov.uk` or a live Kafka broker. Verified with mocked CH
responses:

- ✅ `classify_period` correctly buckets by date, returns `None` outside both ranges
- ✅ Training run splits events into separate period_1/period_2 files with correct counts and metadata
- ✅ Inference run fetches all items for a company, tags them `run_mode=inference, period=None`, routes to the "live" stream (no period multiplexing)
- ✅ All modules import/compile cleanly

Not yet verified against the live API or a real broker — same
recommendation as before: run against 1–2 real company numbers with
`--sink stdout` first before pointing at Kafka.

## Open item

`InferenceReceptor.fetch()` currently pulls **full** filing history per
call, same as training. For a live prediction path this is probably more
than you need per request (a company with a long history re-fetches
everything on every prediction). Worth deciding: cache per company with a
short TTL, or add a "since last known transaction_id" cursor once the
downstream consumer (sequence builder) can tell the receptor what it
already has.
