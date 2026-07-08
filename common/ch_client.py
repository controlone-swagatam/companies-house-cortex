"""
Thin REST client for the Companies House Public Data API.

Deliberately does nothing beyond HTTP concerns: authentication, rate limiting,
retries, and pagination. No filing-type interpretation, no schema mapping —
that logic belongs in a downstream derived layer, not here.
"""
import logging
import time
from collections import deque
from typing import Iterator

import requests

from common.config import ReceptorConfig

logger = logging.getLogger("ch_receptor.client")


class RateLimiter:
    """
    Sliding-window limiter matching CH's published limit: N requests per
    window_seconds, shared across all endpoints on a single API key.
    """

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._timestamps: deque = deque()

    def acquire(self) -> None:
        now = time.monotonic()
        while self._timestamps and now - self._timestamps[0] >= self.window_seconds:
            self._timestamps.popleft()

        if len(self._timestamps) >= self.max_requests:
            sleep_for = self.window_seconds - (now - self._timestamps[0])
            if sleep_for > 0:
                logger.info("Rate limit reached, sleeping %.1fs", sleep_for)
                time.sleep(sleep_for)

        self._timestamps.append(time.monotonic())


class CompaniesHouseClient:
    def __init__(self, config: ReceptorConfig):
        self.config = config
        self.session = requests.Session()
        self.session.auth = (config.ch_api_key, "")  # basic auth, blank password
        self.rate_limiter = RateLimiter(
            config.ch_rate_limit_requests, config.ch_rate_limit_window_seconds
        )

    def _get(self, path: str, params: dict | None = None) -> dict:
        url = f"{self.config.ch_base_url}{path}"
        attempt = 0

        while True:
            self.rate_limiter.acquire()
            attempt += 1
            try:
                resp = self.session.get(
                    url, params=params, timeout=self.config.request_timeout_seconds
                )
            except requests.RequestException as exc:
                if attempt > self.config.max_retries:
                    raise
                backoff = min(2 ** attempt, 60)
                logger.warning(
                    "Request error (%s), retry %d/%d in %.1fs",
                    exc, attempt, self.config.max_retries, backoff,
                )
                time.sleep(backoff)
                continue

            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", 5))
                logger.warning("429 from CH API, backing off %.1fs", retry_after)
                time.sleep(retry_after)
                continue

            if resp.status_code == 404:
                return {}

            if resp.status_code >= 500 and attempt <= self.config.max_retries:
                backoff = min(2 ** attempt, 60)
                logger.warning(
                    "Server error %d, retry %d/%d in %.1fs",
                    resp.status_code, attempt, self.config.max_retries, backoff,
                )
                time.sleep(backoff)
                continue

            resp.raise_for_status()
            return resp.json()

    def get_filing_history_page(
        self, company_number: str, start_index: int, items_per_page: int
    ) -> dict:
        """Single page of raw filingHistoryList. No transformation."""
        return self._get(
            f"/company/{company_number}/filing-history",
            params={"start_index": start_index, "items_per_page": items_per_page},
        )

    def iter_filing_history(self, company_number: str) -> Iterator[dict]:
        """
        Yields raw filingHistoryItem dicts for a company, handling pagination.
        Stops when start_index has walked past total_count, or on an empty page.
        """
        start_index = 0
        page_size = self.config.items_per_page

        while True:
            page = self.get_filing_history_page(company_number, start_index, page_size)
            items = page.get("items", [])
            if not items:
                return

            for item in items:
                yield item

            total_count = page.get("total_count", 0)
            start_index += len(items)
            if start_index >= total_count:
                return
