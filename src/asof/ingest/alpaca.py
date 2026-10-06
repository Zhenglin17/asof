"""Alpaca REST client: asset list and historical bars.

Only the two endpoints the backfill needs. Paging, retry and the free-tier rate limit live
here so the orchestration code can treat ``iter_bars`` as a plain stream of bars. History is
always requested unadjusted (``adjustment=raw``): adjusted prices are rewritten by splits that
happen *after* the bar, which would let a replay see the future.
"""

import json
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime
from typing import Any, Literal

import httpx
from pydantic import AliasChoices, BaseModel, Field, field_validator

from asof.ingest.http import Fetch, FetchResult, make_fetch

__all__ = [
    "AlpacaClient",
    "AlpacaError",
    "Asset",
    "Bar",
    "Fetch",
    "FetchResult",
    "Timeframe",
]

Timeframe = Literal["1Min", "1Day"]
AssetStatus = Literal["active", "inactive"]

PAGE_LIMIT = 10_000
RETRYABLE = frozenset({429, 500, 502, 503, 504})
MAX_BACKOFF = 120.0


class AlpacaError(RuntimeError):
    """A request failed for good: non-retryable status, or retries exhausted."""


class Asset(BaseModel):
    symbol: str
    name: str
    exchange: str
    status: str
    tradable: bool
    # Alpaca serialises the class as "class"; "asset_class" is kept for our own files.
    asset_class: str = Field(validation_alias=AliasChoices("asset_class", "class"))


class Bar(BaseModel):
    symbol: str
    t: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    trade_count: int
    vwap: float

    @field_validator("t")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("bar timestamp must be timezone-aware")
        return value.astimezone(UTC)


def _rfc3339(value: datetime, name: str) -> str:
    if value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _backoff(retry_after: str | None, attempt: int) -> float:
    """Seconds to wait: the server's Retry-After when it is a number, else exponential."""
    delay = float(2**attempt)
    if retry_after:
        try:
            delay = float(retry_after)
        except ValueError:  # HTTP-date form; not worth parsing, fall back
            pass
    return min(delay, MAX_BACKOFF)


class AlpacaClient:
    TRADING_URL = "https://paper-api.alpaca.markets"
    DATA_URL = "https://data.alpaca.markets"

    def __init__(
        self,
        key: str,
        secret: str,
        *,
        fetch: Fetch | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        max_requests_per_minute: int = 190,
        max_retries: int = 5,
    ) -> None:
        self.headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
        self._fetch = fetch or make_fetch(self.headers)
        self._sleep = sleep
        self._clock = clock
        self._max_per_minute = max_requests_per_minute
        self._max_retries = max_retries
        self._recent: deque[float] = deque()

    # -- transport -------------------------------------------------------------------------

    def _throttle(self) -> None:
        """Sliding window: never more than ``max_requests_per_minute`` starts in any 60 s."""
        now = self._clock()
        while self._recent and now - self._recent[0] >= 60.0:
            self._recent.popleft()
        if len(self._recent) >= self._max_per_minute:
            self._sleep(max(0.0, 60.0 - (now - self._recent[0])))
            now = self._clock()
            self._recent.popleft()  # the slot we waited for, even if the clock is coarse
            while self._recent and now - self._recent[0] >= 60.0:
                self._recent.popleft()
        self._recent.append(now)

    def _request(self, url: str, params: dict[str, str]) -> Any:
        for attempt in range(self._max_retries + 1):
            self._throttle()
            try:
                result = self._fetch(url, params)
            except (httpx.TransportError, OSError) as error:
                # Connection reset, timeout, DNS hiccup: retry like a 5xx.
                if attempt == self._max_retries:
                    raise AlpacaError(
                        f"gave up after {attempt + 1} attempts on {url}: {error}"
                    ) from error
                self._sleep(_backoff(None, attempt))
                continue
            if 200 <= result.status < 300:
                return json.loads(result.body) if result.body else None
            if result.status not in RETRYABLE:
                raise AlpacaError(f"{result.status} from {url}: {result.body[:200]!r}")
            if attempt == self._max_retries:
                raise AlpacaError(f"gave up after {attempt + 1} attempts on {url}")
            self._sleep(_backoff(result.headers.get("Retry-After"), attempt))
        raise AssertionError("unreachable")

    # -- endpoints -------------------------------------------------------------------------

    def assets(self, status: AssetStatus, asset_class: str = "us_equity") -> list[Asset]:
        rows = self._request(
            f"{self.TRADING_URL}/v2/assets", {"status": status, "asset_class": asset_class}
        )
        return [Asset.model_validate(row) for row in rows or []]

    def iter_bars(
        self, symbols: Sequence[str], timeframe: Timeframe, start: datetime, end: datetime
    ) -> Iterator[Bar]:
        """All bars for ``symbols`` in ``[start, end]``, across every page. Unordered."""
        params = {
            "symbols": ",".join(symbols),
            "timeframe": timeframe,
            "start": _rfc3339(start, "start"),
            "end": _rfc3339(end, "end"),
            "limit": str(PAGE_LIMIT),
            "adjustment": "raw",
            "feed": "sip",
        }
        return self._iter_pages(params)

    def _iter_pages(self, params: dict[str, str]) -> Iterator[Bar]:
        url = f"{self.DATA_URL}/v2/stocks/bars"
        while True:
            page = self._request(url, params) or {}
            for symbol, rows in (page.get("bars") or {}).items():
                for row in rows:
                    yield Bar(
                        symbol=symbol,
                        t=row["t"],
                        open=row["o"],
                        high=row["h"],
                        low=row["l"],
                        close=row["c"],
                        volume=row["v"],
                        trade_count=row["n"],
                        vwap=row["vw"],
                    )
            token = page.get("next_page_token")
            if not token:
                return
            params = {**params, "page_token": token}
