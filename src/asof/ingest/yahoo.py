"""Yahoo Finance chart API: a second source for daily bars that Alpaca no longer serves.

Alpaca files a security's history under the symbol's current holder. When a symbol is reused
the old holder's history is rewritten (2026-10-16: PARA 2022-02-17 .. 2025-08-06, Paramount
Global's class B, had become Banzai's chain). Yahoo keeps such a lineage under the newest symbol
of the company (SKYD), and on the days both vendors have, open and close agree to the cent
while Yahoo's volume runs about 5% lower (Alpaca's daily volume includes extended hours).

This client is for one-off recoveries registered in ``configs/bar_imports.yaml``, not for the
daily feed: the endpoint is unofficial and may change without notice. Its prices and volumes
are **split-adjusted** (measured 2026-10-16: NVDA 2024-06-07 closes at 120.89 there, 1208.88
raw at Alpaca), so only a lineage with no split after the start of the imported range can be
taken from it; ``import_bars`` refuses the rest. Every bar it returns is
shaped like an Alpaca daily bar (``t`` = midnight New York of the session, prices to four
decimals, ``vwap`` = typical price since Yahoo has none, ``trade_count`` None), and the import
applies the store's own availability rule afterwards.
"""

import json
import time
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from asof.ingest.alpaca import Bar
from asof.ingest.http import Fetch, make_fetch

__all__ = ["YahooClient", "YahooError"]

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
# Yahoo answers nothing without a User-Agent and 429 to a Chrome-like one, while the bare
# token below is served (measured 2026-10-16: the two strings, same URL, one minute apart).
USER_AGENT = "Mozilla/5.0"
RETRYABLE = frozenset({429, 500, 502, 503, 504})
MAX_RETRIES = 3
BACKOFF_BASE = 5.0
MAX_BACKOFF = 60.0
PRICE_DECIMALS = 4
ET = ZoneInfo("America/New_York")


class YahooError(RuntimeError):
    """The request failed for good, or the body is not a daily series."""


def _backoff(retry_after: str | None, attempt: int) -> float:
    """Seconds to wait: the server's Retry-After when it is a number, else 5, 15, 45 (Yahoo
    rate-limits bursts for a while; 2026-10-16 four tries 1-4 s apart all got 429)."""
    delay = BACKOFF_BASE * 3**attempt
    if retry_after:
        try:
            delay = float(retry_after)
        except ValueError:
            pass
    return min(delay, MAX_BACKOFF)


def _midnight_ny(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, tzinfo=ET)


class YahooClient:
    def __init__(
        self,
        *,
        fetch: Fetch | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._fetch = fetch or make_fetch({"User-Agent": USER_AGENT}, follow_redirects=True)
        self._sleep = sleep

    def daily_bars(self, symbol: str, start: date, end: date) -> tuple[list[Bar], bytes]:
        """Daily bars of ``symbol`` for the sessions in ``[start, end]`` (New York dates),
        sorted by time, plus the raw response body for the archive."""
        if start > end:
            raise ValueError(f"start {start} is after end {end}")
        params = {
            "period1": str(int(_midnight_ny(start).timestamp())),
            "period2": str(int(_midnight_ny(end + timedelta(days=1)).timestamp())),
            "interval": "1d",
            "includePrePost": "false",
        }
        body = self._request(CHART_URL.format(symbol=symbol), params)
        return _parse(symbol, body, start, end), body

    def _request(self, url: str, params: dict[str, str]) -> bytes:
        for attempt in range(MAX_RETRIES + 1):
            result = self._fetch(url, params)
            if 200 <= result.status < 300:
                return result.body
            if result.status not in RETRYABLE:
                raise YahooError(f"{result.status} from {url}: {result.body[:200]!r}")
            if attempt == MAX_RETRIES:
                raise YahooError(f"gave up after {attempt + 1} attempts on {url}: {result.status}")
            self._sleep(_backoff(result.headers.get("Retry-After"), attempt))
        raise AssertionError("unreachable")


def _parse(symbol: str, body: bytes, start: date, end: date) -> list[Bar]:
    try:
        data: Any = json.loads(body)
    except ValueError as error:
        raise YahooError(f"{symbol}: body is not JSON: {body[:80]!r}") from error
    chart = data.get("chart") if isinstance(data, dict) else None
    if not isinstance(chart, dict):
        raise YahooError(f"{symbol}: body has no chart: {body[:80]!r}")
    error = chart.get("error")
    if error:
        raise YahooError(f"{symbol}: {error.get('code')}: {error.get('description')}")
    results = chart.get("result") or []
    if not results:
        return []
    result = results[0]
    granularity = (result.get("meta") or {}).get("dataGranularity")
    if granularity != "1d":
        raise YahooError(
            f"{symbol}: Yahoo returned a {granularity!r} series instead of daily bars "
            "(range=max is downgraded to monthly; ask with period1/period2)"
        )
    timestamps = result.get("timestamp") or []
    quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    opens, highs, lows = quote.get("open") or [], quote.get("high") or [], quote.get("low") or []
    closes, volumes = quote.get("close") or [], quote.get("volume") or []

    bars: dict[date, Bar] = {}  # a repeated session keeps the last row seen
    for i, stamp in enumerate(timestamps):
        session = datetime.fromtimestamp(stamp, UTC).astimezone(ET).date()
        close = closes[i] if i < len(closes) else None
        if close is None or not (start <= session <= end):
            continue  # a null close is Yahoo's holiday padding, not a session
        o = opens[i] if i < len(opens) else None
        h = highs[i] if i < len(highs) else None
        lo = lows[i] if i < len(lows) else None
        v = volumes[i] if i < len(volumes) else None
        if o is None or h is None or lo is None or v is None:
            missing = [
                n for n, x in (("open", o), ("high", h), ("low", lo), ("volume", v)) if x is None
            ]
            raise YahooError(
                f"{symbol} {session}: close present but {', '.join(missing)} null; "
                "not a bar the store can keep"
            )
        high = round(float(h), PRICE_DECIMALS)
        low = round(float(lo), PRICE_DECIMALS)
        close_r = round(float(close), PRICE_DECIMALS)
        bars[session] = Bar(
            symbol=symbol,
            t=_midnight_ny(session),
            open=round(float(o), PRICE_DECIMALS),
            high=high,
            low=low,
            close=close_r,
            volume=int(v),
            trade_count=None,
            vwap=round((high + low + close_r) / 3, PRICE_DECIMALS),
        )
    return [bars[day] for day in sorted(bars)]
