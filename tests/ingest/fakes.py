"""Scripted doubles for ingest tests: canned HTTP replies, a fake clock, a fake bar client.

Nothing here touches the network. Every HTTP call the code under test makes goes through the
injectable ``fetch`` callable, which these helpers replace with a list of pre-written answers.
"""

import gzip
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import UTC, date, datetime
from typing import Any, NamedTuple
from zoneinfo import ZoneInfo

from asof.ingest.alpaca import Asset, Bar, FetchResult

# A regular daily bar from Alpaca: the 2026-10-02 session, stamped midnight New York (EDT).
T_DAY = datetime(2026, 10, 2, 4, 0, tzinfo=UTC)
# When that daily bar becomes visible: 20:00 New York, after the post-market session, because
# Alpaca's daily volume includes extended-hours trades.
AVAILABLE_DAY = datetime(2026, 10, 3, 0, 0, tzinfo=UTC)
# The first regular-hours minute bar of the same session: 09:30 New York.
T_MIN = datetime(2026, 10, 2, 13, 30, tzinfo=UTC)
FETCHED_AT = datetime(2026, 10, 5, 1, 0, tzinfo=UTC)


class ScriptedFetch:
    """Returns pre-written results in order and records every (url, params) it was asked for."""

    def __init__(self, results: Iterable[FetchResult] = ()) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, dict[str, str]]] = []

    def __call__(self, url: str, params: Mapping[str, str]) -> FetchResult:
        self.calls.append((url, dict(params)))
        if not self.results:
            raise AssertionError(f"unexpected request #{len(self.calls)}: {url} {dict(params)}")
        return self.results.pop(0)

    @property
    def urls(self) -> list[str]:
        return [url for url, _ in self.calls]

    @property
    def params(self) -> list[dict[str, str]]:
        return [params for _, params in self.calls]


def reply(
    status: int, payload: Any = None, headers: Mapping[str, str] | None = None
) -> FetchResult:
    """A FetchResult whose body is ``payload`` serialised as JSON (``None`` → empty body)."""
    body = b"" if payload is None else json.dumps(payload).encode()
    return FetchResult(status=status, headers=dict(headers or {}), body=body)


def raw_reply(status: int, body: bytes, headers: Mapping[str, str] | None = None) -> FetchResult:
    return FetchResult(status=status, headers=dict(headers or {}), body=body)


def gzipped(payload: Any) -> bytes:
    return gzip.compress(json.dumps(payload).encode())


class FakeTime:
    """A monotonic clock that only moves when told to, or when something sleeps on it."""

    def __init__(self, start: float = 1_000.0) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_asset(
    symbol: str,
    *,
    exchange: str = "NASDAQ",
    status: str = "active",
    tradable: bool = True,
    name: str | None = None,
    asset_class: str = "us_equity",
) -> Asset:
    return Asset(
        symbol=symbol,
        name=name or f"{symbol} Inc.",
        exchange=exchange,
        status=status,
        tradable=tradable,
        asset_class=asset_class,
    )


def make_bar(
    symbol: str = "AAPL",
    t: datetime = T_DAY,
    *,
    open: float = 1.0,
    high: float = 2.0,
    low: float = 0.5,
    close: float = 1.5,
    volume: int = 100,
    trade_count: int = 7,
    vwap: float = 1.2,
) -> Bar:
    return Bar(
        symbol=symbol,
        t=t,
        open=open,
        high=high,
        low=low,
        close=close,
        volume=volume,
        trade_count=trade_count,
        vwap=vwap,
    )


def daily(symbol: str, day: date, **fields: Any) -> Bar:
    """A daily bar stamped the way Alpaca does it: midnight New York of the session."""
    t = datetime(day.year, day.month, day.day, tzinfo=ZoneInfo("America/New_York")).astimezone(UTC)
    return make_bar(symbol, t, **fields)


def bar_json(t: str, **overrides: Any) -> dict[str, Any]:
    """One bar as Alpaca serialises it."""
    bar: dict[str, Any] = {
        "t": t,
        "o": 1.0,
        "h": 2.0,
        "l": 0.5,
        "c": 1.5,
        "v": 100,
        "n": 7,
        "vw": 1.2,
    }
    bar.update(overrides)
    return bar


class BarsCall(NamedTuple):
    symbols: tuple[str, ...]
    timeframe: str
    start: datetime
    end: datetime


class FakeBarClient:
    """Stands in for AlpacaClient in backfill tests; serves bars from memory and logs each call.

    With ``respect_range`` the fake only returns bars inside ``[start, end]`` like the real API
    (Alpaca's ``end`` is inclusive).
    Turn it off to simulate a server that hands back bars outside the requested window.
    """

    def __init__(self, bars: Iterable[Bar] = (), *, respect_range: bool = True) -> None:
        self.bars = list(bars)
        self.respect_range = respect_range
        self.calls: list[BarsCall] = []

    def iter_bars(
        self, symbols: Sequence[str], timeframe: str, start: datetime, end: datetime
    ) -> Iterator[Bar]:
        self.calls.append(BarsCall(tuple(symbols), timeframe, start, end))
        wanted = set(symbols)
        selected = [
            bar
            for bar in self.bars
            if bar.symbol in wanted and (not self.respect_range or start <= bar.t <= end)
        ]
        return iter(selected)
