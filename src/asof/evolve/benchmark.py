"""Benchmarks a strategy has to beat over the same dates: buy-and-hold of index funds and a
mechanical basket, the largest stocks by recent dollar volume held equal weight for a month.

The basket picks its members point in time (``liquid_securities`` at 00:00 New York of the
rebalance day sees bars through the previous session). Values are results, not decisions: the
curve is marked with bars, splits and dividends as known at the window's last close, the same
reading as the replay report's benchmark. Dividends are reinvested at the ex-date close; a
split multiplies the shares before the ex-date open. No trading costs, so the benchmark is a
little stronger than a real one.
"""

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb

from asof.evolve.metrics import TRADING_DAYS
from asof.evolve.protocol import EvaluationProtocol
from asof.store.market import (
    visible_bars,
    visible_dividends,
    visible_master,
    visible_sessions,
    visible_splits,
)
from asof.store.universe import liquid_securities

ET = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class Rebalance:
    session: date
    members: tuple[tuple[str, str], ...]  # (security_id, symbol), in liquid_securities order


@dataclass(frozen=True)
class BenchmarkCurve:
    label: str
    sessions: tuple[date, ...]
    equity: tuple[float, ...]  # value at each close; 1.0 was invested at sessions[0]'s open
    rebalances: tuple[Rebalance, ...] = ()

    @property
    def total_return(self) -> float:
        return self.equity[-1] - 1

    @property
    def annualized_return(self) -> float:
        return (1 + self.total_return) ** (TRADING_DAYS / len(self.sessions)) - 1

    @property
    def max_drawdown(self) -> float:
        peak, worst = 1.0, 0.0
        for value in self.equity:
            peak = max(peak, value)
            worst = max(worst, (peak - value) / peak)
        return worst


@dataclass(frozen=True)
class _Bar:
    symbol: str
    open: float
    close: float


def _close_of(day: date) -> datetime:
    return datetime.combine(day, time(20), tzinfo=ET)


def _sessions(con: duckdb.DuckDBPyConnection, root: Path, start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError(f"end {end} is before start {start}")
    days = [d for d in visible_sessions(con, root, "1Day", _close_of(end)) if start <= d <= end]
    if not days:
        raise ValueError(f"no sessions between {start} and {end}")
    return days


def _bars(
    con: duckdb.DuckDBPyConnection,
    root: Path,
    symbols: Sequence[str],
    start: date,
    end: date,
) -> dict[str, dict[date, _Bar]]:
    """Bars by security_id then session, attributed through the master known at ``end``."""
    rows = visible_bars(
        con,
        root,
        "1Day",
        _close_of(end),
        symbols=sorted(set(symbols)),
        start=datetime.combine(start, time(0), tzinfo=ET),
        end=_close_of(end),
    ).to_pylist()
    out: dict[str, dict[date, _Bar]] = defaultdict(dict)
    for r in rows:
        out[r["security_id"]][r["session_date"]] = _Bar(r["symbol"], r["open"], r["close"])
    return out


def _actions(
    con: duckdb.DuckDBPyConnection, root: Path, symbols: Sequence[str], end: date
) -> tuple[dict[str, list[tuple[date, float]]], dict[str, list[tuple[date, float]]]]:
    """Split ratios and cash dividends per symbol known at ``end``'s close. A row repeated in
    every column is one event (real: QQQ 2022-09-19 is stored twice)."""
    as_of = _close_of(end)
    splits: dict[str, list[tuple[date, float]]] = defaultdict(list)
    seen: set[tuple[str, date, float, float]] = set()
    for s in visible_splits(con, root, as_of, sorted(set(symbols))).to_pylist():
        key = (s["symbol"], s["ex_date"], s["old_rate"], s["new_rate"])
        if key not in seen:
            seen.add(key)
            splits[s["symbol"]].append((s["ex_date"], s["new_rate"] / s["old_rate"]))
    dividends: dict[str, list[tuple[date, float]]] = defaultdict(list)
    for d in visible_dividends(con, root, as_of, sorted(set(symbols))).to_pylist():
        dividends[d["symbol"]].append((d["ex_date"], float(d["rate"])))
    return splits, dividends


def _in(events: Iterable[tuple[date, float]], after: date | None, day: date) -> list[float]:
    """Values dated in (after, day]; with no earlier session, ``day`` alone."""
    return [v for d, v in events if (d == day if after is None else after < d <= day)]


class _Book:
    """Shares per security plus cash; a security without a bar keeps its last close."""

    def __init__(
        self,
        bars: dict[str, dict[date, _Bar]],
        splits: dict[str, list[tuple[date, float]]],
        dividends: dict[str, list[tuple[date, float]]],
    ) -> None:
        self.bars, self.splits, self.dividends = bars, splits, dividends
        self.shares: dict[str, float] = {}
        self.last_close: dict[str, float] = {}
        self.symbol: dict[str, str] = {}
        self.cash = 0.0
        self.owed: dict[str, float] = {}  # dividend cash per security, paid at today's close

    def _symbol(self, sid: str, day: date) -> str | None:
        bar = self.bars.get(sid, {}).get(day)
        if bar is not None:
            self.symbol[sid] = bar.symbol
        return self.symbol.get(sid)

    def open(self, day: date, prev: date | None) -> None:
        for sid in self.shares:
            symbol = self._symbol(sid, day)
            ratio = math.prod(_in(self.splits.get(symbol or "", []), prev, day))
            self.shares[sid] *= ratio
            if sid in self.last_close:
                self.last_close[sid] /= ratio
        # Only shares held before the ex-date open earn the payout; shares bought at this open
        # do not, and shares sold at it still do.
        self.owed = {}
        for sid, n in self.shares.items():
            rate = sum(_in(self.dividends.get(self.symbol.get(sid, ""), []), prev, day))
            if rate:
                self.owed[sid] = n * rate

    def _price(self, sid: str, day: date, field: str) -> float | None:
        bar = self.bars.get(sid, {}).get(day)
        return getattr(bar, field) if bar is not None else self.last_close.get(sid)

    def liquidate(self, day: date) -> float:
        value = self.cash + sum(
            n * (self._price(s, day, "open") or 0.0) for s, n in self.shares.items()
        )
        self.shares, self.cash = {}, 0.0
        return value

    def buy(self, members: Sequence[str], value: float, day: date) -> None:
        if not members:
            self.cash += value
            return
        slice_ = value / len(members)
        for sid in members:
            price = self._price(sid, day, "open")
            self._symbol(sid, day)
            if price:
                self.shares[sid] = self.shares.get(sid, 0.0) + slice_ / price
            else:
                self.cash += slice_  # no price to buy at: the slice waits as cash

    def close(self, day: date) -> float:
        for sid in list(self.shares):
            bar = self.bars.get(sid, {}).get(day)
            if bar is not None:
                self.last_close[sid] = bar.close
        for sid, cash in self.owed.items():
            price = self.last_close.get(sid)
            if sid in self.shares and price:
                self.shares[sid] += cash / price  # reinvested in the same security
            else:
                self.cash += cash  # sold at today's open: the payout waits as cash
        self.owed = {}
        return self.cash + sum(n * self.last_close.get(s, 0.0) for s, n in self.shares.items())


def _curve(
    label: str,
    days: list[date],
    plan: dict[date, list[str]],
    bars: dict[str, dict[date, _Bar]],
    splits: dict[str, list[tuple[date, float]]],
    dividends: dict[str, list[tuple[date, float]]],
    rebalances: tuple[Rebalance, ...] = (),
) -> BenchmarkCurve:
    book = _Book(bars, splits, dividends)
    equity: list[float] = []
    prev: date | None = None
    for day in days:
        book.open(day, prev)
        if day in plan:
            value = book.liquidate(day) if prev is not None else 1.0
            book.buy(plan[day], value, day)
        equity.append(book.close(day))
        prev = day
    return BenchmarkCurve(label, tuple(days), tuple(equity), rebalances)


def buy_and_hold(
    con: duckdb.DuckDBPyConnection, root: Path, symbol: str, start: date, end: date
) -> BenchmarkCurve:
    days = _sessions(con, root, start, end)
    bars = _bars(con, root, [symbol], start, end)
    first = [sid for sid, by_day in bars.items() if days[0] in by_day]
    if not first:
        raise ValueError(f"{symbol} has no bar on {days[0]}")
    splits, dividends = _actions(con, root, [symbol], end)
    label = f"{symbol} buy-and-hold (dividends reinvested)"
    return _curve(label, days, {days[0]: first[:1]}, bars, splits, dividends)


def basket(
    con: duckdb.DuckDBPyConnection,
    root: Path,
    start: date,
    end: date,
    *,
    size: int,
    lookback_sessions: int,
) -> BenchmarkCurve:
    days = _sessions(con, root, start, end)
    firsts = [d for i, d in enumerate(days) if i == 0 or d.month != days[i - 1].month]
    rebalances: list[Rebalance] = []
    for day in firsts:
        tier = liquid_securities(
            con,
            root,
            datetime.combine(day, time(0), tzinfo=ET),
            min_adv_usd=0,
            lookback_sessions=lookback_sessions,
            classes=("common",),
        ).to_pylist()[:size]
        rebalances.append(Rebalance(day, tuple((r["security_id"], r["symbol"]) for r in tier)))
    members = {sid for r in rebalances for sid, _ in r.members}
    # Every symbol a member used in the window, so a rename keeps its bars and actions.
    master = visible_master(con, root, _close_of(end)).to_pylist()
    symbols = sorted({m["symbol"] for m in master if m["security_id"] in members})
    bars = _bars(con, root, symbols, start, end) if symbols else {}
    splits, dividends = _actions(con, root, symbols, end)
    plan = {r.session: [sid for sid, _ in r.members] for r in rebalances}
    label = f"Top-{size} dollar-volume stocks, monthly, equal weight"
    return _curve(label, days, plan, bars, splits, dividends, tuple(rebalances))


def benchmarks(
    con: duckdb.DuckDBPyConnection,
    root: Path,
    protocol: EvaluationProtocol,
    start: date,
    end: date,
) -> list[BenchmarkCurve]:
    spec = protocol.benchmarks
    curves = [buy_and_hold(con, root, s, start, end) for s in spec.symbols]
    curves.append(
        basket(
            con,
            root,
            start,
            end,
            size=spec.basket.size,
            lookback_sessions=spec.basket.lookback_sessions,
        )
    )
    return curves
