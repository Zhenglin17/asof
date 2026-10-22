"""Guards against the replay's in-memory bar cache drifting from the one read path: a cache that
resolves identity once (as of the last day, so a renamed security's old bars appear under its
new id from the start, or a successor's bars appear before the rename was knowable), that hands
out bars before their 20:00 New York availability, or that re-reads Parquet every step.

API under test (asof.replay):

    BarCache(con, root, *, symbols, start, end)   loads the daily bars once
    BarCache.bars_at(as_of) -> pa.Table            must equal visible_bars(con, root, "1Day",
                                                   as_of, symbols=symbols, start=start, end=end)

The worlds are the security-master scenarios of tests/ingest/master_scenarios.py (their
provenance notes apply), built with build_security_master and written to Parquet the way the
backfill writes them. Instants: every instant at which a scenario's master changes (segment
known, end known, CIK known), one microsecond before each, and the first and last bar closes.
"""

from collections.abc import Iterator
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import pyarrow as pa
import pytest

from asof.ingest import security_master as sm
from asof.replay import BarCache
from asof.store.market import market_root, visible_bars
from tests.ingest.fakes import daily
from tests.ingest.master_scenarios import SCENARIOS, Inputs
from tests.store.test_market import store_daily

ET = ZoneInfo("America/New_York")
MICRO = timedelta(microseconds=1)
# Worlds with renames, relabelled copies and late-filed ends (see master_scenarios.py).
RENAME_SCENARIOS = ["world", "h2", "r_bbu_bbuc", "l_arnc", "r_disca_wbd", "f1", "g2"]
MAX_INSTANTS = 40


@pytest.fixture
def con() -> Iterator[duckdb.DuckDBPyConnection]:
    connection = duckdb.connect()
    yield connection
    connection.close()


def close_of(day: date) -> datetime:
    return datetime.combine(day, time(20), tzinfo=ET).astimezone(UTC)


def write_world(root: Path, inputs: Inputs) -> tuple[pa.Table, list[str], list[date]]:
    master, conflicts = sm.build_security_master(
        bars=inputs.bars,
        sec_history=inputs.sec_history,
        name_changes=inputs.name_changes,
        sec_history_start=inputs.sec_history_start,
    )
    sm.write_master(master, conflicts, root)
    rows = inputs.bars.to_pylist()
    store_daily(
        root,
        [
            daily(
                r["symbol"],
                r["session_date"],
                open=r["open"],
                high=r["high"],
                low=r["low"],
                close=r["close"],
                volume=r["volume"],
            )
            for r in rows
        ],
    )
    symbols = sorted({r["symbol"] for r in rows})
    days = sorted({r["session_date"] for r in rows})
    return master, symbols, days


def instants(master: pa.Table, days: list[date]) -> list[datetime]:
    found = {
        t
        for name in ("available_at", "end_available_at", "cik_available_at")
        for t in master.column(name).to_pylist()
        if t is not None
    }
    found |= {t - MICRO for t in found}
    found |= {close_of(days[0]), close_of(days[-1]), close_of(days[len(days) // 2])}
    ordered = sorted(found)
    if len(ordered) > MAX_INSTANTS:
        step = len(ordered) / MAX_INSTANTS
        ordered = [ordered[int(i * step)] for i in range(MAX_INSTANTS)] + [ordered[-1]]
    return ordered


def window(days: list[date]) -> tuple[datetime, datetime]:
    start = datetime.combine(days[0], time(0), tzinfo=ET).astimezone(UTC)
    end = datetime.combine(days[-1] + timedelta(days=1), time(0), tzinfo=ET).astimezone(UTC)
    return start, end


# =============================================================================================
# normal path + leakage: equivalence with visible_bars at every instant
# =============================================================================================


@pytest.mark.parametrize("scenario", RENAME_SCENARIOS)
def test_cache_equals_visible_bars_at_every_instant(
    tmp_path: Path, con: duckdb.DuckDBPyConnection, scenario: str
) -> None:
    # Identity is resolved as of each step: a segment unknown at as_of owns nothing yet, an end
    # not yet knowable is still open. A cache resolved once with the final master fails here.
    root = market_root(tmp_path)
    master, symbols, days = write_world(root, SCENARIOS[scenario]())
    start, end = window(days)
    cache = BarCache(con, root, symbols=symbols, start=start, end=end)
    for as_of in instants(master, days):
        expected = visible_bars(con, root, "1Day", as_of, symbols=symbols, start=start, end=end)
        got = cache.bars_at(as_of)
        assert got.schema.names == expected.schema.names, as_of
        assert got.to_pylist() == expected.to_pylist(), (scenario, as_of)


@pytest.mark.parametrize("scenario", ["world", "r_bbu_bbuc"])
def test_cache_never_hands_out_a_bar_before_it_is_final(
    tmp_path: Path, con: duckdb.DuckDBPyConnection, scenario: str
) -> None:
    root = market_root(tmp_path)
    _, symbols, days = write_world(root, SCENARIOS[scenario]())
    start, end = window(days)
    cache = BarCache(con, root, symbols=symbols, start=start, end=end)
    for day in (days[0], days[len(days) // 2], days[-1]):
        before = close_of(day) - MICRO
        seen = cache.bars_at(before).to_pylist()
        assert all(r["available_at"] <= before for r in seen)
        assert all(r["session_date"] < day for r in seen)


# =============================================================================================
# boundaries
# =============================================================================================


def test_cache_before_the_first_bar_is_empty(
    tmp_path: Path, con: duckdb.DuckDBPyConnection
) -> None:
    root = market_root(tmp_path)
    _, symbols, days = write_world(root, SCENARIOS["h2"]())
    start, end = window(days)
    cache = BarCache(con, root, symbols=symbols, start=start, end=end)
    assert cache.bars_at(close_of(days[0]) - MICRO).num_rows == 0
    assert cache.bars_at(close_of(days[0])).num_rows >= 1


def test_cache_respects_its_symbols_and_range(
    tmp_path: Path, con: duckdb.DuckDBPyConnection
) -> None:
    root = market_root(tmp_path)
    _, symbols, days = write_world(root, SCENARIOS["r_bbu_bbuc"]())
    start, end = window(days[10:20])
    one = symbols[:1]
    cache = BarCache(con, root, symbols=one, start=start, end=end)
    as_of = close_of(days[-1])
    got = cache.bars_at(as_of)
    assert (
        got.to_pylist()
        == visible_bars(con, root, "1Day", as_of, symbols=one, start=start, end=end).to_pylist()
    )
    assert set(got.column("symbol").to_pylist()) <= set(one)


def test_cache_reads_parquet_once(tmp_path: Path, con: duckdb.DuckDBPyConnection) -> None:
    # Loaded once: removing the bar files after construction does not change what it serves.
    root = market_root(tmp_path)
    master, symbols, days = write_world(root, SCENARIOS["h2"]())
    start, end = window(days)
    steps = instants(master, days)
    expected = [
        visible_bars(con, root, "1Day", t, symbols=symbols, start=start, end=end).to_pylist()
        for t in steps
    ]
    cache = BarCache(con, root, symbols=symbols, start=start, end=end)
    for path in (root / "bars_1d").rglob("*.parquet"):
        path.unlink()
    assert [cache.bars_at(t).to_pylist() for t in steps] == expected


@pytest.mark.parametrize("scenario", ["world", "r_bbu_bbuc"])
def test_a_later_start_equals_visible_bars_with_that_start(
    tmp_path: Path, con: duckdb.DuckDBPyConnection, scenario: str
) -> None:
    # The replay asks for one evening at a time; narrowing must not change what is visible.
    root = market_root(tmp_path)
    master, symbols, days = write_world(root, SCENARIOS[scenario]())
    start, end = window(days)
    cache = BarCache(con, root, symbols=symbols, start=start, end=end)
    for as_of in instants(master, days):
        day = as_of.astimezone(ET).date()
        since = datetime.combine(day, time(0), tzinfo=ET)
        expected = visible_bars(con, root, "1Day", as_of, symbols=symbols, start=since, end=end)
        assert cache.bars_at(as_of, start=since).to_pylist() == expected.to_pylist(), as_of
