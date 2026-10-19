"""Guards against the audit missing an exchange-listed company the store holds no bars for --
Inhibrx's INBX (Alpaca dropped the old holder's history when the label was reused) or Barnes
Group's B (Barrick's history now sits under the label) -- or flooding the list with names that
never had exchange bars to lose (mutual funds, OTC tickers, a listing seen once or twice), and
against counting SEC snapshots not yet published at ``as_of``.

Target API (asof.store.audit; implementation pending, step 6-c):

    CHECKS["I10"] = CheckSpec("I10", "exchange-listed ticker with no bars of its own", None,
                              <key pattern accepting "INBX:1739614">), right after I9
    I10_MIN_SNAPSHOTS = 3
    I10 rows: key "TICKER:CIK", then at least ticker, cik, first_listed, last_listed, snapshots
        for (ticker, cik) with ticker fully matching [A-Z]{1,5}, not 5 letters ending in X,
        listed on NASDAQ / NYSE / CBOE in >= 3 exchange snapshots, and no traded bar
        (volume > 0) between its first and last such snapshot that belongs to a master segment
        of that symbol carrying that cik. Reads <market>/symbols/sec_exchange_history.parquet
        (sec_ticker_history.EXCHANGE_HISTORY_PATH); missing file -> no rows, no error.

Fake data provenance (real files read 2026-10-19, see tests/ingest/test_sec_exchange_history.py):
- INBX under 1739614 on Nasdaq in 15 snapshots 2021-07-23 .. 2024-02-04, no INBX bar in the
  store before 2024-05-29 (real). B under 9984 (Barnes) on NYSE 2021-07-23 .. 2024-12-07; after
  6-c rule R every B bar of those years sits in a segment carrying Barrick's 756894 (real
  shape, see tests/ingest/test_security_master_takeovers.py).
- Fixture snapshots are three dates inside the fixture calendar (constructed); exchange values
  are written already upper-cased, as build_exchange_history stores them.
- ASSUMPTION: an exchange snapshot is knowable from 00:00 New York of its snapshot_date (as
  C2's assets snapshots); one dated after ``as_of`` does not count.
- ASSUMPTION: "between first and last listed snapshot" includes both snapshot days.

Real-data check: ``asof market audit`` (manual) lists INBX:1739614 and B:9984 under I10 after the
6-c rebuild; its row count is read once by a human before anything is gated on it.
"""

import re
from collections.abc import Iterator, Sequence
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from asof.ingest import sec_ticker_history as sth
from asof.store import audit as audit_mod
from asof.store.audit import CHECKS, CheckResult, run_audit
from asof.store.market import market_root
from tests.store.test_audit import (
    DAYS,
    LATE,
    SID_D1,
    SID_D2,
    SID_LAC,
    SID_R,
    SID_X1,
    Seg,
    World,
    base_segs,
    flat,
    keys,
    rows,
    series,
)
from tests.store.test_market import known

SNAP_1, SNAP_2, SNAP_3 = date(2020, 1, 3), date(2020, 1, 17), date(2020, 2, 3)
THREE = [SNAP_1, SNAP_2, SNAP_3]
Listing = tuple[str, int, str, Sequence[date]]  # ticker, cik, exchange, snapshot dates


def write_exchange_history(root: Path, listings: Sequence[Listing]) -> Path:
    table = pa.Table.from_pylist(
        [
            {"snapshot_date": day, "ticker": ticker, "cik": cik, "exchange": exchange}
            for ticker, cik, exchange, days in listings
            for day in days
        ],
        schema=sth.EXCHANGE_HISTORY_SCHEMA,
    )
    path = root / sth.EXCHANGE_HISTORY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


def audit_i10(
    con: duckdb.DuckDBPyConnection,
    world: World,
    listings: Sequence[Listing] | None,
    *,
    as_of: datetime = LATE,
) -> CheckResult:
    world.write()
    if listings is not None:
        write_exchange_history(world.root, listings)
    [result] = run_audit(con, world.root, as_of=as_of, liquid_ids=world.liquid_ids, checks=["I10"])
    return result


def by_key(result: CheckResult) -> dict[str, dict[str, Any]]:
    return {r["key"]: r for r in rows(result)}


@pytest.fixture
def con() -> Iterator[duckdb.DuckDBPyConnection]:
    connection = duckdb.connect()
    yield connection
    connection.close()


@pytest.fixture
def world(tmp_path: Path) -> World:
    return World(market_root(tmp_path))


def test_i10_is_registered_after_i9_as_information() -> None:
    assert "I10" in CHECKS
    spec = CHECKS["I10"]
    assert (spec.id, spec.title, spec.gate) == (
        "I10",
        "exchange-listed ticker with no bars of its own",
        None,
    )
    order = list(CHECKS)
    assert order.index("I10") == order.index("I9") + 1
    assert re.fullmatch(spec.key_pattern, "INBX:1739614")
    assert re.fullmatch(spec.key_pattern, "B:9984")
    assert re.fullmatch(spec.key_pattern, "INBX") is None
    assert re.fullmatch(spec.key_pattern, "INBX:2024-05-29") is None
    assert audit_mod.I10_MIN_SNAPSHOTS == 3


def barnes_world(world: World) -> None:
    """B's bars in the fixture window belong to a segment carrying Barrick's CIK."""
    world.segs = [*base_segs(), Seg(SID_X1, "B", cik=756894)]
    world.bars += series("B", DAYS, 17.0, 10_000)


def test_i10_lists_listed_companies_without_their_own_bars(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    barnes_world(world)
    listings: list[Listing] = [
        ("INBX", 1739614, "NASDAQ", THREE),  # no bars at all
        ("B", 9984, "NYSE", THREE),  # bars under B, but in Barrick's segment
        ("B", 756894, "NYSE", THREE),  # Barrick under B: its own bars -> clean
        ("AAA", 1001, "NYSE", THREE),  # a normal listed company with its bars -> clean
        ("CBX", 8001, "CBOE", THREE),  # CBOE counts like the other two
        ("ABCDE", 8101, "NASDAQ", THREE),  # five letters, not ending in X
    ]

    result = audit_i10(con, world, listings)

    assert keys(result) == ["ABCDE:8101", "B:9984", "CBX:8001", "INBX:1739614"]
    names = result.rows.schema.names
    assert names[:3] == ["key", "in_liquid", "excepted"]
    assert {"ticker", "cik", "first_listed", "last_listed", "snapshots"} <= set(names)
    inbx = by_key(result)["INBX:1739614"]
    assert (inbx["ticker"], inbx["cik"]) == ("INBX", 1739614)
    assert (inbx["first_listed"], inbx["last_listed"], inbx["snapshots"]) == (SNAP_1, SNAP_3, 3)
    assert result.fails("liquid") is False and result.fails("all") is False  # information


@pytest.mark.parametrize(
    "listing",
    [
        ("ABCDX", 5555, "NASDAQ", THREE),  # mutual-fund shape
        ("TWO", 6001, "NASDAQ", [SNAP_1, SNAP_2]),  # two snapshots only
        ("OTCY", 7001, "OTC", THREE),  # never on an exchange
        ("MIXD", 7101, "NASDAQ", [SNAP_1, SNAP_2]),  # two exchange sightings ...
        ("BRK.B", 1067983, "NYSE", THREE),  # not [A-Z]{1,5}
        ("ABCDEF", 7201, "NYSE", THREE),  # six letters
        ("BLNK", 7301, "", THREE),  # blank exchange (real: 5,848 rows)
    ],
    ids=["five_letters_ending_x", "two_snapshots", "otc_only", "mixed", "dotted", "six", "blank"],
)
def test_i10_ignores_names_that_are_not_exchange_listings_seen_three_times(
    con: duckdb.DuckDBPyConnection, world: World, listing: Listing
) -> None:
    listings: list[Listing] = [listing]
    if listing[0] == "MIXD":
        listings.append(("MIXD", 7101, "OTC", [SNAP_3]))  # ... plus one OTC sighting

    assert rows(audit_i10(con, world, listings)) == []


@pytest.mark.parametrize(
    ("symbol", "bar_days", "untraded", "reported"),
    [
        ("FLAT", DAYS, True, True),  # only volume-0 placeholders
        ("OUTS", [d for d in DAYS if d > SNAP_3], False, True),  # traded only after the last
        ("FRST", [SNAP_1], False, False),  # one traded bar on the first snapshot's day
        ("LAST", [SNAP_3], False, False),  # one traded bar on the last snapshot's day
    ],
    ids=["placeholders_only", "after_last_snapshot", "on_first_snapshot", "on_last_snapshot"],
)
def test_i10_counts_traded_bars_inside_the_listed_span_only(
    con: duckdb.DuckDBPyConnection,
    world: World,
    symbol: str,
    bar_days: Sequence[date],
    untraded: bool,
    reported: bool,
) -> None:
    world.segs = [*base_segs(), Seg(SID_D1, symbol, cik=9001)]
    world.bars += (
        [flat(symbol, d, 5.0) for d in bar_days] if untraded else series(symbol, bar_days, 5.0, 900)
    )

    result = audit_i10(con, world, [(symbol, 9001, "NYSE", THREE)])

    assert (f"{symbol}:9001" in keys(result)) is reported


def test_i10_bars_in_a_segment_without_a_cik_do_not_count(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    # A bars_only segment (cik None) is not "that cik"; the listing is reported.
    world.segs = [*base_segs(), Seg(SID_D2, "NOCK", cik=None)]
    world.bars += series("NOCK", DAYS, 5.0, 900)

    assert keys(audit_i10(con, world, [("NOCK", 9101, "NASDAQ", THREE)])) == ["NOCK:9101"]


def test_i10_without_an_exchange_history_file_is_empty(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    result = audit_i10(con, world, None)

    assert rows(result) == []
    assert result.rows.schema.names[:3] == ["key", "in_liquid", "excepted"]


def test_i10_snapshot_published_after_as_of_does_not_count(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    # Leakage: the third snapshot is on disk, but a reader the evening before it was published
    # has seen INBX listed twice, which is not enough; from 00:00 New York of its date, three.
    listings: list[Listing] = [("INBX", 1739614, "NASDAQ", THREE)]
    eve = known(SNAP_3) - timedelta(microseconds=1)

    assert rows(audit_i10(con, world, listings, as_of=eve)) == []
    assert keys(audit_i10(con, world, listings, as_of=known(SNAP_3))) == ["INBX:1739614"]


def test_i10_segment_not_yet_knowable_does_not_own_the_bars(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    # Leakage: the segment that gives B's bars to Barnes (9984) is only knowable after the
    # audit's as_of; until then the bars belong to no segment of that cik and B:9984 is listed.
    known_late = known(date(2020, 2, 5))
    world.segs = [
        *base_segs(),
        Seg(SID_LAC, "B", valid_to=date(2020, 1, 2), end_available_at=known_late, cik=756894),
        Seg(SID_R, "B", valid_from=date(2020, 1, 2), available_at=known_late, cik=9984),
    ]
    world.bars += series("B", DAYS, 17.0, 10_000)
    listings: list[Listing] = [("B", 9984, "NYSE", THREE)]

    early = audit_i10(con, world, listings, as_of=known(date(2020, 2, 4)))
    assert keys(early) == ["B:9984"]
    assert rows(audit_i10(con, world, listings, as_of=LATE)) == []
