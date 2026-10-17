"""Guards against the audit missing the NOBL shape (a symbol Alpaca still lists whose bars
stopped long ago because the backfill stopped asking for it) and against the second bar source
confusing the checks that already exist: an ext row that overrides an Alpaca row is one bar, not
a duplicate (P1) and not two rows in the coverage count (C1).

Target API (asof.store.audit; implementation pending):

    CHECKS["C2"] = CheckSpec("C2", "listed asset with stale bars", "all", <I1's key pattern>)
    STALE_SESSIONS = 10
    C1 rows: ... + ext_bars (int) + ext_overrides (int)
    C2 rows: key = symbol, symbol, name, exchange, last_bar (date), sessions_behind (int)
    C2 notes: no_bars (int), snapshot_date (ISO string, None when no snapshot is usable)

Fake data provenance:
- NOBL shape (measured 2026-10-16): active in Alpaca, SEC window closed 2020-07, store holds
  2020 bars only; 197 active symbols like it. The fixtures shorten a symbol's series instead.
- The listing comes from the newest Alpaca assets snapshot on or before ``as_of``
  (``root/assets/snapshot_date=YYYY-MM-DD/assets.parquet``, written by
  ``universe.save_assets_snapshot``; docs). Only status 'active', exchange not 'OTC' and a
  valid symbol count as listed (the universe's own rules; docs).
- "Sessions" are the distinct visible session dates of the store (the derived calendar every
  other check uses; docs). ``sessions_behind`` = index distance between the symbol's last visible
  session and the newest visible session; more than STALE_SESSIONS is reported.
- Ext rows: ``bars_1d_ext`` files in EXT_BAR_SCHEMA with available_at 20:00 New York of the
  session (docs: ingest.bar_imports).
- ASSUMPTION: an assets snapshot is knowable from 00:00 New York of its snapshot_date (the date
  is the download day); a snapshot dated after ``as_of`` is ignored. Real-data check: the
  audit's C2 note ``snapshot_date`` at ``--as-of`` 2026-10-01 must name a snapshot on or before
  that day.

Real-data checks: ``asof market audit`` after the ⑥-a backfill: C2 must list the NOBL-shaped
symbols before the backfill (about 197) and none (or only excepted ones) after; C1's ext_bars
must equal the number of imported PARA sessions (869), as must ext_overrides (Alpaca files a
Banzai row under PARA for every one of those days).
"""

import re
from collections.abc import Iterator, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

import duckdb
import pytest

from asof.ingest.alpaca import Asset
from asof.ingest.universe import save_assets_snapshot
from asof.store.audit import CHECKS, STALE_SESSIONS, AuditException, CheckResult, run_audit
from asof.store.market import market_root
from tests.ingest.fakes import make_asset
from tests.store.test_audit import (
    DAYS,
    HEAD,
    LATE,
    SID_AAA,
    World,
    drop,
    exc,
    find,
    rows,
    series,
)
from tests.store.test_market import avail
from tests.store.test_market_ext import ext_row, write_ext

LISTED = [
    make_asset("AAA", exchange="NYSE", name="Alpha Corp"),
    make_asset("BBB", exchange="NASDAQ", name="Beta Inc"),
    make_asset("CCC", exchange="NASDAQ", name="Gamma Ltd"),
]
SNAP_1 = date(2020, 1, 15)  # inside DAYS, before every stale cut below
SNAP_2 = date(2020, 3, 1)  # after DAYS[-1] (2020-02-06)


def listed_after(bars: Sequence[dict[str, Any]], symbol: str, keep: int) -> list[dict[str, Any]]:
    """``symbol``'s bars cut after its ``keep``-th session (NOBL shape)."""
    return drop(bars, symbol, DAYS[keep + 1 :])


def audit(
    con: duckdb.DuckDBPyConnection,
    world: World,
    *,
    as_of: datetime = LATE,
    checks: Sequence[str] = ("C2",),
    snapshots: Sequence[tuple[date, Sequence[Asset]]] = ((SNAP_1, LISTED),),
    ext: Sequence[dict[str, Any]] = (),
    exceptions: Sequence[AuditException] = (),
) -> dict[str, CheckResult]:
    """Write the world, then the assets snapshots and ext rows beside it, then run the audit."""
    world.write()
    for day, assets in snapshots:
        save_assets_snapshot(list(assets), world.root, day)
    if ext:
        write_ext(world.root, ext)
    results = run_audit(
        con,
        world.root,
        as_of=as_of,
        liquid_ids=world.liquid_ids,
        exceptions=exceptions,
        checks=list(checks),
    )
    return {r.spec.id: r for r in results}


def reported(result: CheckResult) -> list[tuple[str, date, int]]:
    return [(r["symbol"], r["last_bar"], r["sessions_behind"]) for r in rows(result)]


@pytest.fixture
def con() -> Iterator[duckdb.DuckDBPyConnection]:
    connection = duckdb.connect()
    yield connection
    connection.close()


@pytest.fixture
def world(tmp_path: Path) -> World:
    return World(market_root(tmp_path))


# =============================================================================================
# registry
# =============================================================================================


def test_c2_is_registered_right_after_c1_with_i1s_key_shape() -> None:
    assert "C2" in CHECKS
    spec = CHECKS["C2"]
    assert (spec.id, spec.title, spec.gate) == ("C2", "listed asset with stale bars", "all")
    assert spec.key_pattern == CHECKS["I1"].key_pattern
    assert re.fullmatch(spec.key_pattern, "NOBL") and re.fullmatch(spec.key_pattern, "BRK.B")
    assert re.fullmatch(spec.key_pattern, "NOBL:2020-01-07") is None
    order = list(CHECKS)
    assert order.index("C2") == order.index("C1") + 1
    assert STALE_SESSIONS == 10


def test_c2_runs_as_part_of_the_default_check_list(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    results = audit(con, world, checks=list(CHECKS))
    assert list(results)[-2:] == ["C1", "C2"]
    assert rows(results["C2"]) == []


# =============================================================================================
# C2: normal path
# =============================================================================================


def test_c2_reports_a_listed_symbol_whose_bars_stopped(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    world.bars = listed_after(world.bars, "AAA", 10)  # last bar DAYS[10], store ends DAYS[24]

    result = audit(con, world)["C2"]

    assert result.rows.schema.names == [
        *HEAD,
        "symbol",
        "name",
        "exchange",
        "last_bar",
        "sessions_behind",
    ]
    assert rows(result) == [
        {
            "key": "AAA",
            "in_liquid": True,
            "excepted": False,
            "symbol": "AAA",
            "name": "Alpha Corp",
            "exchange": "NYSE",
            "last_bar": DAYS[10],
            "sessions_behind": 14,
        }
    ]
    assert result.notes == {"no_bars": 0, "snapshot_date": SNAP_1.isoformat()}
    assert result.fails("all") is True and result.fails("liquid") is True


def test_c2_clean_world(con: duckdb.DuckDBPyConnection, world: World) -> None:
    result = audit(con, world)["C2"]

    assert rows(result) == []
    assert result.notes == {"no_bars": 0, "snapshot_date": SNAP_1.isoformat()}
    assert result.fails("all") is False


def test_c2_excepted_row_does_not_fail_the_gate(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    world.bars = listed_after(world.bars, "AAA", 10)

    result = audit(con, world, exceptions=[exc("C2", "AAA")])["C2"]

    assert [r["excepted"] for r in rows(result)] == [True]
    assert (result.liquid, result.total, result.excepted) == (0, 0, 1)
    assert result.fails("all") is False


def test_c2_counts_listed_symbols_with_no_bars_in_a_note(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    world.bars = listed_after(world.bars, "CCC", 5)
    world.liquid_ids = set()
    snapshot = [
        *LISTED,
        make_asset("DDD", exchange="NYSE"),  # listed, no bars: counted
        make_asset("GGG", exchange="ARCA"),  # listed, no bars: counted
        make_asset("EEE", exchange="NYSE", status="inactive"),  # not listed any more
        make_asset("FFF", exchange="OTC"),  # OTC is not in the universe
        make_asset("HHH_DELISTED", exchange="NYSE"),  # Alpaca junk entry
    ]

    result = audit(con, world, snapshots=[(SNAP_1, snapshot)])["C2"]

    assert reported(result) == [("CCC", DAYS[5], 19)]
    assert [r["in_liquid"] for r in rows(result)] == [False]
    assert result.notes == {"no_bars": 2, "snapshot_date": SNAP_1.isoformat()}


# =============================================================================================
# C2: boundaries
# =============================================================================================


@pytest.mark.parametrize(("keep", "behind", "listed"), [(14, 10, False), (13, 11, True)])
def test_c2_threshold_is_more_than_ten_sessions(
    con: duckdb.DuckDBPyConnection, world: World, keep: int, behind: int, listed: bool
) -> None:
    world.bars = listed_after(world.bars, "BBB", keep)

    result = audit(con, world)["C2"]

    assert reported(result) == ([("BBB", DAYS[keep], behind)] if listed else [])


def test_c2_counts_sessions_not_calendar_days(con: duckdb.DuckDBPyConnection, world: World) -> None:
    # DAYS[9] is 2020-01-15 (Wed) and DAYS[-1] 2020-02-06: 22 calendar days, 15 sessions
    world.bars = listed_after(world.bars, "AAA", 9)

    assert reported(audit(con, world)["C2"]) == [("AAA", DAYS[9], 15)]
    assert (DAYS[-1] - DAYS[9]).days == 22


def test_c2_inactive_otc_and_junk_symbols_are_not_reported_even_when_stale(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    world.bars = listed_after(world.bars, "AAA", 3)
    world.bars = listed_after(world.bars, "BBB", 3)
    world.bars = listed_after(world.bars, "CCC", 3)
    world.bars += series("DDD", DAYS, 50.0, 2_000)  # keeps the calendar running; not listed
    world.liquid_ids = set()
    snapshot = [
        make_asset("AAA", exchange="NYSE", status="inactive"),
        make_asset("BBB", exchange="OTC"),
        make_asset("CCC", exchange="NASDAQ"),
    ]

    result = audit(con, world, snapshots=[(SNAP_1, snapshot)])["C2"]

    assert reported(result) == [("CCC", DAYS[3], 21)]


def test_c2_only_the_newest_snapshot_defines_the_listing(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    # AAA is in the older snapshot only: it is no longer listed and is not reported
    world.bars = listed_after(world.bars, "AAA", 3)
    world.bars = listed_after(world.bars, "CCC", 3)
    world.liquid_ids = set()
    older = [(SNAP_1, LISTED)]
    newer = [(date(2020, 1, 20), [make_asset("CCC", exchange="NASDAQ")])]

    result = audit(con, world, snapshots=[*older, *newer])["C2"]

    assert reported(result) == [("CCC", DAYS[3], 21)]
    assert result.notes["snapshot_date"] == "2020-01-20"


def test_c2_without_any_snapshot_has_no_rows_and_a_null_snapshot_note(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    world.bars = listed_after(world.bars, "AAA", 3)

    result = audit(con, world, snapshots=[])["C2"]

    assert rows(result) == []
    assert result.notes == {"no_bars": 0, "snapshot_date": None}
    assert result.fails("all") is False


def test_c2_ext_bars_count_as_the_symbols_bars(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    # AAA's Alpaca bars stop at DAYS[10]; the second source carries it to the end of the store
    world.bars = listed_after(world.bars, "AAA", 10)
    ext = [ext_row("AAA", d, close=100.0, source_symbol="AAA") for d in DAYS[11:]]

    assert reported(audit(con, world, ext=ext)["C2"]) == []
    assert reported(audit(con, world, ext=ext[:3])["C2"]) == [("AAA", DAYS[13], 11)]


def test_c2_store_without_bars_has_no_rows(con: duckdb.DuckDBPyConnection, world: World) -> None:
    world.bars = []

    result = audit(con, world)["C2"]

    assert rows(result) == []
    assert result.notes["no_bars"] == 3


# =============================================================================================
# C2: leakage
# =============================================================================================


def test_c2_ignores_a_snapshot_dated_after_as_of(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    # Two snapshots on disk. Audited at the evening of the last session only the first exists:
    # AAA is listed and stale. Audited in 2021 the second one rules and AAA is gone from it.
    world.bars = listed_after(world.bars, "AAA", 10)
    snapshots = [(SNAP_1, LISTED), (SNAP_2, LISTED[1:])]

    early = audit(con, world, as_of=avail(DAYS[-1]), snapshots=snapshots)["C2"]
    late = audit(con, world, as_of=LATE, snapshots=snapshots)["C2"]

    assert reported(early) == [("AAA", DAYS[10], 14)]
    assert early.notes["snapshot_date"] == SNAP_1.isoformat()
    assert reported(late) == []
    assert late.notes["snapshot_date"] == SNAP_2.isoformat()


def test_c2_snapshot_only_after_as_of_means_no_listing(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    world.bars = listed_after(world.bars, "AAA", 3)

    result = audit(con, world, as_of=avail(DAYS[-1]), snapshots=[(SNAP_2, LISTED)])["C2"]

    assert rows(result) == []
    assert result.notes == {"no_bars": 0, "snapshot_date": None}


def test_c2_snapshot_knowable_from_midnight_of_its_date(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    # assumption documented above: the snapshot of DAYS[-1] is usable from 00:00 New York
    world.bars = listed_after(world.bars, "AAA", 3)
    snapshot_day = DAYS[-1]
    on_the_day = audit(con, world, as_of=avail(snapshot_day), snapshots=[(snapshot_day, LISTED)])
    the_evening_before = audit(
        con, world, as_of=avail(DAYS[-2]), snapshots=[(snapshot_day, LISTED)]
    )

    assert reported(on_the_day["C2"]) == [("AAA", DAYS[3], 21)]
    assert reported(the_evening_before["C2"]) == []


def test_c2_bars_not_yet_available_do_not_move_the_store_end_or_the_last_bar(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    # On disk AAA stops at DAYS[10] and the store runs to DAYS[24]. At the evening of DAYS[12]
    # the store's newest visible session is DAYS[12]: AAA is 2 sessions behind, not 14.
    world.bars = listed_after(world.bars, "AAA", 10)

    early = audit(con, world, as_of=avail(DAYS[12]))["C2"]
    late = audit(con, world, as_of=LATE)["C2"]

    assert reported(early) == []
    assert reported(late) == [("AAA", DAYS[10], 14)]
    # and a symbol whose only bars lie in the future has no visible bar at all
    world.bars = series("AAA", DAYS[20:], 100.0, 30_000) + drop(world.bars, "AAA", DAYS)
    assert audit(con, world, as_of=avail(DAYS[12]))["C2"].notes["no_bars"] == 1


# =============================================================================================
# C1 with ext files, P1 across sources
# =============================================================================================


def c1_rows(results: dict[str, CheckResult]) -> list[dict[str, Any]]:
    return rows(results["C1"])


def test_c1_gains_the_ext_columns_and_counts_zero_without_an_ext_directory(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    result = audit(con, world, checks=["C1"])["C1"]

    assert result.rows.schema.names == [
        *HEAD,
        "year",
        "bars",
        "symbols",
        "liquid_symbols",
        "name_changes",
        "name_changes_linked",
        "ext_bars",
        "ext_overrides",
    ]
    [row] = rows(result)
    assert (row["bars"], row["symbols"], row["ext_bars"], row["ext_overrides"]) == (75, 3, 0, 0)


def test_c1_counts_ext_bars_and_the_overrides_among_them(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    override = ext_row("AAA", DAYS[3], close=101.0, source_symbol="AAA")
    extra = ext_row("AAA", date(2020, 2, 7), close=102.0, source_symbol="AAA")  # a new session

    [row] = c1_rows(audit(con, world, checks=["C1"], ext=[override, extra]))

    assert (row["year"], row["bars"], row["symbols"]) == (2020, 76, 3)
    assert (row["ext_bars"], row["ext_overrides"]) == (2, 1)


def test_c1_ext_bars_are_counted_per_year(con: duckdb.DuckDBPyConnection, world: World) -> None:
    ext = [
        ext_row("AAA", date(2019, 12, 31), close=99.0, source_symbol="AAA"),
        ext_row("AAA", DAYS[3], close=101.0, source_symbol="AAA"),
    ]

    by_year = {r["year"]: r for r in c1_rows(audit(con, world, checks=["C1"], ext=ext))}

    assert (by_year[2019]["bars"], by_year[2019]["ext_bars"], by_year[2019]["ext_overrides"]) == (
        1,
        1,
        0,
    )
    assert (by_year[2020]["bars"], by_year[2020]["ext_bars"], by_year[2020]["ext_overrides"]) == (
        75,
        1,
        1,
    )


def test_c1_ext_bar_not_yet_available_is_not_counted(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    ext = [ext_row("AAA", DAYS[-1], close=101.0, source_symbol="AAA")]

    [row] = c1_rows(audit(con, world, as_of=avail(DAYS[-2]), checks=["C1"], ext=ext))

    assert (row["bars"], row["ext_bars"], row["ext_overrides"]) == (72, 0, 0)


def test_p1_is_not_tripped_by_an_ext_row_overriding_an_alpaca_row(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    alpaca = find(world.bars, "AAA", DAYS[3])
    override = ext_row("AAA", DAYS[3], close=alpaca["close"] + 1, source_symbol="AAA")

    results = audit(con, world, checks=["P1", "C1"], ext=[override])

    assert rows(results["P1"]) == []
    [row] = c1_rows(results)
    assert (row["bars"], row["ext_bars"], row["ext_overrides"]) == (75, 1, 1)


def test_p1_still_catches_a_duplicate_inside_the_ext_files(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    twice = ext_row("AAA", DAYS[3], close=101.0, source_symbol="AAA")

    results = audit(con, world, checks=["P1"], ext=[twice, dict(twice)])

    assert [r["key"] for r in rows(results["P1"])] == [f"AAA:{DAYS[3].isoformat()}"]


def test_i1_sees_the_ext_values_not_the_overridden_alpaca_row(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    # AAA's ext row replaces the Alpaca one; both views (raw and resolved) carry the ext values,
    # so nothing is an orphan
    override = ext_row("AAA", DAYS[3], close=101.0, source_symbol="AAA")

    results = audit(con, world, checks=["I1"], ext=[override])

    assert rows(results["I1"]) == []
    assert results["I1"].notes["bars"] == 0


def test_liquid_scope_of_c2_follows_the_symbols_segments(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    world.bars = listed_after(world.bars, "AAA", 3)
    world.bars = listed_after(world.bars, "CCC", 3)

    result = audit(con, world)["C2"]

    assert [(r["symbol"], r["in_liquid"]) for r in rows(result)] == [("AAA", True), ("CCC", False)]
    assert result.liquid == 1 and result.total == 2
    assert SID_AAA in world.liquid_ids
