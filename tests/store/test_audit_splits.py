"""Guards against the split checks S1 / S2 gating on the wrong rows: a split record judged
against a liquid tier from another year, a record whose bar sits under the security's other
name counted as missing, a reverse split whose price moved one session late or whose previous
close belongs to the label's earlier holder counted as "no move" -- and against any of them
deciding with bars or tier rows not yet knowable at ``as_of``.

Target API (asof.store.audit; implementation pending, step 6-c):

    run_audit(..., liquid_tier: pa.Table | None = None)
        rows shaped like asof.store.universe.LIQUID_SCHEMA; only classes in LIQUID_CLASSES
        count. When given, an S1 / S2 row is in_liquid only if the security owning the record's
        symbol on the ex-date (valid_from <= ex_date <= valid_to: the segment that ends that
        day counts) is in the tier at some as_of within SPLIT_LIQUID_DAYS of the ex-date.
        When None, today's rule (liquid_ids, any segment of the symbol).
    SPLIT_LIQUID_DAYS = 92
    SPLIT_TRADED_SESSIONS = 20   S1: reason "not_traded_around" when the owning security has no
                                 traded bar within this many sessions before the ex-date, or
                                 none within this many after; reported, never in_liquid
    SPLIT_LATE_SESSIONS = 2      S2: the move may land on the ex-date or on one of the next two
                                 traded sessions
    S1: a record has its bar when any symbol of the same security has a bar that day.
    S2: the previous close must belong to the same security as the ex-date bar.

Fake data provenance (store probe 2026-10-19 unless noted):
- UTX: reverse_split record 2020-04-03 factor 0.5; UTX's bars stop 04-02, RTX's (identical
  copies until then) carry 04-03 at 51.16 open (real). The master runs UTX up to 04-03 and RTX
  from it, one security (real mechanism; ids constructed).
- ULTY: reverse_split records 2025-11-28 and 2025-12-01, factor 0.1 (10:1); open/prev on
  11-28 is 4.00 / 4.00 = 1.0, on 12-01 39.53 / 4.03 = 9.8 (real). The fixtures use 4.0 -> 40.0
  (constructed round numbers) and only the first record.
- ARNC: reverse_split record 2020-04-01 factor 0.25 on the day Arconic Corp took the label from
  Arconic Inc (real record; the fixture's prices are constructed so that today's S2 reports it).
- Liquid tier rows: one per (as_of, security) at 00:00 New York of a month start
  (asof.store.universe.month_starts, docs). The boundary tests place as_of 92 / 93 calendar days
  from the ex-date instead of at month starts; ASSUMPTION: the distance is measured from the
  ex-date to the New York calendar date of as_of, and the implementation does not require
  month starts.
- Sessions are the distinct visible session dates of the store (the audit's derived calendar).

Interpretations a reviewer should confirm (the spec does not say):
- A tier row whose as_of lies after the audit's ``as_of`` is not knowable yet and does not count
  (``test_tier_rows_after_the_audit_as_of_do_not_count``). The CLI computes the tier only up to
  ``--to`` (default: the audit's as_of), so this only matters for callers passing a wider tier.
- "Within 20 sessions" includes the 20th session before / after the ex-date.

Real-data check: ``asof market audit`` (manual): S1 / S2 clean in the liquid tier after 6-c;
S1's UTX:2020-04-03 and S2's ULTY:2025-11-28 and ARNC:2020-04-01 gone from the report.

S5 (several split events on one day): one row per (symbol, ex_date)
that still has more than one event after ``splits_as_of`` merged repeated records (rule and
the real SNEX / NLSP / MDRR / GWACU shapes: tests/store/test_split_events.py). Key
``SYMBOL:YYYY-MM-DD``; gate "all": every such day needs a reviewed exception, because a
wrong extra record multiplies a held position by its factor. Real-data check: ``asof market
audit`` S5 lists MDRR:2024-07-03, GWACU:2020-12-07 and SAQNU:2020-10-15 on the store of
2026-10-27, nothing for SNEX / NLSP.
"""

import re
from collections.abc import Iterator, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pytest

from asof.store import audit as audit_mod
from asof.store.audit import CheckResult, run_audit
from asof.store.market import market_root
from asof.store.universe import LIQUID_CLASSES, LIQUID_SCHEMA
from tests.store.audit_world import (
    DAYS,
    LATE,
    SID_AAA,
    SID_BBB,
    SID_D1,
    SID_X1,
    SID_X2,
    Seg,
    World,
    base_segs,
    drop,
    exc,
    iso,
    keys,
    rows,
    series,
    split,
)
from tests.store.test_market import avail, known
from tests.store.test_split_events import (
    gwacu_rows,
    mdrr_rows,
    nlsp_rows,
    snex_rows,
    split_row,
)
from tests.store.test_universe import trading_days

EX = DAYS[10]  # 2020-01-16


def tier(entries: Sequence[tuple[date, str, str]]) -> pa.Table:
    """Liquid-tier rows from (as_of day, security_id, class); as_of is 00:00 New York."""
    return pa.Table.from_pylist(
        [
            {
                "as_of": known(day),
                "security_id": sid,
                "symbol": "ANY",
                "class": cls,
                "adv_usd": 1e8,
                "sessions_traded": 20,
            }
            for day, sid, cls in entries
        ],
        schema=LIQUID_SCHEMA,
    )


def audit(
    con: duckdb.DuckDBPyConnection,
    world: World,
    *,
    liquid_tier: pa.Table | None,
    checks: Sequence[str] = ("S1", "S2"),
    as_of: datetime = LATE,
) -> dict[str, CheckResult]:
    world.write()
    # liquid_tier=None is passed by omission, so the guards below also run on today's code
    extra: dict[str, Any] = {} if liquid_tier is None else {"liquid_tier": liquid_tier}
    results = run_audit(
        con, world.root, as_of=as_of, liquid_ids=world.liquid_ids, checks=list(checks), **extra
    )
    return {r.spec.id: r for r in results}


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


def test_split_constants() -> None:
    assert audit_mod.SPLIT_LIQUID_DAYS == 92
    assert audit_mod.SPLIT_TRADED_SESSIONS == 20
    assert audit_mod.SPLIT_LATE_SESSIONS == 2
    assert LIQUID_CLASSES == ("common", "etf")


# =============================================================================================
# gating by date: the tier around the ex-date, not the union over all years
# =============================================================================================


def s1_row_on_aaa(world: World) -> str:
    """AAA split on EX with its EX bar missing: an S1 row (no_bar_that_day)."""
    world.splits = [split("AAA", EX, 1.0, 2.0)]
    world.bars = drop(world.bars, "AAA", [EX])
    return f"AAA:{iso(EX)}"


def test_s1_row_is_liquid_when_the_security_is_in_the_tier_near_the_ex_date(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    key = s1_row_on_aaa(world)

    near = audit(con, world, liquid_tier=tier([(date(2020, 1, 1), SID_AAA, "common")]))["S1"]
    assert by_key(near)[key]["in_liquid"] is True
    assert by_key(near)[key]["reason"] == "no_bar_that_day"
    assert near.fails("liquid") is True
    # AAA is in liquid_ids (the all-years union) but not in the tier around 2020-01: out
    other = audit(con, world, liquid_tier=tier([(date(2020, 1, 1), SID_BBB, "common")]))["S1"]
    assert SID_AAA in world.liquid_ids
    assert by_key(other)[key]["in_liquid"] is False
    assert other.fails("liquid") is False and other.fails("all") is True


@pytest.mark.parametrize(
    ("offset_days", "liquid"),
    [(-93, False), (-92, True), (92, True), (93, False)],
    ids=["93_days_before", "92_days_before", "92_days_after", "93_days_after"],
)
def test_tier_as_of_counts_up_to_92_days_either_side_of_the_ex_date(
    con: duckdb.DuckDBPyConnection, world: World, offset_days: int, liquid: bool
) -> None:
    key = s1_row_on_aaa(world)
    day = EX + timedelta(days=offset_days)

    result = audit(con, world, liquid_tier=tier([(day, SID_AAA, "common")]), checks=["S1"])["S1"]

    assert by_key(result)[key]["in_liquid"] is liquid


@pytest.mark.parametrize(("cls", "liquid"), [("etf", True), ("unknown", False), ("warrant", False)])
def test_only_liquid_classes_in_the_tier_count(
    con: duckdb.DuckDBPyConnection, world: World, cls: str, liquid: bool
) -> None:
    key = s1_row_on_aaa(world)

    result = audit(con, world, liquid_tier=tier([(date(2020, 1, 1), SID_AAA, cls)]), checks=["S1"])

    assert by_key(result["S1"])[key]["in_liquid"] is liquid


def test_s2_row_is_gated_by_the_tier_around_its_ex_date(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    # AAA 4:1 on EX with no price move (test_audit.split_world shape): an S2 row
    world.splits = [split("AAA", EX, 1.0, 4.0)]
    key = f"AAA:{iso(EX)}"

    inside = audit(con, world, liquid_tier=tier([(EX + timedelta(days=92), SID_AAA, "common")]))
    outside = audit(con, world, liquid_tier=tier([(EX + timedelta(days=93), SID_AAA, "common")]))

    assert by_key(inside["S2"])[key]["in_liquid"] is True
    assert by_key(outside["S2"])[key]["in_liquid"] is False


def rename_without_ex_bar(world: World) -> str:
    """OLDN -> NEWN on EX inside one security, no bar under either name on EX: the record is
    filed under OLDN, whose segment ends on EX (the UTX shape without RTX's bar)."""
    world.segs = [
        *base_segs(),
        Seg(SID_X1, "OLDN", valid_to=EX, end_available_at=known(EX)),
        Seg(SID_X1, "NEWN", valid_from=EX, available_at=known(EX)),
    ]
    world.bars += series("OLDN", DAYS[:10], 40.0, 9_000) + series(
        "NEWN", DAYS[11:], 40.0, 9_000, start=11
    )
    world.splits = [split("OLDN", EX, 1.0, 2.0)]
    return f"OLDN:{iso(EX)}"


def test_segment_ending_on_the_ex_date_owns_the_record(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    key = rename_without_ex_bar(world)

    result = audit(con, world, liquid_tier=tier([(date(2020, 1, 1), SID_X1, "common")]))["S1"]

    row = by_key(result)[key]
    # the security traded under OLDN before and under NEWN after: it was trading around EX
    assert row["reason"] == "no_bar_that_day"
    assert row["in_liquid"] is True


def test_without_a_tier_the_union_rule_is_unchanged(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    # Guard (passes today, must keep passing): liquid_tier=None keeps liquid_ids as the scope.
    key = s1_row_on_aaa(world)

    result = audit(con, world, liquid_tier=None, checks=["S1"])["S1"]

    assert by_key(result)[key]["in_liquid"] is True
    world.liquid_ids = {SID_BBB}
    assert (
        by_key(audit(con, world, liquid_tier=None, checks=["S1"])["S1"])[key]["in_liquid"] is False
    )


# =============================================================================================
# S1: the bar may sit under another name of the security; long silence is its own reason
# =============================================================================================


def utx_world(world: World, *, same_security: bool) -> str:
    """UTX's record on EX, its segment ending EX; RTX carries EX's bar from EX on."""
    rtx_sid = SID_X1 if same_security else SID_X2
    world.segs = [
        *base_segs(),
        Seg(SID_X1, "UTX", valid_to=EX, end_available_at=known(EX)),
        Seg(rtx_sid, "RTX", valid_from=EX, available_at=known(EX)),
    ]
    world.bars += series("UTX", DAYS[:10], 90.0, 13_000) + series(
        "RTX", DAYS[10:], 51.0, 23_000, start=10
    )
    world.splits = [split("UTX", EX, 1.0, 0.5, kind="reverse_split")]
    return f"UTX:{iso(EX)}"


def test_s1_bar_under_the_securitys_other_name_counts(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    # Today: UTX has no bar on 2020-04-03, so S1 reports it although RTX's bar is that very
    # security's session.
    key = utx_world(world, same_security=True)

    result = audit(con, world, liquid_tier=None, checks=["S1"])["S1"]

    assert key not in keys(result)


def test_s1_bar_of_another_security_does_not_count(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    key = utx_world(world, same_security=False)

    result = audit(con, world, liquid_tier=None, checks=["S1"])["S1"]

    assert key in keys(result)


LONG = trading_days(date(2020, 1, 2), date(2020, 4, 30))
LONG_EX = LONG[40]


def silent_world(world: World, *, last_before: int, first_after: int) -> str:
    """DEAD (its own security) trades up to ``last_before`` sessions before LONG_EX and again
    from ``first_after`` sessions after it; the split record sits on LONG_EX."""
    i = LONG.index(LONG_EX)
    world.segs = [*base_segs(), Seg(SID_D1, "DEAD", cik=4001)]
    world.bars = (
        series("AAA", LONG, 100.0, 30_000)
        + series("BBB", LONG, 60.0, 20_000)
        + series("CCC", LONG, 10.0, 5_000)
        + series("DEAD", LONG[: i - last_before + 1], 7.0, 8_000)
        + series("DEAD", LONG[i + first_after :], 7.0, 8_000, start=i + first_after)
    )
    world.splits = [split("DEAD", LONG_EX, 1.0, 2.0)]
    return f"DEAD:{iso(LONG_EX)}"


@pytest.mark.parametrize(
    ("last_before", "first_after", "reason"),
    [
        (20, 1, "no_bar_that_day"),
        (21, 1, "not_traded_around"),
        (1, 20, "no_bar_that_day"),
        # reviewer 2026-10-19: DEAD's segment is still open, so silence after the record may be
        # lost bars, not a security that stopped; it stays a gated no_bar_that_day
        (1, 21, "no_bar_that_day"),
    ],
    ids=["traded_20_before", "silent_21_before", "traded_20_after", "silent_21_after_open"],
)
def test_s1_security_silent_for_more_than_20_sessions_around_the_ex_date(
    con: duckdb.DuckDBPyConnection,
    world: World,
    last_before: int,
    first_after: int,
    reason: str,
) -> None:
    key = silent_world(world, last_before=last_before, first_after=first_after)
    in_tier = tier([(date(2020, 3, 1), SID_D1, "common")])

    result = audit(con, world, liquid_tier=in_tier, checks=["S1"])["S1"]

    row = by_key(result)[key]
    assert row["reason"] == reason
    # reported either way; a security that was not trading is never a liquid-tier defect
    assert row["in_liquid"] is (reason == "no_bar_that_day")


# =============================================================================================
# S2: a move one or two sessions late is the move; a previous close of another holder is not
# =============================================================================================


def late_move_world(world: World, *, late: int) -> str:
    """ULT 10:1 reverse split recorded on EX; the price jumps 10x ``late`` sessions after it."""
    i = DAYS.index(EX)
    world.segs = [*base_segs(), Seg(SID_X1, "ULT", cik=5001)]
    world.bars += series("ULT", DAYS[: i + late], 4.0, 9_000_000) + series(
        "ULT", DAYS[i + late :], 40.0, 1_600_000, start=i + late
    )
    world.splits = [split("ULT", EX, 10.0, 1.0, kind="reverse_split")]
    return f"ULT:{iso(EX)}"


@pytest.mark.parametrize(
    ("late", "reported"),
    [(0, False), (1, False), (2, False), (3, True)],
    ids=["on_the_ex_date", "next_session", "two_sessions_late", "three_sessions_late"],
)
def test_s2_move_up_to_two_traded_sessions_after_the_ex_date_is_the_move(
    con: duckdb.DuckDBPyConnection, world: World, late: int, reported: bool
) -> None:
    # ULTY shape. Today only late=0 is clean.
    key = late_move_world(world, late=late)

    result = audit(con, world, liquid_tier=None, checks=["S2"])["S2"]

    assert (key in keys(result)) is reported


def test_s2_late_move_not_yet_visible_does_not_clear_the_record(
    con: duckdb.DuckDBPyConnection, world: World, tmp_path: Path
) -> None:
    # Leakage: the 10x bar of the next session is on disk, but at 20:00 New York of the ex-date
    # only the ex-date's own bar is knowable. The audit then must say exactly what it says about
    # a store in which the later bars do not exist yet; once the next bar is visible the record
    # is cleared.
    key = late_move_world(world, late=1)
    next_session = DAYS[DAYS.index(EX) + 1]
    at_ex = avail(EX)

    seen = audit(con, world, liquid_tier=None, checks=["S2"], as_of=at_ex)["S2"]
    truncated = World(market_root(tmp_path / "truncated"))
    late_move_world(truncated, late=1)
    truncated.bars = [b for b in truncated.bars if b["session_date"] <= EX]
    alone = audit(con, truncated, liquid_tier=None, checks=["S2"], as_of=at_ex)["S2"]
    assert rows(seen) == rows(alone)
    assert seen.notes == alone.notes
    after = audit(con, world, liquid_tier=None, checks=["S2"], as_of=avail(next_session))["S2"]
    assert key not in keys(after)


def holder_change_world(world: World, *, same_security: bool) -> str:
    """LBL changes hands on EX (ARNC shape) with a 4:1 record that day; the new holder trades
    at 60 against the old holder's 100, so open / prev close reads 0.6 against 0.25."""
    new_sid = SID_X1 if same_security else SID_X2
    world.segs = [
        *base_segs(),
        Seg(SID_X1, "LBL", valid_to=EX, end_available_at=known(EX), cik=4281),
        Seg(new_sid, "LBL", valid_from=EX, available_at=known(EX), cik=1790982),
    ]
    i = DAYS.index(EX)
    world.bars += series("LBL", DAYS[:i], 100.0, 40_000) + series(
        "LBL", DAYS[i:], 60.0, 40_000, start=i
    )
    world.splits = [split("LBL", EX, 1.0, 4.0)]
    return f"LBL:{iso(EX)}"


def test_s2_previous_close_of_another_security_is_not_compared(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    key = holder_change_world(world, same_security=False)

    result = audit(con, world, liquid_tier=None, checks=["S2"])["S2"]

    assert key not in keys(result)


def test_s2_previous_close_of_the_same_security_is_still_compared(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    # Guard (passes today, must keep passing): two segments of one security (a CIK switch the
    # build bridged) still compare across the cut.
    key = holder_change_world(world, same_security=True)

    result = audit(con, world, liquid_tier=None, checks=["S2"])["S2"]

    assert key in keys(result)


# =============================================================================================
# leakage: tier rows from after the audit's as_of
# =============================================================================================


def test_tier_rows_after_the_audit_as_of_do_not_count(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    # Interpretation (see the module docstring): a tier row computed at 2020-03-01 is not
    # knowable to an audit run as of 2020-02-10, although it lies within 92 days of the ex-date.
    key = s1_row_on_aaa(world)
    future = tier([(date(2020, 3, 1), SID_AAA, "common")])
    as_of = datetime(2020, 2, 10, 12, tzinfo=UTC)

    early = audit(con, world, liquid_tier=future, checks=["S1"], as_of=as_of)["S1"]
    assert by_key(early)[key]["in_liquid"] is False
    late = audit(con, world, liquid_tier=future, checks=["S1"])["S1"]
    assert by_key(late)[key]["in_liquid"] is True


def test_s1_record_near_the_end_of_the_store_is_not_judged_on_sessions_it_cannot_have(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    # Boundary: a record four sessions before the store ends has no 20 sessions "after" yet. The
    # missing after-side is no evidence that the security stopped trading, so the row stays a
    # gated no_bar_that_day instead of dropping out of the gate as not_traded_around.
    ex = LONG[-5]
    i = LONG.index(ex)
    world.segs = [*base_segs(), Seg(SID_D1, "DEAD", cik=4001)]
    world.bars = (
        series("AAA", LONG, 100.0, 30_000)
        + series("BBB", LONG, 60.0, 20_000)
        + series("CCC", LONG, 10.0, 5_000)
        + series("DEAD", LONG[:i], 7.0, 8_000)
    )
    world.splits = [split("DEAD", ex, 1.0, 2.0)]
    key = f"DEAD:{iso(ex)}"

    result = audit(con, world, liquid_tier=tier([(date(2020, 4, 1), SID_D1, "common")]))["S1"]

    row = by_key(result)[key]
    assert row["reason"] == "no_bar_that_day"
    assert row["in_liquid"] is True


def test_s1_silence_after_a_security_that_ended_is_not_gated(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    # CYBR shape: the record falls on the day the security stops (bought); its segment ends the
    # next session and nothing trades after. That silence is the end of the security, not a
    # missing bar: reported as not_traded_around, outside the gate.
    i = LONG.index(LONG_EX)
    end = LONG[i + 1]
    world.segs = [
        *base_segs(),
        Seg(SID_D1, "DEAD", cik=4001, valid_to=end, end_available_at=known(end)),
    ]
    world.bars = (
        series("AAA", LONG, 100.0, 30_000)
        + series("BBB", LONG, 60.0, 20_000)
        + series("CCC", LONG, 10.0, 5_000)
        + series("DEAD", LONG[:i], 7.0, 8_000)
    )
    world.splits = [split("DEAD", LONG_EX, 1.0, 2.0)]
    key = f"DEAD:{iso(LONG_EX)}"

    result = audit(con, world, liquid_tier=tier([(date(2020, 3, 1), SID_D1, "common")]))["S1"]

    row = by_key(result)[key]
    assert row["reason"] == "not_traded_around"
    assert row["in_liquid"] is False


# =============================================================================================
# S5: several split events on one day
# =============================================================================================


S5_AS_OF = datetime(2026, 10, 27, tzinfo=UTC)  # the real shapes' ex-dates are all knowable


def s5(
    con: duckdb.DuckDBPyConnection, world: World, as_of: datetime = S5_AS_OF, **kwargs: Any
) -> CheckResult:
    return world.audit(con, as_of=as_of, checks=["S5"], **kwargs)["S5"]


def test_s5_is_in_the_check_list_after_s4_and_gates_everywhere() -> None:
    ids = list(audit_mod.CHECKS)
    assert "S5" in ids and ids.index("S5") == ids.index("S4") + 1
    spec = audit_mod.CHECKS["S5"]
    assert spec.gate == "all"
    assert spec.title.strip()


def test_s5_flags_mdrr_and_gwacu_shapes_one_row_per_day(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    world.splits = mdrr_rows() + gwacu_rows()

    result = s5(con, world)

    assert keys(result) == ["GWACU:2020-12-07", "MDRR:2024-07-03"]
    assert result.fails("liquid") is True and result.fails("all") is True


def test_s5_does_not_flag_repeated_records_of_one_event(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    world.splits = snex_rows() + nlsp_rows() + [split_row("AAPL", date(2020, 8, 31), 1.0, 4.0)]

    assert rows(s5(con, world)) == []


def test_s5_a_day_with_three_records_two_of_them_copies_is_one_row(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    first, second = mdrr_rows()
    world.splits = [first, second, {**second, "kind": "unit_split"}]

    assert keys(s5(con, world)) == ["MDRR:2024-07-03"]


def test_s5_exception_silences_the_row(con: duckdb.DuckDBPyConnection, world: World) -> None:
    world.splits = mdrr_rows() + gwacu_rows()

    result = s5(con, world, exceptions=[exc("S5", "MDRR:2024-07-03")])

    excepted = {r["key"]: r["excepted"] for r in rows(result)}
    assert excepted == {"GWACU:2020-12-07": False, "MDRR:2024-07-03": True}
    assert audit_mod.CHECKS["S5"].key_pattern
    assert re.fullmatch(audit_mod.CHECKS["S5"].key_pattern, "MDRR:2024-07-03")


def test_s5_second_event_not_yet_knowable_is_not_flagged(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    # leakage: the second MDRR event is on disk but its available_at lies after the audit's
    # as_of; at that as_of the day has one event. One microsecond later it has two.
    first, second = mdrr_rows()
    as_of = known(first["ex_date"]) + timedelta(hours=1)
    world.splits = [first, {**second, "available_at": as_of + timedelta(microseconds=1)}]

    assert rows(s5(con, world, as_of=as_of)) == []
    assert keys(s5(con, world, as_of=as_of + timedelta(microseconds=1))) == ["MDRR:2024-07-03"]
