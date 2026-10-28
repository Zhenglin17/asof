"""Guards against a spin-off whose factor the prices disprove going unseen: spinoff_events reads
such an event as a restart (reason factor_disproved: the parent's ex-session close plus the
children's value moved more than 15% either way from the previous close), so nothing is
rescaled by it, but the record is worth a look and perhaps a rejection (HON 2026-06-29: +45.5%
on a day nothing happened; BBIG 2022-06-30: -19.6% as TYDE fell on its first day). The check
lists those events, scoped like the split checks, and is not gated.

Target API (asof.store.audit; the check id is not fixed by the spec -- "P1" is taken by the
duplicate-bar check -- so it is a constant here, SPINOFF_CHECK):

    CHECKS[SPINOFF_CHECK]: gate None (listed), key ``PARENT:YYYY-MM-DD`` (the audit's
        _SYM:_DATE), listed right after S5.
    rows: one per spinoff_events(..., as_of) row with status "restart" and reason
        "factor_disproved"; columns include parent_symbol, ex_date, children, total_return.
        The default rejection file applies.
    in_liquid: with ``liquid_tier``, the parent security is in the tier within
        SPLIT_LIQUID_DAYS (92) of the ex-date (as S1 / S2); without it, the parent security is
        in ``liquid_ids``. Out-of-scope rows are listed, not counted by ``fails("liquid")``.

Fake data provenance:
- The world is tests/store/audit_world.py's (AAA 100, BBB 60, CCC 10 over 25 sessions of 2020).
- HON 2026-06-29 is the real case behind the check (store probe 2026-10-27: 232.21 -> 227.80,
  HONA 220.19 at 0.5 -> +0.455); genuine spin-offs on the real store sit within a few percent
  (GE -> GEV -0.022, GE -> GEHC +0.063, LEN -> MRP +0.035, OPEN -0.026).
- 0.15 is asof.store.spinoffs.MAX_TOTAL_RETURN, chosen by the user 2026-10-27. Real events
  above it (store probe 2026-10-27): BBBY 2025-10-02 +0.19, BBIG 2022-06-30 -0.20 (the only
  liquid one), JTAI 2026-07-15 +0.46, LAC 2023-10-04 -0.26, PRSR 2023-12-22 +0.77, WKEY
  2023-05-24 +0.45 and HON 2026-06-29 (rejected).

Real-data check: tests/store/test_golden_spinoffs.py runs the check on the real store with the
CLI's tier and expects BBIG 2022-06-30 listed in the liquid scope and HON 2026-06-29 absent.
"""

import re
from collections.abc import Iterator
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pytest

from asof.store import audit as audit_mod
from asof.store import spinoffs
from asof.store.audit import CheckResult
from asof.store.market import market_root
from tests.store.audit_world import (
    DAYS,
    LATE,
    SID_AAA,
    SID_BBB,
    SID_CCC,
    SID_X1,
    Seg,
    World,
    bar,
    base_segs,
    exc,
    iso,
    keys,
    rows,
    series,
    spin_off,
)
from tests.store.test_audit_splits import audit, by_key, tier
from tests.store.test_market import avail, known

SPINOFF_CHECK = "S6"
EX = DAYS[10]  # 2020-01-16


def spin_world(
    world: World,
    *,
    parent: str = "AAA",
    child: str = "KID",
    after: float = 100.0,
    child_close: float = 40.0,
    ratio: float = 1.0,
    offset: int | None = 0,
) -> str:
    """``parent`` closes ``after`` from EX on (its base price before); ``child`` trades at
    ``child_close`` from ``offset`` sessions after EX (never when None); one record on EX."""
    i = DAYS.index(EX)
    world.bars = [
        bar(parent, b["session_date"], close=after, volume=b["volume"])
        if b["symbol"] == parent and b["session_date"] >= EX
        else b
        for b in world.bars
    ]
    if offset is not None and child != parent:
        first = DAYS[i + offset]
        world.bars += series(child, DAYS[i + offset :], child_close, 7_000, start=i + offset)
        world.segs = [
            *world.segs,
            Seg(SID_X1, child, valid_from=first, available_at=known(first)),
        ]
    world.others = [spin_off(parent, child, EX, ratio)]
    return f"{parent}:{iso(EX)}"


def check(con: duckdb.DuckDBPyConnection, world: World, **kwargs: Any) -> CheckResult:
    kwargs.setdefault("liquid_tier", None)
    return audit(con, world, checks=[SPINOFF_CHECK], **kwargs)[SPINOFF_CHECK]


@pytest.fixture
def con() -> Iterator[duckdb.DuckDBPyConnection]:
    connection = duckdb.connect()
    yield connection
    connection.close()


@pytest.fixture
def world(tmp_path: Path) -> World:
    return World(market_root(tmp_path))


# =============================================================================================
# the registry
# =============================================================================================


def test_check_is_registered_after_s5_listed_not_gated() -> None:
    ids = list(audit_mod.CHECKS)
    assert SPINOFF_CHECK in ids and ids.index(SPINOFF_CHECK) == ids.index("S5") + 1
    spec = audit_mod.CHECKS[SPINOFF_CHECK]
    assert spec.gate is None
    assert spec.title.strip()
    assert re.fullmatch(spec.key_pattern, "HON:2026-06-29")
    assert re.fullmatch(spec.key_pattern, "BRK.B:2020-01-16")


# =============================================================================================
# normal path
# =============================================================================================


def test_adjust_event_with_no_price_drop_is_reported(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    # AAA stays at 100 although every share received a 40 child: +40%
    key = spin_world(world, after=100.0, child_close=40.0, ratio=1.0)

    result = check(con, world)

    row = by_key(result)[key]
    assert row["in_liquid"] is True and row["excepted"] is False
    assert row["parent_symbol"] == "AAA" and row["ex_date"] == EX
    assert row["children"] == "KID"
    assert row["total_return"] == pytest.approx(0.4)
    assert result.liquid == 1
    assert result.fails("liquid") is False  # listed: the event is already a restart


def test_genuine_spin_off_is_not_reported(con: duckdb.DuckDBPyConnection, world: World) -> None:
    # AAA drops to 62 while the child is worth 40: +2%
    spin_world(world, after=62.0, child_close=40.0, ratio=1.0)

    result = check(con, world)

    assert rows(result) == []
    assert result.fails("liquid") is False and result.fails("all") is False


# =============================================================================================
# boundaries
# =============================================================================================


@pytest.mark.parametrize(
    ("after", "reported"),
    [(74.0, False), (76.0, True), (46.0, False), (44.0, True)],
    ids=["plus_14", "plus_16", "minus_14", "minus_16"],
)
def test_threshold_is_fifteen_percent_either_way(
    con: duckdb.DuckDBPyConnection, world: World, after: float, reported: bool
) -> None:
    key = spin_world(world, after=after, child_close=40.0, ratio=1.0)

    assert (key in keys(check(con, world))) is reported


@pytest.mark.parametrize(
    "shape",
    ["same_symbol", "factor_out_of_range", "pending"],
)
def test_events_that_are_not_adjust_are_not_reported(
    con: duckdb.DuckDBPyConnection, world: World, shape: str
) -> None:
    if shape == "same_symbol":
        spin_world(world, child="AAA")
    elif shape == "factor_out_of_range":
        spin_world(world, child_close=150.0, ratio=1.0)  # the child is worth more than AAA was
    else:
        spin_world(world, offset=None)  # no child bar and fewer than 20 sessions after EX

    assert rows(check(con, world)) == []


def test_parent_outside_the_liquid_ids_is_listed_not_counted(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    key = spin_world(world, parent="CCC", after=10.0, child_close=4.0, ratio=1.0)
    assert SID_CCC not in world.liquid_ids

    result = check(con, world)

    assert by_key(result)[key]["in_liquid"] is False
    assert result.liquid == 0 and result.total == 1


@pytest.mark.parametrize(
    ("offset_days", "liquid"),
    [(-93, False), (-92, True), (92, True), (93, False)],
    ids=["93_days_before", "92_days_before", "92_days_after", "93_days_after"],
)
def test_dated_tier_gates_by_the_parent_security_around_the_ex_date(
    con: duckdb.DuckDBPyConnection, world: World, offset_days: int, liquid: bool
) -> None:
    key = spin_world(world)
    day = EX + timedelta(days=offset_days)

    result = check(con, world, liquid_tier=tier([(day, SID_AAA, "common")]))

    assert by_key(result)[key]["in_liquid"] is liquid


def test_dated_tier_of_another_security_does_not_count(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    key = spin_world(world)

    result = check(con, world, liquid_tier=tier([(date(2020, 1, 1), SID_BBB, "common")]))

    assert by_key(result)[key]["in_liquid"] is False


def test_exception_silences_the_row(con: duckdb.DuckDBPyConnection, world: World) -> None:
    key = spin_world(world)
    world.write()
    [result] = audit_mod.run_audit(
        con,
        world.root,
        as_of=LATE,
        liquid_ids=world.liquid_ids,
        checks=[SPINOFF_CHECK],
        exceptions=[exc(SPINOFF_CHECK, key)],
    )

    assert by_key(result)[key]["excepted"] is True
    assert result.fails("liquid") is False


def test_rejected_record_is_not_reported(
    con: duckdb.DuckDBPyConnection, world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = spin_world(world)
    path = tmp_path / "spinoff_rejections.yaml"
    path.write_text(f'- key: "{key}"\n  reason: constructed\n  added: 2026-10-27\n  by: greyson\n')
    monkeypatch.setattr(spinoffs, "SPINOFF_REJECTIONS_FILE", path)

    assert rows(check(con, world)) == []


# =============================================================================================
# leakage
# =============================================================================================


def test_child_price_not_yet_visible_reports_nothing_then_the_row(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    # the child's first bar is on EX+2 and on disk; the evening before, the event is pending
    key = spin_world(world, offset=2)
    i = DAYS.index(EX)

    before = check(con, world, as_of=avail(DAYS[i + 1]))
    assert rows(before) == []
    after = check(con, world, as_of=avail(DAYS[i + 2]))
    assert keys(after) == [key]


def test_record_not_yet_knowable_reports_nothing(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    spin_world(world)

    assert rows(check(con, world, as_of=known(EX) - timedelta(microseconds=1))) == []


def test_base_world_without_spin_offs_is_clean(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    assert world.segs == base_segs() and world.others == []
    assert rows(check(con, world)) == []
