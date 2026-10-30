"""Guards against an audit check that can no longer fail: a check added to ``CHECKS`` without a
fixture row that trips it, or a query edit that makes a check silently find nothing (a renamed
column, a join that never matches, a filter that drops every row). Every check id has one
mutation of the clean world that must make it report; every ``check_master`` rule has one
broken master table that must make it refuse. The registries are compared with
``set(CHECKS)`` and ``set(_MASTER_RULES)``, so a new check or rule without a breaking case
fails here before anything else.

The mutations reuse the shapes the per-check tests already pin (tests/store/test_audit*.py;
their provenance notes apply). The clean world is the audit's base world plus an Alpaca assets
snapshot listing AAA, BBB and CCC (C2 has a listing to judge; the shape of
tests/store/test_audit_c2.py) and no SEC exchange history (I10 has nothing to report).

C1 and I6 report one information row per year even on a clean store; for them "breaks" means
"the row changes", and the clean rows are pinned.

No new assumption about real data is introduced here: every mutation is a shape a per-check
test already documents, and the real-data checks listed there (``asof market audit``) stay the
ones that matter.
"""

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from dataclasses import replace as with_fields
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pytest

from asof.ingest import security_master as sm
from asof.ingest.alpaca import Asset
from asof.ingest.universe import save_assets_snapshot
from asof.store.audit import CHECKS, CheckResult, run_audit
from asof.store.market import market_root
from asof.store.universe import LIQUID_SCHEMA
from tests.ingest.test_security_master import cm_good_rows, cm_violation, master_table
from tests.store.audit_world import (
    DAYS,
    LATE,
    SID_AAA,
    SID_R,
    SID_X1,
    SID_X2,
    START,
    Seg,
    World,
    bar,
    base_bars,
    class_row,
    conflict,
    dividend,
    drop,
    find,
    flat,
    rename,
    replace,
    rows,
    series,
    spin_off,
    split,
)
from tests.store.test_audit import (
    copies_world,
    jump_world,
    lac_world,
    rename_world,
    spike_world,
    split_world,
)
from tests.store.test_audit_c2 import LISTED, SNAP_1, listed_after
from tests.store.test_audit_i10 import THREE, Listing, write_exchange_history
from tests.store.test_market import KNOWN_EARLY, avail, known

INFO_PER_YEAR = ("C1", "I6")  # checks that report a row per year on a clean store


@dataclass
class MetaWorld(World):
    """The audit world plus the two inputs only C2 and I10 read: Alpaca assets snapshots and
    the SEC exchange history, written after the base world on every ``write``."""

    assets: list[tuple[date, list[Asset]]] = field(default_factory=lambda: [(SNAP_1, LISTED)])
    exchange: list[Listing] = field(default_factory=list)
    tier: pa.Table | None = None  # the dated liquid tier (F1 judges its nights); None: not passed

    def write(self) -> Path:
        root = super().write()
        for day, listed in self.assets:
            save_assets_snapshot(list(listed), root, day)
        if self.exchange:
            write_exchange_history(root, self.exchange)
        return root


# =============================================================================================
# one breaking mutation per check
# =============================================================================================


def _p1(w: MetaWorld) -> None:
    w.bars.append(dict(find(w.bars, "AAA", DAYS[3])))  # the same (symbol, t) twice


def _p2(w: MetaWorld) -> None:
    bad = bar("AAA", DAYS[3], close=100.0, volume=30_003, low=0.0)  # a price of zero
    w.bars = replace(w.bars, "AAA", DAYS[3], bad)


def _p3(w: MetaWorld) -> None:
    early_fetch = avail(DAYS[3]) - timedelta(hours=1)
    w.bars = replace(
        w.bars,
        "AAA",
        DAYS[3],
        bar("AAA", DAYS[3], close=100.0, volume=30_003, fetched_at=early_fetch),
    )


def _p4(w: MetaWorld) -> None:
    misfiled = find(w.bars, "AAA", DAYS[3])
    w.bars = drop(w.bars, "AAA", [DAYS[3]])
    w.misfiled = [(2019, misfiled)]


def _p5(w: MetaWorld) -> None:
    moving = bar("AAA", DAYS[3], close=101.0, open=100.0, high=101.0, low=99.0, volume=0)
    w.bars = replace(w.bars, "AAA", DAYS[3], moving)


def _p6(w: MetaWorld) -> None:
    early = avail(DAYS[3]) - timedelta(hours=1)
    w.bars = replace(
        w.bars, "AAA", DAYS[3], bar("AAA", DAYS[3], close=100.0, volume=30_003, available_at=early)
    )


def _m1(w: MetaWorld) -> None:
    # ARNC shape: the old holder's end known at a later snapshot than the new holder's start
    cut = DAYS[10]
    w.segs += [
        Seg(SID_X1, "XXX", START, cut, KNOWN_EARLY, known(DAYS[12])),
        Seg(SID_X2, "XXX", cut, None, known(cut)),
    ]


def _m2(w: MetaWorld) -> None:
    w.segs += [
        Seg(SID_R, "ROLD", START, DAYS[12], KNOWN_EARLY, known(DAYS[12])),
        Seg(SID_R, "RNEW", DAYS[10], None, known(DAYS[10])),
    ]
    w.liquid_ids.add(SID_R)


def _m3(w: MetaWorld) -> None:
    w.segs += [
        Seg(SID_R, "ROLD", START, DAYS[8], KNOWN_EARLY, known(DAYS[8])),
        Seg(SID_R, "RNEW", DAYS[12], None, known(DAYS[12])),
    ]


def _m4(w: MetaWorld) -> None:
    w.class_rows = [class_row(s) for s in w.segs if s.symbol != "BBB"]


def _m5(w: MetaWorld) -> None:
    w.segs = [
        with_fields(s, cls="unknown", rule="none") if s.symbol == "BBB" else s for s in w.segs
    ]


def _m9(w: MetaWorld) -> None:
    # tests/store/test_audit_m9.py shape: BBB tagged etf while its CIK files 10-Qs
    from tests.store.test_audit_shares import fact

    w.segs = [
        with_fields(s, cls="etf", rule="lookup_sec_funds") if s.symbol == "BBB" else s
        for s in w.segs
    ]
    w.shares = [fact("BBB", date(2019, 12, 27), 2e8)]


def _m6(w: MetaWorld) -> None:
    w.conflicts = [conflict("dup_unlinked", ["AAA", "QQQ"], [DAYS[0], DAYS[1]], "2 identical days")]


def _m7(w: MetaWorld) -> None:
    copies_world(w, old_days=20, copied=10)


def _m8(w: MetaWorld) -> None:
    w.conflicts = [
        conflict(
            "cik_switch_discontinuous",
            ["PPP"],
            [DAYS[3], DAYS[4], DAYS[4]],
            "cik 1 -> 2, close None -> 5",
            resolution="new_security",
        )
    ]


def _i1(w: MetaWorld) -> None:
    w.bars += [bar("ORPH", d, close=5.0, volume=2_000 + i) for i, d in enumerate(DAYS[:5])]


def _i2(w: MetaWorld) -> None:
    aaa = series("AAA", DAYS, 100.0, 30_000)
    for d in DAYS[2:5]:
        w.bars = replace(w.bars, "BBB", d, {**find(aaa, "AAA", d), "symbol": "BBB"})


def _i3(w: MetaWorld) -> None:
    first = DAYS[5]
    w.segs.append(Seg(SID_AAA, "AAAX", first, None, known(first)))
    w.bars.append(bar("AAAX", DAYS[7], close=90.0, volume=7_000))


def _i4(w: MetaWorld) -> None:
    lac_world(w)


def _i5(w: MetaWorld) -> None:
    snap_a, snap_b = DAYS[7], DAYS[21]
    w.sec = [
        (snap, sym, 9999 if (snap == snap_a and sym == "BBB") else cik)
        for snap in (sm.SEC_HISTORY_START, snap_a, snap_b)
        for sym, cik in (("AAA", 1001), ("BBB", 1002), ("CCC", 1003))
    ]


def _i6(w: MetaWorld) -> None:
    for d in DAYS[-5:]:
        w.bars = replace(w.bars, "CCC", d, flat("CCC", d, 10.0))


def _i7(w: MetaWorld) -> None:
    dead = DAYS[20]
    for symbol, close in (("AAA", 100.0), ("BBB", 60.0), ("CCC", 10.0)):
        w.bars = replace(w.bars, symbol, dead, flat(symbol, dead, close))


def _i8(w: MetaWorld) -> None:
    gap = DAYS[4:15]  # eleven sessions
    w.bars = drop(w.bars, "CCC", gap) + [flat("CCC", d, 10.0) for d in gap]


def _i9(w: MetaWorld) -> None:
    rename_world(w, old_close=20.0, new_close=10.0)


def _i10(w: MetaWorld) -> None:
    w.exchange = [("INBX", 1739614, "NASDAQ", THREE)]  # listed three times, no bars at all


def _i12(w: MetaWorld) -> None:
    for symbol in ("AAA", "BBB", "CCC"):  # a mid-window NYSE session with no bar at all
        w.bars = drop(w.bars, symbol, [DAYS[10]])


def _s1(w: MetaWorld) -> None:
    w.splits = [split("AAA", date(2020, 1, 11), 1.0, 2.0)]  # a Saturday: no bar that day


def _s2(w: MetaWorld) -> None:
    split_world(w, factor_applied=False)


def _s3(w: MetaWorld) -> None:
    spike_world(w)


def _s4(w: MetaWorld) -> None:
    jump_world(w, before=5.0, after=20.0)


def _s5(w: MetaWorld) -> None:
    # the MDRR 2024-07-03 shape (tests/store/test_split_events.py): two factors on one day
    w.splits = [split("AAA", DAYS[10], 10.0, 1.0), split("AAA", DAYS[10], 1.0, 5.0)]


def _d1(w: MetaWorld) -> None:
    # the FMX 2026-04-22 shape (tests/store/test_audit_invariants.py): one payout stored twice,
    # the second copy with one more digit
    w.dividends = [dividend("AAA", DAYS[5], 1.039579), dividend("AAA", DAYS[5], 1.0395792)]


def _n1(w: MetaWorld) -> None:
    w.name_changes = [rename("CCC", "CCC3", DAYS[-1] + timedelta(days=31))]


def _n2(w: MetaWorld) -> None:
    w.name_changes = [rename("ROLD", "RNEW", DAYS[10])]
    w.bars += series("ROLD", DAYS[:13], 20.0, 3_000)
    w.bars += series("RNEW", DAYS[11:], 20.0, 3_000, start=11)


def _n3(w: MetaWorld) -> None:
    aaa = series("AAA", DAYS, 100.0, 30_000)
    for d in DAYS[:5]:  # 5 of BBB's 25 days: 0.2, inside the "partly identical" band
        w.bars = replace(w.bars, "BBB", d, {**find(aaa, "AAA", d), "symbol": "BBB"})


def _c1(w: MetaWorld) -> None:
    w.bars.append(bar("AAA", date(2019, 12, 31), close=100.0, volume=29_000))


def _c2(w: MetaWorld) -> None:
    w.bars = listed_after(w.bars, "CCC", 13)  # 11 sessions behind the store's end


def _s6(w: MetaWorld) -> None:
    # tests/store/test_audit_spinoffs.py shape: AAA keeps trading at 100 after handing out a
    # child worth 40 per share (+40%)
    ex = DAYS[10]
    w.bars += series("KID", DAYS[10:], 40.0, 7_000, start=10)
    w.segs = [*w.segs, Seg(SID_X1, "KID", valid_from=ex, available_at=known(ex))]
    w.others = [spin_off("AAA", "KID", ex, 1.0)]


def _f1(w: MetaWorld) -> None:
    # a tier common stock on a tier night with no share count at all (no shares file)
    w.tier = pa.Table.from_pylist(
        [
            {
                "as_of": known(date(2020, 2, 1)),
                "security_id": SID_AAA,
                "symbol": "AAA",
                "class": "common",
                "adv_usd": 1e8,
                "sessions_traded": 20,
            }
        ],
        schema=LIQUID_SCHEMA,
    )


def _f2(w: MetaWorld) -> None:
    # a tier common stock whose count (1e3 shares at 100) gives a cap below a day's volume
    from tests.store.test_audit_shares import fact

    _f1(w)
    w.shares = [fact("AAA", date(2019, 12, 27), 1e3)]


BROKEN: dict[str, Callable[[MetaWorld], None]] = {
    "P1": _p1,
    "P2": _p2,
    "P3": _p3,
    "P4": _p4,
    "P5": _p5,
    "P6": _p6,
    "M1": _m1,
    "M2": _m2,
    "M3": _m3,
    "M4": _m4,
    "M5": _m5,
    "M6": _m6,
    "M7": _m7,
    "M8": _m8,
    "M9": _m9,
    "I1": _i1,
    "I2": _i2,
    "I3": _i3,
    "I4": _i4,
    "I5": _i5,
    "I6": _i6,
    "I7": _i7,
    "I8": _i8,
    "I9": _i9,
    "I10": _i10,
    "I12": _i12,
    "S1": _s1,
    "S2": _s2,
    "S3": _s3,
    "S4": _s4,
    "S5": _s5,
    "S6": _s6,
    "D1": _d1,
    "N1": _n1,
    "N2": _n2,
    "N3": _n3,
    "C1": _c1,
    "C2": _c2,
    "F1": _f1,
    "F2": _f2,
}

# the clean world's per-year rows (pinned in tests/store/test_audit.py as well)
CLEAN_PER_YEAR: dict[str, tuple[tuple[str, ...], list[tuple[Any, ...]]]] = {
    "C1": (("key", "bars", "symbols", "liquid_symbols"), [("2020", 75, 3, 2)]),
    "I6": (("key", "untraded", "bars"), [("2020", 0, 75)]),
}


# =============================================================================================
# fixtures and helpers
# =============================================================================================


@pytest.fixture
def con() -> Iterator[duckdb.DuckDBPyConnection]:
    connection = duckdb.connect()
    yield connection
    connection.close()


@pytest.fixture
def world(tmp_path: Path) -> MetaWorld:
    return MetaWorld(market_root(tmp_path))


def run_one(con: duckdb.DuckDBPyConnection, world: MetaWorld, check_id: str) -> CheckResult:
    world.write()
    [result] = run_audit(
        con,
        world.root,
        as_of=LATE,
        liquid_ids=world.liquid_ids,
        checks=[check_id],
        liquid_tier=world.tier,
    )
    return result


def projected(result: CheckResult, columns: tuple[str, ...]) -> list[tuple[Any, ...]]:
    return [tuple(r[c] for c in columns) for r in rows(result)]


# =============================================================================================
# the registry
# =============================================================================================


def test_every_check_has_a_breaking_row() -> None:
    assert set(BROKEN) == set(CHECKS)


def test_every_master_rule_has_a_breaking_table() -> None:
    assert set(MASTER_BROKEN) == set(sm._MASTER_RULES)


def test_clean_world_base_is_the_audit_base_world(world: MetaWorld) -> None:
    # fixture sanity: the meta-test's clean world differs from the per-check tests' clean world
    # only by the C2 listing
    assert world.bars == base_bars()
    assert world.assets == [(SNAP_1, LISTED)] and world.exchange == []
    assert {a.symbol for a in LISTED} == {"AAA", "BBB", "CCC"}


# =============================================================================================
# every check: clean world -> nothing; its mutation -> at least one counting row
# =============================================================================================


@pytest.mark.parametrize("check_id", list(CHECKS))
def test_clean_world_does_not_trip_the_check(
    con: duckdb.DuckDBPyConnection, world: MetaWorld, check_id: str
) -> None:
    result = run_one(con, world, check_id)

    if check_id in INFO_PER_YEAR:
        columns, expected = CLEAN_PER_YEAR[check_id]
        assert projected(result, columns) == expected
    else:
        assert result.total == 0, rows(result)
    assert result.fails("liquid") is False and result.fails("all") is False


@pytest.mark.parametrize("check_id", list(CHECKS))
def test_breaking_mutation_trips_the_check(
    con: duckdb.DuckDBPyConnection, world: MetaWorld, check_id: str
) -> None:
    BROKEN[check_id](world)

    result = run_one(con, world, check_id)

    if check_id in INFO_PER_YEAR:
        columns, clean = CLEAN_PER_YEAR[check_id]
        assert projected(result, columns) != clean
    else:
        assert result.total >= 1, f"{check_id}: the breaking mutation produced no row"
    gate = CHECKS[check_id].gate
    if gate is not None:
        assert result.fails(gate) is True, (check_id, gate, rows(result))


def test_breaking_mutations_leave_other_worlds_alone(
    con: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    # each mutation acts on its own world; a mutation that leaked into the shared defaults
    # (a mutable default list edited in place) would make every later clean run dirty
    for check_id, mutate in BROKEN.items():
        mutate(MetaWorld(market_root(tmp_path / check_id)))
    clean = MetaWorld(market_root(tmp_path / "clean"))
    assert clean.bars == base_bars() and clean.conflicts is None and clean.splits == []
    assert run_one(con, clean, "P1").total == 0


# =============================================================================================
# check_master: one broken table per rule
# =============================================================================================

# rule (the key of _MASTER_RULES, quoted in the error) -> broken table of
# tests/ingest/test_security_master.py::cm_violation
MASTER_BROKEN: dict[str, str] = {
    "null available_at": "null_available_at",
    "valid_to without end_available_at or the reverse": "closed_without_end",
    "valid_to not after valid_from": "zero_length",
    "end_available_at before 00:00 New York of valid_to": "end_known_before_its_day",
    "duplicate segment keys": "duplicate_key",
    "overlapping segments of one symbol": "overlap_two_securities",
    "adjacent segment knowable before the previous one's end": "adjacent_instants_disagree",
    "cik without cik_available_at or the reverse": "cik_without_known",
    "cik_available_at before available_at": "cik_known_before_segment",
}


def test_check_master_accepts_the_good_table() -> None:
    assert sm.check_master(master_table(cm_good_rows())) is None


@pytest.mark.parametrize("rule", list(MASTER_BROKEN))
def test_broken_master_names_its_rule(rule: str) -> None:
    with pytest.raises(ValueError, match=re.escape(rule)):
        sm.check_master(master_table(cm_violation(MASTER_BROKEN[rule])))
