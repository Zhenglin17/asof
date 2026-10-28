"""Guards against store invariants no audit check counted, so a reader trips over them silently.
Covered here: a NaN / infinite price or a negative volume passing the impossible-price check (P2);
a session calendar that disagrees with NYSE (I12); one
dividend payout stored twice with its rate rounded differently, which a reader would pay twice
(D1). Every check is also run at an ``as_of`` before some of its data exists, and that data must
not show.

Target API (asof.store.audit; implementation pending):

    P2 reasons, first match wins: 'not_finite' (any of open/high/low/close NaN or +/-inf),
        'negative_volume' (volume < 0), then the existing four
    CHECKS["I12"] = CheckSpec("I12", "session calendar disagrees with NYSE", "all", DATE)
        rows: day, reason in {'missing_session', 'session_on_holiday'}
    CHECKS["D1"] = CheckSpec("D1", "dividend records differing only by rounding", None, SYM:DATE)
        rows: symbol, ex_date, rates (list)

Fake data provenance:
- The fixture calendar (tests/store/audit_world.py DAYS) is weekdays minus the real NYSE
  holidays, 2020-01-02 .. 2020-02-06; 2020-01-20 is Martin Luther King Jr. Day. I12 compares
  against exchange_calendars' "XNYS" (docs; checked 2026-10-28: its sessions over that span equal
  DAYS and 2020-01-20 is not one).
- Dividend rows: ``available_at`` is 00:00 New York of the ex-date (ingest.corporate_actions;
  the convention of tests/evolve/test_benchmark.py). The rates below are real, read off the live
  store: FMX 2026-04-22 holds 1.039579, 1.0395792, 0.597937, 0.597936 (two payouts, each stored
  twice with a different rounding; relative gaps 1.9e-7 and 1.7e-6); CWSRF 2024-03-27 holds
  0.0376898 and 0.0377261 (relative gap 9.6e-4: two different figures, not a rounding); QQQ
  2022-09-19 holds 0.51856 twice, identical in every column (``visible_dividends`` already reads
  it once); SATA 2026-08-11 holds 0.0516 twice, sub_type null vs 'return_of_capital'.
- The 1e-5 relative tolerance is a design choice that separates those real shapes; it is not
  a property of the feed.
- One bar owned by two securities needs no check here: ``visible_master`` refuses overlapping
  segments of one symbol on every read, so no reader (the audit included) ever sees such bars.
- ASSUMPTION (P2): a NaN or infinite price, or a negative volume, can reach a stored bar (a
  feed float, an import parse). Not observed. Before the new reasons (measured 2026-10-28 on
  these fixtures): a NaN close or open was reported as 'high_below_body' (DuckDB orders NaN
  above every number), an infinite high passed unreported, a -inf low read as 'nonpositive',
  and a negative volume passed unreported.
- ASSUMPTION (I12): the set of stored session dates equals the XNYS sessions over the stored
  span. Unscheduled closures (national days of mourning) are where the two may disagree.
- ASSUMPTION (D1): two rows of one (symbol, ex_date) within 1e-5 of each other are one payout,
  never two payouts of (almost) the same amount.

Real-data checks for the assumptions (``asof market audit`` on the real store):
- P2: rows with reason 'not_finite' or 'negative_volume' -- expected 0; any row is read by hand.
- I12: 0 rows; any row (e.g. an unscheduled closure) is read by a human before an exception
  is added.
- D1: FMX:2026-04-22 and SATA:2026-08-11 listed; CWSRF:2024-03-27 and QQQ:2022-09-19 absent.
"""

import math
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pytest

from asof.store.audit import CHECKS
from asof.store.market import market_root
from tests.store.audit_world import (
    DAYS,
    LATE,
    World,
    bar,
    base_bars,
    dividend,
    drop,
    flat,
    iso,
    keys,
    replace,
    rows,
)
from tests.store.test_market import avail, known

NOW = datetime(2026, 10, 28, tzinfo=UTC)  # every dividend fixture below is knowable by then
MLK = date(2020, 1, 20)  # NYSE holiday inside DAYS
SATURDAY = date(2020, 1, 11)


@pytest.fixture
def con() -> Iterator[duckdb.DuckDBPyConnection]:
    connection = duckdb.connect()
    yield connection
    connection.close()


@pytest.fixture
def world(tmp_path: Path) -> World:
    return World(market_root(tmp_path))


def drop_day(world: World, day: date) -> None:
    for symbol in ("AAA", "BBB", "CCC"):
        world.bars = drop(world.bars, symbol, [day])


def test_fixture_calendar_matches_xnys() -> None:
    import exchange_calendars as xc

    assert DAYS[10] == date(2020, 1, 16) and DAYS[11] == date(2020, 1, 17)
    assert DAYS[12] == date(2020, 1, 21) and MLK not in DAYS and SATURDAY not in DAYS
    assert DAYS[-1] == date(2020, 2, 6)
    cal = xc.get_calendar("XNYS", start=DAYS[0].isoformat(), end=DAYS[-1].isoformat())
    sessions = [d.date() for d in cal.sessions_in_range(DAYS[0], DAYS[-1])]
    assert sessions == DAYS
    assert not cal.is_session(MLK)


def test_registry_specs() -> None:
    assert (CHECKS["I12"].gate, CHECKS["D1"].gate) == ("all", None)


# =============================================================================================
# P2: non-finite prices and negative volume
# =============================================================================================


def clean_bar(day: date, **fields: Any) -> dict[str, Any]:
    base: dict[str, Any] = {"close": 100.0, "open": 100.0, "high": 101.0, "low": 99.0}
    base["volume"] = 30_000 + DAYS.index(day)
    base.update(fields)
    return bar("AAA", day, **base)


@pytest.mark.parametrize(
    "fields",
    [
        {"close": math.nan},
        {"open": math.nan},
        {"high": math.inf},
        {"low": -math.inf},
        {"close": None, "vwap": 100.0},  # NULL slips through every comparison (least() skips it)
        {"vwap": math.nan},  # the liquid tier's dollar volume is vwap x volume
    ],
    ids=["nan_close", "nan_open", "inf_high", "neg_inf_low", "null_close", "nan_vwap"],
)
def test_p2_non_finite_price_is_reported(
    con: duckdb.DuckDBPyConnection, world: World, fields: dict[str, Any]
) -> None:
    world.bars = replace(base_bars(), "AAA", DAYS[3], clean_bar(DAYS[3], **fields))

    result = world.audit(con, checks=["P2"])["P2"]

    assert [(r["key"], r["reason"]) for r in rows(result)] == [
        (f"AAA:{iso(DAYS[3])}", "not_finite")
    ]
    assert result.fails("all") is True


def test_p2_nan_close_and_inf_high_one_row_per_bar(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    world.bars = replace(base_bars(), "AAA", DAYS[3], clean_bar(DAYS[3], close=math.nan))
    world.bars = replace(world.bars, "AAA", DAYS[4], clean_bar(DAYS[4], high=math.inf))
    # both defects on one bar: still one row
    world.bars = replace(
        world.bars, "AAA", DAYS[5], clean_bar(DAYS[5], close=math.nan, high=math.inf)
    )

    result = world.audit(con, checks=["P2"])["P2"]

    assert [(r["key"], r["reason"]) for r in rows(result)] == [
        (f"AAA:{iso(d)}", "not_finite") for d in DAYS[3:6]
    ]


def test_p2_negative_volume(con: duckdb.DuckDBPyConnection, world: World) -> None:
    world.bars = replace(base_bars(), "AAA", DAYS[3], clean_bar(DAYS[3], volume=-5))

    result = world.audit(con, checks=["P2"])["P2"]

    assert [(r["key"], r["reason"]) for r in rows(result)] == [
        (f"AAA:{iso(DAYS[3])}", "negative_volume")
    ]


@pytest.mark.parametrize(
    ("fields", "reason"),
    [
        ({"close": math.nan, "low": 0.0}, "not_finite"),  # before 'nonpositive'
        ({"high": math.inf, "low": 102.0}, "not_finite"),  # before 'high_below_low' etc.
        ({"close": math.nan, "volume": -5}, "not_finite"),  # before 'negative_volume'
        ({"volume": -5, "low": 0.0}, "negative_volume"),  # before 'nonpositive'
        ({"volume": -5, "high": 99.0, "low": 101.0}, "negative_volume"),
    ],
    ids=["nan_and_zero", "inf_and_inverted", "nan_and_negvol", "negvol_and_zero", "negvol_inv"],
)
def test_p2_new_reasons_are_checked_first(
    con: duckdb.DuckDBPyConnection, world: World, fields: dict[str, Any], reason: str
) -> None:
    world.bars = replace(base_bars(), "AAA", DAYS[3], clean_bar(DAYS[3], **fields))

    [row] = rows(world.audit(con, checks=["P2"])["P2"])

    assert row["reason"] == reason


def test_p2_volume_zero_is_not_negative(con: duckdb.DuckDBPyConnection, world: World) -> None:
    # boundary: a flat volume-0 placeholder (real: VTIQ) is a valid bar
    world.bars = replace(base_bars(), "CCC", DAYS[3], flat("CCC", DAYS[3], 10.0))

    assert rows(world.audit(con, checks=["P2"])["P2"]) == []


def test_p2_non_finite_bar_not_yet_available_is_invisible(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    # leakage: the NaN bar of the last session exists on disk but is not knowable before its
    # 20:00 New York of that day
    last = DAYS[-1]
    world.bars = replace(base_bars(), "AAA", last, clean_bar(last, close=math.nan))

    before = world.audit(con, as_of=avail(DAYS[-2]), checks=["P2"])["P2"]
    after = world.audit(con, as_of=avail(last), checks=["P2"])["P2"]

    assert rows(before) == []
    assert [(r["key"], r["reason"]) for r in rows(after)] == [(f"AAA:{iso(last)}", "not_finite")]


# =============================================================================================
# I12: session calendar disagrees with NYSE
# =============================================================================================


def i12(result: Any) -> list[tuple[str, date, str]]:
    return [(r["key"], r["day"], r["reason"]) for r in rows(result)]


def test_i12_clean_world(con: duckdb.DuckDBPyConnection, world: World) -> None:
    result = world.audit(con, checks=["I12"])["I12"]

    assert {"day", "reason"} <= set(result.rows.schema.names)
    assert rows(result) == []


def test_i12_session_with_no_bar_at_all(con: duckdb.DuckDBPyConnection, world: World) -> None:
    drop_day(world, DAYS[10])

    result = world.audit(con, checks=["I12"])["I12"]

    assert i12(result) == [(iso(DAYS[10]), DAYS[10], "missing_session")]
    assert result.fails("all") is True


def test_i12_session_some_symbols_lack_is_not_missing(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    # boundary: only a session with no bar at all counts; CCC alone keeps DAYS[10]
    for symbol in ("AAA", "BBB"):
        world.bars = drop(world.bars, symbol, [DAYS[10]])

    assert rows(world.audit(con, checks=["I12"])["I12"]) == []


def test_i12_session_of_placeholders_only_is_not_missing(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    # boundary: the raw view includes volume-0 bars, so the session exists (I7's business)
    for symbol, close in (("AAA", 100.0), ("BBB", 60.0), ("CCC", 10.0)):
        world.bars = replace(world.bars, symbol, DAYS[10], flat(symbol, DAYS[10], close))

    assert rows(world.audit(con, checks=["I12"])["I12"]) == []


@pytest.mark.parametrize("day", [MLK, SATURDAY], ids=["holiday", "weekend"])
def test_i12_bar_on_a_day_nyse_was_closed(
    con: duckdb.DuckDBPyConnection, world: World, day: date
) -> None:
    world.bars = [*world.bars, bar("AAA", day, close=100.0, volume=29_000)]

    result = world.audit(con, checks=["I12"])["I12"]

    assert i12(result) == [(iso(day), day, "session_on_holiday")]
    assert result.fails("all") is True


@pytest.mark.parametrize("edge", [0, -1], ids=["first", "last"])
def test_i12_days_outside_the_stored_span_are_not_reported(
    con: duckdb.DuckDBPyConnection, world: World, edge: int
) -> None:
    # boundary: the window runs from the first to the last stored session; a missing last day
    # is stale data (C2), not a calendar hole
    drop_day(world, DAYS[edge])

    assert rows(world.audit(con, checks=["I12"])["I12"]) == []


def test_i12_last_two_days_missing_are_not_reported(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    for d in DAYS[-2:]:
        drop_day(world, d)

    assert rows(world.audit(con, checks=["I12"])["I12"]) == []


def test_i12_future_sessions_are_not_missing_at_an_earlier_as_of(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    # leakage: at the close of DAYS[12] the store knows DAYS[0..12] only; the later sessions are
    # on disk but neither visible nor missing
    assert rows(world.audit(con, as_of=avail(DAYS[12]), checks=["I12"])["I12"]) == []


def test_i12_holiday_bar_not_yet_available_is_invisible(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    # leakage: a holiday bar filed later than as_of does not exist yet at as_of
    later = LATE + timedelta(days=1)
    world.bars = [
        *world.bars,
        bar("AAA", MLK, close=100.0, volume=29_000, available_at=later, fetched_at=later),
    ]

    before = world.audit(con, as_of=LATE, checks=["I12"])["I12"]
    after = world.audit(con, as_of=later, checks=["I12"])["I12"]

    assert rows(before) == []
    assert i12(after) == [(iso(MLK), MLK, "session_on_holiday")]


def test_i12_session_whose_bars_arrive_later_is_missing_until_then(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    # leakage, the other way: bars on disk that become knowable only after as_of do not fill
    # the hole at as_of
    later = LATE + timedelta(days=1)
    day = DAYS[10]
    world.bars = [
        {**b, "available_at": later, "fetched_at": later} if b["session_date"] == day else b
        for b in world.bars
    ]

    before = world.audit(con, as_of=LATE, checks=["I12"])["I12"]
    after = world.audit(con, as_of=later, checks=["I12"])["I12"]

    assert i12(before) == [(iso(day), day, "missing_session")]
    assert rows(after) == []


# =============================================================================================
# D1: dividend records differing only by rounding
# =============================================================================================

FMX_DAY = date(2026, 4, 22)
FMX = [
    dividend("FMX", FMX_DAY, 1.039579),
    dividend("FMX", FMX_DAY, 1.0395792),
    dividend("FMX", FMX_DAY, 0.597937),
    dividend("FMX", FMX_DAY, 0.597936),
]


def d1(con: duckdb.DuckDBPyConnection, world: World, as_of: datetime = NOW) -> Any:
    return world.audit(con, as_of=as_of, checks=["D1"])["D1"]


def test_d1_fmx_rounding_twins_are_reported(con: duckdb.DuckDBPyConnection, world: World) -> None:
    world.dividends = list(FMX)

    result = d1(con, world)

    assert {"symbol", "ex_date", "rates"} <= set(result.rows.schema.names)
    [row] = rows(result)
    assert row["key"] == "FMX:2026-04-22"
    assert (row["symbol"], row["ex_date"]) == ("FMX", FMX_DAY)
    assert sorted(row["rates"]) == sorted(r["rate"] for r in FMX)
    assert row["in_liquid"] is False
    assert result.fails("liquid") is False and result.fails("all") is False


def test_d1_clean_store_has_no_rows(con: duckdb.DuckDBPyConnection, world: World) -> None:
    assert rows(d1(con, world)) == []  # no dividends file at all
    world.dividends = [dividend("AAA", DAYS[5], 0.5), dividend("AAA", DAYS[20], 0.5)]
    assert rows(d1(con, world)) == []  # the same rate on two ex-dates is two payouts


def test_d1_cwsrf_two_different_figures_are_not_reported(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    day = date(2024, 3, 27)
    world.dividends = [dividend("CWSRF", day, 0.0376898), dividend("CWSRF", day, 0.0377261)]

    assert rows(d1(con, world)) == []


def test_d1_qqq_rows_identical_in_every_column_are_not_reported(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    day = date(2022, 9, 19)
    world.dividends = [dividend("QQQ", day, 0.51856), dividend("QQQ", day, 0.51856)]

    assert rows(d1(con, world)) == []


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ({"sub_type": None}, {"sub_type": "return_of_capital"}),  # SATA 2026-08-11 (real)
        ({"special": False}, {"special": True}),
    ],
    ids=["sub_type", "special"],
)
def test_d1_equal_rates_differing_elsewhere_are_reported(
    con: duckdb.DuckDBPyConnection,
    world: World,
    first: dict[str, Any],
    second: dict[str, Any],
) -> None:
    day = date(2026, 8, 11)
    world.dividends = [
        dividend("SATA", day, 0.0516, **first),
        dividend("SATA", day, 0.0516, **second),
    ]

    [row] = rows(d1(con, world))

    assert row["key"] == "SATA:2026-08-11"
    assert sorted(row["rates"]) == [0.0516, 0.0516]


@pytest.mark.parametrize(
    ("other", "reported"),
    [(1.000009, True), (1.000011, False)],
    ids=["just_inside", "just_outside"],
)
def test_d1_tolerance_edge(
    con: duckdb.DuckDBPyConnection, world: World, other: float, reported: bool
) -> None:
    # boundary: |a - b| <= 1e-5 * max(a, b)
    world.dividends = [dividend("AAA", DAYS[5], 1.0), dividend("AAA", DAYS[5], other)]

    assert keys(d1(con, world)) == ([f"AAA:{iso(DAYS[5])}"] if reported else [])


def test_d1_other_symbol_same_day_and_rate_is_not_a_twin(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    world.dividends = [dividend("AAA", DAYS[5], 0.5), dividend("BBB", DAYS[5], 0.5)]

    assert rows(d1(con, world)) == []


def test_d1_liquid_flag_follows_the_symbol(con: duckdb.DuckDBPyConnection, world: World) -> None:
    world.dividends = [dividend("AAA", DAYS[5], 1.039579), dividend("AAA", DAYS[5], 1.0395792)]

    result = d1(con, world)

    [row] = rows(result)
    assert (row["key"], row["in_liquid"]) == (f"AAA:{iso(DAYS[5])}", True)
    assert result.fails("liquid") is False  # information, never gated


def test_d1_twin_not_yet_known_is_invisible(con: duckdb.DuckDBPyConnection, world: World) -> None:
    # leakage: the second copy is on disk but filed after as_of
    filed = known(date(2026, 5, 1))
    world.dividends = [
        dividend("FMX", FMX_DAY, 1.039579),
        dividend("FMX", FMX_DAY, 1.0395792, available_at=filed),
    ]

    before = d1(con, world, as_of=datetime(2026, 4, 25, tzinfo=UTC))
    after = d1(con, world, as_of=filed)

    assert rows(before) == []
    assert keys(after) == ["FMX:2026-04-22"]


@pytest.mark.parametrize(
    ("day", "a", "b", "reported"),
    [
        # real: 6 and 7 decimals of one small payout, relative gap 8.1e-5 but absolute 1e-6
        (date(2025, 12, 22), 0.012354, 0.012355, True),
        # real: two figures, absolute gap 8e-6 (relative 3.2e-4)
        (date(2020, 10, 29), 0.0253564, 0.0253484, False),
    ],
    ids=["abev_small_rate_rounding", "ddhrf_two_figures"],
)
def test_d1_small_rates_use_the_absolute_gap(
    con: duckdb.DuckDBPyConnection, world: World, day: date, a: float, b: float, reported: bool
) -> None:
    world.dividends = [dividend("ABEV", day, a), dividend("ABEV", day, b)]

    assert keys(d1(con, world)) == ([f"ABEV:{iso(day)}"] if reported else [])
