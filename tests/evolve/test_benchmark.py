"""Guards against benchmark curves that flatter or punish a strategy for the wrong reason: a split
read as a 50% crash, a dividend left out (an index buy-and-hold without its payouts), a basket
whose members were picked with the rebalance day's own volume or with later bars, an ETF sitting
in a "stocks" basket, a member that stops trading dropping to zero, and corporate actions after
the window leaking into it.

API under test (asof.evolve.benchmark, new module):

    Rebalance(session: date, members: tuple[tuple[str, str], ...])   (security_id, symbol)
    BenchmarkCurve(label, sessions, equity, rebalances=())
        total_return = equity[-1] - 1
        annualized_return = (1 + total_return) ** (252 / len(sessions)) - 1
        max_drawdown = largest peak-to-trough fall, the 1.0 start counted as a peak; >= 0
    buy_and_hold(con, root, symbol, start, end) -> BenchmarkCurve
    basket(con, root, start, end, *, size, lookback_sessions) -> BenchmarkCurve
    benchmarks(con, root, protocol, start, end) -> list[BenchmarkCurve]

Fake data provenance (all prices and volumes are made up and chosen for hand arithmetic):

- Calendar = the sessions that have bars: weekdays 2025-01-02 .. 2025-03-31 minus 2025-01-09
  (national day of mourning), 2025-01-20 and 2025-02-17, the real NYSE closures of the period.
- A daily bar is visible from 20:00 New York of its session (asof.ingest.bars); a split or
  dividend record from 00:00 New York of its ex-date (asof.ingest.corporate_actions).
- Split ratio = new_rate / old_rate (= ``factor``), as the replay applies it (real: NVDA
  2024-06-10 old 1, new 10). Bars on and after the ex-date carry post-split prices.
- Dividend ``rate`` is cash per share (real: SPY 2025-03-21 1.695528, QQQ 2025-03-24 0.71571).
- Dollar volume = vwap * volume with vwap = close, as liquid_securities computes it.

Assumptions these fixtures cannot settle; check them on the real store:
- Stored daily bars are unadjusted, so the ex-date bar is the first post-split price. Check:
  NVDA 2024-06-07 close / 2024-06-10 open is close to 10 (the audit's S2 split check relies on
  the same).
- Dividend rows are cash per share on ex_date and every SPY / QQQ quarterly payout is present.
  Check: four rows per year for SPY and QQQ in corporate_actions/dividends.parquet over
  2020-2026, each ex_date a session with a SPY / QQQ bar.
- Exact duplicate dividend rows exist (real: QQQ 2022-09-19 rate 0.51856 appears twice;
  3,849 (symbol, ex_date) groups have more than one row, 604 of them one distinct rate, 370
  include a special). ASSUMPTION: a row identical in every column is one payment, not two;
  rows that differ (a regular plus a special) are separate payments. Check: the QQQ 2022
  total-return curve against a published total return for that year.
- Standard case: SPY buy-and-hold with dividends over 2025-07-01 .. 2026-06-30 lands within a
  fraction of a percent of SPY's published total return for the same dates.
"""

from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from asof.evolve.benchmark import BenchmarkCurve, Rebalance, basket, benchmarks, buy_and_hold
from asof.ingest.alpaca import Bar
from asof.ingest.corporate_actions import DIVIDEND_SCHEMA
from asof.store.market import market_root
from asof.store.universe import liquid_securities
from tests.evolve.protocols import protocol
from tests.ingest.fakes import daily
from tests.store.test_market import write_splits
from tests.store.test_universe import Seg, known, store_daily, write_symbols

HOLIDAYS = {date(2025, 1, 9), date(2025, 1, 20), date(2025, 2, 17)}
SESSIONS: list[date] = [
    d.date() for d in pd.bdate_range("2025-01-02", "2025-03-31") if d.date() not in HOLIDAYS
]
FEB_FIRST = date(2025, 2, 3)  # first session of February (Feb 1 is a Saturday)
MAR_FIRST = date(2025, 3, 3)  # first session of March (Mar 1 is a Saturday)
LAST = SESSIONS[-1]  # 2025-03-31
FEB_LAST = date(2025, 2, 28)
LATE_FEB = (date(2025, 2, 24), FEB_LAST)  # the 5 sessions March's members are chosen from
LOOKBACK = 5
M = 1_000_000

# Monthly close per stock (January, February, March). The move happens inside the first
# session of the month: its open is the previous month's price, its close the new one.
PRICES: dict[str, tuple[float, float, float]] = {
    "A": (10.0, 12.0, 13.0),
    "B": (20.0, 18.0, 18.0),
    "C": (30.0, 30.0, 36.0),
    "D": (40.0, 40.0, 36.0),
    "SPY": (100.0, 101.0, 103.0),
}
# A gaps up at March's open: last February close 12, March 3 open 13. A basket that sells at
# the previous close instead of the rebalance open would miss it.
OPEN_GAP: dict[tuple[str, date], float] = {("A", MAR_FIRST): 13.0}
# January dollar volume: A 400M > B 300M > C 150M > D 80M; SPY 1B (an ETF, never a member).
JAN_VOLUME = {"A": 40 * M, "B": 15 * M, "C": 5 * M, "D": 2 * M, "SPY": 10 * M}
# Feb 24-28: D 400M > C 300M > B 18M > A 12M. Rest of Feb and March: 1M shares each.
LATE_FEB_VOLUME = {"A": M, "B": M, "C": 10 * M, "D": 10 * M, "SPY": 10 * M}
SEGS = [
    Seg("S_A", "A"),
    Seg("S_B", "B"),
    Seg("S_C", "C"),
    Seg("S_D", "D"),
    Seg("S_SPY", "SPY", cls="etf"),
]

Ohlcv = tuple[float, float, int]  # open, close, volume


def bar(symbol: str, day: date, values: Ohlcv) -> Bar:
    o, c, v = values
    return daily(symbol, day, open=o, high=max(o, c), low=min(o, c), close=c, volume=v, vwap=c)


def default_ohlcv(symbol: str, day: date) -> Ohlcv:
    jan, feb, mar = PRICES[symbol]
    close = {1: jan, 2: feb, 3: mar}[day.month]
    if day == FEB_FIRST:
        open_ = jan
    elif day == MAR_FIRST:
        open_ = feb
    else:
        open_ = close
    open_ = OPEN_GAP.get((symbol, day), open_)
    if day.month == 1:
        volume = JAN_VOLUME[symbol]
    elif LATE_FEB[0] <= day <= LATE_FEB[1]:
        volume = LATE_FEB_VOLUME[symbol]
    else:
        volume = 10 * M if symbol == "SPY" else M
    return (open_, close, volume)


def write_dividends(root: Path, rows: Sequence[tuple[str, date, float]]) -> Path:
    """``(symbol, ex_date, rate)`` rows, available from 00:00 New York of the ex-date."""
    table = pa.table(
        {
            "symbol": pa.array([r[0] for r in rows], pa.string()),
            "ex_date": pa.array([r[1] for r in rows], pa.date32()),
            "rate": pa.array([r[2] for r in rows], pa.float64()),
            "special": pa.array([False for _ in rows], pa.bool_()),
            "sub_type": pa.array([None for _ in rows], pa.string()),
            "available_at": pa.array(
                [known(r[1]) for r in rows], DIVIDEND_SCHEMA.field("available_at").type
            ),
        },
        schema=DIVIDEND_SCHEMA,
    )
    path = root / "corporate_actions" / "dividends.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


def build(
    data_dir: Path,
    bars: Sequence[Bar],
    segs: Sequence[Seg],
    *,
    splits: Sequence[tuple[str, date, float, float]] = (),
    dividends: Sequence[tuple[str, date, float]] = (),
) -> Path:
    root = market_root(data_dir)
    store_daily(root, bars)
    write_symbols(root, segs)
    if splits:
        write_splits(root, splits)
    if dividends:
        write_dividends(root, dividends)
    return root


def world(
    data_dir: Path,
    *,
    through: date = LAST,
    override: Mapping[tuple[str, date], Ohlcv | None] | None = None,
    splits: Sequence[tuple[str, date, float, float]] = (),
    dividends: Sequence[tuple[str, date, float]] = (),
) -> Path:
    """A, B, C, D (common) and SPY (etf) on every session through ``through``. ``override``
    replaces a bar; None removes it."""
    changes = override or {}
    bars = []
    for symbol in PRICES:
        for day in SESSIONS:
            if day > through:
                break
            values = changes.get((symbol, day), default_ohlcv(symbol, day))
            if values is not None:
                bars.append(bar(symbol, day, values))
    return build(data_dir, bars, SEGS, splits=splits, dividends=dividends)


def path_store(
    data_dir: Path,
    symbol: str,
    path: Sequence[tuple[date, float, float] | None],
    *,
    other: str = "OTHR",
    splits: Sequence[tuple[str, date, float, float]] = (),
    dividends: Sequence[tuple[str, date, float]] = (),
) -> Path:
    """``symbol`` follows ``path`` (date, open, close), None = no bar that session; ``other``
    trades flat on the first len(path) sessions of February so the calendar has no holes."""
    days = [d for d in SESSIONS if d >= FEB_FIRST][: len(path)]
    bars = [bar(other, d, (50.0, 50.0, M)) for d in days]
    bars += [bar(symbol, p[0], (p[1], p[2], M)) for p in path if p is not None]
    return build(
        data_dir,
        bars,
        [Seg(f"S_{symbol}", symbol, cls="etf"), Seg(f"S_{other}", other)],
        splits=splits,
        dividends=dividends,
    )


FEB = [d for d in SESSIONS if d.month == 2]  # 19 sessions, Feb 17 closed


def at(curve: BenchmarkCurve, day: date) -> float:
    return curve.equity[curve.sessions.index(day)]


def in_range(start: date, end: date) -> tuple[date, ...]:
    return tuple(d for d in SESSIONS if start <= d <= end)


# =============================================================================================
# normal path
# =============================================================================================


def test_buy_and_hold_without_actions_is_last_close_over_first_open_one_value_per_session(
    con: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    root = world(tmp_path)

    curve = buy_and_hold(con, root, "SPY", FEB_FIRST, LAST)

    # Bought at 100 (Feb 3 open), last close 103.
    assert curve.sessions == in_range(FEB_FIRST, LAST)
    assert len(curve.equity) == len(curve.sessions)
    assert curve.total_return == pytest.approx(103 / 100 - 1)
    assert at(curve, FEB_FIRST) == pytest.approx(1.01)
    assert curve.rebalances == ()
    assert "SPY" in curve.label


def test_max_drawdown_on_a_hand_made_path(con: duckdb.DuckDBPyConnection, tmp_path: Path) -> None:
    d = FEB[:4]
    root = path_store(
        tmp_path, "X", [(d[0], 100, 100), (d[1], 100, 120), (d[2], 120, 90), (d[3], 90, 110)]
    )

    curve = buy_and_hold(con, root, "X", d[0], d[3])

    assert curve.equity == pytest.approx((1.0, 1.2, 0.9, 1.1))
    assert curve.max_drawdown == pytest.approx(0.25)  # 1.2 -> 0.9
    assert curve.total_return == pytest.approx(0.1)


def test_annualized_return_equals_total_return_on_252_sessions_and_compounds_on_126() -> None:
    days = tuple(d.date() for d in pd.bdate_range("2021-01-04", periods=252))
    year = BenchmarkCurve("x", days, tuple(1.0 + 0.3 * i / 251 for i in range(252)))
    half = BenchmarkCurve("x", days[:126], tuple(1.0 + 0.1 * i / 125 for i in range(126)))

    assert year.total_return == pytest.approx(0.3)
    assert year.annualized_return == pytest.approx(0.3)
    assert half.annualized_return == pytest.approx(1.1**2 - 1)


def test_basket_over_two_months_picks_members_by_dollar_volume_and_weights_them_equally(
    con: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    root = world(tmp_path)

    curve = basket(con, root, FEB_FIRST, LAST, size=2, lookback_sessions=LOOKBACK)

    # Feb 3 (January volume): A, B. Mar 3 (Feb 24-28 volume): D, C. SPY's $1B never counts.
    assert curve.rebalances == (
        Rebalance(FEB_FIRST, (("S_A", "A"), ("S_B", "B"))),
        Rebalance(MAR_FIRST, (("S_D", "D"), ("S_C", "C"))),
    )
    assert curve.sessions == in_range(FEB_FIRST, LAST)
    # 0.5 in A at 10 and 0.5 in B at 20; Feb 3 close: 0.05 * 12 + 0.025 * 18 = 1.05.
    assert at(curve, FEB_FIRST) == pytest.approx(1.05)
    assert at(curve, FEB_LAST) == pytest.approx(1.05)
    # Mar 3 open: A sells at 13, B at 18 -> 1.10; 0.55 each into D at 40 and C at 30.
    # Close: 0.55 / 40 * 36 + 0.55 / 30 * 36 = 0.495 + 0.66 = 1.155.
    assert at(curve, MAR_FIRST) == pytest.approx(1.155)
    assert curve.total_return == pytest.approx(0.155)
    assert "2" in curve.label


def test_benchmarks_follows_the_protocol_one_buy_and_hold_per_symbol_then_the_basket(
    con: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    root = world(tmp_path)
    settings = protocol(
        benchmarks={
            "symbols": ["SPY", "A"],
            "basket": {"size": 2, "lookback_sessions": LOOKBACK, "rebalance": "monthly"},
        }
    )

    curves = benchmarks(con, root, settings, FEB_FIRST, LAST)

    assert curves == [
        buy_and_hold(con, root, "SPY", FEB_FIRST, LAST),
        buy_and_hold(con, root, "A", FEB_FIRST, LAST),
        basket(con, root, FEB_FIRST, LAST, size=2, lookback_sessions=LOOKBACK),
    ]
    assert len({c.label for c in curves}) == 3


# =============================================================================================
# boundaries
# =============================================================================================


def test_two_for_one_split_mid_hold_does_not_move_the_value(
    con: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    d = FEB[:4]
    root = path_store(
        tmp_path,
        "X",
        [(d[0], 100, 100), (d[1], 101, 102), (d[2], 51, 50), (d[3], 50, 55)],
        splits=[("X", d[2], 1.0, 2.0)],
    )

    curve = buy_and_hold(con, root, "X", d[0], d[3])

    # 0.01 shares, 0.02 from the ex-date's open: 1.0, 1.02, 0.02 * 50, 0.02 * 55.
    assert curve.equity == pytest.approx((1.0, 1.02, 1.0, 1.1))
    assert curve.max_drawdown == pytest.approx(1 - 1.0 / 1.02)


def test_dividend_is_paid_on_the_ex_date_and_reinvested_at_its_close(
    con: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    d = FEB[:3]
    root = path_store(
        tmp_path,
        "X",
        [(d[0], 100, 100), (d[1], 99, 98), (d[2], 98, 99)],
        dividends=[("X", d[1], 2.0)],
    )

    curve = buy_and_hold(con, root, "X", d[0], d[2])

    # 0.01 shares; ex-date cash 0.02 buys 0.02 / 98 more at the close: equity 0.98 + 0.02.
    shares = 0.01 + 0.02 / 98
    assert curve.equity == pytest.approx((1.0, 1.0, shares * 99))
    assert "dividend" in curve.label.lower()


def test_identical_dividend_rows_are_one_payment(
    con: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    # Real: QQQ 2022-09-19 rate 0.51856 is stored twice, identical in every column.
    d = FEB[:2]
    root = path_store(
        tmp_path,
        "X",
        [(d[0], 100, 100), (d[1], 100, 100)],
        dividends=[("X", d[1], 2.0), ("X", d[1], 2.0)],
    )

    curve = buy_and_hold(con, root, "X", d[0], d[1])

    assert curve.equity == pytest.approx((1.0, 1.02))


def test_buy_and_hold_carries_the_last_close_over_a_session_without_a_bar(
    con: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    d = FEB[:4]
    root = path_store(tmp_path, "X", [(d[0], 100, 110), (d[1], 110, 120), None, (d[3], 120, 90)])

    curve = buy_and_hold(con, root, "X", d[0], d[3])

    assert curve.sessions == tuple(d)
    assert curve.equity == pytest.approx((1.1, 1.2, 1.2, 0.9))


def test_drawdown_counts_the_one_dollar_start_as_a_peak(
    con: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    d = FEB[:2]
    root = path_store(tmp_path, "X", [(d[0], 100, 80), (d[1], 80, 90)])

    curve = buy_and_hold(con, root, "X", d[0], d[1])

    assert curve.equity == pytest.approx((0.8, 0.9))
    assert curve.max_drawdown == pytest.approx(0.2)


def test_single_session_window(con: duckdb.DuckDBPyConnection, tmp_path: Path) -> None:
    root = world(tmp_path)
    day = date(2025, 2, 12)

    held = buy_and_hold(con, root, "SPY", day, day)
    picked = basket(con, root, day, day, size=2, lookback_sessions=LOOKBACK)

    assert held.sessions == (day,)
    assert held.equity == pytest.approx((1.0,))  # SPY opens and closes at 101
    assert held.max_drawdown == 0.0
    assert picked.sessions == (day,)
    assert picked.rebalances == (Rebalance(day, (("S_D", "D"), ("S_C", "C"))),)
    assert picked.equity == pytest.approx((1.0,))


def test_empty_range_and_reversed_range_raise(
    con: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    root = world(tmp_path)
    saturday = date(2025, 2, 1)

    with pytest.raises(ValueError):
        buy_and_hold(con, root, "SPY", saturday, saturday)
    with pytest.raises(ValueError):
        basket(con, root, saturday, saturday, size=2, lookback_sessions=LOOKBACK)
    with pytest.raises(ValueError):
        buy_and_hold(con, root, "SPY", LAST, FEB_FIRST)
    with pytest.raises(ValueError):
        basket(con, root, LAST, FEB_FIRST, size=2, lookback_sessions=LOOKBACK)


def test_buy_and_hold_raises_when_the_symbol_has_no_bar_on_the_first_session(
    con: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    root = world(tmp_path, override={("A", FEB_FIRST): None})

    with pytest.raises(ValueError):
        buy_and_hold(con, root, "A", FEB_FIRST, LAST)
    with pytest.raises(ValueError):
        buy_and_hold(con, root, "ZZZ", FEB_FIRST, LAST)


def test_window_starting_mid_month_rebalances_on_its_first_session(
    con: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    root = world(tmp_path)
    start = date(2025, 2, 12)

    curve = basket(con, root, start, LAST, size=2, lookback_sessions=LOOKBACK)

    # Feb 5-11 volume: D 40M > C 30M > B 18M > A 12M.
    assert curve.rebalances == (
        Rebalance(start, (("S_D", "D"), ("S_C", "C"))),
        Rebalance(MAR_FIRST, (("S_D", "D"), ("S_C", "C"))),
    )
    # Flat February; Mar 3 rebuys 0.5 each at the open: 0.5 * 36 / 40 + 0.5 * 36 / 30 = 1.05.
    assert at(curve, FEB_LAST) == pytest.approx(1.0)
    assert curve.total_return == pytest.approx(0.05)


def test_window_starting_on_a_weekend_first_rebalances_once_on_the_first_session(
    con: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    root = world(tmp_path)

    curve = basket(con, root, date(2025, 3, 1), LAST, size=2, lookback_sessions=LOOKBACK)

    assert curve.sessions[0] == MAR_FIRST
    assert [r.session for r in curve.rebalances] == [MAR_FIRST]


def test_size_larger_than_eligible_weights_equally_among_the_available(
    con: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    root = world(tmp_path)

    curve = basket(con, root, FEB_FIRST, LAST, size=10, lookback_sessions=LOOKBACK)

    assert curve.rebalances == (
        Rebalance(FEB_FIRST, (("S_A", "A"), ("S_B", "B"), ("S_C", "C"), ("S_D", "D"))),
        Rebalance(MAR_FIRST, (("S_D", "D"), ("S_C", "C"), ("S_B", "B"), ("S_A", "A"))),
    )
    # Feb 3 close: 0.25 * (12/10 + 18/20 + 30/30 + 40/40) = 1.025.
    assert at(curve, FEB_FIRST) == pytest.approx(1.025)
    # Mar 3 open: 0.25 * (13/10 + 18/20 + 1 + 1) = 1.05, a quarter of it into each.
    # Close: 0.2625 * (36/40 + 36/30 + 18/18 + 13/13) = 1.07625.
    assert curve.total_return == pytest.approx(0.07625)


def test_an_etf_with_the_largest_dollar_volume_never_enters(
    con: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    root = world(tmp_path)
    # The fixture is meaningful: SPY tops the tier when ETFs are allowed.
    tier = liquid_securities(
        con, root, known(FEB_FIRST), min_adv_usd=0, lookback_sessions=LOOKBACK, classes=None
    )
    assert tier.column("symbol").to_pylist()[0] == "SPY"

    curve = basket(con, root, FEB_FIRST, LAST, size=1, lookback_sessions=LOOKBACK)

    assert curve.rebalances == (
        Rebalance(FEB_FIRST, (("S_A", "A"),)),
        Rebalance(MAR_FIRST, (("S_D", "D"),)),
    )


def test_a_member_that_stops_trading_carries_its_last_close_until_the_next_rebalance(
    con: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    stop = date(2025, 2, 13)
    gone: dict[tuple[str, date], Ohlcv | None] = {("B", d): None for d in SESSIONS if d > stop}
    root = world(tmp_path, override={("B", stop): (18.0, 16.0, M), **gone})

    curve = basket(con, root, FEB_FIRST, LAST, size=2, lookback_sessions=LOOKBACK)

    # From Feb 13: 0.05 * 12 + 0.025 * 16 = 1.0, held at 16 through the end of February.
    assert at(curve, stop) == pytest.approx(1.0)
    assert at(curve, date(2025, 2, 14)) == pytest.approx(1.0)
    assert at(curve, FEB_LAST) == pytest.approx(1.0)
    # Mar 3: A sells at 13 (0.65), B is worth its last close (0.40): 1.05 -> 0.525 into D and C.
    # Close: 0.525 * 36 / 40 + 0.525 * 36 / 30 = 1.1025.
    assert curve.rebalances[1] == Rebalance(MAR_FIRST, (("S_D", "D"), ("S_C", "C")))
    assert curve.total_return == pytest.approx(0.1025)


def test_a_dividend_before_the_window_is_not_paid_on_its_first_session(
    con: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    # Real bug: SPY 2023-01-03 .. 2026-10-16 came out +117.4% instead of +107.7% because every
    # payout since 2020 was credited on the first session.
    d = FEB[:3]
    root = path_store(
        tmp_path,
        "X",
        [(d[0], 100, 100), (d[1], 100, 100), (d[2], 100, 100)],
        dividends=[("X", date(2025, 1, 15), 5.0)],
    )

    curve = buy_and_hold(con, root, "X", d[0], d[2])

    assert curve.equity == pytest.approx((1.0, 1.0, 1.0))


def test_shares_bought_at_the_ex_date_open_earn_no_dividend(
    con: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    # A buyer at or after the ex-date open is not on record for the payout.
    d = FEB[:2]
    root = path_store(
        tmp_path,
        "X",
        [(d[0], 100, 100), (d[1], 100, 100)],
        dividends=[("X", d[0], 5.0)],
    )

    curve = buy_and_hold(con, root, "X", d[0], d[1])

    assert curve.equity == pytest.approx((1.0, 1.0))


def test_basket_members_get_splits_and_dividends(
    con: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    split_day, div_day = date(2025, 3, 10), date(2025, 3, 17)
    halved = {("D", d): (18.0, 18.0, M) for d in SESSIONS if d >= split_day}
    root = world(
        tmp_path,
        override=halved,
        splits=[("D", split_day, 1.0, 2.0)],
        dividends=[("C", div_day, 1.2)],
    )

    curve = basket(con, root, FEB_FIRST, LAST, size=2, lookback_sessions=LOOKBACK)

    # D: 0.55 / 40 shares, doubled on Mar 10 at half the price: still 0.495.
    assert at(curve, split_day) == pytest.approx(1.155)
    # C: 0.55 / 30 shares at 36 = 0.66, plus 1.2 per share = 0.022 reinvested.
    assert at(curve, div_day) == pytest.approx(1.177)
    assert curve.total_return == pytest.approx(0.177)


# =============================================================================================
# leakage
# =============================================================================================


def test_members_and_values_through_a_rebalance_do_not_depend_on_later_bars(
    con: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    # The full store has A trading $1.3B a day after Mar 3; the cut store stops at Mar 3.
    later = {("A", d): (13.0, 13.0, 100 * M) for d in SESSIONS if d > MAR_FIRST}
    full = world(tmp_path / "full", override=later)
    cut = world(tmp_path / "cut", through=MAR_FIRST)

    long = basket(con, full, FEB_FIRST, LAST, size=2, lookback_sessions=LOOKBACK)
    short = basket(con, cut, FEB_FIRST, MAR_FIRST, size=2, lookback_sessions=LOOKBACK)

    assert long.rebalances == short.rebalances
    assert long.rebalances[1] == Rebalance(MAR_FIRST, (("S_D", "D"), ("S_C", "C")))
    assert long.equity[: len(short.equity)] == pytest.approx(short.equity)
    assert long.sessions[: len(short.sessions)] == short.sessions


def test_volume_on_the_rebalance_day_itself_does_not_pick_members(
    con: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    # B trades $20B on Feb 3 and A $13B on Mar 3: both bars are on disk, neither is visible at
    # 00:00 New York of its own session.
    root = world(
        tmp_path,
        override={
            ("B", FEB_FIRST): (20.0, 18.0, 1000 * M),
            ("A", MAR_FIRST): (13.0, 13.0, 1000 * M),
        },
    )

    curve = basket(con, root, FEB_FIRST, LAST, size=2, lookback_sessions=LOOKBACK)

    assert curve.rebalances == (
        Rebalance(FEB_FIRST, (("S_A", "A"), ("S_B", "B"))),
        Rebalance(MAR_FIRST, (("S_D", "D"), ("S_C", "C"))),
    )


def test_buy_and_hold_ignores_bars_and_actions_after_the_window(
    con: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    halved = {("SPY", d): (51.5, 51.5, 10 * M) for d in SESSIONS if d >= MAR_FIRST}
    with_actions = world(
        tmp_path / "with",
        override=halved,
        splits=[("SPY", MAR_FIRST, 1.0, 2.0)],
        dividends=[("SPY", MAR_FIRST, 5.0), ("SPY", date(2025, 3, 21), 5.0)],
    )
    without = world(tmp_path / "without", through=FEB_LAST)

    seen = buy_and_hold(con, with_actions, "SPY", FEB_FIRST, FEB_LAST)
    clean = buy_and_hold(con, without, "SPY", FEB_FIRST, FEB_LAST)

    assert seen.sessions == clean.sessions == in_range(FEB_FIRST, FEB_LAST)
    assert seen.equity == pytest.approx(clean.equity)
    assert seen.total_return == pytest.approx(0.01)


def test_basket_ignores_member_actions_after_the_window(
    con: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    end = date(2025, 3, 7)
    root = world(
        tmp_path,
        splits=[("D", date(2025, 3, 10), 1.0, 2.0)],
        dividends=[("C", date(2025, 3, 17), 1.2)],
    )
    clean = world(tmp_path / "clean")

    seen = basket(con, root, FEB_FIRST, end, size=2, lookback_sessions=LOOKBACK)
    plain = basket(con, clean, FEB_FIRST, end, size=2, lookback_sessions=LOOKBACK)

    assert seen.equity == pytest.approx(plain.equity)
    assert seen.total_return == pytest.approx(0.155)
