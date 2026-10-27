"""Guards against a strategy-versus-benchmark comparison that flatters for the wrong reason: a
window that is not the run's own sessions, a strategy base that already includes the first
day's profit, benchmarks or settings taken from somewhere other than the protocol, exposure
measured against the starting money instead of each day's equity, a criterion that compares
with the wrong benchmark (QQQ or the best of all instead of SPY and the basket), a missing Sharpe
ratio read as a pass, and benchmark bars after the run's last session changing the verdict.

API under test (asof.evolve.versus, new module):

    compare_run(session, con, root, run_id, protocol) -> Versus
        rows = the run's EquityDaily rows by session_date; LookupError when the run does not
        exist or has no rows. start, end = first and last session_date.
        strategy = curve_metrics(equities, base=rows[0].equity - rows[0].realized
                   - rows[0].unrealized, risk_free_rate=protocol.comparison.risk_free_rate)
        benchmarks = [(curve.label, curve_metrics(curve.equity, base=1.0, risk_free_rate=...))
                      for curve in evolve.benchmark.benchmarks(con, root, protocol, start, end)]
        mean_exposure = mean(gross_exposure / equity) over the rows
        high_exposure_share = share of rows with gross_exposure / equity >= high_exposure
        drawdown_not_worse_than_basket = strategy max_drawdown <= basket's
        sharpe_at_least_best = strategy sharpe >= max(SPY's, basket's); False if any is None
        beats_spy_and_basket_total = strategy total_return > SPY's and > basket's
    compare_run(..., window=(first, last)) keeps only rows with first <= session_date <= last;
        start, end are then the first and last kept rows. The base is the equity of the row just
        before the first kept one when there is one (the slice starts after the run's first
        row), else rows[0].equity - realized - unrealized as above. No row in the window ->
        LookupError, like a run without rows. window=None (the default) is the whole run.
    Versus(run_id, start, end, strategy, benchmarks, mean_exposure, high_exposure_share,
           drawdown_not_worse_than_basket, sharpe_at_least_best, beats_spy_and_basket_total)
    "SPY" = the buy-and-hold of protocol.benchmarks.symbols[0]; the basket = the last curve,
    the one not named after a protocol symbol.

Market store: tests/evolve/test_benchmark.py's world (provenance there: real NYSE closures of
Jan-Mar 2025, bars visible from 20:00 New York). Here A also dips on 2025-02-10 (close 9) and
recovers on 02-11 (close 12), so the size-2 basket has a drawdown (1.05 -> 0.90, 14.29%).
Benchmarks over 2025-02-03 .. 03-31 (40 sessions), measured with the current code: SPY total 3%,
drawdown 0, Sharpe 3.41; A total 30%; basket total 15.5%, drawdown 14.29%, Sharpe 1.76.
EquityDaily rows are written by hand (the replay writing them is tested in
tests/replay/test_equity_daily.py); their paths are made up for hand arithmetic.
"""

from collections.abc import Sequence
from datetime import date
from pathlib import Path

import duckdb
import pytest
from sqlmodel import Session

from asof.evolve.benchmark import benchmarks
from asof.evolve.metrics import CurveMetrics, curve_metrics
from asof.evolve.protocol import EvaluationProtocol
from asof.evolve.versus import Versus, compare_run
from asof.store.models import EquityDaily, Run
from tests.evolve.protocols import protocol
from tests.evolve.test_benchmark import FEB_FIRST, LAST, LOOKBACK, SESSIONS, M, Ohlcv, world
from tests.scan.world import close_of

DIP: dict[tuple[str, date], Ohlcv | None] = {
    ("A", date(2025, 2, 10)): (12.0, 9.0, M),
    ("A", date(2025, 2, 11)): (9.0, 12.0, M),
}
WINDOW = [d for d in SESSIONS if FEB_FIRST <= d <= LAST]  # 40 sessions
INITIAL = 100_000.0


def settings(*, risk_free_rate: float = 0.0, high_exposure: float = 0.70) -> EvaluationProtocol:
    """SPY first, then A (a stock that beats everything: no criterion may look at it)."""
    return protocol(
        benchmarks={
            "symbols": ["SPY", "A"],
            "basket": {"size": 2, "lookback_sessions": LOOKBACK, "rebalance": "monthly"},
        },
        comparison={"risk_free_rate": risk_free_rate, "high_exposure": high_exposure},
    )


def new_run(session: Session) -> int:
    run = Run(
        strategy_id="earnings_gap",
        mode="replay",
        as_of=close_of(LAST),
        started_at=close_of(LAST),
        git_sha="test",
    )
    session.add(run)
    session.flush()
    assert run.id is not None
    return run.id


def add_rows(
    session: Session,
    run_id: int,
    days: Sequence[date],
    equity: Sequence[float],
    *,
    gross: Sequence[float] | None = None,
    first_unrealized: float = 0.0,
) -> None:
    """One row per day; realized = equity - INITIAL except what ``first_unrealized`` moves out of
    the first row's realized into its unrealized."""
    for i, (day, value) in enumerate(zip(days, equity, strict=True)):
        unrealized = first_unrealized if i == 0 else 0.0
        exposure = gross[i] if gross is not None else 0.0
        session.add(
            EquityDaily(
                run_id=run_id,
                session_date=day,
                equity=value,
                realized=value - INITIAL - unrealized,
                unrealized=unrealized,
                gross_exposure=exposure,
                open_positions=1 if exposure else 0,
                available_at=close_of(day),
            )
        )
    session.commit()


def strong_path() -> list[float]:
    """+1%, 0%, +1%, ... : never below a previous high, total 1.01 ** 20 - 1 = 22.0%, Sharpe
    sqrt(39 / 40) * sqrt(252) = 15.67."""
    out, value = [], INITIAL
    for i in range(len(WINDOW)):
        value *= 1.01 if i % 2 == 0 else 1.0
        out.append(value)
    return out


def middling_path() -> list[float]:
    """Straight down to 80,000 over 20 sessions, straight up to 110,000 over 20: drawdown 20%,
    total 10% (above SPY's 3%, below the basket's 15.5%), Sharpe 2.85 (above the basket's 1.76,
    below SPY's 3.41)."""
    down = [INITIAL - 20_000 * (i + 1) / 20 for i in range(20)]
    up = [80_000 + 30_000 * (i + 1) / 20 for i in range(20)]
    return down + up


def same(a: CurveMetrics, b: CurveMetrics) -> None:
    assert a.sessions == b.sessions
    assert a.total_return == pytest.approx(b.total_return)
    assert a.annualized_return == pytest.approx(b.annualized_return)
    assert a.max_drawdown == pytest.approx(b.max_drawdown)
    if b.sharpe is None:
        assert a.sharpe is None
    else:
        assert a.sharpe == pytest.approx(b.sharpe)


def same_versus(a: Versus, b: Versus) -> None:
    assert (a.run_id, a.start, a.end) == (b.run_id, b.start, b.end)
    same(a.strategy, b.strategy)
    assert [label for label, _ in a.benchmarks] == [label for label, _ in b.benchmarks]
    for (_, x), (_, y) in zip(a.benchmarks, b.benchmarks, strict=True):
        same(x, y)
    assert a.mean_exposure == pytest.approx(b.mean_exposure)
    assert a.high_exposure_share == pytest.approx(b.high_exposure_share)
    assert (
        a.drawdown_not_worse_than_basket,
        a.sharpe_at_least_best,
        a.beats_spy_and_basket_total,
    ) == (b.drawdown_not_worse_than_basket, b.sharpe_at_least_best, b.beats_spy_and_basket_total)


# =============================================================================================
# normal path
# =============================================================================================


def test_strategy_and_benchmarks_over_the_runs_own_window(
    con: duckdb.DuckDBPyConnection, tmp_path: Path, session: Session
) -> None:
    root = world(tmp_path, override=DIP)
    p = settings()
    run_id = new_run(session)
    add_rows(session, run_id, WINDOW, strong_path())
    other = new_run(session)  # another run's rows must not leak into this one
    add_rows(session, other, WINDOW[:5], [50_000.0] * 5)

    v = compare_run(session, con, root, run_id, p)

    assert (v.run_id, v.start, v.end) == (run_id, FEB_FIRST, LAST)
    same(v.strategy, curve_metrics(strong_path(), base=INITIAL))
    assert v.strategy.total_return == pytest.approx(1.01**20 - 1)
    curves = benchmarks(con, root, p, FEB_FIRST, LAST)
    assert [label for label, _ in v.benchmarks] == [c.label for c in curves]
    assert len(v.benchmarks) == 3
    for (_, got), curve in zip(v.benchmarks, curves, strict=True):
        same(got, curve_metrics(curve.equity, base=1.0))
    # The fixture is what the docstring says it is.
    spy, a, basket = (m for _, m in v.benchmarks)
    assert (spy.total_return, a.total_return, basket.total_return) == pytest.approx(
        (0.03, 0.30, 0.155)
    )
    assert basket.max_drawdown == pytest.approx(1 - 0.90 / 1.05)


def test_a_strategy_that_wins_everywhere_passes_all_three_criteria(
    con: duckdb.DuckDBPyConnection, tmp_path: Path, session: Session
) -> None:
    root = world(tmp_path, override=DIP)
    run_id = new_run(session)
    add_rows(session, run_id, WINDOW, strong_path())

    v = compare_run(session, con, root, run_id, settings())

    # Total 22% is below A's 30%: A is neither SPY nor the basket, so it does not count.
    assert v.drawdown_not_worse_than_basket is True
    assert v.sharpe_at_least_best is True
    assert v.beats_spy_and_basket_total is True


def test_beating_one_of_spy_and_the_basket_is_not_enough(
    con: duckdb.DuckDBPyConnection, tmp_path: Path, session: Session
) -> None:
    root = world(tmp_path, override=DIP)
    run_id = new_run(session)
    add_rows(session, run_id, WINDOW, middling_path())

    v = compare_run(session, con, root, run_id, settings())

    spy, _, basket = (m for _, m in v.benchmarks)
    assert spy.total_return < v.strategy.total_return < basket.total_return
    assert spy.sharpe is not None and basket.sharpe is not None and v.strategy.sharpe is not None
    assert basket.sharpe < v.strategy.sharpe < spy.sharpe
    assert v.strategy.max_drawdown > basket.max_drawdown
    assert v.drawdown_not_worse_than_basket is False
    assert v.sharpe_at_least_best is False
    assert v.beats_spy_and_basket_total is False


def test_exposure_is_measured_against_each_days_equity(
    con: duckdb.DuckDBPyConnection, tmp_path: Path, session: Session
) -> None:
    # Ratios 0.70, 0.69999, 0.70 (35,000 of 50,000), 0: two of four at the 0.70 line. Measured
    # against the 100,000 start the third would be 0.35 and the share 1/4.
    root = world(tmp_path, override=DIP)
    run_id = new_run(session)
    days = WINDOW[:4]
    add_rows(
        session,
        run_id,
        days,
        [INITIAL, INITIAL, 50_000.0, INITIAL],
        gross=[70_000.0, 69_999.0, 35_000.0, 0.0],
    )

    v = compare_run(session, con, root, run_id, settings())

    assert v.high_exposure_share == pytest.approx(0.5)
    assert v.mean_exposure == pytest.approx((0.7 + 0.69999 + 0.7 + 0.0) / 4)


# =============================================================================================
# boundaries
# =============================================================================================


def test_the_base_is_the_equity_before_the_first_sessions_result(
    con: duckdb.DuckDBPyConnection, tmp_path: Path, session: Session
) -> None:
    # First row: equity 100,500 = 100,000 + realized 200 + unrealized 300 -> base 100,000. Taking
    # the first row's equity as the base would hide the first day's +0.5%.
    root = world(tmp_path, override=DIP)
    run_id = new_run(session)
    days = WINDOW[5:15]
    path = [100_500.0, *[101_000.0] * 8, 110_000.0]
    add_rows(session, run_id, days, path, first_unrealized=300.0)

    v = compare_run(session, con, root, run_id, settings())

    assert (v.start, v.end) == (days[0], days[-1])
    assert v.strategy.total_return == pytest.approx(0.10)
    same(v.strategy, curve_metrics(path, base=INITIAL))
    curves = benchmarks(con, root, settings(), days[0], days[-1])
    assert [label for label, _ in v.benchmarks] == [c.label for c in curves]


def test_the_risk_free_rate_and_threshold_come_from_the_protocol(
    con: duckdb.DuckDBPyConnection, tmp_path: Path, session: Session
) -> None:
    root = world(tmp_path, override=DIP)
    p = settings(risk_free_rate=0.05, high_exposure=1.0)
    run_id = new_run(session)
    days = WINDOW[:4]
    add_rows(
        session,
        run_id,
        days,
        [101_000.0, 101_000.0, 102_010.0, 102_010.0],
        gross=[101_000.0, 70_700.0, 0.0, 0.0],
    )

    v = compare_run(session, con, root, run_id, p)

    expected = curve_metrics([101_000.0, 101_000.0, 102_010.0, 102_010.0], base=INITIAL,
                             risk_free_rate=0.05)  # fmt: skip
    same(v.strategy, expected)
    curves = benchmarks(con, root, p, days[0], days[-1])
    for (_, got), curve in zip(v.benchmarks, curves, strict=True):
        same(got, curve_metrics(curve.equity, base=1.0, risk_free_rate=0.05))
    assert v.high_exposure_share == pytest.approx(0.25)  # only the fully invested day


def test_a_run_that_never_traded_has_no_sharpe_and_does_not_pass_on_it(
    con: duckdb.DuckDBPyConnection, tmp_path: Path, session: Session
) -> None:
    root = world(tmp_path, override=DIP)
    run_id = new_run(session)
    add_rows(session, run_id, WINDOW, [INITIAL] * len(WINDOW))

    v = compare_run(session, con, root, run_id, settings())

    assert v.strategy.sharpe is None
    assert v.strategy.total_return == 0.0
    assert (v.mean_exposure, v.high_exposure_share) == (0.0, 0.0)
    assert v.drawdown_not_worse_than_basket is True  # 0 <= 14.29%
    assert v.sharpe_at_least_best is False
    assert v.beats_spy_and_basket_total is False


def test_a_one_session_run_compares_over_that_session(
    con: duckdb.DuckDBPyConnection, tmp_path: Path, session: Session
) -> None:
    root = world(tmp_path, override=DIP)
    run_id = new_run(session)
    day = date(2025, 2, 12)
    add_rows(session, run_id, [day], [101_000.0])

    v = compare_run(session, con, root, run_id, settings())

    assert (v.start, v.end) == (day, day)
    assert v.strategy.sessions == 1
    assert v.strategy.sharpe is None
    assert all(m.sessions == 1 for _, m in v.benchmarks)
    assert v.sharpe_at_least_best is False


def test_a_run_whose_sessions_differ_from_the_benchmarks_is_refused(
    con: duckdb.DuckDBPyConnection, tmp_path: Path, session: Session
) -> None:
    # The run skips a session the store now has (a bar added after the replay): comparing the
    # two series day by day would pair different days, so it must stop instead.
    root = world(tmp_path, override=DIP)
    run_id = new_run(session)
    days = [WINDOW[0], *WINDOW[2:10]]
    add_rows(session, run_id, days, [INITIAL] * len(days))
    with pytest.raises(ValueError):
        compare_run(session, con, root, run_id, settings())


def test_an_unknown_run_raises_lookup_error(
    con: duckdb.DuckDBPyConnection, tmp_path: Path, session: Session
) -> None:
    root = world(tmp_path, override=DIP)
    with pytest.raises(LookupError):
        compare_run(session, con, root, 999, settings())


def test_a_run_without_equity_rows_raises_lookup_error(
    con: duckdb.DuckDBPyConnection, tmp_path: Path, session: Session
) -> None:
    root = world(tmp_path, override=DIP)
    run_id = new_run(session)
    session.commit()
    with pytest.raises(LookupError):
        compare_run(session, con, root, run_id, settings())


# =============================================================================================
# leakage
# =============================================================================================


def test_market_data_after_the_runs_last_session_changes_nothing(
    con: duckdb.DuckDBPyConnection, tmp_path: Path, session: Session
) -> None:
    # The full store has SPY halving and D trading $4B a day after the run's last session
    # (2025-02-28); the cut store ends on it. Both must give the same comparison.
    end = date(2025, 2, 28)
    days = [d for d in WINDOW if d <= end]
    later: dict[tuple[str, date], Ohlcv | None] = {
        **{("SPY", d): (50.0, 50.0, 10 * M) for d in SESSIONS if d > end},
        **{("D", d): (40.0, 40.0, 100 * M) for d in SESSIONS if d > end},
    }
    full = world(tmp_path / "full", override={**DIP, **later})
    cut = world(tmp_path / "cut", override=DIP, through=end)
    run_id = new_run(session)
    add_rows(session, run_id, days, strong_path()[: len(days)])

    seen = compare_run(session, con, full, run_id, settings())
    clean = compare_run(session, con, cut, run_id, settings())

    assert seen.end == end
    same_versus(seen, clean)


# =============================================================================================
# window
# =============================================================================================


def test_a_window_inside_the_run_starts_from_the_previous_rows_equity(
    con: duckdb.DuckDBPyConnection, tmp_path: Path, session: Session
) -> None:
    root = world(tmp_path, override=DIP)
    p = settings()
    run_id = new_run(session)
    path = strong_path()
    add_rows(session, run_id, WINDOW, path)
    first, last = WINDOW[10], WINDOW[29]

    v = compare_run(session, con, root, run_id, p, window=(first, last))

    assert (v.start, v.end) == (first, last)
    # Base = the 2025-02-14 row's equity: the +1% earned on 02-18 (WINDOW[10]) counts; the
    # account's starting 100,000 would add the ten sessions before the window.
    same(v.strategy, curve_metrics(path[10:30], base=path[9]))
    curves = benchmarks(con, root, p, first, last)
    assert [label for label, _ in v.benchmarks] == [c.label for c in curves]
    for (_, got), curve in zip(v.benchmarks, curves, strict=True):
        same(got, curve_metrics(curve.equity, base=1.0))


def test_a_window_wider_than_the_run_is_the_whole_run(
    con: duckdb.DuckDBPyConnection, tmp_path: Path, session: Session
) -> None:
    root = world(tmp_path, override=DIP)
    run_id = new_run(session)
    days = WINDOW[5:15]
    add_rows(
        session, run_id, days, [100_500.0, *[101_000.0] * 8, 110_000.0], first_unrealized=300.0
    )

    wide = compare_run(
        session, con, root, run_id, settings(), window=(date(2025, 1, 1), date(2025, 12, 31))
    )
    whole = compare_run(session, con, root, run_id, settings())

    same_versus(wide, whole)
    assert wide.strategy.total_return == pytest.approx(0.10)  # base 100,000, not 100,500


def test_a_window_starting_before_the_run_uses_the_initial_equity_and_ends_on_its_last_day(
    con: duckdb.DuckDBPyConnection, tmp_path: Path, session: Session
) -> None:
    root = world(tmp_path, override=DIP)
    run_id = new_run(session)
    path = strong_path()
    add_rows(session, run_id, WINDOW, path)

    v = compare_run(session, con, root, run_id, settings(), window=(date(2025, 1, 1), WINDOW[19]))

    assert (v.start, v.end) == (WINDOW[0], WINDOW[19])
    same(v.strategy, curve_metrics(path[:20], base=INITIAL))


def test_a_window_without_rows_raises_lookup_error(
    con: duckdb.DuckDBPyConnection, tmp_path: Path, session: Session
) -> None:
    root = world(tmp_path, override=DIP)
    run_id = new_run(session)
    add_rows(session, run_id, WINDOW[:10], strong_path()[:10])

    with pytest.raises(LookupError):
        compare_run(session, con, root, run_id, settings(), window=(WINDOW[20], WINDOW[30]))


def test_rows_and_market_data_after_the_window_change_nothing(
    con: duckdb.DuckDBPyConnection, tmp_path: Path, session: Session
) -> None:
    # Two runs share their rows up to 02-28; after it one doubles every day and SPY halves in
    # the full store. Over a window ending 02-28 both must read like the cut store's.
    end = date(2025, 2, 28)
    n = len([d for d in WINDOW if d <= end])
    later: dict[tuple[str, date], Ohlcv | None] = {
        ("SPY", d): (50.0, 50.0, 10 * M) for d in SESSIONS if d > end
    }
    full = world(tmp_path / "full", override={**DIP, **later})
    cut = world(tmp_path / "cut", override=DIP, through=end)
    path = strong_path()
    calm = new_run(session)
    add_rows(session, calm, WINDOW, path)
    wild = new_run(session)
    add_rows(session, wild, WINDOW, path[:n] + [path[n - 1] * 2 ** (i + 1) for i in range(40 - n)])

    clean = compare_run(session, con, cut, calm, settings(), window=(WINDOW[0], end))
    for run_id in (calm, wild):
        seen = compare_run(session, con, full, run_id, settings(), window=(WINDOW[0], end))
        assert seen.end == end
        same_versus(seen, Versus(**{**clean.__dict__, "run_id": run_id}))
