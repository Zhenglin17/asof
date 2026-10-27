"""Guards against curve statistics that are quietly off: the first session's return dropped (the
move from the money invested to the first close), a drawdown that ignores a fall below the
starting value, an annualization that does not compound, a risk-free rate taken per year where
it is needed per day, a population standard deviation where the sample one is meant, a Sharpe
ratio invented from one return or a flat line, and a strategy curve and a benchmark curve that
disagree on the same arithmetic.

API under test (asof.evolve.metrics, new module):

    @dataclass(frozen=True) CurveMetrics(sessions: int, total_return: float,
        annualized_return: float, max_drawdown: float, sharpe: float | None)
    curve_metrics(equity: Sequence[float], *, base: float, risk_free_rate: float = 0.0)
        base = value before the first session; r_0 = equity[0] / base - 1,
        r_i = equity[i] / equity[i - 1] - 1; total = equity[-1] / base - 1;
        annualized = (1 + total) ** (252 / len(equity)) - 1;
        max_drawdown = largest (peak - value) / peak, peak starting at base, >= 0;
        daily rf = (1 + risk_free_rate) ** (1 / 252) - 1;
        sharpe = mean(r - rf) / stdev(r - rf, ddof=1) * sqrt(252), None with fewer than 2
        returns or a zero standard deviation. Empty equity or base <= 0 -> ValueError.

No as_of here: curve_metrics is arithmetic on a list it is given and reads no store, so there is
no leakage case (what goes into the list is tested in tests/replay/test_equity_daily.py and
tests/evolve/test_versus.py). The numbers below are made up for hand arithmetic. Assumptions to
check on real data:
- 252 sessions per year is the convention the benchmarks already use (TRADING_DAYS in
  asof.evolve.benchmark). Check: visible_sessions on the real store has 251-253 sessions in each
  full year 2021-2025.
- Standard case: SPY buy-and-hold over 2025-07-01 .. 2026-06-30 through curve_metrics(base=1.0)
  gives a Sharpe ratio within a few hundredths of one recomputed by hand in a spreadsheet from
  the same daily closes and dividends.
"""

import math
from datetime import date

import pandas as pd
import pytest

from asof.evolve.benchmark import BenchmarkCurve
from asof.evolve.metrics import CurveMetrics, curve_metrics

SQRT_252 = math.sqrt(252)


# =============================================================================================
# normal path
# =============================================================================================


def test_hand_computed_three_sessions() -> None:
    # Returns 2%, 0%, 1%: mean 1%, sample sd 1% -> Sharpe sqrt(252). Never below the base.
    m = curve_metrics([102.0, 102.0, 103.02], base=100.0)

    assert m.sessions == 3
    assert m.total_return == pytest.approx(0.0302)
    assert m.annualized_return == pytest.approx(1.0302**84 - 1)
    assert m.max_drawdown == 0.0
    assert m.sharpe == pytest.approx(SQRT_252)


def test_drawdown_from_an_intermediate_peak() -> None:
    # 100 -> 110 -> 99 -> 121: worst fall 110 -> 99 = 10%. Returns 10%, -10%, 2/9.
    m = curve_metrics([110.0, 99.0, 121.0], base=100.0)

    assert m.total_return == pytest.approx(0.21)
    assert m.max_drawdown == pytest.approx(0.1)
    returns = [0.1, -0.1, 2 / 9]
    mean = sum(returns) / 3
    sd = math.sqrt(sum((r - mean) ** 2 for r in returns) / 2)  # sample, ddof = 1
    assert m.sharpe == pytest.approx(mean / sd * SQRT_252)
    assert m.sharpe == pytest.approx(7.2288, abs=1e-4)


def test_risk_free_rate_is_compounded_down_to_a_daily_rate() -> None:
    # 5% a year -> (1.05) ** (1 / 252) - 1 = 0.000193630 a day, subtracted from every return;
    # the spread is unchanged, so only the mean moves.
    daily = 1.05 ** (1 / 252) - 1
    m = curve_metrics([102.0, 102.0, 103.02], base=100.0, risk_free_rate=0.05)

    assert daily == pytest.approx(0.00019363, abs=1e-8)
    assert m.sharpe == pytest.approx((0.01 - daily) / 0.01 * SQRT_252)
    assert m.total_return == pytest.approx(0.0302)  # returns themselves are not excess returns


def test_metrics_do_not_depend_on_the_unit_of_money() -> None:
    dollars = curve_metrics([100_500.0, 99_000.0, 101_000.0], base=100_000.0)
    fraction = curve_metrics([1.005, 0.99, 1.01], base=1.0)

    assert dollars.total_return == pytest.approx(fraction.total_return)
    assert dollars.max_drawdown == pytest.approx(fraction.max_drawdown)
    assert dollars.sharpe == pytest.approx(fraction.sharpe)


def test_curve_metrics_match_a_benchmark_curves_own_properties() -> None:
    days = tuple(d.date() for d in pd.bdate_range("2025-02-03", periods=6))
    curve = BenchmarkCurve("x", days, (1.02, 0.95, 0.97, 1.08, 1.01, 1.05))

    m = curve_metrics(curve.equity, base=1.0)

    assert m.sessions == len(curve.sessions)
    assert m.total_return == pytest.approx(curve.total_return)
    assert m.annualized_return == pytest.approx(curve.annualized_return)
    assert m.max_drawdown == pytest.approx(curve.max_drawdown)


def test_curve_metrics_is_a_frozen_dataclass() -> None:
    m = curve_metrics([101.0, 102.0], base=100.0)
    assert isinstance(m, CurveMetrics)
    with pytest.raises(AttributeError):
        m.sessions = 3  # pyright: ignore[reportAttributeAccessIssue]


# =============================================================================================
# boundaries
# =============================================================================================


def test_a_fall_below_the_base_on_the_first_session_counts_as_drawdown() -> None:
    m = curve_metrics([80.0, 90.0], base=100.0)

    assert m.max_drawdown == pytest.approx(0.2)
    assert m.total_return == pytest.approx(-0.1)


def test_one_session_has_no_sharpe_but_the_other_numbers() -> None:
    # One return, so no standard deviation; annualizing one session is ** 252.
    m = curve_metrics([101.0], base=100.0)

    assert m.sessions == 1
    assert m.total_return == pytest.approx(0.01)
    assert m.annualized_return == pytest.approx(1.01**252 - 1)
    assert m.sharpe is None


def test_two_sessions_are_enough_for_a_sharpe() -> None:
    # Returns 1% and 3%: mean 2%, sample sd sqrt(2) * 1%.
    m = curve_metrics([101.0, 104.03], base=100.0)

    assert m.sharpe == pytest.approx(0.02 / (math.sqrt(2) * 0.01) * SQRT_252)


def test_a_flat_curve_has_zero_return_zero_drawdown_and_no_sharpe() -> None:
    # A strategy that never traded: every return is exactly 0, the spread is 0.
    m = curve_metrics([100_000.0] * 5, base=100_000.0)

    assert (m.sessions, m.total_return, m.annualized_return, m.max_drawdown) == (5, 0.0, 0.0, 0.0)
    assert m.sharpe is None


def test_252_sessions_annualize_to_the_total_and_126_compound() -> None:
    year = [1.0 + 0.3 * (i + 1) / 252 for i in range(252)]
    half = [1.0 + 0.1 * (i + 1) / 126 for i in range(126)]

    assert curve_metrics(year, base=1.0).annualized_return == pytest.approx(0.3)
    assert curve_metrics(half, base=1.0).annualized_return == pytest.approx(1.1**2 - 1)


def test_an_empty_curve_raises() -> None:
    with pytest.raises(ValueError):
        curve_metrics([], base=100.0)


@pytest.mark.parametrize("base", [0.0, -1.0])
def test_a_base_that_is_not_positive_raises(base: float) -> None:
    with pytest.raises(ValueError):
        curve_metrics([101.0, 102.0], base=base)


@pytest.mark.parametrize("equity", [[101.0, 0.0], [101.0, -5.0]])
def test_an_equity_that_is_not_positive_raises(equity: list[float]) -> None:
    # A ratio against zero or a negative account has no meaning: refuse instead of a complex
    # annualized return or a division by zero downstream.
    with pytest.raises(ValueError):
        curve_metrics(equity, base=100.0)


def test_accepts_a_tuple_as_benchmark_curves_store_it() -> None:
    curve = BenchmarkCurve("x", (date(2025, 2, 3), date(2025, 2, 4)), (1.01, 1.02))
    assert curve_metrics(curve.equity, base=1.0).sessions == 2
