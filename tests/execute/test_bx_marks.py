# Guards the numbers the B-Xtrender replay reads off each night: an exit flag raised by a missing
# value (NaN read as "red"), an ATR cut from the wrong rows (the signal bar left out, a window
# one short, too little history passed off as a number), and an account marked without a tranche
# that was sold but not yet booked or without spin-off children already paid.
"""API under test (asof.execute.bx unless stated):

    exit_flags(row: Mapping) -> ExitFlags
        daily_red = bx_daily < 0; weekly_down = bx_weekly < bx_weekly_prev; a NaN / None among a
        flag's values -> that flag False.
    signal_atr(bars: pd.DataFrame, days: int) -> float
        bars of one security (session_date, high, low, close), already split-adjusted; the
        simple mean of the last ``days`` true ranges, the last row's included;
        TR = max(h - l, |h - prev close|, |l - prev close|); NaN with fewer than days + 1 rows.
    BxPositionSim.exposure(last_close) / .unrealized(last_close)
    asof.execute.DecisionSim.exposure(last_close) / .unrealized(last_close)
        exposure = (held shares + waiting order shares) x last_close; unrealized = held shares x
        (last_close - entry) + tranches sold but held back (BX) / the unpaid leg (DecisionSim) at
        their sale price + BX ``open_child``.

Hand-made numbers. Choices pinned here where the spec left one open:
- ``signal_atr`` takes its rows in date order as given (the tests pass them sorted); NaN is
  ``math.isnan``.
- A held-back BX tranche adds nothing to ``exposure`` (its shares are gone).

No leakage case: these functions take no as_of; what they are handed is the replay's business
(tests/replay/test_replay_bx.py holds the as_of cases). Assumptions only real data can confirm:
- The panel columns the replay hands to ``exit_flags`` are named bx_daily / bx_weekly /
  bx_weekly_prev on the real panel too: ``bx_panel`` on any liquid security at a recent close
  has those three columns.
- True ranges across a missing session use the previous visible bar's close: count, on the real
  store, liquid-tier signals whose 15 rows before the signal span a missing session (calendar
  audit I12); a non-zero count means some ATRs bridge a gap.
"""

import math
from datetime import date
from typing import Any

import pandas as pd
import pytest

from asof.execute import DailyBar, DecisionPlan, DecisionSim
from asof.execute.bx import BxPlan, BxPositionSim, ExitFlags, exit_flags, signal_atr
from asof.strategy import BxExecuteConfig, ExecuteConfig
from tests.execute.test_bx import EXECUTE
from tests.strategy.files import DEFAULTS

NAN = float("nan")
X = [d.date() for d in pd.bdate_range("2026-03-02", periods=10)]


def bar(day: date, o: float, h: float, lo: float, c: float) -> DailyBar:
    return DailyBar(session_date=day, open=o, high=h, low=lo, close=c)


# =============================================================================================
# exit_flags
# =============================================================================================


def row(daily: Any, weekly: Any, weekly_prev: Any) -> dict[str, Any]:
    return {"bx_daily": daily, "bx_weekly": weekly, "bx_weekly_prev": weekly_prev}


def test_flags_on_defined_values() -> None:
    assert exit_flags(row(-0.5, 3.0, 4.0)) == ExitFlags(daily_red=True, weekly_down=True)
    assert exit_flags(row(0.5, 4.0, 3.0)) == ExitFlags(daily_red=False, weekly_down=False)


def test_flags_at_their_boundaries_are_not_raised() -> None:
    # 0 is not red; an unchanged weekly bar is not down
    assert exit_flags(row(0.0, 2.0, 2.0)) == ExitFlags(daily_red=False, weekly_down=False)


@pytest.mark.parametrize("missing", [NAN, None])
def test_a_missing_value_lowers_only_its_own_flag(missing: Any) -> None:
    assert exit_flags(row(missing, 1.0, 2.0)) == ExitFlags(daily_red=False, weekly_down=True)
    assert exit_flags(row(-1.0, missing, 2.0)) == ExitFlags(daily_red=True, weekly_down=False)
    assert exit_flags(row(-1.0, 1.0, missing)) == ExitFlags(daily_red=True, weekly_down=False)
    assert exit_flags(row(missing, missing, missing)) == ExitFlags(
        daily_red=False, weekly_down=False
    )


def test_flags_read_a_panel_row_with_other_columns() -> None:
    panel_row = pd.Series(
        {"security_id": "S_AAA", "bx_daily": -2.0, "bx_weekly": 5.0, "bx_weekly_prev": 4.0}
    )
    assert exit_flags(panel_row.to_dict()) == ExitFlags(daily_red=True, weekly_down=False)


# =============================================================================================
# signal_atr
# =============================================================================================

# (high, low, close); true ranges from the second row: 3, 4, 2, 6
HLC = [(11.0, 9.0, 10.0), (12.0, 9.0, 11.0), (15.0, 13.0, 14.0), (14.5, 12.5, 13.0)]
HLC += [(13.0, 7.0, 8.0)]
TRS = [3.0, 4.0, 2.0, 6.0]


def frame(rows: list[tuple[float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_date": X[: len(rows)],
            "high": [r[0] for r in rows],
            "low": [r[1] for r in rows],
            "close": [r[2] for r in rows],
        }
    )


def test_atr_is_the_mean_of_the_last_true_ranges_including_the_last_row() -> None:
    # row 2: gap up, |h - prev close| = 4; row 3: |l - prev| = 1.5 < h - l = 2; row 4: 6
    assert signal_atr(frame(HLC), 2) == pytest.approx((2.0 + 6.0) / 2)
    assert signal_atr(frame(HLC), 4) == pytest.approx(sum(TRS) / 4)


def test_atr_needs_one_row_more_than_its_days() -> None:
    assert signal_atr(frame(HLC), 4) == pytest.approx(3.75)  # exactly days + 1 rows
    assert math.isnan(signal_atr(frame(HLC), 5))  # one short: no first true range to use
    assert math.isnan(signal_atr(frame(HLC[:1]), 1))
    assert math.isnan(signal_atr(frame([]), 1))


def test_atr_ignores_rows_before_its_window() -> None:
    wild = [(100.0, 1.0, 50.0), *HLC]  # a row before the window changes nothing
    assert signal_atr(frame(wild), 2) == pytest.approx(signal_atr(frame(HLC), 2))


# =============================================================================================
# BxPositionSim.exposure / unrealized
# =============================================================================================


def bx_cfg() -> BxExecuteConfig:
    return BxExecuteConfig.model_validate(EXECUTE)  # slippage 0


def bx_sim(shares: int = 100) -> BxPositionSim:
    plan = BxPlan(signal_session=X[0], signal_close=50.0, atr=2.0, shares=shares, equity=1e5)
    return BxPositionSim(plan, bx_cfg())


RED = ExitFlags(daily_red=True, weekly_down=False)
NONE = ExitFlags(daily_red=False, weekly_down=False)


def test_bx_waiting_order_is_exposure_without_unrealized() -> None:
    sim = bx_sim()
    assert sim.exposure(50.0) == pytest.approx(5000.0)
    assert sim.unrealized(50.0) == 0.0


def test_bx_held_shares_are_marked_at_the_last_close() -> None:
    sim = bx_sim()
    sim.step(bar(X[1], 50.5, 51.5, 50.0, 51.0), NONE)  # 100 in at 50.5
    assert sim.exposure(51.0) == pytest.approx(5100.0)
    assert sim.unrealized(51.0) == pytest.approx(50.0)
    sim.step(bar(X[2], 51.0, 52.0, 50.5, 51.5), RED)
    sim.step(bar(X[3], 52.0, 52.5, 51.0, 52.0), NONE)  # 50 sold at 52.0, booked
    assert sim.exposure(52.0) == pytest.approx(50 * 52.0)
    assert sim.unrealized(52.0) == pytest.approx(50 * 1.5)


def test_bx_a_tranche_held_back_for_its_children_counts_at_its_sale_price() -> None:
    sim = bx_sim()
    sim.step(bar(X[1], 50.5, 51.5, 50.0, 51.0), RED)  # 100 in at 50.5; daily_red pending
    sim.begin_spinoff()  # X2 is the ex-date
    assert sim.step(bar(X[2], 48.0, 48.5, 46.5, 47.0), NONE) == []  # 50 sold at 48.0, held back
    assert sim.shares == 50
    # 50 held at 47.0: -175.0; 50 sold at 48.0, not booked yet: -125.0
    assert sim.exposure(47.0) == pytest.approx(50 * 47.0)
    assert sim.unrealized(47.0) == pytest.approx(-175.0 - 125.0)

    released = sim.credit_children(100.0)  # 100 children-dollars over the 100 ex-date shares
    assert [leg.pnl for leg in released] == [pytest.approx(-125.0 + 50.0)]
    # the tranche is booked now; the 50 still held carry 50.0 of children already received
    assert sim.unrealized(47.0) == pytest.approx(-175.0 + 50.0)


def test_bx_children_of_an_earlier_spinoff_carried_by_a_held_back_tranche_count() -> None:
    # 100 in at 50.5; a first spin-off pays 250.0 of children to them (open_child 250); then a
    # second spin-off goes ex while daily_red is pending: half the shares are sold at the ex
    # open, 46.0, and held back with their half of the earlier children (prior_child 125)
    sim = bx_sim()
    sim.step(bar(X[1], 50.5, 51.5, 50.0, 51.0), NONE)
    sim.begin_spinoff()  # X2: the first ex-date
    sim.step(bar(X[2], 48.0, 48.5, 47.0, 47.5), NONE)
    sim.apply_spinoff(0.95)
    assert sim.credit_children(250.0) == []
    assert sim.open_child == pytest.approx(250.0)
    sim.step(bar(X[3], 48.0, 49.0, 47.8, 48.5), RED)  # daily_red pending for X4's open
    sim.begin_spinoff()  # X4: the second ex-date
    assert sim.step(bar(X[4], 46.0, 46.5, 45.0, 45.5), NONE) == []  # 50 sold, held back
    assert sim.shares == 50
    assert sim.open_child == pytest.approx(125.0)
    [back] = sim.held_back
    assert back.prior_child == pytest.approx(125.0)

    # held 50 x (45 - 50.5) = -275; held back 50 x (46 - 50.5) = -225 plus its 125 of earlier
    # children; the 125 still with the shares held
    assert sim.unrealized(45.0) == pytest.approx(-275.0 - 225.0 + 125.0 + 125.0)
    assert sim.exposure(45.0) == pytest.approx(50 * 45.0)


# =============================================================================================
# DecisionSim.exposure / unrealized (the numbers the replay's _open_value / _unrealized give)
# =============================================================================================

# tests/replay/scenario.py: reaction close 56, gap low 54, ATR 2 -> 125 shares; slippage 0
EG_PLAN = DecisionPlan(
    reaction_session=X[0],
    decision_close=56.0,
    gap_low=54.0,
    pre_earnings_close=50.0,
    atr=2.0,
    equity=100_000.0,
)


def eg_sim() -> DecisionSim:
    raw = {**DEFAULTS["execute"], "fills": {"slippage_bps": 0.0}}
    return DecisionSim(EG_PLAN, ExecuteConfig.model_validate(raw))


def test_decision_waiting_order_is_exposure_without_unrealized() -> None:
    sim = eg_sim()
    assert sim.exposure(56.0) == pytest.approx(125 * 56.0)
    assert sim.unrealized(56.0) == 0.0


def test_decision_held_leg_is_marked_at_the_last_close() -> None:
    # tests/replay/test_equity_daily.py row X1: 125 in at 56.5, close 56.8 -> +37.5, 7100
    sim = eg_sim()
    sim.step(bar(X[1], 56.5, 57.0, 56.0, 56.8))
    assert sim.exposure(56.8) == pytest.approx(7100.0)
    assert sim.unrealized(56.8) == pytest.approx(37.5)


def test_decision_a_leg_sold_before_its_children_counts_at_its_sale_price() -> None:
    sim = eg_sim()
    sim.step(bar(X[1], 56.5, 57.0, 56.0, 56.8))
    sim.step(bar(X[2], 56.0, 56.2, 53.0, 53.4))  # close below the zone: break pending
    sim.begin_spinoff()
    assert sim.step(bar(X[3], 53.4, 53.5, 50.2, 50.8)) == []  # sold at 53.4, held back
    assert sim.unpaid is not None
    assert sim.exposure(50.8) == 0.0
    assert sim.unrealized(50.8) == pytest.approx(125 * (53.4 - 56.5))  # -387.5
