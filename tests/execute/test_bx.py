# Guards the B-Xtrender position simulator against acting on information it did not have (an exit
# flag read at a close filled at that same bar, a bar on or before the signal session traded on)
# and against booking the wrong trade: tranches of the wrong size, a flag firing twice, an exit
# when BX is undefined, sizing past one of its three caps, or R / MAE / MFE cut from other levels.
"""Target API (asof.execute.bx, pure, no I/O, no as_of; the caller computes the flags):

    ExitFlags(daily_red, weekly_down)                  frozen, kw_only
    BxPlan(signal_session, signal_close, atr, shares, equity)
    bx_shares(*, equity, close, atr, sizing, gross_exposure) -> int
        = position_shares(budget=equity * risk_pct / 100, price=close,
                          hard=close - atr_mult * atr, equity, max_position_pct, max_gross_pct,
                          gross_exposure)
    BxPositionSim(plan, cfg: BxExecuteConfig)
        step(bar, flags | None) -> list[LegResult]; skip_session(); force_exit(session, price,
        reason) -> list[LegResult]; apply_split(ratio); finished; cancelled; outcome()

Per bar while held (entry bar included): (1) tranches decided at the previous close fill at
this open; (2) the bar's low / high join the held extremes; (3) a True flag of an exit not yet
fired becomes pending for the NEXT open. First exit to fire sells floor(shares / 2) (all of a
1-share position), the second the rest; both on one night -> both at the next open, the floor
half to the exit listed first in cfg.exits. LegResult: leg 1, key_level = entry_price,
hard_stop = entry_price - atr_mult * atr, r_multiple = (exit - entry + child/share) /
(atr_mult * atr), holding_sessions = bars from entry through exit inclusive. MAE / MFE: the
extremes of the bars held before the exit bar plus the exit price, in % of entry_price (the
DecisionSim convention). Choices pinned here where the spec left one open:
- the result order on a both-same-night fill is free; tests look tranches up by exit_reason;
- a tranche booked at an open does not see that bar's range (it left at the open);
- skip_session while held changes nothing (holding_sessions counts bars).

World (made up): signal session X0 closes at 50.0, ATR 2.0, atr_mult 2 -> R unit 4.0; the
caller sized 100 shares; slippage 0 unless stated. Entry X1 open 50.5 -> hard_stop 46.5.

No look-ahead source here other than the flags and the bar order: the leakage analogues are
"a flag fills only at the next open" and "bars on or before the signal session are ignored".
"""

import math
from datetime import date
from typing import Any

import pandas as pd
import pytest

from asof.execute import DailyBar, LegResult, position_shares
from asof.execute.bx import BxPlan, BxPositionSim, ExitFlags, bx_shares
from asof.strategy import BxExecuteConfig

X = [d.date() for d in pd.bdate_range("2026-03-02", periods=30)]
BEFORE = date(2026, 2, 27)  # the session before the signal session X0

EXECUTE: dict[str, Any] = {
    "entry": {"kind": "next_open"},
    "exits": ["daily_red", "weekly_down"],
    "sizing": {
        "risk_pct": 0.5,
        "atr_days": 14,
        "atr_mult": 2.0,
        "max_position_pct": 10.0,
        "max_gross_pct": 80.0,
    },
    "fills": {"slippage_bps": 0.0},
    "account": {"initial_equity": 100_000},
    "data": {"max_missing_sessions": 5, "max_jump_ratio": 1.8, "spinoff_max_wait": 5},
}


def cfg(**changes: Any) -> BxExecuteConfig:
    raw = {**EXECUTE, **changes}
    return BxExecuteConfig.model_validate(raw)


def plan(shares: int = 100, *, atr: float = 2.0) -> BxPlan:
    return BxPlan(signal_session=X[0], signal_close=50.0, atr=atr, shares=shares, equity=100_000.0)


def bar(day: date, o: float, h: float, lo: float, c: float) -> DailyBar:
    return DailyBar(session_date=day, open=o, high=h, low=lo, close=c)


NONE = ExitFlags(daily_red=False, weekly_down=False)
RED = ExitFlags(daily_red=True, weekly_down=False)
DOWN = ExitFlags(daily_red=False, weekly_down=True)
BOTH = ExitFlags(daily_red=True, weekly_down=True)

SIGNAL = bar(X[0], 49.0, 50.2, 48.8, 50.0)
ENTRY = bar(X[1], 50.5, 51.5, 50.0, 51.0)  # buys 100 at 50.5
B2 = bar(X[2], 51.0, 52.0, 50.5, 51.5)
B3 = bar(X[3], 51.2, 51.8, 49.0, 49.5)
B4 = bar(X[4], 49.4, 50.0, 48.8, 49.2)
B5 = bar(X[5], 48.0, 48.5, 47.0, 47.5)  # its low 47.0 comes after a full exit at the open
B6 = bar(X[6], 47.6, 48.0, 47.2, 47.8)
ENTRY_PRICE = 50.5
R_UNIT = 4.0  # atr_mult 2 x atr 2
HARD = ENTRY_PRICE - R_UNIT  # 46.5


def run(sim: BxPositionSim, steps: list[tuple[DailyBar, ExitFlags | None]]) -> list[LegResult]:
    closed: list[LegResult] = []
    for b, f in steps:
        closed += sim.step(b, f)
    return closed


def by_reason(legs: list[LegResult]) -> dict[str, LegResult]:
    out = {leg.exit_reason: leg for leg in legs}
    assert len(out) == len(legs)
    return out


def pct(price: float, basis: float = ENTRY_PRICE) -> float:
    return 100 * (price / basis - 1)


# =============================================================================================
# normal path
# =============================================================================================


def test_daily_red_then_weekly_down_sell_half_each() -> None:
    sim = BxPositionSim(plan(), cfg())
    assert run(sim, [(SIGNAL, NONE), (ENTRY, NONE)]) == []
    assert run(sim, [(B2, RED)]) == []  # decided at X2's close, not filled at X2
    [first] = sim.step(B3, RED)  # X3 open 51.2; the repeated RED is ignored (fired once)
    assert not sim.finished
    assert sim.step(B4, DOWN) == []
    [second] = sim.step(B5, NONE)  # X5 open 48.0
    assert sim.finished

    # first tranche: floor(100 / 2) = 50 at 51.2: 50 x 0.7 = +35.0, R = 0.7 / 4 = +0.175
    assert (first.leg, first.shares, first.exit_reason) == (1, 50, "daily_red")
    assert (first.entry_session, first.entry_price) == (X[1], ENTRY_PRICE)
    assert (first.exit_session, first.exit_price) == (X[3], 51.2)
    assert first.key_level == pytest.approx(ENTRY_PRICE)
    assert first.hard_stop == pytest.approx(HARD)
    assert first.pnl == pytest.approx(35.0)
    assert first.r_multiple == pytest.approx(0.175)
    assert first.holding_sessions == 3  # X1, X2, X3
    # held X1 50.0 .. 51.5 and X2 50.5 .. 52.0, exit 51.2 -> low 50.0, high 52.0
    assert first.mae_pct == pytest.approx(pct(50.0))
    assert first.mfe_pct == pytest.approx(pct(52.0))
    assert first.child_proceeds == 0.0 and first.spinoff is None

    # second: the other 50 at 48.0: 50 x -2.5 = -125.0, R = -2.5 / 4 = -0.625
    assert (second.shares, second.exit_reason) == (50, "weekly_down")
    assert (second.exit_session, second.exit_price) == (X[5], 48.0)
    assert second.entry_price == ENTRY_PRICE
    assert second.hard_stop == pytest.approx(HARD)
    assert second.pnl == pytest.approx(-125.0)
    assert second.r_multiple == pytest.approx(-0.625)
    assert second.holding_sessions == 5  # X1 .. X5
    # held X1 .. X4: low 48.8, high 52.0; the exit 48.0, not X5's later low 47.0
    assert second.mae_pct == pytest.approx(pct(48.0))
    assert second.mfe_pct == pytest.approx(pct(52.0))

    assert sim.outcome().legs == [first, second]
    assert sim.outcome().cancelled is None
    assert sim.step(B6, BOTH) == []  # finished: nothing more


def test_slippage_moves_both_fills() -> None:
    sim = BxPositionSim(plan(), cfg(fills={"slippage_bps": 5.0}))
    run(sim, [(ENTRY, NONE), (B2, RED)])
    [first] = sim.step(B3, NONE)
    buy, sell = 50.5 * (1 + 5 / 1e4), 51.2 * (1 - 5 / 1e4)  # 50.52525, 51.1744
    assert first.entry_price == pytest.approx(buy)
    assert first.exit_price == pytest.approx(sell)
    assert first.pnl == pytest.approx(50 * (sell - buy))
    assert first.hard_stop == pytest.approx(buy - R_UNIT)
    assert first.r_multiple == pytest.approx((sell - buy) / R_UNIT)


def test_both_exits_on_one_night_fill_together_at_the_next_open() -> None:
    sim = BxPositionSim(plan(), cfg())
    assert run(sim, [(ENTRY, NONE), (B2, BOTH)]) == []
    legs = sim.step(B3, NONE)
    assert sim.finished
    got = by_reason(legs)
    assert set(got) == {"daily_red", "weekly_down"}
    assert got["daily_red"].shares == 50  # floor half to the exit listed first
    assert got["weekly_down"].shares == 50
    for leg in legs:
        assert (leg.exit_session, leg.exit_price) == (X[3], 51.2)
        assert leg.pnl == pytest.approx(leg.shares * 0.7)
        assert leg.holding_sessions == 3


# =============================================================================================
# boundaries
# =============================================================================================


def test_odd_shares_sell_the_floor_half_first() -> None:
    # 101 shares: daily_red sells floor(101 / 2) = 50, weekly_down the remaining 51
    sim = BxPositionSim(plan(101), cfg())
    [first] = run(sim, [(ENTRY, NONE), (B2, RED), (B3, NONE)])
    [second] = run(sim, [(B4, DOWN), (B5, NONE)])
    assert (first.shares, second.shares) == (50, 51)
    assert second.pnl == pytest.approx(51 * (48.0 - 50.5))


def test_odd_shares_same_night_follow_the_exits_order() -> None:
    # exits listed weekly_down first: it takes the floor half
    sim = BxPositionSim(plan(101), cfg(exits=["weekly_down", "daily_red"]))
    got = by_reason(run(sim, [(ENTRY, NONE), (B2, BOTH), (B3, NONE)]))
    assert (got["weekly_down"].shares, got["daily_red"].shares) == (50, 51)
    sim = BxPositionSim(plan(101), cfg())
    got = by_reason(run(sim, [(ENTRY, NONE), (B2, BOTH), (B3, NONE)]))
    assert (got["daily_red"].shares, got["weekly_down"].shares) == (50, 51)


def test_weekly_down_first_sells_the_floor_half() -> None:
    sim = BxPositionSim(plan(101), cfg())
    [first] = run(sim, [(ENTRY, NONE), (B2, DOWN), (B3, NONE)])
    [second] = run(sim, [(B4, RED), (B5, NONE)])
    assert (first.exit_reason, first.shares) == ("weekly_down", 50)
    assert (second.exit_reason, second.shares) == ("daily_red", 51)


def test_a_one_share_position_leaves_whole_on_the_first_exit() -> None:
    # floor(1 / 2) = 0, so the first exit sells everything
    sim = BxPositionSim(plan(1), cfg())
    [only] = run(sim, [(ENTRY, NONE), (B2, RED), (B3, NONE)])
    assert sim.finished
    assert (only.shares, only.exit_reason, only.exit_price) == (1, "daily_red", 51.2)
    assert run(sim, [(B4, DOWN), (B5, NONE)]) == []
    assert sim.outcome().legs == [only]


def test_undefined_flags_fire_nothing() -> None:
    sim = BxPositionSim(plan(), cfg())
    assert run(sim, [(ENTRY, None), (B2, None), (B3, None), (B4, None)]) == []
    assert not sim.finished
    [leg] = run(sim, [(B5, RED), (B6, NONE)])  # a defined night still fires
    assert (leg.exit_session, leg.exit_price, leg.shares) == (X[6], 47.6, 50)


def test_a_flag_on_the_entry_bar_fills_at_the_following_open() -> None:
    sim = BxPositionSim(plan(), cfg())
    assert sim.step(ENTRY, RED) == []  # bought at 50.5; red read at X1's close
    [leg] = sim.step(B2, NONE)
    assert (leg.exit_session, leg.exit_price, leg.holding_sessions) == (X[2], 51.0, 2)
    assert leg.pnl == pytest.approx(50 * 0.5)
    # held X1 only (50.0 .. 51.5) plus the exit 51.0
    assert leg.mae_pct == pytest.approx(pct(50.0))
    assert leg.mfe_pct == pytest.approx(pct(51.5))


def test_no_size_cancels_at_construction() -> None:
    sim = BxPositionSim(plan(0), cfg())
    assert sim.finished
    assert sim.cancelled == "no_size"
    assert run(sim, [(ENTRY, NONE), (B2, RED)]) == []
    assert sim.outcome().legs == [] and sim.outcome().cancelled == "no_size"


def test_no_bar_for_the_entry_session_cancels_the_order() -> None:
    sim = BxPositionSim(plan(), cfg())
    sim.step(SIGNAL, NONE)
    sim.skip_session()
    assert sim.finished
    assert sim.outcome().cancelled == "no_bar"
    assert sim.step(B2, NONE) == []
    assert sim.outcome().legs == []


def test_a_skipped_session_while_held_counts_no_bar() -> None:
    sim = BxPositionSim(plan(), cfg())
    run(sim, [(ENTRY, NONE), (B2, RED)])
    sim.skip_session()
    assert not sim.finished and sim.cancelled is None
    [leg] = sim.step(B3, NONE)
    assert leg.holding_sessions == 3


def test_force_exit_sells_whatever_is_left() -> None:
    sim = BxPositionSim(plan(101), cfg())
    [first] = run(sim, [(ENTRY, NONE), (B2, RED), (B3, NONE), (B4, NONE)])
    [rest] = sim.force_exit(X[5], 48.3, "missing_bars")
    assert sim.finished
    assert (rest.shares, rest.exit_price, rest.exit_reason) == (51, 48.3, "missing_bars")
    assert rest.pnl == pytest.approx(51 * (48.3 - 50.5))
    assert sim.outcome().legs == [first, rest]


def test_force_exit_with_the_order_waiting_cancels_it() -> None:
    sim = BxPositionSim(plan(), cfg())
    sim.step(SIGNAL, NONE)
    assert sim.force_exit(X[1], 50.5, "jump") == []
    assert sim.finished
    assert sim.outcome().cancelled == "jump"
    assert sim.outcome().legs == []


def test_a_split_while_half_held_scales_shares_and_levels() -> None:
    # After the first tranche (50 at 51.2), a 2-for-1 before X4: 100 shares at entry 25.25,
    # atr 1.0, hard 25.25 - 2 = 23.25, R unit 2.0. Sold at 24.0: 100 x -1.25 = -125, R -0.625.
    sim = BxPositionSim(plan(), cfg())
    [first] = run(sim, [(ENTRY, NONE), (B2, RED), (B3, NONE)])
    sim.apply_split(2.0)
    assert sim.step(bar(X[4], 24.6, 25.0, 24.4, 24.5), DOWN) == []
    [second] = sim.step(bar(X[5], 24.0, 24.2, 23.0, 23.5), NONE)

    assert (first.shares, first.entry_price) == (50, 50.5)  # booked before: untouched
    assert (second.shares, second.entry_price) == (100, pytest.approx(25.25))
    assert second.key_level == pytest.approx(25.25)
    assert second.hard_stop == pytest.approx(23.25)
    assert second.pnl == pytest.approx(-125.0)
    assert second.r_multiple == pytest.approx(-0.625)
    # pre-split extremes halved: X1 .. X3 low 49.0 -> 24.5, high 52.0 -> 26.0; X4 24.4 .. 25.0
    assert second.mae_pct == pytest.approx(pct(24.0, 25.25))
    assert second.mfe_pct == pytest.approx(pct(26.0, 25.25))


def test_a_split_with_the_order_waiting_scales_the_order() -> None:
    # 101 x 2 = 202 shares bought at the post-split open 25.25; atr 2 / 2 = 1 -> hard 23.25
    sim = BxPositionSim(plan(101), cfg())
    sim.step(SIGNAL, NONE)
    sim.apply_split(2.0)
    sim.step(bar(X[1], 25.25, 25.5, 25.0, 25.3), NONE)
    [leg] = run(
        sim, [(bar(X[2], 25.4, 25.6, 25.1, 25.2), RED), (bar(X[3], 25.0, 25.1, 24.8, 24.9), NONE)]
    )
    assert leg.shares == 101  # floor(202 / 2)
    assert leg.hard_stop == pytest.approx(23.25)
    assert leg.r_multiple == pytest.approx((25.0 - 25.25) / 2.0)


# =============================================================================================
# sizing
# =============================================================================================


SIZING = cfg().sizing


@pytest.mark.parametrize(
    ("close", "atr", "gross", "expected", "binding"),
    [
        # budget 500; risk 2 x 2 = 4 -> 125; position 10000 / 50 = 200; gross 80000 / 50 = 1600
        (50.0, 2.0, 0.0, 125, "risk"),
        # risk 2 x 1.5 = 3 -> floor(166.67) = 166
        (50.0, 1.5, 0.0, 166, "risk"),
        # risk 2 x 0.5 = 1 -> 500; position 200 binds
        (50.0, 0.5, 0.0, 200, "position"),
        # room 80000 - 79000 = 1000 -> 20; risk 500, position 200
        (50.0, 0.5, 79_000.0, 20, "gross"),
        (50.0, 0.5, 80_000.0, 0, "gross_full"),
        (50.0, 0.5, 90_000.0, 0, "gross_over"),
    ],
)
def test_bx_shares_takes_the_smallest_of_the_three_caps(
    close: float, atr: float, gross: float, expected: int, binding: str
) -> None:
    got = bx_shares(equity=100_000.0, close=close, atr=atr, sizing=SIZING, gross_exposure=gross)
    assert got == expected, binding
    assert got == position_shares(
        budget=500.0,
        price=close,
        hard=close - 2.0 * atr,
        equity=100_000.0,
        max_position_pct=10.0,
        max_gross_pct=80.0,
        gross_exposure=gross,
    )


@pytest.mark.parametrize("atr", [0.0, -1.0, math.nan])
def test_bx_shares_is_zero_without_a_positive_atr(atr: float) -> None:
    assert bx_shares(equity=100_000.0, close=50.0, atr=atr, sizing=SIZING, gross_exposure=0.0) == 0


def test_bx_shares_reads_atr_mult_and_risk_pct_from_the_config() -> None:
    wide = cfg(sizing={**EXECUTE["sizing"], "atr_mult": 3.0, "risk_pct": 1.0}).sizing
    # budget 1000; risk 3 x 2 = 6 -> floor(166.67) = 166
    assert bx_shares(equity=100_000.0, close=50.0, atr=2.0, sizing=wide, gross_exposure=0.0) == 166


# =============================================================================================
# leakage analogues
# =============================================================================================


def test_a_flag_never_fills_on_the_bar_it_was_read_at() -> None:
    # X2's flag is read at X2's close; whatever X2's bar looks like, nothing leaves on X2.
    sim = BxPositionSim(plan(), cfg())
    sim.step(ENTRY, NONE)
    assert sim.step(B2, BOTH) == []
    assert sim.outcome().legs == []
    legs = sim.step(B3, NONE)
    assert {leg.exit_session for leg in legs} == {X[3]}
    assert {leg.exit_price for leg in legs} == {51.2}  # X3's open, never X2's close 51.5


def test_bars_on_or_before_the_signal_session_are_ignored() -> None:
    sim = BxPositionSim(plan(), cfg())
    # red flags on bars the plan already knew must neither buy nor arm an exit
    assert sim.step(bar(BEFORE, 48.0, 48.5, 47.5, 48.2), BOTH) == []
    assert sim.step(SIGNAL, BOTH) == []
    assert sim.outcome().legs == [] and not sim.finished
    assert sim.step(ENTRY, NONE) == []
    assert sim.step(B2, NONE) == []  # nothing pending from the ignored bars
    [leg] = run(sim, [(B3, RED), (B4, NONE)])
    assert (leg.entry_session, leg.entry_price) == (X[1], 50.5)
    assert leg.exit_session == X[4]
    # the ignored bars' extremes are not part of the held range (47.5 would be the low)
    assert leg.mae_pct == pytest.approx(pct(49.0))


def test_the_outcome_does_not_depend_on_bars_after_the_last_exit() -> None:
    steps: list[tuple[DailyBar, ExitFlags | None]] = [
        (ENTRY, NONE),
        (B2, RED),
        (B3, NONE),
        (B4, DOWN),
        (B5, NONE),
    ]
    short = BxPositionSim(plan(), cfg())
    run(short, steps)
    long_ = BxPositionSim(plan(), cfg())
    run(long_, [*steps, (B6, BOTH), (bar(X[7], 1.0, 1.0, 1.0, 1.0), BOTH)])
    assert short.outcome() == long_.outcome()


def test_a_reverse_split_rounding_the_waiting_order_to_nothing_cancels_it() -> None:
    # 5 shares ordered; a 1-for-10 before the entry open leaves floor(0.5) = 0 to buy: the
    # order is cancelled for its size, not entered with no shares (which would leave no reason)
    sim = BxPositionSim(plan(5), cfg())
    sim.step(SIGNAL, NONE)
    sim.apply_split(0.1)
    assert sim.finished
    assert sim.cancelled == "no_size"
    assert sim.step(bar(X[1], 505.0, 510.0, 500.0, 505.0), NONE) == []
    assert sim.outcome().legs == []
