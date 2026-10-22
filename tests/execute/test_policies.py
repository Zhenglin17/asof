"""Guards against stop and sizing arithmetic that silently trades a different plan: a zone whose
width cap uses a stale key, a trailing key that ratchets down, a stop checked in the wrong order
on one bar (close before low, low before open), a pending exit that fills at the wrong open,
slippage left off stop fills, and position size computed from the next open (not knowable at
the 20:00 New York decision) instead of the decision-day close.

API under test (asof.execute); dataclasses are built with keyword arguments only:

    zone_bounds(key, atr, zone_atr, zone_cap_pct) -> (bottom, top)
        width = min(zone_atr * atr, zone_cap_pct / 100 * key); zone = key -/+ width
    hard_stop(key, atr, hard_atr) = key - hard_atr * atr
    trail_key(initial_key, highest_close, atr, trail_atr)
        = max(initial_key, highest_close - trail_atr * atr)
    DailyBar(session_date, open, high, low, close)
    LegParams(atr, zone_atr, zone_cap_pct, hard_atr, observe_days, trail_atr, max_sessions,
              slippage_bps)
    LegState(key, highest_close, observe_count, pending_exit, shares, entry_price, initial_hard,
             sessions_held)                         pending_exit: exit reason or None
    Fill(session_date, price, reason)
    step_leg(state, bar, params) -> (LegState, Fill | None)
    position_shares(*, budget, price, hard, equity, max_position_pct, max_gross_pct,
                    gross_exposure) -> int
    reentry_budget(budget, leg1_pnl, risk_mode) -> float
    adjust_for_split(state, ratio) -> LegState

Every number here is hand-made except the golden RDDT path at the bottom, whose bars are the real
2026-05-01 .. 05-11 RDDT daily bars (read off /data/asof on 2026-10-22, rounded to cents as the
spec lists them) and ATR 8.277921 (14 previous true ranges, recomputed from the same bars).

No as_of leakage case at this layer: step_leg is handed one bar and has no way to read another.
The look-ahead risk here is sizing on the next open; test_sizing_uses_the_decision_close_not_
the_next_open pins it. Assumptions to confirm on real data (standard case):
- The golden RDDT numbers in this file and test_simulate.py match a real replay of the
  2026-05-01 decision (`asof replay run --strategy earnings_gap --start 2026-04-01 --end
  2026-06-30 --set execute.fills.slippage_bps=0` then the Trade rows of the RDDT decision).
"""

from dataclasses import replace
from datetime import date

import pytest

from asof.execute import (
    DailyBar,
    Fill,
    LegParams,
    LegState,
    adjust_for_split,
    hard_stop,
    position_shares,
    reentry_budget,
    step_leg,
    trail_key,
    zone_bounds,
)

D = [date(2026, 3, d) for d in (2, 3, 4, 5, 6, 9, 10, 11, 12, 13)]
# atr 2: zone half-width min(0.25 * 2, 1% of key) = 0.5 for key 100; hard = key - 2
P = LegParams(
    atr=2.0,
    zone_atr=0.25,
    zone_cap_pct=1.0,
    hard_atr=1.0,
    observe_days=3,
    trail_atr=1.5,
    max_sessions=60,
    slippage_bps=0.0,
)
# key 100: zone 99.5 .. 100.5, hard 98
S0 = LegState(
    key=100.0,
    highest_close=100.0,
    observe_count=0,
    pending_exit=None,
    shares=10,
    entry_price=101.0,
    initial_hard=98.0,
    sessions_held=1,
)


def bar(i: int, open: float, high: float, low: float, close: float) -> DailyBar:
    return DailyBar(session_date=D[i], open=open, high=high, low=low, close=close)


def above(i: int, close: float = 101.5) -> DailyBar:
    """A bar that touches nothing: well above the hard stop, closing above the zone."""
    return bar(i, close - 0.5, close + 0.5, close - 0.9, close)


def run(
    state: LegState, bars: list[DailyBar], params: LegParams = P
) -> tuple[LegState, Fill | None]:
    fill = None
    for b in bars:
        assert fill is None, "a bar was fed after the leg had exited"
        state, fill = step_leg(state, b, params)
    return state, fill


# =============================================================================================
# zone_bounds / hard_stop / trail_key
# =============================================================================================


def test_zone_width_is_the_atr_part_when_it_is_smaller() -> None:
    assert zone_bounds(100.0, 2.0, 0.25, 1.0) == pytest.approx((99.5, 100.5))


def test_zone_width_is_capped_by_a_percentage_of_the_key() -> None:
    # RDDT: 0.25 * 8.277921 = 2.0695 > 1% of 156.49 = 1.5649
    assert zone_bounds(156.49, 8.277921, 0.25, 1.0) == pytest.approx((154.9251, 158.0549))


def test_zone_width_when_both_limits_are_equal() -> None:
    assert zone_bounds(100.0, 4.0, 0.25, 1.0) == pytest.approx((99.0, 101.0))


def test_hard_stop_is_key_minus_atr_multiple() -> None:
    assert hard_stop(156.49, 8.277921, 1.0) == pytest.approx(148.212079)
    assert hard_stop(100.0, 2.0, 0.5) == pytest.approx(99.0)


def test_trail_key_rises_with_the_highest_close() -> None:
    assert trail_key(156.49, 169.07, 8.277921, 1.5) == pytest.approx(156.6531185)
    assert trail_key(156.49, 171.63, 8.277921, 1.5) == pytest.approx(159.2131185)


def test_trail_key_never_falls_below_the_initial_key() -> None:
    assert trail_key(156.49, 160.0, 8.277921, 1.5) == 156.49
    # exactly at the initial key
    assert trail_key(100.0, 103.0, 2.0, 1.5) == pytest.approx(100.0)


# =============================================================================================
# step_leg: normal path
# =============================================================================================


def test_a_quiet_bar_above_the_zone_holds_and_counts_a_session() -> None:
    state, fill = step_leg(S0, above(0), P)
    assert fill is None
    assert state.pending_exit is None
    assert state.observe_count == 0
    assert state.highest_close == 101.5
    assert state.key == 100.0  # 101.5 - 3 = 98.5 < 100
    assert state.sessions_held == 2
    assert state.shares == 10


def test_close_below_the_zone_exits_at_the_next_open() -> None:
    state, fill = step_leg(S0, bar(0, 100.2, 100.4, 99.0, 99.4), P)
    assert fill is None
    assert state.pending_exit == "break"
    state, fill = step_leg(state, bar(1, 99.0, 99.8, 98.9, 99.6), P)
    assert fill == Fill(session_date=D[1], price=99.0, reason="break")


def test_three_closes_inside_the_zone_time_out() -> None:
    inside = [bar(i, 100.3, 100.6, 99.7, 100.0) for i in range(3)]
    state, fill = run(S0, inside[:2])
    assert fill is None and state.observe_count == 2 and state.pending_exit is None
    state, fill = run(state, inside[2:])
    assert fill is None
    assert state.pending_exit == "observe_timeout"
    state, fill = step_leg(state, bar(3, 100.1, 100.2, 99.9, 100.0), P)
    assert fill == Fill(session_date=D[3], price=100.1, reason="observe_timeout")


def test_trailing_key_moves_the_next_bars_zone_and_hard_stop() -> None:
    # close 110 -> key max(100, 110 - 3) = 107: next zone 106.5 .. 107.5, hard 105
    state, fill = step_leg(S0, above(0, 110.0), P)
    assert fill is None
    assert state.highest_close == 110.0
    assert state.key == pytest.approx(107.0)
    # under the old key this close (106.4) would be far above the zone; under the new one it
    # is a break
    broke, fill = step_leg(state, bar(1, 107.2, 107.5, 106.0, 106.4), P)
    assert fill is None and broke.pending_exit == "break"
    # and the hard stop moved to 105
    _, fill = step_leg(state, bar(1, 107.0, 107.4, 104.9, 106.0), P)
    assert fill is not None
    assert (fill.session_date, fill.reason) == (D[1], "hard")
    assert fill.price == pytest.approx(105.0)


# =============================================================================================
# step_leg: boundaries and order within one bar
# =============================================================================================


def test_open_below_the_hard_stop_fills_at_the_open() -> None:
    _, fill = step_leg(S0, bar(0, 97.5, 99.0, 97.0, 98.5), P)
    assert fill == Fill(session_date=D[0], price=97.5, reason="hard_gap")


def test_open_exactly_at_the_hard_stop_is_a_gap_exit() -> None:
    _, fill = step_leg(S0, bar(0, 98.0, 99.0, 97.0, 98.5), P)
    assert fill == Fill(session_date=D[0], price=98.0, reason="hard_gap")


def test_low_through_the_hard_stop_fills_at_the_stop_not_the_low() -> None:
    _, fill = step_leg(S0, bar(0, 100.8, 101.0, 97.0, 100.9), P)
    assert fill == Fill(session_date=D[0], price=98.0, reason="hard")


def test_low_exactly_at_the_hard_stop_exits() -> None:
    _, fill = step_leg(S0, bar(0, 100.8, 101.0, 98.0, 100.9), P)
    assert fill == Fill(session_date=D[0], price=98.0, reason="hard")


def test_low_one_cent_above_the_hard_stop_holds() -> None:
    state, fill = step_leg(S0, bar(0, 100.8, 101.0, 98.01, 100.9), P)
    assert fill is None and state.pending_exit is None


@pytest.mark.parametrize("close", [99.5, 100.5], ids=["at_bottom", "at_top"])
def test_close_on_a_zone_edge_is_inside(close: float) -> None:
    state, fill = step_leg(S0, bar(0, 100.0, 101.0, 99.0, close), P)
    assert fill is None
    assert state.pending_exit is None
    assert state.observe_count == 1


def test_a_close_above_the_zone_resets_the_observe_count() -> None:
    inside = bar(0, 100.3, 100.6, 99.7, 100.0)
    state, fill = run(S0, [inside, replace(inside, session_date=D[1]), above(2)])
    assert fill is None and state.observe_count == 0
    state, fill = run(state, [replace(inside, session_date=D[3])])
    assert fill is None and state.observe_count == 1 and state.pending_exit is None


def test_a_pending_exit_fills_at_the_open_even_below_the_hard_stop() -> None:
    # decided at the previous close; the gap through the stop does not change the reason
    pending = replace(S0, pending_exit="break")
    _, fill = step_leg(pending, bar(0, 97.0, 97.5, 96.0, 97.2), P)
    assert fill == Fill(session_date=D[0], price=97.0, reason="break")


def test_a_hard_exit_ends_the_bar() -> None:
    # low through the stop AND close below the zone: the hard fill wins, nothing is pending
    state, fill = step_leg(S0, bar(0, 100.0, 100.2, 97.0, 97.5), P)
    assert fill is not None and fill.reason == "hard"


def test_max_sessions_exits_at_the_next_open() -> None:
    params = replace(P, max_sessions=2)
    state, fill = step_leg(S0, above(0), params)  # sessions_held 1 -> 2
    assert fill is None
    assert state.sessions_held == 2
    assert state.pending_exit == "max_sessions"
    _, fill = step_leg(state, above(1), params)
    assert fill == Fill(session_date=D[1], price=101.0, reason="max_sessions")


def test_one_session_before_max_sessions_holds() -> None:
    state, fill = step_leg(S0, above(0), replace(P, max_sessions=3))
    assert fill is None and state.pending_exit is None


def test_a_break_on_the_last_allowed_session_is_reported_as_a_break() -> None:
    state, _ = step_leg(S0, bar(0, 100.2, 100.4, 99.0, 99.4), replace(P, max_sessions=2))
    assert state.pending_exit == "break"


def test_slippage_applies_to_stop_and_open_fills() -> None:
    params = replace(P, slippage_bps=5.0)
    _, fill = step_leg(S0, bar(0, 100.8, 101.0, 97.0, 100.9), params)
    assert fill is not None and fill.price == pytest.approx(98.0 * (1 - 5 / 1e4))
    _, fill = step_leg(replace(S0, pending_exit="break"), bar(0, 99.0, 99.5, 98.5, 99.2), params)
    assert fill is not None and fill.price == pytest.approx(99.0 * (1 - 5 / 1e4))
    _, fill = step_leg(S0, bar(0, 97.5, 99.0, 97.0, 98.5), params)
    assert fill is not None and fill.price == pytest.approx(97.5 * (1 - 5 / 1e4))


def test_the_key_only_ratchets_up() -> None:
    state, _ = run(S0, [above(0, 110.0), bar(1, 108.0, 108.5, 107.6, 107.6)])
    assert state.key == pytest.approx(107.0)
    assert state.highest_close == 110.0


# =============================================================================================
# sizing
# =============================================================================================


def test_sizing_risks_the_budget_between_decision_close_and_hard_stop() -> None:
    # RDDT 2026-05-01: floor(500 / (166.48 - 148.212079)) = 27
    shares = position_shares(
        budget=500.0,
        price=166.48,
        hard=148.212079,
        equity=100_000.0,
        max_position_pct=10.0,
        max_gross_pct=100.0,
        gross_exposure=0.0,
    )
    assert shares == 27


def test_sizing_uses_the_decision_close_not_the_next_open() -> None:
    # Leakage at the sizing step: the 05-04 open (170.08) is not known at 05-01 20:00 New York
    # and would give floor(500 / 21.868) = 22. The caller passes the decision close; the size
    # must be the one computable then.
    at_close = position_shares(
        budget=500.0, price=166.48, hard=148.212079, equity=100_000.0,
        max_position_pct=10.0, max_gross_pct=100.0, gross_exposure=0.0,
    )  # fmt: skip
    at_open = position_shares(
        budget=500.0, price=170.08, hard=148.212079, equity=100_000.0,
        max_position_pct=10.0, max_gross_pct=100.0, gross_exposure=0.0,
    )  # fmt: skip
    assert (at_close, at_open) == (27, 22)


def test_sizing_exact_division_is_not_rounded_down_a_share() -> None:
    shares = position_shares(
        budget=500.0, price=105.0, hard=100.0, equity=100_000.0,
        max_position_pct=20.0, max_gross_pct=100.0, gross_exposure=0.0,
    )  # fmt: skip
    assert shares == 100  # 500 / 5 exactly; the position cap (190) does not bind


def test_sizing_is_capped_by_position_size() -> None:
    shares = position_shares(
        budget=500.0, price=100.0, hard=99.9, equity=100_000.0,
        max_position_pct=10.0, max_gross_pct=100.0, gross_exposure=0.0,
    )  # fmt: skip
    assert shares == 100


def test_sizing_is_capped_by_gross_exposure() -> None:
    shares = position_shares(
        budget=500.0, price=100.0, hard=99.9, equity=100_000.0,
        max_position_pct=10.0, max_gross_pct=100.0, gross_exposure=95_000.0,
    )  # fmt: skip
    assert shares == 50


@pytest.mark.parametrize("gross", [100_000.0, 120_000.0])
def test_no_room_left_means_no_shares(gross: float) -> None:
    shares = position_shares(
        budget=500.0, price=100.0, hard=95.0, equity=100_000.0,
        max_position_pct=10.0, max_gross_pct=100.0, gross_exposure=gross,
    )  # fmt: skip
    assert shares == 0


def test_a_stop_at_or_above_the_price_means_no_shares() -> None:
    for hard in (100.0, 100.5):
        shares = position_shares(
            budget=500.0, price=100.0, hard=hard, equity=100_000.0,
            max_position_pct=10.0, max_gross_pct=100.0, gross_exposure=0.0,
        )  # fmt: skip
        assert shares == 0


def test_shared_reentry_budget_is_what_leg_one_left() -> None:
    assert reentry_budget(500.0, -447.66, "shared") == pytest.approx(52.34)
    assert reentry_budget(500.0, 120.0, "shared") == 500.0  # a profit does not add budget
    assert reentry_budget(500.0, 0.0, "shared") == 500.0
    assert reentry_budget(500.0, -500.0, "shared") <= 0.0
    assert reentry_budget(500.0, -600.0, "shared") <= 0.0


def test_independent_reentry_budget_is_the_full_budget() -> None:
    assert reentry_budget(500.0, -447.66, "independent") == 500.0
    assert reentry_budget(500.0, -600.0, "independent") == 500.0


# =============================================================================================
# splits while holding
# =============================================================================================


def test_a_split_scales_shares_up_and_prices_down() -> None:
    state = replace(S0, key=160.0, highest_close=172.0, entry_price=170.0, initial_hard=152.0)
    after = adjust_for_split(state, 4.0)
    assert after.shares == 40
    assert after.key == pytest.approx(40.0)
    assert after.highest_close == pytest.approx(43.0)
    assert after.entry_price == pytest.approx(42.5)
    assert after.initial_hard == pytest.approx(38.0)
    # value held is unchanged
    assert after.shares * after.entry_price == pytest.approx(state.shares * state.entry_price)


def test_a_split_of_one_changes_nothing() -> None:
    assert adjust_for_split(S0, 1.0) == S0


# =============================================================================================
# golden RDDT leg 1, bar by bar (real bars, ATR 8.277921)
# =============================================================================================

RDDT_P = LegParams(
    atr=8.277921,
    zone_atr=0.25,
    zone_cap_pct=1.0,
    hard_atr=1.0,
    observe_days=3,
    trail_atr=1.5,
    max_sessions=60,
    slippage_bps=0.0,
)
RDDT_LEG1 = [
    DailyBar(session_date=date(2026, 5, 4), open=170.08, high=175.29, low=164.65, close=169.07),
    DailyBar(session_date=date(2026, 5, 5), open=171.00, high=177.13, low=167.62, close=171.63),
    DailyBar(session_date=date(2026, 5, 6), open=172.02, high=172.65, low=162.55, close=166.56),
    DailyBar(session_date=date(2026, 5, 7), open=168.50, high=169.10, low=159.60, close=163.95),
    DailyBar(session_date=date(2026, 5, 8), open=162.62, high=162.90, low=155.21, close=155.80),
    DailyBar(session_date=date(2026, 5, 11), open=153.50, high=162.45, low=153.50, close=159.51),
]


def test_golden_rddt_leg_one_path() -> None:
    state = LegState(
        key=156.49,
        highest_close=156.49,
        observe_count=0,
        pending_exit=None,
        shares=27,
        entry_price=170.08,
        initial_hard=148.212079,
        sessions_held=0,
    )
    state, fill = step_leg(state, RDDT_LEG1[0], RDDT_P)  # entry day, processed after the fill
    assert fill is None and state.key == pytest.approx(156.6531185)
    state, fill = step_leg(state, RDDT_LEG1[1], RDDT_P)
    assert fill is None and state.key == pytest.approx(159.2131185)
    assert zone_bounds(state.key, 8.277921, 0.25, 1.0) == pytest.approx((157.620987, 160.805250))
    assert hard_stop(state.key, 8.277921, 1.0) == pytest.approx(150.935198)
    state, fill = run(state, RDDT_LEG1[2:4], RDDT_P)
    assert fill is None and state.pending_exit is None
    state, fill = step_leg(state, RDDT_LEG1[4], RDDT_P)  # close 155.80 < 157.621
    assert fill is None and state.pending_exit == "break"
    state, fill = step_leg(state, RDDT_LEG1[5], RDDT_P)
    assert fill == Fill(session_date=date(2026, 5, 11), price=153.50, reason="break")
