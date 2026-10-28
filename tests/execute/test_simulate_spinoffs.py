"""Guards against a spin-off read as a crash while a leg is held: a stop fired by the ex-date
drop (the real WDC 2025-02-24 leg booked a fake hard_gap at -3.24R), price levels rescaled before
the factor was knowable, lows from before the ex-date compared raw with prices after it, a child
lost because the parent shares had already been sold at the ex open, an order filled at an open
whose price nobody could plan for, and a re-entry setup judged against a stale or unscaled level.

API under test (asof.execute.DecisionSim), as agreed for the spin-off bookkeeping change:

    begin_spinoff()          the replay calls it before the ex-date bar (the record is known at
                             00:00 New York of the ex-date). Until the event resolves, step() on
                             an open leg runs no stop / zone / observe / trailing / max_sessions
                             logic: it only records the bar's low / high and counts the session.
                             A pending_exit decided at the close before still fills at the ex
                             open (raw, with slippage) and keeps the child entitlement. Re-entry
                             checks and the gap-filled check are skipped. An order that would
                             fill at the ex open is cancelled (leg 1: cancelled = "spinoff_ex").
    apply_spinoff(factor)    after the known_on bar: key and highest_close (adjust_for_spinoff),
                             atr and pre_close scaled; held low / high seen before the ex-date and
                             the MAE / MFE basis (entry_price) scaled; lows / highs from the
                             ex-date on stay raw. Unfreezes. Returns nothing.
    end_spinoff_unpriced()   restart, or the wait ran out: unfreeze with raw levels, nothing
                             scaled, no child; a waiting re-entry setup ends; the leg is flagged
                             spinoff = "restart".
    credit_children(proceeds)
    LegResult.child_proceeds (0.0 when none), LegResult.spinoff (None / "adjust" / "restart").
        pnl = shares * (exit - entry) + child_proceeds; r_multiple with extra_per_share =
        child_proceeds / shares, raw entry and raw initial hard; mae_pct / mfe_pct relative to
        entry_price * factor for adjust legs.

Return channel chosen here (the agreed design left it open): ``credit_children`` and
``end_spinoff_unpriced`` both return ``list[LegResult]``, the legs they complete. A leg entitled
to children whose parent shares leave before the children are credited (a pending exit filled at
the ex open, or any exit after apply_spinoff but before the child sale) is held back: ``step``
returns nothing for it and ``credit_children`` returns it. A leg already credited is completed by
``step`` as before. The replay calls ``credit_children`` once per event with every child's
proceeds together.

Leakage: the factor is unknown until the known_on close. Bars between the ex-date and the
apply_spinoff call must see the raw levels and must not stop, even when they pierce both the raw
and the adjusted hard stop; post-ex lows are not rescaled (that would read a future factor into
prices that were never adjusted).

Fake data provenance: hand-made worlds of tests/execute/test_simulate.py (ATR 2, gap low 54,
pre-earnings close 50, decision close 56 -> 125 shares), extended in the shape of the real WDC
case (leg 1 in 2025-01-31, ex 2025-02-24 with 1/3 SNDK per WDC, WDC open 52.61 after a 68.705
close, factor 0.764209, SNDK first close 48.6 and 02-25 open 48.30; bar store probe). The ratio
0.25, child price 48 and factor 0.8 are round stand-ins.
"""

from pathlib import Path

import pytest

from asof.execute import DecisionSim, LegResult, simulate_decision, spinoff_proceeds
from tests.execute.test_simulate import (
    BREAK_DAY,
    ENTRY_DAY,
    PLAN,
    REACTION,
    X,
    bars_of,
    execute_cfg,
    hard_hit,
    quiet,
)

# After ENTRY_DAY (key 54) a 60.0 close trails the key to 57 (zone 56.5 .. 57.5, hard 55).
UP_DAY = (X[2], 57.0, 60.5, 56.8, 60.0)
# Ex-date X3: 0.25 child per share, child's first close 48.0 -> factor (60 - 12) / 60 = 0.8.
# The open 48.0 is far below the raw hard stop 55: unfrozen, a hard_gap.
EX_BAR = (X[3], 48.0, 49.0, 47.5, 48.4)
FACTOR = 0.8
CHILD = 125 * 0.25 * 48.0  # 1500.0, the child sold at its next open 48.0
# Adjusted from the X3 close: key 45.6, atr 1.6, zone 45.2 .. 46.0, hard 44.0, pre_close 40.


def hold(i: int) -> tuple:
    """Above the adjusted zone; its open is below the raw hard stop 55."""
    return (X[i], 48.3, 48.8, 47.9, 48.5)


def brk(i: int) -> tuple:
    """With the key trailed to 46.1 (zone 45.7 .. 46.5, hard 44.5): a close below the zone."""
    return (X[i], 46.0, 46.2, 45.0, 45.3)


def out(i: int) -> tuple:
    return (X[i], 45.0, 45.5, 44.8, 45.2)


def feed(sim: DecisionSim, rows: list[tuple]) -> list[LegResult]:
    closed: list[LegResult] = []
    for bar in bars_of(rows):
        closed += sim.step(bar)
    return closed


def held_before_ex(tmp_path: Path, *overrides: str) -> DecisionSim:
    sim = DecisionSim(PLAN, execute_cfg(tmp_path, *overrides))
    assert feed(sim, [REACTION, ENTRY_DAY, UP_DAY]) == []
    assert sim.open is not None
    assert sim.open.state.key == pytest.approx(57.0)
    return sim


# =============================================================================================
# normal path
# =============================================================================================


def test_a_leg_held_through_an_adjusted_spinoff_keeps_its_child(tmp_path: Path) -> None:
    sim = held_before_ex(tmp_path)
    sim.begin_spinoff()
    assert feed(sim, [EX_BAR]) == []  # no hard_gap at 48.0
    sim.apply_spinoff(FACTOR)

    assert sim.open is not None
    state = sim.open.state
    assert state.key == pytest.approx(45.6)
    assert state.highest_close == pytest.approx(48.0)  # 60 x 0.8; the frozen bar adds nothing
    assert (state.shares, state.entry_price, state.initial_hard) == (125, 56.5, 52.0)
    assert state.sessions_held == 3  # X1, X2 and the frozen X3
    assert sim.atr == pytest.approx(1.6)
    assert sim.pre_close == pytest.approx(40.0)

    assert feed(sim, [hold(4)]) == []
    assert sim.credit_children(spinoff_proceeds(125, [(0.25, 48.0)], 0.0)) == []
    assert sim.open.state.key == pytest.approx(46.1)  # max(45.6, 48.5 - 1.5 x 1.6)
    assert feed(sim, [brk(5)]) == []
    [leg] = feed(sim, [out(6)])

    assert (leg.leg, leg.shares, leg.entry_price) == (1, 125, 56.5)
    assert (leg.key_level, leg.hard_stop) == (pytest.approx(54.0), pytest.approx(52.0))  # raw
    assert (leg.exit_session, leg.exit_price, leg.exit_reason) == (X[6], 45.0, "break")
    assert leg.child_proceeds == pytest.approx(CHILD)
    assert leg.spinoff == "adjust"
    assert leg.pnl == pytest.approx(125 * (45.0 - 56.5) + 1500.0)  # +62.5
    assert leg.r_multiple == pytest.approx((45.0 - 56.5 + 12.0) / 4.5)  # +0.1111
    assert leg.holding_sessions == 6
    # Basis 56.5 x 0.8 = 45.2. Before X3: X1 56.0 .. 57.0 and X2 56.8 .. 60.5, times 0.8 ->
    # low 44.8, high 48.4. From X3 on, raw: 47.5 .. 49.0, 47.9 .. 48.8, 45.0 .. 46.2, exit 45.0.
    assert leg.mae_pct == pytest.approx(100 * (44.8 / 45.2 - 1))  # -0.8850
    assert leg.mfe_pct == pytest.approx(100 * (49.0 / 45.2 - 1))  # +8.4071


def test_a_restart_unfreezes_with_raw_levels_and_the_stops_work_again(tmp_path: Path) -> None:
    sim = held_before_ex(tmp_path)
    sim.begin_spinoff()
    assert feed(sim, [EX_BAR]) == []
    assert sim.end_spinoff_unpriced() == []  # the leg is still open

    assert sim.open is not None
    assert sim.open.state.key == pytest.approx(57.0)
    assert sim.atr == pytest.approx(2.0)
    [leg] = feed(sim, [hold(4)])  # open 48.3 <= raw hard 55

    assert (leg.exit_session, leg.exit_price, leg.exit_reason) == (X[4], 48.3, "hard_gap")
    assert leg.spinoff == "restart"
    assert leg.child_proceeds == 0.0
    assert leg.pnl == pytest.approx(125 * (48.3 - 56.5))  # -1025
    assert leg.r_multiple == pytest.approx((48.3 - 56.5) / 4.5)
    assert leg.holding_sessions == 4
    # nothing scaled: basis 56.5, the raw X3 low 47.5 and X2 high 60.5
    assert leg.mae_pct == pytest.approx(100 * (47.5 / 56.5 - 1))
    assert leg.mfe_pct == pytest.approx(100 * (60.5 / 56.5 - 1))


# =============================================================================================
# boundaries
# =============================================================================================


def test_a_frozen_bar_does_not_end_the_leg_on_max_sessions(tmp_path: Path) -> None:
    # max_sessions 3: X3 is the third session held. Frozen, it decides nothing; the first bar
    # after the adjustment (X4, the fourth) does, and the leg leaves at X5's open.
    sim = held_before_ex(tmp_path, "execute.exit.max_sessions=3")
    sim.begin_spinoff()
    assert feed(sim, [EX_BAR]) == []
    sim.apply_spinoff(FACTOR)
    assert feed(sim, [hold(4)]) == []
    assert sim.credit_children(CHILD) == []
    [leg] = feed(sim, [brk(5)])
    assert (leg.exit_session, leg.exit_price, leg.exit_reason) == (X[5], 46.0, "max_sessions")
    assert leg.pnl == pytest.approx(125 * (46.0 - 56.5) + CHILD)


def test_an_exit_before_the_child_is_sold_waits_for_the_credit(tmp_path: Path) -> None:
    # The child has no bar for a while (the replay values it later); the parent leg breaks and
    # fills meanwhile. Its result is not complete until the proceeds are known.
    sim = held_before_ex(tmp_path)
    sim.begin_spinoff()
    feed(sim, [EX_BAR])
    sim.apply_spinoff(FACTOR)
    assert feed(sim, [hold(4), brk(5), out(6)]) == []
    assert sim.legs == []
    assert not sim.finished
    [leg] = sim.credit_children(CHILD)
    assert (leg.exit_session, leg.exit_price, leg.exit_reason) == (X[6], 45.0, "break")
    assert leg.child_proceeds == pytest.approx(CHILD)
    assert leg.pnl == pytest.approx(62.5)


# A break decided at X2's close (53.4 < 53.5) fills at the ex-date open X3. The previous close
# 53.4 and a child close of 42.72 at 0.25 give factor (53.4 - 10.68) / 53.4 = 0.8.
PENDING_EX = (X[3], 43.0, 43.5, 42.8, 43.0)
PENDING_CHILD = 125 * 0.25 * 42.72  # 1335.0


def test_a_pending_exit_fills_at_the_ex_open_and_still_gets_the_child(tmp_path: Path) -> None:
    sim = DecisionSim(PLAN, execute_cfg(tmp_path))
    assert feed(sim, [REACTION, ENTRY_DAY, BREAK_DAY]) == []
    sim.begin_spinoff()
    assert feed(sim, [PENDING_EX]) == []  # sold at 43.0, held back until the child is paid
    assert sim.legs == []
    assert not sim.finished
    sim.apply_spinoff(FACTOR)
    assert feed(sim, [(X[4], 43.0, 43.5, 42.8, 43.2)]) == []
    [leg] = sim.credit_children(spinoff_proceeds(125, [(0.25, 42.72)], 0.0))

    assert (leg.exit_session, leg.exit_price, leg.exit_reason) == (X[3], 43.0, "break")
    assert leg.child_proceeds == pytest.approx(PENDING_CHILD)
    assert leg.spinoff == "adjust"
    assert leg.pnl == pytest.approx(125 * (43.0 - 56.5) + PENDING_CHILD)  # -352.5
    assert leg.r_multiple == pytest.approx((43.0 - 56.5 + 10.68) / 4.5)
    assert leg.holding_sessions == 3
    # basis 45.2; X1 56.0 .. 57.0 and X2 53.0 .. 56.2 times 0.8; the exit 43.0 is raw
    assert leg.mae_pct == pytest.approx(100 * (42.4 / 45.2 - 1))
    assert leg.mfe_pct == pytest.approx(100 * (45.6 / 45.2 - 1))
    assert sim.legs == [leg]


def test_a_pending_exit_at_the_ex_open_of_a_restart_is_booked_without_a_child(
    tmp_path: Path,
) -> None:
    sim = DecisionSim(PLAN, execute_cfg(tmp_path))
    assert feed(sim, [REACTION, ENTRY_DAY, BREAK_DAY]) == []
    sim.begin_spinoff()
    assert feed(sim, [PENDING_EX]) == []
    [leg] = sim.end_spinoff_unpriced()
    assert (leg.exit_session, leg.exit_price, leg.exit_reason) == (X[3], 43.0, "break")
    assert leg.child_proceeds == 0.0
    assert leg.spinoff == "restart"
    assert leg.pnl == pytest.approx(125 * (43.0 - 56.5))


def test_a_reentry_after_an_adjusted_spinoff_uses_the_scaled_pre_earnings_close(
    tmp_path: Path,
) -> None:
    # Leg 1 as in the test above. The frozen X3 closes at 43.0, below the raw re-entry zone
    # 49.5 .. 50.5: had the gap-filled check run, the setup would be dead. Scaled, the zone is
    # 39.6 .. 40.4 (pre_close 40, atr 1.6); X5 touches it and closes above its top.
    sim = DecisionSim(PLAN, execute_cfg(tmp_path))
    feed(sim, [REACTION, ENTRY_DAY, BREAK_DAY])
    sim.begin_spinoff()
    feed(sim, [PENDING_EX])
    sim.apply_spinoff(FACTOR)
    feed(sim, [(X[4], 43.0, 43.5, 42.8, 43.2)])
    [first] = sim.credit_children(PENDING_CHILD)
    closed = feed(
        sim,
        [
            (X[5], 41.0, 41.2, 40.2, 40.8),  # confirms
            (X[6], 41.0, 41.5, 40.9, 41.2),  # leg 2 in at 41.0
            (X[7], 40.0, 40.1, 38.0, 38.5),  # hard 40 - 1.6 = 38.4
        ],
    )
    [second] = closed
    assert second.leg == 2
    assert second.key_level == pytest.approx(40.0)
    assert second.hard_stop == pytest.approx(38.4)
    # shared budget 500 - 352.5 (leg 1 with its child) = 147.5; risk 40.8 - 38.4 -> 61 shares
    assert second.shares == 61
    assert (second.entry_session, second.entry_price) == (X[6], 41.0)
    assert (second.exit_price, second.exit_reason) == (pytest.approx(38.4), "hard")
    assert second.spinoff is None
    assert second.child_proceeds == 0.0
    assert sim.outcome().legs == [first, second]


def test_a_waiting_setup_across_an_adjust_without_a_held_leg(tmp_path: Path) -> None:
    # Leg 1 left at X3 (before the ex-date X4) with nothing to receive; the setup waits and its
    # level is scaled. Leg 1 itself is left exactly as it was booked.
    sim = DecisionSim(PLAN, execute_cfg(tmp_path))
    [first] = feed(sim, [REACTION, ENTRY_DAY, BREAK_DAY, (X[3], 53.4, 53.5, 52.0, 52.2)])
    sim.begin_spinoff()
    feed(sim, [quiet(4)])
    sim.apply_spinoff(FACTOR)
    assert sim.pre_close == pytest.approx(40.0)
    assert sim.legs == [first]
    assert first.spinoff is None and first.child_proceeds == 0.0
    [second] = feed(
        sim,
        [
            (X[5], 40.6, 41.0, 40.3, 40.6),  # low 40.3 <= 40.4, close 40.6 >= 40.4
            (X[6], 40.8, 41.0, 40.7, 40.9),
            (X[7], 40.0, 40.1, 38.0, 38.5),
        ],
    )
    # budget 500 - 387.5 = 112.5; risk 40.6 - 38.4 = 2.2 -> 51 shares
    assert (second.leg, second.shares, second.entry_price) == (2, 51, 40.8)
    assert second.key_level == pytest.approx(40.0)
    assert (second.exit_price, second.exit_reason) == (pytest.approx(38.4), "hard")


def test_a_restart_ends_a_waiting_reentry_setup(tmp_path: Path) -> None:
    rows = [REACTION, ENTRY_DAY, BREAK_DAY, (X[3], 53.4, 53.5, 52.0, 52.2), quiet(4)]
    confirm = [(X[5], 51.5, 51.8, 50.4, 50.9), (X[6], 51.2, 52.0, 51.0, 51.8), hard_hit(7)]
    # control: without the event the same bars re-enter
    control = simulate_decision(PLAN, bars_of(rows + confirm), execute_cfg(tmp_path))
    assert [leg.leg for leg in control.legs] == [1, 2]

    sim = DecisionSim(PLAN, execute_cfg(tmp_path))
    [first] = feed(sim, rows[:4])
    sim.begin_spinoff()
    feed(sim, rows[4:])
    assert sim.end_spinoff_unpriced() == []
    assert sim.finished
    assert feed(sim, confirm) == []
    assert sim.outcome().legs == [first]
    assert first.spinoff is None


def test_a_leg_one_order_for_the_ex_open_is_cancelled(tmp_path: Path) -> None:
    sim = DecisionSim(PLAN, execute_cfg(tmp_path))
    feed(sim, [REACTION])
    sim.begin_spinoff()
    assert feed(sim, [ENTRY_DAY]) == []
    assert sim.finished
    outcome = sim.outcome()
    assert outcome.legs == []
    assert outcome.cancelled == "spinoff_ex"


def test_a_leg_two_order_for_the_ex_open_is_cancelled(tmp_path: Path) -> None:
    # Leg 1 leaves at X3 and X3 confirms the re-entry; X4 is the ex-date.
    sim = DecisionSim(PLAN, execute_cfg(tmp_path))
    [first] = feed(sim, [REACTION, ENTRY_DAY, BREAK_DAY, (X[3], 53.4, 53.5, 50.2, 50.8)])
    assert sim.order is not None and sim.order.leg == 2
    sim.begin_spinoff()
    assert feed(sim, [(X[4], 51.0, 51.5, 47.0, 47.5)]) == []
    assert sim.order is None
    assert sim.finished
    outcome = sim.outcome()
    assert outcome.legs == [first]
    assert outcome.cancelled is None  # leg 1 filled; only leg 2 never happened


# =============================================================================================
# leakage: the factor is unknown until the known_on close
# =============================================================================================


def test_bars_before_the_factor_is_known_see_raw_levels_and_never_stop(tmp_path: Path) -> None:
    # known_on X5: X4's open 44.0 and low 43.0 pierce the raw hard stop (55) and the adjusted
    # one (44.0) alike. Nothing may fire and nothing may be scaled yet.
    sim = held_before_ex(tmp_path)
    sim.begin_spinoff()
    assert feed(sim, [EX_BAR, (X[4], 44.0, 48.8, 43.0, 48.5), hold(5)]) == []
    assert sim.open is not None
    state = sim.open.state
    assert state.key == pytest.approx(57.0)
    assert state.highest_close == pytest.approx(60.0)
    assert state.pending_exit is None
    assert state.sessions_held == 5
    assert sim.atr == pytest.approx(2.0)
    assert sim.pre_close == pytest.approx(50.0)

    sim.apply_spinoff(FACTOR)
    assert feed(sim, [hold(6)]) == []
    sim.credit_children(CHILD)
    assert feed(sim, [brk(7)]) == []
    [leg] = feed(sim, [out(8)])
    assert (leg.exit_session, leg.exit_reason) == (X[8], "break")
    # X4's 43.0 came after the ex-date: raw against the adjusted basis 45.2, never x 0.8
    assert leg.mae_pct == pytest.approx(100 * (43.0 / 45.2 - 1))


# =============================================================================================
# review follow-ups: re-entry while leg 1 is unpaid, frozen closes, later splits, overlaps
# =============================================================================================


def test_a_reentry_confirmed_before_the_child_is_paid_counts_leg_one_loss(tmp_path: Path) -> None:
    # Leg 1 sold at the ex open at 43.0 (-1687.5 on the parent shares), its child not yet paid.
    # X4 confirms the scaled re-entry zone the evening after the factor: the shared budget is
    # 500 - 1687.5 < 0, so no leg 2 -- not the full 500 a loss left out would give.
    sim = DecisionSim(PLAN, execute_cfg(tmp_path))
    feed(sim, [REACTION, ENTRY_DAY, BREAK_DAY])
    sim.begin_spinoff()
    feed(sim, [PENDING_EX])
    sim.apply_spinoff(FACTOR)
    assert feed(sim, [(X[4], 41.0, 41.2, 40.2, 40.8)]) == []
    assert sim.order is None
    [leg] = sim.credit_children(PENDING_CHILD)
    assert sim.finished
    assert sim.outcome().legs == [leg]


def test_a_frozen_close_below_the_scaled_zone_ends_the_setup(tmp_path: Path) -> None:
    # Leg 1 left before the ex-date; the frozen X4 closes at 39.0, below the scaled re-entry zone
    # 39.6 .. 40.4: once the factor is known the gap counts as filled.
    sim = DecisionSim(PLAN, execute_cfg(tmp_path))
    [first] = feed(sim, [REACTION, ENTRY_DAY, BREAK_DAY, (X[3], 53.4, 53.5, 52.0, 52.2)])
    sim.begin_spinoff()
    feed(sim, [(X[4], 40.0, 40.5, 38.8, 39.0)])
    sim.apply_spinoff(FACTOR)
    assert sim.finished
    assert feed(sim, [(X[5], 40.6, 41.0, 40.3, 40.6), (X[6], 40.8, 41.0, 40.7, 40.9)]) == []
    assert sim.outcome().legs == [first]


def test_a_split_after_the_ex_date_adds_no_child(tmp_path: Path) -> None:
    sim = held_before_ex(tmp_path)
    sim.begin_spinoff()
    feed(sim, [EX_BAR])
    sim.apply_spinoff(FACTOR)
    sim.apply_split(2.0)
    assert sim.open is not None and sim.open.state.shares == 250
    assert sim.entitled_shares() == 125


def test_a_second_spinoff_during_the_settlement_flags_the_leg(tmp_path: Path) -> None:
    sim = held_before_ex(tmp_path)
    sim.begin_spinoff()
    feed(sim, [EX_BAR])
    sim.apply_spinoff(FACTOR)
    sim.overlap_spinoff()
    assert sim.credit_children(CHILD) == []
    assert feed(sim, [hold(4), brk(5)]) == []
    [leg] = feed(sim, [out(6)])
    assert leg.spinoff == "restart"
    assert leg.child_proceeds == pytest.approx(CHILD)
