# Guards the B-Xtrender position against a spin-off read as information it did not have: an exit
# flag read on a frozen bar (before the factor is known), a child lost because half the shares
# left at the ex open, children paid to the wrong tranche or on shares bought after the ex-date,
# and pre-ex extremes compared raw with post-ex prices.
"""Target API (asof.execute.bx.BxPositionSim), the DecisionSim spin-off conventions
(tests/execute/test_simulate_spinoffs.py) applied to two tranches:

    begin_spinoff()          before the ex-date bar. Frozen: no new exit flag is read; a tranche
                             decided at the close before still fills at the ex open (raw) and is
                             held back, owed its children. An entry order waiting for the ex
                             open is cancelled ("spinoff_ex").
    apply_spinoff(factor)    unfreezes; the MAE / MFE basis becomes entry_price x factor and the
                             extremes seen before the ex-date are scaled by factor; extremes
                             from the ex-date on stay raw. key_level, hard_stop and the R unit
                             stay the raw record.
    credit_children(proceeds) pays the shares held at the ex-date, pro rata by shares: the
                             tranche(s) sold while owed get theirs and are returned now; the
                             shares still held carry theirs into their eventual LegResult.
    end_spinoff_unpriced()   unfreezes on raw levels, no child, spinoff = "restart"; returns the
                             held-back tranches.
    overlap_spinoff()        a second ex-date while one is unsettled: a waiting order is
                             cancelled ("spinoff_ex"); the held shares are flagged "restart".
    entitled_shares()        parent shares the children are owed on (0 when none).

Choice pinned here where the spec left one open: ``hard_stop`` and the R unit of a tranche
stay raw after an adjust (as DecisionSim keeps the raw initial hard stop), so r_multiple =
(exit - entry + child / share) / (atr_mult x raw atr).

World (made up, in the shape of the real WDC 2025-02-24 case of the DecisionSim tests): 100
shares bought at 50.5 on X1 (signal X0 close 50, ATR 2, R unit 4, hard 46.5). Ex-date X4,
0.25 child per share; X3 closes at 51.0 and the child's first close is 40.8, so the factor is
(51.0 - 0.25 x 40.8) / 51.0 = 0.8 and the children of 100 shares sell for 100 x 0.25 x 40.8 =
1020.0 (10.2 a share). The factor is known at X5's close.
"""

import pytest

from asof.execute import DailyBar, LegResult
from asof.execute.bx import BxPositionSim, ExitFlags
from tests.execute.test_bx import BOTH, DOWN, NONE, RED, SIGNAL, X, bar, cfg, pct, plan

FACTOR = 0.8
CHILD_PER_SHARE = 10.2  # 0.25 x 40.8
ENTRY_PRICE = 50.5
BASIS = ENTRY_PRICE * FACTOR  # 40.4
HARD = 46.5

ENTRY = bar(X[1], 50.5, 51.5, 50.0, 51.0)
B2 = bar(X[2], 51.0, 52.0, 50.5, 51.5)
B3 = bar(X[3], 51.5, 52.5, 50.8, 51.0)
EX = bar(X[4], 40.0, 41.0, 39.5, 40.8)  # far below the raw levels; nothing may fire on it
B5 = bar(X[5], 40.8, 41.5, 40.2, 41.2)  # known_on: the factor is known at this close
B6 = bar(X[6], 41.2, 42.0, 40.6, 41.4)
B7 = bar(X[7], 41.0, 41.6, 38.0, 39.0)
B8 = bar(X[8], 39.2, 39.5, 38.5, 39.0)
# Pre-ex extremes X1 .. X3: low 50.0, high 52.5 -> scaled 40.0, 42.0.
# Post-ex raw X4 .. X6: low 39.5, high 42.0.


def run(sim: BxPositionSim, steps: list[tuple[DailyBar, ExitFlags | None]]) -> list[LegResult]:
    closed: list[LegResult] = []
    for b, f in steps:
        closed += sim.step(b, f)
    return closed


def held_through_ex_with_red_pending() -> BxPositionSim:
    """daily_red read at X3's close; the ex-date X4 opens next."""
    sim = BxPositionSim(plan(), cfg())
    assert run(sim, [(SIGNAL, NONE), (ENTRY, NONE), (B2, NONE), (B3, RED)]) == []
    sim.begin_spinoff()
    return sim


# =============================================================================================
# normal path
# =============================================================================================


def test_a_tranche_sold_at_the_ex_open_and_the_rest_both_get_their_children() -> None:
    sim = held_through_ex_with_red_pending()
    assert sim.entitled_shares() == 100
    # X4: the red tranche fills at the ex open 40.0 and is held back; DOWN is not read (frozen)
    assert sim.step(EX, DOWN) == []
    assert sim.step(B5, DOWN) == []
    assert sim.outcome().legs == []
    assert not sim.finished
    sim.apply_spinoff(FACTOR)

    [red] = sim.credit_children(1020.0)
    # 50 of the 100 entitled shares: 510.0. pnl 50 x (40.0 - 50.5) + 510 = -15.0;
    # R = (40.0 - 50.5 + 10.2) / 4 = -0.075
    assert (red.shares, red.exit_reason) == (50, "daily_red")
    assert (red.exit_session, red.exit_price) == (X[4], 40.0)
    assert red.child_proceeds == pytest.approx(510.0)
    assert red.spinoff == "adjust"
    assert red.pnl == pytest.approx(-15.0)
    assert red.r_multiple == pytest.approx(-0.075)
    assert (red.key_level, red.hard_stop) == (pytest.approx(ENTRY_PRICE), pytest.approx(HARD))
    assert red.holding_sessions == 4  # X1 .. X4
    # basis 40.4; pre-ex 40.0 .. 42.0 scaled; the exit 40.0
    assert red.mae_pct == pytest.approx(pct(40.0, BASIS))
    assert red.mfe_pct == pytest.approx(pct(42.0, BASIS))

    assert sim.step(B6, DOWN) == []  # the first flag read after the freeze
    [down] = sim.step(B7, NONE)
    assert sim.finished
    # pnl 50 x (41.0 - 50.5) + 510 = +35.0; R = (41.0 - 50.5 + 10.2) / 4 = +0.175
    assert (down.shares, down.exit_reason, down.exit_session, down.exit_price) == (
        50,
        "weekly_down",
        X[7],
        41.0,
    )
    assert down.child_proceeds == pytest.approx(510.0)
    assert down.spinoff == "adjust"
    assert down.pnl == pytest.approx(35.0)
    assert down.r_multiple == pytest.approx(0.175)
    assert down.hard_stop == pytest.approx(HARD)
    assert down.holding_sessions == 7
    # scaled pre-ex 40.0 .. 42.0, raw post-ex 39.5 .. 42.0, exit 41.0
    assert down.mae_pct == pytest.approx(pct(39.5, BASIS))
    assert down.mfe_pct == pytest.approx(pct(42.0, BASIS))
    assert sim.outcome().legs == [red, down]


def test_half_sold_before_the_ex_date_gets_no_child() -> None:
    # daily_red at X2 -> 50 sold at X3's open 51.5 before the ex-date; only 50 shares are owed
    sim = BxPositionSim(plan(), cfg())
    [first] = run(sim, [(ENTRY, NONE), (B2, RED), (B3, NONE)])
    sim.begin_spinoff()
    assert sim.entitled_shares() == 50
    assert run(sim, [(EX, NONE), (B5, NONE)]) == []
    sim.apply_spinoff(FACTOR)
    assert sim.credit_children(50 * CHILD_PER_SHARE) == []  # nobody sold while owed
    [second] = run(sim, [(B6, DOWN), (B7, NONE)])

    assert first.child_proceeds == 0.0 and first.spinoff is None
    assert first.pnl == pytest.approx(50 * (51.5 - 50.5))
    assert second.child_proceeds == pytest.approx(510.0)
    assert second.pnl == pytest.approx(50 * (41.0 - 50.5) + 510.0)  # +35.0
    assert second.spinoff == "adjust"


# =============================================================================================
# boundaries
# =============================================================================================


def test_both_tranches_sold_at_the_ex_open_split_the_children() -> None:
    sim = BxPositionSim(plan(), cfg())
    run(sim, [(ENTRY, NONE), (B2, NONE), (B3, BOTH)])
    sim.begin_spinoff()
    assert sim.step(EX, NONE) == []
    assert not sim.finished  # sold, but owed
    sim.apply_spinoff(FACTOR)
    legs = sim.credit_children(1020.0)
    assert sim.finished
    assert sorted(leg.exit_reason for leg in legs) == ["daily_red", "weekly_down"]
    for leg in legs:
        assert (leg.shares, leg.exit_price) == (50, 40.0)
        assert leg.child_proceeds == pytest.approx(510.0)
        assert leg.pnl == pytest.approx(-15.0)


def test_odd_shares_split_the_children_pro_rata() -> None:
    # 101 shares at the ex-date; red sells 50 at the ex open, 51 stay: 1030.2 -> 510.0 / 520.2
    sim = BxPositionSim(plan(101), cfg())
    run(sim, [(ENTRY, NONE), (B2, NONE), (B3, RED)])
    sim.begin_spinoff()
    run(sim, [(EX, NONE), (B5, NONE)])
    sim.apply_spinoff(FACTOR)
    [red] = sim.credit_children(101 * CHILD_PER_SHARE)
    [down] = run(sim, [(B6, DOWN), (B7, NONE)])
    assert red.child_proceeds == pytest.approx(50 * CHILD_PER_SHARE)
    assert down.child_proceeds == pytest.approx(51 * CHILD_PER_SHARE)
    assert red.r_multiple == pytest.approx((40.0 - 50.5 + 10.2) / 4)
    assert down.r_multiple == pytest.approx((41.0 - 50.5 + 10.2) / 4)


def test_a_restart_releases_the_held_back_tranche_without_a_child() -> None:
    sim = held_through_ex_with_red_pending()
    run(sim, [(EX, DOWN), (B5, DOWN)])
    [red] = sim.end_spinoff_unpriced()

    assert (red.exit_session, red.exit_price, red.exit_reason) == (X[4], 40.0, "daily_red")
    assert red.child_proceeds == 0.0
    assert red.spinoff == "restart"
    assert red.pnl == pytest.approx(50 * (40.0 - 50.5))  # -525.0
    assert red.r_multiple == pytest.approx(-10.5 / 4)
    # raw basis 50.5: pre-ex 50.0 .. 52.5 unscaled, exit 40.0
    assert red.mae_pct == pytest.approx(pct(40.0))
    assert red.mfe_pct == pytest.approx(pct(52.5))

    [down] = run(sim, [(B6, DOWN), (B7, NONE)])
    assert (down.spinoff, down.child_proceeds) == ("restart", 0.0)
    assert down.pnl == pytest.approx(50 * (41.0 - 50.5))
    assert down.mae_pct == pytest.approx(pct(39.5))  # raw X4 low against the raw basis


def test_an_entry_order_for_the_ex_open_is_cancelled() -> None:
    sim = BxPositionSim(plan(), cfg())
    sim.step(SIGNAL, NONE)
    sim.begin_spinoff()
    assert sim.step(ENTRY, NONE) == []
    assert sim.finished
    assert sim.outcome().cancelled == "spinoff_ex"
    assert sim.outcome().legs == []
    assert sim.entitled_shares() == 0


def test_a_split_after_the_ex_date_adds_no_child() -> None:
    sim = BxPositionSim(plan(), cfg())
    run(sim, [(ENTRY, NONE), (B2, RED), (B3, NONE)])  # 50 left
    sim.begin_spinoff()
    run(sim, [(EX, NONE), (B5, NONE)])
    sim.apply_spinoff(FACTOR)
    sim.apply_split(2.0)
    assert sim.entitled_shares() == 50  # the 100 shares after the split were 50 at the ex-date
    sim.credit_children(50 * CHILD_PER_SHARE)
    [rest] = run(
        sim, [(bar(X[6], 20.6, 21.0, 20.3, 20.7), DOWN), (bar(X[7], 20.5, 20.8, 20.0, 20.1), NONE)]
    )
    assert rest.shares == 100
    assert rest.child_proceeds == pytest.approx(510.0)
    assert rest.pnl == pytest.approx(100 * (20.5 - 25.25) + 510.0)  # -475 + 510 = +35.0


def test_a_second_ex_date_during_the_settlement_flags_the_held_shares() -> None:
    sim = held_through_ex_with_red_pending()
    run(sim, [(EX, NONE), (B5, NONE)])
    sim.apply_spinoff(FACTOR)
    sim.overlap_spinoff()
    [red] = sim.credit_children(1020.0)
    [down] = run(sim, [(B6, DOWN), (B7, NONE)])
    assert red.spinoff == "adjust"  # sold before the second ex-date
    assert down.spinoff == "restart"
    assert down.child_proceeds == pytest.approx(510.0)


def test_a_second_ex_date_cancels_a_waiting_entry() -> None:
    sim = BxPositionSim(plan(), cfg())
    sim.step(SIGNAL, NONE)
    sim.overlap_spinoff()
    assert sim.finished
    assert sim.outcome().cancelled == "spinoff_ex"


# =============================================================================================
# leakage: the factor is unknown until the known_on close
# =============================================================================================


def test_flags_on_frozen_bars_are_never_read() -> None:
    # Both exits signal on the ex-date and the day after; frozen, neither may arm. After the
    # factor is known, quiet nights leave the position whole until a flag is read again.
    sim = BxPositionSim(plan(), cfg())
    run(sim, [(ENTRY, NONE), (B2, NONE), (B3, NONE)])
    sim.begin_spinoff()
    assert run(sim, [(EX, BOTH), (B5, BOTH)]) == []
    sim.apply_spinoff(FACTOR)
    assert sim.credit_children(1020.0) == []
    assert run(sim, [(B6, NONE), (B7, NONE)]) == []
    assert not sim.finished
    [red] = run(sim, [(B8, RED), (bar(X[9], 39.1, 39.4, 38.9, 39.2), NONE)])
    assert (red.exit_session, red.shares) == (X[9], 50)
    assert red.child_proceeds == pytest.approx(510.0)


def test_post_ex_lows_are_not_scaled_by_the_later_factor() -> None:
    # X4's raw low 39.5 is compared with the scaled basis 40.4, never as 39.5 x 0.8 = 31.6
    sim = held_through_ex_with_red_pending()
    run(sim, [(EX, NONE), (B5, NONE)])
    sim.apply_spinoff(FACTOR)
    sim.credit_children(1020.0)
    [down] = run(sim, [(B6, DOWN), (B7, NONE)])
    assert down.mae_pct == pytest.approx(pct(39.5, BASIS))
    assert down.mae_pct != pytest.approx(pct(39.5 * FACTOR, BASIS))


def settled_then_red_into_a_second_ex_date() -> BxPositionSim:
    """The first spin-off settled (1020 paid to the 100 shares held, 10.2 each); daily_red read
    at X7's close; a second spin-off goes ex on X8, so the red tranche fills at X8's open 39.2
    and is held back for the second event's children."""
    sim = BxPositionSim(plan(), cfg())
    run(sim, [(ENTRY, NONE), (B2, NONE), (B3, NONE)])
    sim.begin_spinoff()
    run(sim, [(EX, NONE), (B5, NONE)])
    sim.apply_spinoff(FACTOR)
    assert sim.credit_children(1020.0) == []
    run(sim, [(B6, NONE), (B7, RED)])
    sim.begin_spinoff()
    assert sim.entitled_shares() == 100
    assert sim.step(B8, NONE) == []
    assert sim.step(bar(X[9], 39.0, 39.3, 38.8, 39.1), NONE) == []
    return sim


def test_a_tranche_held_back_by_a_second_spinoff_keeps_the_first_ones_children() -> None:
    # Second event: factor 0.9, 200 for the 100 shares at its ex-date. Red: 50 x 10.2 = 510
    # from the first plus 50 / 100 x 200 = 100 from the second = 610; pnl 50 x (39.2 - 50.5)
    # + 610 = +45.0. The other 50 shares: 510 + 100 = 610 as well.
    sim = settled_then_red_into_a_second_ex_date()
    sim.apply_spinoff(0.9)
    [red] = sim.credit_children(200.0)
    assert red.child_proceeds == pytest.approx(610.0)
    assert red.pnl == pytest.approx(45.0)
    [down] = run(
        sim,
        [(bar(X[10], 39.2, 39.4, 39.0, 39.1), DOWN), (bar(X[11], 39.0, 39.1, 38.9, 39.0), NONE)],
    )
    assert down.child_proceeds == pytest.approx(610.0)


def test_an_unpriced_second_spinoff_still_pays_the_first_ones_children() -> None:
    # The second event cannot be priced: no child from it, the 510 from the first stays.
    sim = settled_then_red_into_a_second_ex_date()
    [red] = sim.end_spinoff_unpriced()
    assert red.child_proceeds == pytest.approx(510.0)
    assert red.pnl == pytest.approx(50 * (39.2 - 50.5) + 510.0)  # -55.0
