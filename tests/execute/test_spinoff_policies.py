"""Guards against spin-off arithmetic that books a distribution the holder never had or loses one
the holder did get: price levels rescaled by the wrong factor (or the share count, entry or R
unit rescaled with them), children credited as whole shares only, slippage left off the child
sale, and an R multiple that ignores the child's value.

API under test (asof.execute):

    adjust_for_spinoff(state: LegState, factor: float) -> LegState
        key and highest_close multiplied by factor; shares, entry_price, initial_hard,
        observe_count, pending_exit and sessions_held unchanged; factor == 1 returns the state
        unchanged.
    spinoff_proceeds(shares, children: Sequence[(ratio, sale price)], slippage_bps) -> float
        sum(shares * ratio * sell_price(price, bps)); fractional child shares are paid at the
        same price (cash in lieu).
    r_multiple(entry_price, exit_price, hard, extra_per_share=0.0)
        = (exit - entry + extra_per_share) / (entry - hard)

No as_of leakage case at this layer: these functions are handed numbers and cannot read any
other bar or record. The look-ahead rule (the factor is unknown before the event's known_on
close) is pinned in tests/execute/test_simulate_spinoffs.py and tests/replay/
test_replay_spinoffs.py.

Fake data provenance: hand-made numbers in the shape of the real WDC case (ex 2025-02-24, 1/3
SanDisk per WDC; factor 0.764209 from the bar store). The 0.25 ratio and 0.8 factor are chosen
round. ASSUMPTION (not checked against any deal's terms): fractional child shares are paid in
cash at the same sale price. Check on real data: the WDC standard case listed in
tests/replay/test_replay_spinoffs.py.
"""

from dataclasses import replace

import pytest

from asof.execute import LegState, adjust_for_spinoff, r_multiple, spinoff_proceeds

# A leg in the canonical shape: 125 shares in at 56.5, initial hard 52, key trailed to 57 after
# a 60 close.
HELD = LegState(
    key=57.0,
    highest_close=60.0,
    observe_count=2,
    pending_exit=None,
    shares=125,
    entry_price=56.5,
    initial_hard=52.0,
    sessions_held=3,
)


# =============================================================================================
# normal path
# =============================================================================================


def test_a_spinoff_scales_the_key_and_the_highest_close_only() -> None:
    after = adjust_for_spinoff(HELD, 0.8)
    assert after.key == pytest.approx(45.6)
    assert after.highest_close == pytest.approx(48.0)
    # the position and its record are the raw entry-time ones: no share is added or removed
    assert after.shares == 125
    assert after.entry_price == 56.5
    assert after.initial_hard == 52.0
    assert after.observe_count == 2
    assert after.pending_exit is None
    assert after.sessions_held == 3


def test_a_pending_exit_survives_the_adjustment() -> None:
    state = replace(HELD, pending_exit="break")
    assert adjust_for_spinoff(state, 0.8).pending_exit == "break"


def test_proceeds_of_one_child() -> None:
    # 125 shares x 0.25 child per share x 48.0
    assert spinoff_proceeds(125, [(0.25, 48.0)], 0.0) == pytest.approx(1500.0)


def test_proceeds_of_several_children_add_up() -> None:
    # 125 x (0.25 x 32 + 0.5 x 8) = 125 x 12
    assert spinoff_proceeds(125, [(0.25, 32.0), (0.5, 8.0)], 0.0) == pytest.approx(1500.0)


def test_child_sale_pays_the_slippage() -> None:
    assert spinoff_proceeds(125, [(0.25, 48.0)], 10.0) == pytest.approx(1500.0 * (1 - 0.001))


def test_r_multiple_counts_the_child_value_per_share() -> None:
    # out at 45.0, child worth 1500 / 125 = 12 per share, risk 56.5 - 52 = 4.5
    assert r_multiple(56.5, 45.0, 52.0, extra_per_share=12.0) == pytest.approx(0.5 / 4.5)


# =============================================================================================
# boundaries
# =============================================================================================


def test_a_factor_of_one_changes_nothing() -> None:
    assert adjust_for_spinoff(HELD, 1.0) == HELD


def test_fractional_child_shares_are_paid_not_floored() -> None:
    # 10 shares x 1/3 = 3.333 children at 30: 100.0, not floor(3.333) x 30 = 90.0
    assert spinoff_proceeds(10, [(1 / 3, 30.0)], 0.0) == pytest.approx(100.0)


def test_no_children_pay_nothing() -> None:
    assert spinoff_proceeds(125, [], 5.0) == 0.0


def test_r_multiple_without_extra_is_unchanged() -> None:
    assert r_multiple(56.5, 53.4, 52.0) == pytest.approx(-3.1 / 4.5)
    assert r_multiple(56.5, 53.4, 52.0, extra_per_share=0.0) == pytest.approx(-3.1 / 4.5)
