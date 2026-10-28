"""Guards against a decision being simulated with information it did not have, or with a plan
other than the approved one: leg 1 sized off the next open, a re-entry that fires before leg 1
has exited, after the window, after the setup was invalidated, more often than max_count, or
with risk the shared budget no longer has; an entry that goes ahead although the open is already
below the zone; and fills that change when bars after the exit are added.

API under test (asof.execute):

    DecisionPlan(reaction_session, decision_close, gap_low, pre_earnings_close, atr, equity)
    simulate_decision(plan, bars: Sequence[DailyBar], execute: ExecuteConfig) -> DecisionOutcome
        bars: the security's daily bars from (at least) the reaction session on; entry is at the
        open of the first bar after the reaction session.
    DecisionOutcome(legs: list[LegResult], cancelled: str | None)
    LegResult(leg, entry_session, entry_price, shares, key_level, hard_stop, exit_session,
              exit_price, exit_reason, pnl, r_multiple)

The golden case uses the real RDDT daily bars of 2026-04-30 .. 05-26 (read off /data/asof on
2026-10-22; 05-14 open 155.775 and 05-26 open 143.495 are listed rounded to cents as the spec
does) and ATR 8.277921 recomputed from the 14 true ranges before 05-01. The other worlds are
hand-made: ATR 2, gap low 54 (zone 53.5 .. 54.5, hard 52), pre-earnings close 50 (zone 49.5 ..
50.5, hard 48), decision close 56 -> 125 shares.

Leakage: bars after a fill must not change it (truncation / edited future), and the size must
not move when the next open moves. Assumption to confirm on real data (standard case): the
golden legs below equal the Trade rows a real replay writes for RDDT's 2026-05-01 decision.
"""

from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from asof.execute import DailyBar, DecisionPlan, DecisionSim, simulate_decision
from asof.strategy import apply_overrides, load_strategy
from tests.strategy.files import write_strategy


def execute_cfg(tmp_path: Path, *overrides: str):  # returns an ExecuteConfig
    base = load_strategy(write_strategy(tmp_path / "strategies"))
    return apply_overrides(base, ["execute.fills.slippage_bps=0", *overrides]).execute


def bars_of(rows: list[tuple[date, float, float, float, float]]) -> list[DailyBar]:
    return [DailyBar(session_date=d, open=o, high=h, low=lo, close=c) for d, o, h, lo, c in rows]


# =============================================================================================
# golden: RDDT 2026-05-01
# =============================================================================================

RDDT = bars_of(
    [
        (date(2026, 4, 30), 146.85, 150.20, 143.39, 147.23),
        (date(2026, 5, 1), 165.51, 173.00, 156.49, 166.48),
        (date(2026, 5, 4), 170.08, 175.29, 164.65, 169.07),
        (date(2026, 5, 5), 171.00, 177.13, 167.62, 171.63),
        (date(2026, 5, 6), 172.02, 172.65, 162.55, 166.56),
        (date(2026, 5, 7), 168.50, 169.10, 159.60, 163.95),
        (date(2026, 5, 8), 162.62, 162.90, 155.21, 155.80),
        (date(2026, 5, 11), 153.50, 162.45, 153.50, 159.51),
        (date(2026, 5, 12), 159.30, 161.04, 150.32, 152.35),
        (date(2026, 5, 13), 152.00, 154.70, 148.50, 154.12),
        (date(2026, 5, 14), 155.78, 157.23, 149.03, 156.31),
        (date(2026, 5, 15), 153.55, 159.16, 153.14, 158.17),
        (date(2026, 5, 18), 157.00, 160.98, 153.57, 159.11),
        (date(2026, 5, 19), 158.23, 161.49, 154.15, 154.88),
        (date(2026, 5, 20), 154.02, 154.76, 143.91, 146.72),
        (date(2026, 5, 21), 144.31, 153.45, 144.10, 150.04),
        (date(2026, 5, 22), 145.23, 145.79, 140.67, 141.67),
        (date(2026, 5, 26), 143.50, 145.89, 139.55, 144.64),
    ]
)
RDDT_PLAN = DecisionPlan(
    reaction_session=date(2026, 5, 1),
    decision_close=166.48,
    gap_low=156.49,
    pre_earnings_close=147.23,
    atr=8.277921,
    equity=100_000.0,
)
HARD_1 = 156.49 - 8.277921  # 148.212079
HARD_2 = 147.23 - 8.277921  # 138.952079


def test_golden_rddt_two_legs_shared_risk(tmp_path: Path) -> None:
    outcome = simulate_decision(RDDT_PLAN, RDDT, execute_cfg(tmp_path))
    assert outcome.cancelled is None
    first, second = outcome.legs

    assert first.leg == 1
    assert first.key_level == pytest.approx(156.49)
    assert first.hard_stop == pytest.approx(HARD_1)
    assert first.shares == 27
    assert (first.entry_session, first.entry_price) == (date(2026, 5, 4), pytest.approx(170.08))
    assert (first.exit_session, first.exit_price) == (date(2026, 5, 11), pytest.approx(153.50))
    assert first.exit_reason == "break"
    assert first.pnl == pytest.approx(-447.66)
    assert first.r_multiple == pytest.approx(-16.58 / (170.08 - HARD_1))  # -0.7582

    # re-entry zone 145.758 .. 148.702; 05-13 low 148.50, close 154.12 confirms
    assert second.leg == 2
    assert second.key_level == pytest.approx(147.23)
    assert second.hard_stop == pytest.approx(HARD_2)
    assert second.shares == 3  # floor((500 - 447.66) / (154.12 - 138.952079))
    assert (second.entry_session, second.entry_price) == (date(2026, 5, 14), pytest.approx(155.78))
    # 05-20 close 146.72 inside the zone, 05-21 close 150.04 resets, 05-22 close 141.67 breaks
    assert (second.exit_session, second.exit_price) == (date(2026, 5, 26), pytest.approx(143.50))
    assert second.exit_reason == "break"
    assert second.pnl == pytest.approx(-36.84)
    assert second.r_multiple == pytest.approx(-12.28 / (155.78 - HARD_2))  # -0.7297

    assert first.pnl + second.pnl == pytest.approx(-484.50)


def test_golden_rddt_independent_risk(tmp_path: Path) -> None:
    outcome = simulate_decision(
        RDDT_PLAN, RDDT, execute_cfg(tmp_path, "execute.reentry.risk_mode=independent")
    )
    first, second = outcome.legs
    assert first.shares == 27
    assert second.shares == 32  # floor(500 / 15.167921)
    assert second.pnl == pytest.approx(32 * (143.50 - 155.78))
    assert first.pnl + second.pnl == pytest.approx(-840.62)


def test_golden_rddt_with_five_bps_slippage(tmp_path: Path) -> None:
    cfg = execute_cfg(tmp_path, "execute.fills.slippage_bps=5")
    first, second = simulate_decision(RDDT_PLAN, RDDT, cfg).legs
    buy, sell = 1 + 5 / 1e4, 1 - 5 / 1e4
    # sizing still uses the decision close and the unslipped stop
    assert first.shares == 27
    assert first.entry_price == pytest.approx(170.08 * buy)
    assert first.exit_price == pytest.approx(153.50 * sell)
    assert first.pnl == pytest.approx(27 * (153.50 * sell - 170.08 * buy))  # -452.028
    assert first.r_multiple == pytest.approx(
        (153.50 * sell - 170.08 * buy) / (170.08 * buy - HARD_1)
    )
    # 500 - 452.028 = 47.97 -> floor(47.97 / 15.168) = 3
    assert second.shares == 3
    assert second.entry_price == pytest.approx(155.78 * buy)
    assert second.exit_price == pytest.approx(143.50 * sell)
    assert first.pnl + second.pnl == pytest.approx(-489.31725)


# =============================================================================================
# hand-made worlds: ATR 2, gap low 54, pre-earnings close 50, decision close 56
# =============================================================================================

X = [d.date() for d in pd.bdate_range("2026-03-02", periods=40)]
PLAN = DecisionPlan(
    reaction_session=X[0],
    decision_close=56.0,
    gap_low=54.0,
    pre_earnings_close=50.0,
    atr=2.0,
    equity=100_000.0,
)
REACTION = (X[0], 55.0, 57.0, 54.0, 56.0)
ENTRY_DAY = (X[1], 56.5, 57.0, 56.0, 56.8)  # entry at 56.5, closes above the zone
BREAK_DAY = (X[2], 56.0, 56.2, 53.0, 53.4)  # 53.4 < 53.5 -> exit at the next open


def quiet(i: int) -> tuple[date, float, float, float, float]:
    """Above the re-entry zone (low 51.5 > 50.5): neither confirms nor invalidates."""
    return (X[i], 52.0, 52.5, 51.5, 52.0)


def hard_hit(i: int) -> tuple[date, float, float, float, float]:
    return (X[i], 51.0, 51.2, 47.0, 47.5)


def test_reentry_confirmed_on_leg_one_exit_day(tmp_path: Path) -> None:
    bars = bars_of(
        [
            REACTION,
            ENTRY_DAY,
            BREAK_DAY,
            (X[3], 53.4, 53.5, 50.2, 50.8),  # leg 1 exits at 53.4; low 50.2, close 50.8 confirm
            (X[4], 51.0, 51.5, 47.0, 47.5),  # leg 2 enters at 51.0 and hits its hard stop 48
            quiet(5),
        ]
    )
    first, second = simulate_decision(PLAN, bars, execute_cfg(tmp_path)).legs
    assert (first.shares, first.entry_price, first.hard_stop) == (125, 56.5, pytest.approx(52.0))
    assert (first.exit_session, first.exit_price, first.exit_reason) == (X[3], 53.4, "break")
    assert first.pnl == pytest.approx(-387.5)
    assert first.r_multiple == pytest.approx(-3.1 / 4.5)
    # budget 500 - 387.5 = 112.5; risk per share 50.8 - 48 = 2.8 -> 40 shares
    assert second.shares == 40
    assert (second.entry_session, second.entry_price) == (X[4], 51.0)
    assert second.key_level == pytest.approx(50.0)
    assert second.hard_stop == pytest.approx(48.0)
    assert (second.exit_session, second.exit_price, second.exit_reason) == (X[4], 48.0, "hard")
    assert second.r_multiple == pytest.approx(-1.0)


def test_a_close_below_the_reentry_zone_invalidates_it_for_good(tmp_path: Path) -> None:
    bars = bars_of(
        [
            REACTION,
            ENTRY_DAY,
            BREAK_DAY,
            (X[3], 53.4, 53.5, 48.9, 49.0),  # leg 1 exits; close 49.0 < 49.5 kills the setup
            (X[4], 49.2, 51.0, 49.1, 50.9),  # would confirm
            quiet(5),
            quiet(6),
        ]
    )
    outcome = simulate_decision(PLAN, bars, execute_cfg(tmp_path))
    assert [leg.leg for leg in outcome.legs] == [1]


def test_shared_budget_used_up_means_no_reentry(tmp_path: Path) -> None:
    bars = bars_of(
        [
            REACTION,
            ENTRY_DAY,
            (X[2], 51.0, 51.5, 50.4, 51.0),  # gap through 52: -687.5; this bar would confirm
            (X[3], 51.2, 52.0, 51.0, 51.8),
            (X[4], 51.5, 51.6, 47.5, 48.2),
            quiet(5),
        ]
    )
    outcome = simulate_decision(PLAN, bars, execute_cfg(tmp_path))
    [first] = outcome.legs
    assert (first.exit_price, first.exit_reason) == (51.0, "hard_gap")
    assert first.pnl == pytest.approx(-687.5)


def test_independent_budget_reenters_after_a_large_loss(tmp_path: Path) -> None:
    bars = bars_of(
        [
            REACTION,
            ENTRY_DAY,
            (X[2], 51.0, 51.5, 50.4, 51.0),
            (X[3], 51.2, 52.0, 51.0, 51.8),
            (X[4], 51.5, 51.6, 47.5, 48.2),
            quiet(5),
        ]
    )
    cfg = execute_cfg(tmp_path, "execute.reentry.risk_mode=independent")
    first, second = simulate_decision(PLAN, bars, cfg).legs
    assert second.shares == 166  # floor(500 / (51.0 - 48))
    assert (second.entry_session, second.entry_price) == (X[3], 51.2)
    assert (second.exit_price, second.exit_reason) == (48.0, "hard")


@pytest.mark.parametrize(("confirm_at", "reenters"), [(20, True), (21, False)])
def test_reentry_window_counts_sessions_after_the_reaction(
    tmp_path: Path, confirm_at: int, reenters: bool
) -> None:
    rows = [REACTION, ENTRY_DAY, BREAK_DAY, (X[3], 53.4, 53.5, 52.0, 52.2)]
    rows += [quiet(i) for i in range(4, confirm_at)]
    rows += [
        (X[confirm_at], 51.5, 51.8, 50.4, 50.9),
        (X[confirm_at + 1], 51.2, 52.0, 51.0, 51.8),
        hard_hit(confirm_at + 2),
        quiet(confirm_at + 3),
    ]
    outcome = simulate_decision(PLAN, bars_of(rows), execute_cfg(tmp_path))
    assert len(outcome.legs) == (2 if reenters else 1)
    if reenters:
        assert outcome.legs[1].entry_session == X[confirm_at + 1]


def test_at_most_max_count_reentries(tmp_path: Path) -> None:
    bars = bars_of(
        [
            REACTION,
            ENTRY_DAY,
            BREAK_DAY,
            (X[3], 53.4, 53.5, 50.2, 50.8),  # leg 1 exits, re-entry confirmed
            (X[4], 51.0, 51.2, 50.0, 50.2),  # leg 2 in at 51.0; close inside 49.5 .. 50.5
            (X[5], 50.2, 50.6, 49.9, 50.3),
            (X[6], 50.3, 50.5, 49.8, 50.1),  # third inside close: observe timeout
            (X[7], 50.2, 51.5, 50.1, 51.2),  # leg 2 out at 50.2; this bar would confirm again
            (X[8], 51.4, 52.0, 51.0, 51.6),
            hard_hit(9),
            quiet(10),
        ]
    )
    outcome = simulate_decision(PLAN, bars, execute_cfg(tmp_path))
    assert [leg.leg for leg in outcome.legs] == [1, 2]
    second = outcome.legs[1]
    assert (second.exit_session, second.exit_price, second.exit_reason) == (
        X[7],
        50.2,
        "observe_timeout",
    )


def test_entry_cancelled_when_the_open_is_at_the_zone_bottom(tmp_path: Path) -> None:
    bars = bars_of([REACTION, (X[1], 53.5, 54.0, 53.4, 53.9), quiet(2), quiet(3)])
    outcome = simulate_decision(PLAN, bars, execute_cfg(tmp_path))
    assert outcome.legs == []
    assert outcome.cancelled == "entry_below_zone"


def test_entry_goes_ahead_one_cent_above_the_zone_bottom(tmp_path: Path) -> None:
    bars = bars_of(
        [REACTION, (X[1], 53.51, 54.0, 53.4, 53.9), (X[2], 53.0, 53.2, 51.5, 52.0), quiet(3)]
    )
    outcome = simulate_decision(PLAN, bars, execute_cfg(tmp_path))
    assert outcome.cancelled is None
    first = outcome.legs[0]
    assert (first.entry_session, first.entry_price) == (X[1], 53.51)
    assert (first.exit_session, first.exit_price, first.exit_reason) == (X[2], 52.0, "hard")


# =============================================================================================
# leakage: later bars and the next open never change what was decided before them
# =============================================================================================


def test_size_does_not_move_with_the_next_open(tmp_path: Path) -> None:
    moved = [replace(b, open=175.0) if b.session_date == date(2026, 5, 4) else b for b in RDDT]
    first = simulate_decision(RDDT_PLAN, moved, execute_cfg(tmp_path)).legs[0]
    assert first.shares == 27
    assert first.entry_price == pytest.approx(175.0)


def test_bars_after_the_last_exit_change_nothing(tmp_path: Path) -> None:
    cfg = execute_cfg(tmp_path)
    future = bars_of(
        [
            (date(2026, 5, 27), 144.50, 158.55, 144.24, 154.19),
            (date(2026, 5, 28), 140.0, 300.0, 1.0, 290.0),  # absurd on purpose
        ]
    )
    assert simulate_decision(RDDT_PLAN, RDDT + future, cfg) == simulate_decision(
        RDDT_PLAN, RDDT, cfg
    )


def test_leg_one_is_the_same_whether_or_not_later_bars_exist(tmp_path: Path) -> None:
    cfg = execute_cfg(tmp_path)
    upto_exit = [b for b in RDDT if b.session_date <= date(2026, 5, 11)]
    assert (
        simulate_decision(RDDT_PLAN, upto_exit, cfg).legs[0]
        == simulate_decision(RDDT_PLAN, RDDT, cfg).legs[0]
    )


# =============================================================================================
# excursions (MAE / MFE) per leg
#
# mae_pct = 100 * (lowest price seen while held / entry_price - 1), mfe_pct likewise with the
# highest. Seen while held: the entry bar's low and high (the fill is at its open, so its whole
# range comes after the fill), every later bar's low and high before the exit bar, and the exit
# price. The exit bar adds only the exit price (and its open before an intraday stop): a daily
# bar does not say whether its high or low came before or after the fill. A leg entered and
# exited on one bar sees only its entry and exit prices (plus that open). Not clipped at 0.
# =============================================================================================


def pct(price: float, entry: float) -> float:
    return 100 * (price / entry - 1)


def test_golden_rddt_excursions(tmp_path: Path) -> None:
    first, second = simulate_decision(RDDT_PLAN, RDDT, execute_cfg(tmp_path)).legs
    # leg 1 holds 05-04 .. 05-08 (lows 164.65 167.62 162.55 159.60 155.21, highs 175.29 177.13
    # 172.65 169.10 162.90) and exits 05-11 at 153.50, below every low it saw
    assert first.mae_pct == pytest.approx(pct(153.50, 170.08))  # -9.7484
    assert first.mfe_pct == pytest.approx(pct(177.13, 170.08))  # +4.1451
    # leg 2 holds 05-14 .. 05-22 (lowest low 140.67 on 05-22, highest high 161.49 on 05-19) and
    # exits 05-26 at the open 143.50; 05-26's own low 139.55 is not counted
    assert second.mae_pct == pytest.approx(pct(140.67, 155.78))  # -9.6996
    assert second.mfe_pct == pytest.approx(pct(161.49, 155.78))  # +3.6654


def test_excursions_of_a_multi_bar_leg_and_a_same_bar_reentry(tmp_path: Path) -> None:
    bars = bars_of(
        [
            REACTION,
            ENTRY_DAY,  # entry 56.5; low 56.0, high 57.0 count
            BREAK_DAY,  # low 53.0, high 56.2 count
            (X[3], 53.4, 53.5, 50.2, 50.8),  # exit bar: only the fill 53.4 counts, not low 50.2
            (X[4], 51.0, 51.5, 47.0, 47.5),  # leg 2 in at 51.0 and out at 48.0 on this bar
            quiet(5),
        ]
    )
    first, second = simulate_decision(PLAN, bars, execute_cfg(tmp_path)).legs
    # leg 1: lowest 53.0, highest 57.0
    assert first.mae_pct == pytest.approx(pct(53.0, 56.5))  # -6.1947
    assert first.mfe_pct == pytest.approx(pct(57.0, 56.5))  # +0.8850
    # leg 2 has its own window and one bar: only 51.0 and 48.0 (not X4's 47.0 / 51.5, not leg
    # 1's 57.0)
    assert second.mae_pct == pytest.approx(pct(48.0, 51.0))  # -5.8824
    assert second.mfe_pct == pytest.approx(0.0)


def test_excursions_when_the_open_gaps_through_the_hard_stop(tmp_path: Path) -> None:
    bars = bars_of(
        [
            REACTION,
            ENTRY_DAY,
            (X[2], 51.0, 51.5, 50.4, 51.0),  # hard_gap at the open 51.0; low 50.4 not counted
            (X[3], 51.2, 52.0, 51.0, 51.8),
            (X[4], 51.5, 51.6, 47.5, 48.2),
            quiet(5),
        ]
    )
    [first] = simulate_decision(PLAN, bars, execute_cfg(tmp_path)).legs
    assert first.exit_reason == "hard_gap"
    assert first.mae_pct == pytest.approx(pct(51.0, 56.5))  # -9.7345
    assert first.mfe_pct == pytest.approx(pct(57.0, 56.5))  # +0.8850


def test_excursions_when_an_intraday_hard_stop_fills(tmp_path: Path) -> None:
    bars = bars_of(
        [
            REACTION,
            (X[1], 53.51, 54.0, 53.4, 53.9),  # entry 53.51; low 53.4, high 54.0 count
            (X[2], 53.0, 53.2, 51.5, 52.0),  # low 51.5 <= 52 -> out at 52.0; 51.5 not counted
            quiet(3),
        ]
    )
    [first] = simulate_decision(PLAN, bars, execute_cfg(tmp_path)).legs
    assert (first.exit_price, first.exit_reason) == (52.0, "hard")
    assert first.mae_pct == pytest.approx(pct(52.0, 53.51))  # -2.8219
    assert first.mfe_pct == pytest.approx(pct(54.0, 53.51))  # +0.9157


def test_the_open_before_an_intraday_stop_counts(tmp_path: Path) -> None:
    # X2 gaps up to 60 and then trades down through the hard stop 52: the open came before the
    # stop filled, so it was seen while held; the rest of X2's range was not.
    bars = bars_of([REACTION, ENTRY_DAY, (X[2], 60.0, 61.0, 51.0, 51.5), quiet(3)])
    [first] = simulate_decision(PLAN, bars, execute_cfg(tmp_path)).legs
    assert (first.exit_price, first.exit_reason) == (52.0, "hard")
    assert first.mfe_pct == pytest.approx(pct(60.0, 56.5))  # +6.1947, not 61.0's +7.96
    assert first.mae_pct == pytest.approx(pct(52.0, 56.5))  # the fill, not 51.0


def test_excursions_are_not_clipped_at_zero(tmp_path: Path) -> None:
    # 5 bps: entry 56.5 * 1.0005 = 56.52825, above the entry bar's high 56.5 and every later
    # high, so the best price seen is below the fill and mfe is negative.
    cfg = execute_cfg(tmp_path, "execute.fills.slippage_bps=5")
    bars = bars_of(
        [
            REACTION,
            (X[1], 56.5, 56.5, 55.0, 55.2),  # close above the zone, key stays 54
            (X[2], 55.0, 55.4, 53.0, 53.4),  # close 53.4 < 53.5 -> break at the next open
            (X[3], 53.4, 53.5, 52.2, 52.4),  # out at 53.4 * 0.9995 = 53.3733
        ]
    )
    first = simulate_decision(PLAN, bars, cfg).legs[0]
    entry = 56.5 * (1 + 5 / 1e4)
    assert first.entry_price == pytest.approx(entry)
    assert first.mae_pct == pytest.approx(pct(53.0, entry))  # -6.2416
    assert first.mfe_pct == pytest.approx(pct(56.5, entry))  # -0.0500
    assert first.mfe_pct < 0


def test_a_split_while_held_leaves_the_excursions_in_percent(tmp_path: Path) -> None:
    # 2:1 before X2: entry 56.5 -> 28.25, the entry bar's 56.0 .. 57.0 -> 28.0 .. 28.5, key 27
    # (zone 26.75 .. 27.25 with ATR 1, hard 26). X3 closes 26.5 -> break, out at X4's 26.6.
    sim = DecisionSim(PLAN, execute_cfg(tmp_path))
    for bar in bars_of([REACTION, ENTRY_DAY]):
        sim.step(bar)
    sim.apply_split(2.0)
    closed = []
    for bar in bars_of(
        [
            (X[2], 28.1, 28.3, 27.9, 28.2),
            (X[3], 28.0, 28.1, 26.4, 26.5),
            (X[4], 26.6, 26.7, 26.3, 26.4),
        ]
    ):
        closed += sim.step(bar)
    [leg] = closed
    assert (leg.entry_price, leg.exit_price) == (pytest.approx(28.25), 26.6)
    # lowest 26.4 (X3), highest 28.5 (the entry bar's 57.0 after the split)
    assert leg.mae_pct == pytest.approx(pct(26.4, 28.25))  # -6.5487
    assert leg.mfe_pct == pytest.approx(pct(28.5, 28.25))  # +0.8850 = pct(57.0, 56.5)


def test_a_forced_exit_counts_the_bars_stepped_and_the_forced_price(tmp_path: Path) -> None:
    sim = DecisionSim(PLAN, execute_cfg(tmp_path))
    for bar in bars_of([REACTION, ENTRY_DAY, (X[2], 56.6, 57.2, 56.0, 57.0)]):
        assert sim.step(bar) == []
    leg = sim.force_exit(X[7], 57.0, "data_gap")
    assert leg is not None
    assert leg.mae_pct == pytest.approx(pct(56.0, 56.5))  # -0.8850
    assert leg.mfe_pct == pytest.approx(pct(57.2, 56.5))  # +1.2389


def test_a_forced_exit_below_every_low_is_the_mae(tmp_path: Path) -> None:
    sim = DecisionSim(PLAN, execute_cfg(tmp_path))
    for bar in bars_of([REACTION, ENTRY_DAY, (X[2], 56.6, 57.2, 56.0, 57.0)]):
        sim.step(bar)
    leg = sim.force_exit(X[3], 55.0, "price_jump")
    assert leg is not None
    assert leg.mae_pct == pytest.approx(pct(55.0, 56.5))  # -2.6549
    assert leg.mfe_pct == pytest.approx(pct(57.2, 56.5))


# leakage: nothing at or after the exit bar's range moves the excursions


def test_bars_after_the_exit_do_not_change_the_excursions(tmp_path: Path) -> None:
    cfg = execute_cfg(tmp_path)
    rows = [
        REACTION,
        ENTRY_DAY,
        (X[2], 51.0, 51.5, 50.4, 51.0),  # hard_gap, the only leg (budget used up)
    ]
    future = [(X[3], 51.2, 400.0, 0.5, 51.8), (X[4], 51.5, 900.0, 0.1, 48.2), quiet(5)]
    [short] = simulate_decision(PLAN, bars_of(rows), cfg).legs
    [full] = simulate_decision(PLAN, bars_of(rows + future), cfg).legs
    assert (full.mae_pct, full.mfe_pct) == (short.mae_pct, short.mfe_pct)
    assert full.mae_pct == pytest.approx(pct(51.0, 56.5))
    assert full == short


@pytest.mark.parametrize(
    ("high", "low"), [(99.0, 50.2), (53.5, 1.0), (99.0, 1.0)], ids=["high", "low", "both"]
)
def test_the_exit_bars_range_does_not_change_the_excursions(
    tmp_path: Path, high: float, low: float
) -> None:
    # Leg 1 exits at X3's open; leg 2 enters and exits on X4. Editing the range of either exit
    # bar (keeping its open and close, so every fill and the re-entry stay as they were) must
    # leave both legs' excursions alone.
    cfg = execute_cfg(tmp_path)

    def world(x3: tuple[float, float], x4: tuple[float, float]) -> list[DailyBar]:
        return bars_of(
            [
                REACTION,
                ENTRY_DAY,
                BREAK_DAY,
                (X[3], 53.4, x3[0], x3[1], 50.8),
                (X[4], 51.0, x4[0], x4[1], 47.5),
                quiet(5),
            ]
        )

    base = simulate_decision(PLAN, world((53.5, 50.2), (51.5, 47.0)), cfg).legs
    edited = simulate_decision(
        PLAN, world((high, low), (max(high, 51.5), min(low, 47.0))), cfg
    ).legs
    assert [(leg.exit_price, leg.exit_reason) for leg in edited] == [
        (leg.exit_price, leg.exit_reason) for leg in base
    ]
    assert [(leg.mae_pct, leg.mfe_pct) for leg in edited] == [
        (leg.mae_pct, leg.mfe_pct) for leg in base
    ]
    assert base[0].mae_pct == pytest.approx(pct(53.0, 56.5))
    assert base[1].mfe_pct == pytest.approx(0.0)


def test_a_reverse_split_rounding_the_waiting_order_to_nothing_cancels_it(tmp_path: Path) -> None:
    # 125 shares ordered at the reaction close; a 1-for-200 before the entry open leaves
    # floor(0.625) = 0: cancelled for its size, never a leg of 0 shares
    sim = DecisionSim(PLAN, execute_cfg(tmp_path))
    for bar in bars_of([REACTION]):
        sim.step(bar)
    sim.apply_split(0.005)
    assert sim.finished
    assert sim.cancelled == "no_size"
    assert sim.step(bars_of([(X[1], 11_300.0, 11_400.0, 11_200.0, 11_360.0)])[0]) == []
    assert sim.outcome().legs == []
