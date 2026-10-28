"""Guards against the replay reading a spin-off as a crash or paying it out wrongly: a held leg
stopped by the ex-date drop (real WDC 2025-02-24: a fake hard_gap at -3.24R, pnl -1821), a
factor or child price used before it was knowable, a record or child bar used before it was on
file, a child paid twice or never, a leg left frozen past the strategy's wait, and other
securities' trades moved by one security's event.

API under test (asof.replay.run_replay), as agreed for the spin-off bookkeeping change:
- Each evening, an active decision whose security has a spin-off record visible at as_of with
  ex_date in (last_session, day] gets DecisionSim.begin_spinoff() before that bar is stepped
  (the security is matched by all its symbols / its security_id).
- Resolution reads asof.store.spinoffs.spinoff_events, only rows with known_on <= day: adjust ->
  apply_spinoff(factor) after the known_on bar; restart -> end_spinoff_unpriced() at its
  known_on. The ex-date bar is wait session 0 (the parent's bars counted); not adjust-resolved
  by the close of session ``execute.data.spinoff_max_wait`` -> end_spinoff_unpriced().
- Children are sold at each child's first open after known_on, read under the same as_of rules;
  a child with no bar for ``max_missing_sessions`` sessions is valued at its last visible
  close. Proceeds = spinoff_proceeds(shares, children, the replay's slippage).
- Trade gains child_proceeds (0.0 when none) and spinoff (None / "adjust" / "restart").

Interface points this file pins down where the design left a choice:
- A leg whose children are still unpaid when its parent shares leave (a pending exit filled at
  the ex open, or any exit before the child sale) is booked when the children are credited:
  its Trade's exit_at stays the parent's fill, its available_at is the close of the evening the
  proceeds were known. Earlier, the result was not knowable.
- A decision whose leg-1 order would have filled at the ex open is recorded as cancelled
  "spinoff_ex" (the existing payload["cancelled"] channel).

Fake data provenance: tests/replay/spinoff_world.py (WDC shape, round numbers). Slippage 0.

Assumptions the fixtures cannot settle; check them on the real store (standard case):
- WDC (security S008254), leg 1 entered 2025-01-31 at 65.2326, 144 shares, key 63.7, hard
  61.3269: a real replay over 2025-01-02 .. 2025-03-31 must not exit it on 2025-02-24
  (exit_reason hard_gap), must book spinoff = "adjust", and child_proceeds = 144 x (1/3) x the
  SNDK 2025-02-25 open 48.30 x (1 - bps / 1e4) = 2318.40 x (1 - bps / 1e4). Command:
  ``asof replay run --strategy earnings_gap --start 2025-01-02 --end 2025-03-31`` then the
  Trade rows WHERE security_id = 'S008254'.
- Every real adjust event's children have a bar within max_missing_sessions of known_on (else
  the last-close fallback prices the sale). Check: for each adjust row of spinoff_events at
  today's as_of, sessions between known_on and each child's next bar; anything above 5 is listed.
- The factor of spinoff_events equals the parent's real price continuity (audit S6 reports the
  ex-date total returns of adjust events; the WDC row reads factor 0.764209).
"""

from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

import duckdb
import pytest
from sqlmodel import Session

from asof.evolve.null import SpinoffAdjust, build_panel, null_returns, spans_of_run
from asof.replay import run_replay
from asof.store.models import Decision
from asof.store.spinoffs import spinoff_events
from asof.strategy import apply_overrides, load_strategy
from tests.replay.scenario import TRADED, X
from tests.replay.spinoff_world import (
    BREAK,
    CHILD_PROCEEDS,
    EX_BAR,
    EXIT,
    FILLER,
    HOLD,
    KIDZ,
    PRE,
    WDC_AFTER,
    Child,
    spin_strategy,
    spin_world,
)
from tests.replay.test_replay import ZERO_SLIP, attributions, ny_date, rows, trade_key, trades
from tests.scan.world import EARNINGS, GAP, RELEASE, Ohlcv, close_of, filing
from tests.store.test_market import write_splits
from tests.store.test_spinoffs import spin_row, write_other_rows

START = date(2026, 2, 12)  # tests/replay/scenario.START: two sessions before D
LATE = datetime(2026, 10, 27, tzinfo=UTC)


@pytest.fixture
def strategy_dir(tmp_path: Path) -> Path:
    return spin_strategy(tmp_path / "strategies")


def end_of(after: list[Ohlcv | None]) -> date:
    return X[len(after)]


def replay(
    session: Session,
    con: duckdb.DuckDBPyConnection,
    root: Path,
    strategy_dir: Path,
    end: date,
    *overrides: str,
) -> int:
    return run_replay(session, con, root, strategy_dir, START, end, (*ZERO_SLIP, *overrides))


# =============================================================================================
# the fixture itself (passes today: shows a failure below is not a fixture error)
# =============================================================================================


def test_the_fixture_event_is_an_adjust_known_on_the_ex_date(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection
) -> None:
    world = spin_world(tmp_path, session)
    [event] = spinoff_events(con, world.root, LATE, rejected=()).to_pylist()
    assert (event["security_id"], event["ex_date"], event["status"]) == ("S_GAPX", X[3], "adjust")
    assert event["factor"] == pytest.approx(0.8)
    assert event["known_on"] == X[3]
    assert event["child_symbols"] == ["KIDZ"]


# =============================================================================================
# normal path
# =============================================================================================


def test_a_leg_held_through_a_spinoff_is_not_stopped_and_books_the_child(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    world = spin_world(tmp_path, session)
    run_id = replay(session, con, world.root, strategy_dir, end_of(WDC_AFTER))

    [only] = trades(session, run_id)
    assert (only.leg, only.shares, only.entry_price) == (1, 125, 56.5)
    assert (only.key_level, only.hard_stop) == (pytest.approx(54.0), pytest.approx(52.0))
    assert only.risk_per_share == pytest.approx(4.5)
    assert (ny_date(only.exit_at), only.exit_price, only.exit_reason) == (X[6], 45.0, "break")
    assert only.child_proceeds == pytest.approx(CHILD_PROCEEDS)
    assert only.spinoff == "adjust"
    assert only.pnl == pytest.approx(125 * (45.0 - 56.5) + CHILD_PROCEEDS)  # +62.5
    assert only.r_multiple == pytest.approx((45.0 - 56.5 + 12.0) / 4.5)
    assert only.holding_sessions == 6
    assert only.mae_pct == pytest.approx(100 * (44.8 / 45.2 - 1))
    assert only.mfe_pct == pytest.approx(100 * (49.0 / 45.2 - 1))
    assert only.available_at == close_of(X[6])
    [attribution] = attributions(session, run_id)
    assert attribution.realized_return == pytest.approx(62.5 / 100_000)


def test_several_children_of_one_day_are_one_event(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    # 0.25 KIDA at 32 + 0.5 KIDB at 8 = 12 per share, factor 0.8 as with KIDZ alone.
    children = (Child("KIDA", 0.25, 3, 32.0), Child("KIDB", 0.5, 3, 8.0))
    world = spin_world(tmp_path, session, children=children)
    run_id = replay(session, con, world.root, strategy_dir, end_of(WDC_AFTER))
    [only] = trades(session, run_id)
    assert (ny_date(only.exit_at), only.exit_price, only.exit_reason) == (X[6], 45.0, "break")
    assert only.child_proceeds == pytest.approx(125 * (0.25 * 32.0 + 0.5 * 8.0))
    assert only.pnl == pytest.approx(62.5)


def othr_traded_world(tmp_path: Path, session: Session, **kwargs: object):  # -> World
    """OTHR gaps on D too, with its own release, and trades the scenario's two-leg path."""
    othr = {X[0]: GAP, **{X[i + 1]: bar for i, bar in enumerate(TRADED[:4]) if bar is not None}}
    world = spin_world(tmp_path, session, othr_bars=othr, **kwargs)  # type: ignore[arg-type]
    filing(
        session,
        source_id=world.source_id,
        entity_id=world.othr,
        accession="0000002222-26-000009",
        available_at=RELEASE,
        exhibits=EARNINGS,
    )
    return world


def test_other_securities_trades_do_not_change_with_a_spinoff_record(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    world = othr_traded_world(tmp_path, session, records=[])
    end = end_of(WDC_AFTER)
    without = replay(session, con, world.root, strategy_dir, end)
    write_other_rows(world.root, [spin_row("GAPX", "KIDZ", X[3], 0.25)])
    with_record = replay(session, con, world.root, strategy_dir, end)

    def othr(run_id: int) -> list[tuple[object, ...]]:
        return [
            (*trade_key(t), t.child_proceeds, t.spinoff)
            for t in trades(session, run_id)
            if t.security_id == "S_OTHR"
        ]

    assert len(othr(without)) == 2
    assert othr(with_record) == othr(without)
    assert all(t.spinoff is None for t in trades(session, with_record) if t.security_id == "S_OTHR")
    # and GAPX is the one that changed: a fake hard_gap without the record
    [gapx_before] = [t for t in trades(session, without) if t.security_id == "S_GAPX"]
    assert (gapx_before.exit_reason, gapx_before.exit_price) == ("hard_gap", 48.0)
    assert gapx_before.child_proceeds == 0.0 and gapx_before.spinoff is None
    [gapx_after] = [t for t in trades(session, with_record) if t.security_id == "S_GAPX"]
    assert gapx_after.spinoff == "adjust"


def test_a_restart_event_unfreezes_at_its_known_on_with_raw_levels(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    # Parent and child on one symbol (real: CR -> CR 2023-04-04): restart, known on the ex-date.
    # The ex bar is frozen; from X4 the raw hard stop 55 is back and X4's open 48.3 gaps it.
    world = spin_world(
        tmp_path, session, children=(), records=[spin_row("GAPX", "GAPX", X[3], 0.25)]
    )
    run_id = replay(session, con, world.root, strategy_dir, end_of(WDC_AFTER))
    [only] = trades(session, run_id)
    assert (ny_date(only.exit_at), only.exit_price, only.exit_reason) == (X[4], 48.3, "hard_gap")
    assert only.spinoff == "restart"
    assert only.child_proceeds == 0.0
    assert only.pnl == pytest.approx(125 * (48.3 - 56.5))


def test_the_null_prices_the_spinoff_leg_like_the_replay(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    world = spin_world(tmp_path, session)
    end = end_of(WDC_AFTER)
    run_id = replay(session, con, world.root, strategy_dir, end)
    [only] = trades(session, run_id)
    spans = spans_of_run(session, run_id)
    cfg = apply_overrides(load_strategy(strategy_dir), list(ZERO_SLIP))

    panel = build_panel(session, con, world.root, spans, cfg=cfg, as_of=close_of(end))

    assert list(panel.spinoffs["S_GAPX"]) == [
        SpinoffAdjust(ex_date=X[3], known_on=X[3], children=((0.25, 48.0),))
    ]
    per_share = sum(r * p for r, p in panel.spinoffs["S_GAPX"][0].children)
    assert only.child_proceeds is not None
    assert per_share == pytest.approx(only.child_proceeds / only.shares)
    assert "S_GAPX" in panel.pools[X[1]]
    only_gapx = replace(panel, pools={X[1]: frozenset({"S_GAPX"})})
    out = null_returns(spans, only_gapx, permutations=1, seed=0, slippage_bps=0)
    assert out[0, 0] == pytest.approx(only.pnl / (only.shares * only.entry_price))


# =============================================================================================
# boundaries
# =============================================================================================

# X3 ex, X4 .. X9 held frozen-able, X10 break, X11 out
WAIT_AFTER: list[Ohlcv | None] = [*PRE, EX_BAR, *[HOLD] * 6, BREAK, EXIT, *[FILLER] * 15]


def test_an_adjust_known_on_the_last_wait_session_still_adjusts(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    # KIDZ first trades on X8 = wait session 5 = spinoff_max_wait: adjust known on X8, sold at
    # X9's open 48.0. X4 .. X8 open at 48.3, through the raw hard stop 55, all frozen.
    world = spin_world(tmp_path, session, after=WAIT_AFTER, children=(replace(KIDZ, first=8),))
    run_id = replay(session, con, world.root, strategy_dir, end_of(WAIT_AFTER))
    [only] = trades(session, run_id)
    assert only.spinoff == "adjust"
    assert only.child_proceeds == pytest.approx(CHILD_PROCEEDS)
    assert (ny_date(only.exit_at), only.exit_price, only.exit_reason) == (X[11], 45.0, "break")
    assert only.pnl == pytest.approx(62.5)


def test_a_wait_one_session_too_long_ends_unpriced(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    # KIDZ first trades on X9 = wait session 6. At X8's close the event is still pending: the
    # leg resumes on raw levels and X9's open 48.3 gaps the hard stop 55. The adjust that
    # becomes known on X9 must not be applied or paid afterwards.
    world = spin_world(tmp_path, session, after=WAIT_AFTER, children=(replace(KIDZ, first=9),))
    run_id = replay(session, con, world.root, strategy_dir, end_of(WAIT_AFTER))
    [only] = trades(session, run_id)
    assert (ny_date(only.exit_at), only.exit_price, only.exit_reason) == (X[9], 48.3, "hard_gap")
    assert only.spinoff == "restart"
    assert only.child_proceeds == 0.0
    assert only.pnl == pytest.approx(125 * (48.3 - 56.5))


def test_the_wait_is_the_strategys_parameter(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    world = spin_world(tmp_path, session, after=WAIT_AFTER, children=(replace(KIDZ, first=9),))
    run_id = replay(
        session, con, world.root, strategy_dir, end_of(WAIT_AFTER),
        "execute.data.spinoff_max_wait=6",
    )  # fmt: skip
    [only] = trades(session, run_id)
    assert only.spinoff == "adjust"
    assert only.child_proceeds == pytest.approx(CHILD_PROCEEDS)  # sold at X10's open 48.0


# X2 closes 53.4 below the zone 53.5: break, filled at the ex open X3 (43.0). Previous close
# 53.4, KIDZ first close 42.72 x 0.25 = 10.68 -> factor 0.8; KIDZ sold at X4's open 42.72.
PENDING_AFTER: list[Ohlcv | None] = [
    (56.5, 57.0, 56.0, 56.8, 1_200_000),
    (56.0, 56.2, 53.0, 53.4, 1_500_000),
    (43.0, 43.5, 42.8, 43.0, 2_000_000),
    *[(43.0, 43.5, 42.8, 43.2, 1_000_000)] * 23,
]


def test_a_pending_exit_at_the_ex_open_still_gets_the_child(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    world = spin_world(
        tmp_path, session, after=PENDING_AFTER, children=(replace(KIDZ, price=42.72),)
    )
    run_id = replay(session, con, world.root, strategy_dir, end_of(PENDING_AFTER))
    [only] = trades(session, run_id)
    assert (ny_date(only.exit_at), only.exit_price, only.exit_reason) == (X[3], 43.0, "break")
    assert only.child_proceeds == pytest.approx(125 * 0.25 * 42.72)  # 1335.0
    assert only.spinoff == "adjust"
    assert only.pnl == pytest.approx(125 * (43.0 - 56.5) + 1335.0)
    assert only.r_multiple == pytest.approx((43.0 - 56.5 + 10.68) / 4.5)
    # the result includes X4's child sale, so it was knowable only at X4's close
    assert only.available_at == close_of(X[4])


def test_a_leg_one_order_for_the_ex_open_is_cancelled(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    # The ex-date is X1, the session leg 1 would buy at the open. Previous close 56 (D),
    # 0.25 x KIDZ 48 -> factor 44 / 56; GAPX opens X1 at 44.5 (total return within 15%).
    after: list[Ohlcv | None] = [
        (44.5, 45.0, 44.0, 44.6, 1_200_000),
        *[(44.6, 45.0, 44.2, 44.6, 1_000_000)] * 22,
    ]
    world = spin_world(tmp_path, session, after=after, ex=1, children=(replace(KIDZ, first=1),))
    run_id = replay(session, con, world.root, strategy_dir, end_of(after))
    assert trades(session, run_id) == []
    [decision] = rows(session, Decision, run_id)
    assert decision.payload["cancelled"] == "spinoff_ex"


def test_a_child_without_a_bar_the_next_session_is_sold_at_its_next_open(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    kid = replace(KIDZ, prices={4: None, 5: 50.0, **{i: 50.0 for i in range(6, 27)}})
    world = spin_world(tmp_path, session, children=(kid,))
    run_id = replay(session, con, world.root, strategy_dir, end_of(WDC_AFTER))
    [only] = trades(session, run_id)
    assert only.child_proceeds == pytest.approx(125 * 0.25 * 50.0)  # X5's open
    assert only.pnl == pytest.approx(125 * (45.0 - 56.5) + 1562.5)


def test_a_child_that_never_trades_again_is_valued_at_its_last_close(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    # KIDZ trades on X3 only. After max_missing_sessions (5: X4 .. X8) its last close 48.0
    # stands in. The parent leaves at X6's open; the leg is booked once the child is valued.
    world = spin_world(tmp_path, session, children=(replace(KIDZ, last=3),))
    run_id = replay(session, con, world.root, strategy_dir, end_of(WDC_AFTER))
    [only] = trades(session, run_id)
    assert (ny_date(only.exit_at), only.exit_price, only.exit_reason) == (X[6], 45.0, "break")
    assert only.child_proceeds == pytest.approx(CHILD_PROCEEDS)
    assert only.available_at == close_of(X[8])


def test_an_ex_date_without_a_parent_bar_applies_to_the_next_bar(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    # GAPX halted on the ex-date X3; X4 is its first bar after it (wait session 0) and opens
    # through the raw hard stop. Known on X4 (the parent's ex-session close), KIDZ sold at X5.
    after: list[Ohlcv | None] = [*PRE, None, EX_BAR, HOLD, BREAK, EXIT, *[FILLER] * 19]
    world = spin_world(tmp_path, session, after=after)
    run_id = replay(session, con, world.root, strategy_dir, end_of(after))
    [only] = trades(session, run_id)
    assert (ny_date(only.exit_at), only.exit_price, only.exit_reason) == (X[7], 45.0, "break")
    assert only.spinoff == "adjust"
    assert only.child_proceeds == pytest.approx(CHILD_PROCEEDS)


def test_a_split_and_a_spinoff_in_one_hold(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    # 2:1 on X2: 250 shares at 28.25, key 27 -> 28.5 after a 30 close (hard 27.5). Ex-date X4:
    # 0.25 KIDZ at 24 per share, factor (30 - 6) / 30 = 0.8 -> key 22.8, atr 0.8. X5 trails it
    # to 23.05 (zone 22.85 .. 23.25, hard 22.25); X6 closes 22.65 -> out at X7's open 22.5.
    after: list[Ohlcv | None] = [
        (56.5, 57.0, 56.0, 56.8, 1_200_000),
        (28.5, 30.25, 28.4, 30.0, 2_400_000),
        (30.0, 30.2, 29.8, 30.0, 2_400_000),
        (24.0, 24.5, 23.75, 24.2, 3_000_000),
        (24.15, 24.4, 23.95, 24.25, 2_400_000),
        (23.0, 23.1, 22.5, 22.65, 2_400_000),
        (22.5, 22.6, 22.4, 22.5, 2_000_000),
        *[(22.5, 22.6, 22.4, 22.5, 2_000_000)] * 19,
    ]
    world = spin_world(
        tmp_path, session, after=after, ex=4, children=(Child("KIDZ", 0.25, 4, 24.0),)
    )
    write_splits(world.root, [("GAPX", X[2], 1.0, 2.0)])
    run_id = replay(session, con, world.root, strategy_dir, end_of(after))
    [only] = trades(session, run_id)
    assert only.shares == 250
    assert only.entry_price == pytest.approx(28.25)
    assert (ny_date(only.exit_at), only.exit_price, only.exit_reason) == (X[7], 22.5, "break")
    assert only.child_proceeds == pytest.approx(250 * 0.25 * 24.0)
    assert only.pnl == pytest.approx(250 * (22.5 - 28.25) + 1500.0)
    assert only.r_multiple == pytest.approx((22.5 - 28.25 + 6.0) / (28.25 - 26.0))


def test_a_spinoff_on_the_reaction_day_changes_nothing(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    # Ex-date D: the decision is made after that bar and leg 1 buys X1 with no entitlement.
    kid = Child("KIDZ", 0.25, 0, 4.0)
    world = spin_world(tmp_path, session, after=TRADED, ex=0, children=(kid,), records=[])
    end = end_of(TRADED)
    without = replay(session, con, world.root, strategy_dir, end)
    write_other_rows(world.root, [spin_row("GAPX", "KIDZ", X[0], 0.25)])
    with_record = replay(session, con, world.root, strategy_dir, end)
    assert [trade_key(t) for t in trades(session, with_record)] == [
        trade_key(t) for t in trades(session, without)
    ]
    assert [(t.child_proceeds, t.spinoff) for t in trades(session, with_record)] == [
        (0.0, None),
        (0.0, None),
    ]


def test_a_leg_closed_before_the_ex_date_is_untouched(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    # Leg 1 leaves at X3 (break), no re-entry confirmed; the ex-date is X5 while the setup waits.
    # Previous close 52, 0.25 KIDZ at 41.6 -> factor 0.8; GAPX trades at 41.6 from X5.
    after: list[Ohlcv | None] = [
        (56.5, 57.0, 56.0, 56.8, 1_200_000),
        (56.0, 56.2, 53.0, 53.4, 1_500_000),
        (53.4, 53.5, 52.0, 52.2, 1_500_000),
        (52.0, 52.5, 51.5, 52.0, 1_000_000),
        *[(41.6, 42.0, 41.3, 41.6, 1_000_000)] * 22,
    ]
    kid = Child("KIDZ", 0.25, 5, 41.6)
    world = spin_world(tmp_path, session, after=after, ex=5, children=(kid,), records=[])
    end = end_of(after)
    without = replay(session, con, world.root, strategy_dir, end)
    write_other_rows(world.root, [spin_row("GAPX", "KIDZ", X[5], 0.25)])
    with_record = replay(session, con, world.root, strategy_dir, end)
    [before] = trades(session, without)
    [after_] = trades(session, with_record)
    assert trade_key(after_) == trade_key(before)
    assert (after_.child_proceeds, after_.spinoff) == (0.0, None)


# =============================================================================================
# leakage
# =============================================================================================

# KIDZ first trades on X5, so the event is known only at X5's close. X4 opens at 44.0 and dips
# to 43.0: through the raw hard stop 55 and the adjusted one 44.0 alike.
LATE_AFTER: list[Ohlcv | None] = [
    *PRE,
    EX_BAR,
    (44.0, 48.8, 43.0, 48.5, 1_500_000),
    HOLD,
    HOLD,  # X6: KIDZ sold at this open
    BREAK,
    EXIT,  # X8: out
    *[FILLER] * 20,
]


@pytest.mark.parametrize("cut", [3, 4, 5, 6])
def test_nothing_moves_before_the_factor_is_known_and_a_short_replay_writes_nothing(
    tmp_path: Path,
    session: Session,
    con: duckdb.DuckDBPyConnection,
    strategy_dir: Path,
    cut: int,
) -> None:
    world = spin_world(tmp_path, session, after=LATE_AFTER, children=(replace(KIDZ, first=5),))
    full = replay(session, con, world.root, strategy_dir, end_of(LATE_AFTER))
    [only] = trades(session, full)
    assert (ny_date(only.exit_at), only.exit_reason) == (X[8], "break")
    assert only.child_proceeds == pytest.approx(CHILD_PROCEEDS)

    # A replay ending while the leg is frozen (or adjusted but still held) books nothing: no
    # hard_gap at X3, no hard stop at X4, and no guess at an event decided after its end.
    short = replay(session, con, world.root, strategy_dir, X[cut])
    assert trades(session, short) == []
    [decision] = rows(session, Decision, short)
    assert "cancelled" not in decision.payload


def test_a_record_filed_after_the_ex_date_is_not_used(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    # The record reached the store only at X10's close: on X3 the replay could not know it, so
    # the drop is a hard_gap as today.
    world = spin_world(tmp_path, session, record_at=close_of(X[10]).astimezone(UTC))
    run_id = replay(session, con, world.root, strategy_dir, end_of(WDC_AFTER))
    [only] = trades(session, run_id)
    assert (ny_date(only.exit_at), only.exit_price, only.exit_reason) == (X[3], 48.0, "hard_gap")
    assert (only.child_proceeds, only.spinoff) == (0.0, None)


def test_a_child_bar_on_file_only_later_is_not_used_for_the_sale(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    # KIDZ's X4 bar (open 60) reached the store at X6's close. On X4's evening the child has no
    # visible bar; X5's open 50 is the sale.
    kid = replace(
        KIDZ,
        prices={4: 60.0, **{i: 50.0 for i in range(5, 27)}},
        late={4: close_of(X[6]).astimezone(UTC)},
    )
    world = spin_world(tmp_path, session, children=(kid,))
    run_id = replay(session, con, world.root, strategy_dir, end_of(WDC_AFTER))
    [only] = trades(session, run_id)
    assert only.child_proceeds == pytest.approx(125 * 0.25 * 50.0)


def test_a_second_spinoff_before_the_first_child_is_sold_is_not_lost(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    # KIDZ (X3) is sold at X4's open; KIDY goes ex on X4 itself, while KIDZ is still owed. The
    # second event is not priced: the leg is flagged restart, KIDZ is still paid.
    kidy = Child("KIDY", 0.1, 4, 5.0)
    records = [spin_row("GAPX", "KIDZ", X[3], 0.25), spin_row("GAPX", "KIDY", X[4], 0.1)]
    world = spin_world(tmp_path, session, children=(KIDZ, kidy), records=records)
    run_id = replay(session, con, world.root, strategy_dir, end_of(WDC_AFTER))
    [only] = trades(session, run_id)
    assert only.spinoff == "restart"
    assert only.child_proceeds == pytest.approx(CHILD_PROCEEDS)
    assert (ny_date(only.exit_at), only.exit_price, only.exit_reason) == (X[6], 45.0, "break")
