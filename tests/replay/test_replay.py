"""Guards against the replay seeing tomorrow: a bar handed to the scanner or the simulator before
its 20:00 New York availability, a filing released after the close of a gap day treated as that
day's catalyst, results up to day E that change when the replay runs past E, and an attribution
written before the re-entry window has closed. Also guards the bookkeeping: one Run per replay
tied to its StrategyVersion, one take-all Decision per candidate, one Trade per leg with the
golden-style numbers, splits and missing data while holding.

API under test (asof.replay):

    run_replay(session, con, root, strategy_dir, start: date, end: date, overrides=()) -> run_id
        per session d (as_of = d 20:00 New York): process held legs with d's bar, then scan ->
        take-all judge -> Decision -> orders for the next open. Writes Run, Decision, Trade,
        Attribution. History before ``start`` is loaded for the scan's lookback.

World: tests/scan/world.py (provenance there). GAPX gaps on D = 2026-02-17 after an 8-K 2.02
released Friday 02-13 16:08 New York; ATR 2, key 54 (zone 53.5 .. 54.5, hard 52), pre-earnings
close 50 (zone 49.5 .. 50.5, hard 48), decision close 56 -> 125 shares. Slippage set to 0.
"""

from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import pytest
from sqlmodel import Session, col, select

from asof.execute import DailyBar, DecisionPlan, simulate_decision
from asof.replay import BarCache, run_replay
from asof.store.models import Attribution, Decision, Run, StrategyVersion, Trade
from asof.strategy import apply_overrides, content_hash, load_strategy
from tests.replay.scenario import END, START, TRADED, X
from tests.scan.world import ET, GAP, Ohlcv, close_of, et, filing, gap_world
from tests.store.test_market import write_splits
from tests.strategy.files import write_strategy

ZERO_SLIP = ("execute.fills.slippage_bps=0",)


def ny_date(value: Any) -> date:
    return value.astimezone(ET).date()


def rows(session: Session, model: Any, run_id: int) -> list[Any]:
    return list(
        session.exec(select(model).where(model.run_id == run_id).order_by(col(model.id))).all()
    )


def trades(session: Session, run_id: int) -> list[Trade]:
    return list(
        session.exec(
            select(Trade).where(Trade.run_id == run_id).order_by(col(Trade.entry_at))
        ).all()
    )


def attributions(session: Session, run_id: int) -> list[Attribution]:
    decision_ids = [d.id for d in rows(session, Decision, run_id)]
    return list(
        session.exec(
            select(Attribution).where(col(Attribution.decision_id).in_(decision_ids))
        ).all()
    )


def trade_key(t: Trade) -> tuple[Any, ...]:
    return (
        t.security_id, t.leg, t.entry_at, t.entry_price, t.shares, t.key_level, t.hard_stop,
        t.exit_at, t.exit_price, t.exit_reason, t.pnl, t.r_multiple, t.available_at,
    )  # fmt: skip


@pytest.fixture
def strategy_dir(tmp_path: Path) -> Path:
    return write_strategy(tmp_path / "strategies")


# =============================================================================================
# normal path
# =============================================================================================


def test_replay_writes_run_decision_trades_and_attribution(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    world = gap_world(tmp_path, session, after=TRADED)
    run_id = run_replay(session, con, world.root, strategy_dir, START, END, ZERO_SLIP)

    run = session.get(Run, run_id)
    assert run is not None
    assert run.mode == "replay"
    assert run.strategy_id == "earnings_gap"
    version = session.get(StrategyVersion, run.strategy_version_id)
    assert version is not None
    assert version.content_hash == content_hash(strategy_dir, ZERO_SLIP)

    [decision] = rows(session, Decision, run_id)
    assert decision.kind == "trade"
    assert decision.direction == "long"
    assert decision.conviction is None
    assert decision.strategy_id == "earnings_gap"
    assert decision.entity_id == world.gapx
    assert decision.as_of == close_of(X[0])
    text = repr(decision.payload)
    assert "take_all" in text
    assert "S_GAPX" in text

    first, second = trades(session, run_id)
    assert first.decision_id == second.decision_id == decision.id
    assert (first.leg, first.shares, first.entry_price) == (1, 125, 56.5)
    assert first.key_level == pytest.approx(54.0)
    assert first.hard_stop == pytest.approx(52.0)
    assert first.risk_per_share == pytest.approx(4.5)
    assert (ny_date(first.entry_at), ny_date(first.exit_at)) == (X[1], X[3])
    assert (first.exit_price, first.exit_reason) == (53.4, "break")
    assert first.pnl == pytest.approx(-387.5)
    assert first.r_multiple == pytest.approx(-3.1 / 4.5)
    assert first.available_at == close_of(X[3])

    assert (second.leg, second.shares, second.entry_price) == (2, 40, 51.0)
    assert second.key_level == pytest.approx(50.0)
    assert second.hard_stop == pytest.approx(48.0)
    assert (ny_date(second.entry_at), ny_date(second.exit_at)) == (X[4], X[4])
    assert (second.exit_price, second.exit_reason) == (48.0, "hard")
    assert second.pnl == pytest.approx(-120.0)
    assert second.r_multiple == pytest.approx(-1.0)
    assert second.available_at == close_of(X[4])

    [attribution] = attributions(session, run_id)
    assert attribution.decision_id == decision.id
    assert attribution.strategy_id == "earnings_gap"
    # re-entry window: 20 sessions after the reaction day
    assert attribution.outcome_available_at == close_of(X[20])
    assert attribution.realized_return == pytest.approx(-507.5 / 100_000)
    assert attribution.is_correct is False


def test_same_inputs_same_results(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    world = gap_world(tmp_path, session, after=TRADED)
    a = run_replay(session, con, world.root, strategy_dir, START, END, ZERO_SLIP)
    b = run_replay(session, con, world.root, strategy_dir, START, END, ZERO_SLIP)
    assert a != b
    assert [trade_key(t) for t in trades(session, a)] == [trade_key(t) for t in trades(session, b)]
    run_a, run_b = session.get(Run, a), session.get(Run, b)
    assert run_a is not None and run_b is not None
    assert run_a.strategy_version_id == run_b.strategy_version_id


# =============================================================================================
# boundaries: splits and missing data while holding
# =============================================================================================


def test_a_split_while_holding_rescales_the_position(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    # 2:1 on X2 (raw bars halve, as the real store's do: AAPL 2020-08-28 close 499.23, 08-31
    # open 127.39 on a 4:1). Key 54 -> 27; X3 closes 26.5, below the zone either way -> break.
    after: list[Ohlcv | None] = [
        (56.5, 57.0, 56.0, 56.8, 1_200_000),
        (28.1, 28.3, 27.9, 28.2, 2_400_000),
        (28.0, 28.1, 26.4, 26.5, 2_400_000),
        (26.6, 26.7, 26.3, 26.4, 2_000_000),
        *[(26.5, 26.7, 26.3, 26.5, 2_000_000)] * 22,
    ]
    world = gap_world(tmp_path, session, after=after)
    write_splits(world.root, [("GAPX", X[2], 1.0, 2.0)])
    run_id = run_replay(session, con, world.root, strategy_dir, START, X[len(after)], ZERO_SLIP)
    [only] = trades(session, run_id)
    assert only.shares == 250
    assert only.entry_price == pytest.approx(28.25)
    assert (ny_date(only.exit_at), only.exit_price, only.exit_reason) == (X[4], 26.6, "break")
    assert only.pnl == pytest.approx(-412.5)  # = 125 * (53.2 - 56.5) before the split
    assert only.r_multiple == pytest.approx(-1.65 / 2.25)


def test_five_missing_sessions_exit_at_the_last_close(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    after: list[Ohlcv | None] = [
        (56.5, 57.0, 56.0, 56.8, 1_200_000),
        (56.6, 57.2, 56.0, 57.0, 1_200_000),  # last GAPX bar
        *[None] * 24,  # OTHR keeps trading, so these are market sessions
    ]
    world = gap_world(tmp_path, session, after=after)
    run_id = run_replay(session, con, world.root, strategy_dir, START, X[len(after)], ZERO_SLIP)
    [only] = trades(session, run_id)
    assert (only.exit_price, only.exit_reason) == (57.0, "data_gap")
    assert only.pnl == pytest.approx(125 * 0.5)
    # known once the fifth missing session (X7) has closed
    assert only.available_at == close_of(X[7])


def test_four_missing_sessions_do_not_exit(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    after: list[Ohlcv | None] = [
        (56.5, 57.0, 56.0, 56.8, 1_200_000),
        (56.6, 57.2, 56.0, 57.0, 1_200_000),
        *[None] * 4,
        (57.0, 57.5, 56.8, 57.2, 1_200_000),
        *[(57.2, 57.6, 56.9, 57.3, 1_000_000)] * 6,
    ]
    world = gap_world(tmp_path, session, after=after)
    run_id = run_replay(session, con, world.root, strategy_dir, START, X[len(after)], ZERO_SLIP)
    assert all(t.exit_reason != "data_gap" for t in trades(session, run_id))


# =============================================================================================
# leakage
# =============================================================================================


def test_no_bar_reaches_the_replay_before_it_is_final(
    tmp_path: Path,
    session: Session,
    con: duckdb.DuckDBPyConnection,
    strategy_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Every bar the replay reads comes through BarCache.bars_at(as_of); none may be newer than
    # that as_of, no as_of may lie past the last session's close, and time only moves forward.
    calls: list[tuple[Any, Any]] = []
    original = BarCache.bars_at

    def spy(self: BarCache, as_of: Any, **kwargs: Any) -> Any:
        table = original(self, as_of, **kwargs)
        calls.append((as_of, table))
        return table

    monkeypatch.setattr(BarCache, "bars_at", spy)
    world = gap_world(tmp_path, session, after=TRADED)
    run_replay(session, con, world.root, strategy_dir, START, END, ZERO_SLIP)

    assert calls, "the replay must read bars through BarCache"
    stamps = [as_of for as_of, _ in calls]
    assert stamps == sorted(stamps)
    assert max(stamps) <= close_of(END)
    for as_of, table in calls:
        assert all(t <= as_of for t in table.column("available_at").to_pylist()), as_of


def test_a_release_after_the_close_of_a_gap_day_waits_for_the_next_session(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    # OTHR gaps on D with heavy volume and files an 8-K 2.02 at 16:08 on D. At D 20:00 the filing
    # is visible and so is the gap, but the gap happened before the release: the reaction
    # session is X1, whose bar is not visible until X1 20:00. OTHR gaps again on X1, so its
    # decision exists -- dated X1, never D.
    world = gap_world(
        tmp_path,
        session,
        after=TRADED,
        othr_bars={X[0]: GAP, X[1]: (60.0, 62.0, 59.0, 61.0, 4_000_000)},
    )
    filing(
        session,
        source_id=world.source_id,
        entity_id=world.othr,
        accession="0000002222-26-000009",
        available_at=et(X[0], 16, 8),
        exhibits=[("ex991.htm", "8-K 2.02,9.01 EX-99.1 PRESS RELEASE")],
    )
    run_id = run_replay(session, con, world.root, strategy_dir, START, END, ZERO_SLIP)
    by_entity = {d.entity_id: d.as_of for d in rows(session, Decision, run_id)}
    assert by_entity == {world.gapx: close_of(X[0]), world.othr: close_of(X[1])}


@pytest.mark.parametrize("cut", [2, 3])
def test_results_up_to_a_day_do_not_depend_on_later_days(
    tmp_path: Path,
    session: Session,
    con: duckdb.DuckDBPyConnection,
    strategy_dir: Path,
    cut: int,
) -> None:
    # A replay ending on X[cut] must write exactly the trades of the full replay that were known
    # by then, and no attribution (the re-entry window is still open).
    world = gap_world(tmp_path, session, after=TRADED)
    full = run_replay(session, con, world.root, strategy_dir, START, END, ZERO_SLIP)
    short = run_replay(session, con, world.root, strategy_dir, START, X[cut], ZERO_SLIP)
    known = [t for t in trades(session, full) if t.available_at <= close_of(X[cut])]
    assert [trade_key(t) for t in trades(session, short)] == [trade_key(t) for t in known]
    assert len(known) == (0 if cut == 2 else 1)
    assert attributions(session, short) == []
    assert len(rows(session, Decision, short)) == 1


def test_no_attribution_before_the_window_closes(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    # Both legs are closed by X4, but a re-entry could have come until X20.
    world = gap_world(tmp_path, session, after=TRADED)
    early = run_replay(session, con, world.root, strategy_dir, START, X[19], ZERO_SLIP)
    assert len(trades(session, early)) == 2
    assert attributions(session, early) == []
    on_time = run_replay(session, con, world.root, strategy_dir, START, X[20], ZERO_SLIP)
    [attribution] = attributions(session, on_time)
    assert attribution.outcome_available_at == close_of(X[20])


# =============================================================================================
# the replay and simulate_decision agree (reviewer H1), splits on a day without a bar (H2)
# =============================================================================================

# Reaction day closes at 49.3, below the re-entry zone 49.5 .. 50.5: the gap is filled on day 0.
FILLED_REACTION: Ohlcv = (55.0, 57.0, 49.0, 49.3, 3_000_000)
FILLED_AFTER: list[Ohlcv | None] = [
    (50.0, 51.0, 49.6, 50.6, 1_200_000),  # X1: leg 1 in at 50.0 (key 49, zone 48.5 .. 49.5)
    (50.0, 51.0, 46.9, 50.8, 1_500_000),  # X2: hard stop 47 hit; low 46.9, close 50.8 would
    #                                       confirm a re-entry if the setup were still valid
    *[(51.0, 51.5, 50.6, 51.2, 1_000_000)] * 24,
]


def legs_of(session: Session, run_id: int) -> list[tuple[Any, ...]]:
    return [
        (t.leg, t.shares, t.entry_price, ny_date(t.exit_at), t.exit_price, t.exit_reason)
        for t in trades(session, run_id)
    ]


def simulated_legs(
    world_bars: list[Any], strategy_dir: Path, reaction: Ohlcv
) -> list[tuple[Any, ...]]:
    cfg = apply_overrides(load_strategy(strategy_dir), list(ZERO_SLIP)).execute
    bars = sorted(
        (
            DailyBar(session_date=b.t.astimezone(ET).date(), open=b.open, high=b.high,
                     low=b.low, close=b.close)
            for b in world_bars
            if b.symbol == "GAPX" and b.t.astimezone(ET).date() >= X[0]
        ),
        key=lambda b: b.session_date,
    )  # fmt: skip
    plan = DecisionPlan(
        reaction_session=X[0],
        decision_close=reaction[3],
        gap_low=reaction[2],
        pre_earnings_close=50.0,
        atr=2.0,
        equity=100_000.0,
    )
    outcome = simulate_decision(plan, bars, cfg)
    return [
        (r.leg, r.shares, r.entry_price, r.exit_session, r.exit_price, r.exit_reason)
        for r in outcome.legs
    ]


@pytest.mark.parametrize(
    ("reaction", "after"),
    [(GAP, TRADED), (FILLED_REACTION, FILLED_AFTER)],
    ids=["two_legs", "filled"],
)
def test_replay_trades_equal_simulate_decision(
    tmp_path: Path,
    session: Session,
    con: duckdb.DuckDBPyConnection,
    strategy_dir: Path,
    reaction: Ohlcv,
    after: list[Ohlcv | None],
) -> None:
    world = gap_world(tmp_path, session, reaction=reaction, after=after)
    run_id = run_replay(session, con, world.root, strategy_dir, START, X[len(after)], ZERO_SLIP)
    assert legs_of(session, run_id) == simulated_legs(world.bars, strategy_dir, reaction)


def test_a_gap_filled_on_the_reaction_day_never_reenters(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    world = gap_world(tmp_path, session, reaction=FILLED_REACTION, after=FILLED_AFTER)
    run_id = run_replay(
        session, con, world.root, strategy_dir, START, X[len(FILLED_AFTER)], ZERO_SLIP
    )
    [only] = trades(session, run_id)
    assert (only.leg, only.exit_price, only.exit_reason) == (1, 47.0, "hard")


def test_a_split_on_a_session_without_a_bar_is_still_applied(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    # 2:1 with ex-date X2, but GAPX has no bar on X2 (halted). X3 trades at half price; the
    # position must be 250 shares at 28.25 by then, not 125 shares gapping through the stop.
    after: list[Ohlcv | None] = [
        (56.5, 57.0, 56.0, 56.8, 1_200_000),
        None,
        (28.0, 28.1, 26.4, 26.5, 2_400_000),
        (26.6, 26.7, 26.3, 26.4, 2_000_000),
        *[(26.5, 26.7, 26.3, 26.5, 2_000_000)] * 22,
    ]
    world = gap_world(tmp_path, session, after=after)
    write_splits(world.root, [("GAPX", X[2], 1.0, 2.0)])
    run_id = run_replay(session, con, world.root, strategy_dir, START, X[len(after)], ZERO_SLIP)
    [only] = trades(session, run_id)
    assert only.shares == 250
    assert (ny_date(only.exit_at), only.exit_price, only.exit_reason) == (X[4], 26.6, "break")
    assert only.pnl == pytest.approx(-412.5)


def test_an_unrecorded_reverse_split_exits_at_the_last_close(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    # Prices jump 4x on X2 with no split record (Alpaca and SEC each recall about 2/3 of splits,
    # DECISIONS 2026-10-07): open / last close 4.0 is above max_jump_ratio 1.8, so the leg leaves
    # at the last trusted close instead of booking a fake gain. Unrecorded drops are left to the
    # stops: 91% of them are real crashes (same decision).
    after: list[Ohlcv | None] = [
        (56.5, 57.0, 56.0, 56.8, 1_200_000),
        (227.0, 228.0, 226.0, 227.5, 300_000),
        *[(227.5, 228.5, 226.5, 227.5, 300_000)] * 24,
    ]
    world = gap_world(tmp_path, session, after=after)
    run_id = run_replay(session, con, world.root, strategy_dir, START, X[len(after)], ZERO_SLIP)
    [only] = trades(session, run_id)
    assert (ny_date(only.exit_at), only.exit_price, only.exit_reason) == (X[2], 56.8, "price_jump")
    assert only.pnl == pytest.approx(125 * 0.3)
