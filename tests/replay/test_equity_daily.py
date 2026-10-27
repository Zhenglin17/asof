"""Guards against a daily equity curve that lies: a session left out (a quiet day without
positions is still a day of the curve), a closed leg counted twice or not at all, a pending order
counted as exposure, a split read as a crash, a session without a bar marked at zero, and a row
for day d that changes when bars after d exist or that is visible before d's close.

API under test (new):

    asof.store.models.EquityDaily (table ``equity_daily``):
        id, run_id (FK run.id, indexed), session_date: date,
        equity = initial equity + realized + unrealized at the session's close,
        realized (cumulative), unrealized, gross_exposure (shares x last close of OPEN
        positions only; orders not filled yet are excluded), open_positions (count),
        available_at = 20:00 New York of session_date (as Trade.available_at).
        UniqueConstraint(run_id, session_date).
    asof.store.queries.visible_equity(session, as_of, *, run_id) -> list[EquityDaily]
        rows with available_at <= as_of, ordered by session_date; naive as_of -> ValueError.
    asof.replay.run_replay writes one EquityDaily row per session in [start, end], after that
    session's step_positions.

World: tests/scan/world.py, traded path tests/replay/scenario.py (provenance in both). Slippage
0, initial equity 100,000 (tests/strategy/files.py). Expected values were computed by hand from
the path and match what ``_Replay.equity()`` returns after each step today:

    session  equity      realized  unrealized  gross   open
    D        100000.0      0.0        0.0        0.0    0   leg 1 ordered at D's close
    X1       100037.5      0.0       37.5     7100.0    1   125 sh in at 56.5, close 56.8
    X2        99612.5      0.0     -387.5     6675.0    1   close 53.4 -> break
    X3        99612.5   -387.5        0.0        0.0    0   leg 1 out at 53.4; leg 2 ordered
    X4        99492.5   -507.5        0.0        0.0    0   leg 2 40 sh in at 51, out at 48

Split: 2:1 ex-date X2; bars after it are raw post-split prices, as the real store's are (real:
AAPL 2020-08-28 close 499.23, 08-31 open 127.39 on a 4:1), the split record visible from 00:00
New York of its ex-date (asof.ingest.corporate_actions).
"""

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pytest
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, col, select

from asof.replay import run_replay
from asof.store.models import EquityDaily, Run, Trade
from asof.store.queries import visible_equity
from tests.replay.scenario import END, START, TRADED, X
from tests.scan.world import SESSIONS, Ohlcv, close_of, gap_world
from tests.store.test_market import write_splits
from tests.strategy.files import write_strategy

ZERO_SLIP = ("execute.fills.slippage_bps=0",)
INITIAL = 100_000.0


@pytest.fixture
def strategy_dir(tmp_path: Path) -> Path:
    return write_strategy(tmp_path / "strategies")


def equity_rows(session: Session, run_id: int) -> list[EquityDaily]:
    session.expire_all()
    return list(
        session.exec(
            select(EquityDaily)
            .where(EquityDaily.run_id == run_id)
            .order_by(col(EquityDaily.session_date))
        ).all()
    )


def by_day(session: Session, run_id: int) -> dict[date, EquityDaily]:
    return {r.session_date: r for r in equity_rows(session, run_id)}


def row_key(r: EquityDaily) -> tuple[Any, ...]:
    return (
        r.session_date, r.equity, r.realized, r.unrealized, r.gross_exposure, r.open_positions,
        r.available_at,
    )  # fmt: skip


def sessions_between(start: date, end: date) -> list[date]:
    return [d for d in SESSIONS if start <= d <= end]


def numbers(r: EquityDaily) -> tuple[float, float, float, float, int]:
    return (r.equity, r.realized, r.unrealized, r.gross_exposure, r.open_positions)


def new_run(session: Session) -> int:
    run = Run(
        strategy_id="earnings_gap",
        mode="replay",
        as_of=close_of(END),
        started_at=close_of(END),
        git_sha="test",
    )
    session.add(run)
    session.flush()
    assert run.id is not None
    return run.id


def flat_row(run_id: int, day: date, equity: float = INITIAL) -> EquityDaily:
    return EquityDaily(
        run_id=run_id,
        session_date=day,
        equity=equity,
        realized=equity - INITIAL,
        unrealized=0.0,
        gross_exposure=0.0,
        open_positions=0,
        available_at=close_of(day),
    )


# =============================================================================================
# normal path
# =============================================================================================


def test_one_row_per_session_with_the_hand_computed_values(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    world = gap_world(tmp_path, session, after=TRADED)
    run_id = run_replay(session, con, world.root, strategy_dir, START, END, ZERO_SLIP)

    rows = equity_rows(session, run_id)
    assert [r.session_date for r in rows] == sessions_between(START, END)
    assert all(r.run_id == run_id for r in rows)
    assert all(r.available_at == close_of(r.session_date) for r in rows)

    days = by_day(session, run_id)
    # Before the reaction day and on it: nothing held, equity is the initial equity.
    for day in sessions_between(START, X[0]):
        assert numbers(days[day]) == (INITIAL, 0.0, 0.0, 0.0, 0), day
    assert numbers(days[X[1]]) == pytest.approx((100_037.5, 0.0, 37.5, 7100.0, 1))
    assert numbers(days[X[2]]) == pytest.approx((99_612.5, 0.0, -387.5, 6675.0, 1))
    # Leg 1 out at X3's open: realized jumps by its pnl, the position leaves gross and count,
    # and leg 2's order (filled only at X4's open) is not exposure.
    assert numbers(days[X[3]]) == pytest.approx((99_612.5, -387.5, 0.0, 0.0, 0))
    assert numbers(days[X[4]]) == pytest.approx((99_492.5, -507.5, 0.0, 0.0, 0))


def test_equity_is_initial_plus_realized_plus_unrealized_and_ends_at_the_trades_sum(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    # Default slippage this time, so the identity is not an artefact of round prices.
    world = gap_world(tmp_path, session, after=TRADED)
    run_id = run_replay(session, con, world.root, strategy_dir, START, END)

    rows = equity_rows(session, run_id)
    for r in rows:
        assert r.equity == pytest.approx(INITIAL + r.realized + r.unrealized), r.session_date
    pnl = [t.pnl for t in session.exec(select(Trade).where(Trade.run_id == run_id)).all()]
    assert len(pnl) == 2
    last = rows[-1]
    assert (last.open_positions, last.gross_exposure, last.unrealized) == (0, 0.0, 0.0)
    assert last.realized == pytest.approx(sum(pnl))
    assert last.equity == pytest.approx(INITIAL + sum(pnl))


def test_realized_moves_only_on_exit_days_by_the_legs_pnl(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    world = gap_world(tmp_path, session, after=TRADED)
    run_id = run_replay(session, con, world.root, strategy_dir, START, END, ZERO_SLIP)

    rows = equity_rows(session, run_id)
    trades = session.exec(select(Trade).where(Trade.run_id == run_id)).all()
    pnl_on = {t.available_at: t.pnl for t in trades}
    previous = 0.0
    for r in rows:
        jump = r.realized - previous
        assert jump == pytest.approx(pnl_on.get(close_of(r.session_date), 0.0)), r.session_date
        previous = r.realized


# =============================================================================================
# boundaries
# =============================================================================================


def test_a_split_while_holding_keeps_equity_continuous(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    # 2:1 on X2: 125 sh at 56.5 become 250 at 28.25. Read naively, 125 sh at a 28.2 close would
    # be a 3,537.5 fake loss; the real move is 56.8 -> 56.4 per old share, -50 on the day.
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

    days = by_day(session, run_id)
    assert numbers(days[X[1]]) == pytest.approx((100_037.5, 0.0, 37.5, 7100.0, 1))
    assert numbers(days[X[2]]) == pytest.approx((99_987.5, 0.0, -12.5, 7050.0, 1))
    assert days[X[2]].equity - days[X[1]].equity == pytest.approx(-50.0)
    # Leg out at X4's open 26.6: 250 * (26.6 - 28.25) = -412.5, the replay test's pnl.
    assert numbers(days[X[4]]) == pytest.approx((99_587.5, -412.5, 0.0, 0.0, 0))


def test_a_session_without_a_bar_carries_the_last_close(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    # GAPX has no bar on X2 (OTHR trades, so X2 is a market session): marked at X1's 56.8.
    after: list[Ohlcv | None] = [
        (56.5, 57.0, 56.0, 56.8, 1_200_000),
        None,
        (56.9, 57.4, 56.6, 57.0, 1_200_000),
        *[(57.0, 57.5, 56.8, 57.2, 1_000_000)] * 23,
    ]
    world = gap_world(tmp_path, session, after=after)
    run_id = run_replay(session, con, world.root, strategy_dir, START, X[len(after)], ZERO_SLIP)

    days = by_day(session, run_id)
    assert X[2] in days
    assert numbers(days[X[2]]) == pytest.approx(numbers(days[X[1]]))
    assert numbers(days[X[2]]) == pytest.approx((100_037.5, 0.0, 37.5, 7100.0, 1))
    assert numbers(days[X[3]]) == pytest.approx((100_062.5, 0.0, 62.5, 7125.0, 1))


def test_a_window_without_any_position_is_flat_at_the_initial_equity(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    # Ends the session before the reaction day: no candidate, no decision, still a full curve.
    world = gap_world(tmp_path, session, after=TRADED)
    end = SESSIONS[SESSIONS.index(X[0]) - 1]
    run_id = run_replay(session, con, world.root, strategy_dir, START, end, ZERO_SLIP)

    rows = equity_rows(session, run_id)
    assert [r.session_date for r in rows] == sessions_between(START, end)
    assert len(rows) == 2
    assert [numbers(r) for r in rows] == [(INITIAL, 0.0, 0.0, 0.0, 0)] * 2


def test_a_one_session_replay_writes_one_row(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    world = gap_world(tmp_path, session, after=TRADED)
    run_id = run_replay(session, con, world.root, strategy_dir, X[1], X[1], ZERO_SLIP)

    [only] = equity_rows(session, run_id)
    assert only.session_date == X[1]
    # Started after the reaction day: nothing was bought, so nothing is marked.
    assert numbers(only) == (INITIAL, 0.0, 0.0, 0.0, 0)


def test_a_second_row_for_the_same_run_and_session_is_refused(session: Session) -> None:
    run_id = new_run(session)
    session.add(flat_row(run_id, X[1]))
    session.flush()
    session.add(flat_row(run_id, X[1]))
    with pytest.raises(IntegrityError):
        session.flush()


def test_visible_equity_orders_by_session_and_keeps_to_its_run(session: Session) -> None:
    mine, other = new_run(session), new_run(session)
    for day in (X[3], X[1], X[2]):
        session.add(flat_row(mine, day))
    session.add(flat_row(other, X[1], equity=1.0))
    session.commit()

    rows = visible_equity(session, close_of(X[3]), run_id=mine)
    assert [r.session_date for r in rows] == [X[1], X[2], X[3]]
    assert all(r.run_id == mine for r in rows)
    assert visible_equity(session, close_of(X[3]), run_id=999) == []


def test_visible_equity_refuses_a_naive_as_of(session: Session) -> None:
    run_id = new_run(session)
    with pytest.raises(ValueError):
        visible_equity(session, close_of(X[1]).replace(tzinfo=None), run_id=run_id)


# =============================================================================================
# leakage
# =============================================================================================


@pytest.mark.parametrize("cut", [1, 2, 3])
def test_rows_up_to_a_day_do_not_depend_on_later_bars(
    tmp_path: Path,
    session: Session,
    con: duckdb.DuckDBPyConnection,
    strategy_dir: Path,
    cut: int,
) -> None:
    # The store holds bars through X26 either way; the short replay stops at X[cut], so its rows
    # can only agree with the full one if no later bar reached any row up to X[cut]. X1 and X2
    # hold an open leg (marked to market), X3 is the exit day.
    world = gap_world(tmp_path, session, after=TRADED)
    full = run_replay(session, con, world.root, strategy_dir, START, END, ZERO_SLIP)
    short = run_replay(session, con, world.root, strategy_dir, START, X[cut], ZERO_SLIP)

    short_rows = [row_key(r) for r in equity_rows(session, short)]
    full_rows = [row_key(r) for r in equity_rows(session, full) if r.session_date <= X[cut]]
    assert short_rows == full_rows
    assert len(short_rows) == len(sessions_between(START, X[cut]))


def test_a_days_row_is_hidden_until_its_close(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    # X2's row exists in the table (the replay ran to END) but its mark uses X2's close, final
    # only at 20:00 New York: one microsecond earlier it must not be visible.
    world = gap_world(tmp_path, session, after=TRADED)
    run_id = run_replay(session, con, world.root, strategy_dir, START, END, ZERO_SLIP)

    just_before = visible_equity(session, close_of(X[2]) - timedelta(microseconds=1), run_id=run_id)
    assert [r.session_date for r in just_before] == sessions_between(START, X[1])
    at_close = visible_equity(session, close_of(X[2]), run_id=run_id)
    assert [r.session_date for r in at_close] == sessions_between(START, X[2])
    everything = visible_equity(session, close_of(END), run_id=run_id)
    assert [r.session_date for r in everything] == sessions_between(START, END)
