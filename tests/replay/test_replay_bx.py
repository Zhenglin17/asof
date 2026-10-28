# Guards the B-Xtrender replay against sizing or deciding on what the signal night could not see
# (later bars, a split record filed after the night, a split whose ex-date is still ahead), against
# exits read from another night's flags, and against bookkeeping that drifts from the spec: half
# then the rest at the next opens, one live position per security, the gross cap stopping the
# night, a data gap leaving at the last close, equity marked with what is still held.
"""API under test:

    asof.replay.run_replay(session, con, root, strategy_dir, start, end, overrides=())
        dispatches on strategy.yaml ``kind``; a bx_trender directory runs the BX replay.

BX replay, night d (as_of = d 20:00 New York): held positions get d's bar and that night's
``exit_flags`` (no row -> both False), then EquityDaily (equity = initial + realized + sum
unrealized), then a finished position is attributed (window_end = d) and dropped (never filled
-> payload["cancelled"]), then candidates in rank order: a security with a live position is
skipped; shares = bx_shares(equity tonight, raw close of d, signal_atr over the bars visible at
as_of adjusted by the splits visible at as_of, gross exposure incl. orders queued tonight);
shares > 0 -> Decision + order for the next open; shares == 0 -> no Decision and the night stops.
Each exit sells floor(half), the last the rest; Trade leg 1, available_at = exit session 20:00.

World and expected numbers: tests/replay/bx_replay_world.py (provenance there). Slippage 5 bps
(the default), initial equity 100,000.

Interface points this file pins down where the spec left a choice:
- The Decision payload holds ``equity_at_decision``, ``atr`` and ``shares`` at its top level;
  the candidate's fields are read from ``payload["candidate"]`` when present, else from the top
  level (``cand``). Dates in the payload may be ISO strings.
- ATR at the signal night is in the price unit of that night's raw close: a split whose ex-date
  is after the night does not rescale it even when its record is already visible.
- A BX Decision carries security_id and symbol (new nullable Decision columns) and no
  entity_id; the relaxed trade check is tested in tests/store/test_decision_upgrade.py.

Assumptions to confirm on the real store (standard case, not checkable here):
- The S1 replay over a dev window writes one Decision per (security, signal night) with no live
  position: ``asof replay run --strategy bx_trender --start 2024-01-02 --end 2024-03-28`` then,
  per security, no two Decisions whose as_of falls between one's entry and its last Trade.
- A real liquid security with a split inside the 14 sessions before a signal (e.g. NVDA
  2024-06-10 10:1) has a Decision atr close to the chart's ATR(14) on adjusted prices, not ten
  times it.
"""

import dataclasses
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pytest
from sqlmodel import Session, col, select

from asof.replay import run_replay
from asof.store.market import market_root
from asof.store.models import Attribution, Decision, EquityDaily, Run, StrategyVersion, Trade
from asof.strategy import content_hash
from tests.features.test_bx import reference
from tests.replay.bx_replay_world import (
    AAA,
    AGAIN_I,
    BBB,
    BOTH_I,
    INITIAL,
    KEEP,
    SIGNAL_I,
    bar_of,
    build_world,
    buy,
    bx_strategy,
    expected_shares,
    first_night,
    flagged,
    hand_atr,
    nights,
    sell,
    sizing_of,
)
from tests.replay.scenario import END, START, TRADED, X
from tests.replay.test_replay import ny_date
from tests.scan.bx_world import CAL, Sec, close_of, known, ref_signal, split_row
from tests.scan.world import close_of as gap_close_of
from tests.scan.world import gap_world
from tests.strategy.files import write_strategy

ATR_DAYS = 14  # BX_DEFAULTS execute.sizing.atr_days
ATR_MULT = 2.0


@pytest.fixture
def strategy_dir(tmp_path: Path) -> Path:
    return bx_strategy(tmp_path / "strategies")


def decisions(session: Session, run_id: int) -> list[Decision]:
    stmt = select(Decision).where(Decision.run_id == run_id).order_by(col(Decision.id))
    return list(session.exec(stmt).all())


def trades(session: Session, run_id: int) -> list[Trade]:
    stmt = select(Trade).where(Trade.run_id == run_id).order_by(col(Trade.exit_at), col(Trade.id))
    return list(session.exec(stmt).all())


def attributions(session: Session, run_id: int) -> list[Attribution]:
    ids = [d.id for d in decisions(session, run_id)]
    stmt = select(Attribution).where(col(Attribution.decision_id).in_(ids))
    return list(session.exec(stmt).all())


def equity_rows(session: Session, run_id: int) -> dict[date, EquityDaily]:
    stmt = select(EquityDaily).where(EquityDaily.run_id == run_id)
    return {r.session_date: r for r in session.exec(stmt).all()}


def cand(decision: Decision) -> dict[str, Any]:
    payload = decision.payload
    inner = payload.get("candidate")
    return inner if isinstance(inner, dict) else payload


def day_of(value: Any) -> date:
    return value if type(value) is date else date.fromisoformat(str(value)[:10])


def sizing_key(decision: Decision) -> tuple[Any, ...]:
    p = decision.payload
    c = cand(decision)
    return (
        decision.as_of,
        c["security_id"],
        int(c["rank"]),
        p["shares"],
        pytest.approx(p["atr"], rel=1e-12),
        pytest.approx(p["equity_at_decision"], rel=1e-12),
    )


def trade_key(t: Trade) -> tuple[Any, ...]:
    return (
        t.security_id, t.leg, t.entry_at, t.entry_price, t.shares, t.exit_at, t.exit_price,
        t.exit_reason, t.pnl, t.available_at,
    )  # fmt: skip


def replay(
    session: Session,
    con: duckdb.DuckDBPyConnection,
    root: Path,
    strategy_dir: Path,
    start: date,
    end: date,
) -> int:
    return run_replay(session, con, root, strategy_dir, start, end)


S = CAL[SIGNAL_I]


@pytest.fixture(scope="module")
def ref() -> pd.DataFrame:
    return flagged([AAA, BBB])


def test_fixture_paths_are_as_documented(ref: pd.DataFrame) -> None:
    # passes today: a failure below is not a fixture error
    assert nights(ref, "S_AAA", "signal", S, S) == [S]
    assert first_night(ref, "S_AAA", "daily_red", CAL[SIGNAL_I + 1]) == CAL[SIGNAL_I + 3]
    assert first_night(ref, "S_AAA", "weekly_down", CAL[SIGNAL_I + 1]) == CAL[SIGNAL_I + 5]
    assert nights(ref, "S_AAA", "signal", CAL[SIGNAL_I + 1], CAL[SIGNAL_I + 13]) == []
    a = CAL[AGAIN_I]
    assert nights(ref, "S_AAA", "signal", a, a) == [a]
    assert first_night(ref, "S_AAA", "daily_red", CAL[AGAIN_I + 1]) == CAL[AGAIN_I + 6]
    assert first_night(ref, "S_AAA", "weekly_down", CAL[AGAIN_I + 1]) == CAL[AGAIN_I + 10]
    assert len(nights(ref, "S_AAA", "signal", CAL[AGAIN_I + 1], CAL[AGAIN_I + 10])) == 3
    assert nights(ref, "S_AAA", "signal", CAL[AGAIN_I + 11], CAL[AGAIN_I + 43]) == [
        CAL[AGAIN_I + 43]
    ]
    b = CAL[BOTH_I]
    assert set(nights(ref, "S_AAA", "signal", b, b) + nights(ref, "S_BBB", "signal", b, b)) == {b}
    assert len(ref[(ref["session_date"] == b) & ref["signal"]]) == 2


# =============================================================================================
# normal path
# =============================================================================================


def test_one_signal_buys_the_next_open_and_sells_half_then_the_rest(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    root = market_root(tmp_path)
    build_world(root, [AAA])
    red, down = CAL[SIGNAL_I + 3], CAL[SIGNAL_I + 5]
    end = CAL[SIGNAL_I + 6]  # the weekly_down tranche fills at this open

    run_id = replay(session, con, root, strategy_dir, S, end)

    run = session.get(Run, run_id)
    assert run is not None
    assert (run.mode, run.strategy_id) == ("replay", "bx_trender")
    version = session.get(StrategyVersion, run.strategy_version_id)
    assert version is not None
    assert version.content_hash == content_hash(strategy_dir)

    atr = hand_atr(AAA.bars(), S, ATR_DAYS)
    close = bar_of(AAA, S)["close"]
    n = expected_shares(sizing_of(strategy_dir), equity=INITIAL, close=close, atr=atr)
    assert n > 1  # fixture: two tranches

    [decision] = decisions(session, run_id)
    assert decision.kind == "trade"
    assert decision.direction == "long"
    assert decision.strategy_id == "bx_trender"
    assert decision.as_of == close_of(S)
    # a BX decision names the security it trades, not an entity
    assert (decision.security_id, decision.symbol) == ("S_AAA", "AAA")
    assert decision.entity_id is None
    assert decision.payload["equity_at_decision"] == pytest.approx(INITIAL)
    assert decision.payload["atr"] == pytest.approx(atr, rel=1e-9)
    assert decision.payload["shares"] == n
    c = cand(decision)
    assert (c["security_id"], c["symbol"], int(c["rank"])) == ("S_AAA", "AAA", 1)
    assert day_of(c["session"]) == S

    entry = buy(bar_of(AAA, CAL[SIGNAL_I + 1])["open"])
    first, second = trades(session, run_id)
    half = n // 2
    for t in (first, second):
        assert t.decision_id == decision.id
        assert (t.leg, t.security_id, t.symbol, t.strategy_id) == (1, "S_AAA", "AAA", "bx_trender")
        assert ny_date(t.entry_at) == CAL[SIGNAL_I + 1]
        assert t.entry_price == pytest.approx(entry, rel=1e-12)
        assert t.hard_stop == pytest.approx(entry - ATR_MULT * atr, rel=1e-9)

    p1 = sell(bar_of(AAA, CAL[SIGNAL_I + 4])["open"])  # daily_red read at `red`, filled next open
    assert (first.shares, first.exit_reason) == (half, "daily_red")
    assert ny_date(first.exit_at) == CAL[SIGNAL_I + 4]
    assert first.exit_price == pytest.approx(p1, rel=1e-12)
    assert first.pnl == pytest.approx(half * (p1 - entry), rel=1e-9)
    assert first.available_at == close_of(CAL[SIGNAL_I + 4])
    assert red < CAL[SIGNAL_I + 4]

    p2 = sell(bar_of(AAA, end)["open"])
    assert (second.shares, second.exit_reason) == (n - half, "weekly_down")
    assert ny_date(second.exit_at) == end
    assert second.exit_price == pytest.approx(p2, rel=1e-12)
    assert second.pnl == pytest.approx((n - half) * (p2 - entry), rel=1e-9)
    assert second.available_at == close_of(end)
    assert down < end

    [attribution] = attributions(session, run_id)
    assert attribution.decision_id == decision.id
    assert attribution.strategy_id == "bx_trender"
    assert attribution.outcome_available_at == close_of(end)  # attributed the night it finished
    total = first.pnl + second.pnl
    assert attribution.realized_return == pytest.approx(total / INITIAL, rel=1e-9)
    assert attribution.is_correct is (total > 0)


def test_equity_marks_what_is_still_held_between_the_two_exits(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    root = market_root(tmp_path)
    build_world(root, [AAA])
    end = CAL[SIGNAL_I + 6]
    run_id = replay(session, con, root, strategy_dir, S, end)
    rows = equity_rows(session, run_id)
    assert set(rows) == {d for d in CAL if S <= d <= end}

    atr = hand_atr(AAA.bars(), S, ATR_DAYS)
    n = expected_shares(
        sizing_of(strategy_dir), equity=INITIAL, close=bar_of(AAA, S)["close"], atr=atr
    )
    entry = buy(bar_of(AAA, CAL[SIGNAL_I + 1])["open"])
    half, rest = n // 2, n - n // 2

    night = rows[S]  # the order is queued, nothing held
    assert (night.equity, night.realized, night.unrealized) == (INITIAL, 0.0, 0.0)
    assert (night.gross_exposure, night.open_positions) == (0.0, 0)

    between = CAL[SIGNAL_I + 4]  # half sold at this open, the rest held at its close
    close = bar_of(AAA, between)["close"]
    realized = half * (sell(bar_of(AAA, between)["open"]) - entry)
    row = rows[between]
    assert row.realized == pytest.approx(realized, rel=1e-9)
    assert row.unrealized == pytest.approx(rest * (close - entry), rel=1e-9)
    assert row.equity == pytest.approx(INITIAL + realized + rest * (close - entry), rel=1e-12)
    assert row.gross_exposure == pytest.approx(rest * close, rel=1e-12)
    assert row.open_positions == 1
    assert row.available_at == close_of(between)

    last = rows[end]
    assert last.unrealized == pytest.approx(0.0, abs=1e-9)
    assert last.open_positions == 0
    assert last.equity == pytest.approx(INITIAL + last.realized, rel=1e-12)


def test_a_split_visible_before_the_signal_night_adjusts_the_atr(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    # 2:1 on S - 3, its record on file from 00:00 of the ex-date (ingest rule): the stored
    # prices before it are twice the walk's, the ATR window spans it
    ex = CAL[SIGNAL_I - 3]
    split = dataclasses.replace(AAA, raw_scale=((None, ex, 2.0),))
    root = market_root(tmp_path)
    build_world(root, [split], splits=[split_row("AAA", ex, 1.0, 2.0)])

    run_id = replay(session, con, root, strategy_dir, S, S)

    atr = hand_atr(split.bars(), S, ATR_DAYS, splits=[(ex, 2.0)])
    raw_atr = hand_atr(split.bars(), S, ATR_DAYS)
    assert raw_atr > 2 * atr  # fixture: the unadjusted ATR would be far off
    [decision] = decisions(session, run_id)
    assert decision.payload["atr"] == pytest.approx(atr, rel=1e-9)
    close = bar_of(split, S)["close"]
    assert decision.payload["shares"] == expected_shares(
        sizing_of(strategy_dir), equity=INITIAL, close=close, atr=atr
    )


# =============================================================================================
# boundaries
# =============================================================================================


def test_the_gross_cap_ends_the_night_at_the_first_candidate_without_shares(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, ref: pd.DataFrame
) -> None:
    # the whole book may take 0.98% of equity: rank 1 (AAA, 16 shares at 58.89 = 942.26) fits,
    # then 37.74 of room is less than one share of rank 2 (BBB at 49.72)
    strategy_dir = bx_strategy(tmp_path / "strategies", sizing={"max_gross_pct": 0.98})
    sizing = sizing_of(strategy_dir)
    root = market_root(tmp_path)
    build_world(root, [AAA, BBB])
    b = CAL[BOTH_I]
    tonight = pd.DataFrame(ref[ref["session_date"] == b])
    tonight["rise"] = tonight["bx_weekly"] - tonight["bx_weekly_prev"]
    order = list(
        tonight.sort_values(["rise", "security_id"], ascending=[False, True])["security_id"]
    )
    secs = {"S_AAA": AAA, "S_BBB": BBB}
    first, second = secs[order[0]], secs[order[1]]
    c1, c2 = bar_of(first, b)["close"], bar_of(second, b)["close"]
    a1, a2 = hand_atr(first.bars(), b, ATR_DAYS), hand_atr(second.bars(), b, ATR_DAYS)
    n1 = expected_shares(sizing, equity=INITIAL, close=c1, atr=a1)
    assert n1 > 0
    assert expected_shares(sizing, equity=INITIAL, close=c2, atr=a2) > 0  # alone it would fit
    assert expected_shares(sizing, equity=INITIAL, close=c2, atr=a2, gross=n1 * c1) == 0

    run_id = replay(session, con, root, strategy_dir, b, b)

    [decision] = decisions(session, run_id)
    assert cand(decision)["security_id"] == first.security_id
    assert int(cand(decision)["rank"]) == 1
    assert decision.payload["shares"] == n1


@pytest.mark.parametrize(
    "sizing",
    [
        # one position may take $55: AAA (58.89) not one share, BBB (49.72) one
        {"max_position_pct": 0.055},
        # $3.205 of risk: below AAA's 2 x ATR 3.212, above BBB's 3.198
        {"risk_pct": 0.003205},
    ],
    ids=["position_cap", "risk_budget"],
)
def test_a_candidate_without_shares_for_another_reason_is_skipped_not_the_night(
    tmp_path: Path,
    session: Session,
    con: duckdb.DuckDBPyConnection,
    ref: pd.DataFrame,
    sizing: dict[str, float],
) -> None:
    # only an exhausted gross room ends the night; rank 1 priced out by its own position cap or
    # by the risk budget is passed over and rank 2 is bought
    strategy_dir = bx_strategy(tmp_path / "strategies", sizing=sizing)
    cfg_sizing = sizing_of(strategy_dir)
    root = market_root(tmp_path)
    build_world(root, [AAA, BBB])
    b = CAL[BOTH_I]
    tonight = pd.DataFrame(ref[ref["session_date"] == b])
    tonight["rise"] = tonight["bx_weekly"] - tonight["bx_weekly_prev"]
    ranked = tonight.sort_values(["rise", "security_id"], ascending=[False, True])
    assert list(ranked["security_id"]) == ["S_AAA", "S_BBB"]  # fixture: AAA ranks first
    c1, c2 = bar_of(AAA, b)["close"], bar_of(BBB, b)["close"]
    a1, a2 = hand_atr(AAA.bars(), b, ATR_DAYS), hand_atr(BBB.bars(), b, ATR_DAYS)
    assert expected_shares(cfg_sizing, equity=INITIAL, close=c1, atr=a1) == 0
    n2 = expected_shares(cfg_sizing, equity=INITIAL, close=c2, atr=a2)
    assert n2 == 1
    room = INITIAL * cfg_sizing.max_gross_pct / 100
    assert room >= c1  # the gross room alone would hold a share of rank 1

    run_id = replay(session, con, root, strategy_dir, b, b)

    [decision] = decisions(session, run_id)
    assert cand(decision)["security_id"] == "S_BBB"
    assert int(cand(decision)["rank"]) == 2
    assert decision.payload["shares"] == n2


def test_a_held_security_gets_no_second_decision_until_it_is_sold(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    # AAA signals three more times while held; after its last share is sold (AGAIN + 11) the
    # next signal, AGAIN + 43, is a new decision sized on the equity then
    root = market_root(tmp_path)
    build_world(root, [AAA])
    start, again = CAL[AGAIN_I], CAL[AGAIN_I + 43]

    run_id = replay(session, con, root, strategy_dir, start, again)

    first, second = decisions(session, run_id)
    assert (first.as_of, second.as_of) == (close_of(start), close_of(again))
    assert cand(second)["security_id"] == "S_AAA"
    done = [t for t in trades(session, run_id) if t.decision_id == first.id]
    assert [t.exit_reason for t in done] == ["daily_red", "weekly_down"]
    assert ny_date(done[-1].exit_at) == CAL[AGAIN_I + 11]
    equity = INITIAL + sum(t.pnl for t in done)
    assert second.payload["equity_at_decision"] == pytest.approx(equity, rel=1e-12)
    atr = hand_atr(AAA.bars(), again, ATR_DAYS)
    assert second.payload["shares"] == expected_shares(
        sizing_of(strategy_dir), equity=equity, close=bar_of(AAA, again)["close"], atr=atr
    )


def test_five_sessions_without_a_bar_leave_at_the_last_close(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    # bought on S + 1 (no exit flag that night); no AAA bar on S + 2 .. S + 6:
    # max_missing_sessions = 5 -> sold on the fifth such night at S + 1's close, less slippage
    root = market_root(tmp_path)
    gap = [CAL[SIGNAL_I + k] for k in range(2, 7)]
    build_world(root, [AAA, KEEP], drop={"S_AAA": gap})
    end = gap[-1]

    run_id = replay(session, con, root, strategy_dir, S, end)

    n = expected_shares(
        sizing_of(strategy_dir),
        equity=INITIAL,
        close=bar_of(AAA, S)["close"],
        atr=hand_atr(AAA.bars(), S, ATR_DAYS),
    )
    [only] = trades(session, run_id)
    assert (only.shares, only.exit_reason) == (n, "data_gap")
    assert only.exit_price == pytest.approx(sell(bar_of(AAA, CAL[SIGNAL_I + 1])["close"]))
    assert ny_date(only.exit_at) == end
    assert only.available_at == close_of(end)
    assert len(attributions(session, run_id)) == 1


def test_four_sessions_without_a_bar_do_not_exit(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    root = market_root(tmp_path)
    gap = [CAL[SIGNAL_I + k] for k in range(2, 6)]
    build_world(root, [AAA, KEEP], drop={"S_AAA": gap})

    run_id = replay(session, con, root, strategy_dir, S, CAL[SIGNAL_I + 6])

    assert trades(session, run_id) == []
    n = expected_shares(
        sizing_of(strategy_dir),
        equity=INITIAL,
        close=bar_of(AAA, S)["close"],
        atr=hand_atr(AAA.bars(), S, ATR_DAYS),
    )
    held_close = bar_of(AAA, CAL[SIGNAL_I + 1])["close"]
    row = equity_rows(session, run_id)[gap[-1]]  # still held, marked at its last close
    assert row.open_positions == 1
    assert row.gross_exposure == pytest.approx(n * held_close, rel=1e-12)


def test_an_order_without_a_bar_at_its_open_is_cancelled(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    root = market_root(tmp_path)
    build_world(root, [AAA, KEEP], drop={"S_AAA": [CAL[SIGNAL_I + 1]]})

    run_id = replay(session, con, root, strategy_dir, S, CAL[SIGNAL_I + 2])

    [decision] = decisions(session, run_id)
    assert trades(session, run_id) == []
    assert attributions(session, run_id) == []
    assert decision.payload["cancelled"] == "no_bar"


def test_an_earnings_gap_directory_still_replays_through_the_dispatch(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection
) -> None:
    # tests/replay/scenario.py's traded path: leg 1 in at 56.5 with 125 shares, out at 53.4
    world = gap_world(tmp_path, session, after=TRADED)
    eg = write_strategy(tmp_path / "eg_strategies")
    run_id = run_replay(session, con, world.root, eg, START, END, ("execute.fills.slippage_bps=0",))
    run = session.get(Run, run_id)
    assert run is not None and run.strategy_id == "earnings_gap"
    legs = trades(session, run_id)
    assert [(t.leg, t.shares, t.entry_price, t.exit_reason) for t in legs] == [
        (1, 125, 56.5, "break"),
        (2, 40, 51.0, "hard"),
    ]
    assert legs[0].available_at == gap_close_of(X[3])


# =============================================================================================
# leakage
# =============================================================================================


def test_bars_after_the_signal_night_do_not_move_its_decision(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    # the same signal night read from a store cut at S, from the full store and from one whose
    # prices from S + 1 on are 30% lower; the longer replays run past S
    cut, full, other = (market_root(tmp_path / n) for n in ("cut", "full", "other"))
    build_world(cut, [AAA], cut=S)
    build_world(full, [AAA])
    build_world(other, [dataclasses.replace(AAA, raw_scale=((CAL[SIGNAL_I + 1], None, 0.7),))])

    [base] = decisions(session, replay(session, con, cut, strategy_dir, S, S))
    for root in (full, other):
        run_id = replay(session, con, root, strategy_dir, S, CAL[SIGNAL_I + 6])
        signal_night = [d for d in decisions(session, run_id) if d.as_of == close_of(S)]
        assert [sizing_key(d) for d in signal_night] == [sizing_key(base)]


@pytest.mark.parametrize("filed", ["on_the_ex_date", "before_the_signal_night"])
def test_a_split_with_its_ex_date_after_the_signal_night_does_not_move_its_decision(
    tmp_path: Path,
    session: Session,
    con: duckdb.DuckDBPyConnection,
    strategy_dir: Path,
    filed: str,
) -> None:
    # 2:1 on S + 2: stored prices from it on are half the walk's. The replay runs through it;
    # the signal night's ATR and shares are those of a store with no split at all, in the unit
    # of the night's raw close
    ex = CAL[SIGNAL_I + 2]
    available = known(ex) if filed == "on_the_ex_date" else known(CAL[SIGNAL_I - 5])
    split = dataclasses.replace(AAA, raw_scale=((ex, None, 0.5),))
    with_split, plain = market_root(tmp_path / "split"), market_root(tmp_path / "plain")
    build_world(
        with_split, [split], splits=[split_row("AAA", ex, 1.0, 2.0, available_at=available)]
    )
    build_world(plain, [AAA], cut=S)

    [base] = decisions(session, replay(session, con, plain, strategy_dir, S, S))
    run_id = replay(session, con, with_split, strategy_dir, S, CAL[SIGNAL_I + 4])
    [decision] = [d for d in decisions(session, run_id) if d.as_of == close_of(S)]

    assert sizing_key(decision) == sizing_key(base)
    assert decision.payload["atr"] == pytest.approx(hand_atr(AAA.bars(), S, ATR_DAYS), rel=1e-9)


SPL_I = 220
SPL_EX = CAL[SPL_I - 3]
# a 1-for-2 reverse split three sessions before the signal night: stored prices before it are
# half the walk's (seed 808, as the scan tests' S_SPL)
SPL = Sec("S_SPL", "SPL", seed=808, raw_scale=((None, SPL_EX, 0.5),))


def test_a_split_filed_after_the_signal_night_does_not_move_its_decision(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    # the record reaches the store an hour after the signal night's close (constructed late
    # feed): that night sees the raw prices, a jump on SPL_EX included, and sizes on them
    night = CAL[SPL_I]
    late = close_of(night) + timedelta(hours=1)
    raw = pd.DataFrame([{**b, "security_id": "S_SPL"} for b in SPL.bars(cut=night)])
    assert ref_signal(reference(raw).iloc[-1].to_dict(), "mid")  # fixture: raw prices signal
    with_rec, without = market_root(tmp_path / "with"), market_root(tmp_path / "without")
    build_world(with_rec, [SPL], splits=[split_row("SPL", SPL_EX, 2.0, 1.0, available_at=late)])
    build_world(without, [SPL], cut=night)

    [base] = decisions(session, replay(session, con, without, strategy_dir, night, night))
    run_id = replay(session, con, with_rec, strategy_dir, night, CAL[SPL_I + 3])
    [decision] = [d for d in decisions(session, run_id) if d.as_of == close_of(night)]

    assert sizing_key(decision) == sizing_key(base)
    raw_atr = hand_atr(SPL.bars(), night, ATR_DAYS)
    assert raw_atr != pytest.approx(hand_atr(SPL.bars(), night, ATR_DAYS, [(SPL_EX, 0.5)]))
    assert decision.payload["atr"] == pytest.approx(raw_atr, rel=1e-9)


def test_an_exit_read_at_a_close_does_not_depend_on_later_bars(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    # daily_red is read at S + 3 and filled at S + 4's open: a store that ends at S + 4 books
    # the same tranche as the full one
    full, cut = market_root(tmp_path / "full"), market_root(tmp_path / "cut")
    build_world(full, [AAA])
    fill = CAL[SIGNAL_I + 4]
    build_world(cut, [AAA], cut=fill)

    long_run = replay(session, con, full, strategy_dir, S, CAL[SIGNAL_I + 6])
    short_run = replay(session, con, cut, strategy_dir, S, fill)

    [short] = trades(session, short_run)
    first = trades(session, long_run)[0]
    assert short.exit_reason == "daily_red"
    assert trade_key(short) == trade_key(first)
