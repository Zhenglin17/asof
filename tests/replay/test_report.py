"""Guards against a replay report that flatters or misstates a run: metrics mixed across runs,
win rate counted per leg instead of per decision, mean R pooled across legs, a drawdown taken
from trade order rather than the equity curve, the top-3 concentration hidden, and the number of
attempts left out (without it a reader cannot discount a lucky parameter set).

API under test (asof.replay). The metrics are computed by a function the
report renders, so they can be checked as numbers rather than parsed out of markdown:

    replay_metrics(session, run_id, initial_equity=100_000.0) -> ReplayMetrics
        decisions, trades_by_leg {leg: n}, win_rate (decisions with total pnl > 0 / decisions),
        mean_r_by_leg {leg: mean r}, mean_pnl_per_trade, total_return (sum pnl / initial
        equity), max_drawdown (fraction, of the realized equity curve stepped at each exit
        day), top3_share (sum of the 3 best decisions' pnl / total pnl), exit_reasons
        {reason: n}, attempt_count
    build_report(session, run_id) -> str (markdown)

No as_of: a report summarises one finished run. Assumptions that only real data can confirm
(check on the first real replay report): the top contributing trades it lists are inspected by
hand together with the bars they cite (PROGRESS: a bad bar would show as an outsized winner),
and attempt_count equals the number of parameter sets in the sweep log.

Hand-computed set (initial equity 100,000):
    D1 leg 1 +300 R 1.5 break, exit 01-10      D1 total +300
    D2 leg 1 -200 R -1.0 hard, exit 01-12      D2 total +300
    D2 leg 2 +500 R 2.5 observe_timeout, 01-20
    D3 leg 1 -100 R -0.5 break, exit 01-15     D3 total -100
    D4 leg 1 +100 R 0.5 max_sessions, 01-25    D4 total +100
  win rate 3/4; mean R leg 1 (1.5 - 1 - 0.5 + 0.5) / 4 = 0.125, leg 2 2.5; mean pnl per trade
  600 / 5 = 120; total return 0.006; equity 100,300 -> 100,100 -> 100,000 -> 100,500 ->
  100,600, max drawdown 300 / 100,300; top 3 (300 + 300 + 100) / 600 = 7/6.
"""

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from sqlmodel import Session

from asof.replay import build_report, replay_metrics
from asof.store.models import Decision, Entity, Run, StrategyVersion, Trade
from tests.store.factories import add, pk

ET = ZoneInfo("America/New_York")


def at(day: date, hh: int = 20) -> datetime:
    return datetime(day.year, day.month, day.day, hh, tzinfo=ET).astimezone(UTC)


def make_run(session: Session, version_id: int) -> Run:
    return add(
        session,
        Run(
            strategy_id="earnings_gap",
            strategy_version_id=version_id,
            mode="replay",
            as_of=at(date(2024, 1, 31)),
            started_at=datetime(2026, 10, 22, tzinfo=UTC),
            git_sha="0" * 40,
        ),
    )


def make_decision(session: Session, run: Run, entity: Entity, day: date) -> Decision:
    return add(
        session,
        Decision(
            run_id=pk(run),
            kind="trade",
            as_of=at(day),
            strategy_id="earnings_gap",
            entity_id=pk(entity),
            direction="long",
            payload={"judge": "take_all"},
        ),
    )


def make_trade(
    session: Session,
    decision: Decision,
    leg: int,
    pnl: float,
    r: float,
    reason: str,
    exit_day: date,
) -> Trade:
    entry, hard, shares = 100.0, 96.0, 50
    return add(
        session,
        Trade(
            decision_id=pk(decision),
            run_id=decision.run_id,
            strategy_id="earnings_gap",
            security_id="S_X",
            leg=leg,
            direction="long",
            entry_at=decision.as_of + timedelta(hours=13, minutes=30),  # next 09:30
            entry_price=entry,
            shares=shares,
            key_level=hard + 2.0,
            hard_stop=hard,
            risk_per_share=entry - hard,
            exit_at=at(exit_day, 9),
            exit_price=entry + pnl / shares,
            exit_reason=reason,
            pnl=pnl,
            r_multiple=r,
            available_at=at(exit_day),
        ),
    )


def version(session: Session, n: int) -> StrategyVersion:
    return add(
        session,
        StrategyVersion(
            strategy_id="earnings_gap",
            content_hash=f"h{n}",
            status="approved" if n == 0 else "proposed",
            proposed_by="human",
            registered_at=datetime(2026, 10, 22, n, tzinfo=UTC),
            change_kind="new_strategy" if n == 0 else "config",
        ),
    )


@pytest.fixture
def run_id(session: Session) -> int:
    entity = add(session, Entity(kind="company", ticker="XYZ", name="XYZ Inc.", cik=999))
    base, _other = version(session, 0), version(session, 1)  # two attempts
    run = make_run(session, pk(base))
    d1, d2, d3, d4 = (make_decision(session, run, entity, date(2024, 1, d)) for d in (2, 4, 8, 16))
    make_trade(session, d1, 1, 300.0, 1.5, "break", date(2024, 1, 10))
    make_trade(session, d2, 1, -200.0, -1.0, "hard", date(2024, 1, 12))
    make_trade(session, d2, 2, 500.0, 2.5, "observe_timeout", date(2024, 1, 20))
    make_trade(session, d3, 1, -100.0, -0.5, "break", date(2024, 1, 15))
    make_trade(session, d4, 1, 100.0, 0.5, "max_sessions", date(2024, 1, 25))
    # another run's trade must not leak into this report
    other_run = make_run(session, pk(base))
    stray = make_decision(session, other_run, entity, date(2024, 1, 2))
    make_trade(session, stray, 1, 10_000.0, 9.0, "break", date(2024, 1, 11))
    return pk(run)


# =============================================================================================
# normal path
# =============================================================================================


def test_metrics_match_the_hand_computation(session: Session, run_id: int) -> None:
    m = replay_metrics(session, run_id)
    assert m.decisions == 4
    assert dict(m.trades_by_leg) == {1: 4, 2: 1}
    assert m.win_rate == pytest.approx(0.75)
    assert m.mean_r_by_leg[1] == pytest.approx(0.125)
    assert m.mean_r_by_leg[2] == pytest.approx(2.5)
    assert m.mean_pnl_per_trade == pytest.approx(120.0)
    assert m.total_return == pytest.approx(0.006)
    assert m.max_drawdown == pytest.approx(300 / 100_300)
    assert m.top3_share == pytest.approx(700 / 600)
    assert m.top10_share == pytest.approx(1.0)  # four decisions: the top 10 are all of them
    assert m.traded_decisions == 4
    assert m.mean_pnl_by_leg == pytest.approx({1: 25.0, 2: 500.0})
    assert dict(m.exit_reasons) == {"break": 2, "hard": 1, "observe_timeout": 1, "max_sessions": 1}
    assert m.attempt_count == 2


def test_report_is_markdown_with_every_section(session: Session, run_id: int) -> None:
    text = build_report(session, run_id).lower()
    assert text.lstrip().startswith("#")
    for label in (
        "decisions",
        "win rate",
        "mean r",
        "total return",
        "max drawdown",
        "top 3",
        "exit reason",
        "holding",
        "attempt",
    ):
        assert label in text, label
    for reason in ("break", "hard", "observe_timeout", "max_sessions"):
        assert reason in text
    assert "75" in text  # win rate, however it is formatted (75%, 75.0 %)


# =============================================================================================
# boundaries
# =============================================================================================


def test_a_run_without_trades_reports_zeros(session: Session) -> None:
    base = version(session, 0)
    run = make_run(session, pk(base))
    m = replay_metrics(session, pk(run))
    assert m.decisions == 0
    assert dict(m.trades_by_leg) == {}
    assert m.total_return == 0.0
    assert m.max_drawdown == 0.0
    assert m.attempt_count == 1
    assert "#" in build_report(session, pk(run))


def test_drawdown_of_a_curve_that_only_rises_is_zero(session: Session) -> None:
    entity = add(session, Entity(kind="company", ticker="UP", name="Up Inc.", cik=998))
    run = make_run(session, pk(version(session, 0)))
    for i, day in enumerate((3, 5, 9)):
        d = make_decision(session, run, entity, date(2024, 2, day))
        make_trade(session, d, 1, 100.0 * (i + 1), 0.5, "break", date(2024, 2, day + 10))
    m = replay_metrics(session, pk(run))
    assert m.max_drawdown == 0.0
    assert m.win_rate == 1.0
    assert m.top3_share == pytest.approx(1.0)


def test_unknown_run_raises(session: Session) -> None:
    with pytest.raises((LookupError, ValueError)):
        replay_metrics(session, 12345)
