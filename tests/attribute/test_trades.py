"""Guards against trade outcomes leaking backwards in time: a Trade row visible before the bar
that closed it was final, an Attribution whose outcome_available_at is earlier than the end of
the re-entry window (so a Researcher at that moment would see a result that could still change),
and against bad rows reaching the ledger at all (a third leg, zero shares) or an R multiple
measured against the wrong stop.

API under test:

    asof.store.models.Trade                  CHECK leg IN (1, 2), CHECK shares > 0
    asof.store.queries.visible_trades(session, as_of) -> list[Trade]   available_at <= as_of
    asof.attribute.r_multiple(entry_price, exit_price, hard_stop) -> float
        (exit - entry) / (entry - hard_stop), long only
    asof.attribute.attribute_decision(decision, trades, *, equity_at_decision, window_end)
        -> Attribution (not added to the session)
        outcome_available_at = max(latest trade available_at, 20:00 New York of window_end)
        realized_return = sum(pnl) / equity_at_decision; is_correct = sum(pnl) > 0

Numbers are the golden RDDT legs (see tests/execute/test_simulate.py) and hand-made ones.
"""

from datetime import UTC, date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session

from asof.attribute import attribute_decision, r_multiple
from asof.store.models import Decision, Entity, Trade
from asof.store.queries import visible_trades
from tests.store.factories import add, new_decision, new_run, pk

ET = ZoneInfo("America/New_York")


def at_close(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, 20, tzinfo=ET).astimezone(UTC)


def at_open(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, 9, 30, tzinfo=ET).astimezone(UTC)


def trade(decision: Decision, **fields: Any) -> Trade:
    """RDDT leg 1 of 2026-05-01 unless overridden."""
    values: dict[str, Any] = {
        "decision_id": pk(decision),
        "run_id": decision.run_id,
        "strategy_id": "earnings_gap",
        "security_id": "S015043",
        "leg": 1,
        "direction": "long",
        "entry_at": at_open(date(2026, 5, 4)),
        "entry_price": 170.08,
        "shares": 27,
        "key_level": 156.49,
        "hard_stop": 148.212079,
        "risk_per_share": 170.08 - 148.212079,
        "exit_at": at_open(date(2026, 5, 11)),
        "exit_price": 153.50,
        "exit_reason": "break",
        "pnl": -447.66,
        "r_multiple": -16.58 / (170.08 - 148.212079),
        "available_at": at_close(date(2026, 5, 11)),
    }
    values.update(fields)
    return Trade(**values)


@pytest.fixture
def decision(session: Session, apple: Entity) -> Decision:
    run = new_run(session)
    return new_decision(
        session,
        run_id=pk(run),
        kind="trade",
        strategy_id="earnings_gap",
        entity_id=pk(apple),
        direction="long",
    )


# =============================================================================================
# Trade table
# =============================================================================================


def test_a_trade_round_trips_with_aware_times(session: Session, decision: Decision) -> None:
    row = add(session, trade(decision))
    assert row.id is not None
    assert row.available_at == at_close(date(2026, 5, 11))
    assert row.available_at.tzinfo is not None
    assert row.entry_at.tzinfo is not None


def test_leg_two_is_accepted(session: Session, decision: Decision) -> None:
    add(session, trade(decision, leg=2, shares=3))


@pytest.mark.parametrize("fields", [{"leg": 0}, {"leg": 3}, {"shares": 0}, {"shares": -27}])
def test_bad_legs_and_share_counts_are_rejected(
    session: Session, decision: Decision, fields: dict[str, Any]
) -> None:
    session.add(trade(decision, **fields))
    with pytest.raises(IntegrityError):
        session.commit()


def test_naive_trade_times_are_rejected(session: Session, decision: Decision) -> None:
    session.add(trade(decision, available_at=datetime(2026, 5, 12)))
    with pytest.raises(Exception, match="timezone"):
        session.commit()


# =============================================================================================
# visible_trades
# =============================================================================================


def test_trades_appear_when_their_exit_bar_is_final(session: Session, decision: Decision) -> None:
    first = add(session, trade(decision))
    add(
        session,
        trade(
            decision,
            leg=2,
            shares=3,
            exit_at=at_open(date(2026, 5, 26)),
            available_at=at_close(date(2026, 5, 26)),
        ),
    )
    exit_known = at_close(date(2026, 5, 11))
    assert [t.id for t in visible_trades(session, exit_known)] == [first.id]  # boundary: equal
    assert len(visible_trades(session, at_close(date(2026, 5, 26)))) == 2


def test_a_trade_is_hidden_until_its_exit_bar_is_final(
    session: Session, decision: Decision
) -> None:
    # Leakage: the leg exited at the 05-11 open, but the row is not knowable until that day's
    # bar is (20:00 New York); at 19:59 it does not exist yet.
    add(session, trade(decision))
    assert visible_trades(session, at_close(date(2026, 5, 11)) - timedelta(minutes=1)) == []
    assert visible_trades(session, at_open(date(2026, 5, 11))) == []


def test_visible_trades_refuses_naive_as_of(session: Session) -> None:
    with pytest.raises(ValueError):
        visible_trades(session, datetime(2026, 5, 12))


# =============================================================================================
# r_multiple / attribute_decision
# =============================================================================================


def test_r_multiple_is_pnl_per_share_over_initial_risk() -> None:
    assert r_multiple(170.08, 153.50, 148.212079) == pytest.approx(-0.758188, abs=1e-6)
    assert r_multiple(155.78, 143.50, 138.952079) == pytest.approx(-0.729740, abs=1e-6)
    assert r_multiple(100.0, 98.0, 98.0) == pytest.approx(-1.0)  # stopped at the stop
    assert r_multiple(100.0, 106.0, 98.0) == pytest.approx(3.0)


def test_attribution_sums_the_legs(session: Session, decision: Decision) -> None:
    legs = [
        trade(decision),
        trade(
            decision,
            leg=2,
            shares=3,
            pnl=-36.84,
            exit_at=at_open(date(2026, 5, 26)),
            available_at=at_close(date(2026, 5, 26)),
        ),
    ]
    # re-entry window: 20 sessions after 2026-05-01 ends 2026-05-29
    row = attribute_decision(
        decision, legs, equity_at_decision=100_000.0, window_end=date(2026, 5, 29)
    )
    assert row.decision_id == decision.id
    assert row.strategy_id == "earnings_gap"
    assert row.realized_return == pytest.approx(-484.50 / 100_000.0)
    assert row.is_correct is False
    assert row.outcome_available_at == at_close(date(2026, 5, 29))


def test_outcome_waits_for_the_last_exit_when_it_is_later_than_the_window(
    session: Session, decision: Decision
) -> None:
    late = trade(
        decision,
        pnl=812.0,
        exit_at=at_open(date(2026, 7, 2)),
        available_at=at_close(date(2026, 7, 2)),
    )
    row = attribute_decision(
        decision, [late], equity_at_decision=100_000.0, window_end=date(2026, 5, 29)
    )
    assert row.outcome_available_at == at_close(date(2026, 7, 2))
    assert row.is_correct is True
    assert row.realized_return == pytest.approx(0.00812)


def test_zero_pnl_is_not_correct(session: Session, decision: Decision) -> None:
    flat = trade(decision, pnl=0.0)
    row = attribute_decision(
        decision, [flat], equity_at_decision=100_000.0, window_end=date(2026, 5, 29)
    )
    assert row.is_correct is False
    assert row.realized_return == 0.0


def test_outcome_is_never_knowable_before_the_window_ends(
    session: Session, decision: Decision
) -> None:
    # Leakage: leg 1 closed on 05-11 but a re-entry was still possible until 05-29; the outcome
    # must stay hidden until then even though every trade row is visible earlier.
    row = attribute_decision(
        decision, [trade(decision)], equity_at_decision=100_000.0, window_end=date(2026, 5, 29)
    )
    assert row.outcome_available_at > at_close(date(2026, 5, 11))
    assert row.outcome_available_at == at_close(date(2026, 5, 29))
