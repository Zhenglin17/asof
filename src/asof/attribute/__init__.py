"""Outcome of a decision: its legs summed, hidden until it can no longer change."""

from collections.abc import Sequence
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from asof.execute.policies import r_multiple
from asof.store.models import Attribution, Decision, Trade

ET = ZoneInfo("America/New_York")

__all__ = ["attribute_decision", "r_multiple"]


def attribute_decision(
    decision: Decision,
    trades: Sequence[Trade],
    *,
    equity_at_decision: float,
    window_end: date,
) -> Attribution:
    """The decision's result once every leg is closed. Until the re-entry window has ended
    (20:00 New York of ``window_end``) another leg was still possible, so the outcome is not
    knowable before then even if every leg closed earlier."""
    window_close = datetime.combine(window_end, time(20), tzinfo=ET)
    known = max([window_close, *(t.available_at for t in trades)])
    total = sum(t.pnl for t in trades)
    return Attribution(
        decision_id=decision.id,  # pyright: ignore[reportArgumentType]
        strategy_id=decision.strategy_id,
        outcome_available_at=known,
        realized_return=total / equity_at_decision,
        is_correct=total > 0,
    )
