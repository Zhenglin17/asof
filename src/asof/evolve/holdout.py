"""The sealed holdout: refuse any run whose window shares a day with it, unless the run opens it
on purpose, and let each strategy open it only once (a second look is an attempt nobody counts).

Only the run window [start, end] is checked. Lookback bars read before ``start`` and positions
still open at ``end`` reveal no result inside the holdout: reports count closed trades only."""

from datetime import UTC, date, datetime

from sqlmodel import Session, select

from asof.evolve.protocol import EvaluationProtocol
from asof.store.models import HoldoutOpening


class HoldoutError(ValueError):
    pass


def overlaps_holdout(protocol: EvaluationProtocol, start: date, end: date) -> bool:
    return start <= protocol.holdout.end and end >= protocol.holdout.start


def check_window(
    session: Session,
    protocol: EvaluationProtocol,
    strategy_id: str,
    start: date,
    end: date,
    *,
    open_holdout: bool,
) -> bool:
    """Call before registering a version or reading any bar. Returns True when the run opens the
    holdout and the caller must ``record_opening``; raises HoldoutError when it may not run."""
    if not overlaps_holdout(protocol, start, end):
        return False
    if strategy_id in protocol.seen_before_protocol:
        return False
    sealed = f"{protocol.holdout.start} .. {protocol.holdout.end}"
    if not open_holdout:
        raise HoldoutError(
            f"{start} .. {end} reaches the sealed holdout {sealed}. "
            "Pass --open-holdout to open it; each strategy may do so once."
        )
    opened = session.exec(
        select(HoldoutOpening).where(HoldoutOpening.strategy_id == strategy_id)
    ).first()
    if opened is not None:
        raise HoldoutError(
            f"{strategy_id} already opened the holdout {sealed} on {opened.opened_at:%Y-%m-%d} "
            f"(version {opened.strategy_version_id}); it cannot be opened again."
        )
    return True


def record_opening(
    session: Session,
    strategy_id: str,
    strategy_version_id: int,
    start: date,
    end: date,
    *,
    run_id: int | None = None,
) -> HoldoutOpening:
    row = HoldoutOpening(
        strategy_id=strategy_id,
        strategy_version_id=strategy_version_id,
        run_id=run_id,
        window_start=start,
        window_end=end,
        opened_at=datetime.now(UTC),
    )
    session.add(row)
    session.flush()
    return row
