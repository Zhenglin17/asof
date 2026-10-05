"""The only read path into the metadata store.

Every function takes `as_of` and returns only what was public at that moment. Other modules
must not write their own SELECTs against these tables; keeping the time filter in one file
is what makes look-ahead leakage reviewable.

Source and Entity are identity lookups, not market information, so they have no visible_*.
"""

from datetime import datetime

from sqlalchemy import ColumnElement, and_
from sqlalchemy.orm import aliased
from sqlmodel import Session, col, select

from asof.store.models import Attribution, Chunk, Decision, Document, Event, StrategyVersion


def _require_aware(as_of: datetime) -> None:
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError(f"as_of must carry a timezone, got naive {as_of!r}")


def _document_is_visible(as_of: datetime) -> ColumnElement[bool]:
    """Published by as_of, and not replaced by a revision that is itself published by as_of."""
    revision = aliased(Document)
    replaced = (
        select(revision.id)
        .where(
            col(revision.supersedes_id) == col(Document.id),
            col(revision.available_at) <= as_of,
        )
        .exists()
    )
    return and_(col(Document.available_at) <= as_of, ~replaced)


def visible_documents(
    session: Session,
    as_of: datetime,
    *,
    entity_id: int | None = None,
    doc_type: str | None = None,
) -> list[Document]:
    _require_aware(as_of)
    stmt = select(Document).where(_document_is_visible(as_of))
    if entity_id is not None:
        stmt = stmt.where(col(Document.entity_id) == entity_id)
    if doc_type is not None:
        stmt = stmt.where(col(Document.doc_type) == doc_type)
    stmt = stmt.order_by(col(Document.available_at), col(Document.id))
    return list(session.exec(stmt).all())


def visible_chunks(
    session: Session, as_of: datetime, *, document_id: int | None = None
) -> list[Chunk]:
    """Chunks of visible documents only; chunks of a replaced version disappear with it."""
    _require_aware(as_of)
    stmt = (
        select(Chunk)
        .join(Document, col(Chunk.document_id) == col(Document.id))
        .where(_document_is_visible(as_of))
    )
    if document_id is not None:
        stmt = stmt.where(col(Chunk.document_id) == document_id)
    stmt = stmt.order_by(col(Chunk.document_id), col(Chunk.ordinal))
    return list(session.exec(stmt).all())


def visible_events(
    session: Session, as_of: datetime, *, entity_id: int | None = None
) -> list[Event]:
    _require_aware(as_of)
    stmt = select(Event).where(col(Event.available_at) <= as_of)
    if entity_id is not None:
        stmt = stmt.where(col(Event.entity_id) == entity_id)
    stmt = stmt.order_by(col(Event.available_at), col(Event.id))
    return list(session.exec(stmt).all())


def visible_decisions(
    session: Session,
    as_of: datetime,
    *,
    kind: str | None = None,
    strategy_id: str | None = None,
) -> list[Decision]:
    _require_aware(as_of)
    stmt = select(Decision).where(col(Decision.as_of) <= as_of)
    if kind is not None:
        stmt = stmt.where(col(Decision.kind) == kind)
    if strategy_id is not None:
        stmt = stmt.where(col(Decision.strategy_id) == strategy_id)
    stmt = stmt.order_by(col(Decision.as_of), col(Decision.id))
    return list(session.exec(stmt).all())


def visible_attributions(
    session: Session, as_of: datetime, *, strategy_id: str | None = None
) -> list[Attribution]:
    """Outcomes not yet knowable at as_of stay hidden, so a Researcher cannot peek at answers."""
    _require_aware(as_of)
    stmt = select(Attribution).where(col(Attribution.outcome_available_at) <= as_of)
    if strategy_id is not None:
        stmt = stmt.where(col(Attribution.strategy_id) == strategy_id)
    stmt = stmt.order_by(col(Attribution.outcome_available_at), col(Attribution.id))
    return list(session.exec(stmt).all())


def visible_strategy_versions(
    session: Session, as_of: datetime, *, strategy_id: str | None = None
) -> list[StrategyVersion]:
    _require_aware(as_of)
    stmt = select(StrategyVersion).where(col(StrategyVersion.registered_at) <= as_of)
    if strategy_id is not None:
        stmt = stmt.where(col(StrategyVersion.strategy_id) == strategy_id)
    stmt = stmt.order_by(col(StrategyVersion.registered_at), col(StrategyVersion.id))
    return list(session.exec(stmt).all())
