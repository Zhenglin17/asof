"""Guards against look-ahead leakage: visible_* must never return data published after as_of."""

from datetime import datetime, timedelta

import pytest
from sqlalchemy.exc import StatementError
from sqlmodel import Session, select

from asof.store.models import Document, Entity, Source
from asof.store.queries import visible_attributions, visible_chunks, visible_documents
from tests.store.factories import (
    T0,
    new_attribution,
    new_chunk,
    new_document,
    new_run,
    new_trade_decision,
    pk,
)

DAY = timedelta(days=1)


def test_documents_published_after_as_of_are_invisible(
    session: Session, source: Source, apple: Entity
) -> None:
    past = new_document(session, source_id=pk(source), available_at=T0 - DAY)
    new_document(session, source_id=pk(source), available_at=T0 + DAY)
    # Describes a period before T0 but was published after it: still invisible.
    new_document(
        session,
        source_id=pk(source),
        source_time=T0 - 30 * DAY,
        available_at=T0 + DAY,
    )

    assert [d.id for d in visible_documents(session, T0)] == [past.id]


def test_document_available_exactly_at_as_of_is_visible(
    session: Session, source: Source, apple: Entity
) -> None:
    doc = new_document(session, source_id=pk(source), available_at=T0)
    assert [d.id for d in visible_documents(session, T0)] == [doc.id]


def test_revision_returns_the_version_known_at_as_of(
    session: Session, source: Source, apple: Entity
) -> None:
    original = new_document(session, source_id=pk(source), available_at=T0 - 10 * DAY)
    revision = new_document(
        session,
        source_id=pk(source),
        available_at=T0 + 10 * DAY,
        supersedes_id=pk(original),
    )
    old_chunk = new_chunk(session, document_id=pk(original))
    new_chunk_row = new_chunk(session, document_id=pk(revision))

    assert [d.id for d in visible_documents(session, T0)] == [original.id]
    assert [d.id for d in visible_documents(session, T0 + 11 * DAY)] == [revision.id]

    assert [c.id for c in visible_chunks(session, T0)] == [old_chunk.id]
    assert [c.id for c in visible_chunks(session, T0 + 11 * DAY)] == [new_chunk_row.id]

    # Revisions append; the original row is never overwritten or deleted.
    assert len(session.exec(select(Document)).all()) == 2


def test_naive_as_of_is_rejected(session: Session) -> None:
    with pytest.raises(ValueError, match="timezone"):
        visible_documents(session, datetime(2024, 7, 15, 13, 30))


def test_naive_datetime_cannot_be_stored(session: Session, source: Source, apple: Entity) -> None:
    with pytest.raises(StatementError, match="timezone"):
        new_document(session, source_id=pk(source), available_at=datetime(2024, 7, 15))


def test_datetimes_round_trip_as_utc(session: Session, source: Source, apple: Entity) -> None:
    doc = new_document(session, source_id=pk(source), available_at=T0)
    session.expire_all()
    stored = session.get(Document, pk(doc))
    assert stored is not None
    assert stored.available_at == T0
    assert stored.available_at.utcoffset() == timedelta(0)


def test_attribution_outcome_after_as_of_is_invisible(session: Session, apple: Entity) -> None:
    decision = new_trade_decision(session, run_id=pk(new_run(session)))
    known = new_attribution(
        session, decision_id=pk(decision), outcome_available_at=T0 + 5 * DAY, is_correct=True
    )

    assert visible_attributions(session, T0 + 4 * DAY) == []
    assert [a.id for a in visible_attributions(session, T0 + 5 * DAY)] == [known.id]
