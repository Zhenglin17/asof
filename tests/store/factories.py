"""Row builders for store tests. Each builder inserts, commits and returns the row."""

import itertools
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlmodel import Session, SQLModel

from asof.store.models import (
    Attribution,
    Chunk,
    Decision,
    Document,
    Entity,
    Run,
    Source,
)

# Simulated moment used throughout: 2024-07-15 09:30 New York = 13:30 UTC.
T0 = datetime(2024, 7, 15, 13, 30, tzinfo=UTC)
APPLE_CIK = 320193
# All Select Sector SPDR funds file under this one CIK and differ only by series id.
SECTOR_SPDR_CIK = 1064641
XLK_SERIES = "S000006415"
XLE_SERIES = "S000006410"

_seq = itertools.count()


def add[R: SQLModel](session: Session, row: R) -> R:
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def pk(row: Any) -> int:
    assert row.id is not None
    return row.id


def new_source(session: Session, name: str = "sec_edgar") -> Source:
    return add(session, Source(name=name, base_url="https://www.sec.gov"))


def new_entity(
    session: Session,
    cik: int | None = APPLE_CIK,
    ticker: str = "AAPL",
    kind: str = "company",
    *,
    name: str = "Apple Inc.",
    series_id: str | None = None,
) -> Entity:
    entity = Entity(kind=kind, ticker=ticker, name=name, cik=cik, series_id=series_id)
    return add(session, entity)


def new_document(
    session: Session,
    *,
    source_id: int,
    available_at: datetime,
    entity_id: int | None = None,
    source_time: datetime | None = None,
    supersedes_id: int | None = None,
    doc_type: str = "10-Q",
) -> Document:
    n = next(_seq)
    doc = Document(
        source_id=source_id,
        entity_id=entity_id,
        doc_type=doc_type,
        external_id=f"0000320193-24-{n:06d}",
        title=f"{doc_type} #{n}",
        url=f"https://www.sec.gov/doc/{n}",
        content_hash=f"hash-{n}",
        source_time=source_time or available_at - timedelta(days=20),
        available_at=available_at,
        received_at=available_at + timedelta(minutes=5),
        supersedes_id=supersedes_id,
    )
    return add(session, doc)


def new_chunk(session: Session, *, document_id: int, ordinal: int = 0) -> Chunk:
    text = "Total net sales increased 5% year over year."
    return add(
        session,
        Chunk(
            document_id=document_id,
            ordinal=ordinal,
            section_path="Part I > Item 2 > Results of Operations",
            char_start=0,
            char_end=len(text),
            text=text,
        ),
    )


def new_run(session: Session) -> Run:
    return add(
        session,
        Run(
            strategy_id="momentum",
            mode="replay",
            as_of=T0,
            started_at=datetime(2026, 10, 4, tzinfo=UTC),
            model_id="test-model",
            model_training_cutoff=datetime(2025, 1, 1, tzinfo=UTC),
            git_sha="0" * 40,
        ),
    )


def new_decision(session: Session, *, run_id: int, **fields: Any) -> Decision:
    return add(session, Decision(run_id=run_id, as_of=T0, **fields))


def new_trade_decision(session: Session, *, run_id: int, entity_id: int) -> Decision:
    return new_decision(
        session,
        run_id=run_id,
        kind="trade",
        strategy_id="momentum",
        entity_id=entity_id,
        direction="long",
    )


def new_observation(session: Session, *, run_id: int, **fields: Any) -> Decision:
    return new_decision(
        session,
        run_id=run_id,
        kind="observation",
        topic="energy flows",
        statement="Money is flowing into energy names.",
        check="In two weeks, XLE relative to SPY is higher than today.",
        **fields,
    )


def new_attribution(
    session: Session, *, decision_id: int, outcome_available_at: datetime, **fields: Any
) -> Attribution:
    return add(
        session,
        Attribution(
            decision_id=decision_id,
            outcome_available_at=outcome_available_at,
            **fields,
        ),
    )
