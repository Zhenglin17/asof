"""Guards against the real evidence path leaking or certifying the wrong thing: NVIDIA's Q2 FY27
results release must be unsearchable and uncitable one minute before EDGAR accepted it, and
searchable and citable one minute after, for NVIDIA only.

Runs on the real store ($ASOF_DATA_DIR/meta.db), opened read-only, and skips when no EDGAR
documents are there, so CI does not run it. The full-text index cannot be built read-only:
`asof evidence index` must have been run on the store first, or the first test fails.
Acceptance time hand-checked on 2026-10-21 (see tests/store/test_golden_documents.py).
"""

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import Engine
from sqlmodel import Session, create_engine, select

from asof.evidence import retrieve
from asof.evidence import validate as v
from asof.store import fts
from asof.store.db import default_db_path
from asof.store.models import Chunk, Document, Entity, Source
from asof.store.queries import visible_chunks

MINUTE = timedelta(minutes=1)
RELEASE = "0001045810-26-000073/q2fy27pr.htm"
# Index page "Accepted 2026-08-26 16:21:19" New York.
PUBLIC = datetime(2026, 8, 26, 20, 21, 19, tzinfo=UTC)
BEFORE = PUBLIC - MINUTE
AFTER = PUBLIC + MINUTE
# A phrase only this release uses, from the CEO quote in its first chunk.
PHRASE = "golden age of new AI labs and startups"
QUOTE = (
    "During the second quarter of fiscal 2027, NVIDIA returned approximately $26.0 billion to "
    "shareholders in the form of shares repurchased and cash dividends."
)
CLAIM = "NVIDIA returned about $26 billion to shareholders in the second quarter of fiscal 2027."


def read_only_engine() -> Engine:
    return create_engine(f"sqlite:///file:{default_db_path()}?mode=ro&uri=true")


def _has_edgar_documents() -> bool:
    """Skip only when the store is absent; any other failure (a locked file) must fail loudly."""
    if not default_db_path().exists():
        return False
    engine = read_only_engine()
    try:
        with Session(engine) as session:
            return (
                session.exec(select(Source).where(Source.name == "sec_edgar")).first() is not None
            )
    finally:
        engine.dispose()


pytestmark = [
    pytest.mark.realdata,
    pytest.mark.skipif(not _has_edgar_documents(), reason="no EDGAR documents in the real store"),
]


@pytest.fixture(scope="module")
def session() -> Iterator[Session]:
    # mode=ro: these tests must never write to the real store.
    engine = read_only_engine()
    with Session(engine) as s:
        yield s
    engine.dispose()


def entity_id(session: Session, ticker: str) -> int:
    found = session.exec(select(Entity).where(Entity.ticker == ticker)).one()
    assert found.id is not None
    return found.id


def release(session: Session) -> Document:
    return session.exec(select(Document).where(Document.external_id == RELEASE)).one()


def quoted_chunk(session: Session) -> Chunk:
    doc = release(session)
    (chunk,) = [c for c in visible_chunks(session, PUBLIC, document_id=doc.id) if QUOTE in c.text]
    return chunk


def citation(session: Session, claim: str = CLAIM) -> v.ClaimCitation:
    return v.ClaimCitation(
        claim=claim, evidence_kind="chunk", chunk_id=quoted_chunk(session).id, quote=QUOTE
    )


def draft(session: Session, as_of: datetime, ticker: str, claim: str = CLAIM) -> v.DecisionDraft:
    return v.DecisionDraft(
        as_of=as_of, entity_id=entity_id(session, ticker), citations=[citation(session, claim)]
    )


def test_index_covers_every_chunk(session: Session) -> None:
    assert fts.chunk_index_mismatch(session) is None, "run `asof evidence index` first"


def test_fixed_facts_still_hold(session: Session) -> None:
    doc = release(session)
    assert doc.available_at == PUBLIC
    assert doc.doc_type == "8-K"
    assert doc.entity_id == entity_id(session, "NVDA")


def test_release_is_not_found_one_minute_before_acceptance(session: Session) -> None:
    nvda = entity_id(session, "NVDA")
    hits = retrieve.search(session, BEFORE, PHRASE, entity_id=nvda, limit=50)
    assert release(session).id not in {h.document_id for h in hits}
    assert all(h.available_at <= BEFORE for h in hits)
    assert all(h.entity_id == nvda for h in hits)


def test_release_is_found_one_minute_after_acceptance(session: Session) -> None:
    nvda = entity_id(session, "NVDA")
    hits = retrieve.search(session, AFTER, PHRASE, entity_id=nvda, limit=10)
    assert release(session).id in {h.document_id for h in hits}

    as_8k = retrieve.search(session, AFTER, PHRASE, entity_id=nvda, base_forms=["8-K"], limit=10)
    assert release(session).id in {h.document_id for h in as_8k}
    as_10q = retrieve.search(session, AFTER, PHRASE, entity_id=nvda, base_forms=["10-Q"], limit=10)
    assert release(session).id not in {h.document_id for h in as_10q}


def test_real_quote_is_accepted_after_acceptance(session: Session) -> None:
    result = v.validate(session, draft(session, AFTER, "NVDA"))
    assert result.ok, result.failures


def test_real_quote_is_not_visible_before_acceptance(session: Session) -> None:
    result = v.validate(session, draft(session, BEFORE, "NVDA"))
    assert [(f.index, f.reason) for f in result.failures] == [(0, v.FailureReason.NOT_VISIBLE)]


def test_one_changed_number_is_rejected(session: Session) -> None:
    claim = CLAIM.replace("$26 billion", "$27 billion")
    result = v.validate(session, draft(session, AFTER, "NVDA", claim))
    assert [(f.index, f.reason) for f in result.failures] == [
        (0, v.FailureReason.NUMBER_NOT_IN_QUOTE)
    ]


def test_nvidia_chunk_cited_for_amd_is_rejected(session: Session) -> None:
    result = v.validate(session, draft(session, AFTER, "AMD"))
    assert [(f.index, f.reason) for f in result.failures] == [(0, v.FailureReason.ENTITY_MISMATCH)]


def test_wrong_scale_is_rejected(session: Session) -> None:
    # The release says "$26.0 billion"; a claim of "$26 million" has the right digits only.
    claim = CLAIM.replace("$26 billion", "$26 million")
    assert claim != CLAIM
    result = v.validate(session, draft(session, AFTER, "NVDA", claim))
    assert [(f.index, f.reason) for f in result.failures] == [
        (0, v.FailureReason.NUMBER_NOT_IN_QUOTE)
    ]


def real_draft(
    session: Session, external_id: str, ticker: str, as_of: datetime, quote: str, claim: str
) -> v.DecisionDraft:
    doc = session.exec(select(Document).where(Document.external_id == external_id)).one()
    (chunk,) = [c for c in visible_chunks(session, as_of, document_id=doc.id) if quote in c.text]
    citation = v.ClaimCitation(claim=claim, evidence_kind="chunk", chunk_id=chunk.id, quote=quote)
    return v.DecisionDraft(as_of=as_of, entity_id=entity_id(session, ticker), citations=[citation])


# PDD 20-F for 2025, accepted 2026-04-29 10:15:54 UTC; chunk 491084 in the store on 2026-10-22.
PDD_20F = "0001104659-26-050727/pdd-20251231x20f.htm"
PDD_AS_OF = datetime(2026, 5, 1, tzinfo=UTC)
PDD_QUOTE = (
    "We recorded a net income of RMB60,026.5 million, RMB112,434.5 million and RMB97,842.5 million"
)


@pytest.mark.parametrize(
    ("claim", "reasons"),
    [
        ("PDD net income was RMB97,842.5 million.", []),
        ("PDD revenue was RMB999.9 billion.", [v.FailureReason.NUMBER_NOT_IN_QUOTE]),
        # Renminbi amounts are not dollars.
        ("PDD net income was $97,842.5 million.", [v.FailureReason.NUMBER_NOT_IN_QUOTE]),
    ],
)
def test_pdd_renminbi_amounts(session: Session, claim: str, reasons: list[v.FailureReason]) -> None:
    result = v.validate(session, real_draft(session, PDD_20F, "PDD", PDD_AS_OF, PDD_QUOTE, claim))
    assert [f.reason for f in result.failures] == reasons


# ONTO 10-Q for Q1 2024, available 2024-05-09 21:18:49 UTC; the closing parenthesis of (281)
# sits in the next table cell (chunk 484269 in the store on 2026-10-22).
ONTO_10Q = "0000950170-24-057270/onto-20240330.htm"
ONTO_AS_OF = datetime(2024, 5, 10, tzinfo=UTC)
ONTO_QUOTE = "Other income (expense), net | 793 | (281 | )"


@pytest.mark.parametrize(
    ("claim", "reasons"),
    [
        ("Other income, net was (281).", []),
        ("Other income, net was 281.", [v.FailureReason.NUMBER_NOT_IN_QUOTE]),
        # Numbers are compared by shape: "-281" is not the table's "(281)" (false rejection).
        ("Other income, net was -281.", [v.FailureReason.NUMBER_NOT_IN_QUOTE]),
    ],
)
def test_onto_table_negative(session: Session, claim: str, reasons: list[v.FailureReason]) -> None:
    draft = real_draft(session, ONTO_10Q, "ONTO", ONTO_AS_OF, ONTO_QUOTE, claim)
    # Note 11 repeats the figures as "Total other income (expense), net | $ | 793 | ...", so
    # the quote matches one chunk only; real_draft asserts that.
    assert [f.reason for f in v.validate(session, draft).failures] == reasons
