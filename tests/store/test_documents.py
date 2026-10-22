"""Guards against unverifiable or leaking documents: silent rewrites, drifting chunk spans,
amendments that hide originals, and documents readable before they were public."""

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlmodel import Session, select

from asof.ingest.filing_text import ChunkSpan
from asof.store.documents import (
    DocumentConflict,
    DocumentSpec,
    add_document,
    audit_documents,
    get_or_create_source,
)
from asof.store.models import Chunk, Document, Entity, Source
from asof.store.queries import visible_chunks, visible_documents
from tests.store.factories import pk

# Measured: NVDA 8-K 0001045810-26-000073 was accepted 2026-08-26 16:21:19 New York.
AVAILABLE = datetime(2026, 8, 26, 20, 21, 19, tzinfo=UTC)
SECOND = timedelta(seconds=1)
TEXT = "Revenue was $96.2 billion.\n\nData Center revenue was $89.0 billion."


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def spans(text: str, section: str = "EX-99.1") -> list[ChunkSpan]:
    """One chunk per blank-line-separated paragraph, built by hand so this file tests the store
    on its own and not the chunker."""
    out: list[ChunkSpan] = []
    start = 0
    for ordinal, block in enumerate(text.split("\n\n")):
        begin = text.index(block, start)
        end = begin + len(block)
        out.append(ChunkSpan(ordinal, section, begin, end, text[begin:end]))
        start = end
    return out


def spec(
    entity_id: int | None,
    *,
    external_id: str = "0001045810-26-000073/q2fy27pr.htm",
    text: str = TEXT,
    doc_type: str = "8-K",
    available_at: datetime = AVAILABLE,
) -> DocumentSpec:
    return DocumentSpec(
        entity_id=entity_id,
        doc_type=doc_type,
        external_id=external_id,
        title="8-K EX-99.1 EX-99.1",
        url="https://www.sec.gov/Archives/edgar/data/1045810/000104581026000073/q2fy27pr.htm",
        text=text,
        source_time=datetime(2026, 8, 26, tzinfo=UTC),
        available_at=available_at,
        received_at=datetime(2026, 10, 21, 12, 0, tzinfo=UTC),
    )


def write_text_file(root: Path, external_id: str, text: str) -> Path:
    path = root / f"{external_id}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def count(session: Session, table: type[Document] | type[Chunk]) -> int:
    return len(session.exec(select(table)).all())


# --- get_or_create_source -------------------------------------------------------------------


def test_get_or_create_source_creates_once(session: Session) -> None:
    first = get_or_create_source(session, "sec_edgar", "https://www.sec.gov")
    second = get_or_create_source(session, "sec_edgar", "https://www.sec.gov")
    assert first.id is not None and first.id == second.id
    assert first.base_url == "https://www.sec.gov"
    assert len(session.exec(select(Source)).all()) == 1


def test_get_or_create_source_returns_an_existing_row(session: Session, source: Source) -> None:
    assert get_or_create_source(session, "sec_edgar").id == source.id


# --- add_document: normal path --------------------------------------------------------------


def test_add_document_writes_the_document_and_its_chunks(
    session: Session, source: Source, apple: Entity
) -> None:
    doc = add_document(session, source, spec(pk(apple)), spans(TEXT))
    assert doc is not None and doc.id is not None
    assert doc.source_id == source.id
    assert doc.entity_id == apple.id
    assert doc.doc_type == "8-K"
    assert doc.external_id == "0001045810-26-000073/q2fy27pr.htm"
    assert doc.content_hash == sha(TEXT)
    assert doc.available_at == AVAILABLE
    assert doc.supersedes_id is None
    session.commit()

    chunks = session.exec(select(Chunk).where(Chunk.document_id == doc.id)).all()
    assert sorted((c.ordinal, c.text) for c in chunks) == [
        (0, "Revenue was $96.2 billion."),
        (1, "Data Center revenue was $89.0 billion."),
    ]
    for c in chunks:
        assert TEXT[c.char_start : c.char_end] == c.text
        assert c.section_path == "EX-99.1"


def test_add_document_flushes_but_leaves_the_commit_to_the_caller(
    session: Session, source: Source, apple: Entity
) -> None:
    doc = add_document(session, source, spec(pk(apple)), spans(TEXT))
    assert doc is not None and doc.id is not None  # flushed: the id exists
    session.rollback()
    assert count(session, Document) == 0
    assert count(session, Chunk) == 0


# --- add_document: boundaries ---------------------------------------------------------------


def test_same_document_twice_is_a_no_op(session: Session, source: Source, apple: Entity) -> None:
    assert add_document(session, source, spec(pk(apple)), spans(TEXT)) is not None
    session.commit()
    assert add_document(session, source, spec(pk(apple)), spans(TEXT)) is None
    session.commit()
    assert count(session, Document) == 1
    assert count(session, Chunk) == 2


def test_same_external_id_with_different_text_conflicts(
    session: Session, source: Source, apple: Entity
) -> None:
    add_document(session, source, spec(pk(apple)), spans(TEXT))
    session.commit()
    changed = TEXT.replace("$96.2", "$96.3")
    with pytest.raises(DocumentConflict):
        add_document(session, source, spec(pk(apple), text=changed), spans(changed))
    assert issubclass(DocumentConflict, ValueError)


def test_same_external_id_under_another_source_is_a_different_document(
    session: Session, source: Source, apple: Entity
) -> None:
    other = get_or_create_source(session, "other_feed")
    add_document(session, source, spec(pk(apple)), spans(TEXT))
    changed = TEXT + " Extra."
    assert add_document(session, other, spec(pk(apple), text=changed), spans(changed)) is not None


def test_chunk_whose_text_differs_from_its_span_is_rejected(
    session: Session, source: Source, apple: Entity
) -> None:
    bad = [ChunkSpan(0, "EX-99.1", 0, 12, "Revenue was")]  # span is 12 chars, text 11
    with pytest.raises(ValueError):
        add_document(session, source, spec(pk(apple)), bad)
    # Validation happens before anything is written.
    assert count(session, Document) == 0


def test_chunk_span_past_the_end_of_the_text_is_rejected(
    session: Session, source: Source, apple: Entity
) -> None:
    tail = TEXT[-10:]
    bad = [ChunkSpan(0, "EX-99.1", len(TEXT) - 10, len(TEXT) + 5, tail)]
    with pytest.raises(ValueError):
        add_document(session, source, spec(pk(apple)), bad)


# --- leakage ----------------------------------------------------------------------------------


def test_document_and_chunks_are_invisible_one_second_before_available_at(
    session: Session, source: Source, apple: Entity
) -> None:
    doc = add_document(session, source, spec(pk(apple)), spans(TEXT))
    assert doc is not None
    session.commit()

    assert visible_documents(session, AVAILABLE - SECOND) == []
    assert visible_chunks(session, AVAILABLE - SECOND) == []
    assert [d.id for d in visible_documents(session, AVAILABLE)] == [doc.id]
    assert len(visible_chunks(session, AVAILABLE, document_id=pk(doc))) == 2


def test_amendment_does_not_hide_the_original_or_its_chunks(
    session: Session, source: Source, apple: Entity
) -> None:
    # Measured shape: NVDA 10-K 0001045810-26-000021 was available 2026-02-26 02:42:19 UTC. The
    # 10-K/A is invented: a partial amendment (e.g. only Part III) filed a month later.
    ten_k_at = datetime(2026, 2, 26, 2, 42, 19, tzinfo=UTC)
    amend_at = ten_k_at + timedelta(days=30)
    ten_k_text = "PART I\n\nItem 1. Business\n\nNVIDIA pioneered accelerated computing."
    amend_text = "PART III\n\nItem 10. Directors\n\nThe board has twelve members."
    ten_k = add_document(
        session,
        source,
        spec(
            pk(apple),
            external_id="0001045810-26-000021/nvda-20260125.htm",
            text=ten_k_text,
            doc_type="10-K",
            available_at=ten_k_at,
        ),
        spans(ten_k_text, "Cover"),
    )
    amendment = add_document(
        session,
        source,
        spec(
            pk(apple),
            external_id="0001045810-26-000099/nvda-20260125x10ka.htm",
            text=amend_text,
            doc_type="10-K/A",
            available_at=amend_at,
        ),
        spans(amend_text, "Cover"),
    )
    assert ten_k is not None and amendment is not None
    session.commit()
    assert amendment.supersedes_id is None

    # Before the amendment is public, only the 10-K exists.
    before = visible_documents(session, amend_at - SECOND)
    assert [d.id for d in before] == [ten_k.id]

    after = visible_documents(session, amend_at + timedelta(days=1))
    assert {d.id for d in after} == {ten_k.id, amendment.id}
    ten_k_chunks = visible_chunks(session, amend_at + timedelta(days=1), document_id=pk(ten_k))
    assert [c.text for c in ten_k_chunks] == ten_k_text.split("\n\n")


# --- audit_documents --------------------------------------------------------------------------


def _stored(session: Session, source: Source, entity_id: int | None, root: Path) -> Document:
    doc = add_document(session, source, spec(entity_id), spans(TEXT))
    assert doc is not None
    session.commit()
    write_text_file(root, doc.external_id, TEXT)
    return doc


def test_audit_is_clean_for_a_well_formed_store(
    session: Session, source: Source, apple: Entity, tmp_path: Path
) -> None:
    _stored(session, source, pk(apple), tmp_path / "text")
    assert audit_documents(session, tmp_path / "text") == []


def test_audit_flags_a_tampered_text_file(
    session: Session, source: Source, apple: Entity, tmp_path: Path
) -> None:
    root = tmp_path / "text"
    doc = _stored(session, source, pk(apple), root)
    # Appended after every chunk span, so only the hash check can see it.
    write_text_file(root, doc.external_id, TEXT + "\n\nInjected line.")
    problems = audit_documents(session, root)
    assert len(problems) == 1
    assert doc.external_id in problems[0]


def test_audit_flags_a_missing_text_file(
    session: Session, source: Source, apple: Entity, tmp_path: Path
) -> None:
    root = tmp_path / "text"
    doc = _stored(session, source, pk(apple), root)
    (root / f"{doc.external_id}.txt").unlink()
    problems = audit_documents(session, root)
    assert problems
    assert any(doc.external_id in p for p in problems)


def test_audit_flags_a_chunk_whose_span_no_longer_matches(
    session: Session, source: Source, apple: Entity, tmp_path: Path
) -> None:
    root = tmp_path / "text"
    doc = _stored(session, source, pk(apple), root)
    chunk = session.exec(select(Chunk).where(Chunk.document_id == doc.id, Chunk.ordinal == 1)).one()
    chunk.char_start += 1  # off by one: text no longer equals file_text[start:end]
    session.add(chunk)
    session.commit()
    problems = audit_documents(session, root)
    assert len(problems) == 1
    assert doc.external_id in problems[0]


def test_audit_flags_a_document_without_chunks(
    session: Session, source: Source, apple: Entity, tmp_path: Path
) -> None:
    root = tmp_path / "text"
    external_id = "0001045810-26-000073/q2fy27cfocommentary.htm"
    session.add(
        Document(
            source_id=pk(source),
            entity_id=pk(apple),
            doc_type="8-K",
            external_id=external_id,
            content_hash=sha(TEXT),
            source_time=datetime(2026, 8, 26, tzinfo=UTC),
            available_at=AVAILABLE,
            received_at=AVAILABLE + timedelta(days=1),
        )
    )
    session.commit()
    write_text_file(root, external_id, TEXT)
    problems = audit_documents(session, root)
    assert len(problems) == 1
    assert external_id in problems[0]


def test_audit_flags_a_document_without_an_entity(
    session: Session, source: Source, tmp_path: Path
) -> None:
    root = tmp_path / "text"
    doc = _stored(session, source, None, root)
    problems = audit_documents(session, root)
    assert len(problems) == 1
    assert doc.external_id in problems[0]
