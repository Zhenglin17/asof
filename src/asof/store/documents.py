"""Writing documents and their chunks, append-only, and checking them against their text files.

A document's full text lives in a file next to the store (``<text_root>/<external_id>.txt``);
the database keeps its hash and the chunks. A citation is only checkable if every chunk is the
exact slice ``text[char_start:char_end]`` of that file, so both writing and auditing enforce it.
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sqlmodel import Session, col, select

from asof.ingest.filing_text import ChunkSpan
from asof.store.models import Chunk, Document, Source


class DocumentConflict(ValueError):
    """The source already published this id with different content; stored rows never change."""


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_or_create_source(session: Session, name: str, base_url: str | None = None) -> Source:
    found = session.exec(select(Source).where(Source.name == name)).first()
    if found is not None:
        return found
    source = Source(name=name, base_url=base_url)
    session.add(source)
    session.flush()
    return source


@dataclass(frozen=True)
class DocumentSpec:
    entity_id: int | None
    doc_type: str
    external_id: str
    title: str | None
    url: str | None
    text: str
    source_time: datetime
    available_at: datetime
    received_at: datetime


def find_document(session: Session, source: Source, external_id: str) -> Document | None:
    stmt = select(Document).where(
        Document.source_id == source.id, Document.external_id == external_id
    )
    return session.exec(stmt).first()


def add_document(
    session: Session, source: Source, spec: DocumentSpec, chunks: Sequence[ChunkSpan]
) -> Document | None:
    """Insert the document and its chunks; None if this exact document is already stored.

    Amendments are separate documents, never revisions: a 10-K/A often carries only Part III, and
    marking it as superseding the 10-K would hide the whole annual report from every later reader.
    Flushes; the caller commits.
    """
    for chunk in chunks:
        if not 0 <= chunk.char_start <= chunk.char_end <= len(spec.text):
            raise ValueError(
                f"{spec.external_id}: chunk {chunk.ordinal} span "
                f"{chunk.char_start}..{chunk.char_end} lies outside the text"
            )
        if spec.text[chunk.char_start : chunk.char_end] != chunk.text:
            raise ValueError(
                f"{spec.external_id}: chunk {chunk.ordinal} text differs from its span"
            )

    digest = content_hash(spec.text)
    existing = find_document(session, source, spec.external_id)
    if existing is not None:
        if existing.content_hash != digest:
            raise DocumentConflict(
                f"{spec.external_id} is stored with hash {existing.content_hash[:12]}, "
                f"the new copy has {digest[:12]}"
            )
        return None

    document = Document(
        source_id=source.id,  # pyright: ignore[reportArgumentType]
        entity_id=spec.entity_id,
        doc_type=spec.doc_type,
        external_id=spec.external_id,
        title=spec.title,
        url=spec.url,
        content_hash=digest,
        source_time=spec.source_time,
        available_at=spec.available_at,
        received_at=spec.received_at,
    )
    session.add(document)
    session.flush()
    session.add_all(
        Chunk(
            document_id=document.id,  # pyright: ignore[reportArgumentType]
            ordinal=chunk.ordinal,
            section_path=chunk.section_path,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
            text=chunk.text,
        )
        for chunk in chunks
    )
    session.flush()
    return document


def text_path(text_root: Path, external_id: str) -> Path:
    return text_root / f"{external_id}.txt"


def audit_documents(session: Session, text_root: Path) -> list[str]:
    """One message per broken document or chunk; empty when every citation can be re-read."""
    problems: list[str] = []
    documents = session.exec(select(Document).order_by(col(Document.id))).all()
    for document in documents:
        name = document.external_id
        if document.entity_id is None:
            problems.append(f"{name}: no entity")
        chunks = session.exec(
            select(Chunk).where(Chunk.document_id == document.id).order_by(col(Chunk.ordinal))
        ).all()
        if not chunks:
            problems.append(f"{name}: no chunks")
        path = text_path(text_root, name)
        if not path.exists():
            problems.append(f"{name}: text file {path} is missing")
            continue
        text = path.read_text(encoding="utf-8")
        if content_hash(text) != document.content_hash:
            problems.append(f"{name}: text file hash differs from the stored content_hash")
        for chunk in chunks:
            if text[chunk.char_start : chunk.char_end] != chunk.text:
                problems.append(
                    f"{name}: chunk {chunk.ordinal} ({chunk.char_start}..{chunk.char_end}) "
                    "no longer matches the text file"
                )
    return problems
