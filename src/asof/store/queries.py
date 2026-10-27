"""The only read path into the metadata store.

Every function takes `as_of` and returns only what was public at that moment. Other modules
must not write their own SELECTs against these tables; keeping the time filter in one file
is what makes look-ahead leakage reviewable.

Source and Entity are identity lookups, not market information, so they have no visible_*.
"""

import math
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import ColumnElement, and_, func
from sqlalchemy import select as sa_select
from sqlalchemy.orm import aliased
from sqlmodel import Session, col, select

from asof.store import fts
from asof.store.models import (
    Attribution,
    Chunk,
    Decision,
    Document,
    EquityDaily,
    Event,
    Run,
    StrategyVersion,
    Trade,
)


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
    since: datetime | None = None,
) -> list[Document]:
    """Documents public at ``as_of``; ``since`` keeps only those that became public at or after
    it (a lower bound, for readers that only want what is new)."""
    _require_aware(as_of)
    stmt = select(Document).where(_document_is_visible(as_of))
    if since is not None:
        _require_aware(since)
        stmt = stmt.where(col(Document.available_at) >= since)
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


# BM25 constants (the usual defaults).
_K1 = 1.2
_B = 0.75
# Query words: runs of letters and digits, case-folded, accents removed, as FTS5's unicode61
# tokenizer splits text. Only the query is split in Python; how often a word occurs in a chunk
# is read from the index itself, so a spelling the two disagree on can only miss, never
# mis-score. "deﬁned" (one ligature character) is a different word to the index than "defined".
_WORD = re.compile(r"[^\W_]+")
# Transition reports (10-KT) and amendments (10-K/A) are the same kind of report as 10-K.
_TRANSITION = {"10-KT": "10-K", "10-QT": "10-Q"}


def base_form(doc_type: str) -> str:
    form = doc_type.removesuffix("/A")
    return _TRANSITION.get(form, form)


def _words(s: str) -> list[str]:
    decomposed = unicodedata.normalize("NFD", s.lower())
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return _WORD.findall(unicodedata.normalize("NFC", stripped))


@dataclass(frozen=True)
class ChunkHit:
    chunk_id: int
    document_id: int
    entity_id: int | None
    doc_type: str
    available_at: datetime
    section_path: str
    # BM25, higher is better. Comparable only between hits of the same query and as_of.
    score: float
    text: str


def _term_counts(
    session: Session,
    as_of: datetime,
    term: str,
    entity_id: int | None,
    doc_types: list[str] | None,
) -> dict[int, tuple[int, int]]:
    """chunk id -> (occurrences of term, length) over the visible chunks that pass the filters.

    Occurrences come from the index's own token list, the same source that decides which
    chunks contain the term, so a chunk is never fetched for a word it does not score on.
    """
    vocab = fts.vocab_table()
    stmt = (
        sa_select(vocab.c.doc, func.count(), col(Chunk.char_end) - col(Chunk.char_start))
        .select_from(vocab)
        .join(Chunk, vocab.c.doc == col(Chunk.id))
        .join(Document, col(Chunk.document_id) == col(Document.id))
        .where(vocab.c.term == term, _document_is_visible(as_of))
        .group_by(vocab.c.doc)
    )
    if entity_id is not None:
        stmt = stmt.where(col(Document.entity_id) == entity_id)
    if doc_types is not None:
        stmt = stmt.where(col(Document.doc_type).in_(doc_types))
    return {doc: (tf, length) for doc, tf, length in session.execute(stmt).all()}


def _visible_chunks_with(session: Session, as_of: datetime, term: str) -> int:
    vocab = fts.vocab_table()
    stmt = (
        sa_select(func.count(func.distinct(vocab.c.doc)))
        .select_from(vocab)
        .join(Chunk, vocab.c.doc == col(Chunk.id))
        .join(Document, col(Chunk.document_id) == col(Document.id))
        .where(vocab.c.term == term, _document_is_visible(as_of))
    )
    return session.execute(stmt).scalar_one()


def visible_chunk_search(
    session: Session,
    as_of: datetime,
    query: str,
    *,
    entity_id: int | None = None,
    base_forms: Iterable[str] | None = None,
    limit: int = 10,
) -> list[ChunkHit]:
    """Chunks visible at as_of that contain any query word, ranked by BM25.

    Ranking statistics come from the chunks visible at as_of only: the number of chunks N,
    how many contain each word, and the average length. FTS5's own bm25() counts every indexed
    chunk, so a word that only becomes common later would already lose weight, and the same
    as_of would rank differently as the store grows. Length is in characters.

    Exact top-k with pruning (MaxScore): one word adds less than idf * (k1 + 1) to any chunk.
    Words are taken rarest first and the chunks containing them scored in full; once the bounds
    of the words not taken yet sum to less than the current k-th score, a chunk containing only
    those words cannot reach the result. Raises RuntimeError when the index is incomplete,
    which would otherwise silently drop chunks and skew the statistics.
    """
    _require_aware(as_of)
    terms = sorted(set(_words(query)))
    if not terms or limit <= 0:
        return []
    if fts.chunk_index_mismatch(session) is not None:
        raise RuntimeError("the chunk index is incomplete; run `asof evidence index`")
    fts.ensure_vocab(session)

    n_chunks, avg_length = session.exec(
        select(func.count(), func.avg(col(Chunk.char_end) - col(Chunk.char_start)))
        .select_from(Chunk)
        .join(Document, col(Chunk.document_id) == col(Document.id))
        .where(_document_is_visible(as_of))
    ).one()
    if not n_chunks:
        return []
    doc_types = None
    if base_forms is not None:
        wanted = set(base_forms)
        types = session.exec(select(Document.doc_type).distinct()).all()
        doc_types = [t for t in types if base_form(t) in wanted]
    counts = {t: _term_counts(session, as_of, t, entity_id, doc_types) for t in terms}
    unfiltered = entity_id is None and doc_types is None
    idf = {}
    for t in terms:
        n = len(counts[t]) if unfiltered else _visible_chunks_with(session, as_of, t)
        idf[t] = math.log(1 + (n_chunks - n + 0.5) / (n + 0.5))

    def score(chunk_id: int, length: int) -> float:
        norm = _K1 * (1 - _B + _B * length / avg_length)
        total = 0.0
        for t in terms:
            found = counts[t].get(chunk_id)
            if found is not None:
                tf = found[0]
                total += idf[t] * tf * (_K1 + 1) / (tf + norm)
        return total

    scores: dict[int, float] = {}
    by_rarity = sorted(terms, key=lambda t: (-idf[t], t))
    for taken, term in enumerate(by_rarity):
        best = sorted(scores.values(), reverse=True)
        threshold = best[limit - 1] if len(best) >= limit else 0.0
        bound = sum(idf[t] * (_K1 + 1) for t in by_rarity[taken:])
        if bound < threshold:
            break
        for chunk_id, (_, length) in counts[term].items():
            if chunk_id not in scores:
                scores[chunk_id] = score(chunk_id, length)
    top = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
    if not top:
        return []

    rows = session.exec(
        select(Chunk, Document)
        .join(Document, col(Chunk.document_id) == col(Document.id))
        .where(col(Chunk.id).in_([chunk_id for chunk_id, _ in top]))
    ).all()
    found = {_pk(chunk.id): (chunk, document) for chunk, document in rows}
    hits = []
    for chunk_id, value in top:
        chunk, document = found[chunk_id]
        hits.append(
            ChunkHit(
                chunk_id=chunk_id,
                document_id=chunk.document_id,
                entity_id=document.entity_id,
                doc_type=document.doc_type,
                available_at=document.available_at,
                section_path=chunk.section_path,
                score=value,
                text=chunk.text,
            )
        )
    return hits


def _pk(value: int | None) -> int:
    assert value is not None
    return value


def visible_chunk(
    session: Session, as_of: datetime, chunk_id: int
) -> tuple[Chunk, Document] | None:
    """The chunk and its document if both were public at as_of, else None."""
    _require_aware(as_of)
    stmt = (
        select(Chunk, Document)
        .join(Document, col(Chunk.document_id) == col(Document.id))
        .where(col(Chunk.id) == chunk_id, _document_is_visible(as_of))
    )
    row = session.exec(stmt).first()
    return None if row is None else (row[0], row[1])


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


def visible_trades(
    session: Session, as_of: datetime, *, strategy_id: str | None = None
) -> list[Trade]:
    """Closed legs whose exit bar was final at as_of."""
    _require_aware(as_of)
    stmt = select(Trade).where(col(Trade.available_at) <= as_of)
    if strategy_id is not None:
        stmt = stmt.where(col(Trade.strategy_id) == strategy_id)
    stmt = stmt.order_by(col(Trade.available_at), col(Trade.id))
    return list(session.exec(stmt).all())


def visible_equity(session: Session, as_of: datetime, *, run_id: int) -> list[EquityDaily]:
    """A run's daily marks whose session had closed by as_of."""
    _require_aware(as_of)
    stmt = (
        select(EquityDaily)
        .where(col(EquityDaily.run_id) == run_id, col(EquityDaily.available_at) <= as_of)
        .order_by(col(EquityDaily.session_date))
    )
    return list(session.exec(stmt).all())


def runs_of_strategy(session: Session, strategy_id: str | None) -> list[Run]:
    """A strategy's runs: the research record of what was tried, like Source and Entity an
    identity lookup rather than information that becomes public, so it takes no as_of."""
    stmt = select(Run).where(col(Run.strategy_id) == strategy_id).order_by(col(Run.id))
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
