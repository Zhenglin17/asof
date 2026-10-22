"""Evidence retrieval for the judge: what was public at as_of and matches a question.

All filtering and ranking live in `store.queries.visible_chunk_search`, so the time rule stays
in the one file that every read goes through.
"""

from collections.abc import Iterable
from datetime import datetime

from sqlmodel import Session

from asof.store.queries import ChunkHit, visible_chunk_search


def search(
    session: Session,
    as_of: datetime,
    query: str,
    *,
    entity_id: int | None = None,
    base_forms: Iterable[str] | None = None,
    limit: int = 10,
) -> list[ChunkHit]:
    return visible_chunk_search(
        session, as_of, query, entity_id=entity_id, base_forms=base_forms, limit=limit
    )
