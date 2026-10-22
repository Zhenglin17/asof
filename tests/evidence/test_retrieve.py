"""Guards against the evidence search bypassing the as_of read path: it must return exactly what
`visible_chunk_search` returns, including hiding a future best match from a limit of one.
"""

from datetime import timedelta

from sqlmodel import Session

from asof.evidence import retrieve
from asof.store.models import Entity, Source
from asof.store.queries import visible_chunk_search
from tests.store.factories import T0, new_chunk, new_document, new_entity, pk

DAY = timedelta(days=1)


def test_search_returns_what_the_read_path_returns(
    session: Session, source: Source, apple: Entity
) -> None:
    microsoft = new_entity(session, 789019, "MSFT", name="Microsoft Corporation")
    for entity, doc_type, text in [
        (apple, "10-Q", "Services revenue reached a record."),
        (apple, "8-K/A", "Services revenue guidance was raised."),
        (microsoft, "10-K", "Services revenue grew on cloud."),
    ]:
        doc = new_document(
            session,
            source_id=pk(source),
            entity_id=pk(entity),
            available_at=T0 - DAY,
            doc_type=doc_type,
        )
        new_chunk(session, document_id=pk(doc), text=text)

    for kwargs in [
        {},
        {"entity_id": pk(apple)},
        {"base_forms": ["8-K"]},
        {"entity_id": pk(apple), "limit": 1},
    ]:
        expected = visible_chunk_search(session, T0, "services revenue", **kwargs)
        assert expected
        assert retrieve.search(session, T0, "services revenue", **kwargs) == expected


def test_search_hides_a_future_best_match(session: Session, source: Source, apple: Entity) -> None:
    past = new_document(session, source_id=pk(source), entity_id=pk(apple), available_at=T0 - DAY)
    weak = new_chunk(session, document_id=pk(past), text="Gross margin was stable.")
    future = new_document(session, source_id=pk(source), entity_id=pk(apple), available_at=T0 + DAY)
    new_chunk(session, document_id=pk(future), text="Gross margin expanded on data center demand.")

    hits = retrieve.search(
        session, T0, "gross margin expanded data center demand", entity_id=pk(apple), limit=1
    )
    assert [h.chunk_id for h in hits] == [weak.id]
