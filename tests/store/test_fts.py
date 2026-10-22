"""Guards against a stale or missing full-text index and a schema migration that loses rows.

The index is an FTS5 table over chunk.text kept in sync by triggers; `init_db` fills it for chunks
written before it existed, search refuses to run on an index that lost rows, and the integrity
check compares the index with the chunk text. `init_db` must also upgrade a
store created by an earlier schema (no `decision_citation.quote`, or `quote` added by ALTER
without its CHECK) by rebuilding the table with every row, or fail and change nothing.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine, inspect, text
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, SQLModel, col, create_engine, select

from asof.store import fts
from asof.store.db import init_db, make_engine
from asof.store.models import Chunk, DecisionCitation, Entity, Source
from asof.store.queries import visible_chunk_search
from tests.store.factories import (
    T0,
    add,
    new_chunk,
    new_document,
    new_entity,
    new_run,
    new_source,
    new_trade_decision,
    pk,
)

# The decision_citation table exactly as the previous schema created it: no quote column.
OLD_CITATION_DDL = """
CREATE TABLE decision_citation (
    id INTEGER NOT NULL PRIMARY KEY,
    decision_id INTEGER NOT NULL REFERENCES decision (id),
    claim VARCHAR NOT NULL,
    evidence_kind VARCHAR NOT NULL,
    evidence_ref VARCHAR,
    chunk_id INTEGER REFERENCES chunk (id),
    CONSTRAINT ck_evidence_kind_enum CHECK (
        "evidence_kind" IN ('chunk', 'fact', 'bar', 'option_snapshot', 'macro_obs')
    )
)
"""


def fts_rowids(session: Session, match: str) -> list[int]:
    rows = session.connection().execute(
        text("SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH :q ORDER BY rowid"), {"q": match}
    )
    return [row[0] for row in rows]


@pytest.fixture
def bare_engine(tmp_path: Path) -> Iterator[Engine]:
    """Tables only, as a store built before the index existed; init_db is not run."""
    eng = make_engine(tmp_path / "old.db")
    SQLModel.metadata.create_all(eng)
    yield eng
    eng.dispose()


def test_init_db_creates_the_chunk_index_as_fts5_over_chunk_text(engine: Engine) -> None:
    with engine.connect() as conn:
        sql = conn.execute(
            text("SELECT sql FROM sqlite_master WHERE name = 'chunk_fts'")
        ).scalar_one()
    assert "fts5" in sql.lower()
    assert "content" in sql.lower() and "chunk" in sql.lower()


def test_new_chunk_is_indexed_by_trigger(session: Session, source: Source, apple: Entity) -> None:
    doc = new_document(session, source_id=pk(source), entity_id=pk(apple), available_at=T0)
    chunk = new_chunk(session, document_id=pk(doc), text="Wearables revenue declined sharply.")
    assert fts_rowids(session, "wearables") == [chunk.id]
    assert fts.chunk_index_mismatch(session) is None


def test_updated_and_deleted_chunks_leave_the_index(
    session: Session, source: Source, apple: Entity
) -> None:
    # Chunks are append-only in practice; the triggers still must not leave stale postings.
    doc = new_document(session, source_id=pk(source), entity_id=pk(apple), available_at=T0)
    kept = new_chunk(session, document_id=pk(doc), ordinal=0, text="Mac revenue rose.")
    dropped = new_chunk(session, document_id=pk(doc), ordinal=1, text="iPad revenue fell.")

    kept.text = "Services revenue rose."
    add(session, kept)
    session.delete(dropped)
    session.commit()

    assert fts_rowids(session, "mac") == []
    assert fts_rowids(session, "services") == [kept.id]
    assert fts_rowids(session, "ipad") == []
    assert fts.chunk_index_mismatch(session) is None


def test_init_db_twice_is_harmless(engine: Engine) -> None:
    init_db(engine)
    init_db(engine)
    with Session(engine) as s:
        assert fts.chunk_index_mismatch(s) is None


def test_file_store_uses_write_ahead_logging(engine: Engine) -> None:
    with engine.connect() as conn:
        assert conn.execute(text("PRAGMA journal_mode")).scalar_one() == "wal"


def test_in_memory_store_still_initialises() -> None:
    # WAL does not apply to :memory:; init_db must not fail there.
    eng = create_engine("sqlite://")
    init_db(eng)
    with eng.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM chunk_fts")).scalar_one() == 0
    eng.dispose()


def test_init_db_indexes_chunks_written_before_the_index(bare_engine: Engine) -> None:
    with Session(bare_engine) as s:
        source = new_source(s)
        apple = new_entity(s)
        doc = new_document(s, source_id=pk(source), entity_id=pk(apple), available_at=T0)
        first = new_chunk(s, document_id=pk(doc), ordinal=0, text="Greater China sales fell.")
        second = new_chunk(s, document_id=pk(doc), ordinal=1, text="Europe sales rose.")
        # Read the ids while the session is open; the objects expire when it closes.
        expected = sorted([pk(first), pk(second)])

    # Creating the index over a store that already has chunks fills it in the same call.
    init_db(bare_engine)
    with Session(bare_engine) as s:
        assert fts.chunk_index_mismatch(s) is None
        assert fts.chunk_index_integrity(s) is None
        hits = visible_chunk_search(s, T0, "sales")
        assert sorted(h.chunk_id for h in hits) == expected

        # Rebuilding stays available and changes nothing.
        assert fts.rebuild_chunk_index(s) == 2
        s.commit()
        assert fts.rebuild_chunk_index(s) == 2
        s.commit()
        assert fts.chunk_index_mismatch(s) is None
        assert sorted(h.chunk_id for h in visible_chunk_search(s, T0, "sales")) == expected


def drop_from_index(session: Session, chunk: Chunk) -> None:
    """Remove one chunk from the index only, as a crash between writes could leave it."""
    session.connection().execute(
        text("INSERT INTO chunk_fts(chunk_fts, rowid, text) VALUES('delete', :id, :text)"),
        {"id": chunk.id, "text": chunk.text},
    )


def test_search_refuses_an_incomplete_index(
    session: Session, source: Source, apple: Entity
) -> None:
    doc = new_document(session, source_id=pk(source), entity_id=pk(apple), available_at=T0)
    kept = new_chunk(session, document_id=pk(doc), ordinal=0, text="Europe sales rose.")
    lost = new_chunk(session, document_id=pk(doc), ordinal=1, text="Greater China sales fell.")
    assert [h.chunk_id for h in visible_chunk_search(session, T0, "europe")] == [kept.id]

    drop_from_index(session, lost)

    assert fts.chunk_index_mismatch(session) is not None
    # A silently missing chunk would be a wrong answer, not an empty one: refuse instead.
    with pytest.raises(RuntimeError, match="index"):
        visible_chunk_search(session, T0, "europe")

    assert fts.rebuild_chunk_index(session) == 2
    assert fts.chunk_index_mismatch(session) is None
    assert [h.chunk_id for h in visible_chunk_search(session, T0, "china")] == [lost.id]


def test_integrity_check_passes_on_a_clean_index(
    session: Session, source: Source, apple: Entity
) -> None:
    assert fts.chunk_index_integrity(session) is None
    doc = new_document(session, source_id=pk(source), entity_id=pk(apple), available_at=T0)
    new_chunk(session, document_id=pk(doc), ordinal=0, text="Europe sales rose.")
    new_chunk(session, document_id=pk(doc), ordinal=1, text="Greater China sales fell.")
    assert fts.chunk_index_integrity(session) is None


def test_integrity_check_reports_an_index_that_differs_from_the_chunks(
    session: Session, source: Source, apple: Entity
) -> None:
    doc = new_document(session, source_id=pk(source), entity_id=pk(apple), available_at=T0)
    new_chunk(session, document_id=pk(doc), ordinal=0, text="Europe sales rose.")
    lost = new_chunk(session, document_id=pk(doc), ordinal=1, text="Greater China sales fell.")

    drop_from_index(session, lost)

    problem = fts.chunk_index_integrity(session)
    assert isinstance(problem, str) and problem
    fts.rebuild_chunk_index(session)
    assert fts.chunk_index_integrity(session) is None


def test_vocabulary_table_stays_out_of_the_main_schema(
    engine: Engine, session: Session, source: Source, apple: Entity
) -> None:
    doc = new_document(session, source_id=pk(source), entity_id=pk(apple), available_at=T0)
    new_chunk(session, document_id=pk(doc), text="Europe sales rose.")
    assert visible_chunk_search(session, T0, "europe sales")
    session.commit()

    # The fts5vocab table used for term statistics is TEMP, per connection.
    with engine.connect() as conn:
        main = conn.execute(text("SELECT name, sql FROM main.sqlite_master")).all()
    assert not [name for name, sql in main if sql and "fts5vocab" in sql.lower()]
    assert not [name for name in inspect(engine).get_table_names() if "vocab" in name]


def test_rebuild_on_an_empty_store_returns_zero(session: Session) -> None:
    assert fts.rebuild_chunk_index(session) == 0
    assert fts.chunk_index_mismatch(session) is None


def test_rebuild_is_repeatable(session: Session, source: Source, apple: Entity) -> None:
    doc = new_document(session, source_id=pk(source), entity_id=pk(apple), available_at=T0)
    new_chunk(session, document_id=pk(doc))
    assert fts.rebuild_chunk_index(session) == 1
    assert fts.rebuild_chunk_index(session) == 1
    assert len(fts_rowids(session, "sales")) == 1


def old_store(path: Path, citation_ddl: str) -> tuple[Engine, int, int]:
    """A store whose decision_citation was built by an earlier schema, with one real decision.

    Every other table comes from the current model; returns the engine, a decision id and a
    chunk id, so old rows can be inserted with foreign keys on, as they were in the real store.
    """
    eng = make_engine(path)
    SQLModel.metadata.create_all(eng)
    with eng.begin() as conn:
        conn.exec_driver_sql("DROP TABLE decision_citation")
        conn.exec_driver_sql(citation_ddl)
    with Session(eng) as s:
        apple = new_entity(s)
        doc = new_document(s, source_id=pk(new_source(s)), entity_id=pk(apple), available_at=T0)
        chunk = new_chunk(s, document_id=pk(doc))
        decision = new_trade_decision(s, run_id=pk(new_run(s)), entity_id=pk(apple))
        return eng, pk(decision), pk(chunk)


def insert_old_citation(eng: Engine, columns: str, values: str) -> None:
    with eng.begin() as conn:
        conn.exec_driver_sql(f"INSERT INTO decision_citation ({columns}) VALUES ({values})")


def citation_sql(eng: Engine) -> str:
    with eng.connect() as conn:
        return conn.execute(
            text("SELECT sql FROM sqlite_master WHERE name = 'decision_citation'")
        ).scalar_one()


def citation_rows(eng: Engine) -> list[tuple[object, ...]]:
    with eng.connect() as conn:
        rows = conn.exec_driver_sql(
            "SELECT id, decision_id, claim, evidence_kind, evidence_ref, chunk_id "
            "FROM decision_citation ORDER BY id"
        )
        return [tuple(r) for r in rows]


def test_init_db_adds_quote_to_an_old_citation_table(tmp_path: Path) -> None:
    eng, decision_id, _ = old_store(tmp_path / "old.db", OLD_CITATION_DDL)
    insert_old_citation(
        eng,
        "decision_id, claim, evidence_kind, evidence_ref",
        f"{decision_id}, 'Closed higher.', 'bar', 'AAPL/2024-07-12'",
    )

    init_db(eng)
    init_db(eng)

    columns = {c["name"] for c in inspect(eng).get_columns("decision_citation")}
    assert "quote" in columns
    assert "ck_chunk_requires_quote" in citation_sql(eng)
    with Session(eng) as s:
        (old,) = s.exec(select(DecisionCitation)).all()
        assert old.claim == "Closed higher."
        assert old.decision_id == decision_id
        assert old.quote is None
    eng.dispose()


def test_quote_column_without_its_check_is_rebuilt(tmp_path: Path) -> None:
    # The real store's shape after an ALTER TABLE ADD COLUMN: quote exists, the CHECK does not.
    altered = OLD_CITATION_DDL.replace(
        "chunk_id INTEGER REFERENCES chunk (id),",
        "chunk_id INTEGER REFERENCES chunk (id),\n    quote VARCHAR,",
    )
    assert "quote VARCHAR" in altered
    eng, decision_id, chunk_id = old_store(tmp_path / "altered.db", altered)
    insert_old_citation(
        eng,
        "decision_id, claim, evidence_kind, chunk_id, quote",
        f"{decision_id}, 'Net sales grew 5%.', 'chunk', {chunk_id}, "
        "'Total net sales increased 5% year over year.'",
    )
    insert_old_citation(
        eng,
        "decision_id, claim, evidence_kind, evidence_ref",
        f"{decision_id}, 'Closed higher.', 'bar', 'AAPL/2024-07-12'",
    )
    before = citation_rows(eng)

    init_db(eng)

    assert "ck_chunk_requires_quote" in citation_sql(eng)
    assert citation_rows(eng) == before
    with Session(eng) as s:
        quotes = [
            c.quote for c in s.exec(select(DecisionCitation).order_by(col(DecisionCitation.id)))
        ]
        assert quotes == ["Total net sales increased 5% year over year.", None]
        with pytest.raises(IntegrityError):
            add(
                s,
                DecisionCitation(
                    decision_id=decision_id,
                    claim="Net sales grew 5%.",
                    evidence_kind="chunk",
                    chunk_id=chunk_id,
                ),
            )
    eng.dispose()


def test_old_chunk_citation_without_quote_fails_the_upgrade_and_changes_nothing(
    tmp_path: Path,
) -> None:
    eng, decision_id, chunk_id = old_store(tmp_path / "old.db", OLD_CITATION_DDL)
    insert_old_citation(
        eng,
        "decision_id, claim, evidence_kind, chunk_id",
        f"{decision_id}, 'Net sales grew 5%.', 'chunk', {chunk_id}",
    )
    before_sql = citation_sql(eng)
    before_rows = citation_rows(eng)

    with pytest.raises(IntegrityError):
        init_db(eng)

    # Rolled back: the old table, without quote, and its row are exactly as they were.
    assert citation_sql(eng) == before_sql
    assert "quote" not in {c["name"] for c in inspect(eng).get_columns("decision_citation")}
    assert citation_rows(eng) == before_rows
    eng.dispose()


def test_chunk_citation_without_quote_is_rejected(
    session: Session, source: Source, apple: Entity
) -> None:
    doc = new_document(session, source_id=pk(source), entity_id=pk(apple), available_at=T0)
    chunk = new_chunk(session, document_id=pk(doc))
    decision = new_trade_decision(session, run_id=pk(new_run(session)), entity_id=pk(apple))
    with pytest.raises(IntegrityError):
        add(
            session,
            DecisionCitation(
                decision_id=pk(decision),
                claim="Net sales grew 5%.",
                evidence_kind="chunk",
                chunk_id=pk(chunk),
            ),
        )


def test_chunk_citation_with_quote_and_bar_citation_without_quote_are_accepted(
    session: Session, source: Source, apple: Entity
) -> None:
    doc = new_document(session, source_id=pk(source), entity_id=pk(apple), available_at=T0)
    chunk: Chunk = new_chunk(session, document_id=pk(doc))
    decision = new_trade_decision(session, run_id=pk(new_run(session)), entity_id=pk(apple))
    cited = add(
        session,
        DecisionCitation(
            decision_id=pk(decision),
            claim="Net sales grew 5%.",
            evidence_kind="chunk",
            chunk_id=pk(chunk),
            quote="Total net sales increased 5% year over year.",
        ),
    )
    bar = add(
        session,
        DecisionCitation(
            decision_id=pk(decision),
            claim="The stock closed higher.",
            evidence_kind="bar",
            evidence_ref="AAPL/2024-07-12",
        ),
    )
    assert cited.quote == "Total net sales increased 5% year over year."
    assert bar.quote is None
