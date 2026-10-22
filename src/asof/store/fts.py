"""The full-text index over chunk.text (SQLite FTS5).

`chunk_fts` is an external-content table: it stores only the postings and reads the text back
from `chunk`, so the text is not stored a third time. Triggers keep it in step with every
insert, update and delete; an index created over existing chunks is filled when it is created.

The index answers which chunks contain a word and how often, through the `fts5vocab` instance
view. Ranking is done by `queries.visible_chunk_search` from statistics over the chunks visible
at as_of, because FTS5's own bm25() counts every indexed chunk, future ones included.
"""

from sqlalchemy import Connection, TableClause, column, table, text
from sqlalchemy.exc import DBAPIError
from sqlmodel import Session

_CREATE = [
    "CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts "
    "USING fts5(text, content='chunk', content_rowid='id')",
    "CREATE TRIGGER IF NOT EXISTS chunk_fts_insert AFTER INSERT ON chunk BEGIN "
    "INSERT INTO chunk_fts(rowid, text) VALUES (new.id, new.text); END",
    "CREATE TRIGGER IF NOT EXISTS chunk_fts_delete AFTER DELETE ON chunk BEGIN "
    "INSERT INTO chunk_fts(chunk_fts, rowid, text) VALUES ('delete', old.id, old.text); END",
    "CREATE TRIGGER IF NOT EXISTS chunk_fts_update AFTER UPDATE ON chunk BEGIN "
    "INSERT INTO chunk_fts(chunk_fts, rowid, text) VALUES ('delete', old.id, old.text); "
    "INSERT INTO chunk_fts(rowid, text) VALUES (new.id, new.text); END",
]


def create_chunk_index(conn: Connection) -> None:
    """Create the index and its triggers if missing; a new index over existing chunks is filled
    at once, so there is no window in which search silently misses them."""
    existed = conn.exec_driver_sql(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'chunk_fts'"
    ).first()
    for statement in _CREATE:
        conn.exec_driver_sql(statement)
    if existed is None:
        conn.exec_driver_sql("INSERT INTO chunk_fts(chunk_fts) VALUES ('rebuild')")


def ensure_vocab(session: Session) -> None:
    """One row per (word, chunk, position) of the index, as a per-connection TEMP table, so it
    also works on a read-only store and never shows up in the schema."""
    session.connection().exec_driver_sql(
        "CREATE VIRTUAL TABLE IF NOT EXISTS temp.chunk_fts_instance "
        "USING fts5vocab(main, chunk_fts, instance)"
    )


def vocab_table() -> TableClause:
    return table("chunk_fts_instance", column("term"), column("doc"), schema="temp")


def _indexed_rows(session: Session) -> int:
    # count(*) on chunk_fts itself reads the content table and always equals the chunk count;
    # the docsize shadow table has one row per chunk actually indexed.
    return session.connection().execute(text("SELECT count(*) FROM chunk_fts_docsize")).scalar_one()


def rebuild_chunk_index(session: Session) -> int:
    """Re-index every chunk from scratch; returns the number of indexed chunks."""
    session.connection().execute(text("INSERT INTO chunk_fts(chunk_fts) VALUES ('rebuild')"))
    return _indexed_rows(session)


def chunk_index_mismatch(session: Session) -> int | None:
    """Chunks minus indexed chunks, or None when they agree. Positive means chunks are missing."""
    chunks = session.connection().execute(text("SELECT count(*) FROM chunk")).scalar_one()
    difference = chunks - _indexed_rows(session)
    return difference or None


def chunk_index_integrity(session: Session) -> str | None:
    """None when FTS5's integrity check, which compares the index with the chunk text, passes;
    otherwise its error. Slower than `chunk_index_mismatch`, which only compares row counts."""
    try:
        session.connection().execute(
            text("INSERT INTO chunk_fts(chunk_fts, rank) VALUES ('integrity-check', 1)")
        )
    except DBAPIError as error:
        # A locked or read-only store is not a broken index; only corruption is reported.
        message = str(error.orig)
        if "malformed" in message or "corrupt" in message:
            return message
        raise
    return None
