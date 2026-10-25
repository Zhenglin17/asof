"""Engine construction and schema creation for the SQLite metadata store."""

import os
import sqlite3
from pathlib import Path

from sqlalchemy import Connection, Engine, event, inspect
from sqlalchemy.pool import ConnectionPoolEntry
from sqlmodel import SQLModel, create_engine

from asof.store import (
    fts,
    models,  # noqa: F401  (registers every table on SQLModel.metadata)
)

DEFAULT_DATA_DIR = Path("/data/asof")


def default_data_dir() -> Path:
    return Path(os.environ.get("ASOF_DATA_DIR", DEFAULT_DATA_DIR))


def default_db_path() -> Path:
    return default_data_dir() / "meta.db"


def _enable_foreign_keys(dbapi_connection: sqlite3.Connection, _: ConnectionPoolEntry) -> None:
    # SQLite ships with foreign keys OFF, and the setting is per connection, not per file.
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def make_engine(path: Path) -> Engine:
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{path}")
    event.listen(engine, "connect", _enable_foreign_keys)
    return engine


def _upgrade_citation_table(conn: Connection) -> None:
    # create_all never alters an existing table, and ALTER cannot add a CHECK constraint. A
    # citation table built before `quote` existed is rebuilt from the model with its rows; a
    # row the new constraints reject (a chunk citation without a quote) fails the upgrade.
    sql = conn.exec_driver_sql(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'decision_citation'"
    ).scalar_one()
    if "ck_chunk_requires_quote" in sql:
        return
    old = [c["name"] for c in inspect(conn).get_columns("decision_citation")]
    rows = [dict(r._mapping) for r in conn.exec_driver_sql("SELECT * FROM decision_citation")]
    conn.exec_driver_sql("DROP TABLE decision_citation")
    table = SQLModel.metadata.tables["decision_citation"]
    table.create(conn)
    if rows:
        conn.execute(table.insert(), [{name: row[name] for name in old} for row in rows])


def _upgrade_trade_table(conn: Connection) -> None:
    # A trade table from before `symbol` existed gains the nullable column; old rows keep NULL.
    if "symbol" not in {c["name"] for c in inspect(conn).get_columns("trade")}:
        conn.exec_driver_sql("ALTER TABLE trade ADD COLUMN symbol VARCHAR")


def _upgrade_holdout_opening_table(conn: Connection) -> None:
    # A holdout_opening table from before `run_id` existed gains the nullable column.
    if "run_id" not in {c["name"] for c in inspect(conn).get_columns("holdout_opening")}:
        conn.exec_driver_sql(
            "ALTER TABLE holdout_opening ADD COLUMN run_id INTEGER REFERENCES run(id)"
        )


def init_db(engine: Engine) -> None:
    """Create or upgrade the schema and the chunk index. Safe to run repeatedly."""
    with engine.connect() as conn:
        # Readers are not blocked by a long write (a docs sync) in WAL mode. Must run outside
        # a transaction; an in-memory database answers "memory" and stays as it is.
        conn.exec_driver_sql("PRAGMA journal_mode=WAL")
    SQLModel.metadata.create_all(engine)
    with engine.begin() as conn:
        # Python's sqlite3 opens a transaction only before INSERT/UPDATE/DELETE, so DROP and
        # CREATE would each commit on their own and a failed upgrade would lose the old rows.
        # An explicit BEGIN makes the rebuild all or nothing.
        conn.exec_driver_sql("BEGIN")
        _upgrade_citation_table(conn)
        _upgrade_trade_table(conn)
        _upgrade_holdout_opening_table(conn)
        fts.create_chunk_index(conn)
