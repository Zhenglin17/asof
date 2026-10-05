"""Engine construction and schema creation for the SQLite metadata store."""

import os
import sqlite3
from pathlib import Path

from sqlalchemy import Engine, event
from sqlalchemy.pool import ConnectionPoolEntry
from sqlmodel import SQLModel, create_engine

from asof.store import models  # noqa: F401  (registers every table on SQLModel.metadata)

DEFAULT_DATA_DIR = Path("/data/asof")


def default_db_path() -> Path:
    return Path(os.environ.get("ASOF_DATA_DIR", DEFAULT_DATA_DIR)) / "meta.db"


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


def init_db(engine: Engine) -> None:
    """Create all tables that do not exist yet. Safe to run repeatedly."""
    SQLModel.metadata.create_all(engine)
