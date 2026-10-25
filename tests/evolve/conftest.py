"""Evolve tests run on the store tests' fresh SQLite file plus the scan tests' in-memory DuckDB."""

from tests.scan.conftest import con
from tests.store.conftest import engine, session

__all__ = ["con", "engine", "session"]
