"""Scan tests run on the store tests' fresh SQLite file plus an in-memory DuckDB connection."""

from collections.abc import Iterator

import duckdb
import pytest

from tests.store.conftest import engine, session

__all__ = ["con", "engine", "session"]


@pytest.fixture
def con() -> Iterator[duckdb.DuckDBPyConnection]:
    connection = duckdb.connect()
    yield connection
    connection.close()
