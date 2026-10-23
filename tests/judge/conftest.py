"""Judge tests run on the same fresh-store fixtures as the store tests, plus the scan tests'
in-memory DuckDB connection for the tests that scan a market store."""

from tests.scan.conftest import con
from tests.store.conftest import engine, session, source

__all__ = ["con", "engine", "session", "source"]
