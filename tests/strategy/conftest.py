"""Strategy tests run on the same fresh-store fixtures as the store tests."""

from tests.store.conftest import engine, session

__all__ = ["engine", "session"]
