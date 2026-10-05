"""Fixtures for store tests: a fresh SQLite file per test, plus common seed rows."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine
from sqlmodel import Session

from asof.store.db import init_db, make_engine
from asof.store.models import Entity, Source
from tests.store.factories import new_entity, new_source


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    eng = make_engine(tmp_path / "meta.db")
    init_db(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    with Session(engine) as s:
        yield s


@pytest.fixture
def source(session: Session) -> Source:
    return new_source(session)


@pytest.fixture
def apple(session: Session) -> Entity:
    return new_entity(session)
