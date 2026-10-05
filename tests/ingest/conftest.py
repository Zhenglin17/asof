"""Fixtures for ingest tests: saved SEC ticker tables and a watchlist file writer."""

from collections.abc import Callable
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def sec_dir() -> Path:
    return FIXTURES / "sec"


@pytest.fixture
def shuffled_sec_dir() -> Path:
    return FIXTURES / "sec_shuffled"


@pytest.fixture
def write_watchlist(tmp_path: Path) -> Callable[[str], Path]:
    def write(text: str) -> Path:
        path = tmp_path / "watchlist.yaml"
        path.write_text(text)
        return path

    return write
