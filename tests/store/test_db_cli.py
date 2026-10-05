"""Guards against a broken setup command: `asof db init` must build every table, repeatably."""

from pathlib import Path

from sqlalchemy import create_engine, inspect
from typer.testing import CliRunner

from asof.cli import app
from tests.store.test_schema import EXPECTED_TABLES

runner = CliRunner()


def test_db_init_creates_all_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "meta.db"

    result = runner.invoke(app, ["db", "init", "--path", str(db_path)])

    assert result.exit_code == 0, result.output
    assert db_path.exists()
    assert set(inspect(create_engine(f"sqlite:///{db_path}")).get_table_names()) == EXPECTED_TABLES


def test_db_init_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "meta.db"
    for _ in range(2):
        result = runner.invoke(app, ["db", "init", "--path", str(db_path)])
        assert result.exit_code == 0, result.output
