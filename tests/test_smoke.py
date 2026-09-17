"""Smoke tests: the package imports and the CLI entry point responds."""

from typer.testing import CliRunner

import asof
from asof.cli import app

runner = CliRunner()


def test_package_has_version() -> None:
    assert isinstance(asof.__version__, str)
    assert asof.__version__


def test_cli_help_exits_zero() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "asof" in result.output


def test_cli_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert asof.__version__ in result.output
