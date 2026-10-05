"""Command-line entry point. Subcommands are added as each pipeline stage lands."""

from pathlib import Path
from typing import Annotated

import typer

from asof import __version__
from asof.store.db import default_db_path, init_db, make_engine

app = typer.Typer(
    name="asof",
    help="Point-in-time correct market research agent.",
    no_args_is_help=True,
    add_completion=False,
)
db_app = typer.Typer(help="Metadata database commands.", no_args_is_help=True)
app.add_typer(db_app, name="db")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"asof {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    """asof command-line interface."""


@db_app.command("init")
def db_init(
    path: Annotated[
        Path | None,
        typer.Option(
            "--path",
            help="SQLite file to create. Defaults to $ASOF_DATA_DIR/meta.db (/data/asof/meta.db).",
        ),
    ] = None,
) -> None:
    """Create the metadata database and every table. Safe to run twice."""
    db_path = path or default_db_path()
    init_db(make_engine(db_path))
    typer.echo(f"Initialized {db_path}")
