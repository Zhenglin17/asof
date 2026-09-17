"""Command-line entry point. Subcommands are added as each pipeline stage lands."""

import typer

from asof import __version__

app = typer.Typer(
    name="asof",
    help="Point-in-time correct market research agent.",
    no_args_is_help=True,
    add_completion=False,
)


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
