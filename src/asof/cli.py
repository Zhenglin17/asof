"""Command-line entry point. Subcommands are added as each pipeline stage lands."""

import os
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from sqlmodel import Session

from asof import __version__
from asof.ingest.sec_tickers import download_sec_listings, load_sec_listings
from asof.ingest.watchlist import load_watchlist, resolve_watchlist
from asof.store.db import default_data_dir, default_db_path, init_db, make_engine
from asof.store.entities import EntityConflict, upsert_entities
from asof.store.models import EntityKind

SEC_USER_AGENT_ENV = "ASOF_SEC_USER_AGENT"

app = typer.Typer(
    name="asof",
    help="Point-in-time correct market research agent.",
    no_args_is_help=True,
    add_completion=False,
)
db_app = typer.Typer(help="Metadata database commands.", no_args_is_help=True)
app.add_typer(db_app, name="db")
entities_app = typer.Typer(help="Tracked instruments.", no_args_is_help=True)
app.add_typer(entities_app, name="entities")


def _fail(message: str) -> NoReturn:
    typer.echo(message)
    raise typer.Exit(1)


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


@entities_app.command("sync")
def entities_sync(
    watchlist: Annotated[Path, typer.Option("--watchlist", help="Watchlist file to load.")] = Path(
        "configs/watchlist.yaml"
    ),
    sec_dir: Annotated[
        Path | None,
        typer.Option(
            "--sec-dir",
            help="Directory holding the SEC ticker tables. Defaults to $ASOF_DATA_DIR/raw/sec.",
        ),
    ] = None,
    db: Annotated[
        Path | None,
        typer.Option("--db", help="SQLite file to write. Defaults to $ASOF_DATA_DIR/meta.db."),
    ] = None,
    download: Annotated[
        bool,
        typer.Option(
            "--download", help=f"Fetch fresh SEC ticker tables first. Needs ${SEC_USER_AGENT_ENV}."
        ),
    ] = False,
) -> None:
    """Write every watchlist instrument into the entity table, or none of them."""
    tables_dir = sec_dir or default_data_dir() / "raw" / "sec"
    if not watchlist.is_file():
        _fail(f"Watchlist file not found: {watchlist}")
    if download:
        user_agent = os.environ.get(SEC_USER_AGENT_ENV)
        if not user_agent:
            _fail(f"Set {SEC_USER_AGENT_ENV} to a contact address; the SEC requires one.")
        try:
            download_sec_listings(tables_dir, user_agent)
        except (ValueError, OSError) as error:
            _fail(f"Download failed, saved tables left as they were: {error}")

    try:
        listings = load_sec_listings(tables_dir)
    except FileNotFoundError:
        _fail(f"SEC ticker tables not found in {tables_dir}. Run again with --download.")
    except (ValueError, KeyError) as error:
        _fail(f"SEC ticker tables in {tables_dir} are unreadable: {error!r}")
    try:
        resolved = resolve_watchlist(load_watchlist(watchlist), listings)
    except ValueError as error:
        _fail(str(error))

    if resolved.missing_companies or resolved.conflicts:
        if resolved.missing_companies:
            typer.echo(f"Companies the SEC does not list: {', '.join(resolved.missing_companies)}")
        if resolved.conflicts:
            typer.echo(
                "Listed for more than one registrant, add `cik:` to the watchlist entry: "
                + ", ".join(resolved.conflicts)
            )
        _fail("Nothing was written.")

    engine = make_engine(db or default_db_path())
    try:
        init_db(engine)
        with Session(engine) as session:
            result = upsert_entities(session, resolved.entities)
            session.commit()
    except EntityConflict as error:
        _fail(f"{error}. Nothing was written.")
    finally:
        engine.dispose()

    for entity in resolved.entities:
        typer.echo(f"{entity.ticker:<6} {entity.kind:<8} {entity.name}")
    if resolved.unmatched_others:
        typer.echo(
            f"Not in the SEC tables, stored without a CIK: {', '.join(resolved.unmatched_others)}"
        )
    # The company table cannot tell a stand-alone trust from an operating company.
    from_company_table = [
        entity.ticker
        for entity in resolved.entities
        if entity.kind == EntityKind.ETF and entity.cik is not None and entity.series_id is None
    ]
    if from_company_table:
        typer.echo(
            f"Funds identified from the company table, check their names: "
            f"{', '.join(from_company_table)}"
        )
    typer.echo(f"{result.inserted} inserted, {result.updated} updated")
