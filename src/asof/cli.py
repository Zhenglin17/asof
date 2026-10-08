"""Command-line entry point. Subcommands are added as each pipeline stage lands."""

import os
import time
from datetime import date, datetime
from pathlib import Path
from typing import Annotated, NoReturn

import duckdb
import httpx
import typer
from sqlmodel import Session

from asof import __version__
from asof.ingest.alpaca import AlpacaClient, AlpacaError, Timeframe
from asof.ingest.bars import ET, Partition, backfill
from asof.ingest.corporate_actions import build_tables, download_all
from asof.ingest.http import make_fetch
from asof.ingest.sec_ticker_history import (
    EXCHANGE_TARGET,
    SNAPSHOT_TARGET,
    build_history,
    download_snapshot,
    list_snapshots,
    symbol_windows,
    write_history,
)
from asof.ingest.sec_tickers import download_sec_listings, load_sec_listings
from asof.ingest.security_master import build_from_store, load_overrides, write_master
from asof.ingest.tickers import is_valid_symbol, normalize_ticker
from asof.ingest.universe import (
    EXCHANGE_DIR,
    HISTORY_PATH,
    active_symbols,
    load_assets,
    load_history,
    load_universe,
    save_assets_snapshot,
)
from asof.ingest.watchlist import load_watchlist, resolve_watchlist
from asof.store.db import default_data_dir, default_db_path, init_db, make_engine
from asof.store.entities import EntityConflict, upsert_entities
from asof.store.market import bars_status, market_root
from asof.store.models import EntityKind

SEC_USER_AGENT_ENV = "ASOF_SEC_USER_AGENT"
ALPACA_KEY_ENV = "ALPACA_API_KEY"
ALPACA_SECRET_ENV = "ALPACA_SECRET_KEY"

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
market_app = typer.Typer(
    help="Market data: symbol universe and bar backfill.", no_args_is_help=True
)
app.add_typer(market_app, name="market")


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


def _alpaca_client() -> AlpacaClient:
    key, secret = os.environ.get(ALPACA_KEY_ENV), os.environ.get(ALPACA_SECRET_ENV)
    if not key or not secret:
        _fail(f"Set {ALPACA_KEY_ENV} and {ALPACA_SECRET_ENV} (Alpaca paper-trading keys).")
    return AlpacaClient(key, secret)


def _market_root(data_dir: Path | None) -> Path:
    return market_root(data_dir or default_data_dir())


def _timeframe(value: str) -> Timeframe:
    if value == "1Day":
        return "1Day"
    if value == "1Min":
        return "1Min"
    _fail("--timeframe must be 1Day or 1Min")


def _parse_symbols(value: str) -> list[str]:
    symbols: list[str] = []
    for raw in value.split(","):
        try:
            symbol = normalize_ticker(raw)
        except ValueError:
            _fail(f"Bad symbol: {raw!r}")
        if not is_valid_symbol(symbol):
            _fail(f"Bad symbol: {raw!r}")
        symbols.append(symbol)
    return sorted(set(symbols))


@market_app.command("universe")
def market_universe(
    data_dir: Annotated[
        Path | None, typer.Option("--data-dir", help="Defaults to $ASOF_DATA_DIR.")
    ] = None,
    history: Annotated[
        bool,
        typer.Option(
            "--history/--no-history",
            help="Also pull historical SEC ticker snapshots from the Wayback Machine.",
        ),
    ] = True,
    since_year: Annotated[int, typer.Option(help="First year of SEC snapshots to fetch.")] = 2018,
) -> None:
    """Snapshot Alpaca's asset list and refresh the SEC ticker history table."""
    root = _market_root(data_dir)
    client = _alpaca_client()
    try:
        assets = client.assets("active") + client.assets("inactive")
    except AlpacaError as error:
        _fail(f"Alpaca asset listing failed: {error}")
    path = save_assets_snapshot(assets, root, datetime.now(ET).date())
    typer.echo(f"{len(assets)} assets -> {path}")

    base = data_dir or default_data_dir()
    if history:
        user_agent = os.environ.get(SEC_USER_AGENT_ENV, "asof")
        fetch = make_fetch(
            {"User-Agent": f"asof ({user_agent})"}, timeout=120.0, follow_redirects=True
        )
        targets = [
            (SNAPSHOT_TARGET, base / "raw" / "sec" / "company_tickers"),
            (EXCHANGE_TARGET, base / EXCHANGE_DIR),
        ]
        for target, snapshots_dir in targets:
            try:
                timestamps = list_snapshots(fetch, since_year=since_year, target=target)
            except (ValueError, httpx.HTTPError) as error:
                _fail(f"Could not list Wayback snapshots of {target}: {error}")
            failed = 0
            for ts in timestamps:
                for attempt in range(3):
                    try:
                        download_snapshot(ts, snapshots_dir, fetch, target=target)
                        break
                    except (ValueError, OSError, httpx.HTTPError) as error:
                        if attempt == 2:
                            failed += 1
                            typer.echo(f"  {ts}: giving up ({error})")
                        else:
                            time.sleep(5.0 * (attempt + 1))
            typer.echo(f"{len(timestamps) - failed} snapshots of {target} -> {snapshots_dir}")
        table = build_history(targets[0][1])
        write_history(table, root / HISTORY_PATH)
        typer.echo(f"{table.num_rows} ticker-history rows -> {root / HISTORY_PATH}")
    typer.echo(f"Universe: {len(load_universe(root, base))} symbols")


@market_app.command("backfill")
def market_backfill(
    timeframe: Annotated[str, typer.Option(help="1Day or 1Min.")] = "1Day",
    start: Annotated[str, typer.Option(help="First session date, YYYY-MM-DD.")] = "2020-10-01",
    end: Annotated[str | None, typer.Option(help="Last session date. Defaults to today.")] = None,
    symbols: Annotated[
        str | None, typer.Option(help="Comma-separated symbols instead of the stored universe.")
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="Refetch complete partitions.")] = False,
    data_dir: Annotated[
        Path | None, typer.Option("--data-dir", help="Defaults to $ASOF_DATA_DIR.")
    ] = None,
) -> None:
    """Download bars into partitioned Parquet; reruns only fetch what is missing or open."""
    tf = _timeframe(timeframe)
    root = _market_root(data_dir)
    universe = (
        _parse_symbols(symbols) if symbols else load_universe(root, data_dir or default_data_dir())
    )
    windows = symbol_windows(load_history(root))
    active = active_symbols(load_assets(root))
    if not universe:
        _fail("Universe is empty. Run `asof market universe` first or pass --symbols.")
    try:
        start_date = date.fromisoformat(start)
        end_date = date.fromisoformat(end) if end else datetime.now(ET).date()
    except ValueError as error:
        _fail(f"Bad date: {error}")
    client = _alpaca_client()
    started = time.monotonic()

    def progress(partition: Partition, outcome: str, rows: int) -> None:
        period = f"{partition.year}" + (f"-{partition.month:02d}" if partition.month else "")
        typer.echo(f"  {period} {partition.bucket}: {outcome} {rows:>9,} rows")

    typer.echo(f"{len(universe)} symbols, {tf}, {start_date} .. {end_date}")
    try:
        report = backfill(
            client,
            root,
            tf,
            start_date,
            end_date,
            universe,
            force=force,
            on_partition=progress,
            windows=windows,
            active=active,
        )
    except AlpacaError as error:
        _fail(f"Stopped: {error}. Completed partitions are kept; rerun to continue.")
    typer.echo(
        f"{len(report.written)} partitions written, {len(report.skipped)} skipped, "
        f"{report.rows:,} rows, {time.monotonic() - started:,.0f}s"
    )


@market_app.command("status")
def market_status(
    timeframe: Annotated[str, typer.Option(help="1Day or 1Min.")] = "1Day",
    data_dir: Annotated[
        Path | None, typer.Option("--data-dir", help="Defaults to $ASOF_DATA_DIR.")
    ] = None,
) -> None:
    """Rows, symbols and time range stored per year."""
    tf = _timeframe(timeframe)
    con = duckdb.connect()
    try:
        rows = bars_status(con, _market_root(data_dir), tf)
    finally:
        con.close()
    if not rows:
        typer.echo("Nothing stored yet.")
        return
    for row in rows:
        first = row["first"].date() if row["first"] else "-"
        last = row["last"].date() if row["last"] else "-"
        typer.echo(
            f"{row['year']}  {row['rows']:>12,} rows  {row['symbols']:>6,} symbols  "
            f"{first} .. {last}"
        )


@market_app.command("actions")
def market_actions(
    start: Annotated[str, typer.Option(help="First effective date, YYYY-MM-DD.")] = "2020-01-01",
    end: Annotated[str | None, typer.Option(help="Last effective date. Defaults to today.")] = None,
    download: Annotated[
        bool, typer.Option("--download/--no-download", help="Fetch from Alpaca before parsing.")
    ] = True,
    data_dir: Annotated[
        Path | None, typer.Option("--data-dir", help="Defaults to $ASOF_DATA_DIR.")
    ] = None,
) -> None:
    """Download Alpaca corporate actions (splits, renames, mergers, dividends) and parse them."""
    data_root = data_dir or default_data_dir()
    root = _market_root(data_dir)
    if download:
        try:
            start_date = date.fromisoformat(start)
            end_date = date.fromisoformat(end) if end else datetime.now(ET).date()
        except ValueError as error:
            _fail(f"Bad date: {error}")
        started = time.monotonic()
        try:
            counts = download_all(_alpaca_client(), data_root, start_date, end_date)
        except AlpacaError as error:
            _fail(f"Stopped: {error}")
        for type_, n in counts.items():
            typer.echo(f"  {type_:<24} {n:>8,} records")
        typer.echo(f"downloaded in {time.monotonic() - started:,.0f}s")
    counts = build_tables(data_root, root)
    for name, n in counts.items():
        typer.echo(f"  {name:<12} {n:>8,} rows")


@market_app.command("identity")
def market_identity(
    overrides: Annotated[Path, typer.Option(help="Human decisions applied last.")] = Path(
        "configs/security_overrides.yaml"
    ),
    data_dir: Annotated[
        Path | None, typer.Option("--data-dir", help="Defaults to $ASOF_DATA_DIR.")
    ] = None,
) -> None:
    """Build the security master (which security each symbol was on each day) and list the
    conflicts the code could not settle."""
    root = _market_root(data_dir)
    started = time.monotonic()
    if not overrides.exists():
        _fail(f"{overrides}: overrides file not found (run from the repo root or pass --overrides)")
    try:
        decisions = load_overrides(overrides)
    except ValueError as error:
        _fail(str(error))
    master, conflicts = build_from_store(root, decisions)
    write_master(master, conflicts, root)
    symbols = len(set(master.column("symbol").to_pylist()))
    securities = len(set(master.column("security_id").to_pylist()))
    typer.echo(
        f"{master.num_rows:,} segments, {symbols:,} symbols, {securities:,} securities, "
        f"{time.monotonic() - started:,.0f}s"
    )
    by_kind: dict[str, int] = {}
    for row in conflicts.to_pylist():
        by_kind[row["kind"]] = by_kind.get(row["kind"], 0) + 1
    for kind, n in sorted(by_kind.items()):
        typer.echo(f"  conflicts {kind:<28} {n:>6,}")
    if not by_kind:
        typer.echo("  no conflicts")
