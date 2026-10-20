"""Command-line entry point. Subcommands are added as each pipeline stage lands."""

import dataclasses
import hashlib
import os
import time
from concurrent.futures.process import BrokenProcessPool
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Annotated, NoReturn

import duckdb
import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import typer
from sqlmodel import Session

from asof import __version__
from asof.ingest.alpaca import AlpacaClient, AlpacaError, Timeframe
from asof.ingest.bar_imports import import_bars, load_bar_imports
from asof.ingest.bars import ET, Partition, backfill, windows_for_requests
from asof.ingest.corporate_actions import build_tables, download_all
from asof.ingest.http import make_fetch
from asof.ingest.instrument_class import (
    CLASSES,
    load_class_overrides,
    load_rules,
    write_instrument_class,
)
from asof.ingest.instrument_class import build_from_store as build_classes_from_store
from asof.ingest.master_truncation import (
    default_instants,
    diff_key,
    gate,
    load_baseline,
    load_instants,
    load_truncation_exceptions,
    prune_baseline,
    run_instants,
)
from asof.ingest.sec_ticker_history import (
    EXCHANGE_HISTORY_PATH,
    EXCHANGE_TARGET,
    SNAPSHOT_TARGET,
    build_exchange_history,
    build_history,
    download_snapshot,
    list_snapshots,
    symbol_windows,
    write_history,
)
from asof.ingest.sec_tickers import (
    COMPANY_FILE,
    FUND_FILE,
    download_sec_listings,
    load_sec_listings,
)
from asof.ingest.security_master import (
    MASTER_FILE,
    build_from_inputs,
    build_from_store,
    load_overrides,
    load_store_inputs,
    write_master,
)
from asof.ingest.tickers import is_valid_symbol, normalize_ticker
from asof.ingest.universe import (
    EXCHANGE_DIR,
    HISTORY_PATH,
    active_symbols,
    load_assets,
    load_history,
    load_universe,
    read_symbols_file,
    rename_chains,
    save_assets_snapshot,
)
from asof.ingest.watchlist import load_watchlist, resolve_watchlist
from asof.ingest.yahoo import YahooClient, YahooError
from asof.store.audit import (
    CHECKS,
    load_exceptions,
    run_audit,
    stale_exceptions,
    write_report,
)
from asof.store.db import default_data_dir, default_db_path, init_db, make_engine
from asof.store.entities import EntityConflict, upsert_entities
from asof.store.market import bars_status, market_root
from asof.store.models import EntityKind
from asof.store.universe import (
    LIQUID_CLASSES,
    MIN_ADV_USD,
    liquid_securities,
    month_starts,
    tier_union,
)

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
        listings = build_exchange_history(targets[1][1])
        if listings.num_rows:  # never replace a usable table with an empty one
            write_history(listings, root / EXCHANGE_HISTORY_PATH)
        typer.echo(f"{listings.num_rows} exchange-listing rows -> {root / EXCHANGE_HISTORY_PATH}")
    typer.echo(f"Universe: {len(load_universe(root, base))} symbols")


@market_app.command("backfill")
def market_backfill(
    timeframe: Annotated[str, typer.Option(help="1Day or 1Min.")] = "1Day",
    start: Annotated[str, typer.Option(help="First session date, YYYY-MM-DD.")] = "2020-10-01",
    end: Annotated[str | None, typer.Option(help="Last session date. Defaults to today.")] = None,
    symbols: Annotated[
        str | None, typer.Option(help="Comma-separated symbols instead of the stored universe.")
    ] = None,
    symbols_file: Annotated[
        Path | None,
        typer.Option(
            "--symbols-file",
            help="File with one symbol per line (# comments) instead of the stored universe.",
        ),
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="Refetch complete partitions.")] = False,
    data_dir: Annotated[
        Path | None, typer.Option("--data-dir", help="Defaults to $ASOF_DATA_DIR.")
    ] = None,
) -> None:
    """Download bars into partitioned Parquet; reruns only fetch what is missing or open.
    Stored bars are never removed: a refetch adds to a partition file, and discarding one means
    deleting the file. With --symbols or --symbols-file plus --force, only those symbols are
    refetched into their partitions (a top-up after a universe fix)."""
    tf = _timeframe(timeframe)
    root = _market_root(data_dir)
    if symbols and symbols_file:
        _fail("Pass either --symbols or --symbols-file, not both.")
    if symbols:
        universe = _parse_symbols(symbols)
    elif symbols_file:
        try:
            universe = read_symbols_file(symbols_file)
        except (OSError, ValueError) as error:
            _fail(str(error))
    else:
        universe = load_universe(root, data_dir or default_data_dir())
    active = active_symbols(load_assets(root))
    # an SEC window only limits symbols Alpaca no longer lists (NOBL's window closed in 2020
    # while it kept trading; 197 listed symbols lost years of bars that way)
    windows = windows_for_requests(symbol_windows(load_history(root)), active)
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
            chains=rename_chains(root),
        )
    except AlpacaError as error:
        _fail(f"Stopped: {error}. Completed partitions are kept; rerun to continue.")
    typer.echo(
        f"{len(report.written)} partitions written, {len(report.skipped)} skipped, "
        f"{report.rows:,} rows, {time.monotonic() - started:,.0f}s"
    )


@market_app.command("import-bars")
def market_import_bars(
    manifest: Annotated[Path, typer.Option(help="Registered second-source imports.")] = Path(
        "configs/bar_imports.yaml"
    ),
    data_dir: Annotated[
        Path | None, typer.Option("--data-dir", help="Defaults to $ASOF_DATA_DIR.")
    ] = None,
) -> None:
    """Fetch every registered import from its source and file it under <market>/bars_1d_ext.
    Rerunning adds nothing for keys already stored; the raw response is kept under
    $ASOF_DATA_DIR/raw/<source>/."""
    root = _market_root(data_dir)
    raw_root = (data_dir or default_data_dir()) / "raw"
    try:
        entries = load_bar_imports(manifest)
    except (OSError, ValueError) as error:
        _fail(str(error))
    if not entries:
        typer.echo(f"{manifest}: nothing registered")
        return
    client = YahooClient()
    for entry in entries:
        fetched_at = datetime.now(UTC)
        try:
            bars, body = client.daily_bars(entry.source_symbol, entry.start, entry.end)
        except (YahooError, httpx.HTTPError) as error:
            _fail(f"{entry.symbol} <- {entry.source} {entry.source_symbol}: {error}")
        archive = (
            raw_root
            / entry.source
            / (f"{entry.source_symbol}_{entry.start}_{entry.end}_{fetched_at:%Y%m%dT%H%M%SZ}.json")
        )
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(body)
        report = import_bars(root, entry, bars, fetched_at)
        typer.echo(
            f"{entry.symbol} <- {entry.source} {entry.source_symbol} {entry.start}..{entry.end}: "
            f"{report.rows_fetched:,} fetched, {report.rows_in_range:,} in range, "
            f"{report.rows_added:,} added, {report.rows_existing:,} already stored"
            + (f" ({report.first_session}..{report.last_session})" if report.first_session else "")
        )
        typer.echo(f"  raw -> {archive}")


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


@market_app.command("instruments")
def market_instruments(
    rules: Annotated[Path, typer.Option(help="Ordered class rules.")] = Path(
        "configs/instrument_class_rules.yaml"
    ),
    overrides: Annotated[Path, typer.Option(help="Human decisions applied last.")] = Path(
        "configs/instrument_class_overrides.yaml"
    ),
    sec_dir: Annotated[
        Path | None,
        typer.Option(
            "--sec-dir",
            help="Directory holding the SEC ticker tables. Defaults to $ASOF_DATA_DIR/raw/sec.",
        ),
    ] = None,
    data_dir: Annotated[
        Path | None, typer.Option("--data-dir", help="Defaults to $ASOF_DATA_DIR.")
    ] = None,
) -> None:
    """Tag every security-master segment with its instrument class (common, etf, warrant...)
    and list how many segments each class and each rule got. Re-run after `market identity`:
    the table is keyed by the master's segments."""
    root = _market_root(data_dir)
    tables_dir = sec_dir or (data_dir or default_data_dir()) / "raw" / "sec"
    if not rules.exists():
        _fail(f"{rules}: rules file not found (run from the repo root or pass --rules)")
    if not overrides.exists():
        _fail(f"{overrides}: overrides file not found (run from the repo root or pass --overrides)")
    if not (root / MASTER_FILE).exists():
        _fail(f"{root / MASTER_FILE}: no security master; run `asof market identity` first")
    for name in (COMPANY_FILE, FUND_FILE):
        if not (tables_dir / name).exists():
            _fail(f"{tables_dir / name}: SEC table missing; run `asof entities sync --download`")
    try:
        loaded = load_rules(rules)
        decisions = load_class_overrides(overrides)
    except ValueError as error:
        _fail(str(error))
    started = time.monotonic()
    try:
        table = build_classes_from_store(root, tables_dir, loaded, decisions)
    except ValueError as error:
        _fail(str(error))
    path = write_instrument_class(table, root)
    typer.echo(f"{table.num_rows:,} segments -> {path} ({time.monotonic() - started:,.0f}s)")
    by_class: dict[str, int] = {}
    by_rule: dict[str, int] = {}
    for cls, rule in zip(
        table.column("class").to_pylist(), table.column("rule").to_pylist(), strict=True
    ):
        by_class[cls] = by_class.get(cls, 0) + 1
        by_rule[rule] = by_rule.get(rule, 0) + 1
    for cls in CLASSES:
        if cls in by_class:
            typer.echo(f"  {cls:<14} {by_class[cls]:>7,}")
    for rule, n in sorted(by_rule.items(), key=lambda kv: -kv[1]):
        typer.echo(f"    rule {rule:<24} {n:>7,}")
    unknown = [
        (row["symbol"], row["name"]) for row in table.to_pylist() if row["class"] == "unknown"
    ]
    for symbol, name in unknown[:20]:
        typer.echo(f"  unknown {symbol:<8} {name}")
    if len(unknown) > 20:
        typer.echo(f"  ... {len(unknown) - 20:,} more unknown")


def _as_of_option(value: datetime) -> datetime:
    """A date or datetime on the command line; naive values are New York time."""
    return value if value.tzinfo is not None else value.replace(tzinfo=ET)


@market_app.command("liquid")
def market_liquid(
    as_of: Annotated[
        datetime | None,
        typer.Option("--as-of", help="One tier at this instant (date = 00:00 New York)."),
    ] = None,
    start: Annotated[
        datetime | None, typer.Option("--from", help="Union mode: first month to include.")
    ] = None,
    end: Annotated[
        datetime | None, typer.Option("--to", help="Union mode: last month to include.")
    ] = None,
    min_adv: Annotated[
        float, typer.Option("--min-adv", min=0.0, help="Average daily dollar volume threshold.")
    ] = MIN_ADV_USD,
    top: Annotated[int, typer.Option(min=0, help="Rows to print in single-tier mode.")] = 20,
    data_dir: Annotated[
        Path | None, typer.Option("--data-dir", help="Defaults to $ASOF_DATA_DIR.")
    ] = None,
) -> None:
    """The liquid tier: securities averaging more than --min-adv dollars a day over the last 20
    sessions, common stocks and ETFs only. Either one tier (--as-of) or one per month start
    (--from/--to) with the union and every qualified-but-unknown segment listed."""
    root = _market_root(data_dir)
    if (as_of is None) == (start is None or end is None):
        _fail("pass either --as-of, or both --from and --to")
    started = time.monotonic()
    con = duckdb.connect()
    try:
        if as_of is not None:
            table = liquid_securities(
                con, root, _as_of_option(as_of), min_adv_usd=min_adv, classes=None
            )
        else:
            assert start is not None and end is not None
            table = tier_union(
                con, root, start.date(), end.date(), min_adv_usd=min_adv, classes=None
            )
    except (FileNotFoundError, ValueError) as error:
        _fail(str(error))
    finally:
        con.close()
    rows = table.to_pylist()
    in_tier = [r for r in rows if r["class"] in LIQUID_CLASSES]
    elapsed = time.monotonic() - started

    if as_of is not None:
        by_class: dict[str, int] = {}
        for r in rows:
            by_class[r["class"]] = by_class.get(r["class"], 0) + 1
        when = _as_of_option(as_of).astimezone(ET)
        typer.echo(
            f"{len(in_tier):,} securities in the liquid tier at {when:%Y-%m-%d %H:%M %Z} "
            f"({len(rows):,} above the threshold in any class, {elapsed:,.1f}s)"
        )
        for cls in CLASSES:
            if cls in by_class:
                typer.echo(f"  {cls:<14} {by_class[cls]:>6,}")
        for r in in_tier[:top]:
            typer.echo(
                f"  {r['symbol']:<8} {r['security_id']:<8} {r['class']:<8} "
                f"{r['adv_usd'] / 1e6:>10,.0f} M  {r['sessions_traded']:>2} traded"
            )
    else:
        assert start is not None and end is not None
        months: dict[datetime, int] = dict.fromkeys(month_starts(start.date(), end.date()), 0)
        for r in in_tier:
            months[r["as_of"]] += 1
        for month, n in months.items():
            typer.echo(
                f"  {month.astimezone(ET).date()}  {n:>6,}"
                + ("  (no visible sessions)" if n == 0 else "")
            )
        union = {r["security_id"] for r in in_tier}
        typer.echo(
            f"{len(union):,} securities ever in the liquid tier over {len(months)} month starts "
            f"({elapsed:,.0f}s)"
        )
        by_class = {}
        for _sid, cls in {(r["security_id"], r["class"]) for r in rows}:
            by_class[cls] = by_class.get(cls, 0) + 1
        for cls in CLASSES:
            if cls in by_class:
                typer.echo(f"  {cls:<14} {by_class[cls]:>6,}")
    unknown: dict[str, int] = {}
    for r in rows:
        if r["class"] == "unknown":
            unknown[r["symbol"]] = unknown.get(r["symbol"], 0) + 1
    if unknown:
        typer.echo(f"{len(unknown)} qualified-but-unknown symbols (add to the class overrides):")
        for symbol, n in sorted(unknown.items(), key=lambda kv: -kv[1]):
            typer.echo(f"  {symbol:<8} {n} month(s)")
        raise typer.Exit(code=2)  # 1 is "could not run"; 2 is "ran and found work for a human"
    typer.echo("  no qualified-but-unknown symbols")


@market_app.command("audit")
def market_audit(
    tier: Annotated[
        str, typer.Option(help="liquid: gated checks count rows in the liquid tier; all: anywhere.")
    ] = "liquid",
    as_of: Annotated[
        datetime | None,
        typer.Option("--as-of", help="Audit the store as known then (date = 00:00 New York)."),
    ] = None,
    start: Annotated[
        datetime, typer.Option("--from", help="First month of the liquid-tier union.")
    ] = datetime(2020, 1, 1),
    end: Annotated[
        datetime | None,
        typer.Option("--to", help="Last month of the liquid-tier union. Defaults to --as-of."),
    ] = None,
    min_adv: Annotated[
        float, typer.Option("--min-adv", min=0.0, help="Liquid-tier dollar volume threshold.")
    ] = MIN_ADV_USD,
    checks: Annotated[
        str | None, typer.Option("--checks", help="Comma-separated check ids. Default: all.")
    ] = None,
    exceptions: Annotated[Path, typer.Option(help="Reviewed exceptions.")] = Path(
        "configs/audit_exceptions.yaml"
    ),
    data_dir: Annotated[
        Path | None, typer.Option("--data-dir", help="Defaults to $ASOF_DATA_DIR.")
    ] = None,
) -> None:
    """Count every known kind of data accident in the store. Exit 0 when the gated checks are
    clean outside the exceptions, 2 when they are not; rows go to <market>/audit/<id>.parquet."""
    if tier not in ("liquid", "all"):
        _fail("--tier must be liquid or all")
    root = _market_root(data_dir)
    when = _as_of_option(as_of) if as_of is not None else datetime.now(ET)
    last = end.date() if end is not None else when.astimezone(ET).date()
    wanted = [c.strip() for c in checks.split(",") if c.strip()] if checks else None
    try:
        reviewed = load_exceptions(exceptions)
    except ValueError as error:
        _fail(str(error))
    started = time.monotonic()
    con = duckdb.connect()
    try:
        # every class above the threshold; the liquid tier is its common / etf part, and an
        # unknown class above the threshold is what M5 must catch
        qualified = tier_union(con, root, start.date(), last, min_adv_usd=min_adv, classes=None)
        qualified_ids = set(qualified.column("security_id").to_pylist())
        liquid_ids = {
            r["security_id"] for r in qualified.to_pylist() if r["class"] in LIQUID_CLASSES
        }
        results = run_audit(
            con,
            root,
            as_of=when,
            liquid_ids=liquid_ids,
            qualified_ids=qualified_ids,
            exceptions=reviewed,
            checks=wanted,
            liquid_tier=qualified,  # split records are gated by the tier around their ex-date
        )
    except (FileNotFoundError, ValueError) as error:
        _fail(str(error))
    finally:
        con.close()
    report = write_report(results, root)

    typer.echo(
        f"as of {when.astimezone(ET):%Y-%m-%d %H:%M %Z}; liquid tier {len(liquid_ids):,} "
        f"securities ({start:%Y-%m}..{last:%Y-%m}, > {min_adv / 1e6:,.0f}M); "
        f"{time.monotonic() - started:,.0f}s"
    )
    typer.echo(f"{'':3}{'check':<4} {'gate':<6} {'liquid':>8} {'all':>9} {'excepted':>8}  title")
    failed = []
    for r in results:
        mark = "X" if r.fails(tier) else " "  # type: ignore[arg-type]
        if r.fails(tier):  # type: ignore[arg-type]
            failed.append(r.spec.id)
        typer.echo(
            f" {mark} {r.spec.id:<4} {r.spec.gate or '-':<6} {r.liquid:>8,} {r.total:>9,} "
            f"{r.excepted:>8,}  {r.spec.title}"
        )
        if r.notes:
            notes = ", ".join(
                f"{k}={v:,.4g}"
                if isinstance(v, float)
                else f"{k}={v:,}"
                if isinstance(v, int)
                else f"{k}={v}"
                for k, v in r.notes.items()
            )
            typer.echo(f"{'':13}{notes}")
    for e in stale_exceptions(results, reviewed):
        typer.echo(f"stale exception: {e.check} {e.key} ({e.reason})")
    typer.echo(f"rows -> {report}/<check>.parquet")
    if failed:
        typer.echo(f"gated checks not clean at tier {tier}: {', '.join(failed)}")
        raise typer.Exit(code=2)
    typer.echo(f"all gated checks clean at tier {tier} ({len(CHECKS)} checks known)")


@market_app.command("truncation")
def market_truncation(
    as_of: Annotated[
        list[datetime] | None,
        typer.Option(
            "--as-of",
            help="Instant to rebuild at (date = 00:00 New York); repeatable. "
            "Default: the instants in --instants, else the named instants plus each "
            "evidence kind's spread.",
        ),
    ] = None,
    overrides: Annotated[Path, typer.Option(help="Human decisions applied last.")] = Path(
        "configs/security_overrides.yaml"
    ),
    exceptions: Annotated[Path, typer.Option(help="Reviewed leaks.")] = Path(
        "configs/truncation_exceptions.yaml"
    ),
    baseline: Annotated[Path, typer.Option(help="Known leaks not fixed yet (a ratchet).")] = Path(
        "configs/truncation_baseline.txt"
    ),
    prune: Annotated[
        bool,
        typer.Option("--prune-baseline", help="Delete baseline keys that no longer occur."),
    ] = False,
    instants_file: Annotated[
        Path, typer.Option("--instants", help="Instants of a default run (fixed by hand).")
    ] = Path("configs/truncation_instants.txt"),
    computed: Annotated[
        bool,
        typer.Option(
            "--computed-instants",
            help="Use the computed spread instead of --instants (a partial run: no pruning).",
        ),
    ] = False,
    jobs: Annotated[
        int, typer.Option("--jobs", help="Rebuilds run at once (6: about 20 GB in all).")
    ] = 6,
    data_dir: Annotated[
        Path | None, typer.Option("--data-dir", help="Defaults to $ASOF_DATA_DIR.")
    ] = None,
) -> None:
    """Rebuild the security master from the inputs knowable at each instant and diff it with the
    full master as seen then. Exit 0 when every leak is in the baseline or the exceptions and,
    on a default run, every baseline key still occurs; 2 when not. Rows go to
    <market>/audit/truncation/<UTC instant>.parquet."""
    root = _market_root(data_dir)
    if not overrides.exists():
        _fail(f"{overrides}: overrides file not found (run from the repo root or pass --overrides)")
    try:
        decisions = load_overrides(overrides)
        reviewed = load_truncation_exceptions(exceptions)
        known = load_baseline(baseline)
        fixed_instants = load_instants(instants_file)
        gate([], baseline=known, exceptions=reviewed, full_run=False)  # overlap, before 26 min
    except ValueError as error:
        _fail(str(error))
    if not 1 <= jobs <= (os.cpu_count() or 1):
        _fail(f"--jobs must be between 1 and {os.cpu_count() or 1} (the cores here), got {jobs}")
    if as_of and computed:
        _fail("--as-of and --computed-instants exclude each other")
    if prune and (as_of or computed):
        _fail("--prune-baseline needs the fixed instants: a partial run cannot tell fixed")
    if not as_of and not computed and not fixed_instants:
        _fail(f"{instants_file}: no instants (pass --instants, --as-of or --computed-instants)")
    started = time.monotonic()
    inputs = load_store_inputs(root)
    full, _ = build_from_inputs(inputs, decisions)
    if as_of:
        instants = sorted({_as_of_option(t).astimezone(UTC) for t in as_of})
    elif not computed:
        instants = fixed_instants
        unlisted = sorted(set(default_instants(full)) - set(fixed_instants))
        if unlisted:  # the data moved: worth a look, never added on its own
            typer.echo(
                f"{len(unlisted)} computed instants not in {instants_file} (not checked): "
                + ", ".join(f"{t.isoformat()}" for t in unlisted[:5])
                + (" ..." if len(unlisted) > 5 else "")
            )
    else:
        instants = default_instants(full)
    out_dir = root / "audit" / "truncation"
    out_dir.mkdir(parents=True, exist_ok=True)
    typer.echo(f"full master {full.num_rows:,} segments; {len(instants)} instants")
    first_at: dict[str, str] = {}

    def record(t: datetime, diff: pa.Table) -> None:
        path = out_dir / f"{t:%Y%m%dT%H%M%S%f}.parquet"
        tmp = path.with_name(path.name + ".tmp")
        pq.write_table(diff, tmp)
        os.replace(tmp, path)
        leaks = [r for r in diff.to_pylist() if r["kind"] == "leak"]
        for r in leaks:
            first_at.setdefault(diff_key(r), f"{t.astimezone(ET):%Y-%m-%d %H:%M} {r['detail']}")
        typer.echo(
            f"  {t.astimezone(ET):%Y-%m-%d %H:%M:%S.%f %Z}  leaks {len(leaks):>5,}  "
            f"delays {diff.num_rows - len(leaks):>5,}  {time.monotonic() - started:,.0f}s"
        )

    try:
        run_instants(full, inputs, decisions, instants, jobs=jobs, progress=record)
    except ValueError as error:
        _fail(str(error))
    except BrokenProcessPool:
        _fail(f"a worker died (most often out of memory): try a smaller --jobs than {jobs}")
    # only the fixed list stands for every instant: the computed one drifts with the master
    full_run = not as_of and not computed
    verdict = gate(first_at, baseline=known, exceptions=reviewed, full_run=full_run)
    # default instants follow the data: a key gone from them may still occur elsewhere
    fingerprint = hashlib.sha256("|".join(t.isoformat() for t in instants).encode()).hexdigest()
    typer.echo(f"{len(instants)} instants, fingerprint {fingerprint[:12]}")
    typer.echo(f"rows -> {out_dir}/<instant>.parquet; {time.monotonic() - started:,.0f}s")
    for key in verdict.stale_exceptions:
        typer.echo(f"stale exception: {key} ({reviewed[key]})")
    if prune and verdict.new:
        for key in verdict.new[:50]:
            typer.echo(f"  new leak: {key}  first at {first_at[key]}")
        _fail(f"{len(verdict.new):,} new leaks: review them before pruning the baseline")
    if prune and verdict.fixed:
        try:
            removed = prune_baseline(baseline, verdict.fixed)
        except ValueError as error:
            _fail(str(error))
        typer.echo(f"pruned {removed:,} keys no longer occurring from {baseline}")
        verdict = dataclasses.replace(verdict, fixed=[])
    typer.echo(
        f"{len(known):,} baseline keys, {len(reviewed):,} exceptions; "
        f"{len(verdict.new):,} new, {len(verdict.fixed):,} no longer occurring"
        + ("" if full_run else " (partial run: not checked)")
    )
    for key in verdict.fixed[:50]:
        typer.echo(f"  no longer occurs at these instants (--prune-baseline): {key}")
    for key in verdict.new[:50]:
        typer.echo(f"  new leak: {key}  first at {first_at[key]}")
    if not verdict.ok:
        raise typer.Exit(code=2)
    typer.echo("no leak outside the baseline and the exceptions")
