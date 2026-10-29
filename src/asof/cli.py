"""Command-line entry point. Subcommands are added as each pipeline stage lands."""

import dataclasses
import json
import os
import time
from concurrent.futures.process import BrokenProcessPool
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Annotated, NoReturn

import duckdb
import httpx
import pyarrow.parquet as pq
import typer
from sqlmodel import Session, select

from asof import __version__
from asof.demo.build import build
from asof.evidence.retrieve import search
from asof.evidence.validate import ClaimCitation, DecisionDraft, validate
from asof.evolve.benchmark import benchmarks
from asof.evolve.entry import entry_test
from asof.evolve.gate import check as gate_measure
from asof.evolve.gate import render_report
from asof.evolve.holdout import HoldoutError
from asof.evolve.null import null_test
from asof.evolve.protocol import load_protocol
from asof.evolve.versus import compare_run
from asof.ingest.alpaca import AlpacaClient, AlpacaError, Timeframe
from asof.ingest.bar_imports import import_bars, load_bar_imports
from asof.ingest.bars import ET, Partition, backfill, windows_for_requests
from asof.ingest.corporate_actions import build_tables, download_all
from asof.ingest.edgar import (
    REQUESTS_PER_SECOND,
    Throttled,
    audit_filings,
    cik_windows,
    coverage_gaps,
    load_coverage_exceptions,
    missing_filings,
    periodic_gaps,
    rate_limited,
    retrying,
    security_segments,
    stale_reports,
    sync_company,
    trading_segments,
)
from asof.ingest.http import make_fetch
from asof.ingest.instrument_class import (
    CLASS_FILE,
    CLASSES,
    load_class_overrides,
    load_rules,
    write_instrument_class,
)
from asof.ingest.instrument_class import build_from_store as build_classes_from_store
from asof.ingest.master_truncation import default_instants, load_instants
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
    MASTER_SCHEMA,
    build_from_inputs,
    build_from_store,
    load_overrides,
    load_store_inputs,
    write_master,
)
from asof.ingest.tickers import is_valid_symbol, normalize_ticker
from asof.ingest.truncation_gate import (
    ImpactCount,
    attribute,
    baseline_fingerprint,
    gate,
    instants_fingerprint,
    load_impact_baseline,
    prune_impact_baseline,
    rebase_impact_baseline,
)
from asof.ingest.truncation_impact import InstantResult, run_instants
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
from asof.judge.cache import CacheMiss
from asof.judge.client import OpenRouterClient
from asof.judge.window import judge_window
from asof.replay import build_report, close_of, run_replay
from asof.replay.compare import Comparison, Section, compare_runs
from asof.scan.bx_trender import load_scan_config, scan_and_pool
from asof.store.audit import (
    CHECKS,
    load_exceptions,
    run_audit,
    stale_exceptions,
    write_report,
)
from asof.store.db import default_data_dir, default_db_path, init_db, make_engine
from asof.store.documents import audit_documents
from asof.store.entities import EntityConflict, upsert_entities
from asof.store.fts import chunk_index_integrity, chunk_index_mismatch, rebuild_chunk_index
from asof.store.market import bars_status, market_root, visible_bars, visible_splits
from asof.store.models import Entity, EntityKind, Run, StrategyVersion
from asof.store.universe import (
    LIQUID_CLASSES,
    MIN_ADV_USD,
    liquid_securities,
    month_starts,
    tier_union,
)
from asof.strategy import (
    AnyStrategyConfig,
    apply_overrides,
    content_hash,
    load_any_strategy,
    load_strategy,
    overrides_of_summary,
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
docs_app = typer.Typer(help="Text documents: SEC filings and their chunks.", no_args_is_help=True)
app.add_typer(docs_app, name="docs")
evidence_app = typer.Typer(
    help="Evidence: search what was public at a moment, check a decision's citations.",
    no_args_is_help=True,
)
app.add_typer(evidence_app, name="evidence")


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


def _cores() -> int:
    """Cores this process may run on: the affinity mask, not the host (a container or CI runner
    gets fewer than the machine has)."""
    return len(os.sched_getaffinity(0)) or 1


@market_app.command("truncation")
def market_truncation(
    as_of: Annotated[
        list[datetime] | None,
        typer.Option(
            "--as-of",
            help="Instant to rebuild at (date = 00:00 New York); repeatable. "
            "Default: the instants in --instants.",
        ),
    ] = None,
    overrides: Annotated[Path, typer.Option(help="Human decisions applied last.")] = Path(
        "configs/security_overrides.yaml"
    ),
    baseline: Annotated[
        Path, typer.Option(help="Known leak impact per key, only ever lowered (a ratchet).")
    ] = Path("configs/truncation_impact_baseline.txt"),
    prune: Annotated[
        bool,
        typer.Option(
            "--prune-baseline",
            help="Delete keys without impact and lower the numbers that shrank.",
        ),
    ] = False,
    rebase: Annotated[
        bool,
        typer.Option(
            "--rebase-baseline",
            help="After --instants changed: re-sum every baseline number over the new list "
            "(each change is printed; new keys are still added by hand).",
        ),
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
        int | None,
        typer.Option(
            "--jobs",
            help="Rebuilds run at once (default: the smaller of 4 and the cores; 4 take 23 GB, "
            "6 ran out of 31 GB).",
        ),
    ] = None,
    data_dir: Annotated[
        Path | None, typer.Option("--data-dir", help="Defaults to $ASOF_DATA_DIR.")
    ] = None,
) -> None:
    """Rebuild the security master from the inputs knowable at each instant, diff it with the
    full master as seen then, and measure what the leaks change for a backtest: the price cells
    of liquid-tier securities and the CIKs of their symbols. Exit 0 when every leak's impact is
    within the baseline and every leak cell is named by a diff key; 2 when not (on a default run
    also when a baseline key has no impact left). Rows go to <market>/audit/truncation/ and
    <market>/audit/truncation_impact/, one <UTC instant>.parquet each."""
    root = _market_root(data_dir)
    if not overrides.exists():
        _fail(f"{overrides}: overrides file not found (run from the repo root or pass --overrides)")
    try:
        decisions = load_overrides(overrides)
        known = load_impact_baseline(baseline)
        recorded = baseline_fingerprint(baseline)
        fixed_instants = load_instants(instants_file)
    except ValueError as error:
        _fail(str(error))
    # the baseline's numbers are sums over the instants named in its header: on any other list
    # they can be neither exceeded nor undercut
    fixed_print = instants_fingerprint(fixed_instants)
    # a non-empty baseline without the line names no list: as good as another one
    drifted = bool(known) and recorded != fixed_print
    if prune and rebase:
        _fail("--prune-baseline and --rebase-baseline exclude each other")
    if prune and drifted:
        _fail(
            f"{instants_file} (fingerprint {fixed_print}) is not the list {baseline}'s numbers "
            f"were summed over ({recorded or 'none recorded'}): --prune-baseline would drop "
            "leaks seen only at the instants removed. Re-sum them with --rebase-baseline"
        )
    if rebase and not drifted:
        # re-summing over the same list could only raise numbers past what the gate allows
        _fail(f"{baseline} is summed over {instants_file} already: nothing to rebase")
    # before a minute of building: the liquid tier reads the master and the classes, and a
    # missing split file would read as "no split", hiding a leak through one
    splits_file = root / "corporate_actions" / "splits.parquet"
    for needed in (root / MASTER_FILE, root / CLASS_FILE, splits_file):
        if not needed.exists():
            _fail(f"{needed}: not found (build the store first)")
    cores = _cores()
    if jobs is None:
        jobs = min(4, cores)  # memory bounds it (~6 GB per late instant); CI has 2-4 cores
    elif not 1 <= jobs <= cores:
        _fail(f"--jobs must be between 1 and {cores} (the cores here), got {jobs}")
    if as_of and computed:
        _fail("--as-of and --computed-instants exclude each other")
    if (prune or rebase) and (as_of or computed):
        flag = "--prune-baseline" if prune else "--rebase-baseline"
        _fail(f"{flag} needs the fixed instants: a partial run cannot tell fixed")
    if not as_of and not computed and not fixed_instants:
        _fail(f"{instants_file}: no instants (pass --instants, --as-of or --computed-instants)")
    started = time.monotonic()
    inputs = load_store_inputs(root)
    full, _ = build_from_inputs(inputs, decisions)
    # the liquid tier names the ids of the master on disk; the full master here must be it
    on_disk = pq.read_table(root / MASTER_FILE).select(MASTER_SCHEMA.names)
    order = [("security_id", "ascending"), ("symbol", "ascending"), ("valid_from", "ascending")]
    if not on_disk.sort_by(order).equals(full.select(MASTER_SCHEMA.names).sort_by(order)):
        _fail(
            f"{root / MASTER_FILE} is not the master these inputs build: run `asof market identity`"
        )
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
    # only the fixed list stands for every instant: the computed one drifts with the master
    full_run = not as_of and not computed and (not drifted or rebase)
    comparable = not drifted and set(instants) <= set(fixed_instants)
    if drifted and not rebase:
        typer.echo(
            f"{instants_file} is not the list the baseline was summed over "
            f"({recorded or 'none recorded'}, now "
            f"{fixed_print}): its numbers compare with nothing and this run fails; re-sum them "
            "with --rebase-baseline"
        )
    elif not comparable and not rebase:
        typer.echo("instants outside the fixed list: only new keys and unmatched cells are judged")
    # the tier through the last fixed instant whatever this run picks: a partial run then
    # measures the same securities as the full one
    last = max([*fixed_instants, *instants]).astimezone(ET).date()
    con = duckdb.connect()
    try:
        tier = tier_union(con, root, date(2020, 1, 1), last)
        splits = visible_splits(con, root, instants[-1])
    except (ValueError, FileNotFoundError) as error:
        _fail(str(error))
    finally:
        con.close()
    liquid = set(tier.column("security_id").to_pylist())
    diff_dir = root / "audit" / "truncation"
    impact_dir = root / "audit" / "truncation_impact"
    for out_dir in (diff_dir, impact_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        if full_run:  # an instant dropped from --instants must not pass for part of this run
            for stale in out_dir.glob("[0-9]*T*.parquet"):
                stale.unlink()
    typer.echo(
        f"full master {full.num_rows:,} segments; {len(liquid):,} securities ever liquid "
        f"2020-01..{last:%Y-%m}; {len(instants)} instants"
    )

    def record(result: InstantResult) -> None:
        name = f"{result.as_of:%Y%m%dT%H%M%S%f}.parquet"
        for out_dir, table in ((diff_dir, result.diff), (impact_dir, result.impact)):
            tmp = out_dir / f"{name}.tmp"
            pq.write_table(table, tmp)
            os.replace(tmp, out_dir / name)
        leaks = sum(1 for k in result.diff.column("kind").to_pylist() if k == "leak")
        days = {
            what: sum(
                r["days"]
                for r in result.impact.to_pylist()
                if (r["kind"], r["what"]) == ("leak", what)
            )
            for what in ("price", "cik")
        }
        typer.echo(
            f"  {result.as_of.astimezone(ET):%Y-%m-%d %H:%M:%S.%f %Z}  leaks {leaks:>5,}  "
            f"delays {result.diff.num_rows - leaks:>5,}  leak days price {days['price']:>6,} "
            f"cik {days['cik']:>6,}  {time.monotonic() - started:,.0f}s"
        )

    try:
        results = run_instants(
            full, inputs, decisions, splits, instants, liquid=liquid, jobs=jobs, progress=record
        )
    except ValueError as error:
        _fail(str(error))
    except BrokenProcessPool:
        _fail(f"a worker died (most often out of memory): try a smaller --jobs than {jobs}")
    found = attribute(results)
    verdict = gate(
        found.counts,
        baseline=known,
        unmatched=len(found.unmatched),
        full_run=full_run,
        comparable=comparable,
    )
    typer.echo(f"{len(instants)} instants, fingerprint {instants_fingerprint(instants)}")
    typer.echo(f"rows -> {diff_dir}/, {impact_dir}/; {time.monotonic() - started:,.0f}s")

    def show(c: ImpactCount) -> str:
        return f"price {c.price:,} cik {c.cik:,}"

    def report() -> None:
        for key in verdict.new:
            typer.echo(f"  new: {key}  {show(found.counts[key])}")
        for key in verdict.grew:
            typer.echo(f"  grew: {key}  {show(found.counts[key])}, baseline {show(known[key])}")
        for key in verdict.fixed:
            typer.echo(f"  no impact left (--prune-baseline): {key}  baseline {show(known[key])}")
        for key in verdict.shrank:
            typer.echo(
                f"  shrank (--prune-baseline lowers it): {key}  {show(found.counts[key])}, "
                f"baseline {show(known[key])}"
            )
        for cell in found.unmatched[:20]:
            typer.echo(
                f"  unmatched: {cell.as_of.astimezone(ET):%Y-%m-%d %H:%M} {cell.what} "
                f"{cell.owner} {cell.day} (rows name {', '.join(cell.segments) or 'nothing'})"
            )
        if len(found.unmatched) > 20:
            typer.echo(f"  ... {len(found.unmatched) - 20:,} more unmatched cells")

    if rebase:
        if verdict.new or verdict.unmatched:
            report()
            _fail("new or unmatched impact: review it before rebasing the baseline")
        try:
            changes = rebase_impact_baseline(baseline, counts=found.counts, fingerprint=fixed_print)
        except ValueError as error:
            _fail(str(error))
        # lowered lines first; the ones to read (a number up, a key gone) last, apart
        kept = [(k, o, n) for k, o, n in changes if n is not None]
        rose = [(k, o, n) for k, o, n in kept if n.price > o.price or n.cik > o.cik]
        for key, old, new in kept:
            if (key, old, new) not in rose:
                typer.echo(f"  rebased: {key}  baseline {show(old)} -> {show(new)}")
        for key, old, new in rose:
            typer.echo(f"  rose: {key}  baseline {show(old)} -> {show(new)}")
        for key, old, new in changes:
            if new is None:
                typer.echo(f"  removed: {key}  baseline {show(old)}")
        typer.echo(
            f"rebased {baseline} onto {instants_file} ({fixed_print}): {len(changes):,} lines "
            "changed; read them before committing"
        )
        return
    if prune and (verdict.new or verdict.grew or verdict.unmatched):
        report()
        _fail("new, grown or unmatched impact: review it before pruning the baseline")
    if prune and (verdict.fixed or verdict.shrank):
        try:
            removed, tightened = prune_impact_baseline(
                baseline,
                remove=verdict.fixed,
                tighten={key: found.counts[key] for key in verdict.shrank},
                fingerprint=fixed_print,
            )
        except ValueError as error:
            _fail(str(error))
        typer.echo(f"pruned {baseline}: {removed:,} keys removed, {tightened:,} lowered")
        verdict = dataclasses.replace(verdict, fixed=[], shrank=[])
    typer.echo(
        f"{len(known):,} baseline keys, {len(found.counts):,} keys with impact; "
        f"{len(verdict.new):,} new, {len(verdict.grew):,} grew, {verdict.unmatched:,} unmatched "
        f"cells"
        + (
            f", {len(verdict.fixed):,} without impact, {len(verdict.shrank):,} shrank"
            if full_run
            else " (partial run: fixed and shrank not checked)"
            if comparable
            else " (grew, fixed and shrank not checked)"
        )
    )
    report()
    if drifted:
        typer.echo("instants changed since the baseline was summed: run --rebase-baseline")
        raise typer.Exit(code=2)
    if not verdict.ok:
        raise typer.Exit(code=2)
    typer.echo("every leak impact within the baseline and named by a key")


def _docs_dirs(data_dir: Path | None) -> tuple[Path, Path]:
    root = data_dir or default_data_dir()
    return root / "raw" / "sec" / "filings", root / "docs" / "sec"


@docs_app.command("sync")
def docs_sync(
    since: Annotated[
        datetime, typer.Option("--since", help="First filing date to store.")
    ] = datetime(2023, 1, 1),
    tickers: Annotated[
        str | None,
        typer.Option("--tickers", help="Comma-separated companies. Default: every company entity."),
    ] = None,
    db: Annotated[
        Path | None,
        typer.Option("--db", help="SQLite file to write. Defaults to $ASOF_DATA_DIR/meta.db."),
    ] = None,
    data_dir: Annotated[
        Path | None, typer.Option("--data-dir", help="Defaults to $ASOF_DATA_DIR.")
    ] = None,
) -> None:
    """Store the earnings 8-Ks, 6-Ks and periodic reports of every company since --since.

    Raw files are written once and reused; a company is committed as soon as it is done, so an
    interrupted run picks up where it stopped. Needs $ASOF_SEC_USER_AGENT.
    """
    user_agent = os.environ.get(SEC_USER_AGENT_ENV)
    if not user_agent:
        _fail(f"Set {SEC_USER_AGENT_ENV} to a contact address; the SEC requires one.")
    raw_dir, text_dir = _docs_dirs(data_dir)
    master_path = market_root(data_dir or default_data_dir()) / MASTER_FILE
    if not master_path.exists():
        _fail(f"{master_path}: no security master; run `asof market identity` first")
    master = pq.read_table(master_path)
    market = market_root(data_dir or default_data_dir())
    con = duckdb.connect()
    now = datetime.now(UTC)

    def traded(symbol: str, start: date, end: date | None) -> tuple[date, date] | None:
        bars = visible_bars(
            con,
            market,
            "1Day",
            now,
            symbols=[symbol],
            start=datetime.combine(start, datetime.min.time(), tzinfo=ET),
            end=datetime.combine(end, datetime.min.time(), tzinfo=ET) if end else None,
            resolve=False,
        )
        days = [t.astimezone(ET).date() for t in bars.column("t").to_pylist()]
        return (min(days), max(days)) if days else None

    fetch = retrying(rate_limited(make_fetch({"User-Agent": user_agent}), REQUESTS_PER_SECOND))
    engine = make_engine(db or default_db_path())
    init_db(engine)
    wanted_tickers = set(_parse_symbols(tickers)) if tickers else None
    totals = [0, 0, 0, 0]
    errors: list[str] = []
    try:
        with Session(engine) as session:
            companies = [
                e
                for e in session.exec(
                    select(Entity).where(Entity.kind == EntityKind.COMPANY).order_by(Entity.ticker)
                ).all()
                if wanted_tickers is None or e.ticker in wanted_tickers
            ]
            if wanted_tickers is not None:
                unknown = wanted_tickers - {e.ticker for e in companies}
                if unknown:
                    _fail(f"Not company entities: {', '.join(sorted(unknown))}")
            for n, company in enumerate(companies, 1):
                assert company.cik is not None
                segments = security_segments(master, company.ticker)
                if not segments:
                    typer.echo(f"{company.ticker}: no open master segment, only CIK {company.cik}")
                windows = cik_windows(company.cik, trading_segments(company.cik, segments, traded))
                started = time.monotonic()
                try:
                    result = sync_company(
                        session,
                        fetch,
                        entity=company,
                        windows=windows,
                        since=since.date(),
                        raw_dir=raw_dir,
                        text_dir=text_dir,
                        now=now,
                    )
                except Throttled as error:
                    session.rollback()
                    _fail(f"{company.ticker}: {error}. Earlier companies are stored.")
                except (RuntimeError, httpx.HTTPError, ValueError) as error:
                    # The listing itself failed; the company is retried on the next run.
                    session.rollback()
                    errors.append(f"{company.ticker}: {error}")
                    continue
                session.commit()
                totals = [
                    totals[0] + result.added,
                    totals[1] + result.skipped,
                    totals[2] + result.files,
                    totals[3] + result.empty,
                ]
                errors.extend(f"{company.ticker} {e}" for e in result.errors)
                ciks = ",".join(str(w.cik) for w in windows)
                typer.echo(
                    f"[{n}/{len(companies)}] {company.ticker:<6} ciks {ciks:<16} "
                    f"+{result.added} docs, {result.skipped} known, {result.files} files, "
                    f"{result.empty} empty, {len(result.errors)} errors, "
                    f"{time.monotonic() - started:.0f}s"
                )
    finally:
        engine.dispose()
        con.close()
    for error in errors:
        typer.echo(f"error: {error}")
    typer.echo(
        f"{totals[0]} documents added, {totals[1]} already stored, {totals[2]} files "
        f"downloaded, {totals[3]} empty files skipped, {len(errors)} errors"
    )
    if errors:
        raise typer.Exit(code=2)


@docs_app.command("audit")
def docs_audit(
    exceptions: Annotated[Path, typer.Option(help="Reviewed coverage gaps.")] = Path(
        "configs/docs_exceptions.yaml"
    ),
    since: Annotated[
        datetime, typer.Option("--since", help="First filing date the sync stored.")
    ] = datetime(2023, 1, 1),
    db: Annotated[
        Path | None,
        typer.Option("--db", help="SQLite file to read. Defaults to $ASOF_DATA_DIR/meta.db."),
    ] = None,
    data_dir: Annotated[
        Path | None, typer.Option("--data-dir", help="Defaults to $ASOF_DATA_DIR.")
    ] = None,
) -> None:
    """Check that every stored citation can be re-read and nothing was public too early.

    Exit 0 when clean; 2 when a chunk, text file or availability time is wrong, a filing on a
    company's saved EDGAR listing is not stored, or a company lacks earnings or periodic
    filings outside the reviewed exceptions.
    """
    raw_dir, text_dir = _docs_dirs(data_dir)
    try:
        excused = load_coverage_exceptions(exceptions)
    except ValueError as error:
        _fail(str(error))
    engine = make_engine(db or default_db_path())
    try:
        with Session(engine) as session:
            problems = audit_documents(session, text_dir) + audit_filings(session, raw_dir)
            unindexed = chunk_index_mismatch(session)
            if unindexed is not None:
                problems.append(
                    f"chunk index: {unindexed:+,} chunks not indexed, run `asof evidence index`"
                )
            elif (broken := chunk_index_integrity(session)) is not None:
                problems.append(f"chunk index: {broken}, run `asof evidence index`")
            session.rollback()
            problems += missing_filings(session, raw_dir, since.date())
            gaps = [f"{ticker}:{kind}" for ticker, kind in coverage_gaps(session)]
            gaps += [
                f"{ticker}:periodic-gap:{later.isoformat()}"
                for ticker, _, later in periodic_gaps(session)
            ]
            gaps += [
                f"{ticker}:periodic-stale:{latest.isoformat()}"
                for ticker, latest in stale_reports(session, datetime.now(UTC).date())
            ]
    finally:
        engine.dispose()
    for problem in problems:
        typer.echo(problem)
    unexcused = [gap for gap in gaps if gap not in excused]
    for gap in unexcused:
        typer.echo(f"coverage: {gap} is missing")
    for gap in gaps:
        if gap in excused:
            typer.echo(f"coverage: {gap} excused ({excused[gap]})")
    for stale in sorted(set(excused) - set(gaps)):
        typer.echo(f"stale exception: {stale} is covered now")
    typer.echo(f"{len(problems)} integrity problems, {len(unexcused)} unexcused coverage gaps")
    if problems or unexcused:
        raise typer.Exit(code=2)


def _db_option_help() -> str:
    return "SQLite file. Defaults to $ASOF_DATA_DIR/meta.db."


@evidence_app.command("index")
def evidence_index(
    force: Annotated[bool, typer.Option(help="Rebuild even when the index checks clean.")] = False,
    db: Annotated[Path | None, typer.Option("--db", help=_db_option_help())] = None,
) -> None:
    """Make sure the full-text index covers every chunk; rebuild it when it does not.

    Creating the index fills it, and new chunks are indexed on insert, so a rebuild is needed
    only after something went wrong (or with --force).
    """
    engine = make_engine(db or default_db_path())
    try:
        init_db(engine)
        with Session(engine) as session:
            stale = chunk_index_mismatch(session) is not None
            stale = stale or chunk_index_integrity(session) is not None
            session.rollback()
            started = time.monotonic()
            if stale or force:
                indexed = rebuild_chunk_index(session)
                session.commit()
                typer.echo(f"indexed {indexed:,} chunks in {time.monotonic() - started:.0f}s")
            mismatch = chunk_index_mismatch(session)
            broken = chunk_index_integrity(session)
            session.rollback()
    finally:
        engine.dispose()
    if mismatch is not None:
        _fail(f"index and chunk table differ by {mismatch:+,} rows")
    if broken is not None:
        _fail(f"index does not match the chunk text: {broken}")
    typer.echo("index complete")


def _entity_by_ticker(session: Session, ticker: str) -> Entity:
    # A command-line convenience: today's ticker of a watchlist entity. Code that reads a
    # historical moment resolves securities to entities by as_of instead.
    rows = session.exec(select(Entity).where(Entity.ticker == ticker.upper())).all()
    if len(rows) != 1:
        _fail(f"{ticker}: {len(rows)} entities have this ticker, expected 1")
    return rows[0]


@evidence_app.command("search")
def evidence_search(
    query: Annotated[str, typer.Argument(help="Plain words; operators are not interpreted.")],
    as_of: Annotated[
        datetime,
        typer.Option("--as-of", help="Only what was public by then (naive = New York time)."),
    ],
    ticker: Annotated[str | None, typer.Option(help="Restrict to one company.")] = None,
    form: Annotated[
        list[str] | None, typer.Option("--form", help="Base form, e.g. 8-K, 10-Q; repeatable.")
    ] = None,
    limit: Annotated[int, typer.Option(min=1)] = 10,
    db: Annotated[Path | None, typer.Option("--db", help=_db_option_help())] = None,
) -> None:
    """Rank the chunks that were public at --as-of by BM25 over that moment's documents."""
    as_of = _as_of_option(as_of)
    engine = make_engine(db or default_db_path())
    try:
        with Session(engine) as session:
            entity_id = _entity_by_ticker(session, ticker).id if ticker else None
            started = time.monotonic()
            try:
                hits = search(
                    session, as_of, query, entity_id=entity_id, base_forms=form, limit=limit
                )
            except RuntimeError as error:
                _fail(str(error))
            elapsed = time.monotonic() - started
    finally:
        engine.dispose()
    for hit in hits:
        public = hit.available_at.astimezone(ET).strftime("%Y-%m-%d %H:%M ET")
        typer.echo(
            f"{hit.score:7.3f}  chunk {hit.chunk_id}  {hit.doc_type:<6} {public}  "
            f"{hit.section_path}"
        )
        typer.echo(f"         {' '.join(hit.text.split())[:200]}")
    typer.echo(f"{len(hits)} hits in {elapsed:.2f}s")


@evidence_app.command("validate")
def evidence_validate(
    draft_file: Annotated[Path, typer.Argument(help="JSON: as_of, ticker, citations[].")],
    db: Annotated[Path | None, typer.Option("--db", help=_db_option_help())] = None,
) -> None:
    """Check a decision draft's citations. Exit 0 when every citation holds, 2 otherwise.

    Each citation is {claim, evidence_kind, chunk_id, quote}; as_of is ISO 8601 with offset.
    """
    try:
        raw = json.loads(draft_file.read_text())
        unknown = set(raw) - {"as_of", "ticker", "citations"} if isinstance(raw, dict) else set()
        if not isinstance(raw, dict) or unknown:
            raise ValueError(f"expected an object with as_of, ticker, citations; got {unknown}")
        as_of = datetime.fromisoformat(raw["as_of"])
        citations = [ClaimCitation(**c) for c in raw["citations"]]
        for c in citations:
            if not (
                isinstance(c.claim, str)
                and isinstance(c.evidence_kind, str)
                and (c.chunk_id is None or type(c.chunk_id) is int)
                and (c.quote is None or isinstance(c.quote, str))
                and (c.evidence_ref is None or isinstance(c.evidence_ref, str))
            ):
                raise TypeError(f"citation fields have the wrong types: {c}")
        ticker = raw["ticker"]
        if not isinstance(ticker, str):
            raise TypeError(f"ticker must be a string, got {ticker!r}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        _fail(f"{draft_file}: not a valid draft ({type(error).__name__}: {error})")
    engine = make_engine(db or default_db_path())
    try:
        with Session(engine) as session:
            draft = DecisionDraft(
                as_of=as_of, entity_id=_entity_by_ticker(session, ticker).id, citations=citations
            )
            result = validate(session, draft)
    except ValueError as error:
        _fail(str(error))
    finally:
        engine.dispose()
    for failure in result.failures:
        typer.echo(f"citation {failure.index}: {failure.reason} - {failure.detail}")
    if not result.ok:
        typer.echo(f"rejected: {len(result.failures)} of {len(draft.citations)} citations fail")
        raise typer.Exit(code=2)
    typer.echo(f"ok: {len(draft.citations)} citations hold")


scan_app = typer.Typer(help="Entry scans run at a session's close.", no_args_is_help=True)
app.add_typer(scan_app, name="scan")


@scan_app.command("bx")
def scan_bx(
    day: Annotated[datetime, typer.Option("--date", help="Session; read at its 20:00 close.")],
    overrides: Annotated[
        list[str] | None, typer.Option("--set", help="Override a scan setting, e.g. variant=long.")
    ] = None,
    top: Annotated[int, typer.Option(min=0, help="Candidates to print.")] = 30,
    strategies_dir: Annotated[Path, typer.Option("--strategies-dir")] = Path("strategies"),
    data_dir: Annotated[
        Path | None, typer.Option("--data-dir", help="Defaults to $ASOF_DATA_DIR.")
    ] = None,
) -> None:
    """B-Xtrender entry candidates of one session: liquid common stocks whose last completed
    state bar (weekly, or monthly with variant=long) and daily histogram both rose."""
    root = _market_root(data_dir)
    try:
        cfg = load_scan_config(strategies_dir / "bx_trender" / "scan.yaml", overrides or [])
    except (FileNotFoundError, ValueError) as error:
        _fail(str(error))
    as_of = close_of(day.date())
    started = time.monotonic()
    con = duckdb.connect()
    try:
        session, found, eligible = scan_and_pool(con, root, as_of, cfg)
    except (FileNotFoundError, ValueError) as error:
        _fail(str(error))
    finally:
        con.close()
    state = "weekly" if cfg.variant == "mid" else "monthly"
    read = session or "no final session"
    typer.echo(
        f"{len(found)} {cfg.variant} candidates on {read} out of {len(eligible)} warm liquid "
        f"common stocks ({time.monotonic() - started:,.1f}s); state = {state}"
    )
    typer.echo(
        f"  {'#':>3} {'symbol':<8} {'security':<8} {'state':>7} {'prev':>7} {'daily':>7} "
        f"{'prev':>7}  {'state bar':<10} {'bars':>4}  {'since':<10} {'ADV $M':>8}"
    )
    for c in found[:top]:
        typer.echo(
            f"  {c.rank:>3} {c.symbol:<8} {c.security_id:<8} {c.state_bx:>7.2f} "
            f"{c.state_bx_prev:>7.2f} {c.daily_bx:>7.2f} {c.daily_bx_prev:>7.2f}  "
            f"{c.state_complete_on!s:<10} {c.state_bars:>4}  {c.bx_since!s:<10} "
            f"{c.adv_usd / 1e6:>8,.0f}"
        )


replay_app = typer.Typer(
    help="Replay a strategy over history and report what it would have done.",
    no_args_is_help=True,
)
app.add_typer(replay_app, name="replay")


def _benchmark(root: Path, start: date, end: date, symbol: str = "SPY") -> tuple[str, float] | None:
    """Price return of buying ``symbol`` at the first session's open and holding to the last
    session's close (dividends not included)."""
    con = duckdb.connect()
    try:
        bars = visible_bars(
            con,
            root,
            "1Day",
            close_of(end),
            symbols=[symbol],
            start=datetime.combine(start, datetime.min.time(), tzinfo=ET),
            end=close_of(end),
        ).to_pylist()
    finally:
        con.close()
    if not bars:
        return None
    return f"{symbol} buy-and-hold (price only)", bars[-1]["close"] / bars[0]["open"] - 1


@replay_app.command("run")
def replay_run(
    strategy: Annotated[
        str, typer.Option("--strategy", help="Directory name under --strategies-dir.")
    ],
    start: Annotated[datetime, typer.Option("--start", formats=["%Y-%m-%d"])],
    end: Annotated[datetime, typer.Option("--end", formats=["%Y-%m-%d"])],
    overrides: Annotated[
        list[str] | None,
        typer.Option("--set", help="Override one setting, e.g. execute.stop.trail_atr=1.0."),
    ] = None,
    open_holdout: Annotated[
        bool,
        typer.Option(
            "--open-holdout",
            help="Allow dates in the sealed holdout (configs/evaluation.yaml); once per strategy.",
        ),
    ] = False,
    strategies_dir: Annotated[Path, typer.Option("--strategies-dir")] = Path("strategies"),
    data_dir: Annotated[
        Path | None, typer.Option("--data-dir", help="Defaults to $ASOF_DATA_DIR.")
    ] = None,
    db: Annotated[
        Path | None, typer.Option("--db", help="Defaults to $ASOF_DATA_DIR/meta.db.")
    ] = None,
) -> None:
    """Replay a strategy session by session. Every distinct --set combination is registered as
    a strategy version, so it counts as one more attempt."""
    first, last = start.date(), end.date()
    if last < first:
        _fail(f"--end {last} is before --start {first}")
    path = strategies_dir / strategy
    try:
        apply_overrides(load_any_strategy(path), overrides or [])
    except (FileNotFoundError, ValueError) as error:
        _fail(f"Strategy {strategy!r}: {error}")
    root = _market_root(data_dir)
    engine = make_engine(db or default_db_path())
    con = duckdb.connect()
    try:
        init_db(engine)
        with Session(engine) as session:
            try:
                run_id = run_replay(
                    session, con, root, path, first, last, overrides or [],
                    open_holdout=open_holdout,
                )  # fmt: skip
            except HoldoutError as error:
                _fail(f"Refused, nothing was registered or run: {error}")
            except (FileNotFoundError, ValueError) as error:
                _fail(
                    f"Replay failed, no run was written (the parameter set is registered): {error}"
                )
            except CacheMiss as error:
                _fail(
                    "Replay failed, no run was written (the parameter set is registered): "
                    f"{error}. Run `asof judge run` over the same window first."
                )
            report = build_report(session, run_id, benchmark=_benchmark(root, first, last))
    finally:
        con.close()
        engine.dispose()
    out = (data_dir or default_data_dir()) / "reports" / f"replay-{run_id}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    typer.echo(f"run {run_id}")
    typer.echo(f"report {out}")


@replay_app.command("report")
def replay_report(
    run: Annotated[int, typer.Option("--run", help="Run id printed by `asof replay run`.")],
    db: Annotated[
        Path | None, typer.Option("--db", help="Defaults to $ASOF_DATA_DIR/meta.db.")
    ] = None,
) -> None:
    """Print the markdown report of a replay run."""
    engine = make_engine(db or default_db_path())
    try:
        init_db(engine)
        with Session(engine) as session:
            try:
                text = build_report(session, run)
            except LookupError as error:
                _fail(str(error))
    finally:
        engine.dispose()
    typer.echo(text, nl=False)


def _money(value: float | None) -> str:
    return "-" if value is None else f"{value:,.2f}"


def _r(value: float | None) -> str:
    return "-" if value is None else f"{value:+.2f}R"


def _section_lines(title: str, section: Section) -> list[str]:
    lines = [f"## {title}"]
    for name, s in (("take_all", section.base), ("llm", section.run)):
        lines.append(
            f"{name:9} decisions {s.decisions}  traded {s.traded}  "
            f"pnl {_money(s.total_pnl)}  mean {_r(s.mean_r)}"
        )
    lines.append("verdict   count  take_all traded  take_all pnl  take_all mean")
    for verdict, v in section.verdicts.items():
        lines.append(
            f"{verdict:9} {v.count:5}  {v.base_outcomes:15}  {_money(v.base_pnl):>12}  "
            f"{_r(v.base_mean_r):>13}"
        )
    lines.append("conviction  count  take_all traded  take_all win rate")
    for b in section.calibration:
        rate = "-" if b.base_win_rate is None else f"{b.base_win_rate:.0%}"
        bracket = "]" if b.high == 1.0 else ")"
        lines.append(
            f"[{b.low:.1f}, {b.high:.1f}{bracket}  {b.count:5}  {b.base_outcomes:15}  {rate:>17}"
        )
    return lines


def _comparison_text(session: Session, c: Comparison) -> str:
    # Stored as 00:00 UTC of the cutoff date: in New York that is still the evening before.
    cutoff = c.cutoff.astimezone(UTC).date().isoformat()
    lines = [
        f"take_all run {c.base_run_id} vs llm run {c.run_id}; model training cutoff {cutoff}",
        "",
        *_section_lines("all decisions", c.all),
        "",
        *_section_lines(f"before the cutoff {cutoff} (possibly contaminated)", c.before_cutoff),
        "",
        *_section_lines(f"on or after the cutoff {cutoff}", c.after_cutoff),
        "",
        f"## unmatched decisions: {len(c.unmatched)}",
    ]
    for u in c.unmatched:
        entity = session.get(Entity, u.entity_id) if u.entity_id is not None else None
        ticker = entity.ticker if entity is not None else "?"
        lines.append(f"run {u.run_id}  {ticker}  {u.as_of.astimezone(ET).date().isoformat()}")
    return "\n".join(lines) + "\n"


@replay_app.command("compare")
def replay_compare(
    base: Annotated[int, typer.Option("--base", help="Run id of the take_all replay.")],
    run: Annotated[int, typer.Option("--run", help="Run id of the llm replay.")],
    db: Annotated[
        Path | None, typer.Option("--db", help="Defaults to $ASOF_DATA_DIR/meta.db.")
    ] = None,
) -> None:
    """Compare an llm replay with the take_all replay of the same window: each verdict scored by
    what take_all earned on the same candidates, split at the model's training cutoff."""
    engine = make_engine(db or default_db_path())
    try:
        init_db(engine)
        with Session(engine) as session:
            try:
                text = _comparison_text(session, compare_runs(session, base, run))
            except (LookupError, ValueError) as error:
                _fail(str(error))
    finally:
        engine.dispose()
    typer.echo(text, nl=False)


judge_app = typer.Typer(help="Ask the llm judge and cache its answers.", no_args_is_help=True)
app.add_typer(judge_app, name="judge")


@judge_app.command("run")
def judge_run(
    strategy: Annotated[
        str, typer.Option("--strategy", help="Directory name under --strategies-dir.")
    ],
    start: Annotated[datetime, typer.Option("--start", formats=["%Y-%m-%d"])],
    end: Annotated[datetime, typer.Option("--end", formats=["%Y-%m-%d"])],
    max_cost: Annotated[
        float, typer.Option("--max-cost", help="Stop before the next candidate past this, USD.")
    ],
    limit: Annotated[
        int | None, typer.Option("--limit", help="Judge at most this many candidates.")
    ] = None,
    overrides: Annotated[
        list[str] | None,
        typer.Option("--set", help="Override one setting, e.g. judge.effort=medium."),
    ] = None,
    open_holdout: Annotated[
        bool,
        typer.Option(
            "--open-holdout",
            help="Allow dates in the sealed holdout (configs/evaluation.yaml); once per strategy.",
        ),
    ] = False,
    strategies_dir: Annotated[Path, typer.Option("--strategies-dir")] = Path("strategies"),
    data_dir: Annotated[
        Path | None, typer.Option("--data-dir", help="Defaults to $ASOF_DATA_DIR.")
    ] = None,
    db: Annotated[
        Path | None, typer.Option("--db", help="Defaults to $ASOF_DATA_DIR/meta.db.")
    ] = None,
) -> None:
    """Judge every candidate of a window with the llm judge (judge.kind=llm is forced) and cache
    the answers, so `asof replay run --set judge.kind=llm` over the window runs offline. Paid
    calls stop once --max-cost is reached (exit 1); cached answers cost nothing."""
    first, last = start.date(), end.date()
    if last < first:
        _fail(f"--end {last} is before --start {first}")
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key.strip():
        _fail("OPENROUTER_API_KEY is not set; nothing was sent.")
    path = strategies_dir / strategy
    try:
        cfg = apply_overrides(load_strategy(path), [*(overrides or []), "judge.kind=llm"])
    except (FileNotFoundError, ValueError) as error:
        _fail(f"Strategy {strategy!r}: {error}")
    model = OpenRouterClient(key)
    root = _market_root(data_dir)
    engine = make_engine(db or default_db_path())
    con = duckdb.connect()
    try:
        init_db(engine)
        with Session(engine) as session:
            try:
                result = judge_window(
                    session, con, root, cfg, first, last, model.complete,
                    max_cost=max_cost, limit=limit, open_holdout=open_holdout,
                )  # fmt: skip
            except HoldoutError as error:
                _fail(f"Refused, nothing was sent: {error}")
            except ValueError as error:
                _fail(f"Judging stopped: {error}. Answers paid for so far are cached.")
    finally:
        con.close()
        engine.dispose()
    for verdict in ("take", "pass", "no_call"):
        typer.echo(f"{verdict} {sum(1 for o in result.outcomes if o.verdict == verdict)}")
    typer.echo(f"fresh calls {result.fresh_calls}")
    typer.echo(f"cached calls {result.cached_calls}")
    typer.echo(f"spent ${result.spent:.4f}")
    if result.error is not None:
        typer.echo(f"error: {result.error}; answers paid for so far are cached, rerun to go on")
        raise typer.Exit(1)
    if result.stopped:
        typer.echo(
            f"stopped: --max-cost {max_cost} reached in this run (the cap is per run; cached "
            "answers are free); rerun to continue from the cache"
        )
        raise typer.Exit(1)


@app.command("benchmark")
def benchmark_cmd(
    start: Annotated[datetime, typer.Option("--start", formats=["%Y-%m-%d"])],
    end: Annotated[datetime, typer.Option("--end", formats=["%Y-%m-%d"])],
    holdings: Annotated[
        bool, typer.Option("--holdings", help="List the basket's members at each rebalance.")
    ] = False,
    data_dir: Annotated[
        Path | None, typer.Option("--data-dir", help="Defaults to $ASOF_DATA_DIR.")
    ] = None,
) -> None:
    """The protocol's benchmarks over [start, end]: index buy-and-hold with dividends
    reinvested and the monthly top dollar-volume basket (configs/evaluation.yaml)."""
    root = _market_root(data_dir)
    con = duckdb.connect()
    try:
        curves = benchmarks(con, root, load_protocol(), start.date(), end.date())
    except ValueError as error:
        _fail(str(error))
    finally:
        con.close()
    for c in curves:
        typer.echo(
            f"{c.label}: total {c.total_return:+.2%}, annualized {c.annualized_return:+.2%}, "
            f"max drawdown {c.max_drawdown:.2%}, {len(c.sessions)} sessions"
        )
        if holdings:
            for r in c.rebalances:
                typer.echo(f"  {r.session} " + " ".join(symbol for _, symbol in r.members))


gate_app = typer.Typer(help="The evaluation gate (configs/evaluation.yaml).", no_args_is_help=True)
app.add_typer(gate_app, name="gate")


def _family_config(
    session: Session, runs: list[int], strategies_dir: Path, current_strategy: bool
) -> dict[int, AnyStrategyConfig]:
    """Each run's config, which gives its random-entry pool and slippage: the strategy files
    must still be the versions the runs replayed (or ``current_strategy``), all of one kind."""
    cfgs = {}
    for run_id in runs:
        run = session.get(Run, run_id)
        if run is None or run.strategy_id is None:
            _fail(f"no replay run {run_id}")
        version = (
            session.get(StrategyVersion, run.strategy_version_id)
            if run.strategy_version_id is not None
            else None
        )
        overrides = overrides_of_summary(version.summary if version else None)
        path = strategies_dir / run.strategy_id
        try:
            cfgs[run_id] = apply_overrides(load_any_strategy(path), overrides)
        except ValueError as error:  # pydantic's ValidationError and a YAML error too
            _fail(f"run {run_id}: its overrides {version.summary if version else ''!r} do not "
                  f"apply to {path}: {error}")  # fmt: skip
        if version is None or content_hash(path, overrides) != version.content_hash:
            if not current_strategy:
                _fail(
                    f"run {run_id}: {path} is no longer the version it replayed; "
                    "--current-strategy uses today's files"
                )
            typer.echo(f"run {run_id}: using today's {path}", err=True)
    kinds = sorted({c.strategy.kind for c in cfgs.values()})
    if len(kinds) > 1:
        _fail(f"the runs mix strategy kinds: {', '.join(kinds)}")
    return cfgs


@gate_app.command("null")
def gate_null(
    runs: Annotated[
        list[int], typer.Option("--run", help="A replay run; several runs form one family.")
    ],
    permutations: Annotated[
        int | None, typer.Option("--permutations", help="Defaults to the protocol's.")
    ] = None,
    seed: Annotated[int, typer.Option("--seed")] = 0,
    current_strategy: Annotated[
        bool,
        typer.Option(
            "--current-strategy",
            help="Use today's strategy files for runs replayed under another version.",
        ),
    ] = False,
    strategies_dir: Annotated[Path, typer.Option("--strategies-dir")] = Path("strategies"),
    data_dir: Annotated[
        Path | None, typer.Option("--data-dir", help="Defaults to $ASOF_DATA_DIR.")
    ] = None,
    db: Annotated[
        Path | None, typer.Option("--db", help="Defaults to $ASOF_DATA_DIR/meta.db.")
    ] = None,
) -> None:
    """Each run's closed legs against random entries: the same dates and exits with a security
    drawn from what the strategy's scanner could have bought that day without its signal.
    p adjusted is corrected for every run given (max statistic). The pool's thresholds and the
    slippage come from the runs' strategy directory."""
    root = _market_root(data_dir)
    engine = make_engine(db or default_db_path())
    con = duckdb.connect()
    try:
        with Session(engine) as session:
            cfg = _family_config(session, runs, strategies_dir, current_strategy)
            try:
                results = null_test(
                    session, con, root, runs, cfg=cfg,
                    permutations=permutations or load_protocol().permutations, seed=seed,
                )  # fmt: skip
            except (LookupError, ValueError) as error:
                _fail(str(error))
    finally:
        con.close()
        engine.dispose()
    typer.echo(f"{'run':>5} {'legs':>5} {'mean':>8} {'random':>8} {'excess':>8} {'z':>6} "
               f"{'p':>7} {'p adj':>7}")  # fmt: skip
    for run_id, r in results.items():
        typer.echo(
            f"{run_id:>5} {r.trades:>5} {r.mean:>+8.2%} {r.null_mean:>+8.2%} {r.excess:>+8.2%} "
            f"{r.z:>6.2f} {r.p_raw:>7.4f} {r.p_adjusted:>7.4f}"
        )


@gate_app.command("entry")
def gate_entry(
    runs: Annotated[
        list[int], typer.Option("--run", help="A replay run; several runs form one family.")
    ],
    holds: Annotated[
        list[int], typer.Option("--hold", help="Sessions each entry is held; give several.")
    ],
    permutations: Annotated[
        int | None, typer.Option("--permutations", help="Defaults to the protocol's.")
    ] = None,
    seed: Annotated[int, typer.Option("--seed")] = 0,
    current_strategy: Annotated[
        bool,
        typer.Option(
            "--current-strategy",
            help="Use today's strategy files for runs replayed under another version.",
        ),
    ] = False,
    strategies_dir: Annotated[Path, typer.Option("--strategies-dir")] = Path("strategies"),
    data_dir: Annotated[
        Path | None, typer.Option("--data-dir", help="Defaults to $ASOF_DATA_DIR.")
    ] = None,
    db: Annotated[
        Path | None, typer.Option("--db", help="Defaults to $ASOF_DATA_DIR/meta.db.")
    ] = None,
) -> None:
    """Each run's entries in the protocol's development dates against random entries, both held
    a fixed number of sessions: the exit no longer follows the real security's path, so what
    is left is the entry. p adjusted is corrected for every (run, hold) row of this call only;
    the interval resamples blocks of hold sessions. Dropped: fixed exit after dev (beyond),
    real security without an exit price (unpriced)."""
    protocol = load_protocol()
    root = _market_root(data_dir)
    engine = make_engine(db or default_db_path())
    con = duckdb.connect()
    try:
        with Session(engine) as session:
            cfg = _family_config(session, runs, strategies_dir, current_strategy)
            try:
                results = entry_test(
                    session, con, root, runs, cfg=cfg, holds=holds, dev=protocol.dev,
                    permutations=permutations or protocol.permutations, seed=seed,
                )  # fmt: skip
            except (LookupError, ValueError) as error:
                _fail(str(error))
    finally:
        con.close()
        engine.dispose()
    typer.echo(f"{'run':>5} {'hold':>5} {'entries':>7} {'mean':>8} {'random':>8} {'excess':>8} "
               f"{'ci low':>8} {'ci high':>8} {'sd ratio':>8} {'p':>7} {'p adj':>7} "
               f"{'beyond':>6} {'unpriced':>8}")  # fmt: skip
    for h in results:
        r = h.result
        typer.echo(
            f"{h.run_id:>5} {h.hold:>5} {r.trades:>7} {r.mean:>+8.2%} {r.null_mean:>+8.2%} "
            f"{r.excess:>+8.2%} {h.ci[0]:>+8.2%} {h.ci[1]:>+8.2%} {h.sd_ratio:>8.2f} "
            f"{r.p_raw:>7.4f} {r.p_adjusted:>7.4f} {h.beyond_window:>6} {h.unpriced:>8}"
        )
    # the correction covers this call's cells only: another call is another family
    typer.echo("family: " + " ".join(f"{h.run_id}:{h.hold}" for h in results))


@gate_app.command("report")
def gate_report(
    runs: Annotated[
        list[int], typer.Option("--run", help="A replay run; several runs form one family.")
    ],
    permutations: Annotated[
        int | None, typer.Option("--permutations", help="Defaults to the protocol's.")
    ] = None,
    seed: Annotated[int, typer.Option("--seed")] = 0,
    current_strategy: Annotated[
        bool,
        typer.Option(
            "--current-strategy",
            help="Use today's strategy files for runs replayed under another version.",
        ),
    ] = False,
    strategies_dir: Annotated[Path, typer.Option("--strategies-dir")] = Path("strategies"),
    data_dir: Annotated[
        Path | None, typer.Option("--data-dir", help="Defaults to $ASOF_DATA_DIR.")
    ] = None,
    db: Annotated[
        Path | None, typer.Option("--db", help="Defaults to $ASOF_DATA_DIR/meta.db.")
    ] = None,
    from_: Annotated[
        datetime | None,
        typer.Option(
            "--from",
            formats=["%Y-%m-%d"],
            help="Measure from this development date on (a variant that cannot trade before it).",
        ),
    ] = None,
    report: Annotated[
        Path | None,
        typer.Option("--report", help="Defaults to <data-dir>/reports/gate-<runs>.md."),
    ] = None,
) -> None:
    """The gate's measurements of a family of runs over the protocol's development dates, side
    by side and not scored: random entries with the max-statistic correction, walk-forward, thin
    edge, the benchmarks, the minimum detectable edge. Writes a markdown report."""
    base = data_dir or default_data_dir()
    root = _market_root(data_dir)
    engine = make_engine(db or default_db_path())
    init_db(engine)
    try:
        protocol = load_protocol()
    except ValueError as error:  # a pydantic ValidationError is a ValueError
        _fail(f"configs/evaluation.yaml: {error}")
    if from_ is not None:
        try:
            protocol = protocol.from_date(from_.date())
        except ValueError as error:
            _fail(str(error))
    con = duckdb.connect()
    try:
        with Session(engine) as session:
            cfg = _family_config(session, runs, strategies_dir, current_strategy)
            try:
                result = gate_measure(
                    session, con, root, runs, cfg=cfg, protocol=protocol,
                    permutations=permutations or protocol.permutations, seed=seed,
                )  # fmt: skip
            except (LookupError, ValueError) as error:
                _fail(str(error))
            text = render_report(session, result, protocol)
    finally:
        con.close()
        engine.dispose()
    suffix = f"-from-{from_:%Y%m%d}" if from_ is not None else ""
    path = report or base / "reports" / f"gate-{'-'.join(str(r) for r in runs)}{suffix}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    best = result.variants[result.candidate]
    typer.echo(
        f"candidate run {result.candidate}: {best.excess:+.2%} a leg over random entries "
        f"(p adj {best.p_adjusted:.3f}), walk-forward {result.walk_forward_positive} of "
        f"{result.walk_forward_scorable} evaluable windows positive"
    )
    typer.echo(f"report {path}")


@gate_app.command("compare")
def gate_compare(
    run_id: Annotated[int, typer.Option("--run", help="A replay run with daily equity.")],
    data_dir: Annotated[
        Path | None, typer.Option("--data-dir", help="Defaults to $ASOF_DATA_DIR.")
    ] = None,
    db: Annotated[
        Path | None, typer.Option("--db", help="Defaults to $ASOF_DATA_DIR/meta.db.")
    ] = None,
) -> None:
    """A run's daily marked-to-market equity against the protocol's benchmarks over the same
    sessions: return, drawdown and Sharpe, how invested it was, and the strategy minus each
    benchmark (shown, not scored)."""
    root = _market_root(data_dir)
    engine = make_engine(db or default_db_path())
    init_db(engine)
    con = duckdb.connect()
    protocol = load_protocol()
    try:
        with Session(engine) as session:
            v = compare_run(session, con, root, run_id, protocol)
    except (LookupError, ValueError) as error:
        _fail(str(error))
    finally:
        con.close()
        engine.dispose()

    def sharpe(value: float | None) -> str:
        return f"{value:>7.2f}" if value is not None else f"{'-':>7}"

    typer.echo(f"run {v.run_id}: {v.start} .. {v.end}, {v.strategy.sessions} sessions")
    rows = [("strategy", v.strategy), *v.benchmarks]
    width = max(len(label) for label, _ in rows)
    typer.echo(f"{'':<{width}} {'total':>9} {'annualized':>11} {'max dd':>8} {'sharpe':>7}")
    for label, m in rows:
        typer.echo(
            f"{label:<{width}} {m.total_return:>+9.2%} {m.annualized_return:>+11.2%} "
            f"{m.max_drawdown:>8.2%} {sharpe(m.sharpe)}"
        )
    high = protocol.comparison.high_exposure
    typer.echo(f"mean exposure {v.mean_exposure:.1%}; high-exposure share (>= {high:.0%}) "
               f"{v.high_exposure_share:.1%}")  # fmt: skip
    typer.echo("the strategy minus each benchmark (max dd above zero: the strategy fell further)")
    typer.echo(f"{'':<{width}} {'total':>11} {'max dd':>10} {'sharpe':>7}")
    for d in v.differences:
        total, drawdown = f"{d.total_return * 100:+.1f} pts", f"{d.max_drawdown * 100:+.1f} pts"
        typer.echo(f"{d.label:<{width}} {total:>11} {drawdown:>10} {sharpe(d.sharpe)}")


demo_app = typer.Typer(help="Data for the static demo page.", no_args_is_help=True)
app.add_typer(demo_app, name="demo")


@demo_app.command("build")
def demo_build(
    out: Annotated[Path, typer.Option("--out", help="Directory for the JSON files.")] = Path(
        "web/data"
    ),
    strategy: Annotated[str, typer.Option("--strategy")] = "earnings_gap",
    start: Annotated[
        datetime, typer.Option("--start", formats=["%Y-%m-%d"], help="Grid window start.")
    ] = datetime(2026, 7, 1),
    end: Annotated[datetime, typer.Option("--end", formats=["%Y-%m-%d"])] = datetime(2026, 10, 21),
    long_start: Annotated[
        datetime,
        typer.Option("--long-start", formats=["%Y-%m-%d"], help="Start of the take-all curve."),
    ] = datetime(2023, 1, 3),
    jobs: Annotated[int, typer.Option("--jobs", help="Worker processes.")] = 1,
    strategies_dir: Annotated[Path, typer.Option("--strategies-dir")] = Path("strategies"),
    data_dir: Annotated[
        Path | None, typer.Option("--data-dir", help="Defaults to $ASOF_DATA_DIR.")
    ] = None,
    db: Annotated[
        Path | None, typer.Option("--db", help="Defaults to $ASOF_DATA_DIR/meta.db.")
    ] = None,
) -> None:
    """Replay the demo grid (take-all and llm over trail, risk and min conviction) on private
    copies of the database and write the page's JSON. The research database is only read: the
    grid adds no strategy versions or runs to it. Llm answers must already be cached."""
    if jobs < 1:
        _fail("--jobs must be at least 1")
    counts = build(
        db or default_db_path(),
        _market_root(data_dir),
        strategies_dir / strategy,
        out,
        start=start.date(),
        end=end.date(),
        long_start=long_start.date(),
        jobs=jobs,
    )
    for name, n in counts.items():
        typer.echo(f"{name} {n}")
