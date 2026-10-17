"""The symbol universe for market-data backfill.

Union of Alpaca's listed assets (active and inactive, exchange-listed only), every symbol that
ever appeared in an SEC ticker snapshot, and both sides of every Alpaca name change. Keeping
inactive and historical names in is what lets a replay see the companies that later
disappeared. The name changes cover what the other two miss: Alpaca prunes old inactive names
from its asset list and the SEC's company table has no ETFs, so SPLG (renamed SPYM) was in
neither and its history was never requested.
"""

import os
from collections.abc import Collection, Iterable, Sequence
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from asof.ingest.alpaca import Asset
from asof.ingest.corporate_actions import actions_dir
from asof.ingest.sec_ticker_history import HISTORY_SCHEMA, otc_only_symbols
from asof.ingest.tickers import is_valid_symbol, normalize_ticker

ASSET_SCHEMA = pa.schema(
    [
        ("symbol", pa.string()),
        ("name", pa.string()),
        ("exchange", pa.string()),
        ("status", pa.string()),
        ("tradable", pa.bool_()),
        ("asset_class", pa.string()),
    ]
)
HISTORY_PATH = Path("symbols") / "sec_history.parquet"
NAME_CHANGES_FILE = "name_changes.parquet"
EXCHANGE_DIR = Path("raw") / "sec" / "company_tickers_exchange"


def build_universe(
    assets: Iterable[Asset],
    historical: Iterable[str],
    renamed: Iterable[str] = (),
    *,
    exclude_exchanges: frozenset[str] = frozenset({"OTC"}),
    exclude_symbols: Collection[str] = frozenset(),
) -> list[str]:
    """Exchange-listed Alpaca assets, plus SEC-history tickers not known to be OTC-only, plus
    renamed symbols (both names of a name change) unless Alpaca lists them on an excluded
    exchange. An SEC listing vouches for a symbol Alpaca files under OTC; a name change does
    not, since Alpaca records renames of its OTC assets too."""
    symbols: set[str] = set()
    excluded_by_alpaca: set[str] = set()
    for asset in assets:
        # Alpaca symbols are already in canonical spelling; only filter out its junk entries.
        if asset.exchange in exclude_exchanges:
            excluded_by_alpaca.add(asset.symbol)
        elif is_valid_symbol(asset.symbol):
            symbols.add(asset.symbol)
    for raw in historical:
        symbol = _canonical(raw)
        if symbol and symbol not in exclude_symbols:
            symbols.add(symbol)
    for raw in renamed:
        symbol = _canonical(raw)
        if symbol and symbol not in exclude_symbols and symbol not in excluded_by_alpaca:
            symbols.add(symbol)
    return sorted(symbols)


def _canonical(raw: str) -> str | None:
    """``raw`` in canonical spelling, or None when it is not a symbol at all."""
    try:
        symbol = normalize_ticker(raw)
    except ValueError:
        return None
    return symbol if is_valid_symbol(symbol) else None


def assets_dir(root: Path) -> Path:
    return root / "assets"


def save_assets_snapshot(assets: Sequence[Asset], root: Path, snapshot_date: date) -> Path:
    """One dated Parquet per listing download; earlier snapshots are never touched."""
    path = assets_dir(root) / f"snapshot_date={snapshot_date.isoformat()}" / "assets.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {name: [getattr(a, name) for a in assets] for name in ASSET_SCHEMA.names},
        schema=ASSET_SCHEMA,
    )
    tmp = path.with_name(path.name + ".tmp")
    try:
        pq.write_table(table, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def assets_snapshots(root: Path) -> list[Path]:
    directory = assets_dir(root)
    if not directory.exists():
        return []
    return sorted(directory.glob("snapshot_date=*/assets.parquet"))


def load_assets(root: Path) -> list[Asset]:
    """Every assets snapshot on disk, not just the newest: Alpaca prunes old inactive symbols
    from its list, and a symbol that was ever listed must stay known."""
    return [
        Asset.model_validate(row)
        for snapshot in assets_snapshots(root)
        for row in pq.read_table(snapshot).to_pylist()
    ]


def load_history(root: Path) -> pa.Table:
    history = root / HISTORY_PATH
    return pq.read_table(history) if history.exists() else HISTORY_SCHEMA.empty_table()


def load_renamed(root: Path) -> list[str]:
    """Every symbol on either side of a stored Alpaca name change; empty when none are stored."""
    path = actions_dir(root) / NAME_CHANGES_FILE
    if not path.exists():
        return []
    table = pq.read_table(path, columns=["old_symbol", "new_symbol"])
    return sorted(
        set(table.column("old_symbol").to_pylist()) | set(table.column("new_symbol").to_pylist())
    )


def rename_chains(root: Path) -> list[frozenset[str]]:
    """The names each security has traded under, grouped: one set per connected chain of stored
    Alpaca name changes (VII -> BNZI -> PARA is one set). A reused symbol joins two securities'
    chains (Paramount's CBS/VIAC/PARA and Banzai's VII/BNZI/PARA become one set): harmless here,
    since chains only decide when a request is repeated, never who owns a bar. Names that are
    not valid symbols (CUSIP placeholders) are left out. Empty when no name changes are stored."""
    path = actions_dir(root) / NAME_CHANGES_FILE
    if not path.exists():
        return []
    table = pq.read_table(path, columns=["old_symbol", "new_symbol"])
    parent: dict[str, str] = {}

    def find(symbol: str) -> str:
        parent.setdefault(symbol, symbol)
        while parent[symbol] != symbol:
            parent[symbol] = parent[parent[symbol]]
            symbol = parent[symbol]
        return symbol

    for old, new in zip(
        table.column("old_symbol").to_pylist(), table.column("new_symbol").to_pylist(), strict=True
    ):
        if is_valid_symbol(old) and is_valid_symbol(new):
            parent[find(old)] = find(new)
    chains: dict[str, set[str]] = {}
    for symbol in parent:
        chains.setdefault(find(symbol), set()).add(symbol)
    return sorted((frozenset(c) for c in chains.values() if len(c) > 1), key=min)


def load_universe(root: Path, data_dir: Path | None = None) -> list[str]:
    """Universe from what is on disk: assets snapshots, SEC history and name changes, minus
    OTC-only tickers."""
    exclude = otc_only_symbols(data_dir / EXCHANGE_DIR) if data_dir else frozenset()
    return build_universe(
        load_assets(root),
        load_history(root).column("ticker").to_pylist(),
        load_renamed(root),
        exclude_symbols=exclude,
    )


def active_symbols(assets: Iterable[Asset]) -> set[str]:
    return {a.symbol for a in assets if a.status == "active"}


def read_symbols_file(path: Path) -> list[str]:
    """A hand-kept symbol list, one per line: blank lines and ``#`` comments are skipped, the
    rest must be valid symbols in canonical spelling. Sorted, without duplicates. A bad line
    stops the run rather than silently shrinking the list."""
    symbols: set[str] = set()
    for number, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if not is_valid_symbol(line):
            raise ValueError(f"{path}:{number}: {line!r} is not a symbol")
        symbols.add(line)
    return sorted(symbols)
