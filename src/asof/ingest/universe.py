"""The symbol universe for market-data backfill.

Union of Alpaca's listed assets (active and inactive, exchange-listed only) and every symbol
that ever appeared in an SEC ticker snapshot. Keeping inactive and historical names in is what
lets a replay see the companies that later disappeared.
"""

import os
from collections.abc import Collection, Iterable, Sequence
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from asof.ingest.alpaca import Asset
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
EXCHANGE_DIR = Path("raw") / "sec" / "company_tickers_exchange"


def build_universe(
    assets: Iterable[Asset],
    historical: Iterable[str],
    *,
    exclude_exchanges: frozenset[str] = frozenset({"OTC"}),
    exclude_symbols: Collection[str] = frozenset(),
) -> list[str]:
    """Exchange-listed Alpaca assets plus SEC-history tickers not known to be OTC-only."""
    symbols: set[str] = set()
    for asset in assets:
        # Alpaca symbols are already in canonical spelling; only filter out its junk entries.
        if asset.exchange not in exclude_exchanges and is_valid_symbol(asset.symbol):
            symbols.add(asset.symbol)
    for raw in historical:
        try:
            symbol = normalize_ticker(raw)
        except ValueError:
            continue
        if is_valid_symbol(symbol) and symbol not in exclude_symbols:
            symbols.add(symbol)
    return sorted(symbols)


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


def load_universe(root: Path, data_dir: Path | None = None) -> list[str]:
    """Universe from what is on disk: assets snapshots plus SEC history minus OTC-only tickers."""
    exclude = otc_only_symbols(data_dir / EXCHANGE_DIR) if data_dir else frozenset()
    return build_universe(
        load_assets(root), load_history(root).column("ticker").to_pylist(), exclude_symbols=exclude
    )


def active_symbols(assets: Iterable[Asset]) -> set[str]:
    return {a.symbol for a in assets if a.status == "active"}
