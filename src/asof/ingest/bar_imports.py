"""Bars imported from a second source, kept apart from the Alpaca files.

Alpaca rewrites a reused symbol's history with the new holder's chain (PARA, 2026-10-16), so a
lost stretch has to come from elsewhere. Imported bars live under ``bars_1d_ext/`` with the same
columns as the Alpaca files plus ``source`` and ``source_symbol``; the Alpaca files are never
touched. Every import is registered in ``configs/bar_imports.yaml`` with the symbol it is filed
under, the source and the source's own symbol, the date range, a reason, a date and a name, and
the read path lets an imported row replace the Alpaca row of the same ``(symbol, t)``: the entry
is the human decision that Alpaca's row for that key is wrong.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Literal, cast

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

from asof.ingest.alpaca import Bar
from asof.ingest.bars import BAR_SCHEMA, _to_table, bucket_of, session_date, write_merged
from asof.ingest.tickers import is_valid_symbol

__all__ = [
    "EXT_BAR_SCHEMA",
    "EXT_DIR",
    "BarImport",
    "ImportReport",
    "import_bars",
    "load_bar_imports",
]

EXT_DIR = "bars_1d_ext"
EXT_BAR_SCHEMA = BAR_SCHEMA.append(pa.field("source", pa.string())).append(
    pa.field("source_symbol", pa.string())
)
ImportSource = Literal["yahoo"]


class BarImport(BaseModel):
    """One registered import: ``symbol`` is the stored symbol, ``source_symbol`` the one the
    source files the series under (Yahoo serves Paramount's lineage as SKYD)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    source: ImportSource
    source_symbol: str
    start: date
    end: date
    reason: str
    added: date
    by: str

    @field_validator("symbol", "source_symbol")
    @classmethod
    def _symbol(cls, value: str) -> str:
        if not is_valid_symbol(value):
            raise ValueError(f"{value!r} is not a symbol in canonical spelling")
        return value

    @field_validator("reason", "by")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value.strip()

    @model_validator(mode="after")
    def _ordered(self) -> "BarImport":
        if self.start > self.end:
            raise ValueError(f"start {self.start} is after end {self.end}")
        return self


class _Loader(yaml.SafeLoader):
    """YAML dates stay strings so that pydantic reports a bad date under its field name."""


def _timestamp_as_string(loader: yaml.SafeLoader, node: yaml.Node) -> str:
    return str(loader.construct_scalar(cast(yaml.ScalarNode, node)))


_Loader.add_constructor("tag:yaml.org,2002:timestamp", _timestamp_as_string)


def load_bar_imports(path: Path) -> list[BarImport]:
    """``configs/bar_imports.yaml``: a list of entries, each validated as ``BarImport``; two
    entries for one symbol may not overlap. A missing file is an error, an empty list is fine."""
    raw = yaml.load(path.read_text(), Loader=_Loader)
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a list of imports")
    loaded: list[BarImport] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: entry {index} is not a mapping: {item!r}")
        try:
            loaded.append(BarImport.model_validate(item))
        except ValidationError as error:
            problems = "; ".join(
                f"{'.'.join(str(part) for part in e['loc']) or 'entry'}: {e['msg']}"
                for e in error.errors()
            )
            raise ValueError(f"{path}: entry {index}: {problems}") from None
    by_symbol: dict[str, list[BarImport]] = {}
    for entry in loaded:
        by_symbol.setdefault(entry.symbol, []).append(entry)
    for symbol, entries in by_symbol.items():
        ordered = sorted(entries, key=lambda e: e.start)
        for earlier, later in zip(ordered, ordered[1:], strict=False):
            if later.start <= earlier.end:
                raise ValueError(
                    f"{path}: entries for {symbol} overlap: {earlier.start}..{earlier.end} and "
                    f"{later.start}..{later.end}"
                )
    return loaded


@dataclass(frozen=True)
class ImportReport:
    rows_fetched: int
    rows_in_range: int
    rows_added: int
    rows_existing: int
    first_session: date | None
    last_session: date | None


def splits_after(root: Path, symbols: Sequence[str], start: date) -> list[tuple[str, date]]:
    """Split records of ``symbols`` with an ex-date after ``start``, from the stored splits
    table (none when the table is absent). Yahoo's prices and volumes are split-adjusted
    (measured 2026-10-16: NVDA 2024-06-07 close 120.89 there, 1208.88 raw), so a split after
    the start of an import range would have rewritten every earlier bar."""
    path = root / "corporate_actions" / "splits.parquet"
    if not path.exists():
        return []
    table = pq.read_table(path, columns=["symbol", "ex_date"])
    wanted = set(symbols)
    found = {
        (symbol, day)
        for symbol, day in zip(
            table.column("symbol").to_pylist(), table.column("ex_date").to_pylist(), strict=True
        )
        if symbol in wanted and day > start
    }
    return sorted(found)


def ext_path(root: Path, symbol: str, year: int) -> Path:
    return root / EXT_DIR / f"year={year}" / f"bucket={bucket_of(symbol)}.parquet"


def import_bars(
    root: Path, entry: BarImport, bars: Sequence[Bar], fetched_at: datetime
) -> ImportReport:
    """File ``bars`` under ``entry.symbol`` in ``bars_1d_ext``; returns what happened.

    Bars outside ``[entry.start, entry.end]`` by session date are left out; a bar not yet final
    at ``fetched_at`` is left for a later run (the store's rule); an existing row is never
    replaced. One file per New York year, merged into like the Alpaca partitions.
    """
    if fetched_at.tzinfo is None:
        raise ValueError("fetched_at must be timezone-aware")
    foreign = sorted({bar.symbol for bar in bars if bar.symbol != entry.source_symbol})
    if foreign:
        raise ValueError(
            f"{entry.symbol}: bars are filed under {', '.join(foreign)}, not the registered "
            f"source symbol {entry.source_symbol}"
        )
    later = splits_after(root, (entry.symbol, entry.source_symbol), entry.start)
    if later:
        listed = ", ".join(f"{symbol} {day}" for symbol, day in later)
        raise ValueError(
            f"{entry.symbol}: a split after the start of the range ({listed}) means the "
            "source's prices and volumes are adjusted for it; the store keeps raw prices, so "
            "this lineage cannot be imported from an adjusted source"
        )
    in_range = [
        bar.model_copy(update={"symbol": entry.symbol})
        for bar in bars
        if entry.start <= session_date(bar.t) <= entry.end
    ]
    table = _to_table(in_range, "1Day", fetched_at)
    n = table.num_rows
    table = table.append_column("source", pa.array([entry.source] * n, pa.string()))
    table = table.append_column("source_symbol", pa.array([entry.source_symbol] * n, pa.string()))
    table = table.cast(EXT_BAR_SCHEMA)

    added = existing = 0
    sessions: list[date] = table.column("session_date").to_pylist()
    for year in sorted({d.year for d in sessions}):
        part = table.filter(pa.array([d.year == year for d in sessions], pa.bool_()))
        _, new = write_merged(ext_path(root, entry.symbol, year), part, EXT_BAR_SCHEMA, fetched_at)
        added += new
        existing += part.num_rows - new
    return ImportReport(
        rows_fetched=len(bars),
        rows_in_range=len(in_range),
        rows_added=added,
        rows_existing=existing,
        first_session=min(sessions) if sessions else None,
        last_session=max(sessions) if sessions else None,
    )
