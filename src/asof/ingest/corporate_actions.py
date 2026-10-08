"""Corporate actions from Alpaca: splits, renames, mergers, dividends and the rest.

``GET /v1beta1/corporate-actions`` returns, per type, the records effective in a date range.
The raw pages are kept verbatim under ``raw/alpaca/corporate_actions/<type>/<year>.json`` and
parsed into a few typed Parquet tables; rows that fail validation go to a ``rejects`` table
with the original JSON instead of being dropped (the real feed has a dividend dated 3026 and
unit splits with a zero rate).

``available_at`` is 00:00 New York of the effective day (``ex_date`` for splits, dividends,
spin-offs and rights; the effective/process date for renames and mergers). The feed carries no
announcement date, and the announcement always precedes the effective day, so this is never
earlier than reality; it is the start of the day rather than its end because the ex-date's
first pre-market print is already at the post-split price and a 09:35 scan must know why.
"""

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any, Protocol
from zoneinfo import ZoneInfo

import pyarrow as pa
import pyarrow.parquet as pq

from asof.ingest.tickers import normalize_ticker

ET = ZoneInfo("America/New_York")
ENDPOINT = "/v1beta1/corporate-actions"
PAGE_LIMIT = 1000
MAX_YEAR = 2100  # anything later is a typo in the feed, not a scheduled event

TS_UTC = pa.timestamp("us", tz="UTC")

SPLIT_SCHEMA = pa.schema(
    [
        ("symbol", pa.string()),
        ("ex_date", pa.date32()),
        ("old_rate", pa.float64()),
        ("new_rate", pa.float64()),
        ("factor", pa.float64()),
        ("kind", pa.string()),
        ("old_cusip", pa.string()),
        ("new_cusip", pa.string()),
        ("available_at", TS_UTC),
    ]
)
NAME_CHANGE_SCHEMA = pa.schema(
    [
        ("old_symbol", pa.string()),
        ("new_symbol", pa.string()),
        ("process_date", pa.date32()),
        ("old_cusip", pa.string()),
        ("new_cusip", pa.string()),
        ("available_at", TS_UTC),
    ]
)
MERGER_SCHEMA = pa.schema(
    [
        ("acquiree_symbol", pa.string()),
        ("acquirer_symbol", pa.string()),
        ("kind", pa.string()),
        ("effective_date", pa.date32()),
        ("rate", pa.float64()),
        ("available_at", TS_UTC),
    ]
)
DIVIDEND_SCHEMA = pa.schema(
    [
        ("symbol", pa.string()),
        ("ex_date", pa.date32()),
        ("rate", pa.float64()),
        ("special", pa.bool_()),
        ("sub_type", pa.string()),
        ("available_at", TS_UTC),
    ]
)
OTHER_SCHEMA = pa.schema(
    [
        ("kind", pa.string()),
        ("symbol", pa.string()),
        ("source_symbol", pa.string()),
        ("new_symbol", pa.string()),
        ("ex_date", pa.date32()),
        ("process_date", pa.date32()),
        ("rate", pa.float64()),
        ("raw_json", pa.string()),
        ("available_at", TS_UTC),
    ]
)
REJECT_SCHEMA = pa.schema(
    [("kind", pa.string()), ("raw_json", pa.string()), ("reason", pa.string())]
)

# Parquet file name under <market root>/corporate_actions/ -> parser.
TABLES = ("splits", "name_changes", "mergers", "dividends", "others")


class ActionsSource(Protocol):
    @property
    def DATA_URL(self) -> str: ...  # noqa: N802 - matches AlpacaClient's class attribute

    def _request(self, url: str, params: dict[str, str]) -> Any: ...


class Rejected(ValueError):
    """A record that must be quarantined; the message is the reason."""


# -- download ----------------------------------------------------------------------------------


def _year_chunks(start: date, end: date) -> list[tuple[date, date]]:
    chunks = []
    for year in range(start.year, end.year + 1):
        lo = max(start, date(year, 1, 1))
        hi = min(end, date(year, 12, 31))
        chunks.append((lo, hi))
    return chunks


def fetch_corporate_actions(
    client: ActionsSource, types: Sequence[str], start: date, end: date
) -> list[dict[str, Any]]:
    """Every record of ``types`` effective in ``[start, end]``, tagged with its response key.

    One paging chain per calendar year keeps each chain short. ``_kind`` is the plural key the
    record sat under (``forward_splits``, ``name_changes``, ...), as the endpoint spells it.
    """
    if start > end:
        raise ValueError(f"start {start} is after end {end}")
    url = client.DATA_URL + ENDPOINT
    rows: list[dict[str, Any]] = []
    for lo, hi in _year_chunks(start, end):
        params = {
            "types": ",".join(types),
            "start": lo.isoformat(),
            "end": hi.isoformat(),
            "limit": str(PAGE_LIMIT),
        }
        while True:
            page = client._request(url, params) or {}
            for kind, items in (page.get("corporate_actions") or {}).items():
                for item in items or []:
                    rows.append({**item, "_kind": kind})
            token = page.get("next_page_token")
            if not token:
                break
            params = {**params, "page_token": token}
    return rows


def raw_path(root: Path, type_: str, year: int) -> Path:
    return root / "raw" / "alpaca" / "corporate_actions" / type_ / f"{year}.json"


def write_raw(rows: Sequence[Mapping[str, Any]], root: Path, type_: str, year: int) -> None:
    """Store the records as received (plus ``_kind``); atomic, replaces an existing file."""
    path = raw_path(root, type_, year)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(list(rows)))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def read_raw(root: Path, type_: str) -> list[dict[str, Any]]:
    """All stored years of one type, oldest first."""
    directory = raw_path(root, type_, 0).parent
    if not directory.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        rows.extend(json.loads(path.read_text()))
    return rows


# -- field readers -----------------------------------------------------------------------------


def _kind_of(row: Mapping[str, Any]) -> str:
    """Singular type name: ``forward_splits`` -> ``forward_split``."""
    return str(row.get("_kind", "unknown")).removesuffix("s")


def _symbol(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not value or not str(value).strip():
        raise Rejected(f"missing {key}")
    try:
        return normalize_ticker(str(value))
    except ValueError as error:
        raise Rejected(f"bad symbol {value!r}") from error


def _optional_symbol(row: Mapping[str, Any], key: str) -> str | None:
    value = row.get(key)
    if not value or not str(value).strip():
        return None
    try:
        return normalize_ticker(str(value))
    except ValueError:
        return str(value).strip().upper()


def _date(row: Mapping[str, Any], key: str) -> date:
    value = row.get(key)
    if not value:
        raise Rejected(f"missing {key} date")
    try:
        parsed = date.fromisoformat(str(value))
    except ValueError as error:
        raise Rejected(f"bad {key} date {value!r}") from error
    if parsed.year > MAX_YEAR:
        raise Rejected(f"{key} date {value!r} is after {MAX_YEAR}")
    return parsed


def _optional_date(row: Mapping[str, Any], key: str) -> date | None:
    if not row.get(key):
        return None
    return _date(row, key)


def _rate(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if value is None:
        raise Rejected(f"missing {key} rate")
    try:
        rate = float(value)
    except (TypeError, ValueError) as error:
        raise Rejected(f"bad {key} rate {value!r}") from error
    if rate <= 0:
        raise Rejected(f"{key} rate {rate} is not positive")
    return rate


def _optional_float(row: Mapping[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _cusip(row: Mapping[str, Any], key: str) -> str | None:
    value = row.get(key)
    return str(value) if value else None  # the feed writes "" when it has none


def available_at(day: date) -> datetime:
    """00:00 New York of the effective day, in UTC."""
    return datetime.combine(day, time(0, 0), tzinfo=ET).astimezone(UTC)


# -- parsers -----------------------------------------------------------------------------------


def _parse(
    rows: Iterable[Mapping[str, Any]],
    schema: pa.Schema,
    convert: Any,
) -> tuple[pa.Table, pa.Table]:
    good: list[dict[str, Any]] = []
    bad: list[dict[str, Any]] = []
    for row in rows:
        try:
            good.append(convert(row))
        except Rejected as error:
            bad.append(
                {"kind": _kind_of(row), "raw_json": json.dumps(dict(row)), "reason": str(error)}
            )
    return pa.Table.from_pylist(good, schema=schema), pa.Table.from_pylist(
        bad, schema=REJECT_SCHEMA
    )


def _split_row(row: Mapping[str, Any]) -> dict[str, Any]:
    kind = _kind_of(row)
    if kind == "unit_split":
        symbol = _symbol(row, "old_symbol")
        day = _date(row, "effective_date")
        old_cusip, new_cusip = _cusip(row, "old_cusip"), _cusip(row, "new_cusip")
    else:
        symbol = _symbol(row, "symbol")
        day = _date(row, "ex_date")
        if kind == "forward_split":
            old_cusip, new_cusip = _cusip(row, "cusip"), None
        else:
            old_cusip, new_cusip = _cusip(row, "old_cusip"), _cusip(row, "new_cusip")
    old_rate, new_rate = _rate(row, "old_rate"), _rate(row, "new_rate")
    return {
        "symbol": symbol,
        "ex_date": day,
        "old_rate": old_rate,
        "new_rate": new_rate,
        "factor": new_rate / old_rate,
        "kind": kind,
        "old_cusip": old_cusip,
        "new_cusip": new_cusip,
        "available_at": available_at(day),
    }


def parse_splits(rows: Iterable[Mapping[str, Any]]) -> tuple[pa.Table, pa.Table]:
    """Forward, reverse and unit splits in one table; ``factor`` = new shares per old share."""
    return _parse(rows, SPLIT_SCHEMA, _split_row)


def _name_change_row(row: Mapping[str, Any]) -> dict[str, Any]:
    day = _date(row, "process_date")
    return {
        "old_symbol": _symbol(row, "old_symbol"),
        "new_symbol": _symbol(row, "new_symbol"),
        "process_date": day,
        "old_cusip": _cusip(row, "old_cusip"),
        "new_cusip": _cusip(row, "new_cusip"),
        "available_at": available_at(day),
    }


def parse_name_changes(rows: Iterable[Mapping[str, Any]]) -> tuple[pa.Table, pa.Table]:
    """Symbol and CUSIP changes; a row with old == new symbol is a CUSIP-only change."""
    return _parse(rows, NAME_CHANGE_SCHEMA, _name_change_row)


def _merger_row(row: Mapping[str, Any]) -> dict[str, Any]:
    kind = _kind_of(row)
    day = _optional_date(row, "effective_date") or _date(row, "process_date")
    # Cash deals quote dollars per share; stock deals quote acquirer shares per share. A mixed
    # deal keeps its share leg here; its cash leg stays in the raw JSON.
    rate = _optional_float(row, "rate" if kind == "cash_merger" else "acquirer_rate")
    return {
        "acquiree_symbol": _symbol(row, "acquiree_symbol"),
        "acquirer_symbol": _optional_symbol(row, "acquirer_symbol"),
        "kind": kind,
        "effective_date": day,
        "rate": rate,
        "available_at": available_at(day),
    }


def parse_mergers(rows: Iterable[Mapping[str, Any]]) -> tuple[pa.Table, pa.Table]:
    return _parse(rows, MERGER_SCHEMA, _merger_row)


def _dividend_row(row: Mapping[str, Any]) -> dict[str, Any]:
    day = _date(row, "ex_date")
    return {
        "symbol": _symbol(row, "symbol"),
        "ex_date": day,
        "rate": _optional_float(row, "rate"),
        "special": bool(row.get("special", False)),
        "sub_type": row.get("sub_type") or None,
        "available_at": available_at(day),
    }


def parse_dividends(rows: Iterable[Mapping[str, Any]]) -> tuple[pa.Table, pa.Table]:
    return _parse(rows, DIVIDEND_SCHEMA, _dividend_row)


def _other_row(row: Mapping[str, Any]) -> dict[str, Any]:
    ex_date = _optional_date(row, "ex_date")
    process_date = _optional_date(row, "process_date")
    day = ex_date or process_date
    if day is None:
        raise Rejected("missing ex_date and process_date")
    symbol = _optional_symbol(row, "symbol")
    source = _optional_symbol(row, "source_symbol")
    if symbol is None and source is None:
        raise Rejected("missing symbol and source_symbol")
    return {
        "kind": _kind_of(row),
        "symbol": symbol,
        "source_symbol": source,
        "new_symbol": _optional_symbol(row, "new_symbol"),
        "ex_date": ex_date,
        "process_date": process_date,
        "rate": _optional_float(row, "rate"),
        "raw_json": json.dumps(dict(row)),
        "available_at": available_at(day),
    }


def parse_others(rows: Iterable[Mapping[str, Any]]) -> tuple[pa.Table, pa.Table]:
    """Spin-offs, stock dividends, rights, worthless removals, redemptions: fixed columns plus
    the original JSON, since these are stored for later and not yet read by anything."""
    return _parse(rows, OTHER_SCHEMA, _other_row)


# -- building the Parquet tables ---------------------------------------------------------------

PARSERS = {
    "splits": (("forward_split", "reverse_split", "unit_split"), parse_splits),
    "name_changes": (("name_change",), parse_name_changes),
    "mergers": (("cash_merger", "stock_merger", "stock_and_cash_merger"), parse_mergers),
    "dividends": (("cash_dividend",), parse_dividends),
    "others": (
        ("spin_off", "stock_dividend", "rights_distribution", "worthless_removal", "redemption"),
        parse_others,
    ),
}
ALL_TYPES = tuple(t for types, _ in PARSERS.values() for t in types)


def actions_dir(market_root: Path) -> Path:
    return market_root / "corporate_actions"


def _write_table(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        pq.write_table(table, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def build_tables(data_root: Path, market_root: Path) -> dict[str, int]:
    """Parse every stored raw file into ``<market root>/corporate_actions/*.parquet``.

    Returns row counts per table, ``rejects`` included.
    """
    counts: dict[str, int] = {}
    rejects: list[pa.Table] = []
    for name, (types, parser) in PARSERS.items():
        rows = [row for type_ in types for row in read_raw(data_root, type_)]
        table, bad = parser(rows)
        _write_table(table, actions_dir(market_root) / f"{name}.parquet")
        counts[name] = table.num_rows
        rejects.append(bad)
    all_bad = pa.concat_tables(rejects) if rejects else REJECT_SCHEMA.empty_table()
    _write_table(all_bad, actions_dir(market_root) / "rejects.parquet")
    counts["rejects"] = all_bad.num_rows
    return counts


def download_all(
    client: ActionsSource, data_root: Path, start: date, end: date, types: Sequence[str] = ALL_TYPES
) -> dict[str, int]:
    """Fetch each type year by year into the raw store; returns record counts per type."""
    counts: dict[str, int] = {}
    for type_ in types:
        total = 0
        for lo, hi in _year_chunks(start, end):
            rows = fetch_corporate_actions(client, [type_], lo, hi)
            existing = raw_path(data_root, type_, lo.year)
            if not rows and existing.exists() and json.loads(existing.read_text()):
                raise RuntimeError(
                    f"{type_} {lo.year}: Alpaca returned no records but {existing} is not "
                    "empty; refusing to overwrite a good file with an empty response"
                )
            write_raw(rows, data_root, type_, lo.year)
            total += len(rows)
        counts[type_] = total
    return counts
