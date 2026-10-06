"""Historical SEC ``company_tickers.json`` snapshots, recovered from the Wayback Machine.

Alpaca serves history for delisted symbols but cannot list them. The SEC ticker table is
archived roughly monthly since 2019; the union of every snapshot is the set of symbols that
existed, and each row keeps the snapshot date and CIK so a reused ticker (BBBY) stays two
different registrants.
"""

import gzip
import json
import os
import re
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import Any, NamedTuple

import pyarrow as pa
import pyarrow.parquet as pq

from asof.ingest.http import Fetch
from asof.ingest.tickers import is_valid_symbol, normalize_ticker

CDX_URL = "http://web.archive.org/cdx/search/cdx"
SNAPSHOT_TARGET = "www.sec.gov/files/company_tickers.json"
EXCHANGE_TARGET = "www.sec.gov/files/company_tickers_exchange.json"
MIN_ENTRIES = 1000  # the real table has ~6k-13k rows; anything smaller is an error page
_DATED = re.compile(r"^\d{8}\.json$")

HISTORY_SCHEMA = pa.schema(
    [
        ("snapshot_date", pa.date32()),
        ("ticker", pa.string()),
        ("cik", pa.int64()),
        ("name", pa.string()),
    ]
)


class SecTickerRow(NamedTuple):
    snapshot_date: date
    ticker: str
    cik: int
    name: str


def list_snapshots(
    fetch: Fetch, *, since_year: int = 2018, target: str = SNAPSHOT_TARGET
) -> list[str]:
    """Wayback timestamps (``YYYYMMDDhhmmss``) of archived copies, at most one per month."""
    result = fetch(
        CDX_URL,
        {
            "url": target,
            "output": "json",
            "from": str(since_year),
            "filter": "statuscode:200",
            "collapse": "timestamp:6",
            "fl": "timestamp",
        },
    )
    rows = json.loads(result.body) if result.body else []
    return [row[0] for row in rows[1:]]  # first row is the header


def snapshot_url(ts: str, target: str = SNAPSHOT_TARGET) -> str:
    # ``id_`` asks for the archived bytes without the Wayback toolbar wrapper.
    return f"http://web.archive.org/web/{ts}id_/https://{target}"


def _validated(body: bytes, target: str) -> bytes:
    if body[:2] == b"\x1f\x8b":
        body = gzip.decompress(body)
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("snapshot body is not JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("snapshot is not a plausible SEC ticker table")
    if target == EXCHANGE_TARGET:
        # {"fields": [...], "data": [[cik, name, ticker, exchange], ...]}
        rows = data.get("data")
        if (
            not isinstance(rows, list)
            or len(rows) < MIN_ENTRIES
            or "ticker" not in data.get("fields", [])
        ):
            raise ValueError("snapshot is not a plausible SEC exchange table")
    elif len(data) < MIN_ENTRIES:
        raise ValueError("snapshot is not a plausible SEC ticker table")
    return body


def _atomic_write(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_bytes(body)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def download_snapshot(
    ts: str, directory: Path, fetch: Fetch, *, target: str = SNAPSHOT_TARGET
) -> Path:
    """Store the snapshot as ``<YYYYMMDD>.json``; an existing file is kept as is."""
    path = directory / f"{ts[:8]}.json"
    if path.exists():
        return path
    result = fetch(snapshot_url(ts, target), {})
    if result.status != 200:
        raise ValueError(f"{result.status} fetching snapshot {ts}")
    _atomic_write(path, _validated(result.body, target))
    return path


def parse_snapshot(raw: bytes, snapshot_date: date) -> list[SecTickerRow]:
    data: dict[str, dict[str, Any]] = json.loads(raw)
    rows: list[SecTickerRow] = []
    for entry in data.values():
        try:
            ticker = normalize_ticker(str(entry["ticker"]))
        except ValueError:
            continue
        if not is_valid_symbol(ticker):
            continue
        rows.append(SecTickerRow(snapshot_date, ticker, int(entry["cik_str"]), str(entry["title"])))
    return rows


def _rows_table(rows: Iterable[SecTickerRow]) -> pa.Table:
    ordered = sorted(rows, key=lambda r: (r.snapshot_date, r.ticker))
    return pa.table(
        {
            "snapshot_date": pa.array([r.snapshot_date for r in ordered], pa.date32()),
            "ticker": pa.array([r.ticker for r in ordered], pa.string()),
            "cik": pa.array([r.cik for r in ordered], pa.int64()),
            "name": pa.array([r.name for r in ordered], pa.string()),
        },
        schema=HISTORY_SCHEMA,
    )


def build_history(directory: Path) -> pa.Table:
    """Every ``<YYYYMMDD>.json`` in ``directory`` as one table, sorted by date then ticker."""
    rows: list[SecTickerRow] = []
    if directory.exists():
        for path in sorted(directory.iterdir()):
            if _DATED.match(path.name):
                snapshot_date = date(int(path.name[:4]), int(path.name[4:6]), int(path.name[6:8]))
                rows.extend(parse_snapshot(path.read_bytes(), snapshot_date))
    return _rows_table(rows)


def write_history(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        pq.write_table(table, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def historical_symbols(table: pa.Table) -> set[str]:
    return set(table.column("ticker").to_pylist())


Window = tuple[date, date | None]


def symbol_windows(table: pa.Table) -> dict[str, Window]:
    """When each ticker could have existed, bracketed by the snapshots around its sightings.

    A ticker first seen in snapshot N appeared somewhere after snapshot N-1; one last seen in
    snapshot M disappeared before snapshot M+1. The window is ``[date(N-1), date(M+1))``, open
    at the end when M is the latest snapshot. Alpaca files a security's history under its
    newest name, so the window is what lets a bar be attributed to the name in use at the time.
    """
    dates = sorted(set(table.column("snapshot_date").to_pylist()))
    before = {d: (dates[i - 1] if i else d) for i, d in enumerate(dates)}
    after = {d: (dates[i + 1] if i + 1 < len(dates) else None) for i, d in enumerate(dates)}
    first: dict[str, date] = {}
    last: dict[str, date] = {}
    tickers: list[str] = table.column("ticker").to_pylist()
    days: list[date] = table.column("snapshot_date").to_pylist()
    for ticker, day in zip(tickers, days, strict=True):
        first[ticker] = min(first.get(ticker, day), day)
        last[ticker] = max(last.get(ticker, day), day)
    return {t: (before[first[t]], after[last[t]]) for t in first}


def _exchange_rows(raw: bytes) -> list[tuple[str, str]]:
    data = json.loads(raw)
    fields = data["fields"]
    ti, ei = fields.index("ticker"), fields.index("exchange")
    rows: list[tuple[str, str]] = []
    for row in data["data"]:
        if not row[ti]:
            continue
        try:
            ticker = normalize_ticker(str(row[ti]))
        except ValueError:
            continue
        if is_valid_symbol(ticker):
            rows.append((ticker, str(row[ei] or "").strip().upper()))
    return rows


def otc_only_symbols(directory: Path) -> set[str]:
    """Tickers the SEC exchange snapshots only ever place on OTC (or nowhere)."""
    seen: dict[str, set[str]] = {}
    if directory.exists():
        for path in sorted(directory.iterdir()):
            if _DATED.match(path.name):
                for ticker, exchange in _exchange_rows(path.read_bytes()):
                    seen.setdefault(ticker, set()).add(exchange)
    return {t for t, ex in seen.items() if "OTC" in ex and ex <= {"OTC", ""}}
