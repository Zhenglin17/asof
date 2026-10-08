"""Read path for market data. ``visible_bars`` is the only way code may read bars.

Bars live in Parquet under ``<data_dir>/market``; DuckDB queries them in place. The single
filter that matters is ``available_at <= as_of``: a bar on disk whose availability lies after
``as_of`` does not exist yet from the caller's point of view.
"""

import re
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa

from asof.ingest.alpaca import Timeframe
from asof.ingest.bars import _DIRS, BAR_SCHEMA
from asof.ingest.corporate_actions import SPLIT_SCHEMA

_YEAR_DIR = re.compile(r"^year=(\d{4})$")


def market_root(data_dir: Path) -> Path:
    return data_dir / "market"


def _aware_utc(value: datetime, name: str) -> datetime:
    if value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(UTC)


def _year_dirs(root: Path, timeframe: Timeframe) -> list[tuple[int, Path]]:
    directory = root / _DIRS[timeframe]
    if not directory.exists():
        return []
    found = []
    for year_dir in sorted(directory.glob("year=*")):
        match = _YEAR_DIR.match(year_dir.name)
        if match:
            found.append((int(match.group(1)), year_dir))
    return found


def _files(root: Path, timeframe: Timeframe) -> list[Path]:
    return [
        p
        for _, year_dir in _year_dirs(root, timeframe)
        for p in sorted(year_dir.rglob("*.parquet"))
    ]


def _source(files: Sequence[Path]) -> str:
    paths = ", ".join("'" + str(p).replace("'", "''") + "'" for p in files)
    return f"read_parquet([{paths}], hive_partitioning = true, union_by_name = true)"


def visible_bars(
    con: duckdb.DuckDBPyConnection,
    root: Path,
    timeframe: Timeframe,
    as_of: datetime,
    *,
    symbols: Sequence[str] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pa.Table:
    """Bars known at ``as_of``, ordered by symbol then time. ``[start, end)`` filters on ``t``.

    Raw store rows, NOT yet resolved against the security master: a symbol may carry another
    security's relabelled history (BK's 2020 bars under BNY) and volume-0 filler bars. Until the
    ``resolve`` step lands, do not feed this into features or counts of "symbols with data".

    Sets the connection's ``TimeZone`` to UTC: DuckDB labels exported timestamps with the
    session zone, and callers must get UTC back.
    """
    as_of_utc = _aware_utc(as_of, "as_of")
    start_utc = _aware_utc(start, "start") if start is not None else None
    end_utc = _aware_utc(end, "end") if end is not None else None
    files = _files(root, timeframe)
    if not files:
        return BAR_SCHEMA.empty_table()

    clauses = ["available_at <= $as_of"]
    params: dict[str, Any] = {"as_of": as_of_utc}
    if symbols is not None:
        clauses.append("symbol IN (SELECT UNNEST($symbols))")
        params["symbols"] = list(symbols)
    if start_utc is not None:
        clauses.append("t >= $start")
        params["start"] = start_utc
    if end_utc is not None:
        clauses.append("t < $end")
        params["end"] = end_utc

    con.execute("SET TimeZone = 'UTC'")
    columns = ", ".join(BAR_SCHEMA.names)
    query = (
        f"SELECT {columns} FROM {_source(files)} WHERE {' AND '.join(clauses)} ORDER BY symbol, t"
    )
    return con.execute(query, params).to_arrow_table()


def visible_sessions(
    con: duckdb.DuckDBPyConnection,
    root: Path,
    timeframe: Timeframe,
    as_of: datetime,
    *,
    last: int | None = None,
) -> list[date]:
    """Trading sessions known at ``as_of``, ascending: the distinct ``session_date`` of bars
    whose ``available_at`` is not after ``as_of``. ``last`` keeps only the most recent ones.

    This is the market calendar as the data shows it. A session only appears once at least one
    bar of it is visible, so a session still in progress is not listed.
    """
    as_of_utc = _aware_utc(as_of, "as_of")
    if last is not None and last <= 0:
        raise ValueError("last must be positive")
    files = _files(root, timeframe)
    if not files:
        return []
    con.execute("SET TimeZone = 'UTC'")
    query = (
        f"SELECT DISTINCT session_date FROM {_source(files)} WHERE available_at <= $as_of "
        "ORDER BY session_date DESC"
    )
    if last is not None:
        query += f" LIMIT {int(last)}"
    sessions = [row[0] for row in con.execute(query, {"as_of": as_of_utc}).fetchall()]
    return sorted(sessions)


def bars_status(con: duckdb.DuckDBPyConnection, root: Path, timeframe: Timeframe) -> list[dict]:
    """Per stored year: row count, distinct symbols, first and last bar time.

    Operational view for the CLI only; it has no ``as_of`` and strategy code must not use it.
    """
    con.execute("SET TimeZone = 'UTC'")
    status: list[dict] = []
    for year, year_dir in _year_dirs(root, timeframe):
        files = sorted(year_dir.rglob("*.parquet"))
        if not files:
            continue
        [summary] = (
            con.execute(
                "SELECT count(*) AS rows, count(DISTINCT symbol) AS symbols, "
                f"min(t) AS first, max(t) AS last FROM {_source(files)}"
            )
            .to_arrow_table()
            .to_pylist()
        )
        status.append({"year": year, **summary})
    return status


def _actions_file(root: Path, name: str) -> Path | None:
    path = root / "corporate_actions" / f"{name}.parquet"
    return path if path.exists() else None


def visible_splits(
    con: duckdb.DuckDBPyConnection,
    root: Path,
    as_of: datetime,
    symbols: Sequence[str] | None = None,
) -> pa.Table:
    """Split records known at ``as_of``: ``available_at`` (00:00 New York of the ex-date) must
    not lie after it. Ordered by symbol then ex_date."""
    as_of_utc = _aware_utc(as_of, "as_of")
    path = _actions_file(root, "splits")
    if path is None:
        return SPLIT_SCHEMA.empty_table()
    clauses = ["available_at <= $as_of"]
    params: dict[str, Any] = {"as_of": as_of_utc}
    if symbols is not None:
        clauses.append("symbol IN (SELECT UNNEST($symbols))")
        params["symbols"] = list(symbols)
    con.execute("SET TimeZone = 'UTC'")
    columns = ", ".join(SPLIT_SCHEMA.names)
    query = (
        f"SELECT {columns} FROM {_source([path])} WHERE {' AND '.join(clauses)} "
        "ORDER BY symbol, ex_date"
    )
    return con.execute(query, params).to_arrow_table()
