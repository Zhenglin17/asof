"""Read path for market data. ``visible_bars`` is the only way code may read bars.

Bars live in Parquet under ``<data_dir>/market``; DuckDB queries them in place. The single
filter that matters is ``available_at <= as_of``: a bar on disk whose availability lies after
``as_of`` does not exist yet from the caller's point of view.

Two more things happen on the way out, both decided by the user on 2026-10-08:

* **Identity.** A symbol is a label. By default every bar is matched to the security-master
  segment that owned its symbol on its session date (``visible_master``), comes back with that
  segment's ``security_id``, and a bar matching no knowable segment is not returned at all --
  BK's 2022 history filed under BNY vanishes, BK's own rows come back as S000730. The master
  itself is read as of ``as_of``: a segment is unknown before its ``available_at`` and its end is
  unknown before its ``end_available_at``, so a reader in 2022 sees BK as an open segment and a
  reader on 2023-10-20 still attributes LAC's bars to LAC although the file says LAC ended on
  10-04 (Alpaca filed the rename on 11-02).
* **Trading.** Alpaca keeps writing a bar a day under an old label after a rename, volume 0 and
  OHLC frozen at the last close (VTIQ at 33.97 for two years; 1.18M bars store-wide, 6.5%).
  ``traded`` is ``volume > 0`` and untraded bars are dropped unless asked for.
"""

import re
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from asof.ingest.alpaca import Timeframe
from asof.ingest.bar_imports import EXT_BAR_SCHEMA, EXT_DIR
from asof.ingest.bars import _DIRS, BAR_SCHEMA
from asof.ingest.corporate_actions import SPLIT_SCHEMA
from asof.ingest.security_master import MASTER_FILE, MASTER_SCHEMA

_YEAR_DIR = re.compile(r"^year=(\d{4})$")
_EXT_DIRS: dict[str, str] = {"1Day": EXT_DIR}  # imported bars exist for daily bars only

# ``source`` names where a bar came from: 'alpaca' for the Alpaca files (nothing on disk says
# so; the read path adds it) or the stored source of an imported row ('yahoo').
UNRESOLVED_BAR_SCHEMA = BAR_SCHEMA.append(pa.field("source", pa.string())).append(
    pa.field("traded", pa.bool_())
)
RESOLVED_BAR_SCHEMA = UNRESOLVED_BAR_SCHEMA.append(pa.field("security_id", pa.string())).append(
    pa.field("split_day", pa.bool_())
)

_MASTER_VIEW = "_asof_market_master"
_SPLITS_VIEW = "_asof_market_splits"


def market_root(data_dir: Path) -> Path:
    return data_dir / "market"


def _aware_utc(value: datetime, name: str) -> datetime:
    if value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(UTC)


def _year_dirs_in(directory: Path) -> list[tuple[int, Path]]:
    if not directory.exists():
        return []
    found = []
    for year_dir in sorted(directory.glob("year=*")):
        match = _YEAR_DIR.match(year_dir.name)
        if match:
            found.append((int(match.group(1)), year_dir))
    return found


def _year_dirs(root: Path, timeframe: Timeframe) -> list[tuple[int, Path]]:
    return _year_dirs_in(root / _DIRS[timeframe])


def _parquet_files(year_dirs: Sequence[tuple[int, Path]]) -> list[Path]:
    return [p for _, year_dir in year_dirs for p in sorted(year_dir.rglob("*.parquet"))]


def _alpaca_files(root: Path, timeframe: Timeframe) -> list[Path]:
    return _parquet_files(_year_dirs(root, timeframe))


def _ext_files(root: Path, timeframe: Timeframe) -> list[Path]:
    """Imported-bar files, each checked to carry the import schema: a file of another shape
    would read as ``source`` NULL and pass for a vendor row."""
    directory = _EXT_DIRS.get(timeframe)
    if not directory:
        return []
    files = _parquet_files(_year_dirs_in(root / directory))
    for path in files:
        if not pq.read_schema(path).equals(EXT_BAR_SCHEMA, check_metadata=False):
            raise ValueError(f"{path}: not an imported-bar file (columns differ from the schema)")
    return files


def _files(root: Path, timeframe: Timeframe) -> list[Path]:
    """Every bar file of ``timeframe``: the Alpaca partitions, then the imported ones."""
    return [*_alpaca_files(root, timeframe), *_ext_files(root, timeframe)]


def bar_files(root: Path, timeframe: Timeframe) -> tuple[list[Path], list[Path]]:
    """The Alpaca partitions and the imported-bar files of ``timeframe``, for callers that
    build their own query over ``bars_source_sql`` (the security master build)."""
    return _alpaca_files(root, timeframe), _ext_files(root, timeframe)


def _source(files: Sequence[Path]) -> str:
    paths = ", ".join("'" + str(p).replace("'", "''") + "'" for p in files)
    return f"read_parquet([{paths}], hive_partitioning = true, union_by_name = true)"


def bars_source_sql(
    files: Sequence[Path], ext_files: Sequence[Path], as_of: str | None = None
) -> str:
    """A SELECT over the bar files yielding the bar columns, the hive ``year`` and ``source``.

    ``files`` are Alpaca partitions (source 'alpaca'), ``ext_files`` imported bars with their
    stored ``source``. When both hold the same ``(symbol, t)`` only the imported row is
    returned: an import is the recorded decision that Alpaca's row for that key is wrong.
    ``as_of`` is a SQL expression (``$as_of`` or a literal); when given, every branch keeps
    only rows with ``available_at`` not after it, so an imported row that is not yet knowable
    does not hide a vendor row that is. At least one list must be non-empty. Used as a subquery
    by ``visible_bars`` and by the audit's raw view, so both see one bar per key.
    """
    columns = ", ".join(BAR_SCHEMA.names)
    knowable = f"available_at <= {as_of}" if as_of else None
    alpaca = f"SELECT {columns}, CAST(year AS BIGINT) AS year, 'alpaca' AS source FROM "
    ext = f"SELECT {columns}, CAST(year AS BIGINT) AS year, source FROM {_source(ext_files)}"
    if knowable:
        ext += f" WHERE {knowable}"
    if not ext_files:
        return alpaca + _source(files) + (f" WHERE {knowable}" if knowable else "")
    if not files:
        return ext
    not_overridden = (
        f"NOT EXISTS (SELECT 1 FROM {_source(ext_files)} e WHERE "
        + (f"{knowable.replace('available_at', 'e.available_at')} AND " if knowable else "")
        + "e.symbol = a.symbol AND e.t = a.t)"
    )
    alpaca_where = f"{knowable.replace('available_at', 'a.available_at')} AND " if knowable else ""
    return f"{alpaca}{_source(files)} a WHERE {alpaca_where}{not_overridden} UNION ALL {ext}"


def visible_bars(
    con: duckdb.DuckDBPyConnection,
    root: Path,
    timeframe: Timeframe,
    as_of: datetime,
    *,
    symbols: Sequence[str] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    resolve: bool = True,
    include_untraded: bool = False,
) -> pa.Table:
    """Bars known at ``as_of``. ``[start, end)`` filters on ``t``; ``symbols`` on the stored
    symbol. Every row carries ``traded`` (``volume > 0``); untraded rows are left out unless
    ``include_untraded``.

    ``resolve=True`` (the default, ``RESOLVED_BAR_SCHEMA``): each bar is attributed to the
    ``visible_master`` segment that owned its symbol on its ``session_date`` and carries that
    segment's ``security_id``; bars owned by no knowable segment are not returned. ``split_day``
    marks a bar whose session is the ex-date of a split known at ``as_of``. Rows come back in
    ``security_id, t`` order, so the two names of a renamed security form one timeline. A missing
    master is an error, never a silent fall-back to raw rows.

    ``resolve=False`` (``UNRESOLVED_BAR_SCHEMA``): the store as it is, ordered by symbol then
    time, for audits and debugging. A symbol may carry another security's relabelled history.

    Sets the connection's ``TimeZone`` to UTC: DuckDB labels exported timestamps with the
    session zone, and callers must get UTC back.
    """
    as_of_utc = _aware_utc(as_of, "as_of")
    start_utc = _aware_utc(start, "start") if start is not None else None
    end_utc = _aware_utc(end, "end") if end is not None else None
    master = visible_master(con, root, as_of_utc) if resolve else None
    files = _alpaca_files(root, timeframe)
    ext_files = _ext_files(root, timeframe)
    if not files and not ext_files:
        return (RESOLVED_BAR_SCHEMA if resolve else UNRESOLVED_BAR_SCHEMA).empty_table()

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
    if not include_untraded:
        clauses.append("volume > 0")

    con.execute("SET TimeZone = 'UTC'")
    columns = ", ".join(f"b.{name}" for name in (*BAR_SCHEMA.names, "source"))
    bars = (
        f"(SELECT * FROM ({bars_source_sql(files, ext_files, '$as_of')}) "
        f"WHERE {' AND '.join(clauses)}) b"
    )
    if master is None:
        query = f"SELECT {columns}, b.volume > 0 AS traded FROM {bars} ORDER BY b.symbol, b.t"
        return con.execute(query, params).to_arrow_table().cast(UNRESOLVED_BAR_SCHEMA)

    splits = visible_splits(con, root, as_of_utc, symbols)
    con.register(_MASTER_VIEW, master)
    con.register(_SPLITS_VIEW, splits.select(["symbol", "ex_date"]))
    try:
        query = f"""
            SELECT {columns}, b.volume > 0 AS traded, s.security_id,
                   sp.symbol IS NOT NULL AS split_day
            FROM {bars}
            JOIN {_MASTER_VIEW} s
              ON s.symbol = b.symbol
             AND b.session_date >= s.valid_from
             AND (s.valid_to IS NULL OR b.session_date < s.valid_to)
            LEFT JOIN (SELECT DISTINCT symbol, ex_date FROM {_SPLITS_VIEW}) sp
              ON sp.symbol = b.symbol AND sp.ex_date = b.session_date
            ORDER BY s.security_id, b.t, b.symbol
            """
        return con.execute(query, params).to_arrow_table().cast(RESOLVED_BAR_SCHEMA)
    finally:
        con.unregister(_MASTER_VIEW)
        con.unregister(_SPLITS_VIEW)


def visible_master(con: duckdb.DuckDBPyConnection, root: Path, as_of: datetime) -> pa.Table:
    """Security-master segments known at ``as_of``, ordered by security_id, valid_from, symbol.

    A segment is listed once its ``available_at`` is not after ``as_of``. Its end is a separate
    fact: while ``end_available_at`` lies after ``as_of`` the segment is returned open, with
    ``valid_to`` and ``end_available_at`` both null -- a reader in 2022 must not learn that BK
    ends in 2026. The two sides of one event (BK's end, BNY's start) share one instant, so there
    is never a moment at which a label is owned twice or by nobody because of this masking.

    The file is checked before use: a closed segment without an end instant (or the reverse)
    means the master predates this column and must be rebuilt; duplicate segment keys and
    overlapping segments of one symbol among the visible rows would hand a bar out twice.
    """
    as_of_utc = _aware_utc(as_of, "as_of")
    path = root / MASTER_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path}: no security master; run `asof market identity`")
    con.execute("SET TimeZone = 'UTC'")
    on_disk = pq.read_table(path)
    missing = [name for name in MASTER_SCHEMA.names if name not in on_disk.schema.names]
    if missing:
        raise ValueError(
            f"{path}: columns {missing} are missing (an older build); rebuild the master with "
            "`asof market identity`"
        )
    con.register(_MASTER_VIEW, on_disk)
    try:
        broken = _scalar(
            con,
            f"SELECT count(*) FROM {_MASTER_VIEW} WHERE available_at IS NULL "
            "OR (valid_to IS NULL) <> (end_available_at IS NULL) "
            "OR valid_to <= valid_from "
            "OR end_available_at < (valid_to::TIMESTAMP AT TIME ZONE 'America/New_York')",
        )
        if broken:
            raise ValueError(
                f"{path}: {broken} segments have no available_at, a valid_to without an "
                "end_available_at (or the reverse), a valid_to not after valid_from, or an end "
                "knowable before its day; rebuild the master with `asof market identity`"
            )
        columns = ", ".join(
            name for name in MASTER_SCHEMA.names if name not in ("valid_to", "end_available_at")
        )
        visible = con.execute(
            f"""
            SELECT {columns},
                   CASE WHEN end_available_at <= $as_of THEN valid_to END AS valid_to,
                   CASE WHEN end_available_at <= $as_of THEN end_available_at END
                       AS end_available_at
            FROM {_MASTER_VIEW}
            WHERE available_at <= $as_of
            ORDER BY security_id, valid_from, symbol
            """,
            {"as_of": as_of_utc},
        ).to_arrow_table()
    finally:
        con.unregister(_MASTER_VIEW)
    visible = visible.select(MASTER_SCHEMA.names).cast(MASTER_SCHEMA)

    con.register(_MASTER_VIEW, visible)
    try:
        duplicates = _scalar(
            con,
            f"SELECT count(*) FROM (SELECT security_id, symbol, valid_from FROM {_MASTER_VIEW} "
            "GROUP BY ALL HAVING count(*) > 1)",
        )
        if duplicates:
            raise ValueError(f"{path}: {duplicates} duplicate segment keys visible at {as_of_utc}")
        overlaps = _scalar(
            con,
            f"SELECT count(*) FROM {_MASTER_VIEW} a JOIN {_MASTER_VIEW} b "
            "ON a.symbol = b.symbol "
            "AND (a.valid_from < b.valid_from "
            "     OR (a.valid_from = b.valid_from AND a.security_id < b.security_id)) "
            "AND (a.valid_to IS NULL OR b.valid_from < a.valid_to)",
        )
        if overlaps:
            raise ValueError(
                f"{path}: {overlaps} overlapping segments of one symbol visible at {as_of_utc}"
            )
    finally:
        con.unregister(_MASTER_VIEW)
    return visible


def _scalar(con: duckdb.DuckDBPyConnection, query: str) -> int:
    row = con.execute(query).fetchone()
    return int(row[0]) if row else 0


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
