"""Daily bars read once for a replay, answered as of each step exactly like ``visible_bars``.

A replay steps hundreds of evenings; re-reading the Parquet store each time is the slow part, not
the query. The cache copies the rows of the requested symbols and range into DuckDB temp tables
once, then runs the same SQL as ``visible_bars`` against them, with the security master
(``visible_master``) and the split records cut at each step's ``as_of``. Identity is resolved
as of every step, never once with the final master: a reader in 2022 must not see a 2024
rename.
"""

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import duckdb
import pyarrow as pa

from asof.ingest.security_master import MASTER_FILE
from asof.store.market import (
    RESOLVED_BAR_SCHEMA,
    bar_files,
    select_bars,
    splits_as_of,
    visible_master,
    visible_splits,
)

_END_OF_TIME = datetime(9999, 1, 1, tzinfo=UTC)


class BarCache:
    def __init__(
        self,
        con: duckdb.DuckDBPyConnection,
        root: Path,
        *,
        symbols: Sequence[str],
        start: datetime,
        end: datetime,
    ) -> None:
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("start and end must be timezone-aware")
        self.con = con
        self.symbols = sorted(set(symbols))
        self.start = start.astimezone(UTC)
        self.end = end.astimezone(UTC)
        tag = uuid4().hex[:12]
        files, ext_files = bar_files(root, "1Day")
        params = {"symbols": self.symbols, "start": self.start, "end": self.end}
        where = "symbol IN (SELECT UNNEST($symbols)) AND t >= $start AND t < $end"
        con.execute("SET TimeZone = 'UTC'")
        self._alpaca = self._copy(f"_asof_cache_bars_{tag}", files, where, params)
        self._ext = self._copy(f"_asof_cache_ext_{tag}", ext_files, where, params)

        self.root = root
        if not (root / MASTER_FILE).exists():
            raise FileNotFoundError(f"{root / MASTER_FILE}: no security master")
        self._splits = f"_asof_cache_splits_{tag}"
        con.register(f"{self._splits}_src", visible_splits(con, root, _END_OF_TIME, self.symbols))
        con.execute(f"CREATE TEMP TABLE {self._splits} AS SELECT * FROM {self._splits}_src")
        con.unregister(f"{self._splits}_src")

    def _copy(
        self, name: str, files: Sequence[Path], where: str, params: dict[str, object]
    ) -> str | None:
        if not files:
            return None
        paths = ", ".join("'" + str(p).replace("'", "''") + "'" for p in files)
        source = f"read_parquet([{paths}], hive_partitioning = true, union_by_name = true)"
        self.con.execute(
            f"CREATE TEMP TABLE {name} AS SELECT * FROM {source} WHERE {where}", params
        )
        return name

    def bars_at(self, as_of: datetime, *, start: datetime | None = None) -> pa.Table:
        """What ``visible_bars(con, root, "1Day", as_of, symbols, start, end)`` returns; a later
        ``start`` narrows the range (the replay asks for one evening's bars only)."""
        if as_of.tzinfo is None or (start is not None and start.tzinfo is None):
            raise ValueError("as_of and start must be timezone-aware")
        if self._alpaca is None and self._ext is None:
            return RESOLVED_BAR_SCHEMA.empty_table()
        as_of_utc = as_of.astimezone(UTC)
        master = visible_master(self.con, self.root, as_of_utc)
        splits = splits_as_of(self.con, self._splits, as_of_utc, self.symbols)
        return select_bars(
            self.con,
            self._alpaca,
            self._ext,
            as_of_utc,
            master=master,
            splits=splits,
            symbols=self.symbols,
            start=max(self.start, start.astimezone(UTC)) if start is not None else self.start,
            end=self.end,
            include_untraded=False,
        )
