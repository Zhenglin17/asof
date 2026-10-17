"""Guards against the second bar source leaking into, or being lost from, the read path: a row
imported into ``bars_1d_ext`` must come back through ``visible_bars`` like any Alpaca bar (same
``available_at`` rule, same master attribution, same filters), must replace the Alpaca row of the
same ``(symbol, t)`` rather than sit beside it, and must say where it came from (``source``).

Target API (asof.store.market; implementation pending):

    _files(root, "1Day") -> Alpaca files, then root/bars_1d_ext/year=*/**/*.parquet
    bars_source_sql(files, ext_files) -> str      # a relation: BAR_SCHEMA columns + source
    UNRESOLVED_BAR_SCHEMA = BAR_SCHEMA + source (string) + traded (bool)
    RESOLVED_BAR_SCHEMA = UNRESOLVED_BAR_SCHEMA + security_id (string) + split_day (bool)

Fake data provenance:
- PARA 2022-02-17 .. 2025-08-06 exists only in the ext files (measured 2026-10-16: Alpaca
  rewrote PARA with Banzai's chain after the 2026-08-07 reuse). The fixtures put an Alpaca PARA
  row on one of those days as well, with Banzai's price level (close 0.1902 on 2024-06-03,
  measured), to prove the ext row wins when both exist.
- Ext rows carry ``available_at`` = 20:00 New York of their session, exactly like Alpaca's daily
  bars (docs: ingest.bars; the import reuses ``available_at(t, "1Day")``).
- Master segments: VIAC [2020-02-13, 2022-02-17) and PARA [2022-02-17, 2026-08-07) both under
  one security (real chain, 2026-10-08 store); the fixture's ids are its own.
- ``source`` values: 'alpaca' for the Alpaca files (there is no such column on disk; the read
  path adds it) and the stored ``source`` for ext files ('yahoo').
"""

from collections import defaultdict
from collections.abc import Iterator, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from asof.ingest.bar_imports import EXT_BAR_SCHEMA, EXT_DIR
from asof.ingest.bars import BAR_SCHEMA, Partition, bucket_of, write_partition
from asof.store.market import (
    RESOLVED_BAR_SCHEMA,
    UNRESOLVED_BAR_SCHEMA,
    _files,
    bars_source_sql,
    market_root,
    visible_bars,
)
from tests.ingest.fakes import FETCHED_AT, daily
from tests.store.test_market import ET, Seg, avail, known, open_master, store_daily, write_master

BOTH_MODES = pytest.mark.parametrize("resolve", [False, True], ids=["raw", "resolved"])
LATE = datetime(2026, 10, 16, tzinfo=ET)
IMPORTED_AT = datetime(2026, 10, 16, 15, 0, tzinfo=ET).astimezone(UTC)
SID_PARAMOUNT, SID_BANZAI = "S000001", "S000002"
DAY = timedelta(days=1)


# =============================================================================================
# builders
# =============================================================================================


def ext_row(
    symbol: str,
    day: date,
    *,
    close: float,
    volume: int = 1_000,
    source: str = "yahoo",
    source_symbol: str = "SKYD",
    available_at: datetime | None = None,
    fetched_at: datetime = IMPORTED_AT,
) -> dict[str, Any]:
    """One ext row as ``import_bars`` writes it (open/high/low derive from close)."""
    t = daily(symbol, day).t
    high, low = round(close * 1.01, 4), round(close * 0.99, 4)
    return {
        "symbol": symbol,
        "t": t,
        "session_date": day,
        "open": round(close * 0.995, 4),
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "trade_count": None,
        "vwap": round((high + low + close) / 3, 4),
        "available_at": available_at or avail(day),
        "fetched_at": fetched_at,
        "source": source,
        "source_symbol": source_symbol,
    }


def write_ext(root: Path, rows: Sequence[dict[str, Any]]) -> list[Path]:
    """Write ext rows into ``bars_1d_ext/year=Y/bucket=X.parquet`` files (one per year/bucket)."""
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["session_date"].year, bucket_of(row["symbol"]))].append(row)
    paths = []
    for (year, bucket), group in sorted(grouped.items()):
        path = root / EXT_DIR / f"year={year}" / f"bucket={bucket}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(group, schema=EXT_BAR_SCHEMA), path)
        paths.append(path)
    return paths


def paramount_master(root: Path) -> None:
    """VIAC -> PARA under one security; Banzai takes PARA on 2026-08-07 (dates real)."""
    write_master(
        root,
        [
            Seg(
                SID_PARAMOUNT,
                "VIAC",
                date(2020, 2, 13),
                date(2022, 2, 17),
                end_available_at=known(date(2022, 2, 17)),
            ),
            Seg(
                SID_PARAMOUNT,
                "PARA",
                date(2022, 2, 17),
                date(2026, 8, 7),
                available_at=known(date(2022, 2, 17)),
                end_available_at=known(date(2026, 8, 7)),
            ),
            Seg(SID_BANZAI, "PARA", date(2026, 8, 7), None, available_at=known(date(2026, 8, 7))),
            Seg("S000003", "AAPL"),
        ],
    )


def sessions(table: pa.Table) -> list[tuple[str, date, str]]:
    return list(
        zip(
            table.column("symbol").to_pylist(),
            table.column("session_date").to_pylist(),
            table.column("source").to_pylist(),
            strict=True,
        )
    )


def one(table: pa.Table) -> dict[str, Any]:
    assert table.num_rows == 1, table.to_pylist()
    return table.to_pylist()[0]


@pytest.fixture
def con() -> Iterator[duckdb.DuckDBPyConnection]:
    connection = duckdb.connect()
    yield connection
    connection.close()


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return market_root(tmp_path)


@pytest.fixture
def mixed(root: Path) -> Path:
    """Alpaca: AAPL 2024-06-03/04, PARA 2024-06-03 (Banzai's level); ext: PARA 2024-06-03 and
    2024-06-04 (Paramount's level, from SKYD)."""
    store_daily(
        root,
        [
            daily("AAPL", date(2024, 6, 3), close=194.03),
            daily("AAPL", date(2024, 6, 4), close=194.35),
            daily("PARA", date(2024, 6, 3), close=0.1902, volume=500),
        ],
    )
    write_ext(
        root,
        [
            ext_row("PARA", date(2024, 6, 3), close=12.80, volume=9_000_000),
            ext_row("PARA", date(2024, 6, 4), close=12.65, volume=8_000_000),
        ],
    )
    paramount_master(root)
    return root


# =============================================================================================
# schemas and file discovery
# =============================================================================================


def test_schemas_carry_the_source_column() -> None:
    assert UNRESOLVED_BAR_SCHEMA.names == [*BAR_SCHEMA.names, "source", "traded"]
    assert RESOLVED_BAR_SCHEMA.names == [*UNRESOLVED_BAR_SCHEMA.names, "security_id", "split_day"]
    assert UNRESOLVED_BAR_SCHEMA.field("source").type == pa.string()


def test_daily_files_list_alpaca_files_then_ext_files(root: Path) -> None:
    store_daily(root, [daily("AAPL", date(2024, 6, 3)), daily("PARA", date(2024, 6, 3))])
    ext = write_ext(root, [ext_row("PARA", date(2024, 6, 3), close=12.8)])

    files = _files(root, "1Day")

    assert files == [
        root / "bars_1d" / "year=2024" / "bucket=A.parquet",
        root / "bars_1d" / "year=2024" / "bucket=P.parquet",
        *ext,
    ]


def test_ext_directory_alone_is_listed(root: Path) -> None:
    ext = write_ext(root, [ext_row("PARA", date(2024, 6, 3), close=12.8)])

    assert _files(root, "1Day") == ext


def test_minute_files_do_not_include_the_ext_directory(root: Path) -> None:
    write_ext(root, [ext_row("PARA", date(2024, 6, 3), close=12.8)])
    minute = Partition("1Min", 2024, 6, "A").path(root)
    write_partition(minute, [], "1Min", FETCHED_AT)

    assert _files(root, "1Min") == [minute]


# =============================================================================================
# bars_source_sql
# =============================================================================================


def alpaca_files(root: Path) -> list[Path]:
    return [p for p in _files(root, "1Day") if EXT_DIR not in p.parts]


def ext_files(root: Path) -> list[Path]:
    return [p for p in _files(root, "1Day") if EXT_DIR in p.parts]


def query(con: duckdb.DuckDBPyConnection, sql: str) -> pa.Table:
    con.execute("SET TimeZone = 'UTC'")
    return con.execute(f"SELECT * FROM ({sql}) ORDER BY symbol, t").to_arrow_table()


def test_source_sql_yields_the_bar_columns_plus_source(
    con: duckdb.DuckDBPyConnection, mixed: Path
) -> None:
    table = query(con, bars_source_sql(alpaca_files(mixed), ext_files(mixed)))

    assert set(table.schema.names) >= {*BAR_SCHEMA.names, "source"}
    assert sessions(table) == [
        ("AAPL", date(2024, 6, 3), "alpaca"),
        ("AAPL", date(2024, 6, 4), "alpaca"),
        ("PARA", date(2024, 6, 3), "yahoo"),  # the Alpaca row of this key is not returned
        ("PARA", date(2024, 6, 4), "yahoo"),
    ]
    para = [
        r
        for r in table.to_pylist()
        if r["symbol"] == "PARA" and r["session_date"] == date(2024, 6, 3)
    ]
    assert [r["close"] for r in para] == [12.8]


def test_source_sql_with_no_ext_files_labels_everything_alpaca(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    store_daily(root, [daily("AAPL", date(2024, 6, 3)), daily("PARA", date(2024, 6, 3))])

    table = query(con, bars_source_sql(alpaca_files(root), []))

    assert table.column("source").to_pylist() == ["alpaca", "alpaca"]
    assert table.num_rows == 2


def test_source_sql_with_no_alpaca_files_returns_the_ext_rows(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    write_ext(root, [ext_row("PARA", date(2024, 6, 3), close=12.8)])

    table = query(con, bars_source_sql([], ext_files(root)))

    assert sessions(table) == [("PARA", date(2024, 6, 3), "yahoo")]


def test_source_sql_is_usable_as_a_subquery_with_a_where_clause(
    con: duckdb.DuckDBPyConnection, mixed: Path
) -> None:
    sql = bars_source_sql(alpaca_files(mixed), ext_files(mixed))
    con.execute("SET TimeZone = 'UTC'")

    count = con.execute(f"SELECT count(*) FROM ({sql}) WHERE source = 'yahoo'").fetchone()
    assert count is not None and count[0] == 2


# =============================================================================================
# visible_bars: normal path
# =============================================================================================


@BOTH_MODES
def test_symbol_with_only_alpaca_rows_has_source_alpaca(
    con: duckdb.DuckDBPyConnection, mixed: Path, resolve: bool
) -> None:
    table = visible_bars(con, mixed, "1Day", LATE, symbols=["AAPL"], resolve=resolve)

    assert table.schema.equals(RESOLVED_BAR_SCHEMA if resolve else UNRESOLVED_BAR_SCHEMA)
    assert sessions(table) == [
        ("AAPL", date(2024, 6, 3), "alpaca"),
        ("AAPL", date(2024, 6, 4), "alpaca"),
    ]


@BOTH_MODES
def test_key_present_in_both_comes_back_once_with_the_ext_values(
    con: duckdb.DuckDBPyConnection, mixed: Path, resolve: bool
) -> None:
    day = datetime(2024, 6, 3, tzinfo=ET)
    row = one(
        visible_bars(
            con, mixed, "1Day", LATE, symbols=["PARA"], start=day, end=day + DAY, resolve=resolve
        )
    )

    assert (row["source"], row["close"], row["volume"], row["trade_count"]) == (
        "yahoo",
        12.8,
        9_000_000,
        None,
    )
    assert row["traded"] is True
    if resolve:
        assert row["security_id"] == SID_PARAMOUNT


@BOTH_MODES
def test_ext_row_for_a_day_alpaca_lacks_is_added(
    con: duckdb.DuckDBPyConnection, mixed: Path, resolve: bool
) -> None:
    table = visible_bars(con, mixed, "1Day", LATE, symbols=["PARA"], resolve=resolve)

    assert sessions(table) == [
        ("PARA", date(2024, 6, 3), "yahoo"),
        ("PARA", date(2024, 6, 4), "yahoo"),
    ]


def test_resolved_rows_keep_the_security_order_across_sources(
    con: duckdb.DuckDBPyConnection, mixed: Path
) -> None:
    table = visible_bars(con, mixed, "1Day", LATE)

    assert [(r["security_id"], r["symbol"], r["session_date"]) for r in table.to_pylist()] == [
        (SID_PARAMOUNT, "PARA", date(2024, 6, 3)),
        (SID_PARAMOUNT, "PARA", date(2024, 6, 4)),
        ("S000003", "AAPL", date(2024, 6, 3)),
        ("S000003", "AAPL", date(2024, 6, 4)),
    ]


def test_ext_source_value_is_the_stored_one(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    write_ext(root, [ext_row("PARA", date(2024, 6, 3), close=12.8, source="other")])
    open_master(root, "PARA")

    assert one(visible_bars(con, root, "1Day", LATE))["source"] == "other"


# =============================================================================================
# visible_bars: boundaries
# =============================================================================================


def test_symbols_filter_matches_the_stored_symbol_not_the_source_symbol(
    con: duckdb.DuckDBPyConnection, mixed: Path
) -> None:
    assert visible_bars(con, mixed, "1Day", LATE, symbols=["SKYD"], resolve=False).num_rows == 0
    assert visible_bars(con, mixed, "1Day", LATE, symbols=["PARA"], resolve=False).num_rows == 2


def test_untraded_ext_row_follows_the_traded_rule(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    write_ext(root, [ext_row("PARA", date(2024, 6, 3), close=12.8, volume=0)])
    open_master(root, "PARA")

    assert visible_bars(con, root, "1Day", LATE).num_rows == 0
    row = one(visible_bars(con, root, "1Day", LATE, include_untraded=True))
    assert (row["traded"], row["source"]) == (False, "yahoo")


def test_time_window_applies_to_ext_rows(con: duckdb.DuckDBPyConnection, mixed: Path) -> None:
    day = datetime(2024, 6, 4, tzinfo=ET)
    table = visible_bars(con, mixed, "1Day", LATE, symbols=["PARA"], start=day, resolve=False)

    assert sessions(table) == [("PARA", date(2024, 6, 4), "yahoo")]


def test_ext_row_outside_every_knowable_segment_is_not_resolved(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # 2022-02-16 is VIAC's last day: a PARA row there belongs to no segment (raw still shows it)
    write_ext(root, [ext_row("PARA", date(2022, 2, 16), close=29.58)])
    paramount_master(root)

    assert visible_bars(con, root, "1Day", LATE).num_rows == 0
    assert visible_bars(con, root, "1Day", LATE, resolve=False).num_rows == 1


def test_store_with_only_ext_files_reads_like_any_store(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    write_ext(root, [ext_row("PARA", date(2024, 6, 3), close=12.8)])
    paramount_master(root)

    row = one(visible_bars(con, root, "1Day", LATE))
    assert (row["symbol"], row["source"], row["security_id"]) == ("PARA", "yahoo", SID_PARAMOUNT)


def test_ext_rows_across_years_are_read_together(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    write_ext(
        root,
        [
            ext_row("PARA", date(2022, 12, 30), close=16.88),
            ext_row("PARA", date(2023, 1, 3), close=17.05),
        ],
    )
    open_master(root, "PARA")

    table = visible_bars(con, root, "1Day", LATE)
    assert [r["session_date"] for r in table.to_pylist()] == [date(2022, 12, 30), date(2023, 1, 3)]


# =============================================================================================
# visible_bars: leakage
# =============================================================================================


@BOTH_MODES
def test_ext_row_is_invisible_before_the_post_market_close_of_its_session(
    con: duckdb.DuckDBPyConnection, mixed: Path, resolve: bool
) -> None:
    day = datetime(2024, 6, 4, tzinfo=ET)
    before = visible_bars(
        con,
        mixed,
        "1Day",
        datetime(2024, 6, 4, 19, 0, tzinfo=ET),
        symbols=["PARA"],
        start=day,
        resolve=resolve,
    )
    at_close = visible_bars(
        con,
        mixed,
        "1Day",
        datetime(2024, 6, 4, 20, 0, tzinfo=ET),
        symbols=["PARA"],
        start=day,
        resolve=resolve,
    )

    assert before.num_rows == 0
    assert sessions(at_close) == [("PARA", date(2024, 6, 4), "yahoo")]


def test_ext_row_with_a_future_available_at_on_disk_is_not_returned(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # the only filter is available_at: an ext row stamped as knowable in 2027 does not exist yet
    write_ext(
        root,
        [
            ext_row(
                "PARA", date(2024, 6, 3), close=12.8, available_at=datetime(2027, 1, 1, tzinfo=UTC)
            )
        ],
    )
    open_master(root, "PARA")

    assert visible_bars(con, root, "1Day", LATE, resolve=False).num_rows == 0
    assert (
        visible_bars(con, root, "1Day", datetime(2027, 1, 1, tzinfo=UTC), resolve=False).num_rows
        == 1
    )


def test_every_returned_row_satisfies_the_invariant_across_sources(
    con: duckdb.DuckDBPyConnection, mixed: Path
) -> None:
    for hour in (0, 12, 19, 20, 23):
        as_of = datetime(2024, 6, 4, hour, tzinfo=ET)
        table = visible_bars(con, mixed, "1Day", as_of, resolve=False)
        assert all(a <= as_of for a in table.column("available_at").to_pylist()), hour


# =============================================================================================
# reviewer 2026-10-16: precedence and the availability rule; ext files must have the schema
# =============================================================================================


def test_future_ext_row_does_not_hide_the_alpaca_row_that_is_already_knowable(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # Alpaca's row is knowable on 2024-06-03 20:00; the ext row for the same key only in 2027.
    # Until then the Alpaca row must be served, not swallowed by the not-yet-visible import.
    store_daily(root, [daily("PARA", date(2024, 6, 3), close=0.1902, volume=500)])
    write_ext(
        root,
        [
            ext_row(
                "PARA", date(2024, 6, 3), close=12.8, available_at=datetime(2027, 1, 1, tzinfo=UTC)
            )
        ],
    )
    paramount_master(root)

    before = one(visible_bars(con, root, "1Day", LATE, symbols=["PARA"]))
    assert (before["source"], before["close"]) == ("alpaca", 0.1902)
    after = one(visible_bars(con, root, "1Day", datetime(2027, 1, 2, tzinfo=UTC), symbols=["PARA"]))
    assert (after["source"], after["close"]) == ("yahoo", 12.8)


def test_ext_file_without_the_source_columns_is_refused(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    path = root / EXT_DIR / "year=2024" / "bucket=P.parquet"
    write_partition(path, [daily("PARA", date(2024, 6, 3))], "1Day", FETCHED_AT)  # Alpaca shape
    open_master(root, "PARA")

    with pytest.raises(ValueError, match="bucket=P.parquet"):
        visible_bars(con, root, "1Day", LATE)
