"""Guards against look-ahead leakage in market data: visible_bars must never return a bar whose
available_at is after as_of, even when the file on disk already holds it."""

from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import duckdb
import pytest

from asof.ingest.alpaca import Bar
from asof.ingest.bars import Partition, write_partition
from asof.store.market import bars_status, market_root, visible_bars
from tests.ingest.fakes import AVAILABLE_DAY, FETCHED_AT, T_DAY, T_MIN, daily, make_bar

MINUTE = timedelta(minutes=1)
DAY = timedelta(days=1)
EXPECTED_COLUMNS = {
    "symbol",
    "t",
    "session_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_count",
    "vwap",
    "available_at",
    "fetched_at",
}


@pytest.fixture
def con() -> Iterator[duckdb.DuckDBPyConnection]:
    connection = duckdb.connect()
    yield connection
    connection.close()


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return market_root(tmp_path)


def store(
    root: Path, partition: Partition, bars: list[Bar], fetched_at: datetime = FETCHED_AT
) -> None:
    write_partition(partition.path(root), bars, partition.timeframe, fetched_at)


def keys(table) -> list[tuple[str, datetime]]:
    return list(zip(table.column("symbol").to_pylist(), table.column("t").to_pylist(), strict=True))


# --- normal path -------------------------------------------------------------------------------


def test_market_root_lives_under_the_data_dir(tmp_path: Path) -> None:
    assert market_root(tmp_path) == tmp_path / "market"


def test_visible_bars_returns_stored_rows_ordered_by_symbol_then_time(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    store(
        root,
        Partition("1Day", 2026, None, "A"),
        [
            daily("AMZN", date(2026, 10, 1)),
            daily("AAPL", date(2026, 10, 2)),
            daily("AAPL", date(2026, 10, 1)),
        ],
    )
    store(root, Partition("1Day", 2026, None, "M"), [daily("MSFT", date(2026, 10, 1))])

    table = visible_bars(con, root, "1Day", as_of=datetime(2026, 10, 3, tzinfo=UTC))

    assert keys(table) == [
        ("AAPL", datetime(2026, 10, 1, 4, tzinfo=UTC)),
        ("AAPL", datetime(2026, 10, 2, 4, tzinfo=UTC)),
        ("AMZN", datetime(2026, 10, 1, 4, tzinfo=UTC)),
        ("MSFT", datetime(2026, 10, 1, 4, tzinfo=UTC)),
    ]
    assert EXPECTED_COLUMNS <= set(table.schema.names)


def test_visible_bars_reads_across_year_partitions(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    store(root, Partition("1Day", 2025, None, "A"), [daily("AAPL", date(2025, 12, 31))])
    store(root, Partition("1Day", 2026, None, "A"), [daily("AAPL", date(2026, 1, 2))])

    table = visible_bars(con, root, "1Day", as_of=datetime(2026, 2, 1, tzinfo=UTC))

    assert [k[1].date() for k in keys(table)] == [date(2025, 12, 31), date(2026, 1, 2)]


def test_symbols_filter_restricts_the_result(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    store(
        root,
        Partition("1Day", 2026, None, "A"),
        [daily("AAPL", date(2026, 10, 1)), daily("AMZN", date(2026, 10, 1))],
    )
    store(root, Partition("1Day", 2026, None, "M"), [daily("MSFT", date(2026, 10, 1))])
    as_of = datetime(2026, 10, 3, tzinfo=UTC)

    assert [
        k[0] for k in keys(visible_bars(con, root, "1Day", as_of, symbols=["MSFT", "AAPL"]))
    ] == ["AAPL", "MSFT"]
    assert keys(visible_bars(con, root, "1Day", as_of, symbols=["ZZZZ"])) == []


def test_time_window_is_start_inclusive_end_exclusive(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    bars = [make_bar("AAPL", T_MIN + i * MINUTE) for i in range(5)]
    store(root, Partition("1Min", 2026, 10, "A"), bars)
    as_of = datetime(2026, 10, 3, tzinfo=UTC)

    table = visible_bars(con, root, "1Min", as_of, start=T_MIN + MINUTE, end=T_MIN + 3 * MINUTE)

    assert [k[1] for k in keys(table)] == [T_MIN + MINUTE, T_MIN + 2 * MINUTE]


def test_bars_status_summarises_each_stored_year(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    store(
        root,
        Partition("1Day", 2025, None, "A"),
        [daily("AAPL", date(2025, 6, 2)), daily("AAPL", date(2025, 12, 31))],
    )
    store(
        root,
        Partition("1Day", 2026, None, "A"),
        [daily("AAPL", date(2026, 1, 2)), daily("AMZN", date(2026, 1, 2))],
    )
    store(root, Partition("1Day", 2026, None, "M"), [daily("MSFT", date(2026, 3, 2))])

    status = bars_status(con, root, "1Day")

    assert sorted(status, key=lambda s: s["year"]) == [
        {
            "year": 2025,
            "rows": 2,
            "symbols": 1,
            "first": datetime(2025, 6, 2, 4, tzinfo=UTC),  # EDT
            "last": datetime(2025, 12, 31, 5, tzinfo=UTC),  # EST
        },
        {
            "year": 2026,
            "rows": 3,
            "symbols": 3,
            "first": datetime(2026, 1, 2, 5, tzinfo=UTC),
            "last": datetime(2026, 3, 2, 5, tzinfo=UTC),
        },
    ]


# --- boundaries --------------------------------------------------------------------------------


def test_no_files_at_all_gives_an_empty_table(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    assert visible_bars(con, root, "1Day", as_of=datetime(2026, 10, 3, tzinfo=UTC)).num_rows == 0
    assert bars_status(con, root, "1Day") == []


def test_empty_partition_file_is_tolerated(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    store(root, Partition("1Day", 2019, None, "Q"), [])
    store(root, Partition("1Day", 2026, None, "A"), [daily("AAPL", date(2026, 1, 2))])

    table = visible_bars(con, root, "1Day", as_of=datetime(2026, 10, 3, tzinfo=UTC))

    assert keys(table) == [("AAPL", datetime(2026, 1, 2, 5, tzinfo=UTC))]
    status = sorted(bars_status(con, root, "1Day"), key=lambda s: s["year"])
    assert status[0] == {"year": 2019, "rows": 0, "symbols": 0, "first": None, "last": None}
    assert status[1]["rows"] == 1


def test_bar_available_exactly_at_as_of_is_visible(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    store(root, Partition("1Day", 2026, None, "A"), [daily("AAPL", date(2026, 10, 2))])

    assert keys(visible_bars(con, root, "1Day", as_of=AVAILABLE_DAY)) == [("AAPL", T_DAY)]
    assert (
        keys(visible_bars(con, root, "1Day", as_of=AVAILABLE_DAY - timedelta(microseconds=1))) == []
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"as_of": datetime(2026, 10, 3)},
        {"as_of": datetime(2026, 10, 3, tzinfo=UTC), "start": datetime(2026, 10, 1)},
        {"as_of": datetime(2026, 10, 3, tzinfo=UTC), "end": datetime(2026, 10, 3)},
    ],
    ids=["naive_as_of", "naive_start", "naive_end"],
)
def test_naive_datetimes_are_rejected(
    con: duckdb.DuckDBPyConnection, root: Path, kwargs: dict
) -> None:
    store(root, Partition("1Day", 2026, None, "A"), [daily("AAPL", date(2026, 10, 1))])

    with pytest.raises(ValueError):
        visible_bars(con, root, "1Day", **kwargs)


def test_as_of_in_another_zone_is_compared_as_an_instant(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    from zoneinfo import ZoneInfo

    store(root, Partition("1Day", 2026, None, "A"), [daily("AAPL", date(2026, 10, 2))])
    new_york = ZoneInfo("America/New_York")

    assert (
        keys(visible_bars(con, root, "1Day", as_of=datetime(2026, 10, 2, 19, 59, tzinfo=new_york)))
        == []
    )
    assert (
        len(
            keys(
                visible_bars(con, root, "1Day", as_of=datetime(2026, 10, 2, 20, 0, tzinfo=new_york))
            )
        )
        == 1
    )


# --- leakage -----------------------------------------------------------------------------------


def test_todays_daily_bar_is_invisible_during_the_session_and_visible_after_post_market(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # The bar for the 2026-10-02 session is on disk (t = 04:00Z) but the market has not closed
    # yet at 10:00 New York (14:00Z); it must only appear once post-market ends at 20:00 New
    # York (00:00Z next day), since the daily volume includes extended-hours trades.
    store(
        root,
        Partition("1Day", 2026, None, "A"),
        [daily("AAPL", date(2026, 10, 1)), daily("AAPL", date(2026, 10, 2))],
    )

    during = visible_bars(con, root, "1Day", as_of=datetime(2026, 10, 2, 14, 0, tzinfo=UTC))
    after = visible_bars(con, root, "1Day", as_of=datetime(2026, 10, 3, 0, 0, tzinfo=UTC))

    assert keys(during) == [("AAPL", datetime(2026, 10, 1, 4, tzinfo=UTC))]
    assert keys(after) == [("AAPL", datetime(2026, 10, 1, 4, tzinfo=UTC)), ("AAPL", T_DAY)]


def test_minute_bar_is_invisible_at_its_own_timestamp(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # A 09:30 bar covers 09:30:00-09:30:59; at 09:30:00 nothing about it is known.
    store(
        root,
        Partition("1Min", 2026, 10, "A"),
        [make_bar("AAPL", T_MIN), make_bar("AAPL", T_MIN + MINUTE)],
    )

    assert keys(visible_bars(con, root, "1Min", as_of=T_MIN)) == []
    assert keys(visible_bars(con, root, "1Min", as_of=T_MIN + MINUTE)) == [("AAPL", T_MIN)]
    assert keys(visible_bars(con, root, "1Min", as_of=T_MIN + 2 * MINUTE)) == [
        ("AAPL", T_MIN),
        ("AAPL", T_MIN + MINUTE),
    ]


def test_fetched_at_does_not_decide_visibility(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    # Re-downloading history stamps a new fetched_at; the bar was still knowable at its close.
    far_future = datetime(2030, 1, 1, tzinfo=UTC)
    store(
        root,
        Partition("1Day", 2026, None, "A"),
        [daily("AAPL", date(2026, 10, 1))],
        fetched_at=far_future,
    )

    table = visible_bars(con, root, "1Day", as_of=datetime(2026, 10, 2, tzinfo=UTC))

    assert keys(table) == [("AAPL", datetime(2026, 10, 1, 4, tzinfo=UTC))]
    assert table.column("fetched_at").to_pylist() == [far_future]


def test_time_window_does_not_override_as_of(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    # Asking for a window that includes the future bar must not make it visible.
    store(
        root,
        Partition("1Day", 2026, None, "A"),
        [daily("AAPL", date(2026, 10, 1)), daily("AAPL", date(2026, 10, 2))],
    )

    table = visible_bars(
        con,
        root,
        "1Day",
        as_of=datetime(2026, 10, 2, 14, 0, tzinfo=UTC),
        start=datetime(2026, 9, 1, tzinfo=UTC),
        end=datetime(2026, 11, 1, tzinfo=UTC),
    )

    assert keys(table) == [("AAPL", datetime(2026, 10, 1, 4, tzinfo=UTC))]


def test_timeframes_are_not_mixed(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    store(root, Partition("1Min", 2026, 10, "A"), [make_bar("AAPL", T_MIN)])
    store(root, Partition("1Day", 2026, None, "A"), [daily("AAPL", date(2026, 10, 1))])
    as_of = datetime(2026, 10, 3, tzinfo=UTC)

    assert keys(visible_bars(con, root, "1Day", as_of)) == [
        ("AAPL", datetime(2026, 10, 1, 4, tzinfo=UTC))
    ]
    assert keys(visible_bars(con, root, "1Min", as_of)) == [("AAPL", T_MIN)]
    assert [s["rows"] for s in bars_status(con, root, "1Min")] == [1]


def test_every_returned_row_satisfies_the_invariant(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    bars = [daily("AAPL", date(2026, 9, 28) + i * DAY) for i in range(10)]
    store(root, Partition("1Day", 2026, None, "A"), bars)

    for hour in (0, 14, 20, 23):
        as_of = datetime(2026, 10, 2, hour, tzinfo=UTC)
        table = visible_bars(con, root, "1Day", as_of)
        assert all(a <= as_of for a in table.column("available_at").to_pylist()), as_of
        assert table.num_rows == sum(1 for b in bars if b.t + timedelta(hours=20) <= as_of)


# --- robustness of the read path ---------------------------------------------------------------


def test_leftover_temp_files_and_junk_year_dirs_are_ignored(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    store(root, Partition("1Day", 2026, None, "A"), [daily("AAPL", date(2026, 10, 1))])
    (root / "bars_1d" / "year=2026" / "bucket=B.parquet.tmp").write_bytes(b"half written")
    (root / "bars_1d" / "year=junk").mkdir()
    (root / "bars_1d" / "year=junk" / "bucket=A.parquet").write_bytes(b"not parquet")

    table = visible_bars(con, root, "1Day", as_of=datetime(2026, 10, 3, tzinfo=UTC))

    assert keys(table) == [("AAPL", datetime(2026, 10, 1, 4, tzinfo=UTC))]
    assert [s["year"] for s in bars_status(con, root, "1Day")] == [2026]


def test_symbols_are_bound_as_parameters_not_spliced_into_sql(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    store(root, Partition("1Day", 2026, None, "A"), [daily("AAPL", date(2026, 10, 1))])

    table = visible_bars(
        con,
        root,
        "1Day",
        as_of=datetime(2026, 10, 3, tzinfo=UTC),
        symbols=["AAPL' OR '1'='1", "x'); DROP TABLE y; --"],
    )

    assert table.num_rows == 0
