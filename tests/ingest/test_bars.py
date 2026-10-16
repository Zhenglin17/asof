"""Guards against bars that claim to be known before the market produced them, half-written
partition files, a backfill that re-downloads what it already has or skips what changed, and a
refetch that loses or overwrites bars an earlier fetch stored.

The ``chains`` tests (renamed names with no SEC window) carry no ``as_of``; what their fake
client assumes about Alpaca has to be checked on real data:

- Requested alone, a pruned old name (SPLG) answers with its own bars: after a backfill with
  ``chains``, count bars_1d rows with symbol SPLG; zero means Alpaca never serves it and the
  solo request is wasted. Measured 2026-10-16: ``--symbols SPLG`` for 2024 gives 252 rows.
- Batched with its new name, the old name comes back empty or relabelled (measured for
  MMC/MRSH on 2026-10-06): compare the SPLG row count of a ``--symbols SPLG,SPYM`` run against
  a ``--symbols SPLG`` run. Measured 2026-10-16: 0 against 252.
- An old name batched with an unrelated symbol answers normally, so the solo request is only
  worth it when a chain partner shared the request (the assumption behind the ``chains``
  rule): compare the SPLG row count of a ``--symbols SPLG,SPY`` run against ``--symbols
  SPLG``. Measured 2026-10-16: 252 against 252.

The merge tests model what Alpaca does to a reused symbol (measured 2026-10-16: VJET's
2020-2024 bars became copies of RMED's; AXNX's 2020-2024 history vanished and 2025 copies of
MIMI appeared). Check after a forced refetch: the row count of a closed partition never goes
down, and ``asof market audit`` reports no symbol whose earliest stored bar moved later.
"""

from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from asof.ingest.alpaca import Bar
from asof.ingest.bars import (
    ET,
    BackfillReport,
    Partition,
    available_at,
    backfill,
    bucket_of,
    is_complete,
    partition_fetched_at,
    plan_partitions,
    session_date,
    write_partition,
)
from tests.ingest.fakes import (
    AVAILABLE_DAY,
    FETCHED_AT,
    T_DAY,
    T_MIN,
    BarsCall,
    FakeBarClient,
    daily,
    make_bar,
)

TS_UTC = pa.timestamp("us", tz="UTC")
BAR_SCHEMA = pa.schema(
    [
        ("symbol", pa.string()),
        ("t", TS_UTC),
        ("session_date", pa.date32()),
        ("open", pa.float64()),
        ("high", pa.float64()),
        ("low", pa.float64()),
        ("close", pa.float64()),
        ("volume", pa.int64()),
        ("trade_count", pa.int64()),
        ("vwap", pa.float64()),
        ("available_at", TS_UTC),
        ("fetched_at", TS_UTC),
    ]
)
TODAY = date(2026, 10, 5)
MINUTE = timedelta(minutes=1)
NOW_2027 = datetime(2027, 1, 15, tzinfo=UTC)


def open_bounds(partition: Partition) -> tuple[datetime, datetime]:
    """Request window for a partition still open at FETCHED_AT: stops short of recent data."""
    start, _ = partition.bounds()
    return start, FETCHED_AT - timedelta(minutes=16)


def leftovers(directory: Path) -> list[str]:
    if not directory.exists():
        return []
    return sorted(str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file())


# =============================================================================================
# session_date / available_at / bucket_of
# =============================================================================================


@pytest.mark.parametrize(
    ("t", "expected"),
    [
        (datetime(2026, 10, 2, 4, 0, tzinfo=UTC), date(2026, 10, 2)),
        (datetime(2026, 10, 2, 3, 59, tzinfo=UTC), date(2026, 10, 1)),
        (datetime(2026, 10, 2, 13, 30, tzinfo=UTC), date(2026, 10, 2)),
        (datetime(2026, 10, 3, 0, 30, tzinfo=UTC), date(2026, 10, 2)),
        (datetime(2026, 1, 5, 5, 0, tzinfo=UTC), date(2026, 1, 5)),
        (datetime(2026, 1, 5, 4, 59, tzinfo=UTC), date(2026, 1, 4)),
        (datetime(2026, 10, 2, 0, 0, tzinfo=ET), date(2026, 10, 2)),
    ],
    ids=[
        "edt_midnight",
        "edt_before_midnight",
        "open",
        "late_evening",
        "est_midnight",
        "est_before",
        "et_input",
    ],
)
def test_session_date_is_the_new_york_calendar_day(t: datetime, expected: date) -> None:
    assert session_date(t) == expected


def test_session_date_rejects_naive_datetimes() -> None:
    with pytest.raises(ValueError):
        session_date(datetime(2026, 10, 2, 4, 0))


def test_minute_bar_is_available_one_minute_after_its_timestamp() -> None:
    got = available_at(T_MIN, "1Min")

    assert got == T_MIN + MINUTE
    assert got.utcoffset() == timedelta(0)


@pytest.mark.parametrize(
    ("t", "expected"),
    [
        (datetime(2026, 10, 2, 4, 0, tzinfo=UTC), datetime(2026, 10, 3, 0, 0, tzinfo=UTC)),
        (datetime(2026, 1, 5, 5, 0, tzinfo=UTC), datetime(2026, 1, 6, 1, 0, tzinfo=UTC)),
    ],
    ids=["edt", "est"],
)
def test_daily_bar_is_available_when_the_post_market_session_ends(
    t: datetime, expected: datetime
) -> None:
    got = available_at(t, "1Day")

    assert got == expected
    assert got.tzinfo is not None
    assert got.utcoffset() == timedelta(0)


@pytest.mark.parametrize("timeframe", ["1Min", "1Day"])
def test_available_at_rejects_naive_datetimes(timeframe: str) -> None:
    with pytest.raises(ValueError):
        available_at(datetime(2026, 10, 2, 4, 0), timeframe)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("symbol", "bucket"), [("AAPL", "A"), ("BRK.B", "B"), ("F", "F"), ("ZZZZ.WS", "Z")]
)
def test_bucket_is_the_first_character(symbol: str, bucket: str) -> None:
    assert bucket_of(symbol) == bucket


# =============================================================================================
# Partition
# =============================================================================================


def test_daily_partition_path(tmp_path: Path) -> None:
    assert Partition("1Day", 2026, None, "A").path(tmp_path) == (
        tmp_path / "bars_1d" / "year=2026" / "bucket=A.parquet"
    )


def test_minute_partition_path_has_a_zero_padded_month(tmp_path: Path) -> None:
    assert Partition("1Min", 2026, 3, "B").path(tmp_path) == (
        tmp_path / "bars_1m" / "year=2026" / "month=03" / "bucket=B.parquet"
    )


def test_yearly_bounds_follow_new_york_midnight() -> None:
    assert Partition("1Day", 2026, None, "A").bounds() == (
        datetime(2026, 1, 1, 5, 0, tzinfo=UTC),
        datetime(2027, 1, 1, 5, 0, tzinfo=UTC),
    )


@pytest.mark.parametrize(
    ("month", "start", "end"),
    [
        (1, datetime(2026, 1, 1, 5, 0, tzinfo=UTC), datetime(2026, 2, 1, 5, 0, tzinfo=UTC)),
        (3, datetime(2026, 3, 1, 5, 0, tzinfo=UTC), datetime(2026, 4, 1, 4, 0, tzinfo=UTC)),
        (7, datetime(2026, 7, 1, 4, 0, tzinfo=UTC), datetime(2026, 8, 1, 4, 0, tzinfo=UTC)),
        (11, datetime(2026, 11, 1, 4, 0, tzinfo=UTC), datetime(2026, 12, 1, 5, 0, tzinfo=UTC)),
        (12, datetime(2026, 12, 1, 5, 0, tzinfo=UTC), datetime(2027, 1, 1, 5, 0, tzinfo=UTC)),
    ],
    ids=["january", "dst_starts", "july", "dst_ends", "december_rollover"],
)
def test_monthly_bounds_follow_new_york_midnight(
    month: int, start: datetime, end: datetime
) -> None:
    got_start, got_end = Partition("1Min", 2026, month, "A").bounds()

    assert (got_start, got_end) == (start, end)
    assert got_start.utcoffset() == timedelta(0)
    assert got_end.utcoffset() == timedelta(0)


@pytest.mark.parametrize(
    ("today", "expected"),
    [
        (date(2026, 1, 1), True),
        (TODAY, True),
        (date(2026, 12, 31), True),
        (date(2027, 1, 1), False),
        (date(2025, 12, 31), False),
    ],
)
def test_yearly_partition_is_open_while_the_year_lasts(today: date, expected: bool) -> None:
    assert Partition("1Day", 2026, None, "A").is_open(today) is expected


@pytest.mark.parametrize(
    ("today", "expected"),
    [
        (date(2026, 10, 1), True),
        (TODAY, True),
        (date(2026, 10, 31), True),
        (date(2026, 11, 1), False),
        (date(2026, 9, 30), False),
    ],
)
def test_monthly_partition_is_open_while_the_month_lasts(today: date, expected: bool) -> None:
    assert Partition("1Min", 2026, 10, "A").is_open(today) is expected


def test_minute_partition_without_a_month_is_invalid(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        Partition("1Min", 2026, None, "A").path(tmp_path)


def test_partitions_are_hashable_values() -> None:
    assert Partition("1Day", 2026, None, "A") == Partition("1Day", 2026, None, "A")
    assert len({Partition("1Day", 2026, None, "A"), Partition("1Day", 2026, None, "A")}) == 1


# =============================================================================================
# plan_partitions
# =============================================================================================


def test_daily_plan_covers_every_year_times_every_bucket_in_order() -> None:
    plan = plan_partitions("1Day", date(2024, 6, 1), date(2026, 2, 1), ["MSFT", "AAPL", "AMZN"])

    assert plan == [
        Partition("1Day", 2024, None, "A"),
        Partition("1Day", 2024, None, "M"),
        Partition("1Day", 2025, None, "A"),
        Partition("1Day", 2025, None, "M"),
        Partition("1Day", 2026, None, "A"),
        Partition("1Day", 2026, None, "M"),
    ]


def test_minute_plan_covers_every_month_touched_by_the_range() -> None:
    plan = plan_partitions("1Min", date(2025, 11, 15), date(2026, 2, 3), ["AAPL"])

    assert plan == [
        Partition("1Min", 2025, 11, "A"),
        Partition("1Min", 2025, 12, "A"),
        Partition("1Min", 2026, 1, "A"),
        Partition("1Min", 2026, 2, "A"),
    ]


def test_plan_for_a_single_day_is_a_single_period() -> None:
    assert plan_partitions("1Day", TODAY, TODAY, ["AAPL"]) == [Partition("1Day", 2026, None, "A")]
    assert plan_partitions("1Min", TODAY, TODAY, ["AAPL"]) == [Partition("1Min", 2026, 10, "A")]


def test_plan_with_no_symbols_is_empty() -> None:
    assert plan_partitions("1Day", date(2020, 1, 1), TODAY, []) == []


def test_plan_rejects_a_start_after_the_end() -> None:
    with pytest.raises(ValueError):
        plan_partitions("1Day", date(2026, 10, 6), TODAY, ["AAPL"])


def test_plan_uses_the_dot_spelling_bucket() -> None:
    assert plan_partitions("1Day", TODAY, TODAY, ["BRK.B", "F.PRD"]) == [
        Partition("1Day", 2026, None, "B"),
        Partition("1Day", 2026, None, "F"),
    ]


# =============================================================================================
# write_partition
# =============================================================================================


def test_write_partition_produces_the_documented_schema_and_rows(tmp_path: Path) -> None:
    path = tmp_path / "bars_1d" / "year=2026" / "bucket=A.parquet"
    bars = [
        make_bar("AMZN", T_DAY, close=3.0),
        make_bar("AAPL", T_DAY + timedelta(days=1), close=2.0),
        make_bar("AAPL", T_DAY, close=1.0),
    ]

    count = write_partition(path, bars, "1Day", FETCHED_AT)

    assert count == 3
    table = pq.read_table(path)
    assert table.schema.equals(BAR_SCHEMA, check_metadata=False)
    rows = table.to_pylist()
    assert [(r["symbol"], r["t"], r["close"]) for r in rows] == [
        ("AAPL", T_DAY, 1.0),
        ("AAPL", T_DAY + timedelta(days=1), 2.0),
        ("AMZN", T_DAY, 3.0),
    ]
    assert [r["session_date"] for r in rows] == [
        date(2026, 10, 2),
        date(2026, 10, 3),
        date(2026, 10, 2),
    ]
    assert all(r["fetched_at"] == FETCHED_AT for r in rows)
    assert rows[0] == {
        "symbol": "AAPL",
        "t": T_DAY,
        "session_date": date(2026, 10, 2),
        "open": 1.0,
        "high": 2.0,
        "low": 0.5,
        "close": 1.0,
        "volume": 100,
        "trade_count": 7,
        "vwap": 1.2,
        "available_at": AVAILABLE_DAY,
        "fetched_at": FETCHED_AT,
    }


def test_write_partition_computes_minute_availability(tmp_path: Path) -> None:
    path = tmp_path / "bucket=A.parquet"
    write_partition(
        path, [make_bar("AAPL", T_MIN), make_bar("AAPL", T_MIN + MINUTE)], "1Min", FETCHED_AT
    )

    rows = pq.read_table(path).to_pylist()

    assert [r["available_at"] for r in rows] == [T_MIN + MINUTE, T_MIN + 2 * MINUTE]
    assert [r["session_date"] for r in rows] == [date(2026, 10, 2), date(2026, 10, 2)]


def test_write_partition_accepts_any_iterable(tmp_path: Path) -> None:
    def gen() -> Iterator[Bar]:
        yield make_bar("AAPL", T_DAY)
        yield make_bar("MSFT", T_DAY)

    assert write_partition(tmp_path / "x.parquet", gen(), "1Day", FETCHED_AT) == 2


def test_duplicate_symbol_and_time_keeps_the_last_occurrence(tmp_path: Path) -> None:
    path = tmp_path / "bucket=A.parquet"
    bars = [make_bar("AAPL", T_DAY, close=1.0), make_bar("AAPL", T_DAY, close=99.0)]

    count = write_partition(path, bars, "1Day", FETCHED_AT)

    assert count == 1
    assert pq.read_table(path).column("close").to_pylist() == [99.0]


def test_zero_bars_still_writes_an_empty_file_with_the_schema(tmp_path: Path) -> None:
    path = tmp_path / "bars_1d" / "year=2019" / "bucket=Q.parquet"

    count = write_partition(path, [], "1Day", FETCHED_AT)

    assert count == 0
    table = pq.read_table(path)
    assert table.num_rows == 0
    assert table.schema.equals(BAR_SCHEMA, check_metadata=False)


def test_write_partition_merges_into_an_existing_file(tmp_path: Path) -> None:
    # Not an overwrite: the row the second call did not carry stays (see the merge section).
    path = tmp_path / "bucket=A.parquet"
    write_partition(path, [make_bar("AAPL", T_DAY), make_bar("AMZN", T_DAY)], "1Day", FETCHED_AT)

    count = write_partition(path, [make_bar("AAPL", T_DAY)], "1Day", FETCHED_AT)

    assert count == 2
    assert pq.read_table(path).column("symbol").to_pylist() == ["AAPL", "AMZN"]
    assert leftovers(tmp_path) == ["bucket=A.parquet"]


def test_write_partition_rejects_naive_fetched_at(tmp_path: Path) -> None:
    path = tmp_path / "bucket=A.parquet"

    with pytest.raises(ValueError):
        write_partition(path, [make_bar("AAPL", T_DAY)], "1Day", datetime(2026, 10, 5, 1, 0))

    assert leftovers(tmp_path) == []


def test_failure_while_iterating_bars_leaves_no_file_and_no_temp_file(tmp_path: Path) -> None:
    path = tmp_path / "bars_1d" / "year=2026" / "bucket=A.parquet"

    def exploding() -> Iterator[Bar]:
        yield make_bar("AAPL", T_DAY)
        raise RuntimeError("connection reset")

    with pytest.raises(RuntimeError, match="connection reset"):
        write_partition(path, exploding(), "1Day", FETCHED_AT)

    assert not path.exists()
    assert leftovers(tmp_path) == []


def test_failure_while_iterating_keeps_the_previous_file_intact(tmp_path: Path) -> None:
    path = tmp_path / "bucket=A.parquet"
    write_partition(path, [make_bar("AAPL", T_DAY, close=1.0)], "1Day", FETCHED_AT)

    def exploding() -> Iterator[Bar]:
        yield make_bar("AAPL", T_DAY, close=2.0)
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        write_partition(path, exploding(), "1Day", FETCHED_AT)

    assert pq.read_table(path).column("close").to_pylist() == [1.0]
    assert leftovers(tmp_path) == ["bucket=A.parquet"]


# --- leakage: the stored visibility time must be after the bar's own time ----------------------


def test_every_stored_bar_becomes_available_strictly_after_its_timestamp(tmp_path: Path) -> None:
    day_path = tmp_path / "d.parquet"
    min_path = tmp_path / "m.parquet"
    write_partition(
        day_path,
        [daily("AAPL", date(2026, 10, 2)), daily("AAPL", date(2026, 1, 5))],
        "1Day",
        FETCHED_AT,
    )
    write_partition(min_path, [make_bar("AAPL", T_MIN)], "1Min", FETCHED_AT)

    for path in (day_path, min_path):
        for row in pq.read_table(path).to_pylist():
            assert row["available_at"] > row["t"], row


def test_daily_bar_is_not_available_at_its_midnight_timestamp(tmp_path: Path) -> None:
    # Alpaca stamps a daily bar at 00:00 New York of its session. Treating that as the moment
    # the bar is known would expose the day's close to a 09:30 scan; and because the daily
    # volume counts post-market trades, even 16:00 would be too early.
    path = tmp_path / "d.parquet"
    write_partition(path, [daily("AAPL", date(2026, 10, 2))], "1Day", FETCHED_AT)

    [row] = pq.read_table(path).to_pylist()

    assert row["t"] == T_DAY
    assert row["available_at"] == AVAILABLE_DAY
    assert row["available_at"] == datetime(2026, 10, 2, 20, 0, tzinfo=ZoneInfo("America/New_York"))


# =============================================================================================
# write_partition merges: a refetch may add bars, never remove or overwrite stored ones
# =============================================================================================
# Alpaca deletes a delisted security's history once its symbol is reused and files the new
# holder's history under the symbol (measured 2026-10-16: VJET's 2020-2024 bars became copies
# of RMED's; AXNX's 2020-2024 history vanished and 2025 copies of MIMI appeared). It also
# revises bars after the fact (888 symbols' 2026-10-05 volumes changed after 20:00 ET). The
# first-fetched row is what a reader saw at the time, so it stays; discarding a partition means
# deleting its file.

LATER = FETCHED_AT + timedelta(days=1)


def rows_of(path: Path) -> list[dict[str, object]]:
    return pq.read_table(path).to_pylist()


def test_merge_adds_new_keys_and_keeps_existing_rows_unchanged(tmp_path: Path) -> None:
    # The VJET case: the second fetch answers the same (symbol, t) with different OHLCV (RMED's
    # bar wearing VJET's label) plus a bar the first fetch did not have.
    path = tmp_path / "bucket=V.parquet"
    write_partition(
        path, [daily("VJET", date(2024, 1, 5), close=1.0, volume=100)], "1Day", FETCHED_AT
    )
    [before] = rows_of(path)

    count = write_partition(
        path,
        [
            daily("VJET", date(2024, 1, 5), close=99.0, volume=5_000),
            daily("VJET", date(2024, 1, 8), close=2.0),
        ],
        "1Day",
        LATER,
    )

    assert count == 2
    rows = rows_of(path)
    assert rows[0] == before  # every column, fetched_at included
    assert [(r["symbol"], r["session_date"], r["close"]) for r in rows] == [
        ("VJET", date(2024, 1, 5), 1.0),
        ("VJET", date(2024, 1, 8), 2.0),
    ]
    assert [r["fetched_at"] for r in rows] == [FETCHED_AT, LATER]


def test_merge_keeps_rows_the_second_fetch_no_longer_returns(tmp_path: Path) -> None:
    # The AXNX case: the second fetch has nothing for AXNX any more. Its rows stay.
    path = tmp_path / "bucket=A.parquet"
    write_partition(
        path, [daily("AXNX", date(2024, 1, 5)), daily("AAPL", date(2024, 1, 5))], "1Day", FETCHED_AT
    )

    count = write_partition(path, [daily("AAPL", date(2024, 1, 5))], "1Day", LATER)

    assert count == 2
    assert [(r["symbol"], r["session_date"]) for r in rows_of(path)] == [
        ("AAPL", date(2024, 1, 5)),
        ("AXNX", date(2024, 1, 5)),
    ]
    assert leftovers(tmp_path) == ["bucket=A.parquet"]


def test_merged_file_is_sorted_by_symbol_and_time(tmp_path: Path) -> None:
    path = tmp_path / "bucket=A.parquet"
    write_partition(
        path, [daily("MSFT", date(2024, 1, 5)), daily("AAPL", date(2024, 1, 8))], "1Day", FETCHED_AT
    )

    write_partition(
        path, [daily("AMZN", date(2024, 1, 5)), daily("AAPL", date(2024, 1, 5))], "1Day", LATER
    )

    assert [(r["symbol"], r["session_date"]) for r in rows_of(path)] == [
        ("AAPL", date(2024, 1, 5)),
        ("AAPL", date(2024, 1, 8)),
        ("AMZN", date(2024, 1, 5)),
        ("MSFT", date(2024, 1, 5)),
    ]


def test_merge_updates_the_fetched_at_metadata_to_the_latest_fetch(tmp_path: Path) -> None:
    path = tmp_path / "bucket=A.parquet"
    write_partition(path, [make_bar("AAPL", T_DAY)], "1Day", FETCHED_AT)

    write_partition(path, [make_bar("AAPL", T_DAY + timedelta(days=1))], "1Day", LATER)

    assert partition_fetched_at(path) == LATER
    assert pq.read_table(path).column("fetched_at").to_pylist() == [FETCHED_AT, LATER]


def test_merge_returns_the_total_row_count_not_the_incoming_one(tmp_path: Path) -> None:
    path = tmp_path / "bucket=A.parquet"
    write_partition(path, [make_bar("AAPL", T_DAY), make_bar("AMZN", T_DAY)], "1Day", FETCHED_AT)

    count = write_partition(
        path, [make_bar("AMZN", T_DAY), make_bar("AAPL", T_DAY + timedelta(days=1))], "1Day", LATER
    )

    assert count == 3
    assert pq.read_table(path).num_rows == 3


# --- boundaries --------------------------------------------------------------------------------


def test_merging_nothing_keeps_the_rows_and_still_stamps_the_new_fetched_at(tmp_path: Path) -> None:
    # A closed year refetched and answered empty (Alpaca pruned the symbol): rows stay, and the
    # metadata records that the refetch happened.
    path = tmp_path / "bucket=A.parquet"
    write_partition(path, [make_bar("AAPL", T_DAY), make_bar("AMZN", T_DAY)], "1Day", FETCHED_AT)
    before = rows_of(path)

    count = write_partition(path, [], "1Day", LATER)

    assert count == 2
    assert rows_of(path) == before
    assert partition_fetched_at(path) == LATER
    assert leftovers(tmp_path) == ["bucket=A.parquet"]


def test_merge_still_drops_incoming_bars_not_yet_final(tmp_path: Path) -> None:
    # Refetch at 10:00 New York on 2026-10-02: yesterday's bar is new and final, today's is
    # still trading. Only the first joins the stored 09-30 bar.
    path = tmp_path / "bucket=A.parquet"
    write_partition(path, [daily("AAPL", date(2026, 9, 30))], "1Day", FETCHED_AT)
    during_session = datetime(2026, 10, 2, 14, 0, tzinfo=UTC)

    count = write_partition(
        path,
        [daily("AAPL", date(2026, 10, 1)), daily("AAPL", date(2026, 10, 2))],
        "1Day",
        during_session,
    )

    assert count == 2
    assert [r["session_date"] for r in rows_of(path)] == [date(2026, 9, 30), date(2026, 10, 1)]
    assert partition_fetched_at(path) == during_session


def test_merge_never_drops_a_stored_row_even_when_fetched_at_runs_backwards(
    tmp_path: Path,
) -> None:
    # The not-yet-final filter applies to incoming bars only. A stored 10-02 bar (final at
    # 20:00 ET that day) stays even if a later call claims a fetched_at from the morning of
    # 10-02, as a skewed clock might.
    path = tmp_path / "bucket=A.parquet"
    write_partition(path, [daily("AAPL", date(2026, 10, 2))], "1Day", FETCHED_AT)
    before = rows_of(path)

    count = write_partition(path, [], "1Day", datetime(2026, 10, 2, 14, 0, tzinfo=UTC))

    assert count == 1
    assert rows_of(path) == before


def test_merge_into_a_file_written_without_metadata_works(tmp_path: Path) -> None:
    # Older files carry fetched_at only in the column; their schema still matches.
    path = tmp_path / "bucket=A.parquet"
    write_partition(path, [make_bar("AAPL", T_DAY)], "1Day", FETCHED_AT)
    pq.write_table(pq.read_table(path).replace_schema_metadata({}), path)

    count = write_partition(path, [make_bar("AMZN", T_DAY)], "1Day", LATER)

    assert count == 2
    assert partition_fetched_at(path) == LATER


def test_writing_to_a_deleted_path_starts_the_file_over(tmp_path: Path) -> None:
    # Discarding a partition is done by deleting its file: nothing from before comes back.
    path = tmp_path / "bucket=A.parquet"
    write_partition(path, [make_bar("AAPL", T_DAY), make_bar("AMZN", T_DAY)], "1Day", FETCHED_AT)
    path.unlink()

    count = write_partition(path, [make_bar("AAPL", T_DAY + timedelta(days=1))], "1Day", LATER)

    assert count == 1
    assert [(r["symbol"], r["t"]) for r in rows_of(path)] == [("AAPL", T_DAY + timedelta(days=1))]
    assert partition_fetched_at(path) == LATER


def test_duplicate_keys_inside_one_fetch_still_collapse_before_the_merge(tmp_path: Path) -> None:
    path = tmp_path / "bucket=A.parquet"
    write_partition(path, [make_bar("AMZN", T_DAY)], "1Day", FETCHED_AT)

    count = write_partition(
        path,
        [make_bar("AAPL", T_DAY, close=1.0), make_bar("AAPL", T_DAY, close=99.0)],
        "1Day",
        LATER,
    )

    assert count == 2
    assert [(r["symbol"], r["close"]) for r in rows_of(path)] == [("AAPL", 99.0), ("AMZN", 1.5)]


def wrong_schema_file(path: Path, *, drop: str | None = None, retype: str | None = None) -> None:
    """A Parquet file at ``path`` whose schema is BAR_SCHEMA minus ``drop`` or with ``retype``
    turned into a string column."""
    fields = [
        pa.field(f.name, pa.string() if f.name == retype else f.type)
        for f in BAR_SCHEMA
        if f.name != drop
    ]
    pq.write_table(pa.schema(fields).empty_table(), path)


@pytest.mark.parametrize(
    "kwargs",
    [{"drop": "vwap"}, {"retype": "volume"}],
    ids=["missing_column", "wrong_type"],
)
def test_merging_into_a_file_with_another_schema_refuses_and_names_the_file(
    tmp_path: Path, kwargs: dict[str, str]
) -> None:
    path = tmp_path / "bucket=A.parquet"
    wrong_schema_file(path, **kwargs)
    before = path.read_bytes()

    with pytest.raises(ValueError, match="delete") as info:
        write_partition(path, [make_bar("AAPL", T_DAY)], "1Day", FETCHED_AT)

    assert str(path) in str(info.value)
    assert path.read_bytes() == before
    assert leftovers(tmp_path) == ["bucket=A.parquet"]


def test_merge_refuses_a_file_with_the_columns_in_another_order(tmp_path: Path) -> None:
    path = tmp_path / "bucket=A.parquet"
    pq.write_table(pa.schema(list(BAR_SCHEMA)[::-1]).empty_table(), path)

    with pytest.raises(ValueError, match="delete"):
        write_partition(path, [make_bar("AAPL", T_DAY)], "1Day", FETCHED_AT)


# --- atomicity: a failed merge leaves the stored file exactly as it was ----------------------


def test_failure_while_writing_the_merged_file_keeps_the_old_file_and_no_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "bucket=A.parquet"
    write_partition(path, [make_bar("AAPL", T_DAY, close=1.0)], "1Day", FETCHED_AT)
    before = path.read_bytes()

    def disk_full(table: pa.Table, where: Path, *args: object, **kwargs: object) -> None:
        Path(where).write_bytes(b"partial")
        raise OSError("No space left on device")

    monkeypatch.setattr("asof.ingest.bars.pq.write_table", disk_full)

    with pytest.raises(OSError, match="No space left"):
        write_partition(path, [make_bar("AAPL", T_DAY + timedelta(days=1))], "1Day", LATER)

    assert path.read_bytes() == before
    assert leftovers(tmp_path) == ["bucket=A.parquet"]


# --- through backfill: force on a complete partition appends, never replaces ----------------


def test_forced_refetch_keeps_bars_the_second_answer_lacks(tmp_path: Path) -> None:
    # The AXNX case: 2024 complete with AXNX and AAPL; a year later Alpaca has no AXNX bars at
    # all (measured 2026-10-16) but a new AAPL bar. The forced run adds the AAPL bar and keeps
    # AXNX.
    first = FakeBarClient([daily("AXNX", date(2024, 1, 5)), daily("AAPL", date(2024, 1, 5))])
    backfill(
        first,
        tmp_path,
        "1Day",
        date(2024, 1, 1),
        date(2024, 12, 31),
        ["AAPL", "AXNX"],
        now=lambda: NOW_2027,
    )
    path = tmp_path / "bars_1d" / "year=2024" / "bucket=A.parquet"
    before = rows_of(path)

    second = FakeBarClient([daily("AAPL", date(2024, 1, 5)), daily("AAPL", date(2024, 1, 8))])
    later = NOW_2027 + timedelta(days=30)
    report = backfill(
        second,
        tmp_path,
        "1Day",
        date(2024, 1, 1),
        date(2024, 12, 31),
        ["AAPL", "AXNX"],
        force=True,
        now=lambda: later,
    )

    assert report.written == [Partition("1Day", 2024, None, "A")]
    assert report.rows == 3
    rows = rows_of(path)
    assert [r for r in rows if r["fetched_at"] == NOW_2027] == before
    assert [(r["symbol"], r["session_date"]) for r in rows] == [
        ("AAPL", date(2024, 1, 5)),
        ("AAPL", date(2024, 1, 8)),
        ("AXNX", date(2024, 1, 5)),
    ]
    assert partition_fetched_at(path) == later


def test_forced_refetch_keeps_bars_the_second_answer_relabels(tmp_path: Path) -> None:
    # The VJET case: VJET's stored 2024 bars; a year later the same (symbol, t) keys come back
    # with RMED's prices (measured 2026-10-16). The stored prices stay; only the key the first
    # fetch did not have is added.
    first = FakeBarClient(
        [daily("VJET", date(2024, 1, 5), close=1.0), daily("VJET", date(2024, 1, 8), close=1.1)]
    )
    backfill(
        first,
        tmp_path,
        "1Day",
        date(2024, 1, 1),
        date(2024, 12, 31),
        ["VJET"],
        now=lambda: NOW_2027,
    )
    path = tmp_path / "bars_1d" / "year=2024" / "bucket=V.parquet"

    second = FakeBarClient(
        [
            daily("VJET", date(2024, 1, 5), close=50.0, volume=9_999),
            daily("VJET", date(2024, 1, 8), close=51.0, volume=9_999),
            daily("VJET", date(2024, 1, 9), close=52.0, volume=9_999),
        ]
    )
    report = backfill(
        second,
        tmp_path,
        "1Day",
        date(2024, 1, 1),
        date(2024, 12, 31),
        ["VJET"],
        force=True,
        now=lambda: NOW_2027 + timedelta(days=30),
    )

    assert report.rows == 3
    assert [(r["session_date"], r["close"], r["volume"]) for r in rows_of(path)] == [
        (date(2024, 1, 5), 1.0, 100),
        (date(2024, 1, 8), 1.1, 100),
        (date(2024, 1, 9), 52.0, 9_999),
    ]


def test_daily_refetch_of_the_open_year_only_appends(tmp_path: Path) -> None:
    # The open year is refetched on every run. A revised volume for a day already stored
    # (888 symbols' 2026-10-05 volumes changed after 20:00 ET) must not replace the value a
    # reader saw at the time.
    day_one = datetime(2026, 10, 6, 1, 0, tzinfo=UTC)
    first = FakeBarClient([daily("AAPL", date(2026, 10, 5), volume=100)])
    backfill(
        first, tmp_path, "1Day", date(2026, 1, 1), date(2026, 10, 6), ["AAPL"], now=lambda: day_one
    )
    path = tmp_path / "bars_1d" / "year=2026" / "bucket=A.parquet"

    day_two = datetime(2026, 10, 7, 1, 0, tzinfo=UTC)
    second = FakeBarClient(
        [daily("AAPL", date(2026, 10, 5), volume=150), daily("AAPL", date(2026, 10, 6), volume=200)]
    )
    report = backfill(
        second, tmp_path, "1Day", date(2026, 1, 1), date(2026, 10, 7), ["AAPL"], now=lambda: day_two
    )

    assert report.written == [Partition("1Day", 2026, None, "A")]
    assert report.rows == 2
    assert [(r["session_date"], r["volume"], r["fetched_at"]) for r in rows_of(path)] == [
        (date(2026, 10, 5), 100, day_one),
        (date(2026, 10, 6), 200, day_two),
    ]
    assert partition_fetched_at(path) == day_two


# =============================================================================================
# backfill
# =============================================================================================


def two_symbol_bars() -> list[Bar]:
    return [
        daily("AAPL", date(2025, 12, 31)),
        daily("AAPL", date(2026, 1, 2)),
        daily("AAPL", date(2026, 10, 2)),
        daily("MSFT", date(2026, 1, 2)),
    ]


def test_backfill_writes_one_file_per_partition_and_reports_rows(tmp_path: Path) -> None:
    client = FakeBarClient(two_symbol_bars())

    report = backfill(
        client,
        tmp_path,
        "1Day",
        date(2026, 1, 1),
        TODAY,
        ["AAPL", "MSFT"],
        now=lambda: FETCHED_AT,
    )

    assert isinstance(report, BackfillReport)
    assert report.written == [
        Partition("1Day", 2026, None, "A"),
        Partition("1Day", 2026, None, "M"),
    ]
    assert report.skipped == []
    assert report.rows == 3
    assert leftovers(tmp_path) == [
        "bars_1d/year=2026/bucket=A.parquet",
        "bars_1d/year=2026/bucket=M.parquet",
    ]
    apple = pq.read_table(tmp_path / "bars_1d" / "year=2026" / "bucket=A.parquet").to_pylist()
    assert [(r["symbol"], r["session_date"]) for r in apple] == [
        ("AAPL", date(2026, 1, 2)),
        ("AAPL", date(2026, 10, 2)),
    ]
    assert all(r["fetched_at"] == FETCHED_AT for r in apple)


def test_backfill_requests_each_bucket_over_the_partition_bounds(tmp_path: Path) -> None:
    client = FakeBarClient()

    backfill(
        client,
        tmp_path,
        "1Day",
        date(2025, 6, 1),
        date(2026, 3, 1),
        ["AAPL", "MSFT"],
        now=lambda: FETCHED_AT,
    )

    assert [(c.symbols, c.timeframe, c.start, c.end) for c in client.calls] == [
        (("AAPL",), "1Day", *Partition("1Day", 2025, None, "A").bounds()),
        (("MSFT",), "1Day", *Partition("1Day", 2025, None, "M").bounds()),
        (("AAPL",), "1Day", *open_bounds(Partition("1Day", 2026, None, "A"))),
        (("MSFT",), "1Day", *open_bounds(Partition("1Day", 2026, None, "M"))),
    ]


def test_backfill_splits_a_bucket_into_batches(tmp_path: Path) -> None:
    symbols = [f"A{i:03d}" for i in range(250)] + ["MSFT"]
    client = FakeBarClient()

    backfill(
        client,
        tmp_path,
        "1Day",
        TODAY,
        TODAY,
        symbols,
        batch_size=100,
        now=lambda: FETCHED_AT,
    )

    a_calls = [c for c in client.calls if c.symbols[0].startswith("A")]
    assert [len(c.symbols) for c in a_calls] == [100, 100, 50]
    requested = [s for c in a_calls for s in c.symbols]
    assert sorted(requested) == sorted(symbols[:250])
    assert len(set(requested)) == 250
    assert all(c.start == a_calls[0].start and c.end == a_calls[0].end for c in a_calls)
    assert [c.symbols for c in client.calls if not c.symbols[0].startswith("A")] == [("MSFT",)]


def test_backfill_handles_minute_partitions_by_month(tmp_path: Path) -> None:
    client = FakeBarClient(
        [make_bar("AAPL", T_MIN), make_bar("AAPL", datetime(2026, 9, 30, 13, 30, tzinfo=UTC))]
    )

    report = backfill(
        client,
        tmp_path,
        "1Min",
        date(2026, 9, 1),
        TODAY,
        ["AAPL"],
        now=lambda: FETCHED_AT,
    )

    assert report.written == [Partition("1Min", 2026, 9, "A"), Partition("1Min", 2026, 10, "A")]
    assert [(c.start, c.end) for c in client.calls] == [
        Partition("1Min", 2026, 9, "A").bounds(),
        open_bounds(Partition("1Min", 2026, 10, "A")),
    ]
    sep = pq.read_table(tmp_path / "bars_1m" / "year=2026" / "month=09" / "bucket=A.parquet")
    oct_ = pq.read_table(tmp_path / "bars_1m" / "year=2026" / "month=10" / "bucket=A.parquet")
    assert sep.column("t").to_pylist() == [datetime(2026, 9, 30, 13, 30, tzinfo=UTC)]
    assert oct_.column("t").to_pylist() == [T_MIN]


def test_backfill_keeps_only_bars_whose_session_falls_inside_the_partition(tmp_path: Path) -> None:
    # A sloppy server answers with bars outside the requested window; the 2026 file must not
    # contain a 2025 session, and the stray bar must not be lost into the wrong year either.
    client = FakeBarClient(two_symbol_bars(), respect_range=False)

    report = backfill(
        client,
        tmp_path,
        "1Day",
        date(2026, 1, 1),
        TODAY,
        ["AAPL"],
        now=lambda: FETCHED_AT,
    )

    rows = pq.read_table(tmp_path / "bars_1d" / "year=2026" / "bucket=A.parquet").to_pylist()
    assert [r["session_date"] for r in rows] == [date(2026, 1, 2), date(2026, 10, 2)]
    assert report.rows == 2


# --- boundaries --------------------------------------------------------------------------------


def test_closed_partitions_are_skipped_and_open_ones_refetched(tmp_path: Path) -> None:
    client = FakeBarClient(two_symbol_bars())
    first = backfill(
        client,
        tmp_path,
        "1Day",
        date(2025, 1, 1),
        TODAY,
        ["AAPL"],
        now=lambda: FETCHED_AT,
    )
    assert first.written == [Partition("1Day", 2025, None, "A"), Partition("1Day", 2026, None, "A")]
    client.calls.clear()

    second = backfill(
        client,
        tmp_path,
        "1Day",
        date(2025, 1, 1),
        TODAY,
        ["AAPL"],
        now=lambda: FETCHED_AT,
    )

    assert second.skipped == [Partition("1Day", 2025, None, "A")]
    assert second.written == [Partition("1Day", 2026, None, "A")]
    assert second.rows == 2
    assert [(c.start, c.end) for c in client.calls] == [
        open_bounds(Partition("1Day", 2026, None, "A"))
    ]


def test_force_refetches_closed_partitions(tmp_path: Path) -> None:
    client = FakeBarClient(two_symbol_bars())
    backfill(
        client,
        tmp_path,
        "1Day",
        date(2025, 1, 1),
        TODAY,
        ["AAPL"],
        now=lambda: FETCHED_AT,
    )
    client.calls.clear()

    report = backfill(
        client,
        tmp_path,
        "1Day",
        date(2025, 1, 1),
        TODAY,
        ["AAPL"],
        force=True,
        now=lambda: FETCHED_AT,
    )

    assert report.skipped == []
    assert report.written == [
        Partition("1Day", 2025, None, "A"),
        Partition("1Day", 2026, None, "A"),
    ]
    assert len(client.calls) == 2


def test_a_partition_far_in_the_past_is_skipped_without_an_explicit_today(tmp_path: Path) -> None:
    client = FakeBarClient()
    backfill(
        client,
        tmp_path,
        "1Day",
        date(2019, 1, 1),
        date(2019, 12, 31),
        ["AAPL"],
        now=lambda: FETCHED_AT,
    )
    client.calls.clear()

    report = backfill(
        client,
        tmp_path,
        "1Day",
        date(2019, 1, 1),
        date(2019, 12, 31),
        ["AAPL"],
        now=lambda: FETCHED_AT,
    )

    assert report.skipped == [Partition("1Day", 2019, None, "A")]
    assert client.calls == []


def test_bucket_with_no_bars_still_gets_a_file_so_it_is_not_refetched(tmp_path: Path) -> None:
    client = FakeBarClient()

    report = backfill(
        client,
        tmp_path,
        "1Day",
        date(2025, 1, 1),
        date(2025, 12, 31),
        ["ZZZZ"],
        now=lambda: FETCHED_AT,
    )

    path = tmp_path / "bars_1d" / "year=2025" / "bucket=Z.parquet"
    assert report.written == [Partition("1Day", 2025, None, "Z")]
    assert report.rows == 0
    assert pq.read_table(path).num_rows == 0


def test_backfill_with_no_symbols_does_nothing(tmp_path: Path) -> None:
    client = FakeBarClient()

    report = backfill(client, tmp_path, "1Day", date(2025, 1, 1), TODAY, [], now=lambda: FETCHED_AT)

    assert (report.written, report.skipped, report.rows) == ([], [], 0)
    assert client.calls == []
    assert leftovers(tmp_path) == []


def test_backfill_rejects_a_start_after_the_end(tmp_path: Path) -> None:
    client = FakeBarClient()

    with pytest.raises(ValueError):
        backfill(
            client,
            tmp_path,
            "1Day",
            TODAY,
            date(2026, 1, 1),
            ["AAPL"],
            now=lambda: FETCHED_AT,
        )

    assert client.calls == []


def test_backfill_rejects_a_naive_now(tmp_path: Path) -> None:
    client = FakeBarClient(two_symbol_bars())

    with pytest.raises(ValueError):
        backfill(
            client,
            tmp_path,
            "1Day",
            date(2026, 1, 1),
            TODAY,
            ["AAPL"],
            now=lambda: datetime(2026, 10, 5, 1, 0),
        )

    assert leftovers(tmp_path) == []


def test_backfill_failure_mid_partition_leaves_earlier_files_and_no_partial_one(
    tmp_path: Path,
) -> None:
    class Flaky(FakeBarClient):
        def iter_bars(self, symbols, timeframe, start, end):  # type: ignore[override]
            if "MSFT" in symbols:
                raise RuntimeError("rate limit exhausted")
            return super().iter_bars(symbols, timeframe, start, end)

    client = Flaky(two_symbol_bars())

    with pytest.raises(RuntimeError):
        backfill(
            client,
            tmp_path,
            "1Day",
            date(2026, 1, 1),
            TODAY,
            ["AAPL", "MSFT"],
            now=lambda: FETCHED_AT,
        )

    assert leftovers(tmp_path) == ["bars_1d/year=2026/bucket=A.parquet"]


# =============================================================================================
# completeness: a file counts as complete only if fetched after its period ended
# =============================================================================================


def test_write_partition_records_fetched_at_in_the_file_metadata(tmp_path: Path) -> None:
    path = tmp_path / "bucket=A.parquet"
    write_partition(path, [make_bar("AAPL", T_DAY)], "1Day", FETCHED_AT)

    assert partition_fetched_at(path) == FETCHED_AT


def test_partition_fetched_at_falls_back_to_the_column_for_files_without_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bucket=A.parquet"
    write_partition(path, [make_bar("AAPL", T_DAY)], "1Day", FETCHED_AT)
    table = pq.read_table(path).replace_schema_metadata({})
    pq.write_table(table, path)

    assert partition_fetched_at(path) == FETCHED_AT


def test_bars_not_final_at_fetch_time_are_not_stored(tmp_path: Path) -> None:
    # Fetched at 10:00 New York on 2026-10-02: that day's daily bar is still being traded.
    path = tmp_path / "bucket=A.parquet"
    fetched_at = datetime(2026, 10, 2, 14, 0, tzinfo=UTC)

    count = write_partition(
        path,
        [daily("AAPL", date(2026, 10, 1)), daily("AAPL", date(2026, 10, 2))],
        "1Day",
        fetched_at,
    )

    assert count == 1
    assert pq.read_table(path).column("session_date").to_pylist() == [date(2026, 10, 1)]


def test_minute_bar_still_in_progress_at_fetch_time_is_not_stored(tmp_path: Path) -> None:
    path = tmp_path / "bucket=A.parquet"

    count = write_partition(
        path, [make_bar("AAPL", T_MIN), make_bar("AAPL", T_MIN + MINUTE)], "1Min", T_MIN + MINUTE
    )

    assert count == 1
    assert pq.read_table(path).column("t").to_pylist() == [T_MIN]


def test_file_written_on_the_last_day_of_its_period_is_redone_next_period(tmp_path: Path) -> None:
    # 2026-12-31 10:00 New York: the 2026 file is written, but the year is not over. On
    # 2027-01-02 the same partition must be fetched again, not skipped.
    client = FakeBarClient([daily("AAPL", date(2026, 12, 30)), daily("AAPL", date(2026, 12, 31))])
    last_day = datetime(2026, 12, 31, 10, 0, tzinfo=ET)
    first = backfill(
        client,
        tmp_path,
        "1Day",
        date(2026, 1, 1),
        date(2026, 12, 31),
        ["AAPL"],
        now=lambda: last_day,
    )
    assert first.written == [Partition("1Day", 2026, None, "A")]
    assert first.rows == 1  # the 12-31 bar was still trading
    client.calls.clear()

    second = backfill(
        client,
        tmp_path,
        "1Day",
        date(2026, 1, 1),
        date(2026, 12, 31),
        ["AAPL"],
        now=lambda: datetime(2027, 1, 2, 10, 0, tzinfo=ET),
    )

    assert second.skipped == []
    assert second.written == [Partition("1Day", 2026, None, "A")]
    assert second.rows == 2
    assert len(client.calls) == 1


def test_is_complete_requires_a_fetch_after_the_period_end(tmp_path: Path) -> None:
    partition = Partition("1Day", 2025, None, "A")
    path = partition.path(tmp_path)
    assert is_complete(partition, path) is False

    write_partition(path, [], "1Day", datetime(2025, 12, 31, 23, 0, tzinfo=ET))
    assert is_complete(partition, path) is False

    write_partition(path, [], "1Day", datetime(2026, 1, 1, 0, 0, tzinfo=ET))
    assert is_complete(partition, path) is True


def test_open_partition_request_stops_short_of_the_recent_data_edge(tmp_path: Path) -> None:
    # The free tier refuses SIP data from the last 15 minutes, so the request end must not
    # reach ``now``; closed periods are unaffected.
    client = FakeBarClient()
    now = datetime(2026, 10, 5, 15, 0, tzinfo=UTC)

    backfill(
        client, tmp_path, "1Day", date(2025, 1, 1), date(2026, 10, 5), ["AAPL"], now=lambda: now
    )

    [closed, open_] = client.calls
    assert (closed.start, closed.end) == Partition("1Day", 2025, None, "A").bounds()
    assert open_.start == Partition("1Day", 2026, None, "A").bounds()[0]
    assert open_.end == now - timedelta(minutes=16)


def test_bar_exactly_at_the_period_end_goes_to_the_next_partition(tmp_path: Path) -> None:
    # Alpaca's ``end`` is inclusive, so a bar stamped at the next period's first instant can
    # come back with this period's request; it belongs to the next file.
    boundary = daily("AAPL", date(2027, 1, 1))
    client = FakeBarClient([daily("AAPL", date(2026, 12, 31)), boundary])

    backfill(
        client,
        tmp_path,
        "1Day",
        date(2026, 1, 1),
        date(2027, 1, 1),
        ["AAPL"],
        now=lambda: datetime(2027, 6, 1, tzinfo=UTC),
    )

    y2026 = pq.read_table(tmp_path / "bars_1d" / "year=2026" / "bucket=A.parquet")
    y2027 = pq.read_table(tmp_path / "bars_1d" / "year=2027" / "bucket=A.parquet")
    assert y2026.column("session_date").to_pylist() == [date(2026, 12, 31)]
    assert y2027.column("session_date").to_pylist() == [date(2027, 1, 1)]


# =============================================================================================
# ticker chains: Alpaca files a security's whole history under whichever of its names it is
# asked for, and when old and new names share a request it keeps only the newest
# =============================================================================================


class ChainClient(FakeBarClient):
    """Mimics Alpaca: history of MMC (renamed MRSH) is served under whichever name is asked;
    if both are in one request, only MRSH comes back."""

    CHAIN = {"MRSH": "MMC"}  # new -> old

    def iter_bars(self, symbols, timeframe, start, end):  # type: ignore[override]
        self.calls.append(BarsCall(tuple(symbols), timeframe, start, end))
        wanted = set(symbols)
        out: list[Bar] = []
        for bar in self.bars:
            if start <= bar.t <= end:
                if bar.symbol in wanted:
                    out.append(bar)
                elif bar.symbol == "MMC" and "MRSH" in wanted:
                    out.append(bar.model_copy(update={"symbol": "MRSH"}))
        if "MMC" in wanted and "MRSH" in wanted:
            out = [b for b in out if b.symbol != "MMC"]
        return iter(out)


MMC_BARS = [daily("MMC", date(2026, 1, 5)), daily("MMC", date(2026, 2, 3))]
MRSH_BARS = [daily("MRSH", date(2026, 4, 1))]
WINDOWS = {"MMC": (date(2018, 1, 22), date(2026, 3, 29)), "MRSH": (date(2026, 2, 28), None)}


def test_symbols_without_a_window_are_kept_whole(tmp_path: Path) -> None:
    path = tmp_path / "bucket=S.parquet"
    count = write_partition(path, [daily("SPY", date(2026, 1, 5))], "1Day", NOW_2027)
    assert count == 1


def test_active_and_inactive_symbols_are_requested_in_separate_batches(tmp_path: Path) -> None:
    client = ChainClient(MMC_BARS + MRSH_BARS)

    backfill(
        client,
        tmp_path,
        "1Day",
        date(2026, 1, 1),
        date(2026, 12, 31),
        ["MMC", "MRSH", "MSFT"],
        now=lambda: NOW_2027,
        windows=WINDOWS,
        active={"MRSH", "MSFT"},
    )

    first_pass = [set(c.symbols) for c in client.calls[:2]]
    assert {"MRSH", "MSFT"} in first_pass
    assert {"MMC"} in first_pass
    # MRSH's request brings MMC's history back relabelled MRSH; since 4b-2b it is written as
    # returned (the security master hides it at read time).
    rows = pq.read_table(tmp_path / "bars_1d" / "year=2026" / "bucket=M.parquet").to_pylist()
    assert [(r["symbol"], r["session_date"]) for r in rows] == [
        ("MMC", date(2026, 1, 5)),
        ("MMC", date(2026, 2, 3)),
        ("MRSH", date(2026, 1, 5)),
        ("MRSH", date(2026, 2, 3)),
        ("MRSH", date(2026, 4, 1)),
    ]


def test_symbol_merged_away_in_a_batch_is_refetched_alone(tmp_path: Path) -> None:
    # Both names inactive (bankruptcy: SIVB -> SIVBQ) land in the same batch; SIVB comes back
    # empty and must get a request of its own, which Alpaca answers correctly.
    class Bankrupt(ChainClient):
        CHAIN = {"SIVBQ": "SIVB"}

        def iter_bars(self, symbols, timeframe, start, end):  # type: ignore[override]
            self.calls.append(BarsCall(tuple(symbols), timeframe, start, end))
            wanted = set(symbols)
            out = [b for b in self.bars if b.symbol in wanted and start <= b.t <= end]
            if "SIVB" in wanted and "SIVBQ" in wanted:
                out = [b.model_copy(update={"symbol": "SIVBQ"}) for b in out]
            return iter(out)

    # 2023 is the year both windows overlap (SIVB until April, SIVBQ from March).
    bars = [daily("SIVB", date(2023, 1, 5)), daily("SIVB", date(2023, 2, 3))]
    client = Bankrupt(bars)
    windows = {"SIVB": (date(2018, 1, 22), date(2023, 4, 5)), "SIVBQ": (date(2023, 3, 5), None)}

    report = backfill(
        client,
        tmp_path,
        "1Day",
        date(2023, 1, 1),
        date(2023, 12, 31),
        ["SIVB", "SIVBQ"],
        now=lambda: NOW_2027,
        windows=windows,
        active=set(),
    )

    assert [c.symbols for c in client.calls] == [("SIVB", "SIVBQ"), ("SIVB",)]
    # The batch answer (SIVB's bars labelled SIVBQ) is kept as returned; the solo request adds
    # the same bars under SIVB. Which label was in use when is the security master's call.
    rows = pq.read_table(tmp_path / "bars_1d" / "year=2023" / "bucket=S.parquet").to_pylist()
    assert [(r["symbol"], r["session_date"]) for r in rows] == [
        ("SIVB", date(2023, 1, 5)),
        ("SIVB", date(2023, 2, 3)),
        ("SIVBQ", date(2023, 1, 5)),
        ("SIVBQ", date(2023, 2, 3)),
    ]
    assert report.rows == 4


def test_symbols_whose_window_misses_the_period_are_not_requested(tmp_path: Path) -> None:
    client = FakeBarClient()
    windows = {"ELNK": (date(2018, 1, 22), date(2019, 6, 1)), "AAPL": (date(2018, 1, 22), None)}

    backfill(
        client,
        tmp_path,
        "1Day",
        date(2022, 1, 1),
        date(2022, 12, 31),
        ["AAPL", "ELNK", "SPY"],
        now=lambda: NOW_2027,
        windows=windows,
        active={"AAPL", "SPY"},
    )

    requested = {s for c in client.calls for s in c.symbols}
    assert requested == {"AAPL", "SPY"}


def test_zero_row_symbol_without_a_window_is_not_retried(tmp_path: Path) -> None:
    client = FakeBarClient()

    backfill(
        client,
        tmp_path,
        "1Day",
        date(2022, 1, 1),
        date(2022, 12, 31),
        ["ZZZZ"],
        now=lambda: NOW_2027,
    )

    assert [c.symbols for c in client.calls] == [("ZZZZ",)]


# =============================================================================================
# chains: a renamed name that answered nothing is asked for alone only when another name of
# the same security shared its request (that is the only case Alpaca hides it); the SEC window
# rule above is unchanged
# =============================================================================================
# Measured on real data: retrying every renamed name unconditionally costs 24,337 solo requests
# a year (about 145 min at the free-tier rate), twelve times that for monthly minute
# partitions, and Alpaca only hides an old name when a newer name of the same security is in
# the request (measured for MMC/MRSH and SIVB/SIVBQ). Symbols in ``chains`` that are not in
# the run are ignored; ``chains=None`` or empty leaves the second pass to the windows.


class PrunedRename(FakeBarClient):
    """Mimics Alpaca for SPLG -> SPYM (a real ETF rename; SPLG is gone from the asset list):
    asked alone, each name answers with its own bars; when both share a request, SPLG's bars
    come back labelled SPYM, as measured for MMC/MRSH. Whether that holds for ETFs is an
    assumption (see the module docstring)."""

    def iter_bars(self, symbols, timeframe, start, end):  # type: ignore[override]
        self.calls.append(BarsCall(tuple(symbols), timeframe, start, end))
        wanted = set(symbols)
        out = [b for b in self.bars if b.symbol in wanted and start <= b.t <= end]
        if "SPLG" in wanted and "SPYM" in wanted:
            out = [b.model_copy(update={"symbol": "SPYM"}) for b in out]
        return iter(out)


SPLG_BARS = [daily("SPLG", date(2024, 1, 5)), daily("SPLG", date(2024, 2, 5))]
SPYM_BARS = [daily("SPYM", date(2024, 6, 3))]
SPLG_CHAIN = [{"SPLG", "SPYM"}]


def test_chain_member_merged_away_in_a_shared_batch_is_refetched_alone(tmp_path: Path) -> None:
    # No SEC windows (ETFs) and neither name in ``active`` (SPLG pruned): both land in one
    # request and SPLG comes back as SPYM. Without ``chains`` the windows-only second pass
    # would never ask for SPLG again, because it has no window.
    client = PrunedRename(SPLG_BARS + SPYM_BARS)

    report = backfill(
        client,
        tmp_path,
        "1Day",
        date(2024, 1, 1),
        date(2024, 12, 31),
        ["SPLG", "SPYM"],
        now=lambda: NOW_2027,
        windows={},
        active=set(),
        chains=SPLG_CHAIN,
    )

    assert [c.symbols for c in client.calls] == [("SPLG", "SPYM"), ("SPLG",)]
    rows = pq.read_table(tmp_path / "bars_1d" / "year=2024" / "bucket=S.parquet").to_pylist()
    assert [(r["symbol"], r["session_date"]) for r in rows] == [
        ("SPLG", date(2024, 1, 5)),
        ("SPLG", date(2024, 2, 5)),
        ("SPYM", date(2024, 1, 5)),
        ("SPYM", date(2024, 2, 5)),
        ("SPYM", date(2024, 6, 3)),
    ]
    assert report.rows == 5


def test_chains_work_without_windows_or_active_and_accept_any_collections(tmp_path: Path) -> None:
    client = PrunedRename(SPLG_BARS + SPYM_BARS)

    backfill(
        client,
        tmp_path,
        "1Day",
        date(2024, 1, 1),
        date(2024, 12, 31),
        ["SPLG", "SPYM"],
        now=lambda: NOW_2027,
        chains=(["SPYM", "SPLG"],),
    )

    assert [c.symbols for c in client.calls] == [("SPLG", "SPYM"), ("SPLG",)]
    rows = pq.read_table(tmp_path / "bars_1d" / "year=2024" / "bucket=S.parquet")
    assert rows.column("symbol").to_pylist().count("SPLG") == 2


def test_three_name_chain_retries_only_the_member_whose_partner_shared_its_batch(
    tmp_path: Path,
) -> None:
    # A synthetic three-name chain in one bucket (real three-name chains such as
    # VII -> BNZI -> PARA span buckets and never share a request). QNEW is active and alone in
    # its batch; QOLD and QMID are inactive and share one. The batch answer carries QMID's own
    # bars and QOLD's relabelled QMID (the MMC/MRSH behaviour). QOLD answered nothing with a
    # partner present: retried. QNEW answered nothing without a partner: not retried.
    class ThreeNames(FakeBarClient):
        def iter_bars(self, symbols, timeframe, start, end):  # type: ignore[override]
            self.calls.append(BarsCall(tuple(symbols), timeframe, start, end))
            wanted = set(symbols)
            out = [b for b in self.bars if b.symbol in wanted and start <= b.t <= end]
            if "QOLD" in wanted and "QMID" in wanted:
                out = [b.model_copy(update={"symbol": "QMID"}) for b in out]
            return iter(out)

    client = ThreeNames([daily("QOLD", date(2024, 1, 5)), daily("QMID", date(2024, 6, 3))])

    report = backfill(
        client,
        tmp_path,
        "1Day",
        date(2024, 1, 1),
        date(2024, 12, 31),
        ["QOLD", "QMID", "QNEW"],
        now=lambda: NOW_2027,
        active={"QNEW"},
        chains=[{"QOLD", "QMID", "QNEW"}],
    )

    assert [c.symbols for c in client.calls] == [("QNEW",), ("QOLD", "QMID"), ("QOLD",)]
    rows = pq.read_table(tmp_path / "bars_1d" / "year=2024" / "bucket=Q.parquet").to_pylist()
    assert [(r["symbol"], r["session_date"]) for r in rows] == [
        ("QMID", date(2024, 1, 5)),
        ("QMID", date(2024, 6, 3)),
        ("QOLD", date(2024, 1, 5)),
    ]
    assert report.rows == 3


# --- boundaries: when the partner did not share the request, no solo request -----------------


def test_chain_member_whose_partner_is_in_the_other_active_group_is_not_retried(
    tmp_path: Path,
) -> None:
    # SPYM active, SPLG inactive: they never share a request, so SPLG's empty answer is final.
    client = FakeBarClient(SPYM_BARS)

    backfill(
        client,
        tmp_path,
        "1Day",
        date(2024, 1, 1),
        date(2024, 12, 31),
        ["SPLG", "SPYM"],
        now=lambda: NOW_2027,
        active={"SPYM"},
        chains=SPLG_CHAIN,
    )

    assert [c.symbols for c in client.calls] == [("SPYM",), ("SPLG",)]


def test_chain_member_separated_from_its_partner_by_the_batch_size_is_not_retried(
    tmp_path: Path,
) -> None:
    client = FakeBarClient(SPYM_BARS)

    backfill(
        client,
        tmp_path,
        "1Day",
        date(2024, 1, 1),
        date(2024, 12, 31),
        ["SPLG", "SPYM"],
        batch_size=1,
        now=lambda: NOW_2027,
        active=set(),
        chains=SPLG_CHAIN,
    )

    assert [c.symbols for c in client.calls] == [("SPLG",), ("SPYM",)]


def test_chain_member_whose_partner_is_not_in_the_run_is_not_retried(tmp_path: Path) -> None:
    # SIVB's window ends in 2023, so it is left out of 2024 and, answering nothing, asked
    # again alone in 2022 and 2023 (window rule). SPLG has no window and its partner SPYM is
    # not in the run: requested every year, never retried.
    client = FakeBarClient()
    windows = {"SIVB": (date(2018, 1, 22), date(2023, 4, 5))}

    backfill(
        client,
        tmp_path,
        "1Day",
        date(2022, 1, 1),
        date(2024, 12, 31),
        ["SIVB", "SPLG"],
        now=lambda: NOW_2027,
        windows=windows,
        active=set(),
        chains=SPLG_CHAIN,
    )

    by_year: dict[int, list[tuple[str, ...]]] = {}
    for call in client.calls:
        by_year.setdefault(call.start.astimezone(ET).year, []).append(call.symbols)
    assert by_year == {
        2022: [("SIVB", "SPLG"), ("SIVB",)],
        2023: [("SIVB", "SPLG"), ("SIVB",)],
        2024: [("SPLG",)],
    }


def test_chain_symbols_absent_from_the_run_are_ignored(tmp_path: Path) -> None:
    client = FakeBarClient()

    backfill(
        client,
        tmp_path,
        "1Day",
        date(2024, 1, 1),
        date(2024, 12, 31),
        ["SPYM"],
        now=lambda: NOW_2027,
        chains=[{"SPLG", "SPYM"}, {"VII", "BNZI", "PARA"}],
    )

    assert [c.symbols for c in client.calls] == [("SPYM",)]


def test_chain_member_that_answered_in_the_shared_batch_is_not_requested_again(
    tmp_path: Path,
) -> None:
    client = FakeBarClient(SPLG_BARS + SPYM_BARS)

    backfill(
        client,
        tmp_path,
        "1Day",
        date(2024, 1, 1),
        date(2024, 12, 31),
        ["SPLG", "SPYM"],
        now=lambda: NOW_2027,
        active=set(),
        chains=SPLG_CHAIN,
    )

    assert [c.symbols for c in client.calls] == [("SPLG", "SPYM")]


def test_two_chain_members_both_empty_in_a_shared_batch_are_each_retried(tmp_path: Path) -> None:
    # The rule is about who shared the request, not who answered: with neither name answering
    # (the security has no bars that year) both get a solo request. Pinned so a change here is
    # a deliberate one.
    client = FakeBarClient()

    backfill(
        client,
        tmp_path,
        "1Day",
        date(2024, 1, 1),
        date(2024, 12, 31),
        ["SPLG", "SPYM"],
        now=lambda: NOW_2027,
        active=set(),
        chains=SPLG_CHAIN,
    )

    assert [c.symbols for c in client.calls] == [("SPLG", "SPYM"), ("SPLG",), ("SPYM",)]


def test_windowed_symbol_is_still_retried_when_its_chain_partner_was_elsewhere(
    tmp_path: Path,
) -> None:
    # Rule (a) is untouched by ``chains``: MMC has a window over 2026, answers nothing in its
    # inactive batch while MRSH sits in the active one, and is asked again alone regardless.
    client = FakeBarClient(MRSH_BARS)

    backfill(
        client,
        tmp_path,
        "1Day",
        date(2026, 1, 1),
        date(2026, 12, 31),
        ["MMC", "MRSH"],
        now=lambda: NOW_2027,
        windows=WINDOWS,
        active={"MRSH"},
        chains=[{"MMC", "MRSH"}],
    )

    assert [c.symbols for c in client.calls] == [("MRSH",), ("MMC",), ("MMC",)]


def test_empty_chains_behave_like_none(tmp_path: Path) -> None:
    for chains in (None, set(), [], [set()]):
        client = FakeBarClient()

        backfill(
            client,
            tmp_path,
            "1Day",
            date(2024, 1, 1),
            date(2024, 12, 31),
            ["ZZZZ"],
            now=lambda: NOW_2027,
            force=True,
            chains=chains,
        )

        assert [c.symbols for c in client.calls] == [("ZZZZ",)], chains


def test_retry_keyword_is_gone(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        backfill(
            FakeBarClient(),
            tmp_path,
            "1Day",
            date(2024, 1, 1),
            date(2024, 12, 31),
            ["SPLG"],
            now=lambda: NOW_2027,
            retry={"SPLG"},  # type: ignore[call-arg]
        )


# --- invariants: chains shape requests, never what is kept -----------------------------------


def test_chains_write_whatever_the_solo_request_returns_under_its_own_label(
    tmp_path: Path,
) -> None:
    # The solo answer for SPLG carries one bar labelled SPLG and one the server filed under
    # SPYM. Both are written as returned; nothing is dropped or relabelled on the way in.
    class OddLabels(PrunedRename):
        def iter_bars(self, symbols, timeframe, start, end):  # type: ignore[override]
            if tuple(symbols) == ("SPLG",):
                self.calls.append(BarsCall(tuple(symbols), timeframe, start, end))
                return iter([daily("SPLG", date(2024, 1, 5)), daily("SPYM", date(2024, 1, 8))])
            return super().iter_bars(symbols, timeframe, start, end)

    client = OddLabels(SPLG_BARS)

    report = backfill(
        client,
        tmp_path,
        "1Day",
        date(2024, 1, 1),
        date(2024, 12, 31),
        ["SPLG", "SPYM"],
        now=lambda: NOW_2027,
        active=set(),
        chains=SPLG_CHAIN,
    )

    rows = pq.read_table(tmp_path / "bars_1d" / "year=2024" / "bucket=S.parquet").to_pylist()
    assert [(r["symbol"], r["session_date"]) for r in rows] == [
        ("SPLG", date(2024, 1, 5)),
        ("SPYM", date(2024, 1, 5)),
        ("SPYM", date(2024, 1, 8)),
        ("SPYM", date(2024, 2, 5)),
    ]
    assert report.rows == 4


def test_chains_none_leaves_the_second_pass_to_the_windows(tmp_path: Path) -> None:
    # Windows-only behaviour, spelled out: a windowed symbol that answers nothing is asked
    # again alone; a symbol without a window is not, even though SPLG and SPYM shared a batch.
    client = FakeBarClient()
    windows = {"SIVB": (date(2018, 1, 22), date(2023, 4, 5))}

    backfill(
        client,
        tmp_path,
        "1Day",
        date(2023, 1, 1),
        date(2023, 12, 31),
        ["SIVB", "SPLG", "SPYM"],
        now=lambda: NOW_2027,
        windows=windows,
        active=set(),
        chains=None,
    )

    assert [c.symbols for c in client.calls] == [("SIVB", "SPLG", "SPYM"), ("SIVB",)]
