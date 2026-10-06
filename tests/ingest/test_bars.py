"""Guards against bars that claim to be known before the market produced them, half-written
partition files, and a backfill that re-downloads what it already has or skips what changed."""

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


def test_write_partition_overwrites_an_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "bucket=A.parquet"
    write_partition(path, [make_bar("AAPL", T_DAY), make_bar("AMZN", T_DAY)], "1Day", FETCHED_AT)

    count = write_partition(path, [make_bar("AAPL", T_DAY)], "1Day", FETCHED_AT)

    assert count == 1
    assert pq.read_table(path).column("symbol").to_pylist() == ["AAPL"]
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
NOW_2027 = datetime(2027, 1, 15, tzinfo=UTC)


def test_bars_outside_a_symbols_sec_window_are_dropped_at_write_time(tmp_path: Path) -> None:
    path = tmp_path / "bucket=M.parquet"
    relabelled = [b.model_copy(update={"symbol": "MRSH"}) for b in MMC_BARS] + MRSH_BARS

    count = write_partition(path, relabelled, "1Day", NOW_2027, windows=WINDOWS)

    assert count == 1
    rows = pq.read_table(path).to_pylist()
    assert [(r["symbol"], r["session_date"]) for r in rows] == [("MRSH", date(2026, 4, 1))]


def test_symbols_without_a_window_are_kept_whole(tmp_path: Path) -> None:
    path = tmp_path / "bucket=S.parquet"
    count = write_partition(
        path, [daily("SPY", date(2026, 1, 5))], "1Day", NOW_2027, windows=WINDOWS
    )
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
    rows = pq.read_table(tmp_path / "bars_1d" / "year=2026" / "bucket=M.parquet").to_pylist()
    assert [(r["symbol"], r["session_date"]) for r in rows] == [
        ("MMC", date(2026, 1, 5)),
        ("MMC", date(2026, 2, 3)),
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
    rows = pq.read_table(tmp_path / "bars_1d" / "year=2023" / "bucket=S.parquet").to_pylist()
    assert [(r["symbol"], r["session_date"]) for r in rows] == [
        ("SIVB", date(2023, 1, 5)),
        ("SIVB", date(2023, 2, 3)),
    ]
    assert report.rows == 2


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
