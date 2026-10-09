"""Guards against look-ahead leakage in market data, on two layers.

Store layer: ``visible_bars`` must never return a bar whose ``available_at`` is after ``as_of``,
even when the file on disk already holds it. Identity layer: a bar is handed out under a
``security_id`` only through a security-master segment knowable at ``as_of``, and a segment's
END is a fact of its own (``end_available_at``): a reader in 2022 must see BK as an open segment
although the file on disk says it ends 2026-05-21, and must keep owning the old symbol's bars
until the end is knowable (BBUC's cut is on disk as 2026-03-31 but was filed 04-21; 14 traded
BBUC bars sit in between).

Target API (asof.store.market; implementation pending, design approved 2026-10-08):

    UNRESOLVED_BAR_SCHEMA = BAR_SCHEMA + traded (bool)
    RESOLVED_BAR_SCHEMA = BAR_SCHEMA + traded (bool) + security_id (string) + split_day (bool)
    visible_master(con, root, as_of) -> pa.Table                      schema MASTER_SCHEMA
    visible_bars(con, root, timeframe, as_of, *, symbols=None, start=None, end=None,
                 resolve=True, include_untraded=False) -> pa.Table

Fake data provenance ("real" = read off the store / the security master on 2026-10-08):
- Daily bar availability is 20:00 New York of its session (documented in ingest.bars; real: the
  2020-01-02 bar has available_at 2020-01-03 01:00 UTC). Minute bars: t + 1 minute.
- BK -> BNY: master row BK [2020-01-02, 2026-05-21), security S000730, end knowable 2026-05-21
  04:00 UTC = BNY's available_at (real). The raw store holds a copy of BK's history under BNY
  and the master leaves that copy without a segment (real).
- VTIQ -> NKLA: security S006538, NKLA segment valid_from 2020-06-04, available_at 2020-06-08
  04:00 UTC (real). After the rename Alpaca keeps writing VTIQ bars with volume 0 and OHLC
  frozen at 33.97 (real: 2020-06-04 .. 2022-09-21, 214 bars).
- Volume-0 bars: 1,176,813 in the store, all flat placeholders (real; user decision 2026-10-08:
  ``traded`` = volume > 0 and untraded rows are dropped by default).
- "LAC -> LAAC" fixture: a label whose on-disk cut is filed weeks later while the successor
  carries a new symbol. Real instances: BBUC cut 2026-03-31, end knowable 2026-04-21 04:00 UTC,
  14 traded bars in between; GGRW cut 2022-04-05 known 05-03 (54 such bars / 12 symbols
  store-wide on 2026-10-08). The real LAC is NOT this shape: Alpaca filed LAC.WI -> LAC on
  2023-10-04 itself, so the old LAC's end is knowable that day and the 10-04 .. 11-01 LAC bars
  are the new company's (see test_security_master H2). The dates below are the fixture's, kept
  for readability.
- Splits: ``available_at`` is 00:00 New York of the ex_date (documented in
  ingest.corporate_actions). AAPL 4:1 on 2020-08-31 is a real split; its CUSIPs are not asserted.
- Segment valid_from of VTIQ / LAC / BK in these fixtures: a convenient early date, not the real
  one; only the cut dates and the knowability instants are asserted.

Real-data checks for the ASSUMPTIONS (audit / standard cases):
- `SELECT * FROM security_master WHERE symbol = 'BBUC'`: cut 2026-03-31, end_available_at
  2026-04-21 04:00 UTC. visible_bars(as_of=2026-04-10, symbols=['BBUC']) must return BBUC's
  04-01 .. 04-09 bars under BBUC's security_id; at as_of 2026-04-22 they are gone.
- Store-wide: every back-to-back pair of segments of one symbol has old.end_available_at ==
  new.available_at (723 pairs, 0 mismatches on 2026-10-08).
- BK at as_of 2022-03-01: visible_master shows BK with valid_to NULL; visible_bars(symbols=['BK',
  'BNY']) returns only BK rows, security_id S000730.
- NKLA at as_of 2020-06-06 00:00 UTC: no NKLA bars; at 2020-06-08 04:00 UTC VTIQ's and NKLA's
  bars come back under S006538 as one timeline.
- Store-wide: bars with volume = 0 and open != close must count 0 (the "all placeholders"
  premise behind traded = volume > 0).
"""

from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from asof.ingest.alpaca import Bar
from asof.ingest.bars import BAR_SCHEMA, Partition, available_at, bucket_of, write_partition
from asof.ingest.corporate_actions import SPLIT_SCHEMA
from asof.ingest.security_master import MASTER_FILE, MASTER_SCHEMA
from asof.store.market import (
    RESOLVED_BAR_SCHEMA,
    UNRESOLVED_BAR_SCHEMA,
    bars_status,
    market_root,
    visible_bars,
    visible_master,
    visible_sessions,
)
from tests.ingest.fakes import AVAILABLE_DAY, FETCHED_AT, T_DAY, T_MIN, daily, make_bar

ET = ZoneInfo("America/New_York")
MINUTE = timedelta(minutes=1)
DAY = timedelta(days=1)
TS_UTC = pa.timestamp("us", tz="UTC")
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
    "traded",
}
# 2016-01-04 00:00 New York: the start of the bar store, when every opening segment is known.
KNOWN_EARLY = datetime(2016, 1, 4, tzinfo=ET).astimezone(UTC)
BOTH_MODES = pytest.mark.parametrize("resolve", [False, True], ids=["raw", "resolved"])


# =============================================================================================
# builders
# =============================================================================================


def known(day: date) -> datetime:
    """00:00 New York of ``day`` as UTC: when a rename-created segment, and the end of the one
    it replaces, become knowable (real: BNY 2026-05-21 04:00 UTC)."""
    return datetime.combine(day, datetime.min.time(), tzinfo=ET).astimezone(UTC)


def avail(day: date) -> datetime:
    """When the daily bar of ``day`` becomes visible (20:00 New York), per the ingest rule."""
    return available_at(daily("X", day).t, "1Day")


def t_of(day: date) -> datetime:
    return daily("X", day).t


def filler(symbol: str, day: date, close: float = 33.97) -> Bar:
    """What Alpaca writes under an old label after a rename (real: VTIQ at 33.97, volume 0)."""
    return daily(
        symbol, day, open=close, high=close, low=close, close=close, volume=0, trade_count=0
    )


@dataclass(frozen=True)
class Seg:
    security_id: str
    symbol: str
    valid_from: date = date(2016, 1, 4)
    valid_to: date | None = None
    available_at: datetime = KNOWN_EARLY
    end_available_at: datetime | None = None  # must be set iff valid_to is set


def write_master(root: Path, segs: Sequence[Seg]) -> Path:
    table = pa.table(
        {
            "security_id": pa.array([s.security_id for s in segs], pa.string()),
            "symbol": pa.array([s.symbol for s in segs], pa.string()),
            "valid_from": pa.array([s.valid_from for s in segs], pa.date32()),
            "valid_to": pa.array([s.valid_to for s in segs], pa.date32()),
            "cik": pa.array([None for _ in segs], pa.int64()),
            "cusip": pa.array([None for _ in segs], pa.string()),
            "evidence": pa.array(["bars" for _ in segs], pa.string()),
            "available_at": pa.array([s.available_at for s in segs], TS_UTC),
            "end_available_at": pa.array([s.end_available_at for s in segs], TS_UTC),
        },
        schema=MASTER_SCHEMA,
    )
    path = root / MASTER_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


def open_master(root: Path, *symbols: str) -> Path:
    """One open segment per symbol, known since the store began: the master a test about the
    raw store's guarantees needs so that it can run under the default ``resolve=True``."""
    return write_master(root, [Seg(f"S_{s}", s) for s in symbols])


def write_splits(
    root: Path,
    rows: Sequence[tuple[str, date, float, float]],
    available_at: datetime | None = None,
) -> Path:
    """``(symbol, ex_date, old_rate, new_rate)`` rows; available_at per the ingest rule (00:00
    New York of the ex_date) unless given, which leak tests use to push a record into the future."""
    table = pa.table(
        {
            "symbol": pa.array([r[0] for r in rows], pa.string()),
            "ex_date": pa.array([r[1] for r in rows], pa.date32()),
            "old_rate": pa.array([r[2] for r in rows], pa.float64()),
            "new_rate": pa.array([r[3] for r in rows], pa.float64()),
            "factor": pa.array([r[3] / r[2] for r in rows], pa.float64()),
            "kind": pa.array(["forward" for _ in rows], pa.string()),
            "old_cusip": pa.array([None for _ in rows], pa.string()),
            "new_cusip": pa.array([None for _ in rows], pa.string()),
            "available_at": pa.array([available_at or known(r[1]) for r in rows], TS_UTC),
        },
        schema=SPLIT_SCHEMA,
    )
    path = root / "corporate_actions" / "splits.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


def store(
    root: Path, partition: Partition, bars: list[Bar], fetched_at: datetime = FETCHED_AT
) -> None:
    write_partition(partition.path(root), bars, partition.timeframe, fetched_at)


def store_daily(root: Path, bars: Sequence[Bar], fetched_at: datetime = FETCHED_AT) -> None:
    """Write daily bars into the partition files the backfill would use (one per New York year
    and first-letter bucket). Writing a partition replaces it, so pass every bar at once."""
    grouped: dict[tuple[int, str], list[Bar]] = defaultdict(list)
    for bar in bars:
        grouped[(bar.t.astimezone(ET).year, bucket_of(bar.symbol))].append(bar)
    for (year, bucket), group in grouped.items():
        write_partition(Partition("1Day", year, None, bucket).path(root), group, "1Day", fetched_at)


def keys(table: pa.Table) -> list[tuple[str, datetime]]:
    return list(zip(table.column("symbol").to_pylist(), table.column("t").to_pylist(), strict=True))


def owned(table: pa.Table) -> list[tuple[str, str, date]]:
    """``(security_id, symbol, session_date)`` per resolved row, in the table's order."""
    return list(
        zip(
            table.column("security_id").to_pylist(),
            table.column("symbol").to_pylist(),
            table.column("session_date").to_pylist(),
            strict=True,
        )
    )


def seg_rows(table: pa.Table) -> list[tuple[str, str, date, date | None, datetime | None]]:
    return [
        (r["security_id"], r["symbol"], r["valid_from"], r["valid_to"], r["end_available_at"])
        for r in table.to_pylist()
    ]


@pytest.fixture
def con() -> Iterator[duckdb.DuckDBPyConnection]:
    connection = duckdb.connect()
    yield connection
    connection.close()


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return market_root(tmp_path)


# =============================================================================================
# schemas
# =============================================================================================


def test_schemas_extend_the_bar_schema() -> None:
    assert UNRESOLVED_BAR_SCHEMA.names == [*BAR_SCHEMA.names, "traded"]
    assert RESOLVED_BAR_SCHEMA.names == [*BAR_SCHEMA.names, "traded", "security_id", "split_day"]
    for schema in (UNRESOLVED_BAR_SCHEMA, RESOLVED_BAR_SCHEMA):
        for field in BAR_SCHEMA:
            assert schema.field(field.name).type == field.type
        assert schema.field("traded").type == pa.bool_()
    assert RESOLVED_BAR_SCHEMA.field("security_id").type == pa.string()
    assert RESOLVED_BAR_SCHEMA.field("split_day").type == pa.bool_()
    assert MASTER_SCHEMA.names[-1] == "end_available_at"


# =============================================================================================
# normal path
# =============================================================================================


def test_market_root_lives_under_the_data_dir(tmp_path: Path) -> None:
    assert market_root(tmp_path) == tmp_path / "market"


def test_raw_bars_are_the_stored_rows_ordered_by_symbol_then_time(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # resolve=False is the raw-store view: no master is needed and nothing is attributed
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

    table = visible_bars(con, root, "1Day", as_of=datetime(2026, 10, 3, tzinfo=UTC), resolve=False)

    assert keys(table) == [
        ("AAPL", datetime(2026, 10, 1, 4, tzinfo=UTC)),
        ("AAPL", datetime(2026, 10, 2, 4, tzinfo=UTC)),
        ("AMZN", datetime(2026, 10, 1, 4, tzinfo=UTC)),
        ("MSFT", datetime(2026, 10, 1, 4, tzinfo=UTC)),
    ]
    assert table.schema.names == UNRESOLVED_BAR_SCHEMA.names
    assert EXPECTED_COLUMNS <= set(table.schema.names)
    assert table.column("traded").to_pylist() == [True] * 4


def test_resolved_bars_carry_the_security_id_and_are_ordered_by_security_then_time(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # ids chosen so that security order differs from symbol order: AMZN first, then AAPL
    write_master(root, [Seg("S2", "AAPL"), Seg("S1", "AMZN"), Seg("S3", "MSFT")])
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

    assert table.schema.names == RESOLVED_BAR_SCHEMA.names
    assert owned(table) == [
        ("S1", "AMZN", date(2026, 10, 1)),
        ("S2", "AAPL", date(2026, 10, 1)),
        ("S2", "AAPL", date(2026, 10, 2)),
        ("S3", "MSFT", date(2026, 10, 1)),
    ]
    assert table.column("traded").to_pylist() == [True] * 4
    assert table.column("split_day").to_pylist() == [False] * 4  # no splits file at all


def test_visible_bars_reads_across_year_partitions(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    open_master(root, "AAPL")
    store(root, Partition("1Day", 2025, None, "A"), [daily("AAPL", date(2025, 12, 31))])
    store(root, Partition("1Day", 2026, None, "A"), [daily("AAPL", date(2026, 1, 2))])

    table = visible_bars(con, root, "1Day", as_of=datetime(2026, 2, 1, tzinfo=UTC))

    assert [k[1].date() for k in keys(table)] == [date(2025, 12, 31), date(2026, 1, 2)]


@BOTH_MODES
def test_symbols_filter_restricts_the_result(
    con: duckdb.DuckDBPyConnection, root: Path, resolve: bool
) -> None:
    open_master(root, "AAPL", "AMZN", "MSFT")
    store(
        root,
        Partition("1Day", 2026, None, "A"),
        [daily("AAPL", date(2026, 10, 1)), daily("AMZN", date(2026, 10, 1))],
    )
    store(root, Partition("1Day", 2026, None, "M"), [daily("MSFT", date(2026, 10, 1))])
    as_of = datetime(2026, 10, 3, tzinfo=UTC)

    table = visible_bars(con, root, "1Day", as_of, symbols=["MSFT", "AAPL"], resolve=resolve)
    assert [k[0] for k in keys(table)] == ["AAPL", "MSFT"]
    assert keys(visible_bars(con, root, "1Day", as_of, symbols=["ZZZZ"], resolve=resolve)) == []


def test_time_window_is_start_inclusive_end_exclusive(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    open_master(root, "AAPL")
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


# --- traded: volume-0 placeholder bars ----------------------------------------------------------


@BOTH_MODES
def test_untraded_bars_are_dropped_by_default_and_flagged_on_request(
    con: duckdb.DuckDBPyConnection, root: Path, resolve: bool
) -> None:
    # real shape: VTIQ's 2020-06-04 .. 2020-06-08 bars are flat 33.97 x4 with volume 0
    open_master(root, "VTIQ")
    store_daily(
        root,
        [
            daily("VTIQ", date(2020, 6, 2), close=33.5),
            daily("VTIQ", date(2020, 6, 3), close=33.97),
            filler("VTIQ", date(2020, 6, 4)),
            filler("VTIQ", date(2020, 6, 5)),
        ],
    )
    as_of = avail(date(2020, 6, 5))

    default = visible_bars(con, root, "1Day", as_of, resolve=resolve)
    assert [k[1].date() for k in keys(default)] == [date(2020, 6, 2), date(2020, 6, 3)]
    assert default.column("traded").to_pylist() == [True, True]

    everything = visible_bars(con, root, "1Day", as_of, resolve=resolve, include_untraded=True)
    assert [k[1].date() for k in keys(everything)] == [
        date(2020, 6, 2),
        date(2020, 6, 3),
        date(2020, 6, 4),
        date(2020, 6, 5),
    ]
    assert everything.column("traded").to_pylist() == [True, True, False, False]
    assert everything.column("volume").to_pylist() == [100, 100, 0, 0]


@BOTH_MODES
def test_traded_is_volume_above_zero_not_trade_count(
    con: duckdb.DuckDBPyConnection, root: Path, resolve: bool
) -> None:
    # the flag is defined on volume alone (user decision 2026-10-08); a bar with one share is traded
    open_master(root, "THIN")
    store_daily(
        root,
        [
            daily("THIN", date(2026, 10, 1), volume=1, trade_count=1),
            daily("THIN", date(2026, 10, 2), volume=0, trade_count=5),
        ],
    )

    table = visible_bars(con, root, "1Day", AVAILABLE_DAY, resolve=resolve, include_untraded=True)

    assert table.column("traded").to_pylist() == [True, False]
    assert keys(visible_bars(con, root, "1Day", AVAILABLE_DAY, resolve=resolve)) == [
        ("THIN", t_of(date(2026, 10, 1)))
    ]


# --- visible_master -----------------------------------------------------------------------------

# real master rows: BK S000730 [2020-01-02, 2026-05-21) known 2020-01-03 01:00 UTC, end known
# 2026-05-21 04:00 UTC; BNY S000730 [2026-05-21, open) known 2026-05-21 04:00 UTC.
BK_FIRST = date(2020, 1, 2)
BK_CUT = date(2026, 5, 21)
BK_END_KNOWN = known(BK_CUT)
BK_SEGS = [
    Seg("S000730", "BK", BK_FIRST, BK_CUT, avail(BK_FIRST), BK_END_KNOWN),
    Seg("S000730", "BNY", BK_CUT, None, BK_END_KNOWN, None),
    Seg("S_SPY", "SPY"),
]


def test_fixture_instants_match_the_real_rows() -> None:
    assert avail(BK_FIRST) == datetime(2020, 1, 3, 1, tzinfo=UTC)  # real
    assert BK_END_KNOWN == datetime(2026, 5, 21, 4, tzinfo=UTC)  # real: BNY's available_at
    assert known(date(2020, 6, 8)) == datetime(2020, 6, 8, 4, tzinfo=UTC)  # real: NKLA's
    assert known(date(2023, 11, 2)) == datetime(2023, 11, 2, 4, tzinfo=UTC)


def test_visible_master_returns_known_segments_with_the_master_schema(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    write_master(root, BK_SEGS)

    table = visible_master(con, root, datetime(2026, 10, 1, tzinfo=UTC))

    assert table.schema.equals(MASTER_SCHEMA)
    # ordered by security_id, valid_from, symbol; the ended BK row shows its end once knowable
    assert seg_rows(table) == [
        ("S000730", "BK", BK_FIRST, BK_CUT, BK_END_KNOWN),
        ("S000730", "BNY", BK_CUT, None, None),
        ("S_SPY", "SPY", date(2016, 1, 4), None, None),
    ]


def test_visible_master_masks_an_end_that_is_not_yet_knowable(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # a reader in 2022 sees BK as an open segment: both valid_to and end_available_at are None,
    # and nothing of BNY
    write_master(root, BK_SEGS)

    table = visible_master(con, root, datetime(2022, 3, 1, tzinfo=UTC))

    assert seg_rows(table) == [
        ("S000730", "BK", BK_FIRST, None, None),
        ("S_SPY", "SPY", date(2016, 1, 4), None, None),
    ]


def test_visible_master_orders_by_security_then_valid_from_then_symbol(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    d = date(2020, 6, 4)
    write_master(
        root,
        [
            Seg("S2", "ZZZ"),
            Seg("S1", "B", d, None, known(d)),
            Seg("S1", "A", d, None, known(d)),
            Seg("S1", "C", date(2016, 1, 4), d, KNOWN_EARLY, known(d)),
        ],
    )

    table = visible_master(con, root, datetime(2021, 1, 1, tzinfo=UTC))

    assert [(r[0], r[1], r[2]) for r in seg_rows(table)] == [
        ("S1", "C", date(2016, 1, 4)),
        ("S1", "A", d),
        ("S1", "B", d),
        ("S2", "ZZZ", date(2016, 1, 4)),
    ]


# =============================================================================================
# boundaries
# =============================================================================================


def test_no_files_at_all_gives_an_empty_table(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    as_of = datetime(2026, 10, 3, tzinfo=UTC)
    raw = visible_bars(con, root, "1Day", as_of, resolve=False)
    assert raw.num_rows == 0 and raw.schema.names == UNRESOLVED_BAR_SCHEMA.names
    assert bars_status(con, root, "1Day") == []

    open_master(root, "AAPL")  # a master but no bars
    resolved = visible_bars(con, root, "1Day", as_of)
    assert resolved.num_rows == 0 and resolved.schema.names == RESOLVED_BAR_SCHEMA.names


def test_resolving_without_a_master_is_an_error_not_a_fallback(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    store(root, Partition("1Day", 2026, None, "A"), [daily("AAPL", date(2026, 10, 1))])
    as_of = datetime(2026, 10, 3, tzinfo=UTC)

    with pytest.raises(FileNotFoundError):
        visible_bars(con, root, "1Day", as_of)
    with pytest.raises(FileNotFoundError):
        visible_bars(con, root, "1Day", as_of, resolve=True, include_untraded=True)
    with pytest.raises(FileNotFoundError):
        visible_master(con, root, as_of)
    assert keys(visible_bars(con, root, "1Day", as_of, resolve=False)) == [
        ("AAPL", datetime(2026, 10, 1, 4, tzinfo=UTC))
    ]


def test_empty_master_file_resolves_nothing(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    write_master(root, [])
    store(root, Partition("1Day", 2026, None, "A"), [daily("AAPL", date(2026, 10, 1))])
    as_of = datetime(2026, 10, 3, tzinfo=UTC)

    master = visible_master(con, root, as_of)
    assert master.num_rows == 0 and master.schema.equals(MASTER_SCHEMA)
    table = visible_bars(con, root, "1Day", as_of)
    assert table.num_rows == 0 and table.schema.names == RESOLVED_BAR_SCHEMA.names


def test_empty_partition_file_is_tolerated(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    open_master(root, "AAPL")
    store(root, Partition("1Day", 2019, None, "Q"), [])
    store(root, Partition("1Day", 2026, None, "A"), [daily("AAPL", date(2026, 1, 2))])

    table = visible_bars(con, root, "1Day", as_of=datetime(2026, 10, 3, tzinfo=UTC))

    assert keys(table) == [("AAPL", datetime(2026, 1, 2, 5, tzinfo=UTC))]
    status = sorted(bars_status(con, root, "1Day"), key=lambda s: s["year"])
    assert status[0] == {"year": 2019, "rows": 0, "symbols": 0, "first": None, "last": None}
    assert status[1]["rows"] == 1


@BOTH_MODES
def test_bar_available_exactly_at_as_of_is_visible(
    con: duckdb.DuckDBPyConnection, root: Path, resolve: bool
) -> None:
    open_master(root, "AAPL")
    store(root, Partition("1Day", 2026, None, "A"), [daily("AAPL", date(2026, 10, 2))])

    assert keys(visible_bars(con, root, "1Day", as_of=AVAILABLE_DAY, resolve=resolve)) == [
        ("AAPL", T_DAY)
    ]
    just_before = AVAILABLE_DAY - timedelta(microseconds=1)
    assert keys(visible_bars(con, root, "1Day", as_of=just_before, resolve=resolve)) == []


def test_segment_known_exactly_at_as_of_is_visible_and_its_end_likewise(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    write_master(root, BK_SEGS)

    at = visible_master(con, root, BK_END_KNOWN)
    assert [(r[1], r[3]) for r in seg_rows(at)] == [("BK", BK_CUT), ("BNY", None), ("SPY", None)]
    before = visible_master(con, root, BK_END_KNOWN - timedelta(microseconds=1))
    assert [(r[1], r[3]) for r in seg_rows(before)] == [("BK", None), ("SPY", None)]
    first = visible_master(con, root, avail(BK_FIRST))
    assert [r[1] for r in seg_rows(first)] == ["BK", "SPY"]
    assert [r[1] for r in seg_rows(visible_master(con, root, avail(BK_FIRST) - MINUTE))] == ["SPY"]


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
    open_master(root, "AAPL")
    store(root, Partition("1Day", 2026, None, "A"), [daily("AAPL", date(2026, 10, 1))])

    with pytest.raises(ValueError):
        visible_bars(con, root, "1Day", **kwargs)
    with pytest.raises(ValueError):
        visible_bars(con, root, "1Day", resolve=False, **kwargs)
    with pytest.raises(ValueError):
        visible_master(con, root, datetime(2026, 10, 3))


def test_as_of_in_another_zone_is_compared_as_an_instant(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    open_master(root, "AAPL")
    store(root, Partition("1Day", 2026, None, "A"), [daily("AAPL", date(2026, 10, 2))])

    assert (
        keys(visible_bars(con, root, "1Day", as_of=datetime(2026, 10, 2, 19, 59, tzinfo=ET))) == []
    )
    assert (
        len(keys(visible_bars(con, root, "1Day", as_of=datetime(2026, 10, 2, 20, 0, tzinfo=ET))))
        == 1
    )


def test_bars_on_the_segment_edges_follow_valid_from_inclusive_valid_to_exclusive(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # X owned by S_A up to and excluding 2026-10-02, by S_B from that day on; both known
    cut = date(2026, 10, 2)
    write_master(
        root,
        [
            Seg("S_A", "X", date(2026, 9, 1), cut, KNOWN_EARLY, known(cut)),
            Seg("S_B", "X", cut, None, known(cut)),
        ],
    )
    store_daily(
        root, [daily("X", date(2026, 9, 30)), daily("X", date(2026, 10, 1)), daily("X", cut)]
    )

    table = visible_bars(con, root, "1Day", datetime(2026, 10, 4, tzinfo=UTC))

    assert owned(table) == [
        ("S_A", "X", date(2026, 9, 30)),
        ("S_A", "X", date(2026, 10, 1)),
        ("S_B", "X", cut),
    ]


def test_bars_before_a_symbols_first_segment_are_not_returned(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # a bar dated before valid_from has no owner (an opening segment starts at the first bar on
    # the real master; here the fixture deliberately leaves one day uncovered)
    write_master(root, [Seg("S_X", "X", date(2026, 10, 1))])
    store_daily(root, [daily("X", date(2026, 9, 30)), daily("X", date(2026, 10, 1))])

    table = visible_bars(con, root, "1Day", datetime(2026, 10, 4, tzinfo=UTC))

    assert owned(table) == [("S_X", "X", date(2026, 10, 1))]


# --- visible_master: a broken file is refused ---------------------------------------------------


def test_master_violating_the_end_invariant_is_rejected(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    as_of = datetime(2026, 10, 1, tzinfo=UTC)
    store(root, Partition("1Day", 2026, None, "A"), [daily("A", date(2026, 9, 1))])

    # valid_to set, end unknown: an older build without the column's semantics
    write_master(root, [Seg("S_A", "A", date(2016, 1, 4), date(2026, 9, 1), KNOWN_EARLY, None)])
    with pytest.raises(ValueError, match="rebuild"):
        visible_master(con, root, as_of)
    with pytest.raises(ValueError, match="rebuild"):
        visible_bars(con, root, "1Day", as_of)

    # end set on an open segment
    write_master(
        root, [Seg("S_A", "A", date(2016, 1, 4), None, KNOWN_EARLY, known(date(2026, 9, 1)))]
    )
    with pytest.raises(ValueError, match="rebuild"):
        visible_master(con, root, as_of)


def test_master_with_duplicate_visible_keys_is_rejected(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    write_master(root, [Seg("S_A", "A"), Seg("S_A", "A")])
    with pytest.raises(ValueError, match="duplicate"):
        visible_master(con, root, datetime(2026, 10, 1, tzinfo=UTC))


def test_master_with_overlapping_visible_segments_of_one_symbol_is_rejected(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    write_master(
        root,
        [
            Seg(
                "S_A", "X", date(2020, 1, 2), date(2020, 6, 4), KNOWN_EARLY, known(date(2020, 6, 4))
            ),
            Seg("S_B", "X", date(2020, 3, 2), None, known(date(2020, 3, 2))),
        ],
    )
    with pytest.raises(ValueError, match="overlap"):
        visible_master(con, root, datetime(2026, 10, 1, tzinfo=UTC))


def test_overlap_created_by_masking_an_unknowable_end_is_rejected(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # On disk the two X segments are back to back, but the taker's row is knowable (06-04) before
    # the previous holder's end is (06-08): at an as_of in between the first row is masked open
    # and both own X. A consistent master ties the two instants together; this file is broken.
    write_master(
        root,
        [
            Seg(
                "S_A", "X", date(2020, 1, 2), date(2020, 6, 4), KNOWN_EARLY, known(date(2020, 6, 8))
            ),
            Seg("S_B", "X", date(2020, 6, 4), None, known(date(2020, 6, 4))),
        ],
    )
    with pytest.raises(ValueError, match="overlap"):
        visible_master(con, root, datetime(2020, 6, 6, tzinfo=UTC))
    # once both facts are knowable the file reads fine
    assert len(seg_rows(visible_master(con, root, datetime(2020, 6, 9, tzinfo=UTC)))) == 2


def test_checks_run_on_visible_segments_only(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    # a duplicate / overlap that is not yet knowable is not this reader's problem
    future = datetime(2030, 1, 1, tzinfo=UTC)
    write_master(
        root,
        [
            Seg("S_A", "A"),
            Seg("S_A", "A", available_at=future),
            Seg("S_B", "A", date(2020, 1, 2), None, future),
        ],
    )
    assert [
        r[0] for r in seg_rows(visible_master(con, root, datetime(2026, 1, 1, tzinfo=UTC)))
    ] == ["S_A"]
    with pytest.raises(ValueError):
        visible_master(con, root, future)


# =============================================================================================
# leakage
# =============================================================================================


@BOTH_MODES
def test_todays_daily_bar_is_invisible_during_the_session_and_visible_after_post_market(
    con: duckdb.DuckDBPyConnection, root: Path, resolve: bool
) -> None:
    # The bar for the 2026-10-02 session is on disk (t = 04:00Z) but the market has not closed
    # yet at 10:00 New York (14:00Z); it must only appear once post-market ends at 20:00 New
    # York (00:00Z next day), since the daily volume includes extended-hours trades.
    open_master(root, "AAPL")
    store(
        root,
        Partition("1Day", 2026, None, "A"),
        [daily("AAPL", date(2026, 10, 1)), daily("AAPL", date(2026, 10, 2))],
    )

    during = visible_bars(
        con, root, "1Day", as_of=datetime(2026, 10, 2, 14, 0, tzinfo=UTC), resolve=resolve
    )
    after = visible_bars(
        con, root, "1Day", as_of=datetime(2026, 10, 3, 0, 0, tzinfo=UTC), resolve=resolve
    )

    assert keys(during) == [("AAPL", datetime(2026, 10, 1, 4, tzinfo=UTC))]
    assert keys(after) == [("AAPL", datetime(2026, 10, 1, 4, tzinfo=UTC)), ("AAPL", T_DAY)]


@BOTH_MODES
def test_minute_bar_is_invisible_at_its_own_timestamp(
    con: duckdb.DuckDBPyConnection, root: Path, resolve: bool
) -> None:
    # A 09:30 bar covers 09:30:00-09:30:59; at 09:30:00 nothing about it is known.
    open_master(root, "AAPL")
    store(
        root,
        Partition("1Min", 2026, 10, "A"),
        [make_bar("AAPL", T_MIN), make_bar("AAPL", T_MIN + MINUTE)],
    )

    assert keys(visible_bars(con, root, "1Min", as_of=T_MIN, resolve=resolve)) == []
    assert keys(visible_bars(con, root, "1Min", as_of=T_MIN + MINUTE, resolve=resolve)) == [
        ("AAPL", T_MIN)
    ]
    assert keys(visible_bars(con, root, "1Min", as_of=T_MIN + 2 * MINUTE, resolve=resolve)) == [
        ("AAPL", T_MIN),
        ("AAPL", T_MIN + MINUTE),
    ]


@BOTH_MODES
def test_fetched_at_does_not_decide_visibility(
    con: duckdb.DuckDBPyConnection, root: Path, resolve: bool
) -> None:
    # Re-downloading history stamps a new fetched_at; the bar was still knowable at its close.
    open_master(root, "AAPL")
    far_future = datetime(2030, 1, 1, tzinfo=UTC)
    store(
        root,
        Partition("1Day", 2026, None, "A"),
        [daily("AAPL", date(2026, 10, 1))],
        fetched_at=far_future,
    )

    table = visible_bars(
        con, root, "1Day", as_of=datetime(2026, 10, 2, tzinfo=UTC), resolve=resolve
    )

    assert keys(table) == [("AAPL", datetime(2026, 10, 1, 4, tzinfo=UTC))]
    assert table.column("fetched_at").to_pylist() == [far_future]


@BOTH_MODES
def test_time_window_does_not_override_as_of(
    con: duckdb.DuckDBPyConnection, root: Path, resolve: bool
) -> None:
    # Asking for a window that includes the future bar must not make it visible.
    open_master(root, "AAPL")
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
        resolve=resolve,
    )

    assert keys(table) == [("AAPL", datetime(2026, 10, 1, 4, tzinfo=UTC))]


@BOTH_MODES
def test_timeframes_are_not_mixed(
    con: duckdb.DuckDBPyConnection, root: Path, resolve: bool
) -> None:
    open_master(root, "AAPL")
    store(root, Partition("1Min", 2026, 10, "A"), [make_bar("AAPL", T_MIN)])
    store(root, Partition("1Day", 2026, None, "A"), [daily("AAPL", date(2026, 10, 1))])
    as_of = datetime(2026, 10, 3, tzinfo=UTC)

    assert keys(visible_bars(con, root, "1Day", as_of, resolve=resolve)) == [
        ("AAPL", datetime(2026, 10, 1, 4, tzinfo=UTC))
    ]
    assert keys(visible_bars(con, root, "1Min", as_of, resolve=resolve)) == [("AAPL", T_MIN)]
    assert [s["rows"] for s in bars_status(con, root, "1Min")] == [1]


@BOTH_MODES
def test_every_returned_row_satisfies_the_invariant(
    con: duckdb.DuckDBPyConnection, root: Path, resolve: bool
) -> None:
    open_master(root, "AAPL")
    bars = [daily("AAPL", date(2026, 9, 28) + i * DAY) for i in range(10)]
    store(root, Partition("1Day", 2026, None, "A"), bars)

    for hour in (0, 14, 20, 23):
        as_of = datetime(2026, 10, 2, hour, tzinfo=UTC)
        table = visible_bars(con, root, "1Day", as_of, resolve=resolve)
        assert all(a <= as_of for a in table.column("available_at").to_pylist()), as_of
        assert table.num_rows == sum(1 for b in bars if b.t + timedelta(hours=20) <= as_of)


# --- leakage through identity: the master's own availability ------------------------------------


def test_relabelled_copy_under_the_new_name_is_never_returned_and_bk_stays_bk(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # The raw store holds BK's 2022 bar twice: under BK and under BNY (Alpaca files a security's
    # history under its newest name). resolve=True returns the BK row with S000730 and drops the
    # BNY copy, which falls in no segment; resolve=False shows both, as the store does.
    write_master(root, BK_SEGS)
    bk_2022 = daily("BK", date(2022, 3, 1), close=50.0)
    store_daily(
        root,
        [
            bk_2022,
            daily("BNY", date(2022, 3, 1), close=50.0),  # the relabelled copy
            daily("BK", date(2026, 5, 20), close=90.0),
            daily("BNY", date(2026, 5, 20), close=90.0),  # copy of BK's last day
            daily("BNY", date(2026, 5, 21), close=91.0),  # BNY's own first day
        ],
    )

    on_0301 = avail(date(2022, 3, 1))  # 2022-03-02 01:00 UTC (EST)
    in_2022 = visible_bars(con, root, "1Day", on_0301)
    assert owned(in_2022) == [("S000730", "BK", date(2022, 3, 1))]
    assert keys(visible_bars(con, root, "1Day", on_0301, resolve=False)) == [
        ("BK", bk_2022.t),
        ("BNY", bk_2022.t),
    ]

    later = visible_bars(con, root, "1Day", avail(BK_CUT))  # 2026-05-22 00:00 UTC
    assert owned(later) == [
        ("S000730", "BK", date(2022, 3, 1)),
        ("S000730", "BK", date(2026, 5, 20)),
        ("S000730", "BNY", date(2026, 5, 21)),
    ]


# real: NKLA segment valid_from 2020-06-04, available 2020-06-08 04:00 UTC; VTIQ's last traded
# bar 2020-06-03 (close 33.97), fillers after it; security S006538
NKLA_FIRST = date(2020, 6, 4)
NKLA_KNOWN = known(date(2020, 6, 8))
VTIQ_SEGS = [
    Seg("S006538", "VTIQ", date(2020, 1, 2), NKLA_FIRST, avail(date(2020, 1, 2)), NKLA_KNOWN),
    Seg("S006538", "NKLA", NKLA_FIRST, None, NKLA_KNOWN, None),
    Seg("S_SPY", "SPY"),
]
VTIQ_BARS = [
    daily("VTIQ", date(2020, 6, 1), close=30.0),
    daily("VTIQ", date(2020, 6, 2), close=32.0),
    daily("VTIQ", date(2020, 6, 3), close=33.97),
    filler("VTIQ", date(2020, 6, 4)),
    filler("VTIQ", date(2020, 6, 5)),
    filler("VTIQ", date(2020, 6, 8)),
    daily("NKLA", date(2020, 6, 4), close=35.0),
    daily("NKLA", date(2020, 6, 5), close=36.0),
    daily("NKLA", date(2020, 6, 8), close=37.0),
]


def test_successor_segment_and_its_bars_are_invisible_until_the_rename_is_knowable(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # Reader on 06-05 evening: NKLA's row is on disk but not knowable, so no NKLA bar comes back
    # (they have no owner yet) and VTIQ is open-ended. The VTIQ fillers of 06-04 / 06-05 belong
    # to VTIQ at that instant, flagged untraded.
    write_master(root, VTIQ_SEGS)
    store_daily(root, VTIQ_BARS)
    on_0605 = avail(date(2020, 6, 5))  # 2020-06-06 00:00 UTC
    assert on_0605 < NKLA_KNOWN

    assert owned(visible_bars(con, root, "1Day", on_0605)) == [
        ("S006538", "VTIQ", date(2020, 6, 1)),
        ("S006538", "VTIQ", date(2020, 6, 2)),
        ("S006538", "VTIQ", date(2020, 6, 3)),
    ]
    with_fillers = visible_bars(con, root, "1Day", on_0605, include_untraded=True)
    assert owned(with_fillers) == [
        ("S006538", "VTIQ", date(2020, 6, 1)),
        ("S006538", "VTIQ", date(2020, 6, 2)),
        ("S006538", "VTIQ", date(2020, 6, 3)),
        ("S006538", "VTIQ", date(2020, 6, 4)),
        ("S006538", "VTIQ", date(2020, 6, 5)),
    ]
    assert with_fillers.column("traded").to_pylist() == [True, True, True, False, False]
    assert seg_rows(visible_master(con, root, on_0605)) == [
        ("S006538", "VTIQ", date(2020, 1, 2), None, None),
        ("S_SPY", "SPY", date(2016, 1, 4), None, None),
    ]
    # the raw store knows nothing of this: NKLA's bars are there for anyone who asks
    raw = visible_bars(con, root, "1Day", on_0605, resolve=False)
    assert [k[0] for k in keys(raw)] == ["NKLA", "NKLA", "VTIQ", "VTIQ", "VTIQ"]


def test_both_names_come_back_as_one_timeline_once_the_rename_is_knowable(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # At 2020-06-08 04:00 UTC the 06-04 and 06-05 NKLA bars (already on disk and available) appear
    # under S006538 right after VTIQ's, ordered by time, not grouped by symbol. VTIQ's fillers
    # now fall outside its segment and under no other: gone even with include_untraded.
    write_master(root, VTIQ_SEGS)
    store_daily(root, VTIQ_BARS)

    table = visible_bars(con, root, "1Day", NKLA_KNOWN)

    assert owned(table) == [
        ("S006538", "VTIQ", date(2020, 6, 1)),
        ("S006538", "VTIQ", date(2020, 6, 2)),
        ("S006538", "VTIQ", date(2020, 6, 3)),
        ("S006538", "NKLA", date(2020, 6, 4)),
        ("S006538", "NKLA", date(2020, 6, 5)),
    ]
    assert table.column("t").to_pylist() == sorted(table.column("t").to_pylist())
    everything = visible_bars(con, root, "1Day", NKLA_KNOWN, include_untraded=True)
    assert owned(everything) == owned(table)
    # the symbol filter is on the stored symbol: asking for NKLA alone gives NKLA's rows only
    assert owned(visible_bars(con, root, "1Day", NKLA_KNOWN, symbols=["NKLA"])) == [
        ("S006538", "NKLA", date(2020, 6, 4)),
        ("S006538", "NKLA", date(2020, 6, 5)),
    ]
    assert seg_rows(visible_master(con, root, NKLA_KNOWN))[:2] == [
        ("S006538", "VTIQ", date(2020, 1, 2), NKLA_FIRST, NKLA_KNOWN),
        ("S006538", "NKLA", NKLA_FIRST, None, None),
    ]


def test_minute_bars_resolve_on_their_session_date(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # 09:30 New York on 2020-06-04 is 13:30 UTC; the session date decides the segment, so NKLA's
    # first minutes are invisible on 06-05 and owned by S006538 from 06-08 04:00 UTC.
    write_master(root, VTIQ_SEGS)
    t0 = datetime(2020, 6, 4, 13, 30, tzinfo=UTC)
    store(
        root, Partition("1Min", 2020, 6, "N"), [make_bar("NKLA", t0), make_bar("NKLA", t0 + MINUTE)]
    )
    store(root, Partition("1Min", 2020, 6, "V"), [make_bar("VTIQ", t0 - 2 * DAY)])  # 06-02 09:30

    on_0605 = avail(date(2020, 6, 5))
    assert owned(visible_bars(con, root, "1Min", on_0605)) == [
        ("S006538", "VTIQ", date(2020, 6, 2)),
    ]
    assert owned(visible_bars(con, root, "1Min", NKLA_KNOWN)) == [
        ("S006538", "VTIQ", date(2020, 6, 2)),
        ("S006538", "NKLA", NKLA_FIRST),
        ("S006538", "NKLA", NKLA_FIRST),
    ]
    assert keys(visible_bars(con, root, "1Min", NKLA_KNOWN))[1:] == [
        ("NKLA", t0),
        ("NKLA", t0 + MINUTE),
    ]


# Fixture shape (see the module docstring: the real analogue is BBUC, not LAC): the cut is on
# disk at 10-04 but filed 11-02, and the old symbol keeps trading in between.
LAC_CUT = date(2023, 10, 4)
LAC_KNOWN = known(date(2023, 11, 2))
LAC_SEGS = [
    Seg("S_LAC", "LAC", date(2020, 1, 2), LAC_CUT, avail(date(2020, 1, 2)), LAC_KNOWN),
    Seg("S_LAC", "LAAC", LAC_CUT, None, LAC_KNOWN, None),
]
LAC_DAYS = [date(2023, 10, 2), date(2023, 10, 3), LAC_CUT, date(2023, 10, 5), date(2023, 10, 19)]
LAAC_DAYS = [LAC_CUT, date(2023, 10, 5), date(2023, 10, 19)]


def test_old_symbol_keeps_its_bars_after_the_cut_until_the_end_is_knowable(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # The LAC row's valid_to (10-04) is on disk; a reader on 10-20 does not know it yet and must
    # keep attributing LAC's 10-04 .. 10-19 bars to S_LAC, while LAAC's bars have no owner. From
    # 11-02 04:00 UTC the cut applies: LAC's post-cut bars fall in no segment, LAAC's take over.
    write_master(root, LAC_SEGS)
    store_daily(root, [daily("LAC", d) for d in LAC_DAYS] + [daily("LAAC", d) for d in LAAC_DAYS])

    on_1020 = datetime(2023, 10, 20, tzinfo=UTC)
    assert on_1020 < LAC_KNOWN
    assert owned(visible_bars(con, root, "1Day", on_1020)) == [
        ("S_LAC", "LAC", date(2023, 10, 2)),
        ("S_LAC", "LAC", date(2023, 10, 3)),
        ("S_LAC", "LAC", LAC_CUT),
        ("S_LAC", "LAC", date(2023, 10, 5)),
        ("S_LAC", "LAC", date(2023, 10, 19)),
    ]
    assert [r[3] for r in seg_rows(visible_master(con, root, on_1020))] == [None]

    after = visible_bars(con, root, "1Day", LAC_KNOWN)
    assert owned(after) == [
        ("S_LAC", "LAC", date(2023, 10, 2)),
        ("S_LAC", "LAC", date(2023, 10, 3)),
        ("S_LAC", "LAAC", LAC_CUT),
        ("S_LAC", "LAAC", date(2023, 10, 5)),
        ("S_LAC", "LAAC", date(2023, 10, 19)),
    ]
    # no session is handed out twice for one security
    sessions = [(r[0], r[2]) for r in owned(after)]
    assert len(sessions) == len(set(sessions))


def test_the_moment_the_end_becomes_knowable_is_sharp(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    write_master(root, LAC_SEGS)
    store_daily(root, [daily("LAC", LAC_CUT), daily("LAAC", LAC_CUT)])

    just_before = LAC_KNOWN - timedelta(microseconds=1)
    assert owned(visible_bars(con, root, "1Day", just_before)) == [("S_LAC", "LAC", LAC_CUT)]
    assert owned(visible_bars(con, root, "1Day", LAC_KNOWN)) == [("S_LAC", "LAAC", LAC_CUT)]


# --- split_day ----------------------------------------------------------------------------------


def test_split_day_marks_the_ex_date_bar_of_that_symbol_only(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # real split: AAPL 4-for-1, ex-date 2020-08-31; MSFT trades the same days without one
    open_master(root, "AAPL", "MSFT")
    days = [date(2020, 8, 28), date(2020, 8, 31), date(2020, 9, 1)]
    store_daily(root, [daily("AAPL", d) for d in days] + [daily("MSFT", d) for d in days])
    write_splits(root, [("AAPL", date(2020, 8, 31), 1.0, 4.0)])

    table = visible_bars(con, root, "1Day", avail(date(2020, 9, 1)))

    assert [(r[1], r[2]) for r in owned(table)] == [("AAPL", d) for d in days] + [
        ("MSFT", d) for d in days
    ]
    assert table.column("split_day").to_pylist() == [False, True, False, False, False, False]
    # the raw view has no such column
    raw = visible_bars(con, root, "1Day", avail(date(2020, 9, 1)), resolve=False)
    assert "split_day" not in raw.schema.names


def test_split_not_yet_knowable_does_not_mark_the_bar(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # Leakage through the splits table: the record is on disk but its available_at lies after
    # as_of, so the ex-date bar is an ordinary bar; one microsecond later it is a split day.
    open_master(root, "AAPL")
    ex = date(2020, 8, 31)
    store_daily(root, [daily("AAPL", date(2020, 8, 28)), daily("AAPL", ex)])
    as_of = avail(ex)  # the bar itself is visible
    write_splits(root, [("AAPL", ex, 1.0, 4.0)], available_at=as_of + timedelta(microseconds=1))

    before = visible_bars(con, root, "1Day", as_of)
    assert before.column("split_day").to_pylist() == [False, False]
    after = visible_bars(con, root, "1Day", as_of + timedelta(microseconds=1))
    assert after.column("split_day").to_pylist() == [False, True]


def test_two_securities_starting_one_symbol_on_the_same_day_are_rejected(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    d = date(2020, 6, 4)
    write_master(root, [Seg("S_A", "X", d, None, known(d)), Seg("S_B", "X", d, None, known(d))])
    with pytest.raises(ValueError, match="overlap"):
        visible_master(con, root, datetime(2026, 10, 1, tzinfo=UTC))


def test_master_with_a_zero_length_segment_is_rejected(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    d = date(2020, 6, 4)
    write_master(root, [Seg("S_A", "X", d, d, known(d), known(d))])
    with pytest.raises(ValueError, match="rebuild"):
        visible_master(con, root, datetime(2026, 10, 1, tzinfo=UTC))


def test_split_day_is_false_everywhere_without_a_splits_file(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    open_master(root, "AAPL")
    store_daily(root, [daily("AAPL", date(2020, 8, 31))])

    table = visible_bars(con, root, "1Day", avail(date(2020, 8, 31)))

    assert table.column("split_day").to_pylist() == [False]


# =============================================================================================
# robustness of the read path
# =============================================================================================


def test_leftover_temp_files_and_junk_year_dirs_are_ignored(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    open_master(root, "AAPL")
    store(root, Partition("1Day", 2026, None, "A"), [daily("AAPL", date(2026, 10, 1))])
    (root / "bars_1d" / "year=2026" / "bucket=B.parquet.tmp").write_bytes(b"half written")
    (root / "bars_1d" / "year=junk").mkdir()
    (root / "bars_1d" / "year=junk" / "bucket=A.parquet").write_bytes(b"not parquet")

    table = visible_bars(con, root, "1Day", as_of=datetime(2026, 10, 3, tzinfo=UTC))

    assert keys(table) == [("AAPL", datetime(2026, 10, 1, 4, tzinfo=UTC))]
    assert [s["year"] for s in bars_status(con, root, "1Day")] == [2026]


@BOTH_MODES
def test_symbols_are_bound_as_parameters_not_spliced_into_sql(
    con: duckdb.DuckDBPyConnection, root: Path, resolve: bool
) -> None:
    open_master(root, "AAPL")
    store(root, Partition("1Day", 2026, None, "A"), [daily("AAPL", date(2026, 10, 1))])

    table = visible_bars(
        con,
        root,
        "1Day",
        as_of=datetime(2026, 10, 3, tzinfo=UTC),
        symbols=["AAPL' OR '1'='1", "x'); DROP TABLE y; --"],
        resolve=resolve,
    )

    assert table.num_rows == 0


# =============================================================================================
# visible_sessions: the market calendar as the data shows it
# =============================================================================================


def test_visible_sessions_are_distinct_ascending_and_last_keeps_the_most_recent(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    days = [date(2026, 9, 28), date(2026, 9, 29), date(2026, 9, 30), date(2026, 10, 1)]
    store(root, Partition("1Day", 2026, None, "A"), [daily("AAPL", d) for d in days])
    store(root, Partition("1Day", 2026, None, "M"), [daily("MSFT", d) for d in days[1:]])
    as_of = datetime(2026, 10, 2, tzinfo=UTC)

    assert visible_sessions(con, root, "1Day", as_of) == days
    assert visible_sessions(con, root, "1Day", as_of, last=2) == days[-2:]
    assert visible_sessions(con, root, "1Day", as_of, last=10) == days


def test_visible_sessions_leave_out_a_session_whose_bar_is_not_yet_available(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # the 2026-10-02 bar is on disk during the session; it only counts from 20:00 New York
    store(
        root,
        Partition("1Day", 2026, None, "A"),
        [daily("AAPL", date(2026, 10, 1)), daily("AAPL", date(2026, 10, 2))],
    )
    during = datetime(2026, 10, 2, 13, tzinfo=UTC)  # 09:00 New York
    assert visible_sessions(con, root, "1Day", during) == [date(2026, 10, 1)]
    assert visible_sessions(con, root, "1Day", AVAILABLE_DAY) == [
        date(2026, 10, 1),
        date(2026, 10, 2),
    ]
    assert visible_sessions(con, root, "1Day", AVAILABLE_DAY - MINUTE, last=1) == [
        date(2026, 10, 1)
    ]


def test_visible_sessions_empty_store_and_bad_arguments(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    assert visible_sessions(con, root, "1Day", datetime(2026, 10, 2, tzinfo=UTC)) == []
    store(root, Partition("1Day", 2026, None, "A"), [daily("AAPL", date(2026, 10, 1))])
    with pytest.raises(ValueError):
        visible_sessions(con, root, "1Day", datetime(2026, 10, 2))  # naive
    with pytest.raises(ValueError):
        visible_sessions(con, root, "1Day", datetime(2026, 10, 2, tzinfo=UTC), last=0)
