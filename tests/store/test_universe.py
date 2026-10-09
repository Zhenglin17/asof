"""Guards against the liquid tier being computed from information that was not yet available:
a bar on disk whose session has not closed, a security-master segment (a rename) not yet
published, bars written later changing an earlier answer. Also guards the ADV arithmetic: the
divisor is the number of sessions in the window, not the number of days the security traded, so
a new listing or a one-day spike cannot buy its way in.

Target API (asof.store.universe, new module):

    LIQUID_SCHEMA, MIN_ADV_USD (50e6), LOOKBACK_SESSIONS (20), LIQUID_CLASSES (("common", "etf"))
    liquid_securities(con, root, as_of, *, min_adv_usd, lookback_sessions, classes) -> pa.Table
    tier_union(con, root, start, end, *, same keyword args) -> pa.Table
    month_starts(start, end) -> list[datetime]   (00:00 New York on the 1st, as UTC)

Fake data provenance -- "real" means read off the store / the security master on 2026-10-08:

- A daily bar becomes visible at 20:00 New York of its session (real: the 2020-01-02 bar has
  available_at 2020-01-03 01:00 UTC); the tests take the instant from ``bars.available_at``.
- The trading calendar is derived from the bars themselves. The fake calendar is weekdays minus
  the real NYSE holidays of the period (2019-12-25, 2020-01-01, 2020-01-20, 2020-02-17,
  2020-05-25). ASSUMPTION: on real data the derived calendar matches NYSE (about 252 sessions a
  year, none on a holiday); check with the audit (distinct session_date per year, and none of
  the dates above as a session).
- VTIQ -> NKLA (security S006538): segment NKLA valid_from 2020-06-04, available_at 2020-06-08
  (real). VTIQ's class row is tagged ``unit`` here only so the join on the latest segment is
  observable; the real class of the VTIQ era is not asserted anywhere.
- BNY carries a copy of BK's 2020-2026 history and the master leaves it without a segment (real).
- TVIX in 2020-H1 traded about $3.4B/day and sits in class ``unknown`` (real).
- Volume-0 placeholder bars exist (real: 1.18M). ASSUMPTION: a session on which only volume-0
  bars exist still counts as a session of the calendar; not asserted here, check on real data
  that no session has zero traded bars store-wide.
- Bars with a null vwap: ASSUMPTION that they occur in the store (the spec prices them off
  close); check with the audit (count of bars with vwap IS NULL).
- Divisor = sessions in the window even when the security traded fewer: a design decision whose
  real-data check is the standard case "a listing in its first month is not in the tier".
- Master segments carry ``end_available_at`` (2026-10-08 design): the instant the event that
  ended the segment became knowable; NULL iff valid_to is NULL. Bars reach the tier through
  ``visible_master``, which masks valid_to while its end is not yet knowable. Real: BK
  [2020-01-02, 2026-05-21) ends knowably at 2026-05-21 04:00 UTC = BNY's available_at.
- "LAC -> LAAC" fixture: a label whose on-disk cut is filed weeks later while the successor
  carries a new symbol. Real analogue: BBUC cut 2026-03-31, end knowable 2026-04-21 04:00 UTC,
  14 traded bars in between (the real LAC is not this shape: Alpaca filed LAC.WI -> LAC on
  2023-10-04 itself). Real-data check: liquid_securities at 2026-04-10 counts BBUC's 04-01 ..
  04-09 bars under BBUC's security_id.
- Every price is 100 (vwap) / 99 (close) so that dollar volume = 100 * volume; the sums are
  exact in float64.
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
from asof.ingest.instrument_class import CLASS_FILE, CLASS_SCHEMA
from asof.ingest.security_master import MASTER_FILE, MASTER_SCHEMA
from asof.store.market import market_root
from asof.store.universe import (
    LIQUID_CLASSES,
    LIQUID_SCHEMA,
    LOOKBACK_SESSIONS,
    MIN_ADV_USD,
    liquid_securities,
    month_starts,
    tier_union,
)
from tests.ingest.fakes import FETCHED_AT, daily

ET = ZoneInfo("America/New_York")
PRICE = 100.0  # vwap of every fake bar: dollar volume = PRICE * volume
CLOSE = 99.0  # differs from vwap so a bar priced off close is distinguishable
M = 1_000_000.0
# 2016-01-04 00:00 New York: the start of the bar store, when every opening segment is known.
KNOWN_EARLY = datetime(2016, 1, 4, tzinfo=ET).astimezone(UTC)
NYSE_HOLIDAYS = {  # real NYSE holidays inside the periods used below
    date(2019, 12, 25),
    date(2020, 1, 1),
    date(2020, 1, 20),
    date(2020, 2, 17),
    date(2020, 5, 25),
    date(2023, 9, 4),
}


# =============================================================================================
# builders
# =============================================================================================


def trading_days(start: date, end: date) -> list[date]:
    days = []
    day = start
    while day <= end:
        if day.weekday() < 5 and day not in NYSE_HOLIDAYS:
            days.append(day)
        day += timedelta(days=1)
    return days


def avail(day: date) -> datetime:
    """When the daily bar of ``day`` becomes visible (20:00 New York), per the ingest rule."""
    return available_at(daily("X", day).t, "1Day")


def usd_bars(symbol: str, days: Sequence[date], usd_per_day: float) -> list[Bar]:
    volume = int(round(usd_per_day / PRICE))
    return [daily(symbol, d, volume=volume, close=CLOSE, vwap=PRICE) for d in days]


def store_daily(root: Path, bars: Sequence[Bar], fetched_at: datetime = FETCHED_AT) -> None:
    """Write bars into the partition files the backfill would use (one per year and bucket).

    Writing a partition replaces it, so pass every bar of a (year, bucket) in one call.
    """
    grouped: dict[tuple[int, str], list[Bar]] = defaultdict(list)
    for bar in bars:
        grouped[(bar.t.astimezone(ET).year, bucket_of(bar.symbol))].append(bar)
    for (year, bucket), group in grouped.items():
        write_partition(Partition("1Day", year, None, bucket).path(root), group, "1Day", fetched_at)


def store_raw(root: Path, rows: Sequence[dict], year: int, bucket: str) -> None:
    """Write a bars file straight from dicts: the only way to put a null vwap on disk."""
    table = pa.Table.from_pylist(rows, schema=BAR_SCHEMA)
    path = Partition("1Day", year, None, bucket).path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def raw_row(symbol: str, day: date, *, volume: int, close: float, vwap: float | None) -> dict:
    bar = daily(symbol, day)
    return {
        "symbol": symbol,
        "t": bar.t,
        "session_date": day,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": volume,
        "trade_count": 1,
        "vwap": vwap,
        "available_at": avail(day),
        "fetched_at": FETCHED_AT,
    }


@dataclass(frozen=True)
class Seg:
    """One security-master segment plus the class its row in the class table carries."""

    security_id: str
    symbol: str
    valid_from: date = date(2016, 1, 4)
    valid_to: date | None = None
    cls: str = "common"
    available_at: datetime = KNOWN_EARLY
    end_available_at: datetime | None = None  # must be set iff valid_to is set


def known(day: date) -> datetime:
    """00:00 New York of ``day`` as UTC: when a rename-created segment (and the end of the one it
    replaces) becomes knowable. Real: BNY's segment, 2026-05-21 04:00 UTC."""
    return datetime.combine(day, datetime.min.time(), tzinfo=ET).astimezone(UTC)


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
            "available_at": pa.array(
                [s.available_at for s in segs], MASTER_SCHEMA.field("available_at").type
            ),
            "end_available_at": pa.array(
                [s.end_available_at for s in segs], pa.timestamp("us", tz="UTC")
            ),
        },
        schema=MASTER_SCHEMA,
    )
    path = root / MASTER_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


def write_classes(root: Path, segs: Sequence[Seg]) -> Path:
    table = pa.table(
        {
            "security_id": pa.array([s.security_id for s in segs], pa.string()),
            "symbol": pa.array([s.symbol for s in segs], pa.string()),
            "valid_from": pa.array([s.valid_from for s in segs], pa.date32()),
            "valid_to": pa.array([s.valid_to for s in segs], pa.date32()),
            "name": pa.array([f"{s.symbol} name" for s in segs], pa.string()),
            "name_source": pa.array(["alpaca_active" for _ in segs], pa.string()),
            "class": pa.array([s.cls for s in segs], pa.string()),
            "rule": pa.array(["test" for _ in segs], pa.string()),
        },
        schema=CLASS_SCHEMA,
    )
    path = root / CLASS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


def write_symbols(root: Path, segs: Sequence[Seg]) -> None:
    write_master(root, segs)
    write_classes(root, segs)


def rows(table: pa.Table) -> list[dict]:
    return table.to_pylist()


def symbols(table: pa.Table) -> list[str]:
    return table.column("symbol").to_pylist()


# =============================================================================================
# fixtures
# =============================================================================================


@pytest.fixture
def con() -> Iterator[duckdb.DuckDBPyConnection]:
    connection = duckdb.connect()
    yield connection
    connection.close()


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return market_root(tmp_path)


# 25 sessions, 2020-01-02 .. 2020-02-06 (EST: a session's bar is visible next day 01:00 UTC)
JAN = trading_days(date(2020, 1, 2), date(2020, 2, 6))
JAN_WINDOW = JAN[-LOOKBACK_SESSIONS:]
JAN_END = avail(JAN[-1])  # 2020-02-07 01:00 UTC

JAN_SEGS = [
    Seg("S_SPY", "SPY", cls="etf"),
    Seg("S_AAPL", "AAPL", cls="common"),
    Seg("S_SMALL", "SMALL", cls="common"),
    Seg("S_ACME_WS", "ACME.WS", cls="warrant"),
    Seg("S_TVIX", "TVIX", cls="unknown"),  # real: TVIX 2020-H1 is unknown at ~$3.4B/day
]


@pytest.fixture
def jan_world(root: Path) -> Path:
    """Five securities trading every one of the 25 sessions at a constant dollar volume."""
    write_symbols(root, JAN_SEGS)
    store_daily(
        root,
        usd_bars("SPY", JAN, 200 * M)
        + usd_bars("AAPL", JAN, 100 * M)
        + usd_bars("SMALL", JAN, 10 * M)
        + usd_bars("ACME.WS", JAN, 500 * M)
        + usd_bars("TVIX", JAN, 3_400 * M),
    )
    return root


# =============================================================================================
# fixture sanity: the facts the arithmetic below relies on
# =============================================================================================


def test_fixture_calendar_and_availability() -> None:
    assert len(JAN) == 25
    assert JAN[0] == date(2020, 1, 2)
    assert date(2020, 1, 20) not in JAN  # MLK day
    assert JAN_WINDOW[0] == date(2020, 1, 9)
    assert avail(date(2020, 1, 2)) == datetime(2020, 1, 3, 1, tzinfo=UTC)  # real
    assert avail(date(2020, 6, 5)) == datetime(2020, 6, 6, 0, tzinfo=UTC)  # EDT
    assert JAN_END == datetime(2020, 2, 7, 1, tzinfo=UTC)


# =============================================================================================
# normal path
# =============================================================================================


def test_constants_and_schema() -> None:
    assert MIN_ADV_USD == 50e6
    assert LOOKBACK_SESSIONS == 20
    assert LIQUID_CLASSES == ("common", "etf")
    assert LIQUID_SCHEMA.names == [
        "as_of",
        "security_id",
        "symbol",
        "class",
        "adv_usd",
        "sessions_traded",
    ]
    assert LIQUID_SCHEMA.field("as_of").type == pa.timestamp("us", tz="UTC")
    assert LIQUID_SCHEMA.field("adv_usd").type == pa.float64()
    assert LIQUID_SCHEMA.field("sessions_traded").type == pa.int64()
    for name in ("security_id", "symbol", "class"):
        assert LIQUID_SCHEMA.field(name).type == pa.string()


def test_liquid_tier_rows_order_values_and_schema(
    con: duckdb.DuckDBPyConnection, jan_world: Path
) -> None:
    table = liquid_securities(con, jan_world, JAN_END)

    assert table.schema.equals(LIQUID_SCHEMA)
    out = rows(table)
    assert [(r["security_id"], r["symbol"], r["class"]) for r in out] == [
        ("S_SPY", "SPY", "etf"),
        ("S_AAPL", "AAPL", "common"),
    ]
    assert [r["adv_usd"] for r in out] == pytest.approx([200 * M, 100 * M])
    assert [r["sessions_traded"] for r in out] == [20, 20]
    assert [r["as_of"] for r in out] == [JAN_END, JAN_END]


def test_classes_none_returns_every_class_including_unknown(
    con: duckdb.DuckDBPyConnection, jan_world: Path
) -> None:
    # the audit's view: an unknown segment at $3.4B/day must be surfaced, not hidden
    table = liquid_securities(con, jan_world, JAN_END, classes=None)
    assert symbols(table) == ["TVIX", "ACME.WS", "SPY", "AAPL"]
    assert table.column("class").to_pylist() == ["unknown", "warrant", "etf", "common"]


def test_classes_argument_is_an_exact_filter(
    con: duckdb.DuckDBPyConnection, jan_world: Path
) -> None:
    assert symbols(liquid_securities(con, jan_world, JAN_END, classes=("etf",))) == ["SPY"]
    assert symbols(liquid_securities(con, jan_world, JAN_END, classes=("unknown",))) == ["TVIX"]
    assert symbols(liquid_securities(con, jan_world, JAN_END, classes=())) == []


def test_thresholds_are_keyword_arguments(con: duckdb.DuckDBPyConnection, jan_world: Path) -> None:
    assert symbols(liquid_securities(con, jan_world, JAN_END, min_adv_usd=5 * M)) == [
        "SPY",
        "AAPL",
        "SMALL",
    ]
    assert symbols(liquid_securities(con, jan_world, JAN_END, min_adv_usd=150 * M)) == ["SPY"]
    short = liquid_securities(con, jan_world, JAN_END, lookback_sessions=5)
    assert rows(short)[0]["sessions_traded"] == 5
    assert rows(short)[0]["adv_usd"] == pytest.approx(200 * M)


def test_as_of_in_another_zone_is_reported_as_utc(
    con: duckdb.DuckDBPyConnection, jan_world: Path
) -> None:
    as_of_ny = JAN_END.astimezone(ET)
    table = liquid_securities(con, jan_world, as_of_ny)
    assert set(table.column("as_of").to_pylist()) == {JAN_END}
    assert all(v.utcoffset() == timedelta(0) for v in table.column("as_of").to_pylist())


# -- renames: dollar volume is summed per security across symbols ---------------------------------

# real: VTIQ became NKLA on 2020-06-04; the master's NKLA segment is dated 2020-06-04 but was
# only knowable on 2020-06-08 (available_at 00:00 New York).
RENAME_DAY = date(2020, 6, 4)
NKLA_KNOWN = datetime(2020, 6, 8, tzinfo=ET).astimezone(UTC)
MAY_JUN = trading_days(date(2020, 5, 4), date(2020, 6, 12))
VTIQ_DAYS = [d for d in MAY_JUN if d < RENAME_DAY]
NKLA_DAYS = [d for d in MAY_JUN if d >= RENAME_DAY]
VTIQ_USD = 60 * M
NKLA_USD = 300 * M


def rename_segments(*, vtiq_closed: bool = True) -> list[Seg]:
    return [
        Seg(
            "S006538",
            "VTIQ",
            valid_to=RENAME_DAY if vtiq_closed else None,
            cls="unit",
            end_available_at=NKLA_KNOWN if vtiq_closed else None,  # one event, one instant
        ),
        Seg("S006538", "NKLA", valid_from=RENAME_DAY, cls="common", available_at=NKLA_KNOWN),
        Seg("S_SPY", "SPY", cls="etf"),
    ]


@pytest.fixture
def rename_world(root: Path) -> Path:
    write_symbols(root, rename_segments())
    store_daily(
        root,
        usd_bars("VTIQ", VTIQ_DAYS, VTIQ_USD)
        + usd_bars("NKLA", NKLA_DAYS, NKLA_USD)
        + usd_bars("SPY", MAY_JUN, 200 * M),
    )
    return root


def test_rename_fixture_sanity() -> None:
    assert len(MAY_JUN) == 29
    assert date(2020, 5, 25) not in MAY_JUN
    assert MAY_JUN[-20] == date(2020, 5, 15)
    assert len(NKLA_DAYS) == 7 and len(VTIQ_DAYS) == 22


def test_dollar_volume_is_summed_per_security_across_a_rename(
    con: duckdb.DuckDBPyConnection, rename_world: Path
) -> None:
    # window 2020-05-15 .. 2020-06-12: 13 VTIQ sessions + 7 NKLA sessions, one security
    table = liquid_securities(con, rename_world, avail(date(2020, 6, 12)))
    [spy, nkla] = rows(table)
    assert spy["symbol"] == "SPY"
    assert (nkla["security_id"], nkla["symbol"], nkla["class"]) == ("S006538", "NKLA", "common")
    assert nkla["adv_usd"] == pytest.approx((13 * VTIQ_USD + 7 * NKLA_USD) / 20)
    assert nkla["sessions_traded"] == 20
    # the two names never appear as two rows
    assert symbols(table).count("VTIQ") == 0


def test_reported_symbol_and_class_come_from_the_latest_visible_segment(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # same volume under both names: only the segment dates decide which name is reported
    write_symbols(
        root,
        [
            Seg("S1", "OLD", valid_to=JAN[10], cls="unit", end_available_at=known(JAN[10])),
            Seg("S1", "NEW", valid_from=JAN[10], cls="common", available_at=known(JAN[10])),
        ],
    )
    store_daily(root, usd_bars("OLD", JAN[:10], 100 * M) + usd_bars("NEW", JAN[10:], 100 * M))

    [row] = rows(liquid_securities(con, root, JAN_END))
    assert (row["security_id"], row["symbol"], row["class"]) == ("S1", "NEW", "common")
    assert row["adv_usd"] == pytest.approx(100 * M)


# -- month_starts / tier_union --------------------------------------------------------------------


def test_month_starts_are_midnight_new_york_as_utc_inclusive_of_both_months() -> None:
    starts = month_starts(date(2021, 3, 15), date(2021, 7, 1))
    assert starts == [
        datetime(2021, 3, 1, 5, tzinfo=UTC),  # EST
        datetime(2021, 4, 1, 4, tzinfo=UTC),  # EDT
        datetime(2021, 5, 1, 4, tzinfo=UTC),
        datetime(2021, 6, 1, 4, tzinfo=UTC),
        datetime(2021, 7, 1, 4, tzinfo=UTC),
    ]
    assert all(s.utcoffset() == timedelta(0) for s in starts)
    assert month_starts(date(2021, 3, 1), date(2021, 3, 31)) == [
        datetime(2021, 3, 1, 5, tzinfo=UTC)
    ]
    assert month_starts(date(2020, 12, 31), date(2021, 1, 1)) == [
        datetime(2020, 12, 1, 5, tzinfo=UTC),
        datetime(2021, 1, 1, 5, tzinfo=UTC),
    ]


DEC = trading_days(date(2019, 12, 2), date(2019, 12, 31))
JAN_FULL = trading_days(date(2020, 1, 2), date(2020, 1, 31))
UNION_SEGS = [
    Seg("S_SPY", "SPY", cls="etf"),
    Seg("S_LATE", "LATE", cls="common"),
    Seg("S_TVIX", "TVIX", cls="unknown"),
]


@pytest.fixture
def union_world(root: Path) -> Path:
    """SPY and TVIX trade in December and January; LATE only starts trading in January."""
    write_symbols(root, UNION_SEGS)
    store_daily(
        root,
        usd_bars("SPY", DEC + JAN_FULL, 200 * M)
        + usd_bars("TVIX", DEC + JAN_FULL, 3_400 * M)
        + usd_bars("LATE", JAN_FULL, 100 * M),
    )
    return root


def test_tier_union_is_one_liquid_tier_per_month_start(
    con: duckdb.DuckDBPyConnection, union_world: Path
) -> None:
    assert len(DEC) == 21 and len(JAN_FULL) == 21
    jan1 = datetime(2020, 1, 1, 5, tzinfo=UTC)
    feb1 = datetime(2020, 2, 1, 5, tzinfo=UTC)

    table = tier_union(con, union_world, date(2020, 1, 1), date(2020, 2, 15))

    assert table.schema.equals(LIQUID_SCHEMA)
    assert [(r["as_of"], r["symbol"]) for r in rows(table)] == [
        (jan1, "SPY"),
        (feb1, "SPY"),
        (feb1, "LATE"),
    ]
    # each month's rows equal the single-as_of call
    assert [r for r in rows(table) if r["as_of"] == feb1] == rows(
        liquid_securities(con, union_world, feb1)
    )


def test_tier_union_passes_the_keyword_arguments_through(
    con: duckdb.DuckDBPyConnection, union_world: Path
) -> None:
    everything = tier_union(con, union_world, date(2020, 1, 1), date(2020, 2, 1), classes=None)
    assert [(r["as_of"].month, r["symbol"]) for r in rows(everything)] == [
        (1, "TVIX"),
        (1, "SPY"),
        (2, "TVIX"),
        (2, "SPY"),
        (2, "LATE"),
    ]
    strict = tier_union(con, union_world, date(2020, 1, 1), date(2020, 2, 1), min_adv_usd=150 * M)
    assert symbols(strict) == ["SPY", "SPY"]


# =============================================================================================
# boundaries
# =============================================================================================


def test_window_is_the_last_sessions_not_calendar_days(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # EARLY trades $5B/day on the first five sessions only; the 20-session window starts at the
    # sixth session, so it is out -- and in again when the window is widened to 25.
    write_symbols(root, [Seg("S_SPY", "SPY", cls="etf"), Seg("S_EARLY", "EARLY")])
    store_daily(root, usd_bars("SPY", JAN, 200 * M) + usd_bars("EARLY", JAN[:5], 5_000 * M))

    assert symbols(liquid_securities(con, root, JAN_END)) == ["SPY"]
    wide = liquid_securities(con, root, JAN_END, lookback_sessions=25)
    assert symbols(wide) == ["EARLY", "SPY"]
    [early, _] = rows(wide)
    assert early["adv_usd"] == pytest.approx(5 * 5_000 * M / 25)
    assert early["sessions_traded"] == 5


def test_sparse_security_divides_by_the_window_not_by_its_own_sessions(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # a listing that traded 5 of the 20 sessions at $400M/day has ADV $100M, not $400M
    write_symbols(root, [Seg("S_SPY", "SPY", cls="etf"), Seg("S_NEW", "NEW")])
    store_daily(root, usd_bars("SPY", JAN, 200 * M) + usd_bars("NEW", JAN[-5:], 400 * M))

    [spy, new] = rows(liquid_securities(con, root, JAN_END))
    assert (new["symbol"], new["sessions_traded"]) == ("NEW", 5)
    assert new["adv_usd"] == pytest.approx(5 * 400 * M / 20)
    assert spy["sessions_traded"] == 20


def test_one_day_spike_does_not_qualify(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    # a SPAC that printed $900M on one session and nothing else: 900M / 20 = 45M < 50M
    write_symbols(root, [Seg("S_SPY", "SPY", cls="etf"), Seg("S_SPAC", "SPAC")])
    store_daily(root, usd_bars("SPY", JAN, 200 * M) + usd_bars("SPAC", JAN[-1:], 900 * M))

    assert symbols(liquid_securities(con, root, JAN_END)) == ["SPY"]


def test_threshold_is_strict(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    write_symbols(root, [Seg("S_AT", "AT"), Seg("S_ABOVE", "ABOVE")])
    store_daily(
        root,
        [daily("AT", d, volume=500_000, close=CLOSE, vwap=PRICE) for d in JAN]  # 50,000,000
        + [daily("ABOVE", d, volume=500_001, close=CLOSE, vwap=PRICE) for d in JAN],  # 50,000,100
    )

    table = liquid_securities(con, root, JAN_END)
    assert symbols(table) == ["ABOVE"]
    assert rows(table)[0]["adv_usd"] == pytest.approx(50_000_100.0)
    assert symbols(liquid_securities(con, root, JAN_END, min_adv_usd=49_999_999.0)) == [
        "ABOVE",
        "AT",
    ]


def test_equal_adv_orders_by_security_id(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    write_symbols(root, [Seg("S_B", "ZZZ"), Seg("S_A", "YYY")])
    store_daily(root, usd_bars("ZZZ", JAN, 100 * M) + usd_bars("YYY", JAN, 100 * M))

    assert [r["security_id"] for r in rows(liquid_securities(con, root, JAN_END))] == ["S_A", "S_B"]


def test_fewer_visible_sessions_than_lookback_divides_by_what_exists(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # only three sessions in the whole store; AAPL traded on two of them
    write_symbols(root, [Seg("S_SPY", "SPY", cls="etf"), Seg("S_AAPL", "AAPL")])
    store_daily(root, usd_bars("SPY", JAN[:3], 200 * M) + usd_bars("AAPL", JAN[1:3], 100 * M))

    [spy, aapl] = rows(liquid_securities(con, root, avail(JAN[2])))
    assert spy["adv_usd"] == pytest.approx(200 * M)
    assert aapl["adv_usd"] == pytest.approx(2 * 100 * M / 3)
    assert (spy["sessions_traded"], aapl["sessions_traded"]) == (3, 2)


def test_volume_zero_bars_add_nothing_and_are_not_traded_sessions(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # real: 1.18M placeholder bars with volume 0 sit in the store
    write_symbols(root, [Seg("S_SPY", "SPY", cls="etf"), Seg("S_GAP", "GAP")])
    filler = [daily("GAP", d, volume=0, close=CLOSE, vwap=PRICE) for d in JAN[:-5]]
    store_daily(root, usd_bars("SPY", JAN, 200 * M) + filler + usd_bars("GAP", JAN[-5:], 400 * M))

    [_, gap] = rows(liquid_securities(con, root, JAN_END))
    assert (gap["symbol"], gap["sessions_traded"]) == ("GAP", 5)
    assert gap["adv_usd"] == pytest.approx(5 * 400 * M / 20)


def test_null_vwap_falls_back_to_close(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    # ASSUMPTION: bars with vwap NULL exist in the store; priced off close instead
    write_symbols(root, [Seg("S_NOVW", "NOVW")])
    store_raw(
        root,
        [raw_row("NOVW", d, volume=2_000_000, close=50.0, vwap=None) for d in JAN],
        2020,
        "N",
    )

    [row] = rows(liquid_securities(con, root, JAN_END))
    assert row["adv_usd"] == pytest.approx(50.0 * 2_000_000)


def test_bars_outside_their_segment_dates_are_ignored(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # valid_from inclusive, valid_to exclusive: bars on JAN[10] .. JAN[14] belong, JAN[15] not
    write_symbols(
        root,
        [Seg("S_X", "X", valid_from=JAN[10], valid_to=JAN[15], end_available_at=known(JAN[15]))],
    )
    store_daily(root, usd_bars("X", JAN, 400 * M))

    [row] = rows(liquid_securities(con, root, JAN_END))
    assert row["sessions_traded"] == 5
    assert row["adv_usd"] == pytest.approx(5 * 400 * M / 20)

    write_symbols(
        root,
        [Seg("S_X", "X", valid_from=JAN[10], valid_to=JAN[16], end_available_at=known(JAN[16]))],
    )
    [row] = rows(liquid_securities(con, root, JAN_END))
    assert row["sessions_traded"] == 6


def test_bars_of_a_symbol_without_a_segment_are_ignored(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # real: BNY holds a copy of BK's history that the master deliberately leaves unsegmented
    write_symbols(root, [Seg("S_BK", "BK")])
    store_daily(root, usd_bars("BK", JAN, 100 * M) + usd_bars("BNY", JAN, 100 * M))

    assert symbols(liquid_securities(con, root, JAN_END)) == ["BK"]
    assert symbols(liquid_securities(con, root, JAN_END, classes=None)) == ["BK"]


def test_empty_store_gives_an_empty_table_with_the_schema(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    write_symbols(root, JAN_SEGS)
    table = liquid_securities(con, root, JAN_END)
    assert table.num_rows == 0
    assert table.schema.equals(LIQUID_SCHEMA)
    union = tier_union(con, root, date(2020, 1, 1), date(2020, 2, 1))
    assert union.num_rows == 0
    assert union.schema.equals(LIQUID_SCHEMA)


def test_missing_master_or_class_file_raises(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    store_daily(root, usd_bars("SPY", JAN, 200 * M))
    with pytest.raises(FileNotFoundError):
        liquid_securities(con, root, JAN_END)
    write_master(root, JAN_SEGS)  # master only
    with pytest.raises(FileNotFoundError):
        liquid_securities(con, root, JAN_END)
    (root / MASTER_FILE).unlink()
    write_classes(root, JAN_SEGS)  # class table only
    with pytest.raises(FileNotFoundError):
        liquid_securities(con, root, JAN_END)


def test_visible_segment_without_a_class_row_is_a_stale_class_table(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # real failure mode: the master was rebuilt, the class table was not
    write_master(root, [Seg("S_SPY", "SPY", cls="etf"), Seg("S_AAPL", "AAPL")])
    write_classes(root, [Seg("S_SPY", "SPY", cls="etf")])
    store_daily(root, usd_bars("SPY", JAN, 200 * M) + usd_bars("AAPL", JAN, 1 * M))

    with pytest.raises(ValueError, match=r"stale|market instruments"):
        liquid_securities(con, root, JAN_END)


def test_segment_not_yet_visible_may_lack_a_class_row(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # the staleness check is on visible segments: a future segment without a row is not an error
    write_master(
        root,
        [Seg("S_SPY", "SPY", cls="etf"), Seg("S_F", "F", available_at=JAN_END + timedelta(days=1))],
    )
    write_classes(root, [Seg("S_SPY", "SPY", cls="etf")])
    store_daily(root, usd_bars("SPY", JAN, 200 * M))

    assert symbols(liquid_securities(con, root, JAN_END)) == ["SPY"]


def test_duplicate_segment_keys_are_rejected(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    # two master rows with the same (security_id, symbol, valid_from) would list SPY twice
    write_symbols(root, [Seg("S_SPY", "SPY", cls="etf"), Seg("S_SPY", "SPY", cls="etf")])
    store_daily(root, usd_bars("SPY", JAN, 200 * M))
    with pytest.raises(ValueError, match="duplicate"):
        liquid_securities(con, root, JAN_END)


def test_overlapping_segments_of_one_symbol_are_rejected(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # X owned by two securities over JAN[5..10): each bar would be counted twice
    write_symbols(
        root,
        [
            Seg("S_A", "X", valid_to=JAN[10], end_available_at=known(JAN[10])),
            Seg("S_B", "X", valid_from=JAN[5]),
        ],
    )
    store_daily(root, usd_bars("X", JAN, 400 * M))
    with pytest.raises(ValueError, match="overlap"):
        liquid_securities(con, root, JAN_END)
    # back to back is fine: S_A has 5 sessions in the window (5 * 400M / 20 = 100M), S_B 15
    write_symbols(
        root,
        [
            Seg("S_A", "X", valid_to=JAN[10], end_available_at=known(JAN[10])),
            Seg("S_B", "X", valid_from=JAN[10], available_at=known(JAN[10])),
        ],
    )
    assert [r["security_id"] for r in rows(liquid_securities(con, root, JAN_END))] == ["S_B", "S_A"]


def test_segment_without_available_at_is_rejected(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    write_symbols(root, [Seg("S_SPY", "SPY", cls="etf")])
    table = pq.read_table(root / MASTER_FILE)
    idx = table.schema.get_field_index("available_at")
    table = table.set_column(
        idx, "available_at", pa.array([None], MASTER_SCHEMA.field("available_at").type)
    )
    pq.write_table(table, root / MASTER_FILE)
    store_daily(root, usd_bars("SPY", JAN, 200 * M))
    with pytest.raises(ValueError, match="available_at"):
        liquid_securities(con, root, JAN_END)


def test_closed_segment_without_end_available_at_is_rejected(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # the master invariant valid_to IS NULL <=> end_available_at IS NULL; a file violating it
    # was built by an older version and must be rebuilt, not read half-way
    write_symbols(root, [Seg("S_SPY", "SPY", cls="etf"), Seg("S_X", "X", valid_to=JAN[10])])
    store_daily(root, usd_bars("SPY", JAN, 200 * M) + usd_bars("X", JAN, 400 * M))
    with pytest.raises(ValueError, match="rebuild"):
        liquid_securities(con, root, JAN_END)
    # and the other direction
    write_symbols(
        root, [Seg("S_SPY", "SPY", cls="etf"), Seg("S_X", "X", end_available_at=known(JAN[10]))]
    )
    with pytest.raises(ValueError, match="rebuild"):
        liquid_securities(con, root, JAN_END)


def test_naive_as_of_is_rejected(con: duckdb.DuckDBPyConnection, jan_world: Path) -> None:
    with pytest.raises(ValueError):
        liquid_securities(con, jan_world, datetime(2020, 2, 7, 1))


# =============================================================================================
# leakage
# =============================================================================================


def test_bar_on_disk_but_not_yet_available_is_not_counted(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # SPIKE prints $2B on the last session; that bar is on disk during the session (08:00 New
    # York) but only becomes available at 20:00 New York. as_of sits between the previous
    # session's availability and this one's.
    write_symbols(root, [Seg("S_SPY", "SPY", cls="etf"), Seg("S_SPIKE", "SPIKE")])
    store_daily(
        root,
        usd_bars("SPY", JAN, 200 * M)
        + usd_bars("SPIKE", JAN[:-1], 10 * M)
        + usd_bars("SPIKE", JAN[-1:], 2_000 * M),
    )
    during = avail(JAN[-2]) + timedelta(hours=12)  # 2020-02-06 13:00 UTC, 08:00 New York
    assert avail(JAN[-2]) < during < avail(JAN[-1])

    assert symbols(liquid_securities(con, root, during)) == ["SPY"]
    assert symbols(liquid_securities(con, root, avail(JAN[-1]) - timedelta(microseconds=1))) == [
        "SPY"
    ]
    after = liquid_securities(con, root, avail(JAN[-1]))
    assert symbols(after) == ["SPY", "SPIKE"]
    assert rows(after)[1]["adv_usd"] == pytest.approx((19 * 10 * M + 2_000 * M) / 20)


def test_window_does_not_slide_into_the_future(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    # at an as_of in the middle of the store the window ends at the last visible session, so
    # a security that only trades afterwards is absent and the earlier one is still inside
    write_symbols(
        root, [Seg("S_SPY", "SPY", cls="etf"), Seg("S_FUT", "FUT"), Seg("S_PAST", "PAST")]
    )
    store_daily(
        root,
        usd_bars("SPY", JAN, 200 * M)
        + usd_bars("FUT", JAN[20:], 5_000 * M)
        + usd_bars("PAST", JAN[:3], 5_000 * M),
    )

    # sessions 0..19 visible: window = JAN[0:20], PAST's three sessions inside it
    table = liquid_securities(con, root, avail(JAN[19]))
    assert symbols(table) == ["PAST", "SPY"]
    assert rows(table)[0]["adv_usd"] == pytest.approx(3 * 5_000 * M / 20)
    # all 25 visible: window = JAN[5:25], PAST out, FUT in
    assert symbols(liquid_securities(con, root, JAN_END)) == ["FUT", "SPY"]


def test_master_segment_not_yet_available_does_not_exist(
    con: duckdb.DuckDBPyConnection, rename_world: Path
) -> None:
    # real: NKLA's segment is dated 2020-06-04 but available 2020-06-08. A reader on 06-05
    # sees only the VTIQ segment: NKLA's bars have no owner and the security is still VTIQ.
    on_0605 = avail(date(2020, 6, 5))
    assert on_0605 < NKLA_KNOWN
    [_, row] = rows(liquid_securities(con, rename_world, on_0605, classes=None))
    assert (row["security_id"], row["symbol"], row["class"]) == ("S006538", "VTIQ", "unit")
    # window 05-08 .. 06-05 has 20 sessions; VTIQ traded 18 of them (06-04 and 06-05 are NKLA's)
    assert row["sessions_traded"] == 18
    assert row["adv_usd"] == pytest.approx(18 * VTIQ_USD / 20)
    # with the default classes the "unit" era is simply not in the tier
    assert symbols(liquid_securities(con, rename_world, on_0605)) == ["SPY"]

    # one session after the segment is published: NKLA, both names summed
    on_0608 = avail(date(2020, 6, 8))
    assert on_0608 > NKLA_KNOWN
    [_, row] = rows(liquid_securities(con, rename_world, on_0608))
    assert (row["symbol"], row["class"], row["sessions_traded"]) == ("NKLA", "common", 20)
    assert row["adv_usd"] == pytest.approx((17 * VTIQ_USD + 3 * NKLA_USD) / 20)


def test_valid_to_written_later_does_not_change_the_earlier_adv(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # The master on disk closes VTIQ on 2020-06-04 -- a fact only knowable on 06-08. A reader on
    # 06-05 must get the same answer whether or not that valid_to has been written: Alpaca files
    # the bars from 06-04 on under NKLA, so there is nothing for the closed segment to lose.
    bars = usd_bars("VTIQ", VTIQ_DAYS, VTIQ_USD) + usd_bars("NKLA", NKLA_DAYS, NKLA_USD)
    bars += usd_bars("SPY", MAY_JUN, 200 * M)
    store_daily(root, bars)
    on_0605 = avail(date(2020, 6, 5))

    write_symbols(root, rename_segments(vtiq_closed=True))
    closed = rows(liquid_securities(con, root, on_0605, classes=None))
    write_symbols(root, rename_segments(vtiq_closed=False))
    still_open = rows(liquid_securities(con, root, on_0605, classes=None))

    assert closed == still_open
    assert [r["symbol"] for r in closed] == ["SPY", "VTIQ"]


# Fixture shape (module docstring: the real analogue is BBUC): the cut is on disk at 10-04 but
# filed 11-02, and the old symbol keeps printing traded bars in between (21 sessions here).
LAC_CUT = date(2023, 10, 4)
LAC_KNOWN = known(date(2023, 11, 2))  # 2023-11-02 04:00 UTC
SEP_NOV = trading_days(date(2023, 9, 5), date(2023, 11, 10))
LAC_LAST_TRADED = date(2023, 11, 1)
LAC_DAYS = [d for d in SEP_NOV if d <= LAC_LAST_TRADED]
LAAC_DAYS = [d for d in SEP_NOV if d >= LAC_CUT]
LAC_USD = 80 * M
LAAC_USD = 120 * M


def test_lac_fixture_sanity() -> None:
    assert len([d for d in LAC_DAYS if d >= LAC_CUT]) == 21  # fixture: 21 traded post-cut bars
    assert LAC_KNOWN == datetime(2023, 11, 2, 4, tzinfo=UTC)
    assert SEP_NOV.index(date(2023, 10, 20)) == 33 and SEP_NOV.index(date(2023, 11, 3)) == 43


def test_old_symbol_keeps_earning_adv_until_its_end_is_knowable(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # On disk LAC ends 10-04 and LAAC starts there, both facts knowable 11-02. A reader on 10-20
    # knows neither: LAC is open and owns every LAC bar in the window, LAAC's bars have no owner.
    # Once the record is knowable, LAC's bars from 10-04 on fall in no segment (the cut applies)
    # and only LAAC's are counted -- the same days are never counted twice.
    write_symbols(
        root,
        [
            Seg("S_LAC", "LAC", valid_to=LAC_CUT, end_available_at=LAC_KNOWN),
            Seg("S_LAC", "LAAC", valid_from=LAC_CUT, available_at=LAC_KNOWN),
            Seg("S_SPY", "SPY", cls="etf"),
        ],
    )
    store_daily(
        root,
        usd_bars("LAC", LAC_DAYS, LAC_USD)
        + usd_bars("LAAC", LAAC_DAYS, LAAC_USD)
        + usd_bars("SPY", SEP_NOV, 200 * M),
    )

    # window 09-25 .. 10-20, all 20 sessions LAC bars, 13 of them after the on-disk valid_to
    between = avail(date(2023, 10, 20))
    assert between < LAC_KNOWN
    [_, lac] = rows(liquid_securities(con, root, between))
    assert (lac["security_id"], lac["symbol"]) == ("S_LAC", "LAC")
    assert lac["sessions_traded"] == 20
    assert lac["adv_usd"] == pytest.approx(LAC_USD)

    # window 10-09 .. 11-03: LAAC owns 20 sessions; LAC's 18 post-cut bars count for nobody
    after = avail(date(2023, 11, 3))
    assert after > LAC_KNOWN
    [_, laac] = rows(liquid_securities(con, root, after))
    assert (laac["security_id"], laac["symbol"]) == ("S_LAC", "LAAC")
    assert laac["sessions_traded"] == 20
    assert laac["adv_usd"] == pytest.approx(LAAC_USD)
    assert symbols(liquid_securities(con, root, after)).count("LAC") == 0


def test_later_bars_written_to_the_store_do_not_change_an_earlier_answer(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    write_symbols(root, [Seg("S_SPY", "SPY", cls="etf"), Seg("S_SMALL", "SMALL")])
    first = usd_bars("SPY", JAN[:20], 200 * M) + usd_bars("SMALL", JAN[:20], 10 * M)
    store_daily(root, first)
    as_of = avail(JAN[19])
    before = rows(liquid_securities(con, root, as_of))
    assert [r["symbol"] for r in before] == ["SPY"]

    # the backfill re-fetches the year: same partitions, now with five more sessions in which
    # SMALL trades $5B/day
    later = usd_bars("SPY", JAN[20:], 200 * M) + usd_bars("SMALL", JAN[20:], 5_000 * M)
    store_daily(root, first + later, fetched_at=FETCHED_AT + timedelta(days=30))

    assert rows(liquid_securities(con, root, as_of)) == before
    assert symbols(liquid_securities(con, root, JAN_END)) == ["SMALL", "SPY"]


def test_every_tier_row_only_uses_visible_bars(
    con: duckdb.DuckDBPyConnection, jan_world: Path
) -> None:
    # at any as_of, a security's ADV equals the arithmetic over bars with available_at <= as_of
    for k in (0, 4, 12, 19, 24):
        as_of = avail(JAN[k])
        window = JAN[: k + 1][-LOOKBACK_SESSIONS:]
        out = rows(liquid_securities(con, jan_world, as_of))
        assert [r["symbol"] for r in out] == ["SPY", "AAPL"], as_of
        assert out[0]["adv_usd"] == pytest.approx(200 * M), as_of  # constant volume, any window
        assert out[0]["sessions_traded"] == len(window), as_of
