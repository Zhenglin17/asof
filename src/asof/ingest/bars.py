"""Backfill of Alpaca bars into partitioned Parquet files.

Layout under the market root::

    bars_1d/year=2026/bucket=A.parquet
    bars_1m/year=2026/month=03/bucket=A.parquet

Periods are New York calendar years / months, not UTC ones: a post-market minute at 19:59 ET
in winter is already the next UTC day and must not land in the next month's file. A partition
file is the unit of resume: it is written atomically and records when it was fetched; it counts
as complete only if it was fetched after its period ended, otherwise the next run redoes it.

Every row carries ``available_at``, the first instant the bar could have been known, which is
the only column ``visible_bars`` filters on.

Ticker chains: Alpaca stores a security's whole history under its newest name and, when an old
and a new name share one request, answers only for the new one. Three measures keep each bar
under the name in use at the time: active and inactive symbols are requested separately, bars
outside a symbol's SEC validity window are dropped, and a symbol that came back empty although
its window overlaps the period is requested again on its own.
"""

import os
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Protocol
from zoneinfo import ZoneInfo

import pyarrow as pa
import pyarrow.parquet as pq

from asof.ingest.alpaca import Bar, Timeframe

ET = ZoneInfo("America/New_York")
# Alpaca's 1Day bar: OHLC from the regular session, but volume and trade_count include
# pre- and post-market trades (verified 2026-10-05 against 1Min bars). The bar is only
# final once the post-market session ends.
DAILY_FINAL = time(20, 0)

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
_DIRS: dict[str, str] = {"1Day": "bars_1d", "1Min": "bars_1m"}
FETCHED_AT_KEY = b"asof.fetched_at"
# The free tier refuses SIP data from the last 15 minutes; stay clear of that edge.
RECENT_DATA_MARGIN = timedelta(minutes=16)


def _require_aware(value: datetime, name: str) -> None:
    if value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")


def session_date(t: datetime) -> date:
    """The New York calendar day a bar belongs to."""
    _require_aware(t, "t")
    return t.astimezone(ET).date()


def available_at(t: datetime, timeframe: Timeframe) -> datetime:
    """First instant the bar is complete: next minute for 1Min, 20:00 New York for 1Day."""
    _require_aware(t, "t")
    if timeframe == "1Min":
        return t.astimezone(UTC) + timedelta(minutes=1)
    return datetime.combine(session_date(t), DAILY_FINAL, tzinfo=ET).astimezone(UTC)


def bucket_of(symbol: str) -> str:
    return symbol[0]


def _period_start(year: int, month: int | None) -> datetime:
    return datetime(year, month or 1, 1, tzinfo=ET)


def _next_period_start(year: int, month: int | None) -> datetime:
    if month is None:
        return datetime(year + 1, 1, 1, tzinfo=ET)
    return datetime(year + (month == 12), month % 12 + 1, 1, tzinfo=ET)


@dataclass(frozen=True)
class Partition:
    timeframe: Timeframe
    year: int
    month: int | None
    bucket: str

    def _check(self) -> None:
        if self.timeframe == "1Min" and self.month is None:
            raise ValueError("minute partitions need a month")

    def path(self, root: Path) -> Path:
        self._check()
        directory = root / _DIRS[self.timeframe] / f"year={self.year}"
        if self.month is not None:
            directory = directory / f"month={self.month:02d}"
        return directory / f"bucket={self.bucket}.parquet"

    def bounds(self) -> tuple[datetime, datetime]:
        """``[start, end)`` in UTC of the New York calendar period."""
        self._check()
        return (
            _period_start(self.year, self.month).astimezone(UTC),
            _next_period_start(self.year, self.month).astimezone(UTC),
        )

    def contains(self, day: date) -> bool:
        return (
            _period_start(self.year, self.month).date()
            <= day
            < _next_period_start(self.year, self.month).date()
        )

    def is_open(self, today: date) -> bool:
        """Still growing: the period has not ended yet, so it must be refetched on each run."""
        return self.contains(today)


def plan_partitions(
    timeframe: Timeframe, start: date, end: date, symbols: Sequence[str]
) -> list[Partition]:
    if start > end:
        raise ValueError(f"start {start} is after end {end}")
    buckets = sorted({bucket_of(s) for s in symbols})
    if not buckets:
        return []
    periods: list[tuple[int, int | None]] = []
    if timeframe == "1Day":
        periods = [(year, None) for year in range(start.year, end.year + 1)]
    else:
        year, month = start.year, start.month
        while (year, month) <= (end.year, end.month):
            periods.append((year, month))
            year, month = year + (month == 12), month % 12 + 1
    return [Partition(timeframe, y, m, b) for y, m in periods for b in buckets]


Window = tuple[date, date | None]


def in_window(windows: Mapping[str, Window] | None, symbol: str, day: date) -> bool:
    """True unless ``symbol`` has a window and ``day`` falls outside it."""
    if not windows or symbol not in windows:
        return True
    start, end = windows[symbol]
    return start <= day and (end is None or day < end)


def _overlaps(windows: Mapping[str, Window] | None, symbol: str, partition: "Partition") -> bool:
    if not windows or symbol not in windows:
        return True
    start, end = windows[symbol]
    p_start = _period_start(partition.year, partition.month).date()
    p_end = _next_period_start(partition.year, partition.month).date()
    return start < p_end and (end is None or end > p_start)


def _to_table(
    bars: Iterable[Bar],
    timeframe: Timeframe,
    fetched_at: datetime,
    windows: Mapping[str, Window] | None,
) -> pa.Table:
    fetched = fetched_at.astimezone(UTC)
    latest: dict[tuple[str, datetime], Bar] = {}
    for bar in bars:
        # A bar that was not final when we fetched it (today's daily bar during the session)
        # would be stored as if complete; leave it for a later run. A bar outside the symbol's
        # SEC window belongs to another name of the same security.
        if available_at(bar.t, timeframe) <= fetched and in_window(
            windows, bar.symbol, session_date(bar.t)
        ):
            latest[(bar.symbol, bar.t)] = bar  # a duplicate (symbol, t) keeps the last one seen
    ordered = [latest[key] for key in sorted(latest)]
    columns: dict[str, list[object]] = {
        "symbol": [b.symbol for b in ordered],
        "t": [b.t for b in ordered],
        "session_date": [session_date(b.t) for b in ordered],
        "open": [b.open for b in ordered],
        "high": [b.high for b in ordered],
        "low": [b.low for b in ordered],
        "close": [b.close for b in ordered],
        "volume": [b.volume for b in ordered],
        "trade_count": [b.trade_count for b in ordered],
        "vwap": [b.vwap for b in ordered],
        "available_at": [available_at(b.t, timeframe) for b in ordered],
        "fetched_at": [fetched] * len(ordered),
    }
    return pa.table(
        {name: pa.array(values, BAR_SCHEMA.field(name).type) for name, values in columns.items()},
        schema=BAR_SCHEMA,
    )


def write_partition(
    path: Path,
    bars: Iterable[Bar],
    timeframe: Timeframe,
    fetched_at: datetime,
    *,
    windows: Mapping[str, Window] | None = None,
) -> int:
    """Write one partition file atomically; returns the row count. Nothing is left on failure.

    Bars not yet final at ``fetched_at`` are dropped; ``fetched_at`` is also stored as file
    metadata so a later run can tell whether the file was written after its period ended.
    """
    _require_aware(fetched_at, "fetched_at")
    table = _to_table(bars, timeframe, fetched_at, windows)  # consume fully before writing
    table = table.replace_schema_metadata({FETCHED_AT_KEY: fetched_at.astimezone(UTC).isoformat()})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        pq.write_table(table, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return table.num_rows


def partition_fetched_at(path: Path) -> datetime | None:
    """When the file was fetched, from its metadata (or the column, for older files)."""
    metadata = pq.read_metadata(path).metadata or {}
    if FETCHED_AT_KEY in metadata:
        return datetime.fromisoformat(metadata[FETCHED_AT_KEY].decode())
    values = [
        v for v in pq.read_table(path, columns=["fetched_at"]).column("fetched_at").to_pylist() if v
    ]
    return max(values) if values else None


def is_complete(partition: Partition, path: Path) -> bool:
    """A file is complete only if it was fetched after the period it covers had ended."""
    if not path.exists():
        return False
    fetched_at = partition_fetched_at(path)
    return fetched_at is not None and fetched_at >= partition.bounds()[1]


class BarSource(Protocol):
    def iter_bars(
        self, symbols: Sequence[str], timeframe: Timeframe, start: datetime, end: datetime
    ) -> Iterator[Bar]: ...


@dataclass
class BackfillReport:
    written: list[Partition] = field(default_factory=list)
    skipped: list[Partition] = field(default_factory=list)
    rows: int = 0


def _fetch_partition(
    client: BarSource,
    partition: Partition,
    symbols: Sequence[str],
    batch_size: int,
    fetched_at: datetime,
    windows: Mapping[str, Window] | None,
    active: Collection[str] | None,
) -> Iterator[Bar]:
    start, end = partition.bounds()
    end = min(end, fetched_at - RECENT_DATA_MARGIN)  # open period: do not ask for the future
    if end <= start:
        return
    wanted = [s for s in symbols if _overlaps(windows, s, partition)]
    # A renamed security has one current name: keeping active and inactive names in separate
    # requests keeps old and new names of one chain apart.
    groups = (
        [[s for s in wanted if s in active], [s for s in wanted if s not in active]]
        if active is not None
        else [wanted]
    )
    seen: set[str] = set()
    for group in groups:
        for i in range(0, len(group), batch_size):
            for bar in client.iter_bars(group[i : i + batch_size], partition.timeframe, start, end):
                seen.add(bar.symbol)
                # Defensive: never file a bar under a period its session does not belong to.
                if partition.contains(session_date(bar.t)):
                    yield bar
    # Second pass: a symbol with an SEC window over this period that came back empty may have
    # been folded into a newer name sharing its request. Alone, Alpaca answers for it.
    for symbol in wanted:
        if symbol in seen or not (windows and symbol in windows):
            continue
        for bar in client.iter_bars([symbol], partition.timeframe, start, end):
            if partition.contains(session_date(bar.t)):
                yield bar


def backfill(
    client: BarSource,
    root: Path,
    timeframe: Timeframe,
    start: date,
    end: date,
    symbols: Sequence[str],
    *,
    force: bool = False,
    batch_size: int = 100,
    now: Callable[[], datetime] | None = None,
    on_partition: Callable[[Partition, str, int], None] | None = None,
    windows: Mapping[str, Window] | None = None,
    active: Collection[str] | None = None,
) -> BackfillReport:
    """Fetch and write every partition in range, skipping complete ones unless ``force``.

    ``now`` supplies ``fetched_at`` (taken once per partition); it decides which bars are final
    and how far the request may reach. ``windows`` (SEC validity per symbol) and ``active``
    (Alpaca's currently listed symbols) drive the ticker-chain handling described above.
    """
    clock = now or (lambda: datetime.now(UTC))
    _require_aware(clock(), "now()")
    plan = plan_partitions(timeframe, start, end, symbols)
    by_bucket: dict[str, list[str]] = {}
    for symbol in symbols:
        by_bucket.setdefault(bucket_of(symbol), []).append(symbol)

    report = BackfillReport()
    for partition in plan:
        path = partition.path(root)
        if not force and is_complete(partition, path):
            report.skipped.append(partition)
            if on_partition:
                on_partition(partition, "skipped", 0)
            continue
        fetched_at = clock()
        bars = _fetch_partition(
            client, partition, by_bucket[partition.bucket], batch_size, fetched_at, windows, active
        )
        rows = write_partition(path, bars, timeframe, fetched_at, windows=windows)
        report.written.append(partition)
        report.rows += rows
        if on_partition:
            on_partition(partition, "written", rows)
    return report
