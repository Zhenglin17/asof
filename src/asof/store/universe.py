"""The liquid tier: securities with real money flowing through them, as of a point in time.

The scanner cannot look at 20,000 symbols a morning; most are warrants and shells nobody trades.
A security is *liquid* at ``as_of`` when its average daily dollar volume over the last
``LOOKBACK_SESSIONS`` trading sessions exceeds ``MIN_ADV_USD`` and its instrument class is a
common stock or an ETF (no price floor: HIVE traded $50-150M a day at $3-4 in 2026). The tier
is the audit's scope (identity conflicts must be settled inside it) and the minute-bar universe.

Everything is computed from what was visible at ``as_of``:

- bars come through ``visible_bars`` and the calendar through ``visible_sessions``, so a session
  still in progress is not in the window;
- ``visible_bars`` attributes each bar to a security through the master as known at ``as_of``
  (``visible_master``): NKLA's segment starts 2020-06-04 but was published 2020-06-08, so a
  reader on 06-05 still sees VTIQ; LAC's rename to LAAC was filed a month after the cut, so a
  reader in between keeps counting LAC's bars under LAC; bars matching no knowable segment
  (BNY's copy of BK's history) count for nobody;
- the divisor is the number of sessions in the window, not the number of days the security
  traded: a listing in its first week must earn its place and a one-day SPAC spike cannot
  ($900M on one day is $45M a day over 20).

Dollar volume per bar is ``coalesce(vwap, close) * volume``; the 1.18M volume-0 placeholder bars
Alpaca writes after a rename are dropped by ``visible_bars`` and would add nothing anyway.
Symbols of one security are summed (VTIQ and NKLA across the rename window) and the tier reports
the symbol of the latest visible segment.
"""

from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from asof.ingest.instrument_class import CLASS_FILE
from asof.store.market import visible_bars, visible_master, visible_sessions

ET = ZoneInfo("America/New_York")

MIN_ADV_USD = 50e6  # average daily dollar volume; user decision 2026-10-07, no price floor
LOOKBACK_SESSIONS = 20  # trading sessions in the window
LIQUID_CLASSES = ("common", "etf")  # leveraged ETFs, warrants, units, preferreds stay out

LIQUID_SCHEMA = pa.schema(
    [
        ("as_of", pa.timestamp("us", tz="UTC")),
        ("security_id", pa.string()),
        ("symbol", pa.string()),  # the symbol in use at as_of (latest visible segment)
        ("class", pa.string()),
        ("adv_usd", pa.float64()),
        ("sessions_traded", pa.int64()),  # bars with volume > 0 inside the window
    ]
)

_SEGMENTS_VIEW = "_asof_visible_segments"
_CLASS_VIEW = "_asof_instrument_class"
_BARS_VIEW = "_asof_window_bars"


def _aware_utc(value: datetime, name: str) -> datetime:
    if value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(UTC)


def _empty() -> pa.Table:
    return LIQUID_SCHEMA.empty_table()


def month_starts(start: date, end: date) -> list[datetime]:
    """00:00 New York on the first of every month from ``start``'s month through ``end``'s,
    as UTC instants. A reader at that instant sees every bar through the previous session."""
    if end < start:
        raise ValueError("end must not be before start")
    starts: list[datetime] = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        starts.append(datetime(year, month, 1, tzinfo=ET).astimezone(UTC))
        year, month = year + (month == 12), month % 12 + 1
    return starts


def liquid_securities(
    con: duckdb.DuckDBPyConnection,
    root: Path,
    as_of: datetime,
    *,
    min_adv_usd: float = MIN_ADV_USD,
    lookback_sessions: int = LOOKBACK_SESSIONS,
    classes: Sequence[str] | None = LIQUID_CLASSES,
) -> pa.Table:
    """Securities whose average daily dollar volume over the last ``lookback_sessions`` visible
    sessions exceeds ``min_adv_usd``, ordered by ``adv_usd`` descending then ``security_id``.

    ``classes`` keeps only those instrument classes; ``None`` returns every class, which is how
    the audit finds ADV-qualified segments still tagged ``unknown``.
    """
    as_of_utc = _aware_utc(as_of, "as_of")
    if lookback_sessions <= 0:
        raise ValueError("lookback_sessions must be positive")
    class_path = root / CLASS_FILE
    if not class_path.exists():
        raise FileNotFoundError(f"{class_path}: build it with `asof market instruments`")

    sessions = visible_sessions(con, root, "1Day", as_of_utc, last=lookback_sessions)
    if not sessions:
        return _empty()
    window_start = datetime.combine(sessions[0], datetime.min.time(), tzinfo=ET)
    # Attributed to securities as the master was known at as_of; untraded bars are already gone.
    bars = visible_bars(con, root, "1Day", as_of_utc, start=window_start)
    segments = visible_master(con, root, as_of_utc)

    con.register(_SEGMENTS_VIEW, segments)
    con.register(_CLASS_VIEW, pq.read_table(class_path))
    con.register(_BARS_VIEW, bars)
    try:
        con.execute("SET TimeZone = 'UTC'")
        _check_class_rows(con)
        params: dict[str, Any] = {"sessions": len(sessions), "min_adv": float(min_adv_usd)}
        class_clause = ""
        if classes is not None:
            class_clause = 'AND c."class" IN (SELECT UNNEST($classes))'
            params["classes"] = list(classes)
        result = con.execute(
            f"""
            WITH adv AS (
                SELECT security_id,
                       sum(coalesce(vwap, close) * volume) / $sessions AS adv_usd,
                       count(DISTINCT session_date) FILTER (WHERE volume > 0) AS sessions_traded
                FROM {_BARS_VIEW} GROUP BY security_id
            ),
            latest AS (
                SELECT security_id, symbol, valid_from FROM {_SEGMENTS_VIEW}
                QUALIFY row_number() OVER (
                    PARTITION BY security_id ORDER BY valid_from DESC, symbol
                ) = 1
            )
            SELECT a.security_id, l.symbol, c."class", a.adv_usd, a.sessions_traded
            FROM adv a
            JOIN latest l USING (security_id)
            JOIN {_CLASS_VIEW} c
              ON c.security_id = l.security_id AND c.symbol = l.symbol
             AND c.valid_from = l.valid_from
            WHERE a.adv_usd > $min_adv {class_clause}
            ORDER BY a.adv_usd DESC, a.security_id
            """,
            params,
        ).to_arrow_table()
    finally:
        for name in (_SEGMENTS_VIEW, _CLASS_VIEW, _BARS_VIEW):
            con.unregister(name)

    as_of_column = pa.array([as_of_utc] * result.num_rows, LIQUID_SCHEMA.field("as_of").type)
    table = result.add_column(0, "as_of", as_of_column)
    return table.select(LIQUID_SCHEMA.names).cast(LIQUID_SCHEMA)


def _check_class_rows(con: duckdb.DuckDBPyConnection) -> None:
    """Every visible segment must have a class row: the class table is keyed by the master's
    segments and goes stale when the master is rebuilt without it. (Duplicate keys and
    overlapping segments are refused by ``visible_master`` itself.)"""
    missing = _scalar(
        con,
        f"SELECT count(*) FROM {_SEGMENTS_VIEW} m ANTI JOIN {_CLASS_VIEW} c "
        "ON m.security_id = c.security_id AND m.symbol = c.symbol AND m.valid_from = c.valid_from",
    )
    if missing:
        raise ValueError(
            f"instrument class table is stale: {missing} visible master segments have no class "
            "row; rerun `asof market instruments`"
        )


def _scalar(con: duckdb.DuckDBPyConnection, query: str) -> int:
    row = con.execute(query).fetchone()
    return int(row[0]) if row else 0


def tier_union(
    con: duckdb.DuckDBPyConnection,
    root: Path,
    start: date,
    end: date,
    *,
    min_adv_usd: float = MIN_ADV_USD,
    lookback_sessions: int = LOOKBACK_SESSIONS,
    classes: Sequence[str] | None = LIQUID_CLASSES,
) -> pa.Table:
    """The liquid tier recomputed at the start of every month from ``start`` through ``end``,
    stacked: one block of rows per ``as_of`` in ascending order. The set of ``security_id`` over
    all blocks is the "ever liquid" universe the audit and the minute-bar backfill use."""
    blocks = [
        liquid_securities(
            con,
            root,
            as_of,
            min_adv_usd=min_adv_usd,
            lookback_sessions=lookback_sessions,
            classes=classes,
        )
        for as_of in month_starts(start, end)
    ]
    blocks = [b for b in blocks if b.num_rows]
    if not blocks:
        return _empty()
    return pa.concat_tables(blocks)
