"""The audit's fixture world: a 25-session store of three clean symbols (AAA, BBB, CCC) with
its master, class table and SEC history, rewritten from scratch on every ``World.write``.

Shared by the per-check audit tests (tests/store/test_audit*.py) and the meta-test that keeps
every check breakable (tests/store/test_audit_meta.py). Fake data provenance is documented in
tests/store/test_audit.py.
"""

import json
import shutil
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from asof.ingest.bars import bucket_of
from asof.ingest.corporate_actions import (
    DIVIDEND_SCHEMA,
    NAME_CHANGE_SCHEMA,
    OTHER_SCHEMA,
    SPLIT_SCHEMA,
)
from asof.ingest.instrument_class import CLASS_FILE, CLASS_SCHEMA
from asof.ingest.sec_ticker_history import HISTORY_SCHEMA
from asof.ingest.security_master import (
    CONFLICT_SCHEMA,
    CONFLICTS_FILE,
    MASTER_FILE,
    MASTER_SCHEMA,
    SEC_HISTORY_START,
)
from asof.store.audit import AuditException, CheckResult, run_audit
from tests.ingest.fakes import FETCHED_AT, daily
from tests.store.test_market import KNOWN_EARLY, TS_UTC, avail, known
from tests.store.test_universe import store_raw, trading_days

START = date(2016, 1, 4)  # valid_from of every opening segment, known since KNOWN_EARLY
DAYS = trading_days(date(2020, 1, 2), date(2020, 2, 6))  # 25 sessions
LATE = datetime(2021, 1, 4, tzinfo=UTC)  # an as_of at which every fixture fact is knowable
SID_AAA, SID_BBB, SID_CCC = "S000001", "S000002", "S000003"
SID_LAC, SID_R, SID_X1, SID_X2, SID_D1, SID_D2 = (f"S00000{n}" for n in range(4, 10))
BASE_LIQUID = frozenset({SID_AAA, SID_BBB})
ADDED = date(2026, 10, 8)


# =============================================================================================
# builders
# =============================================================================================


def iso(day: date) -> str:
    return day.isoformat()


def t_of(day: date) -> datetime:
    return daily("X", day).t


def bar(
    symbol: str,
    day: date,
    *,
    close: float,
    volume: int,
    open: float | None = None,
    high: float | None = None,
    low: float | None = None,
    vwap: float | None = None,
    available_at: datetime | None = None,
    fetched_at: datetime = FETCHED_AT,
) -> dict[str, Any]:
    """One raw daily bar as a dict (the only way to put a wrong available_at, a null vwap or an
    impossible price on disk). OHLC derive from ``close`` unless given; all rounded so that the
    values read back from Parquet compare equal to the fixture's."""
    return {
        "symbol": symbol,
        "t": t_of(day),
        "session_date": day,
        "open": round(close * 0.995, 4) if open is None else open,
        "high": round(close * 1.01, 4) if high is None else high,
        "low": round(close * 0.99, 4) if low is None else low,
        "close": close,
        "volume": volume,
        "trade_count": 1 if volume else 0,
        "vwap": round(close * 1.002, 4) if vwap is None else vwap,
        "available_at": available_at or avail(day),
        "fetched_at": fetched_at,
    }


def series(
    symbol: str, days: Sequence[date], close: float, volume: int, *, start: int = 0
) -> list[dict[str, Any]]:
    """Flat close, volume rising by one a day so that no two days of one symbol are identical
    and no two symbols collide unless a test copies bars on purpose."""
    return [bar(symbol, d, close=close, volume=volume + i) for i, d in enumerate(days, start)]


def flat(symbol: str, day: date, close: float) -> dict[str, Any]:
    """What Alpaca writes under an old label after a rename (real: VTIQ at 33.97, volume 0)."""
    return bar(symbol, day, close=close, volume=0, open=close, high=close, low=close, vwap=close)


def relabel(rows: Sequence[dict[str, Any]], symbol: str) -> list[dict[str, Any]]:
    """The same bars filed under another name (real: BK's history under BNY)."""
    return [{**r, "symbol": symbol} for r in rows]


def replace(
    rows: Sequence[dict[str, Any]], symbol: str, day: date, new: dict[str, Any]
) -> list[dict[str, Any]]:
    return [new if (r["symbol"] == symbol and r["session_date"] == day) else r for r in rows]


def drop(rows: Sequence[dict[str, Any]], symbol: str, days: Sequence[date]) -> list[dict[str, Any]]:
    gone = set(days)
    return [r for r in rows if not (r["symbol"] == symbol and r["session_date"] in gone)]


def find(rows: Sequence[dict[str, Any]], symbol: str, day: date) -> dict[str, Any]:
    [row] = [r for r in rows if r["symbol"] == symbol and r["session_date"] == day]
    return row


@dataclass(frozen=True)
class Seg:
    """A master segment plus the class-table row that goes with it."""

    security_id: str
    symbol: str
    valid_from: date = START
    valid_to: date | None = None
    available_at: datetime = KNOWN_EARLY
    end_available_at: datetime | None = None  # must be set iff valid_to is set
    cik: int | None = None
    cls: str = "common"
    name_source: str = "alpaca_active"
    rule: str = "symbol_shape"


def class_row(seg: Seg) -> dict[str, Any]:
    return {
        "security_id": seg.security_id,
        "symbol": seg.symbol,
        "valid_from": seg.valid_from,
        "valid_to": seg.valid_to,
        "name": f"{seg.symbol} Inc",
        "name_source": seg.name_source,
        "class": seg.cls,
        "rule": seg.rule,
    }


def write_master(root: Path, segs: Sequence[Seg]) -> Path:
    """Like tests.store.test_market.write_master, with a cik column (I5 needs one)."""
    table = pa.table(
        {
            "security_id": pa.array([s.security_id for s in segs], pa.string()),
            "symbol": pa.array([s.symbol for s in segs], pa.string()),
            "valid_from": pa.array([s.valid_from for s in segs], pa.date32()),
            "valid_to": pa.array([s.valid_to for s in segs], pa.date32()),
            "cik": pa.array([s.cik for s in segs], pa.int64()),
            "cik_available_at": pa.array(
                [None if s.cik is None else s.available_at for s in segs], TS_UTC
            ),
            "cusip": pa.array([None for _ in segs], pa.string()),
            "evidence": pa.array(["bars" for _ in segs], pa.string()),
            "available_at": pa.array([s.available_at for s in segs], TS_UTC),
            "end_available_at": pa.array([s.end_available_at for s in segs], TS_UTC),
        },
        schema=MASTER_SCHEMA,
    )
    return _write(root / MASTER_FILE, table)


def write_class_rows(root: Path, rows: Sequence[dict[str, Any]]) -> Path:
    return _write(root / CLASS_FILE, pa.Table.from_pylist(list(rows), schema=CLASS_SCHEMA))


def write_sec(root: Path, rows: Sequence[tuple[date, str, int]]) -> Path:
    table = pa.Table.from_pylist(
        [
            {"snapshot_date": d, "ticker": sym, "cik": cik, "name": f"{sym} Inc"}
            for d, sym, cik in sorted(rows)
        ],
        schema=HISTORY_SCHEMA,
    )
    return _write(root / "symbols" / "sec_history.parquet", table)


def default_sec(segs: Sequence[Seg]) -> list[tuple[date, str, int]]:
    """Snapshots that agree with the master on every cik it carries."""
    return [
        (snap, s.symbol, s.cik)
        for snap in (SEC_HISTORY_START, date(2020, 1, 13), date(2020, 7, 1))
        for s in segs
        if s.cik is not None
    ]


def split(
    symbol: str, ex_date: date, old_rate: float, new_rate: float, *, kind: str = "forward"
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "ex_date": ex_date,
        "old_rate": old_rate,
        "new_rate": new_rate,
        "factor": new_rate / old_rate,
        "kind": kind,
        "old_cusip": None,
        "new_cusip": None,
        "available_at": known(ex_date),  # ingest rule: 00:00 New York of the ex_date
    }


def dividend(
    symbol: str,
    ex_date: date,
    rate: float,
    *,
    special: bool = False,
    sub_type: str | None = None,
    available_at: datetime | None = None,
) -> dict[str, Any]:
    """A dividends.parquet row as the ingest writes it (the convention of
    tests/evolve/test_benchmark.py::write_dividends, plus ``special`` / ``sub_type`` and a
    settable ``available_at`` so a record not yet knowable can be put on disk)."""
    return {
        "symbol": symbol,
        "ex_date": ex_date,
        "rate": rate,
        "special": special,
        "sub_type": sub_type,
        "available_at": available_at or known(ex_date),  # ingest rule: 00:00 New York
    }


def spin_off(parent: str, child: str, ex_date: date, ratio: float) -> dict[str, Any]:
    """A spin_off row of others.parquet as the ingest writes it (rate = new / source rate);
    the full builder lives in tests/store/test_spinoffs.py."""
    raw = {
        "ex_date": ex_date.isoformat(),
        "new_rate": ratio,
        "new_symbol": child,
        "source_rate": 1,
        "source_symbol": parent,
        "_kind": "spin_offs",
    }
    return {
        "kind": "spin_off",
        "symbol": None,
        "source_symbol": parent,
        "new_symbol": child,
        "ex_date": ex_date,
        "process_date": ex_date,
        "rate": ratio,
        "raw_json": json.dumps(raw),
        "available_at": known(ex_date),  # ingest rule: 00:00 New York of the ex_date
    }


def rename(old: str, new: str, process_date: date) -> dict[str, Any]:
    return {
        "old_symbol": old,
        "new_symbol": new,
        "process_date": process_date,
        "old_cusip": None,
        "new_cusip": None,
        "available_at": known(process_date),
    }


def conflict(
    kind: str,
    symbols: Sequence[str],
    dates: Sequence[date],
    detail: str,
    resolution: str = "unresolved",
) -> dict:
    return {
        "kind": kind,
        "symbols": list(symbols),
        "dates": list(dates),
        "detail": detail,
        "resolution": resolution,
    }


def _write(path: Path, table: pa.Table) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


def exc(check: str, key: str) -> AuditException:
    return AuditException(check=check, key=key, reason="reviewed", added=ADDED, by="greyson")


def rows(result: CheckResult) -> list[dict[str, Any]]:
    return result.rows.to_pylist()


def keys(result: CheckResult) -> list[str]:
    return result.rows.column("key").to_pylist()


def base_segs() -> list[Seg]:
    return [
        Seg(SID_AAA, "AAA", cik=1001),
        Seg(SID_BBB, "BBB", cik=1002),
        Seg(SID_CCC, "CCC", cik=1003),
    ]


def base_bars() -> list[dict[str, Any]]:
    # volumes between 1,000 (the duplicate floor) and 50,000 (S3's thin-volume ceiling)
    return (
        series("AAA", DAYS, 100.0, 30_000)
        + series("BBB", DAYS, 60.0, 20_000)
        + series("CCC", DAYS, 10.0, 5_000)
    )


@dataclass
class World:
    """Everything the audit reads, rewritten from scratch on every ``write``."""

    root: Path
    bars: list[dict[str, Any]] = field(default_factory=base_bars)
    segs: list[Seg] = field(default_factory=base_segs)
    liquid_ids: set[str] = field(default_factory=lambda: set(BASE_LIQUID))
    class_rows: list[dict[str, Any]] | None = None  # None: one row per segment
    splits: list[dict[str, Any]] = field(default_factory=list)
    name_changes: list[dict[str, Any]] = field(default_factory=list)
    others: list[dict[str, Any]] = field(default_factory=list)  # spin-offs and the rest
    dividends: list[dict[str, Any]] = field(default_factory=list)
    conflicts: list[dict[str, Any]] | None = None  # None: no file
    sec: list[tuple[date, str, int]] | None = None  # None: agrees with the master
    misfiled: list[tuple[int, dict[str, Any]]] = field(default_factory=list)  # (hive year, bar)

    def write(self) -> Path:
        shutil.rmtree(self.root, ignore_errors=True)
        grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
        for b in self.bars:
            grouped[(b["session_date"].year, bucket_of(b["symbol"]))].append(b)
        for year, b in self.misfiled:
            grouped[(year, bucket_of(b["symbol"]))].append(b)
        for (year, bucket), group in grouped.items():
            store_raw(self.root, group, year, bucket)
        write_master(self.root, self.segs)
        classes = [class_row(s) for s in self.segs] if self.class_rows is None else self.class_rows
        write_class_rows(self.root, classes)
        write_sec(self.root, default_sec(self.segs) if self.sec is None else self.sec)
        if self.splits:
            _write(
                self.root / "corporate_actions" / "splits.parquet",
                pa.Table.from_pylist(self.splits, schema=SPLIT_SCHEMA),
            )
        if self.name_changes:
            _write(
                self.root / "corporate_actions" / "name_changes.parquet",
                pa.Table.from_pylist(self.name_changes, schema=NAME_CHANGE_SCHEMA),
            )
        if self.others:
            _write(
                self.root / "corporate_actions" / "others.parquet",
                pa.Table.from_pylist(self.others, schema=OTHER_SCHEMA),
            )
        if self.dividends:
            _write(
                self.root / "corporate_actions" / "dividends.parquet",
                pa.Table.from_pylist(self.dividends, schema=DIVIDEND_SCHEMA),
            )
        if self.conflicts is not None:
            _write(
                self.root / CONFLICTS_FILE,
                pa.Table.from_pylist(self.conflicts, schema=CONFLICT_SCHEMA),
            )
        return self.root

    def audit(
        self,
        con: duckdb.DuckDBPyConnection,
        *,
        as_of: datetime = LATE,
        checks: Sequence[str] | None = None,
        exceptions: Sequence[AuditException] = (),
    ) -> dict[str, CheckResult]:
        self.write()
        results = run_audit(
            con,
            self.root,
            as_of=as_of,
            liquid_ids=self.liquid_ids,
            exceptions=exceptions,
            checks=checks,
        )
        return {r.spec.id: r for r in results}
