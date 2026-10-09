"""Guards against the store audit missing a defect it is meant to catch, blaming the wrong
row, or seeing data that was not yet knowable at ``as_of``: every check in the table has one
fixture that trips it (with the exact key and columns the report promises), one that is clean,
and -- for gated checks -- one that is tripped but excepted; the exception file is validated
entry by entry; and a master segment whose ``available_at`` lies after ``as_of`` must leave the
whole audit unchanged.

Target API (asof.store.audit; implementation pending, design approved 2026-10-08, see the
interface spec): ``CHECKS``, ``CheckSpec``, ``CheckResult``, ``AuditException``,
``load_exceptions``, ``run_audit``, ``stale_exceptions``, ``write_report``.

Fake data provenance ("real" = read off the store / the security master on 2026-10-08):
- A daily bar is visible from 20:00 New York of its session (documented in ingest.bars; real:
  the 2020-01-02 bar has available_at 2020-01-03 01:00 UTC). The fixtures take the instant from
  ``tests.store.test_market.avail``.
- The session calendar is derived from the bars themselves; the fake one is weekdays minus the
  real NYSE holidays (2020-01-20 is MLK day, so 2020-01-17 -> 2020-01-21 is one session step).
- Segments created by a rename, and the end of the one they replace, become knowable at 00:00
  New York of the effective day (real: BNY / BK, 2026-05-21 04:00 UTC); opening segments are
  known since the store began (``KNOWN_EARLY``, 2016-01-04). Security ids look like S000730.
- Splits: ``available_at`` is 00:00 New York of the ex_date (ingest.corporate_actions);
  ``factor`` = new shares per old share (AAPL 4:1 on 2020-08-31 -> factor 4, price divided by
  4). The spec's I9 wording "divide first_close by the product of factors" would move the AAPL
  ratio the wrong way under this convention; the test asserts the behaviour (a split exactly at
  the cut must not be reported), not the formula.
- Volume-0 bars are flat placeholders Alpaca keeps writing under a dead label (real: VTIQ at
  33.97, 1.18M store-wide). "traded" = volume > 0 (user decision 2026-10-08).
- "LAC -> LAAC" shape (I4): the on-disk cut precedes the day it became knowable while the old
  label kept trading. Real analogue: BBUC cut 2026-03-31, end knowable 2026-04-21, 14 traded bars
  in between (54 bars / 12 symbols store-wide).
- Relabelled history (I1, N3): Alpaca files a security's past under its newest name, so the raw
  store holds BK's bars under BNY as well, identical on every day (real).
- ARNC-type instant mismatch (M1): before the "earliest instant of the day" rule 5 of 723
  back-to-back same-symbol pairs had old.end_available_at != new.available_at (real, fixed).
- SEC history: snapshots before SEC_HISTORY_START (2019-10-02) are one stale file and must be
  ignored (real); a CIK listed under a different number for one snapshot interval is the ARNC
  2020-04-01 .. 07-10 shape (real).
- ASSUMPTION (I7): a session on which only volume-0 placeholder bars exist still appears in the
  derived calendar with 0 traded symbols. Real-data check: the audit's I7 notes
  ``min_traded_symbols`` on the real store (expected well above 0 on every session).
- ASSUMPTION (I5): the real master never carries a cik the SEC history contradicts outside the
  ARNC-type cases; the audit's I5 row count on the real store is the check.
- ASSUMPTION (S3): the EXEEW 2026-02-03 bar (one-day 3x spike on thin volume, no split) is
  reported by S3 on the real store and nothing else is (standard case).
- ASSUMPTION (M4): rows for class rows whose master segment is not yet knowable at a historical
  ``as_of`` are not asserted either way; the CLI audits at "now", where every segment is visible.
- The exception file's wording ("reason", "added", "by") and the YAML list shape are the
  design's, not observed anywhere.

Real-data checks for the assumptions (audit command, then the standard cases):
- ``asof market audit`` on the real store: I4 reports BBUC with 14 bars; M1 is empty; S3 lists
  EXEEW 2026-02-03; I7's min_traded_symbols > 0; I5 rows all of the ARNC shape; N3 pairs with a
  ratio inside [0.1, 0.9] are read by a human before any exception is added.
"""

import re
import shutil
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from asof.ingest.bars import bucket_of
from asof.ingest.corporate_actions import NAME_CHANGE_SCHEMA, SPLIT_SCHEMA
from asof.ingest.instrument_class import CLASS_FILE, CLASS_SCHEMA
from asof.ingest.sec_ticker_history import HISTORY_SCHEMA
from asof.ingest.security_master import (
    CONFLICT_SCHEMA,
    CONFLICTS_FILE,
    MASTER_FILE,
    MASTER_SCHEMA,
    SEC_HISTORY_START,
)
from asof.store.audit import (
    CHECKS,
    AuditException,
    CheckResult,
    CheckSpec,
    load_exceptions,
    run_audit,
    stale_exceptions,
    write_report,
)
from asof.store.market import market_root
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

# gate per check, in report order (the spec's table)
GATES: dict[str, str | None] = {
    "P1": "all",
    "P2": "all",
    "P3": "all",
    "P4": "all",
    "P5": "all",
    "P6": "all",
    "M1": "all",
    "M2": "liquid",  # an identity conflict like M6: the long tail is reported, not gated
    "M3": None,
    "M4": "all",
    "M5": "liquid",
    "M6": "liquid",
    "I1": None,
    "I2": "liquid",
    "I3": "liquid",
    "I4": None,
    "I5": None,
    "I6": None,
    "I7": "all",
    "I8": None,
    "I9": None,
    "S1": "liquid",
    "S2": "liquid",
    "S3": "liquid",
    "S4": None,
    "N1": None,
    "N2": None,
    "N3": None,
    "C1": None,
}
HEAD = ["key", "in_liquid", "excepted"]


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


def rename(old: str, new: str, process_date: date) -> dict[str, Any]:
    return {
        "old_symbol": old,
        "new_symbol": new,
        "process_date": process_date,
        "old_cusip": None,
        "new_cusip": None,
        "available_at": known(process_date),
    }


def conflict(kind: str, symbols: Sequence[str], dates: Sequence[date], detail: str) -> dict:
    return {"kind": kind, "symbols": list(symbols), "dates": list(dates), "detail": detail}


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


@pytest.fixture
def con() -> Iterator[duckdb.DuckDBPyConnection]:
    connection = duckdb.connect()
    yield connection
    connection.close()


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return market_root(tmp_path)


@pytest.fixture
def world(root: Path) -> World:
    return World(root)


# =============================================================================================
# fixture sanity: the facts the arithmetic below relies on
# =============================================================================================


def test_fixture_calendar_and_instants() -> None:
    assert len(DAYS) == 25
    assert DAYS[3] == date(2020, 1, 7) and DAYS[10] == date(2020, 1, 16)
    assert DAYS[11] == date(2020, 1, 17) and DAYS[12] == date(2020, 1, 21)  # MLK day skipped
    assert avail(DAYS[0]) == datetime(2020, 1, 3, 1, tzinfo=UTC)  # real
    assert known(DAYS[10]) == datetime(2020, 1, 16, 5, tzinfo=UTC)
    assert LATE > avail(DAYS[-1]) and LATE > known(date(2020, 9, 1))
    aaa = series("AAA", DAYS[:2], 100.0, 30_000)
    assert [(r["open"], r["high"], r["low"], r["vwap"]) for r in aaa] == [
        (99.5, 101.0, 99.0, 100.2)
    ] * 2
    assert [r["volume"] for r in aaa] == [30_000, 30_001]


# =============================================================================================
# CHECKS / CheckSpec
# =============================================================================================


def test_checks_are_the_spec_table_in_report_order() -> None:
    assert list(CHECKS) == list(GATES)
    for check_id, spec in CHECKS.items():
        assert isinstance(spec, CheckSpec)
        assert spec.id == check_id
        assert spec.gate == GATES[check_id], check_id
        assert spec.title.strip()
        re.compile(spec.key_pattern)  # must compile
    assert len({s.id for s in CHECKS.values()}) == len(CHECKS)


SAMPLE_KEYS = {
    "P1": "AAA:2020-01-07",
    "P2": "AAA:2020-01-07",
    "P3": "AAA:2020-01-07",
    "P4": "AAA:2020-01-07",
    "P5": "AAA:2020-01-07",
    "P6": "AAA:2020-01-07",
    "M1": "XXX:2020-01-16",
    "M2": "S000005:2020-01-16",
    "M3": "S000005:2020-01-21",
    "M4": "BBB:2016-01-04",
    "M5": "BBB:2016-01-04",
    "M6": "AAA/QQQ:dup_unlinked:2020-01-02,2020-01-03",
    "I1": "ORPH",
    "I2": "AAA/BBB",
    "I3": "S000001",
    "I4": "LAC",
    "I5": "BBB:2016-01-04",
    "I6": "2020",
    "I7": "2020-01-31",
    "I8": "S000003:2020-01-24",
    "I9": "S000005:2020-01-16",
    "S1": "BRK.B:2020-01-11",
    "S2": "AAA:2020-01-16",
    "S3": "AAA:2020-01-16",
    "S4": "AAA:2020-01-16",
    "N1": "CCC/CCC3:2020-03-08",
    "N2": "ROLD/RNEW:2020-01-16",
    "N3": "AAA/BBB",
    "C1": "2020",
}


@pytest.mark.parametrize("check_id", list(GATES))
def test_key_pattern_accepts_the_documented_key_shape(check_id: str) -> None:
    pattern = re.compile(CHECKS[check_id].key_pattern)
    assert pattern.fullmatch(SAMPLE_KEYS[check_id]), (check_id, pattern.pattern)


@pytest.mark.parametrize(
    ("check_id", "bad"),
    [
        ("P1", "aaa:2020-01-07"),  # lowercase symbol
        ("P1", "AAA"),  # date missing
        ("P1", "AAA/BBB"),
        ("I2", "AAA:2020-01-07"),
        ("C1", "2020-01-07"),
        ("I7", "2020"),
        ("I1", "AAA:2020-01-07"),
        ("N1", "CCC:2020-03-08"),
    ],
)
def test_key_pattern_rejects_other_shapes(check_id: str, bad: str) -> None:
    assert not re.fullmatch(CHECKS[check_id].key_pattern, bad)


# =============================================================================================
# CheckResult counting
# =============================================================================================


def result_with(spec: CheckSpec, flags: Sequence[tuple[bool, bool]]) -> CheckResult:
    table = pa.table(
        {
            "key": pa.array([f"K{i}" for i in range(len(flags))], pa.string()),
            "in_liquid": pa.array([f[0] for f in flags], pa.bool_()),
            "excepted": pa.array([f[1] for f in flags], pa.bool_()),
        }
    )
    return CheckResult(spec=spec, rows=table, notes={})


# (in_liquid, excepted) per row: 2 liquid live, 1 non-liquid live, 2 excepted
MIXED = [(True, False), (True, True), (False, False), (False, True), (True, False)]


@pytest.mark.parametrize(
    ("check_id", "expect_liquid_tier", "expect_all_tier"),
    [("P1", True, True), ("I2", True, True), ("I1", False, False)],
    ids=["gate_all", "gate_liquid", "gate_none"],
)
def test_counts_and_fails_per_gate(
    check_id: str, expect_liquid_tier: bool, expect_all_tier: bool
) -> None:
    result = result_with(CHECKS[check_id], MIXED)
    assert (result.liquid, result.total, result.excepted) == (2, 3, 2)
    assert result.fails("liquid") is expect_liquid_tier
    assert result.fails("all") is expect_all_tier


def test_liquid_gate_only_fails_the_liquid_tier_on_liquid_rows() -> None:
    only_non_liquid = result_with(CHECKS["I2"], [(False, False), (False, False)])
    assert (only_non_liquid.liquid, only_non_liquid.total) == (0, 2)
    assert only_non_liquid.fails("liquid") is False
    assert only_non_liquid.fails("all") is True
    # the "all" gate does not care about the tier
    all_gate = result_with(CHECKS["P1"], [(False, False)])
    assert all_gate.fails("liquid") is True and all_gate.fails("all") is True


def test_excepted_rows_count_for_nothing_but_excepted() -> None:
    for check_id in ("P1", "I2", "I1"):
        result = result_with(CHECKS[check_id], [(True, True), (False, True)])
        assert (result.liquid, result.total, result.excepted) == (0, 0, 2)
        assert result.fails("liquid") is False and result.fails("all") is False
    empty = result_with(CHECKS["P1"], [])
    assert (empty.liquid, empty.total, empty.excepted) == (0, 0, 0)
    assert empty.fails("all") is False


# =============================================================================================
# load_exceptions
# =============================================================================================

GOOD_YAML = """\
- check: P1
  key: "AAA:2020-01-07"
  reason: duplicate from the 2026-10 refetch, harmless
  added: 2026-10-08
  by: greyson
- check: I2
  key: AAA/BBB
  reason: two share classes with one print feed
  added: "2026-10-01"
  by: greyson
"""


def test_load_exceptions_missing_file_and_empty_list(tmp_path: Path) -> None:
    assert load_exceptions(tmp_path / "absent.yaml") == []
    empty = tmp_path / "empty.yaml"
    empty.write_text("[]\n")
    assert load_exceptions(empty) == []


def test_load_exceptions_reads_every_field(tmp_path: Path) -> None:
    path = tmp_path / "audit_exceptions.yaml"
    path.write_text(GOOD_YAML)

    loaded = load_exceptions(path)

    assert loaded == [
        AuditException(
            check="P1",
            key="AAA:2020-01-07",
            reason="duplicate from the 2026-10 refetch, harmless",
            added=date(2026, 10, 8),
            by="greyson",
        ),
        AuditException(
            check="I2",
            key="AAA/BBB",
            reason="two share classes with one print feed",
            added=date(2026, 10, 1),
            by="greyson",
        ),
    ]
    assert all(isinstance(e.added, date) for e in loaded)


def entry(**overrides: Any) -> str:
    fields = {
        "check": "P1",
        "key": '"AAA:2020-01-07"',
        "reason": "reviewed",
        "added": "2026-10-08",
        "by": "greyson",
    }
    fields.update(overrides)
    lines = [f"  {k}: {v}" for k, v in fields.items() if v is not None]
    lines[0] = "-" + lines[0][1:]
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("check: P1\n", r"list"),
        ("- P1\n", r"P1"),
        (entry(by=None), r"AAA:2020-01-07"),  # a key missing
        (entry(note="x"), r"AAA:2020-01-07"),  # an unknown key
        (entry(reason='""'), r"AAA:2020-01-07"),  # empty value
        (entry(check="Z9"), r"Z9"),  # unknown check
        (entry(key='"aaa:2020-01-07"'), r"aaa:2020-01-07"),  # lowercase does not match SYM
        (entry(check="I2", key='"AAA:2020-01-07"'), r"AAA:2020-01-07"),  # wrong shape for I2
        (entry(added="yesterday"), r"yesterday"),
        (entry(added='"2026/10/08"'), r"2026/10/08"),
    ],
    ids=[
        "not_a_list",
        "item_not_a_mapping",
        "missing_field",
        "extra_field",
        "empty_field",
        "unknown_check",
        "key_not_matching",
        "key_wrong_shape",
        "added_not_a_date",
        "added_not_iso",
    ],
)
def test_load_exceptions_rejects_bad_entries(tmp_path: Path, text: str, match: str) -> None:
    path = tmp_path / "audit_exceptions.yaml"
    path.write_text(text)
    with pytest.raises(ValueError, match=match):
        load_exceptions(path)


# =============================================================================================
# run_audit: shape, ordering, arguments, errors
# =============================================================================================


def test_clean_world_trips_nothing_and_every_result_has_the_report_shape(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    results = run_audit(con, world.write(), as_of=LATE, liquid_ids=world.liquid_ids)

    assert [r.spec.id for r in results] == list(CHECKS)
    for result in results:
        assert result.spec is CHECKS[result.spec.id]
        assert result.rows.schema.names[:3] == HEAD
        assert result.rows.schema.field("key").type == pa.string()
        assert result.rows.schema.field("in_liquid").type == pa.bool_()
        assert result.rows.schema.field("excepted").type == pa.bool_()
        assert keys(result) == sorted(keys(result))
        assert result.fails("liquid") is False and result.fails("all") is False, result.spec.id
    with_rows = {r.spec.id for r in results if r.rows.num_rows}
    assert with_rows == {"I6", "C1"}  # the per-year information rows


def test_checks_subset_keeps_report_order_and_rejects_unknown_ids(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    root = world.write()
    subset = run_audit(con, root, as_of=LATE, liquid_ids=set(), checks=["C1", "P2", "I6"])
    assert [r.spec.id for r in subset] == ["P2", "I6", "C1"]
    assert [r.spec.id for r in run_audit(con, root, as_of=LATE, liquid_ids=set(), checks=[])] == []
    with pytest.raises(ValueError, match="Z9"):
        run_audit(con, root, as_of=LATE, liquid_ids=set(), checks=["P1", "Z9"])


def test_naive_as_of_is_rejected(con: duckdb.DuckDBPyConnection, world: World) -> None:
    with pytest.raises(ValueError):
        run_audit(con, world.write(), as_of=datetime(2021, 1, 4), liquid_ids=set())


def test_missing_master_or_class_table_is_an_error(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    root = world.write()
    (root / CLASS_FILE).unlink()
    with pytest.raises(FileNotFoundError):
        run_audit(con, root, as_of=LATE, liquid_ids=set())
    write_class_rows(root, [class_row(s) for s in world.segs])
    (root / MASTER_FILE).unlink()
    with pytest.raises(FileNotFoundError):
        run_audit(con, root, as_of=LATE, liquid_ids=set())


def test_store_without_bars_or_corporate_actions_audits_to_empty_tables(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    world.bars = []
    results = world.audit(con)
    assert all(r.rows.num_rows == 0 for r in results.values())
    assert all(r.rows.schema.names[:3] == HEAD for r in results.values())


def test_exception_marks_only_the_matching_check_and_key(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    dup = find(world.bars, "AAA", DAYS[3])
    world.bars.append(dict(dup))
    key = f"AAA:{iso(DAYS[3])}"

    results = world.audit(con, exceptions=[exc("P2", key), exc("P1", "BBB:2020-01-07")])
    assert rows(results["P1"])[0]["excepted"] is False  # other check / other key: no effect
    results = world.audit(con, exceptions=[exc("P1", key)])
    [row] = rows(results["P1"])
    assert (row["key"], row["excepted"]) == (key, True)  # excepted rows stay in the table
    assert (results["P1"].liquid, results["P1"].total, results["P1"].excepted) == (0, 0, 1)
    assert results["P1"].fails("all") is False


# =============================================================================================
# stale_exceptions / write_report
# =============================================================================================


def test_stale_exceptions_are_those_matching_no_row_among_the_checks_that_ran(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    world.bars.append(dict(find(world.bars, "AAA", DAYS[3])))
    hit = exc("P1", f"AAA:{iso(DAYS[3])}")
    miss = exc("P1", "AAA:2020-01-08")
    other = exc("I2", "AAA/BBB")  # I2 ran and has no such row
    skipped = exc("S1", "AAA:2020-01-11")  # S1 did not run
    exceptions = [hit, miss, other, skipped]

    results = world.audit(con, checks=["P1", "I2"], exceptions=exceptions)

    assert stale_exceptions(list(results.values()), exceptions) == [miss, other]
    assert stale_exceptions([], exceptions) == []


def test_write_report_writes_one_parquet_per_result_including_empty_ones(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    world.bars.append(dict(find(world.bars, "AAA", DAYS[3])))
    root = world.write()
    results = run_audit(con, root, as_of=LATE, liquid_ids=world.liquid_ids)
    report = root / "audit"
    report.mkdir(parents=True)
    (report / "P1.parquet").write_bytes(b"stale junk from an earlier run")

    out = write_report(results, root)

    assert out == report
    assert sorted(p.name for p in report.iterdir()) == sorted(f"{c}.parquet" for c in CHECKS)
    p1 = pq.read_table(report / "P1.parquet")
    assert p1.num_rows == 1 and p1.column("key").to_pylist() == [f"AAA:{iso(DAYS[3])}"]
    for check_id, result in zip(CHECKS, results, strict=True):
        table = pq.read_table(report / f"{check_id}.parquet")
        assert table.schema.names == result.rows.schema.names
        assert table.num_rows == result.rows.num_rows
    assert not list(report.glob("*.tmp"))


def test_write_report_of_a_subset_only_touches_those_files(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    root = world.write()
    subset = run_audit(con, root, as_of=LATE, liquid_ids=set(), checks=["I6"])
    out = write_report(subset, root)
    assert [p.name for p in out.iterdir()] == ["I6.parquet"]
    assert pq.read_table(out / "I6.parquet").num_rows == 1


# =============================================================================================
# in_liquid rules
# =============================================================================================


def test_symbol_key_is_in_liquid_if_any_segment_of_the_symbol_is(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    # XOLD belonged to a non-liquid security, then (same label, retaken) to a liquid one; a P2
    # row dated inside the first piece is still in the liquid scope
    cut = DAYS[5]
    world.segs += [
        Seg(SID_X1, "XOLD", START, cut, KNOWN_EARLY, known(cut)),
        Seg(SID_X2, "XOLD", cut, None, known(cut)),
    ]
    world.bars += series("XOLD", DAYS, 5.0, 2_000)
    world.bars = replace(
        world.bars, "XOLD", DAYS[2], bar("XOLD", DAYS[2], close=5.0, volume=2_002, low=0.0)
    )
    world.liquid_ids = {SID_X2}

    [row] = rows(world.audit(con, checks=["P2"])["P2"])
    assert (row["key"], row["in_liquid"]) == (f"XOLD:{iso(DAYS[2])}", True)

    world.liquid_ids = {SID_X1}
    assert rows(world.audit(con, checks=["P2"])["P2"])[0]["in_liquid"] is True
    world.liquid_ids = {SID_AAA}
    assert rows(world.audit(con, checks=["P2"])["P2"])[0]["in_liquid"] is False


def test_security_key_is_in_liquid_iff_that_id_is(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    world.bars = drop(world.bars, "CCC", DAYS[4:16])  # 12 sessions missing: an I8 row
    world.liquid_ids = {SID_CCC}
    [row] = rows(world.audit(con, checks=["I8"])["I8"])
    assert (row["key"], row["in_liquid"]) == (f"{SID_CCC}:{iso(DAYS[16])}", True)
    world.liquid_ids = {SID_AAA, SID_BBB}
    assert rows(world.audit(con, checks=["I8"])["I8"])[0]["in_liquid"] is False


def test_date_and_year_keys_are_never_in_liquid(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    dead = DAYS[20]  # an I7 session: only placeholders
    for symbol, close in (("AAA", 100.0), ("BBB", 60.0), ("CCC", 10.0)):
        world.bars = replace(world.bars, symbol, dead, flat(symbol, dead, close))
    world.liquid_ids = {SID_AAA, SID_BBB, SID_CCC}

    results = world.audit(con, checks=["I6", "I7", "C1"])

    assert [r["in_liquid"] for r in rows(results["I6"])] == [False]
    assert [r["in_liquid"] for r in rows(results["I7"])] == [False]
    assert [r["in_liquid"] for r in rows(results["C1"])] == [False]


def test_empty_liquid_ids_puts_every_row_outside_the_liquid_scope(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    world.bars.append(dict(find(world.bars, "AAA", DAYS[3])))
    world.liquid_ids = set()
    result = world.audit(con, checks=["P1"])["P1"]
    assert rows(result)[0]["in_liquid"] is False
    assert (result.liquid, result.total) == (0, 1)
    assert result.fails("liquid") is True  # gate "all": the tier does not matter


# =============================================================================================
# P: the raw bar files
# =============================================================================================


def test_p1_duplicate_symbol_and_t(con: duckdb.DuckDBPyConnection, world: World) -> None:
    dup = find(world.bars, "AAA", DAYS[3])
    world.bars += [dict(dup), dict(dup)]  # three copies in the file
    key = f"AAA:{iso(DAYS[3])}"

    result = world.audit(con, checks=["P1"])["P1"]

    assert result.rows.schema.names == [*HEAD, "symbol", "t", "copies"]
    assert rows(result) == [
        {
            "key": key,
            "in_liquid": True,
            "excepted": False,
            "symbol": "AAA",
            "t": t_of(DAYS[3]),
            "copies": 3,
        }
    ]
    assert result.fails("all") is True and result.fails("liquid") is True

    excepted = world.audit(con, checks=["P1"], exceptions=[exc("P1", key)])["P1"]
    assert rows(excepted)[0]["excepted"] is True
    assert (excepted.liquid, excepted.total, excepted.excepted) == (0, 0, 1)
    assert excepted.fails("all") is False


def test_p2_impossible_prices_name_the_first_matching_rule(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    cases: dict[str, dict[str, Any]] = {
        "nonpositive": {"low": 0.0},
        "high_below_low": {"high": 99.0, "low": 101.0},
        "low_above_body": {"low": 100.5, "high": 101.0},
        "high_below_body": {"high": 99.5, "low": 99.0},
    }
    reasons: dict[str, str] = {}
    for name, fields in cases.items():
        bad = bar("AAA", DAYS[3], close=100.0, open=100.0, volume=30_003, **fields)
        world.bars = replace(base_bars(), "AAA", DAYS[3], bad)
        result = world.audit(con, checks=["P2"])["P2"]
        assert result.rows.schema.names == [
            *HEAD,
            "symbol",
            "session_date",
            "open",
            "high",
            "low",
            "close",
            "reason",
        ]
        [row] = rows(result)
        assert row["key"] == f"AAA:{iso(DAYS[3])}" and row["in_liquid"] is True
        assert (row["open"], row["high"], row["low"], row["close"]) == (
            bad["open"],
            bad["high"],
            bad["low"],
            bad["close"],
        )
        assert re.fullmatch(r"[a-z_]+", row["reason"]), row["reason"]
        reasons[name] = row["reason"]
    assert len(set(reasons.values())) == 4
    # a bar breaking several rules is reported once, under the first rule in the spec's order
    multi = bar("AAA", DAYS[3], close=100.0, open=-1.0, high=99.0, low=101.0, volume=30_003)
    world.bars = replace(base_bars(), "AAA", DAYS[3], multi)
    [row] = rows(world.audit(con, checks=["P2"])["P2"])
    assert row["reason"] == reasons["nonpositive"]


def test_p2_clean_and_excepted(con: duckdb.DuckDBPyConnection, world: World) -> None:
    assert rows(world.audit(con, checks=["P2"])["P2"]) == []
    world.bars = replace(
        world.bars, "CCC", DAYS[3], bar("CCC", DAYS[3], close=10.0, volume=5_003, low=0.0)
    )
    key = f"CCC:{iso(DAYS[3])}"
    result = world.audit(con, checks=["P2"], exceptions=[exc("P2", key)])["P2"]
    assert [(r["key"], r["in_liquid"], r["excepted"]) for r in rows(result)] == [(key, False, True)]
    assert result.fails("all") is False


def test_p3_available_after_fetched(con: duckdb.DuckDBPyConnection, world: World) -> None:
    early_fetch = avail(DAYS[3]) - timedelta(hours=1)
    world.bars = replace(
        world.bars,
        "AAA",
        DAYS[3],
        bar("AAA", DAYS[3], close=100.0, volume=30_003, fetched_at=early_fetch),
    )
    key = f"AAA:{iso(DAYS[3])}"

    result = world.audit(con, checks=["P3"])["P3"]

    assert result.rows.schema.names == [
        *HEAD,
        "symbol",
        "session_date",
        "available_at",
        "fetched_at",
    ]
    assert rows(result) == [
        {
            "key": key,
            "in_liquid": True,
            "excepted": False,
            "symbol": "AAA",
            "session_date": DAYS[3],
            "available_at": avail(DAYS[3]),
            "fetched_at": early_fetch,
        }
    ]
    # fetched exactly at availability is fine (not strictly after)
    world.bars = replace(
        world.bars,
        "AAA",
        DAYS[3],
        bar("AAA", DAYS[3], close=100.0, volume=30_003, fetched_at=avail(DAYS[3])),
    )
    assert rows(world.audit(con, checks=["P3"])["P3"]) == []
    world.bars = replace(
        world.bars,
        "AAA",
        DAYS[3],
        bar("AAA", DAYS[3], close=100.0, volume=30_003, fetched_at=early_fetch),
    )
    excepted = world.audit(con, checks=["P3"], exceptions=[exc("P3", key)])["P3"]
    assert excepted.total == 0 and excepted.excepted == 1


def test_p4_bar_filed_under_the_wrong_hive_year(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    misfiled = find(world.bars, "AAA", DAYS[3])
    world.bars = drop(world.bars, "AAA", [DAYS[3]])
    world.misfiled = [(2019, misfiled)]
    key = f"AAA:{iso(DAYS[3])}"

    result = world.audit(con, checks=["P4"])["P4"]

    assert result.rows.schema.names == [*HEAD, "symbol", "session_date", "year"]
    assert rows(result) == [
        {
            "key": key,
            "in_liquid": True,
            "excepted": False,
            "symbol": "AAA",
            "session_date": DAYS[3],
            "year": 2019,
        }
    ]
    excepted = world.audit(con, checks=["P4"], exceptions=[exc("P4", key)])["P4"]
    assert excepted.total == 0 and excepted.excepted == 1
    world.misfiled = [(2020, misfiled)]
    assert rows(world.audit(con, checks=["P4"])["P4"]) == []


def test_p5_untraded_bar_that_is_not_flat(con: duckdb.DuckDBPyConnection, world: World) -> None:
    moving = bar("AAA", DAYS[3], close=101.0, open=100.0, high=101.0, low=99.0, volume=0)
    world.bars = replace(world.bars, "AAA", DAYS[3], moving)
    key = f"AAA:{iso(DAYS[3])}"

    result = world.audit(con, checks=["P5"])["P5"]

    assert result.rows.schema.names == [
        *HEAD,
        "symbol",
        "session_date",
        "open",
        "high",
        "low",
        "close",
    ]
    assert rows(result) == [
        {
            "key": key,
            "in_liquid": True,
            "excepted": False,
            "symbol": "AAA",
            "session_date": DAYS[3],
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 101.0,
        }
    ]
    excepted = world.audit(con, checks=["P5"], exceptions=[exc("P5", key)])["P5"]
    assert excepted.total == 0 and excepted.excepted == 1
    # the real placeholder (VTIQ at 33.97, volume 0, flat) is not a finding
    world.bars = replace(world.bars, "AAA", DAYS[3], flat("AAA", DAYS[3], 33.97))
    assert rows(world.audit(con, checks=["P5"])["P5"]) == []


def test_p6_bar_knowable_before_the_evening_of_its_session(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    # the rule all as_of filtering rests on (real: 0 of 18.2M bars break it); one hour early is
    # a look-ahead into the after-hours volume
    early = avail(DAYS[3]) - timedelta(hours=1)
    world.bars = replace(
        world.bars,
        "AAA",
        DAYS[3],
        bar("AAA", DAYS[3], close=100.0, volume=30_003, available_at=early),
    )
    key = f"AAA:{iso(DAYS[3])}"
    result = world.audit(con, checks=["P6"])["P6"]
    assert result.rows.schema.names == [
        *HEAD,
        "symbol",
        "session_date",
        "t",
        "available_at",
        "expected_available_at",
    ]
    [row] = rows(result)
    assert (row["key"], row["available_at"], row["expected_available_at"]) == (
        key,
        early,
        avail(DAYS[3]),
    )
    assert result.fails("all") is True
    excepted = world.audit(con, checks=["P6"], exceptions=[exc("P6", key)])["P6"]
    assert excepted.total == 0 and excepted.excepted == 1
    # a timestamp on another New York day than the session date is caught too
    world.bars = replace(
        base_bars(), "AAA", DAYS[3], {**find(base_bars(), "AAA", DAYS[3]), "t": t_of(DAYS[4])}
    )
    assert keys(world.audit(con, checks=["P6"])["P6"]) == [key]


def test_m5_unknown_class_is_judged_against_every_class_above_the_threshold(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    # A security of unknown class is never in the liquid tier (common / etf only), so judged
    # against liquid_ids an unknown class could never fail the liquid gate. The caller passes
    # the union over every class as qualified_ids; M5 uses it.
    world.segs = base_segs()[:2] + [Seg(SID_CCC, "CCC", cik=1003, cls="unknown", rule="none")]
    world.write()
    [result] = run_audit(
        con,
        world.root,
        as_of=LATE,
        liquid_ids=BASE_LIQUID,
        qualified_ids=BASE_LIQUID | {SID_CCC},
        checks=["M5"],
    )
    assert [(r["key"], r["in_liquid"]) for r in rows(result)] == [(f"CCC:{iso(START)}", True)]
    assert result.fails("liquid") is True
    # the other checks keep the liquid tier as their scope
    [p1] = run_audit(
        con, world.root, as_of=LATE, liquid_ids=set(), qualified_ids={SID_AAA}, checks=["P1"]
    )
    assert p1.rows.num_rows == 0


def test_s1_ignores_a_split_whose_ex_date_bar_is_not_visible_yet(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    # a split is known from 00:00 New York of its ex-date, the bar only from 20:00; at noon the
    # missing bar is not a defect
    world.splits = [split("AAA", DAYS[-1], 1.0, 2.0)]
    noon = known(DAYS[-1]) + timedelta(hours=12)
    assert rows(world.audit(con, as_of=noon, checks=["S1"])["S1"]) == []
    world.bars = drop(world.bars, "AAA", [DAYS[-1]])
    assert rows(world.audit(con, as_of=noon, checks=["S1"])["S1"]) == []
    assert keys(world.audit(con, checks=["S1"])["S1"]) == [f"AAA:{iso(DAYS[-1])}"]


def test_s2_counts_the_split_records_it_cannot_test(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    # only a placeholder on the ex-date (real: 24 records), and a traded ex-date bar with no
    # traded bar in the five sessions before it
    world.splits = [split("AAA", DAYS[10], 1.0, 4.0), split("CCC", DAYS[12], 1.0, 4.0)]
    world.bars = replace(world.bars, "AAA", DAYS[10], flat("AAA", DAYS[10], 100.0))
    world.bars = drop(world.bars, "CCC", DAYS[6:12])
    result = world.audit(con, checks=["S2"])["S2"]
    assert rows(result) == []
    assert result.notes == {"unchecked_untraded": 1, "unchecked_no_prev": 1}
    clean = World(world.root)
    assert clean.audit(con, checks=["S2"])["S2"].notes == {
        "unchecked_untraded": 0,
        "unchecked_no_prev": 0,
    }


# =============================================================================================
# M: the security master and the class table
# =============================================================================================


def test_m1_adjacent_same_symbol_segments_disagree_on_the_instant(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    # ARNC shape: the old holder's end is dated at a later SEC snapshot (DAYS[12]) while the new
    # holder's start was knowable on the cut day (DAYS[10])
    cut = DAYS[10]
    world.segs += [
        Seg(SID_X1, "XXX", START, cut, KNOWN_EARLY, known(DAYS[12])),
        Seg(SID_X2, "XXX", cut, None, known(cut)),
    ]
    key = f"XXX:{iso(cut)}"

    result = world.audit(con, checks=["M1"])["M1"]

    assert result.rows.schema.names == [
        *HEAD,
        "symbol",
        "old_security_id",
        "new_security_id",
        "valid_from",
        "end_available_at",
        "available_at",
    ]
    assert rows(result) == [
        {
            "key": key,
            "in_liquid": False,
            "excepted": False,
            "symbol": "XXX",
            "old_security_id": SID_X1,
            "new_security_id": SID_X2,
            "valid_from": cut,
            "end_available_at": known(DAYS[12]),
            "available_at": known(cut),
        }
    ]
    assert result.fails("all") is True
    excepted = world.audit(con, checks=["M1"], exceptions=[exc("M1", key)])["M1"]
    assert excepted.total == 0 and excepted.excepted == 1


def test_m1_matching_instants_and_gap_pairs_are_clean(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    cut = DAYS[10]
    world.segs = base_segs() + [
        Seg(SID_X1, "XXX", START, cut, KNOWN_EARLY, known(cut)),
        Seg(SID_X2, "XXX", cut, None, known(cut)),
    ]
    assert rows(world.audit(con, checks=["M1"])["M1"]) == []
    # FFR shape: vacated, dead for a while, retaken: the two instants need not agree
    world.segs = base_segs() + [
        Seg(SID_X1, "XXX", START, DAYS[8], KNOWN_EARLY, known(DAYS[8])),
        Seg(SID_X2, "XXX", DAYS[12], None, known(DAYS[12])),
    ]
    assert rows(world.audit(con, checks=["M1"])["M1"]) == []


def test_m2_consecutive_segments_of_one_security_overlap(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    world.segs += [
        Seg(SID_R, "ROLD", START, DAYS[12], KNOWN_EARLY, known(DAYS[12])),
        Seg(SID_R, "RNEW", DAYS[10], None, known(DAYS[10])),
    ]
    world.liquid_ids.add(SID_R)
    key = f"{SID_R}:{iso(DAYS[10])}"

    result = world.audit(con, checks=["M2"])["M2"]

    assert result.rows.schema.names == [
        *HEAD,
        "security_id",
        "old_symbol",
        "new_symbol",
        "old_valid_to",
        "new_valid_from",
    ]
    assert rows(result) == [
        {
            "key": key,
            "in_liquid": True,
            "excepted": False,
            "security_id": SID_R,
            "old_symbol": "ROLD",
            "new_symbol": "RNEW",
            "old_valid_to": DAYS[12],
            "new_valid_from": DAYS[10],
        }
    ]
    excepted = world.audit(con, checks=["M2"], exceptions=[exc("M2", key)])["M2"]
    assert excepted.total == 0 and excepted.excepted == 1 and excepted.fails("all") is False
    # back to back is clean
    world.segs = base_segs() + [
        Seg(SID_R, "ROLD", START, DAYS[10], KNOWN_EARLY, known(DAYS[10])),
        Seg(SID_R, "RNEW", DAYS[10], None, known(DAYS[10])),
    ]
    assert rows(world.audit(con, checks=["M2"])["M2"]) == []


def test_m3_gap_between_segments_of_one_security_counts_traded_bars_in_it(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    world.segs += [
        Seg(SID_R, "ROLD", START, DAYS[8], KNOWN_EARLY, known(DAYS[8])),
        Seg(SID_R, "RNEW", DAYS[12], None, known(DAYS[12])),
    ]
    # ROLD trades two sessions into the gap and has a placeholder on a third; RNEW starts one
    # session early
    world.bars += series("ROLD", DAYS[:10], 20.0, 3_000) + [flat("ROLD", DAYS[10], 20.0)]
    world.bars += series("RNEW", DAYS[11:], 20.0, 3_000, start=11)
    key = f"{SID_R}:{iso(DAYS[12])}"

    result = world.audit(con, checks=["M3"])["M3"]

    assert result.rows.schema.names == [
        *HEAD,
        "security_id",
        "old_symbol",
        "new_symbol",
        "old_valid_to",
        "new_valid_from",
        "gap_days",
        "traded_bars_in_gap",
    ]
    assert rows(result) == [
        {
            "key": key,
            "in_liquid": False,
            "excepted": False,
            "security_id": SID_R,
            "old_symbol": "ROLD",
            "new_symbol": "RNEW",
            "old_valid_to": DAYS[8],
            "new_valid_from": DAYS[12],
            "gap_days": (DAYS[12] - DAYS[8]).days,
            "traded_bars_in_gap": 3,
        }
    ]
    assert result.notes == {"traded_bars_in_gaps": 3}
    assert result.fails("all") is False and result.fails("liquid") is False  # information only


def test_m3_back_to_back_segments_are_not_a_gap(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    world.segs += [
        Seg(SID_R, "ROLD", START, DAYS[10], KNOWN_EARLY, known(DAYS[10])),
        Seg(SID_R, "RNEW", DAYS[10], None, known(DAYS[10])),
    ]
    result = world.audit(con, checks=["M3"])["M3"]
    assert rows(result) == [] and result.notes == {"traded_bars_in_gaps": 0}


def test_m4_master_and_class_table_disagree_on_both_sides(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    world.class_rows = [class_row(s) for s in world.segs if s.symbol != "BBB"] + [
        class_row(Seg("S000099", "ZZZ"))
    ]

    result = world.audit(con, checks=["M4"])["M4"]

    assert result.rows.schema.names == [*HEAD, "security_id", "symbol", "valid_from", "side"]
    assert rows(result) == [
        {
            "key": f"BBB:{iso(START)}",
            "in_liquid": True,
            "excepted": False,
            "security_id": SID_BBB,
            "symbol": "BBB",
            "valid_from": START,
            "side": "master",
        },
        {
            "key": f"ZZZ:{iso(START)}",
            "in_liquid": False,
            "excepted": False,
            "security_id": "S000099",
            "symbol": "ZZZ",
            "valid_from": START,
            "side": "class",
        },
    ]
    assert result.fails("all") is True
    excepted = world.audit(
        con,
        checks=["M4"],
        exceptions=[exc("M4", f"BBB:{iso(START)}"), exc("M4", f"ZZZ:{iso(START)}")],
    )["M4"]
    assert excepted.total == 0 and excepted.excepted == 2 and excepted.fails("all") is False


def test_m4_is_keyed_on_the_full_segment_key_not_the_symbol(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    # a class row whose valid_from differs from the segment's matches nothing: two rows
    world.class_rows = [class_row(s) for s in world.segs if s.symbol != "CCC"] + [
        class_row(Seg(SID_CCC, "CCC", DAYS[0]))
    ]
    result = world.audit(con, checks=["M4"])["M4"]
    assert [(r["key"], r["side"]) for r in rows(result)] == [
        (f"CCC:{iso(START)}", "master"),
        (f"CCC:{iso(DAYS[0])}", "class"),
    ]


def test_m5_unknown_class_rows_and_the_notes_about_weak_evidence(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    cut = DAYS[5]
    world.segs = [
        Seg(SID_AAA, "AAA", cik=1001, name_source="sec", rule="lookup_sec_companies"),
        Seg(SID_BBB, "BBB", cik=1002, cls="unknown", name_source="none", rule="none"),
        # non-liquid: carries the same weak evidence as AAA but must not be counted in the notes
        Seg(SID_CCC, "CCC", cik=1003, name_source="none", rule="lookup_sec_companies"),
        # one symbol, two classes
        Seg(SID_D1, "DDD", START, cut, KNOWN_EARLY, known(cut), cls="common"),
        Seg(SID_D2, "DDD", cut, None, known(cut), cls="etf"),
    ]
    key = f"BBB:{iso(START)}"

    result = world.audit(con, checks=["M5"])["M5"]

    assert result.rows.schema.names == [
        *HEAD,
        "security_id",
        "symbol",
        "valid_from",
        "name",
        "rule",
    ]
    assert rows(result) == [
        {
            "key": key,
            "in_liquid": True,
            "excepted": False,
            "security_id": SID_BBB,
            "symbol": "BBB",
            "valid_from": START,
            "name": "BBB Inc",
            "rule": "none",
        }
    ]
    assert result.notes == {
        "liquid_lookup_sec_companies": 1,
        "liquid_name_from_sec_or_none_common_etf": 1,
        "symbols_with_mixed_classes": 1,
    }
    assert result.fails("liquid") is True and result.fails("all") is True
    excepted = world.audit(con, checks=["M5"], exceptions=[exc("M5", key)])["M5"]
    assert excepted.liquid == 0 and excepted.fails("liquid") is False


def test_m5_unknown_outside_the_liquid_scope_fails_only_the_all_tier(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    world.segs = base_segs()[:2] + [Seg(SID_CCC, "CCC", cik=1003, cls="unknown", rule="none")]
    result = world.audit(con, checks=["M5"])["M5"]
    assert [(r["key"], r["in_liquid"]) for r in rows(result)] == [(f"CCC:{iso(START)}", False)]
    assert result.fails("liquid") is False and result.fails("all") is True
    assert result.notes == {
        "liquid_lookup_sec_companies": 0,
        "liquid_name_from_sec_or_none_common_etf": 0,
        "symbols_with_mixed_classes": 0,
    }


def test_m6_identity_conflicts_are_listed_one_row_each(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    world.conflicts = [
        conflict("dup_unlinked", ["AAA", "QQQ"], [DAYS[0], DAYS[1]], "2 identical days"),
        conflict("reuse_looks_continuous", ["PPP"], [DAYS[4]], "ratio 1.01 across the reuse"),
    ]

    result = world.audit(con, checks=["M6"])["M6"]

    assert result.rows.schema.names == [*HEAD, "kind", "symbols", "dates", "detail"]
    assert rows(result) == [
        {
            "key": f"AAA/QQQ:dup_unlinked:{iso(DAYS[0])},{iso(DAYS[1])}",
            "in_liquid": True,
            "excepted": False,
            "kind": "dup_unlinked",
            "symbols": "AAA/QQQ",
            "dates": f"{iso(DAYS[0])},{iso(DAYS[1])}",
            "detail": "2 identical days",
        },
        {
            "key": f"PPP:reuse_looks_continuous:{iso(DAYS[4])}",
            "in_liquid": False,
            "excepted": False,
            "kind": "reuse_looks_continuous",
            "symbols": "PPP",
            "dates": iso(DAYS[4]),
            "detail": "ratio 1.01 across the reuse",
        },
    ]
    assert result.fails("liquid") is True
    excepted = world.audit(
        con,
        checks=["M6"],
        exceptions=[exc("M6", f"AAA/QQQ:dup_unlinked:{iso(DAYS[0])},{iso(DAYS[1])}")],
    )["M6"]
    assert (excepted.liquid, excepted.total, excepted.excepted) == (0, 1, 1)
    assert excepted.fails("liquid") is False and excepted.fails("all") is True
    world.conflicts = None  # no file at all
    assert rows(world.audit(con, checks=["M6"])["M6"]) == []


# =============================================================================================
# I: identity and integrity across bars and master
# =============================================================================================


def test_i1_orphan_traded_bars_per_symbol_minus_relabel_copies(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    orph = [
        bar("ORPH", d, close=5.0, vwap=5.0, volume=v)
        for d, v in zip(DAYS[:5], [2_000, 2_001, 2_002, 300_000, 300_001], strict=True)
    ]
    world.bars += orph + [flat("ORPH", DAYS[5], 5.0)]  # the placeholder is not traded
    world.bars += relabel(series("AAA", DAYS, 100.0, 30_000), "AAA2")  # BNY-shape copy

    result = world.audit(con, checks=["I1"])["I1"]

    assert result.rows.schema.names == [*HEAD, "symbol", "bars", "first", "last", "usd"]
    [row] = rows(result)
    assert {
        k: row[k] for k in ("key", "in_liquid", "excepted", "symbol", "bars", "first", "last")
    } == {
        "key": "ORPH",
        "in_liquid": False,
        "excepted": False,
        "symbol": "ORPH",
        "bars": 5,
        "first": DAYS[0],
        "last": DAYS[4],
    }
    assert row["usd"] == pytest.approx(5.0 * (2_000 + 2_001 + 2_002 + 300_000 + 300_001))
    assert result.notes == {"bars": 5, "bars_over_1m_usd": 2, "symbols_over_1m_usd": 1}
    assert result.fails("all") is False  # information only


def test_i1_relabel_copy_with_one_differing_day_reports_that_day_only(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    copy = relabel(series("AAA", DAYS, 100.0, 30_000), "AAA2")
    copy[-1] = bar("AAA2", DAYS[-1], close=101.0, volume=30_024)  # not AAA's bar
    world.bars += copy
    [row] = rows(world.audit(con, checks=["I1"])["I1"])
    assert (row["symbol"], row["bars"], row["first"], row["last"]) == (
        "AAA2",
        1,
        DAYS[-1],
        DAYS[-1],
    )


def test_i1_null_vwap_prices_the_bar_off_close(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    world.bars += [{**bar("ORPH", DAYS[0], close=5.0, volume=1_000), "vwap": None}]
    [row] = rows(world.audit(con, checks=["I1"])["I1"])
    assert row["usd"] == pytest.approx(5_000.0)


def test_i2_identical_bars_under_two_symbols_in_the_resolved_view(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    aaa = series("AAA", DAYS, 100.0, 30_000)
    for d in DAYS[2:5]:
        world.bars = replace(world.bars, "BBB", d, {**find(aaa, "AAA", d), "symbol": "BBB"})
    # identical at volume 1000 exactly: below the floor, not counted
    thin = bar("AAA", DAYS[5], close=100.0, volume=1_000)
    world.bars = replace(world.bars, "AAA", DAYS[5], thin)
    world.bars = replace(world.bars, "BBB", DAYS[5], {**thin, "symbol": "BBB"})

    result = world.audit(con, checks=["I2"])["I2"]

    assert result.rows.schema.names == [
        *HEAD,
        "symbol_a",
        "symbol_b",
        "security_a",
        "security_b",
        "days",
        "first",
        "last",
    ]
    assert rows(result) == [
        {
            "key": "AAA/BBB",
            "in_liquid": True,
            "excepted": False,
            "symbol_a": "AAA",
            "symbol_b": "BBB",
            "security_a": SID_AAA,
            "security_b": SID_BBB,
            "days": 3,
            "first": DAYS[2],
            "last": DAYS[4],
        }
    ]
    assert result.fails("liquid") is True
    excepted = world.audit(con, checks=["I2"], exceptions=[exc("I2", "AAA/BBB")])["I2"]
    assert excepted.liquid == 0 and excepted.fails("liquid") is False


def test_i2_ignores_bars_outside_the_resolved_view(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    # the BNY-shape copy has no segment: identical, but not a resolved pair
    world.bars += relabel(series("AAA", DAYS, 100.0, 30_000), "AAA2")
    assert rows(world.audit(con, checks=["I2"])["I2"]) == []


def test_i3_one_security_traded_under_two_symbols_on_one_session(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    first = DAYS[5]
    world.segs.append(Seg(SID_AAA, "AAAX", first, None, known(first)))
    aaa = series("AAA", DAYS, 100.0, 30_000)
    world.bars += [
        {**find(aaa, "AAA", DAYS[5]), "symbol": "AAAX"},
        {**find(aaa, "AAA", DAYS[6]), "symbol": "AAAX"},
        bar("AAAX", DAYS[7], close=90.0, volume=7_000),
        flat("AAAX", DAYS[8], 90.0),  # untraded: not a double day
    ]

    result = world.audit(con, checks=["I3"])["I3"]

    assert result.rows.schema.names == [*HEAD, "security_id", "symbols", "days", "identical_days"]
    assert rows(result) == [
        {
            "key": SID_AAA,
            "in_liquid": True,
            "excepted": False,
            "security_id": SID_AAA,
            "symbols": "AAA/AAAX",
            "days": 3,
            "identical_days": 2,
        }
    ]
    assert result.fails("liquid") is True
    excepted = world.audit(con, checks=["I3"], exceptions=[exc("I3", SID_AAA)])["I3"]
    assert excepted.liquid == 0 and excepted.excepted == 1


def test_i3_rename_is_one_timeline_not_a_double_day(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    cut = DAYS[10]
    world.segs += [
        Seg(SID_R, "ROLD", START, cut, KNOWN_EARLY, known(cut)),
        Seg(SID_R, "RNEW", cut, None, known(cut)),
    ]
    world.bars += series("ROLD", DAYS[:10], 20.0, 3_000) + series("RNEW", DAYS[10:], 20.0, 3_000)
    world.bars += [flat("ROLD", d, 20.0) for d in DAYS[10:13]]  # VTIQ-style placeholders
    assert rows(world.audit(con, checks=["I3"])["I3"]) == []


def lac_world(world: World) -> World:
    """BBUC shape: LAC's cut is DAYS[10] on disk, knowable DAYS[14]; LAC keeps trading."""
    cut, knowable = DAYS[10], DAYS[14]
    world.segs += [
        Seg(SID_LAC, "LAC", START, cut, KNOWN_EARLY, known(knowable)),
        Seg(SID_LAC, "LAAC", cut, None, known(knowable)),
    ]
    world.bars += series("LAC", DAYS[:15], 20.0, 3_000)  # through DAYS[14] inclusive
    world.bars += series("LAAC", DAYS[10:], 20.0, 4_000, start=10)
    return world


def test_i4_old_label_traded_between_the_cut_and_the_day_it_became_knowable(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    lac_world(world)

    result = world.audit(con, checks=["I4"])["I4"]

    assert result.rows.schema.names == [
        *HEAD,
        "symbol",
        "security_id",
        "valid_to",
        "end_knowable",
        "bars",
    ]
    # DAYS[10..13]: four bars; the DAYS[14] bar is on the knowable day itself and not counted
    assert rows(result) == [
        {
            "key": "LAC",
            "in_liquid": False,
            "excepted": False,
            "symbol": "LAC",
            "security_id": SID_LAC,
            "valid_to": DAYS[10],
            "end_knowable": DAYS[14],
            "bars": 4,
        }
    ]
    assert result.fails("all") is False  # information only


def test_i4_is_empty_while_the_end_is_not_yet_knowable_and_on_a_clean_rename(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    lac_world(world)
    # a reader before the filing still sees LAC open: those bars resolve, nothing to report
    assert rows(world.audit(con, checks=["I4"], as_of=avail(DAYS[13]))["I4"]) == []
    # a rename filed on its day (BK -> BNY shape) has no such window
    world.segs = base_segs() + [
        Seg(SID_LAC, "LAC", START, DAYS[10], KNOWN_EARLY, known(DAYS[10])),
        Seg(SID_LAC, "LAAC", DAYS[10], None, known(DAYS[10])),
    ]
    assert rows(world.audit(con, checks=["I4"])["I4"]) == []


def test_i5_sec_history_lists_the_symbol_under_another_cik_for_an_interval(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    snap_a, snap_b = DAYS[7], DAYS[21]  # 2020-01-13, 2020-02-03
    world.sec = [
        (SEC_HISTORY_START, "AAA", 1001),
        (SEC_HISTORY_START, "BBB", 1002),
        (SEC_HISTORY_START, "CCC", 1003),
        (snap_a, "AAA", 1001),
        (snap_a, "BBB", 9999),  # ARNC shape: a different number for one interval
        (snap_a, "CCC", 1003),
        (snap_b, "AAA", 1001),
        (snap_b, "BBB", 1002),
        (snap_b, "CCC", 1003),
    ]
    in_interval = [d for d in DAYS if snap_a <= d < snap_b]
    assert len(in_interval) == 14

    result = world.audit(con, checks=["I5"])["I5"]

    assert result.rows.schema.names == [
        *HEAD,
        "security_id",
        "symbol",
        "valid_from",
        "segment_cik",
        "sec_cik",
        "bars",
    ]
    assert rows(result) == [
        {
            "key": f"BBB:{iso(START)}",
            "in_liquid": True,
            "excepted": False,
            "security_id": SID_BBB,
            "symbol": "BBB",
            "valid_from": START,
            "segment_cik": 1002,
            "sec_cik": 9999,
            "bars": 14,
        }
    ]
    assert result.fails("all") is False


def test_i5_ignores_stale_snapshots_and_segments_without_a_cik(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    stale = date(2019, 7, 1)
    assert stale < SEC_HISTORY_START
    world.segs = [Seg(SID_AAA, "AAA", cik=1001), Seg(SID_BBB, "BBB", cik=1002), Seg(SID_CCC, "CCC")]
    world.sec = [
        (stale, "AAA", 7777),  # would cover every January session if it were used
        (stale, "BBB", 1002),
        (DAYS[21], "AAA", 1001),
        (DAYS[21], "BBB", 1002),
        (DAYS[21], "CCC", 4444),  # CCC's segment has no cik: nothing to contradict
        (date(2020, 7, 1), "AAA", 1001),
    ]
    assert rows(world.audit(con, checks=["I5"])["I5"]) == []


def test_i5_ignores_rows_the_sec_later_corrected_in_bulk(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    # Real: between the 2020-06-01 and 2020-07-10 snapshots the SEC re-keyed 126 tickers at
    # once; AFC had been listed under Allied Capital (CIK 3906, acquired 2010) until then. The
    # master is built from the table with such bulk corrections dropped, and I5 must read the
    # same table, or every corrected ticker shows up as a contradiction for half a year.
    snap_a, snap_b = DAYS[7], DAYS[21]
    sec = [(SEC_HISTORY_START, sym, cik) for sym, cik in (("AAA", 1001), ("CCC", 1003))]
    sec += [(d, sym, cik) for d in (snap_a, snap_b) for sym, cik in (("AAA", 1001), ("CCC", 1003))]
    sec += [(SEC_HISTORY_START, "BBB", 9999), (snap_a, "BBB", 9999), (snap_b, "BBB", 1002)]
    world.sec = sec
    assert keys(world.audit(con, checks=["I5"])["I5"]) == [f"BBB:{iso(START)}"]  # one switch
    # 50 more tickers switch in the same snapshot pair, their old CIKs never seen again: a bulk
    # correction (TABLE_CORRECTION_MIN_SWITCHES = 50), so BBB's 9999 rows are dropped too
    world.sec = sec + [
        (d, f"Z{n:02d}", (5000 if d < snap_b else 6000) + n)
        for n in range(50)
        for d in (snap_a, snap_b)
    ]
    assert rows(world.audit(con, checks=["I5"])["I5"]) == []


def test_i6_untraded_share_per_year(con: duckdb.DuckDBPyConnection, world: World) -> None:
    for d in DAYS[-5:]:
        world.bars = replace(world.bars, "CCC", d, flat("CCC", d, 10.0))
    world.bars.append(bar("AAA", date(2019, 12, 31), close=100.0, volume=29_000))

    result = world.audit(con, checks=["I6"])["I6"]

    assert result.rows.schema.names == [*HEAD, "year", "untraded", "bars", "share"]
    out = rows(result)
    assert [(r["key"], r["in_liquid"], r["year"], r["untraded"], r["bars"]) for r in out] == [
        ("2019", False, 2019, 0, 1),
        ("2020", False, 2020, 5, 75),
    ]
    assert [r["share"] for r in out] == pytest.approx([0.0, 5 / 75])
    assert result.notes["untraded"] == 5 and result.notes["bars"] == 76
    assert result.notes["share"] == pytest.approx(5 / 76)


def test_i6_clean_store_has_a_zero_row_per_year(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    result = world.audit(con, checks=["I6"])["I6"]
    assert [(r["key"], r["untraded"], r["bars"], r["share"]) for r in rows(result)] == [
        ("2020", 0, 75, 0.0)
    ]
    assert result.notes == {"untraded": 0, "bars": 75, "share": 0.0}


def test_i7_session_with_far_fewer_traded_symbols_than_the_median(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    dead = DAYS[20]
    for symbol, close in (("AAA", 100.0), ("BBB", 60.0), ("CCC", 10.0)):
        world.bars = replace(world.bars, symbol, dead, flat(symbol, dead, close))
    # one of three traded on DAYS[21]: 1 >= 0.1 * 3, not reported
    for symbol, close in (("BBB", 60.0), ("CCC", 10.0)):
        world.bars = replace(world.bars, symbol, DAYS[21], flat(symbol, DAYS[21], close))

    result = world.audit(con, checks=["I7"])["I7"]

    assert result.rows.schema.names == [*HEAD, "session_date", "traded_symbols"]
    assert rows(result) == [
        {
            "key": iso(dead),
            "in_liquid": False,
            "excepted": False,
            "session_date": dead,
            "traded_symbols": 0,
        }
    ]
    assert result.notes == {"min_traded_symbols": 0, "median_traded_symbols": 3, "sessions": 25}
    assert result.fails("all") is True and result.fails("liquid") is True
    excepted = world.audit(con, checks=["I7"], exceptions=[exc("I7", iso(dead))])["I7"]
    assert excepted.total == 0 and excepted.fails("all") is False


@pytest.mark.parametrize(("missing", "reported"), [(10, False), (11, True)])
def test_i8_gap_of_more_than_ten_sessions_in_a_securitys_traded_bars(
    con: duckdb.DuckDBPyConnection, world: World, missing: int, reported: bool
) -> None:
    gap = DAYS[4 : 4 + missing]
    world.bars = drop(world.bars, "CCC", gap)
    world.bars += [flat("CCC", d, 10.0) for d in gap]  # placeholders do not bridge a gap
    after = DAYS[4 + missing]

    result = world.audit(con, checks=["I8"])["I8"]

    assert result.rows.schema.names == [
        *HEAD,
        "security_id",
        "symbol",
        "gap_from",
        "gap_to",
        "sessions_missing",
    ]
    expected = (
        [
            {
                "key": f"{SID_CCC}:{iso(after)}",
                "in_liquid": False,
                "excepted": False,
                "security_id": SID_CCC,
                "symbol": "CCC",
                "gap_from": DAYS[3],
                "gap_to": after,
                "sessions_missing": missing,
            }
        ]
        if reported
        else []
    )
    assert rows(result) == expected
    assert result.fails("all") is False


def test_i8_counts_sessions_across_a_rename_and_not_calendar_days(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    # one security, two names: ROLD stops at DAYS[3], RNEW starts at DAYS[16]: 12 sessions
    # missing (MLK day is not a session and does not count)
    cut = DAYS[10]
    world.segs += [
        Seg(SID_R, "ROLD", START, cut, KNOWN_EARLY, known(cut)),
        Seg(SID_R, "RNEW", cut, None, known(cut)),
    ]
    world.bars += series("ROLD", DAYS[:4], 20.0, 3_000) + series("RNEW", DAYS[16:], 20.0, 3_000)
    [row] = rows(world.audit(con, checks=["I8"])["I8"])
    assert (row["key"], row["symbol"], row["gap_from"], row["gap_to"], row["sessions_missing"]) == (
        f"{SID_R}:{iso(DAYS[16])}",
        "RNEW",
        DAYS[3],
        DAYS[16],
        12,
    )


def rename_world(world: World, *, old_close: float, new_close: float, old_until: int = 10) -> World:
    cut = DAYS[10]
    world.segs += [
        Seg(SID_R, "ROLD", START, cut, KNOWN_EARLY, known(cut)),
        Seg(SID_R, "RNEW", cut, None, known(cut)),
    ]
    world.bars += series("ROLD", DAYS[:old_until], old_close, 3_000)
    world.bars += series("RNEW", DAYS[10:], new_close, 3_000, start=10)
    return world


def test_i9_price_jump_across_a_rename(con: duckdb.DuckDBPyConnection, world: World) -> None:
    rename_world(world, old_close=20.0, new_close=10.0)
    key = f"{SID_R}:{iso(DAYS[10])}"

    result = world.audit(con, checks=["I9"])["I9"]

    assert result.rows.schema.names == [
        *HEAD,
        "security_id",
        "old_symbol",
        "new_symbol",
        "last_close",
        "last_date",
        "first_close",
        "first_date",
        "ratio",
        "gap_days",
    ]
    [row] = rows(result)
    assert row["ratio"] == pytest.approx(0.5)
    del row["ratio"]
    assert row == {
        "key": key,
        "in_liquid": False,
        "excepted": False,
        "security_id": SID_R,
        "old_symbol": "ROLD",
        "new_symbol": "RNEW",
        "last_close": 20.0,
        "last_date": DAYS[9],
        "first_close": 10.0,
        "first_date": DAYS[10],
        "gap_days": (DAYS[10] - DAYS[9]).days,
    }
    assert result.fails("all") is False


def test_i9_continuous_prices_are_clean_and_a_split_at_the_cut_explains_a_jump(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    rename_world(world, old_close=20.0, new_close=20.5)
    assert rows(world.audit(con, checks=["I9"])["I9"]) == []
    # AAPL-shape 4:1 under the new name on the cut day: 100 -> 25 is no jump
    world.bars = base_bars()
    world.segs = base_segs()
    rename_world(world, old_close=100.0, new_close=25.0)
    world.splits = [split("RNEW", DAYS[10], 1.0, 4.0)]
    assert rows(world.audit(con, checks=["I9"])["I9"]) == []
    # the same split recorded under the old name counts too
    world.splits = [split("ROLD", DAYS[10], 1.0, 4.0)]
    assert rows(world.audit(con, checks=["I9"])["I9"]) == []
    # a split before the last old bar does not explain anything
    world.splits = [split("RNEW", DAYS[8], 1.0, 4.0)]
    assert [r["ratio"] for r in rows(world.audit(con, checks=["I9"])["I9"])] == pytest.approx(
        [0.25]
    )


def test_i9_long_calendar_gap_or_missing_close_is_reported_whatever_the_ratio(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    rename_world(world, old_close=20.0, new_close=20.0, old_until=2)  # last old bar DAYS[1]
    [row] = rows(world.audit(con, checks=["I9"])["I9"])
    assert (row["last_date"], row["gap_days"]) == (DAYS[1], (DAYS[10] - DAYS[1]).days)
    assert row["gap_days"] > 10 and row["ratio"] == pytest.approx(1.0)
    # exactly ten calendar days is not "more than ten"
    world.bars = base_bars()
    world.segs = base_segs()
    rename_world(world, old_close=20.0, new_close=20.0, old_until=3)  # DAYS[2] = 01-06 -> 01-16
    assert (DAYS[10] - DAYS[2]).days == 10
    assert rows(world.audit(con, checks=["I9"])["I9"]) == []
    # the new segment never traded: a row with first_close missing
    world.bars = base_bars()
    world.segs = base_segs()
    rename_world(world, old_close=20.0, new_close=20.0)
    world.bars = drop(world.bars, "RNEW", DAYS)
    [row] = rows(world.audit(con, checks=["I9"])["I9"])
    assert row["first_close"] is None and row["last_close"] == 20.0


# =============================================================================================
# S: splits and price jumps
# =============================================================================================


def test_s1_split_records_without_a_bar_on_the_ex_date(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    saturday = date(2020, 1, 11)
    assert saturday not in DAYS
    world.splits = [
        split("AAA", saturday, 1.0, 2.0),
        split("NOPE", DAYS[9], 1.0, 2.0),
        split("BBB", DAYS[9], 1.0, 2.0),  # has a bar: not S1's business
    ]

    result = world.audit(con, checks=["S1"])["S1"]

    assert result.rows.schema.names == [*HEAD, "symbol", "ex_date", "kind", "factor", "reason"]
    assert rows(result) == [
        {
            "key": f"AAA:{iso(saturday)}",
            "in_liquid": True,
            "excepted": False,
            "symbol": "AAA",
            "ex_date": saturday,
            "kind": "forward",
            "factor": 2.0,
            "reason": "no_bar_that_day",
        },
        {
            "key": f"NOPE:{iso(DAYS[9])}",
            "in_liquid": False,
            "excepted": False,
            "symbol": "NOPE",
            "ex_date": DAYS[9],
            "kind": "forward",
            "factor": 2.0,
            "reason": "symbol_never_stored",
        },
    ]
    assert result.notes == {"symbol_never_stored": 1, "no_bar_that_day": 1}
    assert result.fails("liquid") is True
    excepted = world.audit(con, checks=["S1"], exceptions=[exc("S1", f"AAA:{iso(saturday)}")])["S1"]
    assert (excepted.liquid, excepted.total, excepted.excepted) == (0, 1, 1)
    assert excepted.fails("liquid") is False and excepted.fails("all") is True


def split_world(world: World, *, factor_applied: bool, kind: str = "forward") -> World:
    """AAA 4:1 on DAYS[10]; with ``factor_applied`` the price actually drops to a quarter."""
    world.splits = [split("AAA", DAYS[10], 1.0, 4.0, kind=kind)]
    if factor_applied:
        aaa = series("AAA", DAYS[:10], 100.0, 30_000) + series(
            "AAA", DAYS[10:], 25.0, 30_000, start=10
        )
        world.bars = [b for b in world.bars if b["symbol"] != "AAA"] + aaa
    return world


def test_s2_split_whose_price_did_not_move(con: duckdb.DuckDBPyConnection, world: World) -> None:
    split_world(world, factor_applied=False)
    key = f"AAA:{iso(DAYS[10])}"

    result = world.audit(con, checks=["S2"])["S2"]

    assert result.rows.schema.names == [
        *HEAD,
        "symbol",
        "ex_date",
        "kind",
        "factor",
        "prev_close",
        "open",
        "observed",
        "expected",
    ]
    [row] = rows(result)
    assert row["observed"] == pytest.approx(99.5 / 100.0)
    assert row["expected"] == pytest.approx(0.25)
    del row["observed"], row["expected"]
    assert row == {
        "key": key,
        "in_liquid": True,
        "excepted": False,
        "symbol": "AAA",
        "ex_date": DAYS[10],
        "kind": "forward",
        "factor": 4.0,
        "prev_close": 100.0,
        "open": 99.5,
    }
    assert result.fails("liquid") is True
    excepted = world.audit(con, checks=["S2"], exceptions=[exc("S2", key)])["S2"]
    assert excepted.liquid == 0 and excepted.fails("liquid") is False


def test_s2_clean_when_the_price_moved_and_skipped_for_unit_splits_or_untraded_days(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    split_world(world, factor_applied=True)
    results = world.audit(con, checks=["S2", "S3", "S4"])
    assert rows(results["S2"]) == []
    # the split record also explains the 4x drop to S3 and S4
    assert rows(results["S3"]) == [] and rows(results["S4"]) == []

    unit = World(world.root)
    split_world(unit, factor_applied=False, kind="unit_split")
    assert rows(unit.audit(con, checks=["S2"])["S2"]) == []

    untraded = World(world.root)
    split_world(untraded, factor_applied=False)
    untraded.bars = replace(untraded.bars, "AAA", DAYS[10], flat("AAA", DAYS[10], 100.0))
    assert rows(untraded.audit(con, checks=["S2"])["S2"]) == []


def spike_world(
    world: World,
    *,
    volume: int = 30_010,
    close: float = 100.0,
    high_mult: float = 4.0,
    next_close: float | None = None,
    split_day: date | None = None,
) -> World:
    """EXEEW shape: one bar whose high is ``high_mult`` x the previous close, then back."""
    aaa = series("AAA", DAYS, close, 30_000)
    spike = bar("AAA", DAYS[10], close=close, volume=volume, high=round(close * high_mult, 4))
    aaa = replace(aaa, "AAA", DAYS[10], spike)
    if next_close is not None:
        aaa = replace(aaa, "AAA", DAYS[11], bar("AAA", DAYS[11], close=next_close, volume=30_011))
    world.bars = [b for b in world.bars if b["symbol"] != "AAA"] + aaa
    if split_day is not None:
        world.splits = [split("AAA", split_day, 1.0, 2.0)]
    return world


def test_s3_one_day_spike_on_thin_volume_without_a_split(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    spike_world(world)
    key = f"AAA:{iso(DAYS[10])}"

    result = world.audit(con, checks=["S3"])["S3"]

    assert result.rows.schema.names == [
        *HEAD,
        "symbol",
        "session_date",
        "prev_close",
        "open",
        "high",
        "close",
        "next_close",
        "volume",
    ]
    assert rows(result) == [
        {
            "key": key,
            "in_liquid": True,
            "excepted": False,
            "symbol": "AAA",
            "session_date": DAYS[10],
            "prev_close": 100.0,
            "open": 99.5,
            "high": 400.0,
            "close": 100.0,
            "next_close": 100.0,
            "volume": 30_010,
        }
    ]
    assert result.fails("liquid") is True
    excepted = world.audit(con, checks=["S3"], exceptions=[exc("S3", key)])["S3"]
    assert excepted.liquid == 0 and excepted.fails("liquid") is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"volume": 50_000},  # not thin
        {"high_mult": 2.9},  # not a 3x move
        {"next_close": 250.0},  # the move persisted: not a one-day print
        {"close": 0.9},  # prev_close below $1: penny noise, not S3's business
        {"split_day": DAYS[13]},  # a split within three sessions explains it
        {"split_day": DAYS[7]},
    ],
    ids=["volume_at_floor", "below_3x", "persistent", "sub_dollar", "split_after", "split_before"],
)
def test_s3_boundaries_are_not_reported(
    con: duckdb.DuckDBPyConnection, world: World, kwargs: dict[str, Any]
) -> None:
    spike_world(world, **kwargs)
    assert rows(world.audit(con, checks=["S3"])["S3"]) == []


def test_s3_previous_close_may_come_from_a_placeholder(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    # EXEEW (real): flat volume-0 bars at 105 for weeks, then one print at 0.0101 on 2026-02-03
    # (volume 101), then 125 the next day. The last *traded* bar is far more than five sessions
    # back, but the placeholder carries the last close, so the spike still has a previous close.
    sym, spike_day = "EXW", DAYS[10]
    world.bars += [bar(sym, d, close=105.0, volume=900 + i) for i, d in enumerate(DAYS[:2])]
    world.bars += [flat(sym, d, 105.0) for d in DAYS[2:10]]
    world.bars += [
        bar(sym, spike_day, close=0.0101, open=0.0101, high=0.0101, low=0.0101, volume=101),
        bar(sym, DAYS[11], close=125.0, open=70.0, high=138.56, low=70.0, volume=2_959),
    ]
    [row] = rows(world.audit(con, checks=["S3"])["S3"])
    assert (row["key"], row["prev_close"], row["close"], row["next_close"]) == (
        f"{sym}:{iso(spike_day)}",
        105.0,
        0.0101,
        125.0,
    )
    # a placeholder more than five sessions back is not a previous close
    world.bars = [
        b for b in world.bars if not (b["symbol"] == sym and b["session_date"] in DAYS[4:10])
    ]
    assert rows(world.audit(con, checks=["S3"])["S3"]) == []


def test_s3_split_four_sessions_away_does_not_explain_the_spike(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    spike_world(world, split_day=DAYS[14])
    assert keys(world.audit(con, checks=["S3"])["S3"]) == [f"AAA:{iso(DAYS[10])}"]


def test_s3_needs_a_prev_and_a_next_traded_bar_within_five_sessions(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    spike_world(world)
    world.bars = drop(world.bars, "AAA", DAYS[11:16])  # next traded bar six sessions later
    assert rows(world.audit(con, checks=["S3"])["S3"]) == []
    spike_world(world)
    world.bars = drop(world.bars, "AAA", DAYS[11:15])  # five sessions later: still "next"
    assert keys(world.audit(con, checks=["S3"])["S3"]) == [f"AAA:{iso(DAYS[10])}"]


def jump_world(
    world: World, *, before: float, after: float, split_day: date | None = None
) -> World:
    ccc = series("CCC", DAYS[:10], before, 5_000) + series("CCC", DAYS[10:], after, 5_000, start=10)
    world.bars = [b for b in world.bars if b["symbol"] != "CCC"] + ccc
    if split_day is not None:
        world.splits = [split("CCC", split_day, 1.0, 4.0)]
    return world


@pytest.mark.parametrize(
    ("before", "after", "bucket"),
    [(0.5, 2.0, "lt1"), (5.0, 20.0, "1to10"), (50.0, 200.0, "gt10"), (50.0, 15.0, "gt10")],
    ids=["lt1", "1to10", "gt10_up", "gt10_down"],
)
def test_s4_close_jump_without_a_split_bucketed_by_price(
    con: duckdb.DuckDBPyConnection, world: World, before: float, after: float, bucket: str
) -> None:
    jump_world(world, before=before, after=after)
    key = f"CCC:{iso(DAYS[10])}"

    result = world.audit(con, checks=["S4"])["S4"]

    assert result.rows.schema.names == [
        *HEAD,
        "symbol",
        "session_date",
        "prev_close",
        "close",
        "ratio",
        "usd",
        "bucket",
    ]
    [row] = rows(result)
    assert row["ratio"] == pytest.approx(after / before)
    assert row["usd"] == pytest.approx(round(after * 1.002, 4) * 5_010)
    del row["ratio"], row["usd"]
    assert row == {
        "key": key,
        "in_liquid": False,
        "excepted": False,
        "symbol": "CCC",
        "session_date": DAYS[10],
        "prev_close": before,
        "close": after,
        "bucket": bucket,
    }
    assert result.notes == {"lt1": 0, "1to10": 0, "gt10": 0} | {bucket: 1}
    assert result.fails("all") is False


def test_s4_split_nearby_or_a_smaller_move_is_clean(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    jump_world(world, before=5.0, after=20.0, split_day=DAYS[10])
    assert rows(world.audit(con, checks=["S4"])["S4"]) == []
    jump_world(world, before=5.0, after=14.0)
    assert rows(world.audit(con, checks=["S4"])["S4"]) == []


# =============================================================================================
# N: Alpaca name changes against the bars
# =============================================================================================


def test_n1_name_change_lag_histogram_and_rows_over_thirty_days(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    last = DAYS[-1]  # 2020-02-06, every base symbol's last traded bar
    world.name_changes = [
        rename("AAA", "AAA2", last - timedelta(days=5)),  # le0
        rename("BBB", "BBB2", last + timedelta(days=5)),  # d1_10
        rename("CCC", "CCC2", last + timedelta(days=30)),  # d11_30, not a row
        rename("CCC", "CCC3", last + timedelta(days=31)),  # d31_180
        rename("AAA", "AAA3", last + timedelta(days=208)),  # gt180
        rename("NOPE", "NOPE2", last + timedelta(days=4)),  # no bars under the old name
    ]

    result = world.audit(con, checks=["N1"])["N1"]

    assert result.rows.schema.names == [
        *HEAD,
        "old_symbol",
        "new_symbol",
        "process_date",
        "last_traded",
        "lag_days",
    ]
    assert rows(result) == [
        {
            "key": f"AAA/AAA3:{iso(last + timedelta(days=208))}",
            "in_liquid": True,
            "excepted": False,
            "old_symbol": "AAA",
            "new_symbol": "AAA3",
            "process_date": last + timedelta(days=208),
            "last_traded": last,
            "lag_days": 208,
        },
        {
            "key": f"CCC/CCC3:{iso(last + timedelta(days=31))}",
            "in_liquid": False,
            "excepted": False,
            "old_symbol": "CCC",
            "new_symbol": "CCC3",
            "process_date": last + timedelta(days=31),
            "last_traded": last,
            "lag_days": 31,
        },
    ]
    assert result.notes == {
        "le0": 1,
        "d1_10": 1,
        "d11_30": 1,
        "d31_180": 1,
        "gt180": 1,
        "no_old_bars": 1,
    }
    assert result.fails("all") is False


def test_n1_last_traded_ignores_placeholders(con: duckdb.DuckDBPyConnection, world: World) -> None:
    for d in DAYS[-5:]:
        world.bars = replace(world.bars, "CCC", d, flat("CCC", d, 10.0))
    world.name_changes = [rename("CCC", "CCC2", DAYS[-6] + timedelta(days=31))]
    [row] = rows(world.audit(con, checks=["N1"])["N1"])
    assert (row["last_traded"], row["lag_days"]) == (DAYS[-6], 31)


def test_n2_both_names_trade_after_the_process_date(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    cut = DAYS[10]
    world.name_changes = [rename("ROLD", "RNEW", cut)]
    world.bars += series("ROLD", DAYS[:13], 20.0, 3_000)  # trades through DAYS[12]
    world.bars += series("RNEW", DAYS[11:], 20.0, 3_000, start=11)  # first bar after the cut

    result = world.audit(con, checks=["N2"])["N2"]

    assert result.rows.schema.names == [
        *HEAD,
        "old_symbol",
        "new_symbol",
        "process_date",
        "old_last_traded",
        "new_first_traded",
    ]
    assert rows(result) == [
        {
            "key": f"ROLD/RNEW:{iso(cut)}",
            "in_liquid": False,
            "excepted": False,
            "old_symbol": "ROLD",
            "new_symbol": "RNEW",
            "process_date": cut,
            "old_last_traded": DAYS[12],
            "new_first_traded": DAYS[11],
        }
    ]
    assert result.fails("all") is False


def test_n2_clean_shapes(con: duckdb.DuckDBPyConnection, world: World) -> None:
    cut = DAYS[10]
    world.name_changes = [rename("ROLD", "RNEW", cut)]
    # the usual: old stops before the cut, new starts on it
    world.bars = (
        base_bars()
        + series("ROLD", DAYS[:10], 20.0, 3_000)
        + series("RNEW", DAYS[10:], 20.0, 3_000, start=10)
    )
    assert rows(world.audit(con, checks=["N2"])["N2"]) == []
    # old keeps trading, but new began ON the process date (not after): no row
    world.bars = (
        base_bars()
        + series("ROLD", DAYS[:13], 20.0, 3_000)
        + series("RNEW", DAYS[10:], 20.0, 3_000, start=10)
    )
    assert rows(world.audit(con, checks=["N2"])["N2"]) == []
    # old only has placeholders after the cut (VTIQ): no row
    world.bars = (
        base_bars()
        + series("ROLD", DAYS[:10], 20.0, 3_000)
        + series("RNEW", DAYS[11:], 20.0, 3_000, start=11)
    )
    world.bars += [flat("ROLD", d, 20.0) for d in DAYS[10:13]]
    assert rows(world.audit(con, checks=["N2"])["N2"]) == []


def test_n3_partial_duplicates_between_clearly_relabelled_and_accidental(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    aaa = series("AAA", DAYS, 100.0, 30_000)
    # BBB: 20 traded days, 5 of them AAA's bars -> 5 / 20 = 0.25
    world.bars = drop(world.bars, "BBB", DAYS[20:])
    for d in DAYS[:5]:
        world.bars = replace(world.bars, "BBB", d, {**find(aaa, "AAA", d), "symbol": "BBB"})
    # CCC: 2 of 25 -> 0.08 (accident); DDD2: a full copy of DDD -> 1.0 (relabel). CCC copies
    # days BBB does not, or BBB/CCC would be a third pair (2 / 20 = 0.1)
    for d in DAYS[5:7]:
        world.bars = replace(world.bars, "CCC", d, {**find(aaa, "AAA", d), "symbol": "CCC"})
    ddd = series("DDD", DAYS, 7.0, 4_000)
    world.bars += ddd + relabel(ddd, "DDD2")

    result = world.audit(con, checks=["N3"])["N3"]

    assert result.rows.schema.names == [
        *HEAD,
        "symbol_a",
        "symbol_b",
        "identical_days",
        "shorter_days",
        "ratio",
    ]
    [row] = rows(result)
    assert row["ratio"] == pytest.approx(0.25)
    del row["ratio"]
    assert row == {
        "key": "AAA/BBB",
        "in_liquid": True,
        "excepted": False,
        "symbol_a": "AAA",
        "symbol_b": "BBB",
        "identical_days": 5,
        "shorter_days": 20,
    }
    assert result.notes == {"lt10": 1, "p10_50": 1, "p50_90": 0, "ge90": 1}
    assert result.fails("all") is False


def test_n3_clean_store_has_no_pairs(con: duckdb.DuckDBPyConnection, world: World) -> None:
    result = world.audit(con, checks=["N3"])["N3"]
    assert rows(result) == []
    assert result.notes == {"lt10": 0, "p10_50": 0, "p50_90": 0, "ge90": 0}


# =============================================================================================
# C: coverage per year
# =============================================================================================


def test_c1_coverage_per_year(con: duckdb.DuckDBPyConnection, world: World) -> None:
    rename_world(world, old_close=20.0, new_close=20.0)
    world.liquid_ids.add(SID_R)
    world.name_changes = [rename("ROLD", "RNEW", DAYS[10]), rename("CCC", "CCC2", DAYS[12])]
    world.bars.append(bar("AAA", date(2019, 12, 31), close=100.0, volume=29_000))

    result = world.audit(con, checks=["C1"])["C1"]

    assert result.rows.schema.names == [
        *HEAD,
        "year",
        "bars",
        "symbols",
        "liquid_symbols",
        "name_changes",
        "name_changes_linked",
    ]
    assert rows(result) == [
        {
            "key": "2019",
            "in_liquid": False,
            "excepted": False,
            "year": 2019,
            "bars": 1,
            "symbols": 1,
            "liquid_symbols": 3,  # AAA, BBB and ROLD's open-ended-in-2019 segment
            "name_changes": 0,
            "name_changes_linked": 0,
        },
        {
            "key": "2020",
            "in_liquid": False,
            "excepted": False,
            "year": 2020,
            "bars": 75 + 10 + 15,
            "symbols": 5,
            "liquid_symbols": 4,
            "name_changes": 2,
            "name_changes_linked": 1,
        },
    ]
    assert result.notes == {"name_changes": 2, "name_changes_linked": 1}
    assert result.fails("all") is False


def test_c1_clean_world(con: duckdb.DuckDBPyConnection, world: World) -> None:
    [row] = rows(world.audit(con, checks=["C1"])["C1"])
    assert (row["bars"], row["symbols"], row["liquid_symbols"]) == (75, 3, 2)
    assert (row["name_changes"], row["name_changes_linked"]) == (0, 0)


# =============================================================================================
# leakage
# =============================================================================================


def comparable(results: dict[str, CheckResult]) -> dict[str, tuple[list[dict], dict]]:
    return {k: (rows(r), r.notes) for k, r in results.items()}


def test_master_segment_not_yet_knowable_leaves_the_audit_unchanged(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    # AAAX is a second label of S000001 dated DAYS[5], filed on DAYS[20]. Before the filing its
    # bars are orphans (I1) and there is no double day (I3); after it, the reverse. At the early
    # as_of the audit must be identical to one run on a master that does not have the row at
    # all -- for every check (M4 aside: whether a class row of an unknowable segment is "a class
    # row without a segment" is left to the implementation).
    first, filed = DAYS[5], DAYS[20]
    world.segs.append(Seg(SID_AAA, "AAAX", first, None, known(filed)))
    world.bars += series("AAAX", DAYS[5:8], 90.0, 7_000, start=5)
    early = avail(DAYS[15])
    assert early < known(filed)

    with_future = world.audit(con, as_of=early)
    world.segs = base_segs()
    without = world.audit(con, as_of=early)

    assert {k: v for k, v in comparable(with_future).items() if k != "M4"} == {
        k: v for k, v in comparable(without).items() if k != "M4"
    }
    assert rows(with_future["I3"]) == []
    assert [(r["symbol"], r["bars"]) for r in rows(with_future["I1"])] == [("AAAX", 3)]

    world.segs = base_segs() + [Seg(SID_AAA, "AAAX", first, None, known(filed))]
    late = world.audit(con, as_of=known(filed))
    assert [(r["key"], r["days"], r["identical_days"]) for r in rows(late["I3"])] == [
        (SID_AAA, 3, 0)
    ]
    assert rows(late["I1"]) == []


def test_bars_and_splits_not_yet_available_are_not_audited(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    # a duplicate on the last session and a split dated the day after: both on disk, neither
    # knowable at the evening of the second-to-last session
    world.bars.append(dict(find(world.bars, "AAA", DAYS[-1])))
    after = DAYS[-1] + timedelta(days=1)
    world.splits = [split("NOPE", after, 1.0, 2.0)]
    before = avail(DAYS[-2])

    early = world.audit(con, as_of=before, checks=["P1", "S1", "I6", "I7"])
    assert rows(early["P1"]) == [] and rows(early["S1"]) == []
    assert rows(early["I6"])[0]["bars"] == 72
    assert early["I7"].notes["sessions"] == 24

    on_time = world.audit(con, as_of=avail(DAYS[-1]), checks=["P1", "S1", "I6"])
    assert keys(on_time["P1"]) == [f"AAA:{iso(DAYS[-1])}"]
    assert rows(on_time["S1"]) == []  # the split's available_at (00:00 NY of 02-07) is later
    assert rows(on_time["I6"])[0]["bars"] == 76
    # known at 00:00 of its ex-date, but that day's bars are not: no finding until the session
    # is visible (another symbol's bar on that day), then NOPE's missing bar is one
    assert rows(world.audit(con, as_of=known(after), checks=["S1"])["S1"]) == []
    world.bars.append(bar("AAA", after, close=100.0, volume=31_000))
    assert keys(world.audit(con, as_of=avail(after), checks=["S1"])["S1"]) == [f"NOPE:{iso(after)}"]


def test_end_of_a_segment_not_yet_knowable_keeps_the_old_label_resolved(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    # the I9 / M3 / I4 family all read the master as of ``as_of``: with LAC's end still unknown
    # on DAYS[13], LAAC does not exist and no segment pair is examined
    lac_world(world)
    results = world.audit(con, as_of=avail(DAYS[13]), checks=["I4", "I9", "M3", "I3"])
    assert all(rows(r) == [] for r in results.values())
    results = world.audit(con, as_of=known(DAYS[14]), checks=["I4", "I9", "M3", "I3"])
    assert keys(results["I4"]) == ["LAC"]
    assert rows(results["I9"]) == [] and rows(results["M3"]) == [] and rows(results["I3"]) == []
