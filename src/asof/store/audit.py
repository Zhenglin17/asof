"""Store audit: one number per known kind of data accident, over the store as known at ``as_of``.

An operational view like ``bars_status``; strategy code must not import it. The tests of the
ingest code only prove that the code does what its author imagined (4b-2 had 347 green tests
and wrong data); this module looks at the real files for the shapes we have actually met --
BK's history filed under BNY, a split with no price move, a one-day 40x print on a warrant, a
rename filed weeks after the old name stopped trading -- and counts them.

Every check yields rows keyed by a short string (``AAA:2020-01-07``, ``S000730``, ``AAA/BBB``)
so a human can except one in ``configs/audit_exceptions.yaml`` with a reason. A row is *in the
liquid scope* when the security, or any segment of the symbol, belongs to the liquid-tier union
the caller passes in; split records (S1 / S2) are scoped by date instead when the caller passes
the monthly tier itself: the security that owned the record on its ex-date must be in the tier
within ``SPLIT_LIQUID_DAYS`` of it. Gated checks fail the run when rows are left outside the
exceptions: ``gate="all"`` anywhere, ``gate="liquid"`` inside the liquid scope (or anywhere with
``tier="all"``). Ungated checks are information: they are read before deciding whether to gate.

Three readings of the store are used, all as of ``as_of``:

* raw: every stored bar whose ``available_at`` is not after ``as_of`` (``visible_bars`` with
  ``resolve=False``, plus the hive ``year`` the file sits under);
* resolved: traded bars attributed to securities through ``visible_master``
  (``visible_bars`` with the defaults), the view every strategy reads;
* the master as known at ``as_of`` (``visible_master``): an end not yet knowable is open.
"""

import os
import re
import statistics
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

import duckdb
import exchange_calendars as xc
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from asof.ingest.bars import BAR_SCHEMA
from asof.ingest.corporate_actions import NAME_CHANGE_SCHEMA
from asof.ingest.instrument_class import CLASS_FILE, SHELF_MIN_TICKERS
from asof.ingest.sec_ticker_history import (
    EXCHANGE_HISTORY_PATH,
    EXCHANGE_HISTORY_SCHEMA,
    HISTORY_SCHEMA,
)
from asof.ingest.security_master import (
    CONFLICT_SCHEMA,
    CONFLICTS_FILE,
    COPY_MIN_DAYS,
    DUP_MIN_DAYS,
    DUP_MIN_VOLUME,
    MASTER_FILE,
    SEC_HISTORY_START,
    WHEN_ISSUED,
    drop_table_corrections,
    normalize_spelling,
    traded_runs,
)
from asof.ingest.tickers import is_valid_symbol
from asof.ingest.universe import ASSET_SCHEMA, assets_snapshots
from asof.store.market import (
    _alpaca_files,
    _ext_files,
    _source,
    bars_source_sql,
    visible_bars,
    visible_dividends,
    visible_master,
    visible_sessions,
    visible_splits,
)
from asof.store.shares import implausible, market_caps, share_facts, visible_shares
from asof.store.spinoffs import spinoff_events
from asof.store.universe import LIQUID_CLASSES

ET = ZoneInfo("America/New_York")

Gate = Literal["liquid", "all"] | None
Tier = Literal["liquid", "all"]

REPORT_DIR = "audit"

# key shapes
_SYM = r"[A-Z0-9.]+"
_SID = r"S\d+"
_DATE = r"\d{4}-\d{2}-\d{2}"
_YEAR = r"\d{4}"
_KIND = r"[a-z_]+"

# thresholds (design 2026-10-07 §8; reviewer follow-ups 2026-10-07/08)
GAP_SESSIONS = 10  # I8: a security's traded sessions further apart than this
CONTINUOUS = (0.8, 1.25)  # I9: close after / close before across a segment change
CONTINUOUS_DAYS = 10  # I9: calendar days between the two closes
NEIGHBOUR_SESSIONS = 5  # S2-S4: how far back / ahead a "previous" / "next" traded bar may be
SPLIT_WINDOW = 3  # S3, S4: a split this many sessions away explains a jump
SPLIT_TOLERANCE = 0.35  # S2: observed open / prev close vs 1 / factor
JUMP = 3.0  # S3, S4: ratio to the previous close that counts as a jump
BAD_BAR_MAX_VOLUME = 50_000  # S3: thin
BAD_BAR_RETURN = (0.5, 1.5)  # S3: next close / prev close, i.e. the price came back
SPARSE_SESSION = 0.1  # I7: traded symbols below this share of the median
RENAME_LAG_DAYS = 30  # N1: rows beyond this many days between last old bar and the record
RELABEL_BAND = (0.1, 0.9)  # N3: identical-day share between "accident" and "relabelled"
STALE_SESSIONS = 10  # C2: a listed symbol whose last bar is further behind the store's end
COPY_SHARE_WARN = 0.10  # M7: share of an old-holder segment's days copied from the latest holder
SPLIT_LIQUID_DAYS = 92  # S1, S2: tier as_of this many days either side of the ex-date gates it
SPLIT_TRADED_SESSIONS = 20  # S1: no traded bar this many sessions before (or after, if it ended)
SPLIT_LATE_SESSIONS = 2  # S2: the move may land this many sessions after the ex-date (ULTY: 1)
DIVIDEND_TWIN_TOLERANCE = 1e-5  # D1: relative gap within which two rates are one payout
# D1: or this absolute gap -- a small rate stored at 6 and 7 decimals (ABEV 2025-12-22: 0.012354 /
# 0.012355); real gaps are <= 1.1e-6 or >= 8e-6 (DDHRF 2020-10-29, two figures)
DIVIDEND_TWIN_ABS = 1.5e-6
CALENDAR = "XNYS"  # I12: the exchange calendar sessions are checked against
REVIEWED_RESOLUTIONS = ("unresolved", "relabelled")  # M6: conflicts a human reads once
I10_MIN_SNAPSHOTS = 3  # I10: exchange snapshots that make a listing (fewer: stale or not started)
# I10: the SEC exchange table's spellings of a listing; "OTC" and "" are none
LISTING_EXCHANGES = ("NASDAQ", "NYSE", "CBOE", "CBOE US", "NAS")
# F1: a count older than this is missing; a 20-F count is a year old when the next one is filed
# (TSM: as of 2023-12-31, filed 2024-04-18), so two years leave room for a late filer
SHARE_MAX_AGE_DAYS = 730
SHARE_MAX_TURNOVER = 1.0  # F2: average dollar volume / cap above this is no real count
SHARE_MAX_CAP = 5e12  # F2: a cap above this is no real count (the largest, 2025: ~4.5e12)
SCAN_CLOSE = time(20)  # F2: a night's caps are read at its scan time


@dataclass(frozen=True)
class CheckSpec:
    id: str
    title: str
    gate: Gate
    key_pattern: str


def _spec(id: str, title: str, gate: Gate, key_pattern: str) -> tuple[str, CheckSpec]:
    return id, CheckSpec(id, title, gate, key_pattern)


CHECKS: dict[str, CheckSpec] = dict(
    [
        _spec("P1", "duplicate (symbol, t) rows", "all", f"{_SYM}:{_DATE}"),
        _spec("P2", "impossible prices", "all", f"{_SYM}:{_DATE}"),
        _spec("P3", "available_at after fetched_at", "all", f"{_SYM}:{_DATE}"),
        _spec("P4", "bar filed under another year's partition", "all", f"{_SYM}:{_DATE}"),
        _spec("P5", "volume 0 but prices not flat", "all", f"{_SYM}:{_DATE}"),
        _spec(
            "P6",
            "daily bar not knowable at 20:00 New York of its session",
            "all",
            f"{_SYM}:{_DATE}",
        ),  # fmt: skip
        _spec(
            "M1",
            "a symbol's next segment known before its previous one's end",
            "all",
            f"{_SYM}:{_DATE}",
        ),  # fmt: skip
        _spec("M2", "segments of one security overlap", "liquid", f"{_SID}:{_DATE}"),
        _spec("M3", "gap between segments of one security", None, f"{_SID}:{_DATE}"),
        _spec("M4", "master and class table disagree", "all", f"{_SYM}:{_DATE}"),
        _spec("M5", "instrument class unknown", "liquid", f"{_SYM}:{_DATE}"),
        _spec(
            "M6",
            "identity conflicts awaiting a human (unresolved, relabelled)",
            "liquid",
            f"{_SYM}(/{_SYM})*:{_KIND}:{_DATE}(,{_DATE})*",
        ),  # fmt: skip
        _spec(
            "M7",
            "old-holder segment partly copies of the label's latest holder",
            "liquid",
            f"{_SID}:{_SYM}:{_DATE}",
        ),  # fmt: skip
        _spec(
            "M8",
            "identity decisions settled by the build",
            None,
            f"{_SYM}(/{_SYM})*:{_KIND}:{_DATE}(,{_DATE})*",
        ),  # fmt: skip
        _spec("M9", "fund-class segment of a company filer", "liquid", f"{_SYM}:{_DATE}"),
        _spec("I1", "traded bars no segment owns (loss)", None, _SYM),
        _spec("I2", "two securities share an identical bar", "liquid", f"{_SYM}/{_SYM}"),
        _spec("I3", "one security traded under two symbols on one day", "liquid", _SID),
        _spec("I4", "old label traded between its cut and the day the cut was known", None, _SYM),
        _spec("I5", "segment CIK contradicted by the SEC table", None, f"{_SYM}:{_DATE}"),
        _spec("I6", "volume-0 placeholder bars per year", None, _YEAR),
        _spec("I7", "session with almost nothing traded", "all", _DATE),
        _spec("I8", "security silent for more than 10 sessions", None, f"{_SID}:{_DATE}"),
        _spec("I9", "price discontinuity across a segment change", None, f"{_SID}:{_DATE}"),
        _spec("I10", "exchange-listed ticker with no bars of its own", None, rf"{_SYM}:\d+"),
        _spec("I12", "session calendar disagrees with NYSE", "all", _DATE),
        _spec("S1", "split record with no bar on its ex-date", "liquid", f"{_SYM}:{_DATE}"),
        _spec("S2", "split record without the price move", "liquid", f"{_SYM}:{_DATE}"),
        _spec("S3", "one-day thin-volume spike, no split", "liquid", f"{_SYM}:{_DATE}"),
        _spec("S4", "close jump of 3x without a split", None, f"{_SYM}:{_DATE}"),
        # gated everywhere, not only in the liquid tier: every reader multiplies the events of a
        # day, so a wrong one silently rescales any position held across it
        _spec("S5", "several split events on one day", "all", f"{_SYM}:{_DATE}"),
        # listed, not gated: the event is already read as a restart, nothing is rescaled by it
        _spec("S6", "spin-off factor the prices disprove (restarted)", None, f"{_SYM}:{_DATE}"),
        # listed, not gated: two real payouts of one amount on one day (F 2025-02-18, regular
        # and supplemental) look the same; only the benchmark reads dividends
        _spec("D1", "dividend records differing only by rounding", None, f"{_SYM}:{_DATE}"),
        _spec(
            "N1", "rename recorded long after the old name stopped", None, f"{_SYM}/{_SYM}:{_DATE}"
        ),  # fmt: skip
        _spec("N2", "both names traded after the rename date", None, f"{_SYM}/{_SYM}:{_DATE}"),
        _spec("N3", "symbol pairs partly identical", None, f"{_SYM}/{_SYM}"),
        _spec("C1", "coverage per year", None, _YEAR),
        _spec("C2", "listed asset with stale bars", "all", _SYM),
        # listed, not gated: a stock without a usable count (F1) or with an implausible one (F2)
        # drops out of a cap filter, which reports how many it dropped
        _spec("F1", "liquid security without a visible share count", None, _SYM),
        _spec("F2", "liquid security with an implausible market cap", None, f"{_SYM}:{_DATE}"),
    ]
)


# -- exceptions ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditException:
    check: str
    key: str
    reason: str
    added: date
    by: str


# share counts from these forms mark an operating company (M9); funds file N-CSR / 485BPOS
COMPANY_FORMS = frozenset({"10-K", "10-K/A", "10-Q", "10-Q/A", "20-F", "20-F/A", "40-F", "40-F/A"})
_EXCEPTION_FIELDS = ("check", "key", "reason", "added", "by")


def load_exceptions(path: Path) -> list[AuditException]:
    """``configs/audit_exceptions.yaml``: a list of ``check / key / reason / added / by``.

    Every entry is validated: a known check, a key of that check's shape, an ISO date, nothing
    empty and nothing extra. A missing file is an empty list.
    """
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text())
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a list of exceptions")
    loaded: list[AuditException] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError(f"{path}: entry {item!r} is not a mapping")
        if set(item) != set(_EXCEPTION_FIELDS):
            raise ValueError(f"{path}: entry {item!r} must have exactly {_EXCEPTION_FIELDS}")
        if any(item[name] is None or str(item[name]).strip() == "" for name in _EXCEPTION_FIELDS):
            raise ValueError(f"{path}: entry {item!r} has an empty field")
        check, key = str(item["check"]), str(item["key"])
        if check not in CHECKS:
            raise ValueError(f"{path}: entry {item!r}: unknown check {check!r}")
        if not re.fullmatch(CHECKS[check].key_pattern, key):
            raise ValueError(
                f"{path}: entry {item!r}: key {key!r} does not match {CHECKS[check].key_pattern}"
            )
        added = item["added"]
        if isinstance(added, datetime) or not isinstance(added, date):
            try:
                added = date.fromisoformat(str(added))
            except ValueError:
                raise ValueError(
                    f"{path}: entry {item!r}: added {item['added']!r} is not YYYY-MM-DD"
                ) from None
        loaded.append(AuditException(check, key, str(item["reason"]), added, str(item["by"])))
    return loaded


# -- results -------------------------------------------------------------------------------------


@dataclass
class CheckResult:
    spec: CheckSpec
    rows: pa.Table  # key, in_liquid, excepted, then the check's own columns; sorted by key
    notes: dict[str, int | float | str | None]

    def _count(self, liquid_only: bool) -> int:
        n = 0
        for in_liquid, excepted in zip(
            self.rows.column("in_liquid").to_pylist(),
            self.rows.column("excepted").to_pylist(),
            strict=True,
        ):
            if not excepted and (in_liquid or not liquid_only):
                n += 1
        return n

    @property
    def liquid(self) -> int:
        return self._count(liquid_only=True)

    @property
    def total(self) -> int:
        return self._count(liquid_only=False)

    @property
    def excepted(self) -> int:
        return sum(1 for e in self.rows.column("excepted").to_pylist() if e)

    def fails(self, tier: Tier) -> bool:
        if self.spec.gate is None:
            return False
        if self.spec.gate == "all" or tier == "all":
            return self.total > 0
        return self.liquid > 0


def stale_exceptions(
    results: Sequence[CheckResult], exceptions: Sequence[AuditException]
) -> list[AuditException]:
    """Exceptions for a check that ran and matched no row: the defect is gone (or the key was
    mistyped), so the entry should go too."""
    keys = {r.spec.id: set(r.rows.column("key").to_pylist()) for r in results}
    return [e for e in exceptions if e.check in keys and e.key not in keys[e.check]]


def write_report(results: Sequence[CheckResult], root: Path) -> Path:
    """``<root>/audit/<id>.parquet`` per result, empty ones too, each replaced atomically."""
    directory = root / REPORT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    for result in results:
        path = directory / f"{result.spec.id}.parquet"
        tmp = path.with_name(path.name + ".tmp")
        try:
            pq.write_table(result.rows, tmp)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
    return directory


# -- the run -------------------------------------------------------------------------------------

_RAW = "_audit_raw"
_RES = "_audit_res"
_MASTER = "_audit_master"
_MASTER_FILE = "_audit_master_file"
_CLASS = "_audit_class"
_SPLITS = "_audit_splits"
_SPLIT_DAYS = "_audit_split_days"
_NAMES = "_audit_names"
_SEC = "_audit_sec"
_LIQ_IDS = "_audit_liquid_ids"
_QUAL_IDS = "_audit_qualified_ids"
_LIQ_SYMS = "_audit_liquid_symbols"
_SESSIONS = "_audit_sessions"
_TRADED = "_audit_traded"
_ORPHANS = "_audit_orphans"
_ALPACA_KEYS = "_audit_alpaca_keys"  # (symbol, t) of every Alpaca row, imported rows aside
_LISTED = "_audit_listed"  # the Alpaca listing as of the audit (C2)
_TIER = "_audit_tier"  # (security_id, day): liquid-tier rows knowable at as_of (S1, S2)
_EXCHANGES = "_audit_exchanges"  # SEC exchange snapshots knowable at as_of (I10)
_SPLIT_OWNERS = "_audit_split_owners"  # split record -> security owning its symbol that day
_DIVIDENDS = "_audit_dividends"  # dividends as visible_dividends returns them (D1)
_CAL = "_audit_calendar"  # exchange sessions over the stored span (I12)

_REGISTERED = (
    _RES,
    _MASTER,
    _MASTER_FILE,
    _CLASS,
    _SPLITS,
    _NAMES,
    _SEC,
    _LIQ_IDS,
    _QUAL_IDS,
    _LISTED,
    _TIER,
    _EXCHANGES,
    _DIVIDENDS,
    _CAL,
)
_TEMP_TABLES = (_LIQ_SYMS, _SESSIONS, _TRADED, _ORPHANS, _SPLIT_DAYS, _SPLIT_OWNERS)
_TEMP_VIEWS = (_RAW, _ALPACA_KEYS)


def _liq_sym(column: str) -> str:
    return f"coalesce({column} IN (SELECT symbol FROM {_LIQ_SYMS}), false)"


def _liq_sid(column: str) -> str:
    return f"coalesce({column} IN (SELECT security_id FROM {_LIQ_IDS}), false)"


def _day(column: str) -> str:
    return f"CAST({column} AS VARCHAR)"


def _ny_date(column: str) -> str:
    return f"CAST(timezone('America/New_York', {column}) AS DATE)"


class _Audit:
    def __init__(self, con: duckdb.DuckDBPyConnection, root: Path, as_of: datetime) -> None:
        self.con = con
        self.root = root
        self.as_of = as_of
        self._traded_ready = False
        self.snapshot_date: date | None = None  # the listing C2 used; set by load()

    # -- setup -----------------------------------------------------------------------------------

    def load(
        self,
        liquid_ids: Collection[str],
        qualified_ids: Collection[str],
        liquid_tier: pa.Table | None = None,
    ) -> None:
        con, root, as_of = self.con, self.root, self.as_of
        # DuckDB caches file contents by path and mtime; a file rewritten within the same second
        # (a rebuild right before the audit) would be read stale
        con.execute("SET enable_external_file_cache = false")
        if not (root / CLASS_FILE).exists():
            raise FileNotFoundError(f"{root / CLASS_FILE}: build it with `asof market instruments`")
        master = visible_master(con, root, as_of)  # FileNotFoundError when absent
        resolved = visible_bars(con, root, "1Day", as_of)
        splits = visible_splits(con, root, as_of)
        con.execute("SET TimeZone = 'UTC'")

        con.register(_MASTER, master)
        con.register(_MASTER_FILE, pq.read_table(root / MASTER_FILE))
        con.register(_CLASS, pq.read_table(root / CLASS_FILE))
        con.register(_RES, resolved)
        con.register(_SPLITS, splits)
        con.register(_NAMES, self._names())
        con.register(_LIQ_IDS, pa.table({"security_id": pa.array(sorted(liquid_ids), pa.string())}))
        con.register(
            _QUAL_IDS, pa.table({"security_id": pa.array(sorted(qualified_ids), pa.string())})
        )

        files, ext_files = _alpaca_files(root, "1Day"), _ext_files(root, "1Day")
        if files or ext_files:
            # one bar per (symbol, t): an imported row replaces the Alpaca row of its key
            stamp = f"TIMESTAMPTZ '{as_of.isoformat()}'"
            con.execute(
                f"CREATE OR REPLACE TEMP VIEW {_RAW} AS "
                f"SELECT * FROM ({bars_source_sql(files, ext_files, stamp)})"
            )
        else:
            empty = (
                BAR_SCHEMA.append(pa.field("year", pa.int64()))
                .append(pa.field("source", pa.string()))
                .empty_table()
            )
            con.register("_audit_raw_empty", empty)
            con.execute(f"CREATE OR REPLACE TEMP VIEW {_RAW} AS SELECT * FROM _audit_raw_empty")
        con.register(_SEC, self._sec())
        if files:
            con.execute(
                f"CREATE OR REPLACE TEMP VIEW {_ALPACA_KEYS} AS "
                f"SELECT symbol, t FROM {_source(files)}"
            )
        else:
            con.register(
                "_audit_alpaca_keys_empty", BAR_SCHEMA.empty_table().select(["symbol", "t"])
            )
            con.execute(
                f"CREATE OR REPLACE TEMP VIEW {_ALPACA_KEYS} AS "
                "SELECT * FROM _audit_alpaca_keys_empty"
            )
        con.register(_LISTED, self._listed())
        con.register(_EXCHANGES, self._exchanges())
        # a tier row computed after as_of is not knowable yet
        self.dated_tier = liquid_tier is not None
        self.liquid_tier = liquid_tier
        self.liquid_ids = frozenset(liquid_ids)
        tier = [
            {"security_id": r["security_id"], "day": r["as_of"].astimezone(ET).date()}
            for r in (liquid_tier.to_pylist() if liquid_tier is not None else [])
            if r["class"] in LIQUID_CLASSES and r["as_of"] <= as_of
        ]
        con.register(
            _TIER,
            pa.Table.from_pylist(
                tier, schema=pa.schema([("security_id", pa.string()), ("day", pa.date32())])
            ),
        )

        con.execute(
            f"CREATE OR REPLACE TEMP TABLE {_LIQ_SYMS} AS SELECT DISTINCT symbol FROM {_MASTER} "
            f"WHERE security_id IN (SELECT security_id FROM {_LIQ_IDS})"
        )
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE {_SESSIONS} AS
            SELECT session_date,
                   row_number() OVER o AS idx,
                   coalesce(lag(session_date, {SPLIT_WINDOW}) OVER o, DATE '1900-01-01') AS lo,
                   coalesce(lead(session_date, {SPLIT_WINDOW}) OVER o, DATE '2999-12-31') AS hi
            FROM (SELECT DISTINCT session_date FROM {_RAW})
            WINDOW o AS (ORDER BY session_date)
            """
        )
        con.execute(
            f"CREATE OR REPLACE TEMP TABLE {_SPLIT_DAYS} AS "
            f"SELECT DISTINCT symbol, ex_date FROM {_SPLITS}"
        )
        self.liquid_symbols = {
            row[0] for row in con.execute(f"SELECT symbol FROM {_LIQ_SYMS}").fetchall()
        }
        # The security owning a split record's symbol on its ex-date: the segment covering the
        # day, else the one ending that day (UTX's record of 2020-04-03, the day it became RTX).
        # The session index is the first session on or after the ex-date.
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE {_SPLIT_OWNERS} AS
            SELECT sp.symbol, sp.ex_date,
                   (SELECT m.security_id FROM {_MASTER} m
                    WHERE m.symbol = sp.symbol AND m.valid_from <= sp.ex_date
                      AND (m.valid_to IS NULL OR sp.ex_date <= m.valid_to)
                    ORDER BY (m.valid_to = sp.ex_date) NULLS FIRST, m.valid_from DESC
                    LIMIT 1) AS security_id,
                   (SELECT min(idx) FROM {_SESSIONS} s WHERE s.session_date >= sp.ex_date)
                       AS ex_idx
            FROM {_SPLIT_DAYS} sp
            """
        )

    def _exchanges(self) -> pa.Table:
        """SEC exchange snapshots knowable at ``as_of`` (from 00:00 New York of their date)."""
        path = self.root / EXCHANGE_HISTORY_PATH
        if not path.exists():
            return EXCHANGE_HISTORY_SCHEMA.empty_table()
        table = pq.read_table(path)
        keep = [
            datetime(d.year, d.month, d.day, tzinfo=ET) <= self.as_of
            for d in table.column("snapshot_date").to_pylist()
        ]
        return table.filter(pa.array(keep, pa.bool_()))

    def _names(self) -> pa.Table:
        path = self.root / "corporate_actions" / "name_changes.parquet"
        if not path.exists():
            return NAME_CHANGE_SCHEMA.empty_table()
        table = pq.read_table(path)
        keep = [a <= self.as_of for a in table.column("available_at").to_pylist()]
        return table.filter(pa.array(keep, pa.bool_()))

    def _sec(self) -> pa.Table:
        path = self.root / "symbols" / "sec_history.parquet"
        if not path.exists():
            return HISTORY_SCHEMA.empty_table()
        # the same preparation the master was built from: one spelling per ticker, and the
        # SEC's table clean-ups (AFC still listed under Allied Capital until 2020-07) dropped
        return drop_table_corrections(
            normalize_spelling(pq.read_table(path)), traded_runs(self.con, _RAW)
        )

    def _listed(self) -> pa.Table:
        """Alpaca's listing as of the audit: the newest assets snapshot dated on or before
        ``as_of`` (a snapshot is knowable from 00:00 New York of its date), kept to the symbols
        the universe would take: active, not OTC, a valid symbol. ``snapshot_date`` is None when
        no snapshot is usable."""
        self.snapshot_date = None
        usable: list[tuple[date, Path]] = []
        for path in assets_snapshots(self.root):
            try:
                day = date.fromisoformat(path.parent.name.split("=", 1)[1])
            except (IndexError, ValueError):
                continue
            if datetime(day.year, day.month, day.day, tzinfo=ET) <= self.as_of:
                usable.append((day, path))
        if not usable:
            return ASSET_SCHEMA.empty_table()
        self.snapshot_date, path = max(usable)
        table = pq.read_table(path)
        rows = [
            r
            for r in table.to_pylist()
            if r["status"] == "active" and r["exchange"] != "OTC" and is_valid_symbol(r["symbol"])
        ]
        return pa.Table.from_pylist(rows, schema=ASSET_SCHEMA)

    def close(self) -> None:
        for name in (
            *_REGISTERED,
            "_audit_raw_empty",
            "_audit_alpaca_keys_empty",
            "_audit_spinoffs",
        ):
            try:
                self.con.unregister(name)
            except duckdb.Error:
                pass
        for name in _TEMP_VIEWS:
            self.con.execute(f"DROP VIEW IF EXISTS {name}")
        for name in _TEMP_TABLES:
            self.con.execute(f"DROP TABLE IF EXISTS {name}")

    def _q(self, query: str, params: dict[str, Any] | None = None) -> pa.Table:
        return self.con.execute(query, params or {}).to_arrow_table()

    def _scalar(self, query: str) -> Any:
        row = self.con.execute(query).fetchone()
        return row[0] if row else None

    def _traded(self) -> str:
        """Traded raw bars with their session index and their neighbours: the previous and
        next traded bar (``prev_close`` / ``next_close``), and the previous bar of any kind
        (``prev_any_close``). A volume-0 placeholder carries the last close (real: EXEEW sat at
        105 for weeks before its 0.0101 print), so it is a fair "previous close" for a spike."""
        if not self._traded_ready:
            self.con.execute(
                f"""
                CREATE OR REPLACE TEMP TABLE {_TRADED} AS
                WITH a AS (
                    SELECT r.symbol, r.t, r.session_date, r.open, r.high, r.low, r.close,
                           r.volume, r.vwap, s.idx, s.lo, s.hi,
                           lag(r.close) OVER w AS prev_any_close, lag(s.idx) OVER w AS prev_any_idx
                    FROM {_RAW} r JOIN {_SESSIONS} s USING (session_date)
                    WINDOW w AS (PARTITION BY r.symbol ORDER BY r.t)
                )
                SELECT *, lag(close) OVER w AS prev_close, lag(idx) OVER w AS prev_idx,
                       lead(close) OVER w AS next_close, lead(idx) OVER w AS next_idx
                FROM a WHERE volume > 0
                WINDOW w AS (PARTITION BY symbol ORDER BY t)
                """
            )
            self._traded_ready = True
        return _TRADED

    # -- P: raw files ----------------------------------------------------------------------------

    def p1(self) -> tuple[pa.Table, dict]:
        return self._q(
            f"""
            SELECT symbol || ':' || {_day("min(session_date)")} AS key,
                   {_liq_sym("symbol")} AS _liq, symbol, t, count(*) AS copies
            FROM {_RAW} GROUP BY symbol, t HAVING count(*) > 1
            """
        ), {}

    def p2(self) -> tuple[pa.Table, dict]:
        return self._q(
            f"""
            SELECT symbol || ':' || {_day("session_date")} AS key, {_liq_sym("symbol")} AS _liq,
                   symbol, session_date, open, high, low, close, reason
            FROM (
                SELECT *, CASE
                    -- before the comparisons below: DuckDB orders NaN above every number, so a
                    -- NaN close read as 'high_below_body' and an infinite high passed
                    WHEN NOT coalesce(
                        isfinite(open) AND isfinite(high) AND isfinite(low)
                        AND isfinite(close) AND volume IS NOT NULL
                        -- vwap may be absent (imported bars) but never NaN: the liquid tier's
                        -- dollar volume is vwap x volume
                        AND (vwap IS NULL OR isfinite(vwap)),
                        false
                    ) THEN 'not_finite'
                    WHEN volume < 0 THEN 'negative_volume'
                    WHEN least(open, high, low, close) <= 0 THEN 'nonpositive'
                    WHEN high < low THEN 'high_below_low'
                    WHEN low > least(open, close) THEN 'low_above_body'
                    WHEN high < greatest(open, close) THEN 'high_below_body'
                END AS reason
                FROM {_RAW}
            ) WHERE reason IS NOT NULL
            """
        ), {}

    def _raw_rows(self, columns: str, where: str) -> pa.Table:
        return self._q(
            f"SELECT symbol || ':' || {_day('session_date')} AS key, {_liq_sym('symbol')} AS _liq, "
            f"{columns} FROM {_RAW} WHERE {where}"
        )

    def p3(self) -> tuple[pa.Table, dict]:
        return self._raw_rows(
            "symbol, session_date, available_at, fetched_at", "available_at > fetched_at"
        ), {}

    def p4(self) -> tuple[pa.Table, dict]:
        return self._raw_rows("symbol, session_date, year", "year <> year(session_date)"), {}

    def p5(self) -> tuple[pa.Table, dict]:
        return self._raw_rows(
            "symbol, session_date, open, high, low, close",
            "volume = 0 AND NOT (open = high AND high = low AND low = close)",
        ), {}

    def p6(self) -> tuple[pa.Table, dict]:
        # the rule every as_of filter rests on: a daily bar (its volume includes the after-hours
        # session) is final at 20:00 New York of its session date, and that date is the New York
        # date of its timestamp. Real 2026-10-08: 0 of 18,199,756 bars break it.
        expected = "timezone('America/New_York', session_date + INTERVAL 20 HOUR)"
        return self._raw_rows(
            f"symbol, session_date, t, available_at, {expected} AS expected_available_at",
            f"available_at <> {expected} OR session_date <> {_ny_date('t')}",
        ), {}

    # -- M: master and class table ---------------------------------------------------------------

    def m1(self) -> tuple[pa.Table, dict]:
        return self._q(
            f"""
            SELECT a.symbol || ':' || {_day("b.valid_from")} AS key, {_liq_sym("a.symbol")} AS _liq,
                   a.symbol, a.security_id AS old_security_id, b.security_id AS new_security_id,
                   b.valid_from, a.end_available_at, b.available_at
            FROM {_MASTER} a JOIN {_MASTER} b
              ON a.symbol = b.symbol AND a.valid_to <= b.valid_from  -- also across a gap
            WHERE b.available_at < a.end_available_at  -- a gap until the first bar is fine
            """
        ), {}

    _PAIRS = f"""
        WITH s AS (
            SELECT security_id, symbol AS old_symbol, valid_from AS old_from, valid_to AS old_to,
                   lead(symbol) OVER w AS new_symbol, lead(valid_from) OVER w AS new_from,
                   lead(valid_to) OVER w AS new_to
            FROM {_MASTER}
            WINDOW w AS (PARTITION BY security_id ORDER BY valid_from, symbol)
        ),
        pairs AS (SELECT * FROM s WHERE new_from IS NOT NULL)
    """

    def m2(self) -> tuple[pa.Table, dict]:
        return self._q(
            f"""
            {self._PAIRS}
            SELECT security_id || ':' || {_day("new_from")} AS key,
                   {_liq_sid("security_id")} AS _liq, security_id, old_symbol, new_symbol,
                   old_to AS old_valid_to, new_from AS new_valid_from
            FROM pairs
            WHERE new_symbol <> old_symbol AND (old_to IS NULL OR new_from < old_to)
            """
        ), {}

    def m3(self) -> tuple[pa.Table, dict]:
        table = self._q(
            f"""
            {self._PAIRS},
            gaps AS (
                SELECT * FROM pairs WHERE new_symbol <> old_symbol AND new_from > old_to
            )
            SELECT g.security_id || ':' || {_day("g.new_from")} AS key,
                   {_liq_sid("g.security_id")} AS _liq,
                   g.security_id, g.old_symbol, g.new_symbol, g.old_to AS old_valid_to,
                   g.new_from AS new_valid_from, g.new_from - g.old_to AS gap_days,
                   count(r.symbol) AS traded_bars_in_gap
            FROM gaps g LEFT JOIN {_RAW} r
              ON r.symbol IN (g.old_symbol, g.new_symbol) AND r.volume > 0
             AND r.session_date >= g.old_to AND r.session_date < g.new_from
            GROUP BY ALL
            """
        )
        return table, {"traded_bars_in_gaps": sum(table.column("traded_bars_in_gap").to_pylist())}

    def m4(self) -> tuple[pa.Table, dict]:
        # a class row is compared with the whole master file: a segment not yet knowable at a
        # historical as_of is not a reason to call its class row an orphan
        keys = (
            "x.security_id = y.security_id AND x.symbol = y.symbol AND x.valid_from = y.valid_from"
        )
        return self._q(
            f"""
            SELECT symbol || ':' || {_day("valid_from")} AS key, {_liq_sym("symbol")} AS _liq,
                   security_id, symbol, valid_from, side
            FROM (
                SELECT x.security_id, x.symbol, x.valid_from, 'master' AS side
                FROM {_MASTER} x ANTI JOIN {_CLASS} y ON {keys}
                UNION ALL
                SELECT x.security_id, x.symbol, x.valid_from, 'class' AS side
                FROM {_CLASS} x ANTI JOIN {_MASTER_FILE} y ON {keys}
            )
            """
        ), {}

    def m5(self) -> tuple[pa.Table, dict]:
        visible = (
            f"SELECT c.* FROM {_CLASS} c SEMI JOIN {_MASTER} m ON c.security_id = m.security_id "
            "AND c.symbol = m.symbol AND c.valid_from = m.valid_from"
        )
        table = self._q(
            f"""
            SELECT symbol || ':' || {_day("valid_from")} AS key,
                   coalesce(security_id IN (SELECT security_id FROM {_QUAL_IDS}), false) AS _liq,
                   security_id, symbol, valid_from, name, rule
            FROM ({visible}) WHERE "class" = 'unknown'
            """
        )
        liquid = f"FROM ({visible}) WHERE {_liq_sid('security_id')}"
        notes = {
            "liquid_lookup_sec_companies": int(
                self._scalar(f"SELECT count(*) {liquid} AND rule = 'lookup_sec_companies'")
            ),
            "liquid_name_from_sec_or_none_common_etf": int(
                self._scalar(
                    f"SELECT count(*) {liquid} AND name_source IN ('sec', 'none') "
                    "AND \"class\" IN ('common', 'etf')"
                )
            ),
            "symbols_with_mixed_classes": int(
                self._scalar(
                    f"SELECT count(*) FROM (SELECT symbol FROM {_CLASS} GROUP BY symbol "
                    'HAVING count(DISTINCT "class") > 1)'
                )
            ),
        }
        return table, notes

    def _conflict_rows(self, *, unresolved: bool) -> pa.Table:
        """The build's conflict table as audit rows: the ones awaiting a human (M6) or the ones
        the build settled (M8, carrying the resolution). ``relabelled`` waits for a human too:
        rule R merges two filers on the absence of the taker's own bars, which a coincidence of
        prices can also produce, so each case is read once and excepted with a reason. A
        conflicts file written before the ``resolution`` column existed reads as all
        unresolved."""
        path = self.root / CONFLICTS_FILE
        conflicts = pq.read_table(path) if path.exists() else CONFLICT_SCHEMA.empty_table()
        rows = []
        for c in conflicts.to_pylist():
            resolution = c.get("resolution") or "unresolved"
            if (resolution in REVIEWED_RESOLUTIONS) != unresolved:
                continue
            symbols = list(c["symbols"] or [])
            row = {
                # the dates make the key unique: ADRA has three cik_switch_discontinuous
                # conflicts, and one exception must not silence all three
                "key": "/".join(symbols)
                + ":"
                + c["kind"]
                + ":"
                + ",".join(d.isoformat() for d in c["dates"] or []),
                "_liq": any(s in self.liquid_symbols for s in symbols),
                "kind": c["kind"],
                "symbols": "/".join(symbols),
                "dates": ",".join(d.isoformat() for d in c["dates"] or []),
                "detail": c["detail"],
            }
            if not unresolved:
                row["resolution"] = resolution
            rows.append(row)
        fields = [
            ("key", pa.string()),
            ("_liq", pa.bool_()),
            ("kind", pa.string()),
            ("symbols", pa.string()),
            ("dates", pa.string()),
            ("detail", pa.string()),
        ]
        if not unresolved:
            fields.append(("resolution", pa.string()))
        return pa.Table.from_pylist(rows, schema=pa.schema(fields))

    def m6(self) -> tuple[pa.Table, dict]:
        return self._conflict_rows(unresolved=True), {}

    def m7(self) -> tuple[pa.Table, dict]:
        # A label held by two or more securities: how much of each earlier holder's segment is
        # OHLC-identical to the latest holder's chain on the same day. The build drops such a
        # segment at 90% (rule Z) and cuts copied tails (rule F); what is left above
        # COPY_SHARE_WARN is the next PARA / SPWR -- a history rewritten by the vendor that no
        # rule has caught yet.
        same = " AND ".join(f"c.{x} = t.{x}" for x in ("open", "high", "low", "close"))
        return self._q(
            f"""
            WITH latest AS (
                SELECT symbol, arg_max(security_id, valid_from) AS latest_security
                FROM {_MASTER} GROUP BY symbol HAVING count(DISTINCT security_id) > 1
            ), old AS (
                SELECT m.security_id, m.symbol, m.valid_from, m.valid_to, l.latest_security
                FROM {_MASTER} m JOIN latest l USING (symbol)
                WHERE m.security_id <> l.latest_security
            ), chain_bars AS (
                SELECT c.security_id AS latest_security, r.symbol, r.session_date,
                       r.open, r.high, r.low, r.close
                FROM {_RAW} r
                JOIN (SELECT DISTINCT security_id, symbol FROM {_MASTER}) c USING (symbol)
                WHERE r.volume > 0 AND c.security_id IN (SELECT latest_security FROM latest)
            ), traded AS (
                SELECT o.security_id, o.symbol, o.valid_from, o.valid_to, o.latest_security,
                       t.session_date, t.open, t.high, t.low, t.close
                FROM old o JOIN {_RAW} t ON t.symbol = o.symbol AND t.volume > 0
                 AND t.session_date >= o.valid_from
                 AND (o.valid_to IS NULL OR t.session_date < o.valid_to)
            ), scored AS (
                SELECT t.security_id, t.symbol, t.valid_from, t.valid_to, t.latest_security,
                       t.session_date,
                       EXISTS (SELECT 1 FROM chain_bars c
                               WHERE c.latest_security = t.latest_security
                                 AND c.session_date = t.session_date AND c.symbol <> t.symbol
                                 AND {same}) AS copied
                FROM traded t
            )
            SELECT security_id || ':' || symbol || ':' || {_day("valid_from")} AS key,
                   {_liq_sym("symbol")} AS _liq,
                   security_id, symbol, valid_from, valid_to, latest_security,
                   count(*) AS traded_days, count(*) FILTER (WHERE copied) AS copied_days,
                   count(*) FILTER (WHERE copied) / count(*) AS share
            FROM scored
            GROUP BY ALL
            HAVING count(*) FILTER (WHERE copied) >= {COPY_MIN_DAYS}
               AND count(*) FILTER (WHERE copied) / count(*) >= {COPY_SHARE_WARN}
            """
        ), {}

    def m8(self) -> tuple[pa.Table, dict]:
        return self._conflict_rows(unresolved=False), {}

    def m9(self) -> tuple[pa.Table, dict]:
        """Fund-class segments whose CIK files company reports: a segment tagged ``etf`` or
        ``leveraged_etf`` whose CIK, known by ``as_of``, has a share count from a 10-K, 10-Q,
        20-F or 40-F public by then. Facebook (CIK 1326801) was tagged etf because today's SEC
        fund table gives FB to a ProShares fund; PRTO's 2020 segment (Proteon, a 10-K filer)
        carries a later ETF's name. A bank's ETN shelf (a CIK listing ``SHELF_MIN_TICKERS``
        or more tickers in the SEC history up to ``as_of``) and a human override are exempt.

        Blind spots: a segment without a CIK, and a company whose CIKs the shares ingest never
        fetched (it reads the CIKs of the liquid common tier), give no row."""
        schema = pa.schema(
            [
                ("key", pa.string()),
                ("_liq", pa.bool_()),
                ("security_id", pa.string()),
                ("symbol", pa.string()),
                ("valid_from", pa.date32()),
                ("name", pa.string()),
                ("rule", pa.string()),
                ("cik", pa.int64()),
            ]
        )
        filers = {
            cik
            for cik, facts in share_facts(self.root).items()
            if any(f["form"] in COMPANY_FORMS and f["available_at"] <= self.as_of for f in facts)
        }
        day = self.as_of.astimezone(ET).date()
        sec = self.con.execute(
            f"SELECT cik, count(DISTINCT ticker) AS n FROM {_SEC} "
            "WHERE snapshot_date >= $start AND snapshot_date <= $day GROUP BY cik",
            {"start": SEC_HISTORY_START, "day": day},
        )
        shelves = {int(cik) for cik, n in sec.fetchall() if n >= SHELF_MIN_TICKERS}
        cik_of = {
            (seg["security_id"], seg["symbol"], seg["valid_from"]): int(seg["cik"])
            for seg in visible_master(self.con, self.root, self.as_of).to_pylist()
            if seg["cik"] is not None
            and (seg["cik_available_at"] is None or seg["cik_available_at"] <= self.as_of)
        }
        qualified = {
            r[0] for r in self.con.execute(f"SELECT security_id FROM {_QUAL_IDS}").fetchall()
        }
        rows = []
        for r in self._q(f"SELECT * FROM {_CLASS}").to_pylist():
            cik = cik_of.get((r["security_id"], r["symbol"], r["valid_from"]))
            if (
                r["class"] in ("etf", "leveraged_etf")
                and r["rule"] != "override"
                and cik in filers
                and cik not in shelves
            ):
                rows.append(
                    {
                        "key": f"{r['symbol']}:{r['valid_from'].isoformat()}",
                        "_liq": r["security_id"] in qualified,
                        "security_id": r["security_id"],
                        "symbol": r["symbol"],
                        "valid_from": r["valid_from"],
                        "name": r["name"],
                        "rule": r["rule"],
                        "cik": cik,
                    }
                )
        return pa.Table.from_pylist(rows, schema=schema), {"shelf_ciks": len(shelves)}

    # -- I: identity across bars and master ------------------------------------------------------

    def i1(self) -> tuple[pa.Table, dict]:
        same = " AND ".join(f"v.{c} = r.{c}" for c in ("open", "high", "low", "close", "volume"))
        self.con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE {_ORPHANS} AS
            SELECT r.symbol, r.session_date, coalesce(r.vwap, r.close) * r.volume AS usd
            FROM {_RAW} r
            ANTI JOIN {_RES} o ON o.symbol = r.symbol AND o.t = r.t
            ANTI JOIN {_RES} v ON v.session_date = r.session_date AND v.symbol <> r.symbol
                AND {same}
            WHERE r.volume > 0
            """
        )
        table = self._q(
            f"""
            SELECT symbol AS key, {_liq_sym("symbol")} AS _liq, symbol, count(*) AS bars,
                   min(session_date) AS "first", max(session_date) AS "last", sum(usd) AS usd
            FROM {_ORPHANS} GROUP BY symbol
            """
        )
        row = self.con.execute(
            f"SELECT count(*), count(*) FILTER (WHERE usd > 1e6), "
            f"count(DISTINCT symbol) FILTER (WHERE usd > 1e6) FROM {_ORPHANS}"
        ).fetchone()
        assert row is not None
        notes = {
            "bars": int(row[0]),
            "bars_over_1m_usd": int(row[1]),
            "symbols_over_1m_usd": int(row[2]),
        }
        return table, notes

    def i2(self) -> tuple[pa.Table, dict]:
        same = " AND ".join(f"a.{c} = b.{c}" for c in ("open", "high", "low", "close", "volume"))
        return self._q(
            f"""
            SELECT a.symbol || '/' || b.symbol AS key,
                   {_liq_sym("a.symbol")} OR {_liq_sym("b.symbol")} AS _liq,
                   a.symbol AS symbol_a, b.symbol AS symbol_b,
                   min(a.security_id) AS security_a, min(b.security_id) AS security_b,
                   count(*) AS days, min(a.session_date) AS "first", max(a.session_date) AS "last"
            FROM {_RES} a JOIN {_RES} b
              ON a.session_date = b.session_date AND a.symbol < b.symbol AND {same}
             AND a.security_id <> b.security_id
            WHERE a.volume > {DUP_MIN_VOLUME}
              AND NOT regexp_full_match(a.symbol, '{WHEN_ISSUED.pattern}')
              AND NOT regexp_full_match(b.symbol, '{WHEN_ISSUED.pattern}')
            GROUP BY a.symbol, b.symbol
            HAVING count(*) >= {DUP_MIN_DAYS}
            """
        ), {}

    def i3(self) -> tuple[pa.Table, dict]:
        flat = " AND ".join(
            f"min({c}) = max({c})" for c in ("open", "high", "low", "close", "volume")
        )
        return self._q(
            f"""
            WITH d AS (
                SELECT security_id, session_date, list(DISTINCT symbol) AS syms,
                       {flat} AS identical
                FROM {_RES} GROUP BY security_id, session_date
                HAVING count(DISTINCT symbol) > 1
            )
            SELECT security_id AS key, {_liq_sid("security_id")} AS _liq, security_id,
                   array_to_string(list_sort(list_distinct(flatten(list(syms)))), '/') AS symbols,
                   count(*) AS days, count(*) FILTER (WHERE identical) AS identical_days
            FROM d GROUP BY security_id
            """
        ), {}

    def i4(self) -> tuple[pa.Table, dict]:
        # bars a reader between the cut and its filing gave to the old security although they
        # belong to another one (or to none) as far as the store knows today
        knowable = _ny_date("m.end_available_at")
        return self._q(
            f"""
            SELECT m.symbol AS key, {_liq_sym("m.symbol")} AS _liq, m.symbol, m.security_id,
                   m.valid_to, {knowable} AS end_knowable, count(*) AS bars
            FROM {_MASTER} m
            JOIN {_RAW} r ON r.symbol = m.symbol AND r.volume > 0
             AND r.session_date >= m.valid_to AND r.session_date < {knowable}
            LEFT JOIN {_RES} v ON v.symbol = r.symbol AND v.t = r.t
            WHERE m.valid_to IS NOT NULL AND v.security_id IS DISTINCT FROM m.security_id
            GROUP BY m.symbol, m.security_id, m.valid_to, m.end_available_at
            """
        ), {}

    def i5(self) -> tuple[pa.Table, dict]:
        as_of_day = self.as_of.astimezone(UTC).date()
        return self._q(
            f"""
            WITH snaps AS (
                SELECT DISTINCT snapshot_date FROM {_SEC}
                WHERE snapshot_date >= $start AND snapshot_date <= $as_of_day
            ),
            iv AS (
                SELECT snapshot_date AS lo, lead(snapshot_date) OVER (ORDER BY snapshot_date) AS hi
                FROM snaps
            ),
            sec AS (
                SELECT s.ticker, s.cik, iv.lo, iv.hi
                FROM {_SEC} s JOIN iv ON s.snapshot_date = iv.lo
            ),
            bars AS (
                SELECT v.session_date, m.security_id, m.symbol, m.valid_from, m.cik
                FROM {_RES} v JOIN {_MASTER} m
                  ON m.security_id = v.security_id AND m.symbol = v.symbol
                 AND v.session_date >= m.valid_from
                 AND (m.valid_to IS NULL OR v.session_date < m.valid_to)
                WHERE m.cik IS NOT NULL
            )
            SELECT b.symbol || ':' || {_day("b.valid_from")} AS key, {_liq_sym("b.symbol")} AS _liq,
                   b.security_id, b.symbol, b.valid_from, b.cik AS segment_cik, s.cik AS sec_cik,
                   count(*) AS bars
            FROM bars b JOIN sec s
              ON s.ticker = b.symbol AND b.session_date >= s.lo
             AND (s.hi IS NULL OR b.session_date < s.hi)
            WHERE s.cik <> b.cik
            GROUP BY b.security_id, b.symbol, b.valid_from, b.cik, s.cik
            """,
            {"start": SEC_HISTORY_START, "as_of_day": as_of_day},
        ), {}

    def i6(self) -> tuple[pa.Table, dict]:
        table = self._q(
            f"""
            SELECT {_day("year(session_date)")} AS key, false AS _liq, year(session_date) AS year,
                   count(*) FILTER (WHERE volume = 0) AS untraded, count(*) AS bars,
                   count(*) FILTER (WHERE volume = 0) / count(*) AS share
            FROM {_RAW} GROUP BY year(session_date)
            """
        )
        untraded = sum(table.column("untraded").to_pylist())
        bars = sum(table.column("bars").to_pylist())
        return table, {
            "untraded": untraded,
            "bars": bars,
            "share": untraded / bars if bars else 0.0,
        }

    def i7(self) -> tuple[pa.Table, dict]:
        counts = self._q(
            f"SELECT session_date, count(DISTINCT symbol) FILTER (WHERE volume > 0) "
            f"AS traded_symbols FROM {_RAW} GROUP BY session_date"
        )
        traded = counts.column("traded_symbols").to_pylist()
        median = statistics.median(traded) if traded else 0
        self.con.register("_audit_counts", counts)
        try:
            table = self._q(
                f"""
                SELECT {_day("session_date")} AS key, false AS _liq, session_date, traded_symbols
                FROM _audit_counts WHERE traded_symbols < $limit
                """,
                {"limit": SPARSE_SESSION * median},
            )
        finally:
            self.con.unregister("_audit_counts")
        notes = {
            "min_traded_symbols": min(traded) if traded else 0,
            "median_traded_symbols": median,
            "sessions": len(traded),
        }
        return table, notes

    def i8(self) -> tuple[pa.Table, dict]:
        return self._q(
            f"""
            WITH sd AS (
                SELECT security_id, session_date, min(symbol) AS symbol
                FROM {_RES} GROUP BY security_id, session_date
            ),
            x AS (
                SELECT sd.*, s.idx, lag(s.idx) OVER w AS prev_idx,
                       lag(sd.session_date) OVER w AS prev_date
                FROM sd JOIN {_SESSIONS} s USING (session_date)
                WINDOW w AS (PARTITION BY sd.security_id ORDER BY sd.session_date)
            )
            SELECT security_id || ':' || {_day("session_date")} AS key,
                   {_liq_sid("security_id")} AS _liq, security_id, symbol,
                   prev_date AS gap_from, session_date AS gap_to,
                   idx - prev_idx - 1 AS sessions_missing
            FROM x WHERE idx - prev_idx - 1 > {GAP_SESSIONS}
            """
        ), {}

    def i9(self) -> tuple[pa.Table, dict]:
        lo, hi = CONTINUOUS
        return self._q(
            f"""
            {self._PAIRS},
            closed AS (SELECT * FROM pairs WHERE old_to IS NOT NULL),
            last_b AS (
                SELECT p.security_id, p.new_from, arg_max(v.close, v.t) AS last_close,
                       max(v.session_date) AS last_date
                FROM closed p JOIN {_RES} v
                  ON v.security_id = p.security_id AND v.symbol = p.old_symbol
                 AND v.session_date >= p.old_from AND v.session_date < p.old_to
                GROUP BY p.security_id, p.new_from
            ),
            first_b AS (
                SELECT p.security_id, p.new_from, arg_min(v.close, v.t) AS first_close,
                       min(v.session_date) AS first_date
                FROM closed p JOIN {_RES} v
                  ON v.security_id = p.security_id AND v.symbol = p.new_symbol
                 AND v.session_date >= p.new_from
                 AND (p.new_to IS NULL OR v.session_date < p.new_to)
                GROUP BY p.security_id, p.new_from
            ),
            j AS (
                SELECT p.*, l.last_close, l.last_date, f.first_close, f.first_date
                FROM closed p
                LEFT JOIN last_b l USING (security_id, new_from)
                LEFT JOIN first_b f USING (security_id, new_from)
            ),
            adj AS (
                -- factor = new shares per old share: 4:1 turns 100 into 25, so multiply back
                SELECT j.*, (
                    SELECT product(factor) FROM (
                        SELECT DISTINCT ex_date, factor FROM {_SPLITS} sp
                        WHERE sp.symbol IN (j.old_symbol, j.new_symbol)
                          AND sp.ex_date > j.last_date AND sp.ex_date <= j.first_date
                    )
                ) AS factor
                FROM j
            ),
            r AS (
                SELECT *, first_close * coalesce(factor, 1) / last_close AS ratio,
                       first_date - last_date AS gap_days
                FROM adj
            )
            SELECT security_id || ':' || {_day("new_from")} AS key,
                   {_liq_sid("security_id")} AS _liq, security_id, old_symbol, new_symbol,
                   last_close, last_date, first_close, first_date, ratio, gap_days
            FROM r
            WHERE last_close IS NULL OR first_close IS NULL OR ratio <= {lo} OR ratio >= {hi}
               OR gap_days > {CONTINUOUS_DAYS}
            """
        ), {}

    def i10(self) -> tuple[pa.Table, dict]:
        """A company the SEC lists on an exchange that owns no traded bar of its listed span:
        its history is not in the store (Inhibrx's INBX, dropped by the vendor when the label
        was reused; Barnes's B, overwritten with Barrick's). Information for a second source."""
        listing = ", ".join(f"'{e}'" for e in LISTING_EXCHANGES)
        table = self._q(
            f"""
            WITH listed AS (
                SELECT ticker, cik, min(snapshot_date) AS first_listed,
                       max(snapshot_date) AS last_listed,
                       count(DISTINCT snapshot_date) AS snapshots
                FROM {_EXCHANGES}
                WHERE exchange IN ({listing})
                  AND regexp_full_match(ticker, '[A-Z]{{1,5}}')
                  AND NOT regexp_full_match(ticker, '[A-Z]{{4}}X')  -- mutual fund tickers
                GROUP BY ticker, cik
                HAVING count(DISTINCT snapshot_date) >= {I10_MIN_SNAPSHOTS}
            )
            SELECT l.ticker || ':' || CAST(l.cik AS VARCHAR) AS key, false AS _liq,
                   l.ticker, l.cik, l.first_listed, l.last_listed, l.snapshots
            FROM listed l
            WHERE NOT EXISTS (
                SELECT 1 FROM {_MASTER} m
                JOIN {_RAW} r ON r.symbol = m.symbol AND r.volume > 0
                     AND r.session_date >= m.valid_from
                     AND (m.valid_to IS NULL OR r.session_date < m.valid_to)
                WHERE m.symbol = l.ticker AND m.cik = l.cik
                  AND r.session_date BETWEEN l.first_listed AND l.last_listed
            )
            """
        )
        return table, {}

    def i12(self) -> tuple[pa.Table, dict]:
        """Session dates against the exchange calendar, from the first to the last stored
        session: an exchange session with no bar at all, or a bar dated on a day the exchange
        was closed. Every reader takes its calendar from the bars (``visible_sessions``), so a
        missing day silently stretches a 20-session window. Days after the last stored session
        are stale data, C2's business."""
        stored = [
            r[0] for r in self.con.execute(f"SELECT session_date FROM {_SESSIONS}").fetchall()
        ]
        days: list[date] = []
        if stored:
            calendar = xc.get_calendar(
                CALENDAR, start=min(stored).isoformat(), end=max(stored).isoformat()
            )
            days = [d.date() for d in calendar.sessions_in_range(min(stored), max(stored))]
        self.con.register(_CAL, pa.table({"day": pa.array(days, pa.date32())}))
        table = self._q(
            f"""
            SELECT {_day("day")} AS key, false AS _liq, day, reason FROM (
                SELECT day, 'missing_session' AS reason FROM {_CAL}
                WHERE day NOT IN (SELECT session_date FROM {_SESSIONS})
                UNION ALL
                SELECT session_date AS day, 'session_on_holiday' AS reason FROM {_SESSIONS}
                WHERE session_date NOT IN (SELECT day FROM {_CAL})
            )
            """
        )
        return table, {"sessions": len(stored), "exchange_sessions": len(days)}

    # -- S: splits and price jumps ---------------------------------------------------------------

    def _split_liq(self, owner: str, record: str) -> str:
        """Liquid scope of a split record: with the monthly tier, the owning security in the
        tier within ``SPLIT_LIQUID_DAYS`` of the ex-date; without it, the all-years union."""
        if not self.dated_tier:
            return _liq_sym(f"{record}.symbol")
        return (
            f"coalesce(EXISTS (SELECT 1 FROM {_TIER} l WHERE l.security_id = {owner}.security_id "
            f"AND abs(date_diff('day', l.day, {record}.ex_date)) <= {SPLIT_LIQUID_DAYS}), false)"
        )

    def s1(self) -> tuple[pa.Table, dict]:
        n = SPLIT_TRADED_SESSIONS
        table = self._q(
            f"""
            WITH stored AS (SELECT DISTINCT symbol FROM {_RAW}),
            sp AS (
                SELECT DISTINCT symbol, ex_date, kind, factor FROM {_SPLITS}
                -- a split is known from 00:00 of its ex-date, its bar only from 20:00
                WHERE ex_date <= (SELECT max(session_date) FROM {_SESSIONS})
            ),
            own AS (SELECT sp.*, o.security_id, o.ex_idx FROM sp JOIN {_SPLIT_OWNERS} o
                    USING (symbol, ex_date)),
            -- bars around the ex-date under any name of the owning security, or of the record's
            -- own symbol when no segment owns it
            near AS (
                SELECT o.symbol, o.ex_date, r.session_date, r.volume, s.idx
                FROM own o JOIN {_MASTER} m ON m.security_id = o.security_id
                JOIN {_RAW} r ON r.symbol = m.symbol AND r.session_date >= m.valid_from
                     AND (m.valid_to IS NULL OR r.session_date < m.valid_to)
                JOIN {_SESSIONS} s ON s.session_date = r.session_date
                WHERE s.idx BETWEEN o.ex_idx - {n} AND o.ex_idx + {n}
                UNION ALL
                SELECT o.symbol, o.ex_date, r.session_date, r.volume, s.idx
                FROM own o JOIN {_RAW} r ON r.symbol = o.symbol
                JOIN {_SESSIONS} s ON s.session_date = r.session_date
                WHERE o.security_id IS NULL AND s.idx BETWEEN o.ex_idx - {n} AND o.ex_idx + {n}
            ),
            judged AS (
                SELECT o.*,
                       EXISTS (SELECT 1 FROM near b WHERE b.symbol = o.symbol
                               AND b.ex_date = o.ex_date AND b.session_date = o.ex_date) AS has_bar,
                       -- a side is judged only when the store holds all its sessions (the newest
                       -- records have no "after" yet)
                       o.ex_idx > {n} AND NOT EXISTS (
                           SELECT 1 FROM near b WHERE b.symbol = o.symbol AND b.ex_date = o.ex_date
                           AND b.volume > 0 AND b.idx < o.ex_idx) AS silent_before,
                       o.ex_idx + {n} <= (SELECT max(idx) FROM {_SESSIONS}) AND NOT EXISTS (
                           SELECT 1 FROM near b WHERE b.symbol = o.symbol AND b.ex_date = o.ex_date
                           AND b.volume > 0 AND b.idx > o.ex_idx)
                       -- silence after the record excuses it only when the security ended
                       -- (bought, delisted); an open security gone quiet may have lost its bars
                       AND NOT EXISTS (
                           SELECT 1 FROM {_MASTER} m WHERE m.security_id = o.security_id
                           AND (m.valid_to IS NULL OR m.valid_to > (
                               SELECT session_date FROM {_SESSIONS}
                               WHERE idx = o.ex_idx + {n}))) AS silent_after
                FROM own o
            ),
            rows AS (
                SELECT j.*,
                       CASE WHEN j.symbol NOT IN (SELECT symbol FROM stored)
                                THEN 'symbol_never_stored'
                            WHEN j.silent_before OR j.silent_after THEN 'not_traded_around'
                            ELSE 'no_bar_that_day' END AS reason
                FROM judged j WHERE NOT j.has_bar
            )
            SELECT x.symbol || ':' || {_day("x.ex_date")} AS key,
                   x.reason <> 'not_traded_around' AND {self._split_liq("x", "x")} AS _liq,
                   x.symbol, x.ex_date, x.kind, x.factor, x.reason
            FROM rows x
            """
        )
        reasons = table.column("reason").to_pylist()
        notes = {
            r: reasons.count(r)
            for r in ("symbol_never_stored", "no_bar_that_day", "not_traded_around")
        }
        return table, notes

    def _has_prev(self, alias: str = "t") -> str:
        return f"{alias}.prev_close > 0 AND {alias}.idx - {alias}.prev_idx <= {NEIGHBOUR_SESSIONS}"

    def _near_split(self, alias: str = "t") -> str:
        return (
            f"EXISTS (SELECT 1 FROM {_SPLIT_DAYS} d WHERE d.symbol = {alias}.symbol "
            f"AND d.ex_date BETWEEN {alias}.lo AND {alias}.hi)"
        )

    def s2(self) -> tuple[pa.Table, dict]:
        traded = self._traded()
        tol, late = SPLIT_TOLERANCE, SPLIT_LATE_SESSIONS

        def off(open_: str) -> str:  # this open, against the close before the ex-date
            return f"abs(({open_} / t.prev_close) * sp.factor - 1) > {tol}"

        table = self._q(
            f"""
            WITH ahead AS (
                SELECT *, lead(open, 1) OVER w AS open1, lead(idx, 1) OVER w AS idx1,
                       lead(open, 2) OVER w AS open2, lead(idx, 2) OVER w AS idx2
                FROM {traded} WINDOW w AS (PARTITION BY symbol ORDER BY t)
            )
            SELECT sp.symbol || ':' || {_day("sp.ex_date")} AS key,
                   {self._split_liq("o", "sp")} AS _liq,
                   sp.symbol, sp.ex_date, sp.kind, sp.factor, t.prev_close, t.open,
                   t.open / t.prev_close AS observed, 1.0 / sp.factor AS expected
            FROM (SELECT DISTINCT symbol, ex_date, kind, factor FROM {_SPLITS}
                  WHERE kind <> 'unit_split' AND factor > 0) sp
            JOIN ahead t ON t.symbol = sp.symbol AND t.session_date = sp.ex_date
            LEFT JOIN {_SPLIT_OWNERS} o ON o.symbol = sp.symbol AND o.ex_date = sp.ex_date
            JOIN {_SESSIONS} ps ON ps.idx = t.prev_idx
            WHERE {self._has_prev()}
              -- the previous close must be the same security's (ARNC changed holder that day)
              AND (SELECT m.security_id FROM {_MASTER} m
                   WHERE m.symbol = t.symbol AND m.valid_from <= ps.session_date
                     AND (m.valid_to IS NULL OR ps.session_date < m.valid_to)
                   LIMIT 1) IS NOT DISTINCT FROM o.security_id
              -- the move may land up to SPLIT_LATE_SESSIONS after the record (ULTY)
              AND {off("t.open")}
              AND (t.open1 IS NULL OR t.idx1 - t.idx > {late} OR {off("t.open1")})
              AND (t.open2 IS NULL OR t.idx2 - t.idx > {late} OR {off("t.open2")})
            """
        )
        # records S2 cannot test, so a clean S2 is not read as "every split moved the price"
        row = self.con.execute(
            f"""
            WITH sp AS (SELECT DISTINCT symbol, ex_date FROM {_SPLITS}
                        WHERE kind <> 'unit_split' AND factor > 0),
            day AS (
                SELECT sp.symbol, sp.ex_date, count(r.symbol) AS bars,
                       count(r.symbol) FILTER (WHERE r.volume > 0) AS traded
                FROM sp LEFT JOIN {_RAW} r ON r.symbol = sp.symbol AND r.session_date = sp.ex_date
                GROUP BY ALL
            )
            SELECT count(*) FILTER (WHERE bars > 0 AND traded = 0),
                   (SELECT count(*) FROM sp JOIN {traded} t
                      ON t.symbol = sp.symbol AND t.session_date = sp.ex_date
                    WHERE NOT ({self._has_prev()}) OR t.prev_close IS NULL)
            FROM day
            """
        ).fetchone()
        assert row is not None
        return table, {"unchecked_untraded": int(row[0]), "unchecked_no_prev": int(row[1])}

    def s3(self) -> tuple[pa.Table, dict]:
        traded = self._traded()
        back_lo, back_hi = BAD_BAR_RETURN
        return self._q(
            f"""
            SELECT symbol || ':' || {_day("session_date")} AS key, {_liq_sym("symbol")} AS _liq,
                   symbol, session_date, prev_any_close AS prev_close, open, high, close,
                   next_close, volume
            FROM {traded} t
            WHERE t.prev_any_close >= 1 AND t.idx - t.prev_any_idx <= {NEIGHBOUR_SESSIONS}
              AND t.next_close IS NOT NULL AND t.next_idx - t.idx <= {NEIGHBOUR_SESSIONS}
              AND (greatest(t.open, t.high, t.close) / t.prev_any_close >= {JUMP}
                   OR least(t.open, t.high, t.close) / t.prev_any_close <= 1.0 / {JUMP})
              AND t.next_close / t.prev_any_close BETWEEN {back_lo} AND {back_hi}
              AND t.volume < {BAD_BAR_MAX_VOLUME}
              AND NOT {self._near_split()}
            """
        ), {}

    def s4(self) -> tuple[pa.Table, dict]:
        traded = self._traded()
        table = self._q(
            f"""
            SELECT symbol || ':' || {_day("session_date")} AS key, {_liq_sym("symbol")} AS _liq,
                   symbol, session_date, prev_close, close, close / prev_close AS ratio,
                   coalesce(vwap, close) * volume AS usd,
                   CASE WHEN prev_close < 1 THEN 'lt1' WHEN prev_close <= 10 THEN '1to10'
                        ELSE 'gt10' END AS bucket
            FROM {traded} t
            WHERE {self._has_prev()}
              AND (t.close / t.prev_close >= {JUMP} OR t.close / t.prev_close <= 1.0 / {JUMP})
              AND NOT {self._near_split()}
            """
        )
        buckets = table.column("bucket").to_pylist()
        return table, {b: buckets.count(b) for b in ("lt1", "1to10", "gt10")}

    def s5(self) -> tuple[pa.Table, dict]:
        # repeated records are already one event here (splits_as_of); what is left are days the
        # readers multiply, right only when the events are genuinely chained (MDRR 2024-07-03)
        table = self._q(
            f"""
            SELECT symbol || ':' || {_day("ex_date")} AS key, {_liq_sym("symbol")} AS _liq,
                   symbol, ex_date, count(*) AS events, product(factor) AS net_factor,
                   string_agg(kind || ' ' || factor, ', ' ORDER BY factor) AS records
            FROM {_SPLITS}
            GROUP BY symbol, ex_date
            HAVING count(*) > 1
            """
        )
        return table, {}

    def s6(self) -> tuple[pa.Table, dict]:
        # restarts whose factor the ex-session prices disprove (BBIG 2022-06-30: -19.6%): a
        # record worth a look, perhaps a rejection, though no reader rescales anything by it
        events = spinoff_events(self.con, self.root, self.as_of)
        self.con.execute("SET TimeZone = 'UTC'")
        self.con.register("_audit_spinoffs", events)
        if self.dated_tier:
            liq = (
                f"coalesce(EXISTS (SELECT 1 FROM {_TIER} l WHERE l.security_id = e.security_id "
                f"AND abs(date_diff('day', l.day, e.ex_date)) <= {SPLIT_LIQUID_DAYS}), false)"
            )
        else:
            liq = f"e.security_id IN (SELECT security_id FROM {_LIQ_IDS})"
        table = self._q(
            f"""
            SELECT e.parent_symbol || ':' || {_day("e.ex_date")} AS key, {liq} AS _liq,
                   e.parent_symbol, e.security_id, e.ex_date,
                   array_to_string(e.child_symbols, ',') AS children,
                   e.parent_prev_close, array_to_string(e.child_closes, ',') AS child_closes,
                   array_to_string(e.ratio, ',') AS ratios, e.total_return
            FROM _audit_spinoffs e
            WHERE e.status = 'restart' AND e.reason = 'factor_disproved'
            ORDER BY e.parent_symbol, e.ex_date
            """
        )
        statuses = events.column("status").to_pylist()
        return table, {
            status: statuses.count(status) for status in ("adjust", "restart", "pending")
        }

    # -- D: dividends ----------------------------------------------------------------------------

    def d1(self) -> tuple[pa.Table, dict]:
        """One payout stored twice with its rate rounded differently (FMX 2026-04-22: 1.039579
        and 1.0395792), or once more under another flag (SATA 2026-08-11: sub_type null and
        'return_of_capital'): the benchmark pays every visible row. Read through
        ``visible_dividends``, so rows identical in every column already count once."""
        self.con.register(_DIVIDENDS, visible_dividends(self.con, self.root, self.as_of))
        self.con.execute("SET TimeZone = 'UTC'")
        table = self._q(
            f"""
            WITH d AS (SELECT row_number() OVER () AS id, * FROM {_DIVIDENDS}),
            twins AS (
                SELECT DISTINCT a.symbol, a.ex_date FROM d a JOIN d b
                  ON a.symbol = b.symbol AND a.ex_date = b.ex_date AND a.id < b.id
                 AND (abs(a.rate - b.rate) <= {DIVIDEND_TWIN_TOLERANCE} * greatest(a.rate, b.rate)
                      OR abs(a.rate - b.rate) <= {DIVIDEND_TWIN_ABS})
            )
            SELECT d.symbol || ':' || {_day("d.ex_date")} AS key, {_liq_sym("d.symbol")} AS _liq,
                   d.symbol, d.ex_date, list(d.rate ORDER BY d.rate) AS rates
            FROM d JOIN twins USING (symbol, ex_date)
            GROUP BY d.symbol, d.ex_date
            """
        )
        return table, {}

    # -- N: name changes against the bars --------------------------------------------------------

    def n1(self) -> tuple[pa.Table, dict]:
        all_rows = self._q(
            f"""
            WITH lt AS (
                SELECT symbol, max(session_date) AS last_traded FROM {_RAW}
                WHERE volume > 0 GROUP BY symbol
            )
            SELECT DISTINCT nc.old_symbol || '/' || nc.new_symbol || ':' ||
                       {_day("nc.process_date")} AS key,
                   {_liq_sym("nc.old_symbol")} OR {_liq_sym("nc.new_symbol")} AS _liq,
                   nc.old_symbol, nc.new_symbol, nc.process_date, lt.last_traded,
                   nc.process_date - lt.last_traded AS lag_days
            FROM {_NAMES} nc LEFT JOIN lt ON lt.symbol = nc.old_symbol
            """
        )
        notes = dict.fromkeys(("le0", "d1_10", "d11_30", "d31_180", "gt180", "no_old_bars"), 0)
        for lag in all_rows.column("lag_days").to_pylist():
            if lag is None:
                notes["no_old_bars"] += 1
            elif lag <= 0:
                notes["le0"] += 1
            elif lag <= 10:
                notes["d1_10"] += 1
            elif lag <= 30:
                notes["d11_30"] += 1
            elif lag <= 180:
                notes["d31_180"] += 1
            else:
                notes["gt180"] += 1
        keep = [
            lag is not None and lag > RENAME_LAG_DAYS for lag in all_rows["lag_days"].to_pylist()
        ]
        return all_rows.filter(pa.array(keep, pa.bool_())), notes

    def n2(self) -> tuple[pa.Table, dict]:
        return self._q(
            f"""
            WITH tr AS (
                SELECT symbol, min(session_date) AS first_traded, max(session_date) AS last_traded
                FROM {_RAW} WHERE volume > 0 GROUP BY symbol
            )
            SELECT DISTINCT nc.old_symbol || '/' || nc.new_symbol || ':' ||
                       {_day("nc.process_date")} AS key,
                   {_liq_sym("nc.old_symbol")} OR {_liq_sym("nc.new_symbol")} AS _liq,
                   nc.old_symbol, nc.new_symbol, nc.process_date,
                   o.last_traded AS old_last_traded, n.first_traded AS new_first_traded
            FROM {_NAMES} nc
            JOIN tr o ON o.symbol = nc.old_symbol
            JOIN tr n ON n.symbol = nc.new_symbol
            WHERE o.last_traded > nc.process_date AND n.first_traded > nc.process_date
            """
        ), {}

    def n3(self) -> tuple[pa.Table, dict]:
        same = " AND ".join(f"a.{c} = b.{c}" for c in ("open", "high", "low", "close", "volume"))
        pairs = self._q(
            f"""
            WITH td AS (
                SELECT symbol, count(DISTINCT session_date) AS days FROM {_RAW}
                WHERE volume > 0 GROUP BY symbol
            ),
            dup AS (
                SELECT a.symbol AS symbol_a, b.symbol AS symbol_b, count(*) AS identical_days
                FROM {_RAW} a JOIN {_RAW} b
                  ON a.session_date = b.session_date AND a.symbol < b.symbol AND {same}
                WHERE a.volume > {DUP_MIN_VOLUME}
                GROUP BY a.symbol, b.symbol
            )
            SELECT d.symbol_a || '/' || d.symbol_b AS key,
                   {_liq_sym("d.symbol_a")} OR {_liq_sym("d.symbol_b")} AS _liq,
                   d.symbol_a, d.symbol_b, d.identical_days,
                   least(ta.days, tb.days) AS shorter_days,
                   d.identical_days / least(ta.days, tb.days) AS ratio
            FROM dup d JOIN td ta ON ta.symbol = d.symbol_a JOIN td tb ON tb.symbol = d.symbol_b
            """
        )
        lo, hi = RELABEL_BAND
        ratios = pairs.column("ratio").to_pylist()
        notes = {
            "lt10": sum(1 for r in ratios if r < lo),
            "p10_50": sum(1 for r in ratios if lo <= r < 0.5),
            "p50_90": sum(1 for r in ratios if 0.5 <= r < hi),
            "ge90": sum(1 for r in ratios if r >= hi),
        }
        keep = pa.array([lo <= r < hi for r in ratios], pa.bool_())
        return pairs.filter(keep), notes

    # -- C: coverage -----------------------------------------------------------------------------

    def c1(self) -> tuple[pa.Table, dict]:
        table = self._q(
            f"""
            WITH b AS (
                SELECT year(session_date) AS year, count(*) AS bars,
                       count(DISTINCT symbol) AS symbols
                FROM {_RAW} GROUP BY 1
            ),
            nc AS (
                SELECT year(process_date) AS year, count(*) AS name_changes,
                       count(*) FILTER (WHERE linked) AS name_changes_linked
                FROM (
                    SELECT n.process_date, EXISTS (
                        SELECT 1 FROM {_MASTER} a JOIN {_MASTER} c
                          ON a.security_id = c.security_id
                        WHERE a.symbol = n.old_symbol AND c.symbol = n.new_symbol
                    ) AS linked
                    FROM {_NAMES} n
                ) GROUP BY 1
            ),
            ext AS (
                SELECT year(r.session_date) AS year, count(*) AS ext_bars,
                       count(*) FILTER (WHERE EXISTS (
                           SELECT 1 FROM {_ALPACA_KEYS} k WHERE k.symbol = r.symbol AND k.t = r.t
                       )) AS ext_overrides
                FROM {_RAW} r WHERE r.source <> 'alpaca' GROUP BY 1
            ),
            y AS (SELECT year FROM b UNION SELECT year FROM nc),
            liq AS (
                SELECT y.year, count(DISTINCT m.symbol) AS liquid_symbols
                FROM y JOIN {_MASTER} m
                  ON m.valid_from <= make_date(y.year, 12, 31)
                 AND (m.valid_to IS NULL OR m.valid_to > make_date(y.year, 1, 1))
                WHERE m.security_id IN (SELECT security_id FROM {_LIQ_IDS})
                GROUP BY y.year
            )
            SELECT {_day("y.year")} AS key, false AS _liq, y.year,
                   coalesce(b.bars, 0) AS bars, coalesce(b.symbols, 0) AS symbols,
                   coalesce(liq.liquid_symbols, 0) AS liquid_symbols,
                   coalesce(nc.name_changes, 0) AS name_changes,
                   coalesce(nc.name_changes_linked, 0) AS name_changes_linked,
                   coalesce(ext.ext_bars, 0) AS ext_bars,
                   coalesce(ext.ext_overrides, 0) AS ext_overrides
            FROM y LEFT JOIN b USING (year) LEFT JOIN nc USING (year) LEFT JOIN liq USING (year)
                   LEFT JOIN ext USING (year)
            """
        )
        notes = {
            "name_changes": sum(table.column("name_changes").to_pylist()),
            "name_changes_linked": sum(table.column("name_changes_linked").to_pylist()),
        }
        return table, notes

    def c2(self) -> tuple[pa.Table, dict]:
        """Listed (Alpaca active, not OTC) symbols whose last visible bar lies more than
        ``STALE_SESSIONS`` sessions before the store's newest visible session: the backfill
        stopped asking for them (NOBL, whose SEC window closed in 2020) or they halted. Listed
        symbols with no visible bar at all are only counted (a listing of a few days old and a
        warrant that never traded are not errors). "Last bar" is any stored bar, a volume-0
        placeholder included: the question is whether the vendor is still asked for and still
        serving the symbol, not whether it still trades."""
        table = self._q(
            f"""
            WITH last AS (
                SELECT symbol, max(session_date) AS last_bar FROM {_RAW} GROUP BY symbol
            ),
            newest AS (SELECT coalesce(max(idx), 0) AS idx FROM {_SESSIONS})
            SELECT l.symbol AS key, {_liq_sym("l.symbol")} AS _liq, l.symbol, l.name,
                   l.exchange, b.last_bar, CAST(newest.idx - s.idx AS BIGINT) AS sessions_behind
            FROM {_LISTED} l
            JOIN last b USING (symbol)
            JOIN {_SESSIONS} s ON s.session_date = b.last_bar
            CROSS JOIN newest
            WHERE newest.idx - s.idx > {STALE_SESSIONS}
            ORDER BY key
            """
        )
        no_bars = self._scalar(
            f"SELECT count(*) FROM {_LISTED} WHERE symbol NOT IN (SELECT symbol FROM {_RAW})"
        )
        notes = {
            "no_bars": int(no_bars or 0),
            "snapshot_date": self.snapshot_date.isoformat() if self.snapshot_date else None,
        }
        return table, notes

    def f1(self) -> tuple[pa.Table, dict]:
        """Tier common stocks without a count ``visible_shares`` would use on a tier night, read
        with what was public that night: one row per symbol, with the nights it is missing.
        Without the dated tier there are no nights to judge."""
        schema = pa.schema(
            [
                ("key", pa.string()),
                ("_liq", pa.bool_()),
                ("nights", pa.int64()),
                ("first_night", pa.date32()),
                ("last_night", pa.date32()),
            ]
        )
        if self.liquid_tier is None:
            return schema.empty_table(), {"nights": 0}
        by_night: dict[datetime, list[str]] = {}
        for r in self.liquid_tier.to_pylist():
            if r["class"] == "common" and r["as_of"] <= self.as_of:
                by_night.setdefault(r["as_of"], []).append(r["security_id"])
        missing: dict[str, list[date]] = {}
        for at, ids in sorted(by_night.items()):
            have = set(
                visible_shares(self.con, self.root, at, ids, max_age_days=SHARE_MAX_AGE_DAYS)
                .column("security_id")
                .to_pylist()
            )
            night = at.astimezone(ET).date()
            symbol = {
                seg["security_id"]: seg["symbol"]
                for seg in visible_master(self.con, self.root, at).to_pylist()
                if seg["valid_from"] <= night
                and (seg["valid_to"] is None or night < seg["valid_to"])
            }
            for sid in ids:
                if sid not in have:
                    missing.setdefault(symbol.get(sid, sid), []).append(night)
        rows = [
            {"key": k, "_liq": True, "nights": len(v), "first_night": min(v), "last_night": max(v)}
            for k, v in missing.items()
        ]
        return pa.Table.from_pylist(rows, schema=schema), {"nights": len(by_night)}

    def f2(self) -> tuple[pa.Table, dict]:
        """Tier common stocks whose count gives a cap no count could (``implausible``), on the
        last session before each tier night: one row per stock and count. A cap filter reads
        such a stock as having no count, so the rows are listed, not gated."""
        schema = pa.schema(
            [
                ("key", pa.string()),
                ("_liq", pa.bool_()),
                ("nights", pa.int64()),
                ("first_night", pa.date32()),
                ("last_night", pa.date32()),
                ("concept", pa.string()),
                ("shares", pa.float64()),
                ("raw_cap", pa.float64()),
                ("adv_usd", pa.float64()),
            ]
        )
        if self.liquid_tier is None:
            return schema.empty_table(), {"nights": 0}
        by_night: dict[datetime, dict[str, float]] = {}
        for r in self.liquid_tier.to_pylist():
            if r["class"] == "common" and r["as_of"] <= self.as_of:
                by_night.setdefault(r["as_of"], {})[r["security_id"]] = r["adv_usd"]
        sessions = visible_sessions(self.con, self.root, "1Day", self.as_of)
        found: dict[str, dict[str, Any]] = {}
        for at, adv in sorted(by_night.items()):
            day = at.astimezone(ET).date()
            before = [d for d in sessions if d < day]
            if not before:
                continue
            evening = datetime.combine(before[-1], SCAN_CLOSE, tzinfo=ET)
            caps = market_caps(
                self.con, self.root, evening, list(adv), max_age_days=SHARE_MAX_AGE_DAYS
            )
            bad = implausible(
                caps, adv_usd=adv, max_turnover=SHARE_MAX_TURNOVER, max_cap=SHARE_MAX_CAP
            )
            for r in caps.to_pylist():
                if r["security_id"] not in bad:
                    continue
                key = f"{r['symbol']}:{r['end'].isoformat()}"
                row = found.setdefault(
                    key,
                    {
                        "key": key,
                        "_liq": True,
                        "nights": 0,
                        "first_night": before[-1],
                        "concept": r["concept"],
                        "shares": r["shares"],
                        "raw_cap": r["raw_cap"],
                        "adv_usd": adv[r["security_id"]],
                    },
                )
                row["nights"] += 1
                row["last_night"] = before[-1]
        return pa.Table.from_pylist(list(found.values()), schema=schema), {"nights": len(by_night)}


def _finish(spec: CheckSpec, table: pa.Table, excepted_keys: set[str]) -> pa.Table:
    """``key, in_liquid, excepted`` first, then the check's own columns, sorted by key."""
    keys = table.column("key").cast(pa.string())
    in_liquid = table.column("_liq").cast(pa.bool_())
    excepted = pa.array([k in excepted_keys for k in keys.to_pylist()], pa.bool_())
    rest = [name for name in table.schema.names if name not in ("key", "_liq")]
    out = pa.table(
        [keys, in_liquid, excepted, *(table.column(n) for n in rest)],
        names=["key", "in_liquid", "excepted", *rest],
    )
    return out.sort_by([("key", "ascending")])


def run_audit(
    con: duckdb.DuckDBPyConnection,
    root: Path,
    *,
    as_of: datetime,
    liquid_ids: Collection[str],
    qualified_ids: Collection[str] | None = None,
    exceptions: Sequence[AuditException] = (),
    checks: Sequence[str] | None = None,
    liquid_tier: pa.Table | None = None,
) -> list[CheckResult]:
    """Run ``checks`` (all by default) over the store under ``root`` as known at ``as_of``, in
    ``CHECKS`` order. ``liquid_ids`` is the liquid-tier union (``tier_union``).

    ``qualified_ids`` is the same union taken over every instrument class (``classes=None``)
    and scopes M5 and M9: a security whose class is unknown (or a leveraged ETF) can never be
    in the liquid tier, so judged against ``liquid_ids`` it would never fail the liquid gate.
    Defaults to ``liquid_ids``.

    ``liquid_tier`` is the monthly tier itself (``tier_union`` rows); when given, split records
    are scoped by date (``SPLIT_LIQUID_DAYS``) rather than by the union."""
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    wanted = list(CHECKS) if checks is None else list(checks)
    unknown = [c for c in wanted if c not in CHECKS]
    if unknown:
        raise ValueError(f"unknown checks: {', '.join(unknown)}")
    selected = [c for c in CHECKS if c in set(wanted)]
    if not selected:
        return []

    audit = _Audit(con, root, as_of.astimezone(UTC))
    try:
        audit.load(liquid_ids, liquid_ids if qualified_ids is None else qualified_ids, liquid_tier)
        results = []
        for check_id in selected:
            method: Callable[[], tuple[pa.Table, dict]] = getattr(audit, check_id.lower())
            table, notes = method()
            excepted = {e.key for e in exceptions if e.check == check_id}
            spec = CHECKS[check_id]
            results.append(CheckResult(spec, _finish(spec, table, excepted), notes))
        return results
    finally:
        audit.close()
