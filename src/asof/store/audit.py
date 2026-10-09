"""Store audit: one number per known kind of data accident, over the store as known at ``as_of``.

An operational view like ``bars_status``; strategy code must not import it. The tests of the
ingest code only prove that the code does what its author imagined (4b-2 had 347 green tests
and wrong data); this module looks at the real files for the shapes we have actually met --
BK's history filed under BNY, a split with no price move, a one-day 40x print on a warrant, a
rename filed weeks after the old name stopped trading -- and counts them.

Every check yields rows keyed by a short string (``AAA:2020-01-07``, ``S000730``, ``AAA/BBB``)
so a human can except one in ``configs/audit_exceptions.yaml`` with a reason. A row is *in the
liquid scope* when the security, or any segment of the symbol, belongs to the liquid-tier union
the caller passes in. Gated checks fail the run when rows are left outside the exceptions:
``gate="all"`` anywhere, ``gate="liquid"`` inside the liquid scope (or anywhere with
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
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from asof.ingest.bars import BAR_SCHEMA
from asof.ingest.corporate_actions import NAME_CHANGE_SCHEMA
from asof.ingest.instrument_class import CLASS_FILE
from asof.ingest.sec_ticker_history import HISTORY_SCHEMA
from asof.ingest.security_master import (
    CONFLICT_SCHEMA,
    CONFLICTS_FILE,
    DUP_MIN_VOLUME,
    MASTER_FILE,
    SEC_HISTORY_START,
    drop_table_corrections,
    normalize_spelling,
)
from asof.store.market import _files, _source, visible_bars, visible_master, visible_splits

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
            "back-to-back segments of a symbol known at two instants",
            "all",
            f"{_SYM}:{_DATE}",
        ),  # fmt: skip
        _spec("M2", "segments of one security overlap", "liquid", f"{_SID}:{_DATE}"),
        _spec("M3", "gap between segments of one security", None, f"{_SID}:{_DATE}"),
        _spec("M4", "master and class table disagree", "all", f"{_SYM}:{_DATE}"),
        _spec("M5", "instrument class unknown", "liquid", f"{_SYM}:{_DATE}"),
        _spec(
            "M6",
            "identity conflicts the build could not settle",
            "liquid",
            f"{_SYM}(/{_SYM})*:{_KIND}:{_DATE}(,{_DATE})*",
        ),  # fmt: skip
        _spec("I1", "traded bars no segment owns (loss)", None, _SYM),
        _spec("I2", "two securities share an identical bar", "liquid", f"{_SYM}/{_SYM}"),
        _spec("I3", "one security traded under two symbols on one day", "liquid", _SID),
        _spec("I4", "old label traded between its cut and the day the cut was known", None, _SYM),
        _spec("I5", "segment CIK contradicted by the SEC table", None, f"{_SYM}:{_DATE}"),
        _spec("I6", "volume-0 placeholder bars per year", None, _YEAR),
        _spec("I7", "session with almost nothing traded", "all", _DATE),
        _spec("I8", "security silent for more than 10 sessions", None, f"{_SID}:{_DATE}"),
        _spec("I9", "price discontinuity across a segment change", None, f"{_SID}:{_DATE}"),
        _spec("S1", "split record with no bar on its ex-date", "liquid", f"{_SYM}:{_DATE}"),
        _spec("S2", "split record without the price move", "liquid", f"{_SYM}:{_DATE}"),
        _spec("S3", "one-day thin-volume spike, no split", "liquid", f"{_SYM}:{_DATE}"),
        _spec("S4", "close jump of 3x without a split", None, f"{_SYM}:{_DATE}"),
        _spec(
            "N1", "rename recorded long after the old name stopped", None, f"{_SYM}/{_SYM}:{_DATE}"
        ),  # fmt: skip
        _spec("N2", "both names traded after the rename date", None, f"{_SYM}/{_SYM}:{_DATE}"),
        _spec("N3", "symbol pairs partly identical", None, f"{_SYM}/{_SYM}"),
        _spec("C1", "coverage per year", None, _YEAR),
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
    notes: dict[str, int | float]

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

_REGISTERED = (_RES, _MASTER, _MASTER_FILE, _CLASS, _SPLITS, _NAMES, _SEC, _LIQ_IDS, _QUAL_IDS)
_TEMP_TABLES = (_LIQ_SYMS, _SESSIONS, _TRADED, _ORPHANS, _SPLIT_DAYS)


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

    # -- setup -----------------------------------------------------------------------------------

    def load(self, liquid_ids: Collection[str], qualified_ids: Collection[str]) -> None:
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
        con.register(_SEC, self._sec())
        con.register(_LIQ_IDS, pa.table({"security_id": pa.array(sorted(liquid_ids), pa.string())}))
        con.register(
            _QUAL_IDS, pa.table({"security_id": pa.array(sorted(qualified_ids), pa.string())})
        )

        files = _files(root, "1Day")
        if files:
            con.execute(
                f"CREATE OR REPLACE TEMP VIEW {_RAW} AS "
                f"SELECT * REPLACE (CAST(year AS BIGINT) AS year) FROM {_source(files)} "
                f"WHERE available_at <= TIMESTAMPTZ '{as_of.isoformat()}'"
            )
        else:
            empty = BAR_SCHEMA.append(pa.field("year", pa.int64())).empty_table()
            con.register("_audit_raw_empty", empty)
            con.execute(f"CREATE OR REPLACE TEMP VIEW {_RAW} AS SELECT * FROM _audit_raw_empty")

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
        return drop_table_corrections(normalize_spelling(pq.read_table(path)))

    def close(self) -> None:
        for name in (*_REGISTERED, "_audit_raw_empty"):
            try:
                self.con.unregister(name)
            except duckdb.Error:
                pass
        self.con.execute(f"DROP VIEW IF EXISTS {_RAW}")
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
              ON a.symbol = b.symbol AND a.valid_to = b.valid_from
            WHERE a.end_available_at IS DISTINCT FROM b.available_at
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

    def m6(self) -> tuple[pa.Table, dict]:
        path = self.root / CONFLICTS_FILE
        conflicts = pq.read_table(path) if path.exists() else CONFLICT_SCHEMA.empty_table()
        rows = []
        for c in conflicts.to_pylist():
            symbols = list(c["symbols"] or [])
            rows.append(
                {
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
            )
        schema = pa.schema(
            [
                ("key", pa.string()),
                ("_liq", pa.bool_()),
                ("kind", pa.string()),
                ("symbols", pa.string()),
                ("dates", pa.string()),
                ("detail", pa.string()),
            ]
        )
        return pa.Table.from_pylist(rows, schema=schema), {}

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
            GROUP BY a.symbol, b.symbol
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

    # -- S: splits and price jumps ---------------------------------------------------------------

    def s1(self) -> tuple[pa.Table, dict]:
        table = self._q(
            f"""
            WITH stored AS (SELECT DISTINCT symbol FROM {_RAW}),
            sp AS (SELECT DISTINCT symbol, ex_date, kind, factor FROM {_SPLITS})
            SELECT sp.symbol || ':' || {_day("sp.ex_date")} AS key, {_liq_sym("sp.symbol")} AS _liq,
                   sp.symbol, sp.ex_date, sp.kind, sp.factor,
                   CASE WHEN sp.symbol IN (SELECT symbol FROM stored) THEN 'no_bar_that_day'
                        ELSE 'symbol_never_stored' END AS reason
            FROM sp ANTI JOIN {_RAW} r ON r.symbol = sp.symbol AND r.session_date = sp.ex_date
            -- a split is known from 00:00 of its ex-date, its bar only from 20:00
            WHERE sp.ex_date <= (SELECT max(session_date) FROM {_SESSIONS})
            """
        )
        reasons = table.column("reason").to_pylist()
        notes = {
            "symbol_never_stored": reasons.count("symbol_never_stored"),
            "no_bar_that_day": reasons.count("no_bar_that_day"),
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
        table = self._q(
            f"""
            SELECT sp.symbol || ':' || {_day("sp.ex_date")} AS key, {_liq_sym("sp.symbol")} AS _liq,
                   sp.symbol, sp.ex_date, sp.kind, sp.factor, t.prev_close, t.open,
                   t.open / t.prev_close AS observed, 1.0 / sp.factor AS expected
            FROM (SELECT DISTINCT symbol, ex_date, kind, factor FROM {_SPLITS}
                  WHERE kind <> 'unit_split' AND factor > 0) sp
            JOIN {traded} t ON t.symbol = sp.symbol AND t.session_date = sp.ex_date
            WHERE {self._has_prev()}
              AND abs((t.open / t.prev_close) * sp.factor - 1) > {SPLIT_TOLERANCE}
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
                   coalesce(nc.name_changes_linked, 0) AS name_changes_linked
            FROM y LEFT JOIN b USING (year) LEFT JOIN nc USING (year) LEFT JOIN liq USING (year)
            """
        )
        notes = {
            "name_changes": sum(table.column("name_changes").to_pylist()),
            "name_changes_linked": sum(table.column("name_changes_linked").to_pylist()),
        }
        return table, notes


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
) -> list[CheckResult]:
    """Run ``checks`` (all by default) over the store under ``root`` as known at ``as_of``, in
    ``CHECKS`` order. ``liquid_ids`` is the liquid-tier union (``tier_union``).

    ``qualified_ids`` is the same union taken over every instrument class (``classes=None``)
    and scopes M5 only: a security whose class is unknown can never be in the liquid tier, so
    judged against ``liquid_ids`` an unknown class would never fail the liquid gate. Defaults
    to ``liquid_ids``."""
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
        audit.load(liquid_ids, liquid_ids if qualified_ids is None else qualified_ids)
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
