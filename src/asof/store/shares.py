"""Shares outstanding and a rough market cap, as a reader at ``as_of`` could have known them.

A security reads the facts of the CIK its master segment carries on the night (the New York date
of ``as_of``), filed by then (``available_at``) and with a count no older than ``max_age_days``.
The first concept of ``CONCEPTS`` that has such a fact wins, then the latest ``end``; a later
filing breaks a tie (a restated count), then the larger accession number. A split between the
count's unit date (``unit_date``) and the night multiplies it: AAPL's 10-Q of 2020-07-31
counted 4.28bn shares as of 07-17, before the 4-for-1 of 08-31, so from that night the count is
17.1bn.

A 20-F filer reports its ordinary shares while what trades here may be an ADS standing for
several of them (TSM: 5) or a fraction of one (PKX: 1/4), and no ratio is on file:
``ratio_unknown`` marks such a count, and ``cap_passes`` holds it to a higher bar. A 40-F filer
(Canadian) lists its own shares, so its cap is read like a domestic one.
"""

import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from asof.ingest.shares import CONCEPTS, SHARES_FILE
from asof.store.market import visible_bars, visible_master, visible_sessions, visible_splits

NEW_YORK = ZoneInfo("America/New_York")
COVER_CONCEPT = "{}:{}".format(*CONCEPTS[0])
_PRIORITY = {f"{taxonomy}:{tag}": i for i, (taxonomy, tag) in enumerate(CONCEPTS)}
_ANNUAL_FORMS = frozenset({"10-K", "10-K/A", "10-KT", "20-F", "20-F/A", "40-F", "40-F/A"})
_FOREIGN_FORMS = frozenset({"20-F", "20-F/A", "6-K", "6-K/A"})
SHARES_COLUMNS = pa.schema(
    [
        ("security_id", pa.string()),
        ("symbol", pa.string()),
        ("cik", pa.int64()),
        ("concept", pa.string()),
        ("form", pa.string()),
        ("end", pa.date32()),
        ("filed", pa.date32()),
        ("shares", pa.float64()),
        ("split_factor", pa.float64()),
        ("age_days", pa.int64()),
        ("ratio_unknown", pa.bool_()),
    ]
)
CAP_COLUMNS = SHARES_COLUMNS.append(pa.field("close", pa.float64())).append(
    pa.field("raw_cap", pa.float64())
)


def _aware(as_of: datetime) -> datetime:
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    return as_of


def _night(as_of: datetime) -> date:
    return as_of.astimezone(NEW_YORK).date()


@lru_cache(maxsize=4)
def _facts_cached(path: str, mtime_ns: int, size: int) -> dict[int, list[dict[str, Any]]]:
    """The shares file grouped by CIK; the file's mtime and size key a rewritten file anew."""
    by_cik: dict[int, list[dict[str, Any]]] = {}
    for row in pq.read_table(path).to_pylist():
        by_cik.setdefault(row["cik"], []).append(row)
    return by_cik


def share_facts(root: Path) -> dict[int, list[dict[str, Any]]]:
    """Every stored fact by CIK, public or not yet; empty when there is no shares file."""
    path = root / SHARES_FILE
    if not path.exists():
        return {}
    stat = path.stat()
    return _facts_cached(str(path), stat.st_mtime_ns, stat.st_size)


def _covers(seg: dict[str, Any], day: date) -> bool:
    return seg["valid_from"] <= day and (seg["valid_to"] is None or day < seg["valid_to"])


def split_factors(
    master: Sequence[dict[str, Any]], splits: pa.Table
) -> dict[str, list[tuple[date, float]]]:
    """security_id -> (ex_date, new/old) of each split record, given to the security that owned
    its symbol on the ex-date (as ``asof.evolve.null.build_panel`` does)."""
    by_symbol: dict[str, list[dict[str, Any]]] = {}
    for seg in master:
        by_symbol.setdefault(seg["symbol"], []).append(seg)
    out: dict[str, set[tuple[date, float]]] = {}
    for s in splits.to_pylist():
        for seg in by_symbol.get(s["symbol"], ()):
            if _covers(seg, s["ex_date"]):
                ratio = float(s["new_rate"]) / float(s["old_rate"])
                out.setdefault(seg["security_id"], set()).add((s["ex_date"], ratio))
    return {sid: sorted(rows) for sid, rows in out.items()}


def unit_date(fact: dict[str, Any]) -> date:
    """The day whose share unit a count is in. A cover count is as of its ``end``; an average
    is restated for every split before its filing (SHOP's 2021 average, 0.12bn in the 10-K of
    2022-02, is 1.25bn in that of 2023-02 after the 10-for-1 of 2022-06), so it is in the unit
    of ``filed``."""
    return fact["end"] if fact["concept"] == COVER_CONCEPT else fact["filed"]


def factor_between(rows: Sequence[tuple[date, float]], after: date, through: date) -> float:
    """The product of the ratios with ``after < ex_date <= through``."""
    return math.prod(r for d, r in rows if after < d <= through)


def ratio_unknown(facts: Sequence[dict[str, Any]], chosen: dict[str, Any], as_of: datetime) -> bool:
    """Whether the company files as a foreign private issuer: its latest annual report public at
    ``as_of`` is a 20-F (a count tagged in its 6-K is still of ordinary shares; 4,665 such facts,
    measured 2026-10-29), or, before any annual report, the chosen count came with a 20-F or
    6-K. A company moving to the 10-K is read as domestic from its first 10-K."""
    annual = [f for f in facts if f["available_at"] <= as_of and f["form"] in _ANNUAL_FORMS]
    if annual:
        latest = max(annual, key=lambda f: (f["filed"], f["accn"]))
        return latest["form"].startswith("20-F")
    return chosen["form"] in _FOREIGN_FORMS


def choose_fact(
    facts: Sequence[dict[str, Any]], as_of: datetime, night: date, max_age_days: int
) -> dict[str, Any] | None:
    eligible = [
        f
        for f in facts
        if f["available_at"] <= as_of
        and (night - f["end"]).days <= max_age_days
        and f["concept"] in _PRIORITY
    ]
    if not eligible:
        return None
    top = min(_PRIORITY[f["concept"]] for f in eligible)
    return max(
        (f for f in eligible if _PRIORITY[f["concept"]] == top),
        # same end and filing: the shortest period (a quarter over the half-year it ends)
        key=lambda f: (f["end"], f["filed"], f.get("start") or date.min, f["accn"]),
    )


def _segments_on(
    master: Sequence[dict[str, Any]], night: date, wanted: set[str]
) -> dict[str, dict[str, Any]]:
    return {
        seg["security_id"]: seg
        for seg in master
        if seg["security_id"] in wanted and _covers(seg, night)
    }


def visible_shares(
    con: duckdb.DuckDBPyConnection,
    root: Path,
    as_of: datetime,
    security_ids: Sequence[str],
    *,
    max_age_days: int,
) -> pa.Table:
    """One row per security of ``security_ids`` with a usable count on the night."""
    as_of = _aware(as_of)
    night = _night(as_of)
    by_cik = share_facts(root)
    if not by_cik:
        return SHARES_COLUMNS.empty_table()
    master = visible_master(con, root, as_of).to_pylist()
    segments = _segments_on(master, night, set(security_ids))
    factors = split_factors(master, visible_splits(con, root, as_of))
    rows = []
    for sid in security_ids:
        seg = segments.get(sid)
        if seg is None or seg["cik"] is None:
            continue
        chosen = choose_fact(by_cik.get(int(seg["cik"]), ()), as_of, night, max_age_days)
        if chosen is None:
            continue
        factor = factor_between(factors.get(sid, ()), unit_date(chosen), night)
        rows.append(
            {
                "security_id": sid,
                "symbol": seg["symbol"],
                "cik": int(seg["cik"]),
                "concept": chosen["concept"],
                "form": chosen["form"],
                "end": chosen["end"],
                "filed": chosen["filed"],
                "shares": chosen["shares"] * factor,
                "split_factor": float(factor),
                "age_days": (night - chosen["end"]).days,
                "ratio_unknown": ratio_unknown(by_cik.get(int(seg["cik"]), ()), chosen, as_of),
            }
        )
    return pa.Table.from_pylist(rows, schema=SHARES_COLUMNS)


def market_caps(
    con: duckdb.DuckDBPyConnection,
    root: Path,
    as_of: datetime,
    security_ids: Sequence[str],
    *,
    max_age_days: int,
) -> pa.Table:
    """``visible_shares`` times the night's raw close; a security without a traded bar that
    session (or not visible yet at ``as_of``) is left out."""
    night = _night(as_of)
    if night not in visible_sessions(con, root, "1Day", as_of):
        raise ValueError(
            f"{night} is not a session whose bars are visible at {as_of}: read a cap at or "
            "after 20:00 New York of a session"
        )
    shares = visible_shares(con, root, as_of, security_ids, max_age_days=max_age_days)
    if shares.num_rows == 0:
        return CAP_COLUMNS.empty_table()
    start = datetime.combine(night, time(0), tzinfo=NEW_YORK)
    bars = visible_bars(
        con,
        root,
        "1Day",
        as_of,
        symbols=sorted(set(shares.column("symbol").to_pylist())),
        start=start,
        end=start + timedelta(days=1),
    )
    close = {
        b["security_id"]: float(b["close"]) for b in bars.to_pylist() if b["session_date"] == night
    }
    rows = [
        {**r, "close": close[r["security_id"]], "raw_cap": r["shares"] * close[r["security_id"]]}
        for r in shares.to_pylist()
        if r["security_id"] in close
    ]
    return pa.Table.from_pylist(rows, schema=CAP_COLUMNS)


def implausible(
    caps: pa.Table, *, adv_usd: Mapping[str, float], max_turnover: float, max_cap: float
) -> frozenset[str]:
    """Securities whose cap no count could give: one trading more than ``max_turnover`` of
    itself a day on average (``adv_usd`` / cap; VALE's IFRS count of 5.18m, in thousands, made
    a $305m-a-day stock worth $61m) or worth more than ``max_cap`` (BNTX's 2020 cover count).
    A real squeeze can trade more than its cap in a day too (NAKD 2021-01); such a stock is far
    below any cap bar either way."""
    out = set()
    for r in caps.to_pylist():
        sid = r["security_id"]
        if sid not in adv_usd:
            raise ValueError(f"{sid}: no average dollar volume to judge its cap by")
        if adv_usd[sid] > max_turnover * r["raw_cap"] or r["raw_cap"] > max_cap:
            out.add(sid)
    return frozenset(out)


def cap_passes(
    caps: pa.Table,
    *,
    adv_usd: Mapping[str, float],
    min_cap: float,
    min_raw_cap_ratio_unknown: float,
    max_turnover: float,
    max_cap: float,
) -> frozenset[str]:
    """The securities of ``adv_usd`` (every candidate) whose cap reaches ``min_cap``; a
    ``ratio_unknown`` cap (ordinary shares times the price of what trades here) must reach
    ``min_raw_cap_ratio_unknown`` instead. A security without a usable count (none in ``caps``,
    or an ``implausible`` one) passes when it trades ``max_turnover`` x ``min_cap`` a day: a
    smaller cap would turn over more than ``max_turnover`` of itself daily (V, BRK.B, STZ tag no
    all-class count; V trades about $2bn a day)."""
    bad = implausible(caps, adv_usd=adv_usd, max_turnover=max_turnover, max_cap=max_cap)
    usable = {r["security_id"]: r for r in caps.to_pylist() if r["security_id"] not in bad}
    out = set()
    for sid, adv in adv_usd.items():
        row = usable.get(sid)
        if row is None:
            if adv >= max_turnover * min_cap:
                out.add(sid)
        elif row["raw_cap"] >= (min_raw_cap_ratio_unknown if row["ratio_unknown"] else min_cap):
            out.add(sid)
    return frozenset(out)
