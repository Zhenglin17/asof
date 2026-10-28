"""Spin-offs as of a point in time: the vendor's records, and what each one does to the parent.

A spin-off hands the parent's holders shares of a new company: GE holders received one GEV
share per four GE shares on 2024-04-02, and GE opened about 20% lower. Without an adjustment
that drop reads as a crash in the parent's history; held through it, a position books a loss
the holder never had.

``spinoff_events`` turns the records into one decision per (parent security, ex-date):

- ``adjust``: every child traded soon enough to price it. The parent's prices before the
  ex-date are multiplied by ``factor = (prev_close - sum(child_close * ratio)) / prev_close``,
  the first child close being the price of what was handed out.
- ``restart``: the event cannot be priced (no child bar, a child listed too late, parent and
  child on one symbol, no parent price, no ratio, two ratios for one child, a factor outside
  (0, 1)), or the prices disprove the factor: the parent's ex-session close plus the children's
  value is more than ``MAX_TOTAL_RETURN`` away from the previous close (``factor_disproved``,
  the row keeps its ``total_return``). Readers start the parent's series again at the ex-date
  instead of guessing.
- ``pending``: not decidable yet at ``as_of`` (the child has not traded and the waiting window
  is still open).

``known_on`` is the day the decision became possible; a reader must not use it earlier, which
is what makes a one-shot computation over all history equal the day-by-day one.
"""

import functools
from collections.abc import Collection, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd
import pyarrow as pa

from asof.store.market import (
    _actions_file,
    _aware_utc,
    _source,
    load_split_rejections,
    visible_bars,
    visible_master,
)

ET = ZoneInfo("America/New_York")
SPINOFF_REJECTIONS_FILE = (
    Path(__file__).resolve().parents[3] / "configs" / "spinoff_rejections.yaml"
)
MAX_CHILD_DELAY = 20  # parent sessions; real lags: 2-13 sessions vs 35+ (MPTI.WS, TOPS -> RUBI)
# |ex-date total return| above this disproves the factor (HON 2026-06-29 +45.5%, BBIG 2022-06-30
# -19.6% as TYDE fell 3.10 -> 2.03 on its first day); real liquid adjusts sit within +-7%
MAX_TOTAL_RETURN = 0.15
_STALE_DAYS = 7  # as features/bx.py: an earlier holder of the label, not the parent

SPINOFF_SCHEMA = pa.schema(
    [
        ("parent_symbol", pa.string()),
        ("child_symbol", pa.string()),
        ("ex_date", pa.date32()),
        ("ratio", pa.float64()),
        ("available_at", pa.timestamp("us", tz="UTC")),
    ]
)
EVENT_SCHEMA = pa.schema(
    [
        ("security_id", pa.string()),
        ("parent_symbol", pa.string()),
        ("child_symbols", pa.list_(pa.string())),
        ("ex_date", pa.date32()),
        ("ratio", pa.list_(pa.float64())),
        ("parent_prev_session", pa.date32()),
        ("parent_prev_close", pa.float64()),
        ("child_session", pa.date32()),
        ("child_closes", pa.list_(pa.float64())),
        ("factor", pa.float64()),
        ("total_return", pa.float64()),
        ("status", pa.string()),
        ("reason", pa.string()),
        ("known_on", pa.date32()),
    ]
)


# -- rejections -----------------------------------------------------------------------------------


@functools.lru_cache(maxsize=4)
def _shipped(path: Path, mtime_ns: int, size: int) -> frozenset[tuple[str, date]]:
    return load_split_rejections(path)


def _rejections(rejected: Collection[tuple[str, date]] | None) -> Collection[tuple[str, date]]:
    """The shipped file is required: without it HON 2026-06-29 would rescale HON's history."""
    if rejected is not None:
        return rejected
    path = SPINOFF_REJECTIONS_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; pass rejected=() to read every record")
    stat = path.stat()
    return _shipped(path, stat.st_mtime_ns, stat.st_size)


# -- records --------------------------------------------------------------------------------------


def visible_spinoffs(
    con: duckdb.DuckDBPyConnection,
    root: Path,
    as_of: datetime,
    symbols: Sequence[str] | None = None,
    *,
    rejected: Collection[tuple[str, date]] | None = None,
) -> pa.Table:
    """Spin-off records known at ``as_of`` (``available_at``, 00:00 New York of the ex-date, not
    after it), ``symbols`` filtering on the parent. Rows equal in (parent, child, ex_date,
    ratio) come back once, the earliest knowable copy: the feed stores some twice (OPEN ->
    OPENW 2025-11-18), and a reader would pay the child out twice. Every row of a rejected
    (parent, ex_date) is left out at any ``as_of``: ``rejected=None`` reads
    ``SPINOFF_REJECTIONS_FILE``, an explicit collection replaces it."""
    as_of_utc = _aware_utc(as_of, "as_of")
    skip = sorted(f"{symbol}:{day.isoformat()}" for symbol, day in _rejections(rejected))
    path = _actions_file(root, "others")
    if path is None:
        return SPINOFF_SCHEMA.empty_table()
    clauses = ["kind = 'spin_off'", "available_at <= $as_of"]
    params: dict[str, Any] = {"as_of": as_of_utc}
    if symbols is not None:
        clauses.append("source_symbol IN (SELECT UNNEST($symbols))")
        params["symbols"] = list(symbols)
    if skip:
        clauses.append(
            "source_symbol || ':' || strftime(ex_date, '%Y-%m-%d') NOT IN (SELECT UNNEST($skip))"
        )
        params["skip"] = skip
    con.execute("SET TimeZone = 'UTC'")
    query = f"""
        SELECT source_symbol AS parent_symbol, new_symbol AS child_symbol, ex_date,
               rate AS ratio, available_at
        FROM {_source([path])} WHERE {" AND ".join(clauses)}
        QUALIFY row_number() OVER (
            PARTITION BY source_symbol, new_symbol, ex_date, round(rate, 9) ORDER BY available_at
        ) = 1
        ORDER BY parent_symbol, ex_date, child_symbol
    """
    return con.execute(query, params).to_arrow_table().cast(SPINOFF_SCHEMA)


# -- events ---------------------------------------------------------------------------------------


def _ny_date(value: Any) -> date:
    return value.astimezone(ET).date()  # a tz-aware pandas Timestamp


def _child_key(child: tuple[str, float | None]) -> tuple[str, float]:
    return (child[0], -1.0 if child[1] is None else child[1])


def _parent_security(by_sid: dict[str, pd.DataFrame], symbol: str, ex: date) -> str | None:
    """The security carrying ``symbol`` at the ex-date boundary: its last bar before ``ex`` has
    the symbol (and it still trades, or stopped within a week), else its first bar on/after
    ``ex`` has it (a record filed under the new name of a rename day)."""
    before: list[tuple[date, str]] = []
    after: list[str] = []
    for sid, bars in by_sid.items():
        days = bars["session_date"]
        prior = bars.loc[days < ex]
        later = bars.loc[days >= ex]
        if not prior.empty and prior["symbol"].iloc[-1] == symbol:
            last = prior["session_date"].iloc[-1]
            if not later.empty or (ex - last).days <= _STALE_DAYS:
                before.append((last, sid))
        elif not later.empty and later["symbol"].iloc[0] == symbol:
            after.append(sid)
    if before:
        return max(before)[1]
    return min(after) if after else None


def _decide(
    parent: pd.DataFrame | None,
    symbols: set[str],
    ex: date,
    children: list[tuple[str, float | None]],
    child_bars: dict[str, pd.DataFrame],
    max_child_delay: int,
    records_known: date,
    max_total_return: float = MAX_TOTAL_RETURN,
) -> dict[str, Any]:
    """One event's columns other than the identity ones; see the module docstring. ``symbols``
    are the parent symbols the event's records name; ``records_known`` is the New York day the
    last of its records became visible."""
    names = [c for c, _ in children]
    out: dict[str, Any] = {
        "child_symbols": names,
        "ratio": [r for _, r in children],
        "parent_prev_session": None,
        "parent_prev_close": None,
        "child_session": None,
        "child_closes": [],
        "factor": None,
        "total_return": None,
        "status": "restart",
        "reason": None,
        "known_on": ex,
    }
    prior = parent.loc[parent["session_date"] < ex] if parent is not None else None
    prev_known = ex
    if prior is not None and not prior.empty:
        out["parent_prev_session"] = prior["session_date"].iloc[-1]
        out["parent_prev_close"] = float(prior["close"].iloc[-1])
        prev_known = _ny_date(prior["available_at"].iloc[-1])
    if symbols & set(names):
        return {**out, "reason": "same_symbol"}
    if out["parent_prev_close"] is None:
        return {**out, "reason": "no_parent_price"}
    if any(r is None for _, r in children):
        return {**out, "reason": "no_ratio"}
    if len(set(names)) < len(names):
        # one child at two ratios: taking either (the newer may be visible only later) guesses
        return {**out, "reason": "conflicting_ratio"}

    assert parent is not None
    sessions = parent.loc[parent["session_date"] >= ex, "session_date"].tolist()
    window = sessions[max_child_delay] if len(sessions) > max_child_delay else None
    firsts = []  # (session, close, the day it became knowable) of each child's first bar
    for name in names:
        bars = child_bars.get(name)
        bars = bars.loc[bars["session_date"] >= ex] if bars is not None else None
        if bars is None or bars.empty:
            firsts.append(None)
            continue
        first = bars.iloc[0]
        session = first["session_date"]
        firsts.append(
            (session, float(first["close"]), max(session, _ny_date(first["available_at"])))
        )

    priced = all(f is not None for f in firsts)
    # the factor needs every input: each child's first bar, the records and the previous close
    known = max([*(f[2] for f in firsts if f is not None), records_known, prev_known])
    if priced and (window is None or known <= window):
        closes = [f[1] for f in firsts if f is not None]
        prev = out["parent_prev_close"]
        paid = sum(
            c * float(r) for c, (_, r) in zip(closes, children, strict=True) if r is not None
        )
        factor = (prev - paid) / prev
        out.update(
            child_session=max(f[0] for f in firsts if f is not None),
            child_closes=closes,
            known_on=known,
        )
        if not 0 < factor < 1:
            return {**out, "reason": "factor_out_of_range"}
        # a parent with no bar from the ex-date on has nothing to adjust and no return to check
        later = parent.loc[parent["session_date"] >= ex]
        if later.empty:
            # the check needs the parent's ex-session close; a parent that never trades again
            # stays pending, which costs nothing: it has no later session to adjust
            return {
                **out,
                "child_session": None,
                "child_closes": [],
                "status": "pending",
                "known_on": None,
            }
        total = (float(later["close"].iloc[0]) + paid) / prev - 1
        # the check reads the parent's ex-session close: decided once that bar is visible too
        out["known_on"] = max(known, _ny_date(later["available_at"].iloc[0]))
        if abs(total) > max_total_return:
            return {**out, "total_return": total, "reason": "factor_disproved"}
        return {**out, "factor": factor, "total_return": total, "status": "adjust"}
    if window is not None:
        reason = "child_too_late" if priced else "no_child_price"
        return {**out, "reason": reason, "known_on": window}
    return {**out, "status": "pending", "known_on": None}


def spinoff_events(
    con: duckdb.DuckDBPyConnection,
    root: Path,
    as_of: datetime,
    parents: Sequence[str] | None = None,
    *,
    rejected: Collection[tuple[str, date]] | None = None,
    max_child_delay: int = MAX_CHILD_DELAY,
    max_total_return: float = MAX_TOTAL_RETURN,
) -> pa.Table:
    """One row per (parent security, ex-date) of the records visible at ``as_of``, decided from
    the bars visible at ``as_of`` (``EVENT_SCHEMA``). Several children of one day are one event.
    The child delay counts the parent's own sessions, the ex-date's being number 0; a child bar
    counts from the later of its session and the New York day it became visible."""
    records = visible_spinoffs(con, root, as_of, parents, rejected=rejected).to_pylist()
    if not records:
        return EVENT_SCHEMA.empty_table()
    parent_symbols = sorted({r["parent_symbol"] for r in records})
    child_symbols = sorted({r["child_symbol"] for r in records})

    # bars under the parents' and children's symbols, then every bar of each security that
    # carried a parent symbol (a parent renamed on its ex-date trades on under another name)
    first = visible_bars(
        con, root, "1Day", as_of, symbols=sorted({*parent_symbols, *child_symbols})
    ).to_pandas()
    sids = sorted(set(first.loc[first["symbol"].isin(parent_symbols), "security_id"]))
    master = visible_master(con, root, as_of).to_pandas()
    names = sorted(set(master.loc[master["security_id"].isin(sids), "symbol"]))
    bars = visible_bars(con, root, "1Day", as_of, symbols=names).to_pandas() if names else first
    bars = bars[bars["security_id"].isin(sids)].sort_values(["security_id", "session_date"])
    by_sid = {sid: g.reset_index(drop=True) for sid, g in bars.groupby("security_id", sort=True)}
    kids = first[first["symbol"].isin(child_symbols)].sort_values("session_date")
    child_bars = {c: g.reset_index(drop=True) for c, g in kids.groupby("symbol", sort=True)}

    # one event per (parent security, ex-date): a security named by two records of a day (old
    # and new name of a rename day) gets its children once, not two factors
    grouped: dict[tuple[str, date], list[dict[str, Any]]] = {}
    for r in records:
        symbol, ex = r["parent_symbol"], r["ex_date"]
        holders = {sid: b for sid, b in by_sid.items() if symbol in set(b["symbol"])}
        sid = _parent_security(holders, symbol, ex)
        grouped.setdefault((sid or f"symbol:{symbol}", ex), []).append({**r, "security_id": sid})
    rows = []
    for (_, ex), group in grouped.items():
        sid = group[0]["security_id"]
        symbols = {r["parent_symbol"] for r in group}
        children = sorted({(r["child_symbol"], r["ratio"]) for r in group}, key=_child_key)
        decided = _decide(
            by_sid.get(sid) if sid is not None else None,
            symbols,
            ex,
            children,
            child_bars,
            max_child_delay,
            max(_ny_date(r["available_at"]) for r in group),
            max_total_return,
        )
        row = {"security_id": sid, "parent_symbol": min(symbols), "ex_date": ex}
        rows.append({**row, **decided})
    rows.sort(key=lambda r: (r["parent_symbol"], r["ex_date"]))
    return pa.Table.from_pylist(rows, schema=EVENT_SCHEMA)
