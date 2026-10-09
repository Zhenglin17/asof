"""Security master: which security each (symbol, day) in the bar store belongs to.

A ticker is a label, not an identity. The same label passes from one company to another (BBBY,
FFR), one security wears several labels over time (BK -> BNY), and Alpaca files a security's
whole history under whichever of its names it is asked for, so the raw store holds BK's 2020
bars under BNY as well. This module turns three sources into one table of **segments** --
``(symbol, valid_from, valid_to) -> security_id`` -- that the read path applies to bars:

* Alpaca ``name_change`` records: exact dates and CUSIPs, but they miss about a third of the
  renames the SEC history shows (2026-10-06 probe).
* SEC ticker snapshots: the same CIK carrying ticker A in one snapshot and B in the next; no
  exact date, but it covers what Alpaca misses.
* The bars themselves: identical OHLCV on the same day is reported as a conflict, never merged
  automatically (two unrelated companies can share one relabelled history).

CUSIPs ride along as attributes; they are not identities (a reverse split changes the CUSIP,
4,466 of 4,865 in the feed). Human decisions live in an overrides list applied last.

Three things the first real build (2026-10-07) taught, now rules:

* SEC snapshots before ``SEC_HISTORY_START`` are one frozen, stale file (MRNA listed as Marina
  Biotech; 597 tickers vanish at the 2019-10-02 refresh) and are not used at all.
* A snapshot pair in which ``TABLE_CORRECTION_MIN_SWITCHES`` or more tickers change CIK is the
  SEC correcting its table (142 between 2020-06-01 and 2020-07-10; normal pairs show at most
  14). A switching ticker whose old CIK never appears again was a zombie row (AFC as Allied
  Capital, acquired 2010) and loses its rows before the correction; an old CIK that lives on
  under another ticker (Arconic Inc -> HWM) is a real reorganisation and keeps its switch.
* A label taken over by a rename may carry, before the takeover, the taker's own history under
  the new name (BNY held BK's 2020-2026 bars while the SEC still listed BNY as a BlackRock fund).
  Such a piece is identical to the taker's old name on nearly every day and gets no segment.
  A takeover whose two sides share a CIK and join at a continuous price is one security
  (temporary tickers like 1ORIC -> ORIC, SPAC completions); different CIKs at a continuous
  price are reported for a human.

Every segment carries ``available_at``: the moment the event that created it became knowable
(00:00 New York of the rename's effective day, of the SEC snapshot, or the first bar's
availability for a symbol's opening segment). A reader at ``as_of`` filters on it like on any
other table. A segment's end is a fact of its own with its own instant, ``end_available_at``
(the rename, takeover or CIK switch that closed it; null while the segment is open): a reader in
2022 must see BK as open although the file says it ends 2026-05-21, so ``visible_master`` masks
``valid_to`` until that instant. A closed segment and the one that replaces it become knowable
together, at the same instant, because they are two sides of one event.
"""

import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from asof.ingest.corporate_actions import NAME_CHANGE_SCHEMA

ET = ZoneInfo("America/New_York")
TS_UTC = pa.timestamp("us", tz="UTC")

EDGE_SCHEMA = pa.schema(
    [
        ("old_symbol", pa.string()),
        ("new_symbol", pa.string()),
        ("date", pa.date32()),
        ("date_lower", pa.date32()),
        ("evidence", pa.string()),
        ("available_at", TS_UTC),
        ("old_cusip", pa.string()),
        ("new_cusip", pa.string()),
    ]
)
MASTER_SCHEMA = pa.schema(
    [
        ("security_id", pa.string()),
        ("symbol", pa.string()),
        ("valid_from", pa.date32()),
        ("valid_to", pa.date32()),
        ("cik", pa.int64()),
        ("cusip", pa.string()),
        ("evidence", pa.string()),
        ("available_at", TS_UTC),
        ("end_available_at", TS_UTC),  # when the end became knowable; null while open
    ]
)
CONFLICT_SCHEMA = pa.schema(
    [
        ("kind", pa.string()),
        ("symbols", pa.list_(pa.string())),
        ("dates", pa.list_(pa.date32())),
        ("detail", pa.string()),
    ]
)
DUP_SCHEMA = pa.schema([("symbol_a", pa.string()), ("symbol_b", pa.string()), ("days", pa.int64())])

DUP_MIN_VOLUME = 1000  # thin names can match by accident; identical days above this cannot
CONTINUOUS = (0.8, 1.25)  # close after / close before: a rename or CIK switch, not a new security
MAX_GAP_DAYS = 10  # bars this far apart are not "continuous" whatever the ratio
SEC_HISTORY_START = date(2019, 10, 2)  # first refreshed SEC snapshot; earlier ones are stale
TABLE_CORRECTION_MIN_SWITCHES = 50  # CIK switches in one snapshot pair; normal pairs: <= 14
RELABEL_MIN_IDENTICAL = 0.9  # share of a pre-takeover piece's days identical to the taker's bars
SEC_LAG_MAX_DAYS = 183  # an SEC snapshot may record a rename this long after Alpaca's date
SEC_GAP_MAX_SNAPSHOTS = 1  # a CIK absent from more snapshots than this was dropped, not renamed


def _midnight(day: date) -> datetime:
    return datetime.combine(day, time(0, 0), tzinfo=ET).astimezone(UTC)


# -- SEC spelling --------------------------------------------------------------------------------


def normalize_spelling(sec_history: pa.Table) -> pa.Table:
    """Same CIK, same letters once dots are removed -> one spelling, the dotted one.

    The SEC wrote ``BRKB`` until 2020 and ``BRK.B`` afterwards; treating them as two tickers
    would turn a spelling change into a rename.
    """
    tickers: list[str] = sec_history.column("ticker").to_pylist()
    ciks: list[int] = sec_history.column("cik").to_pylist()
    spellings: dict[tuple[int, str], set[str]] = {}
    for ticker, cik in zip(tickers, ciks, strict=True):
        spellings.setdefault((cik, ticker.replace(".", "")), set()).add(ticker)
    canonical = {
        key: sorted(group, key=lambda s: ("." not in s, s))[0] for key, group in spellings.items()
    }
    fixed = [canonical[(cik, t.replace(".", ""))] for t, cik in zip(tickers, ciks, strict=True)]
    return sec_history.set_column(
        sec_history.schema.get_field_index("ticker"), "ticker", pa.array(fixed, pa.string())
    )


# -- edges -------------------------------------------------------------------------------------


def _edge_table(rows: list[dict[str, Any]]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=EDGE_SCHEMA)


def edges_from_name_changes(name_changes: pa.Table) -> pa.Table:
    """One edge per Alpaca rename that changes the symbol; CUSIP-only changes are not edges."""
    rows = []
    for r in name_changes.to_pylist():
        if r["old_symbol"] == r["new_symbol"]:
            continue
        rows.append(
            {
                "old_symbol": r["old_symbol"],
                "new_symbol": r["new_symbol"],
                "date": r["process_date"],
                "date_lower": None,
                "evidence": "name_change",
                "available_at": r.get("available_at") or _midnight(r["process_date"]),
                "old_cusip": r.get("old_cusip"),
                "new_cusip": r.get("new_cusip"),
            }
        )
    return _edge_table(rows)


def _by_cik(sec_history: pa.Table) -> dict[int, dict[date, set[str]]]:
    out: dict[int, dict[date, set[str]]] = {}
    for r in sec_history.select(["snapshot_date", "ticker", "cik"]).to_pylist():
        out.setdefault(r["cik"], {}).setdefault(r["snapshot_date"], set()).add(r["ticker"])
    return out


def edges_from_sec(sec_history: pa.Table) -> pa.Table:
    """Same CIK, one ticker gone and one new between two consecutive snapshots -> a rename
    somewhere in ``(date_lower, date]``. Delistings and CIK switches produce no edge.

    Two things that look like renames are not: the SEC file flickering between two labels of
    one filer (the old ticker is listed again later, or the new one had been listed before --
    VIAC vanished from the 2020-07-10 snapshot while CBS reappeared; 264 of 639 such edges on
    the real file), and a filer that was absent from the table for more than
    ``SEC_GAP_MAX_SNAPSHOTS`` snapshots and came back under another ticker (a relisting:
    BULL 2023-06 ... BLSH 2025-08).
    """
    all_snaps = sorted(set(sec_history.column("snapshot_date").to_pylist()))
    rows = []
    for _cik, by_snapshot in _by_cik(sec_history).items():
        snaps = sorted(by_snapshot)
        for i, (prev, cur) in enumerate(zip(snaps, snaps[1:], strict=False)):
            gone = by_snapshot[prev] - by_snapshot[cur]
            came = by_snapshot[cur] - by_snapshot[prev]
            if len(gone) == 1 and len(came) == 1:
                old, new = next(iter(gone)), next(iter(came))
                if any(old in by_snapshot[d] for d in snaps[i + 2 :]):
                    continue  # flicker: the old label comes back under this filer
                if any(new in by_snapshot[d] for d in snaps[:i]):
                    continue  # flicker: the new label was this filer's before
                if sum(1 for d in all_snaps if prev < d < cur) > SEC_GAP_MAX_SNAPSHOTS:
                    continue  # the filer left the table and came back: a relisting
                rows.append(
                    {
                        "old_symbol": old,
                        "new_symbol": new,
                        "date": cur,
                        "date_lower": prev,
                        "evidence": "sec",
                        "available_at": _midnight(cur),
                        "old_cusip": None,
                        "new_cusip": None,
                    }
                )
    rows.sort(key=lambda r: (r["date"], r["old_symbol"], r["new_symbol"]))
    return _edge_table(rows)


def merge_edges(alpaca_edges: pa.Table, sec_edges: pa.Table) -> pa.Table:
    """One row per (old, new). Alpaca's exact date wins; seen in both -> evidence ``both``.

    Alpaca sometimes files one rename twice (BIGT -> MAGS at 2023-11-09 and 2025-02-03): the
    earliest date is the event. An SEC-only edge into a label that Alpaca shows being created
    (by any old name) up to ``SEC_LAG_MAX_DAYS`` before the snapshot is the same event seen
    late (Alpaca HWM.WI -> HWM 2020-04-01, SEC ARNC -> HWM in the 2020-07-10 snapshot): it
    takes Alpaca's date so the label is not started twice.
    """
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    from_sec: set[tuple[str, str]] = set()
    for r in sec_edges.to_pylist():
        merged[(r["old_symbol"], r["new_symbol"])] = r
        from_sec.add((r["old_symbol"], r["new_symbol"]))
    for r in alpaca_edges.to_pylist():
        key = (r["old_symbol"], r["new_symbol"])
        if key in from_sec:
            from_sec.discard(key)
            merged[key] = {**r, "evidence": "both"}
        elif key not in merged:
            merged[key] = r
        elif r["date"] < merged[key]["date"]:  # a duplicate row never moves the date later
            merged[key] = {**r, "evidence": merged[key]["evidence"]}
    alpaca_starts: dict[str, list[date]] = {}  # after de-duplication: one date per rename
    for (_old, new), r in merged.items():
        if r["evidence"] != "sec":
            alpaca_starts.setdefault(new, []).append(r["date"])
    for key, r in merged.items():
        if r["evidence"] != "sec":
            continue
        old, new = key
        # The SEC saw the new label appear (Alpaca dates its creation), or saw the old label
        # leave (Alpaca dates another filer taking it: Inhibrx's INXB -> INBX is the moment the
        # old INBX became INBXV). Either way the snapshot is late and Alpaca's day is the cut.
        same_event = [
            d
            for d in alpaca_starts.get(new, []) + alpaca_starts.get(old, [])
            if d <= r["date"] and (r["date"] - d).days <= SEC_LAG_MAX_DAYS
        ]
        if same_event:
            merged[key] = {**r, "date": max(same_event), "date_lower": None}
    rows = sorted(merged.values(), key=lambda r: (r["date"], r["old_symbol"], r["new_symbol"]))
    return _edge_table(rows)


# -- duplicates ----------------------------------------------------------------------------------


def ohlcv_duplicates(bars: pa.Table, con: duckdb.DuckDBPyConnection | None = None) -> pa.Table:
    """Pairs of symbols with identical open/high/low/close/volume on the same day, counted over
    days with volume above ``DUP_MIN_VOLUME``."""
    own = con is None
    con = con or duckdb.connect()
    try:
        con.register("_dup_bars", bars)
        table = con.execute(
            f"""
            SELECT a.symbol AS symbol_a, b.symbol AS symbol_b, count(*) AS days
            FROM _dup_bars a JOIN _dup_bars b
              ON a.session_date = b.session_date AND a.symbol < b.symbol
             AND a.open = b.open AND a.high = b.high AND a.low = b.low
             AND a.close = b.close AND a.volume = b.volume
            WHERE a.volume > {DUP_MIN_VOLUME}
            GROUP BY 1, 2 ORDER BY 3 DESC, 1, 2
            """
        ).to_arrow_table()
        con.unregister("_dup_bars")
    finally:
        if own:
            con.close()
    return table.cast(DUP_SCHEMA) if table.num_rows else DUP_SCHEMA.empty_table()


# -- segments ------------------------------------------------------------------------------------


@dataclass
class _Edge:
    old: str
    new: str
    date: date  # the cut as finally used (SEC-only edges may be refined by the old name's bars)
    evidence: str
    available_at: datetime
    old_cusip: str | None
    new_cusip: str | None


@dataclass
class _Segment:
    symbol: str
    valid_from: date | None  # None until the first bar is known
    valid_to: date | None
    start: _Edge | None = None  # the rename that created it
    end: _Edge | None = None  # the rename that ended it
    cik_cut: bool = False  # created by a CIK switch inside one symbol
    cik: int | None = None
    cusip: str | None = None
    evidence: str = "bars_only"
    available_at: datetime | None = None
    end_available_at: datetime | None = None  # set exactly when valid_to is
    n_bars: int = 0
    chain: int = -1
    key: int = field(default=-1)


class _Union:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _symbol_stats(con: duckdb.DuckDBPyConnection) -> dict[str, tuple[date, date]]:
    rows = con.execute(
        "SELECT symbol, min(session_date), max(session_date) FROM _bars GROUP BY 1"
    ).fetchall()
    return {s: (lo, hi) for s, lo, hi in rows}


def _refine_sec_edges(con: duckdb.DuckDBPyConnection, edges: list[dict[str, Any]]) -> None:
    """For SEC-only edges the cut is the day after the old name's last traded bar inside the
    snapshot interval: Alpaca stops filing real bars under the old name on the rename day,
    which pins the date better than a snapshot that may be months later. No bar there -> keep
    the bound. "Traded" = volume > 0: after a rename Alpaca keeps writing volume-0 bars with a
    frozen close under the old name for years (VTIQ at 33.97 from 2020-06-04 to 2022-09-21)."""
    todo = [e for e in edges if e["evidence"] == "sec" and e["date_lower"] is not None]
    if not todo:
        return
    con.register(
        "_sec_edges",
        pa.table(
            {
                "old_symbol": [e["old_symbol"] for e in todo],
                "date_lower": pa.array([e["date_lower"] for e in todo], pa.date32()),
                "date": pa.array([e["date"] for e in todo], pa.date32()),
            }
        ),
    )
    rows = con.execute(
        """
        SELECT e.old_symbol, e.date, max(b.session_date)
        FROM _sec_edges e JOIN _bars b ON b.symbol = e.old_symbol
         AND b.session_date > e.date_lower AND b.session_date <= e.date AND b.volume > 0
        GROUP BY 1, 2
        """
    ).fetchall()
    con.unregister("_sec_edges")
    last_bar = {(s, d): last for s, d, last in rows}
    for e in todo:
        last = last_bar.get((e["old_symbol"], e["date"]))
        if last is not None and last + timedelta(days=1) < e["date"]:
            e["date"] = last + timedelta(days=1)


def _refine_alpaca_edges(con: duckdb.DuckDBPyConnection, edges: list[dict[str, Any]]) -> None:
    """Alpaca's ``process_date`` can trail the first trading day under the new name by a few
    days (NKLA traded from 2020-06-04, the VTIQ -> NKLA record says 06-08). When the old name's
    last traded bar (volume > 0; see ``_refine_sec_edges`` on filler bars) is within
    ``MAX_GAP_DAYS`` before the record and the new name already has bars in between, the cut is
    the day after that last bar. The edge's ``available_at`` is left alone:
    the rename was knowable only when Alpaca filed it."""
    todo = [e for e in edges if e["evidence"] != "sec"]
    if not todo:
        return
    con.register(
        "_alp_edges",
        pa.table(
            {
                "old_symbol": [e["old_symbol"] for e in todo],
                "new_symbol": [e["new_symbol"] for e in todo],
                "date": pa.array([e["date"] for e in todo], pa.date32()),
            }
        ),
    )
    rows = con.execute(
        f"""
        WITH last_old AS (
            SELECT e.old_symbol, e.new_symbol, e.date, max(b.session_date) AS last_bar
            FROM _alp_edges e JOIN _bars b ON b.symbol = e.old_symbol
             AND b.session_date < e.date AND b.volume > 0
            GROUP BY 1, 2, 3
        )
        SELECT l.old_symbol, l.new_symbol, l.date, l.last_bar
        FROM last_old l
        WHERE l.date - l.last_bar - 1 BETWEEN 1 AND {MAX_GAP_DAYS}
          AND EXISTS (
              SELECT 1 FROM _bars n
              WHERE n.symbol = l.new_symbol
                AND n.session_date > l.last_bar AND n.session_date < l.date
          )
        """
    ).fetchall()
    con.unregister("_alp_edges")
    moved = {(o, n, d): last + timedelta(days=1) for o, n, d, last in rows}
    for e in todo:
        new_date = moved.get((e["old_symbol"], e["new_symbol"], e["date"]))
        if new_date is not None:
            e["date"] = new_date


def _sec_presence(sec_history: pa.Table) -> dict[str, list[tuple[date, int]]]:
    out: dict[str, list[tuple[date, int]]] = {}
    for r in sec_history.select(["snapshot_date", "ticker", "cik"]).to_pylist():
        out.setdefault(r["ticker"], []).append((r["snapshot_date"], r["cik"]))
    for rows in out.values():
        rows.sort()
    return out


def _cik_switches(presence: list[tuple[date, int]]) -> list[tuple[date, date]]:
    """``(prev_snapshot, snapshot]`` intervals in which a symbol's CIK set changed completely."""
    by_snap: dict[date, set[int]] = {}
    for day, cik in presence:
        by_snap.setdefault(day, set()).add(cik)
    snaps = sorted(by_snap)
    return [
        (prev, cur)
        for prev, cur in zip(snaps, snaps[1:], strict=False)
        if not (by_snap[prev] & by_snap[cur])
    ]


def drop_table_corrections(sec_history: pa.Table) -> pa.Table:
    """Remove zombie rows exposed by a mass correction of the SEC table.

    A consecutive snapshot pair in which ``TABLE_CORRECTION_MIN_SWITCHES`` or more tickers change
    CIK is the SEC fixing its file, not that many simultaneous events. For each switching ticker
    whose old CIK is in no snapshot from the correction on, the rows before the correction are
    dropped (the later CIK applied all along). Old CIKs that continue under another ticker are
    real reorganisations and are left alone.
    """
    presence = _sec_presence(sec_history)
    by_pair: dict[tuple[date, date], list[str]] = {}
    for ticker, rows in presence.items():
        for lower, upper in _cik_switches(rows):
            by_pair.setdefault((lower, upper), []).append(ticker)
    snapshot_dates: list[date] = sec_history.column("snapshot_date").to_pylist()
    ciks: list[int] = sec_history.column("cik").to_pylist()
    drop_before: dict[str, date] = {}
    for (lower, upper), tickers in by_pair.items():
        if len(tickers) < TABLE_CORRECTION_MIN_SWITCHES:
            continue
        later = {c for d, c in zip(snapshot_dates, ciks, strict=True) if d >= upper}
        for ticker in tickers:
            old = {c for d, c in presence[ticker] if d == lower}
            if not old & later:
                drop_before[ticker] = max(upper, drop_before.get(ticker, upper))
    if not drop_before:
        return sec_history
    tickers_col: list[str] = sec_history.column("ticker").to_pylist()
    keep = [
        not (t in drop_before and d < drop_before[t])
        for t, d in zip(tickers_col, snapshot_dates, strict=True)
    ]
    return sec_history.filter(pa.array(keep, pa.bool_()))


def _pieces_for(
    symbol: str,
    first_bar: date,
    starts: list[_Edge],
    ends: list[_Edge],
    presence: list[tuple[date, int]],
    ciks_of: Mapping[str, set[int]],
) -> list[_Segment]:
    """Cut one symbol's timeline at every rename touching it and every CIK switch, and decide
    which pieces are real: a label is dead between being vacated and being taken, and history
    filed under a label before it was ever in use (Alpaca's relabelling) is no segment.

    A CIK switch in the same snapshot interval as a rename into the label is the rename seen
    by the SEC only if the renamer (``ciks_of[old name]``) is the filer the label now carries;
    otherwise two things happened (Inhibrx: INXB -> INBX, then a new holding company took INBX)
    and the switch stays an event.
    """
    events: list[
        tuple[date, int, _Edge | None]
    ] = []  # (day, kind, edge); kind 0 end, 1 start, 2 cik
    for e in ends:
        events.append((e.date, 0, e))
    for e in starts:
        events.append((e.date, 1, e))
    for lower, upper in _cik_switches(presence):
        now = {cik for day, cik in presence if day == upper}
        explained = any(lower < e.date <= upper and ciks_of.get(e.old, set()) & now for e in starts)
        if not explained:
            events.append((upper, 2, None))
    events.sort(key=lambda t: (t[0], t[1]))

    # Renames dated on or before the first bar only settle the label's state when our bars
    # begin; they cannot end a piece that has not started.
    pre = [ev for ev in events if ev[1] != 2 and ev[0] <= first_bar]
    rest = [ev for ev in events if not (ev[1] != 2 and ev[0] <= first_bar)]
    current: _Segment | None
    if pre:
        day, kind, edge = pre[-1]
        if kind == 1:  # taken before our bars: the taker holds it from that day
            current = _Segment(symbol, day, None, start=edge)
        elif any(k == 1 for _, k, _ in rest):  # vacated, taken later: dead until then
            current = None
        else:  # vacated before our bars yet never retaken: nobody's relabelled history can
            # sit under this label, so the vacate cannot be right and the bars are its own
            current = _Segment(symbol, first_bar, None)
    else:
        first_start = min((e.date for e in starts), default=None)
        first_end = min((e.date for e in ends), default=None)
        # Vacated strictly before being taken -> it was in use before our bars (FFR). A same-day
        # round trip (ORIC -> 1ORIC -> ORIC) says nothing about the label's past: 1ORIC only
        # lives if the SEC ever listed it, otherwise its bars are ORIC's relabelled history.
        alive = (
            first_start is None
            or (first_end is not None and first_end < first_start)
            or any(day < first_start for day, _ in presence)
        )
        current = _Segment(symbol, first_bar, None) if alive else None
    pieces: list[_Segment] = []
    for day, kind, edge in rest:
        if kind == 0:  # vacated
            if current is not None:
                current.valid_to, current.end = day, edge
                pieces.append(current)
                current = None
        elif kind == 1:  # taken
            if current is not None:  # taken while in use: the previous holder's piece ends
                current.valid_to = day
                pieces.append(current)
            current = _Segment(symbol, day, None, start=edge)
        else:  # CIK switch: cuts a live label; on a dead one a new filer has taken it
            if current is not None:
                current.valid_to = day
                pieces.append(current)
            current = _Segment(symbol, day, None, cik_cut=True)
    # The end of a piece is knowable when the event that ends it is: the rename's or the
    # takeover's record, or midnight of the SEC snapshot that shows the new CIK -- the same
    # instant at which the piece that follows becomes knowable. Several events may fall on one
    # day (Arconic Inc vacated ARNC for HWM, seen by the SEC on 2020-07-10, while Alpaca recorded
    # ARNC.WI -> ARNC on 04-01 itself; WBD got two takeover records on 2022-04-11): the earliest
    # of them already tells a reader the label changed hands that day, so it is the instant.
    ends_known: dict[date, datetime] = {}
    for day, _kind, edge in rest:
        known = edge.available_at if edge is not None else _midnight(day)
        ends_known[day] = min(ends_known.get(day, known), known)
    for p in pieces:
        if p.valid_to is not None:
            p.end_available_at = ends_known[p.valid_to]
    if current is not None:
        pieces.append(current)
    return pieces


def _fill_bar_stats(con: duckdb.DuckDBPyConnection, pieces: list[_Segment]) -> None:
    con.register(
        "_pieces",
        pa.table(
            {
                "key": pa.array([p.key for p in pieces], pa.int64()),
                "symbol": [p.symbol for p in pieces],
                "lo": pa.array([p.valid_from for p in pieces], pa.date32()),
                "hi": pa.array([p.valid_to for p in pieces], pa.date32()),
            }
        ),
    )
    rows = (
        con.execute(
            """
        SELECT p.key, count(*), min(b.session_date), min(b.available_at)
        FROM _pieces p JOIN _bars b ON b.symbol = p.symbol
         AND (p.lo IS NULL OR b.session_date >= p.lo)
         AND (p.hi IS NULL OR b.session_date < p.hi)
        GROUP BY 1
        """
        )
        .to_arrow_table()
        .to_pylist()
    )
    con.unregister("_pieces")
    stats = {r["key"]: tuple(r.values())[1:] for r in rows}
    for p in pieces:
        n, first, avail = stats.get(p.key, (0, None, None))
        p.n_bars = n
        if p.valid_from is None and first is not None:
            p.valid_from = first
        if p.start is not None:
            p.available_at = p.start.available_at
        elif p.cik_cut and p.valid_from is not None:
            p.available_at = _midnight(p.valid_from)
        elif avail is not None:
            p.available_at = avail.replace(tzinfo=UTC) if avail.tzinfo is None else avail


def _relabelled_pieces(con: duckdb.DuckDBPyConnection, pieces: list[_Segment]) -> set[int]:
    """Keys of pieces that are the taker's own history filed under a label it took later.

    For every takeover (a piece opened by a rename) each earlier piece of the same label is
    compared with the taker's old name day by day; identical OHLCV on at least
    ``RELABEL_MIN_IDENTICAL`` of the piece's days means Alpaca relabelling, not a previous
    holder. Real: BNY carried BK's bars while the SEC listed BNY as a BlackRock fund.
    """
    by_symbol: dict[str, list[_Segment]] = {}
    for p in pieces:
        by_symbol.setdefault(p.symbol, []).append(p)
    cands: list[tuple[int, str, str, date, date, int]] = []
    for p in pieces:
        if p.start is None or p.valid_from is None:
            continue
        for q in by_symbol[p.symbol]:
            if q.valid_from is None or q.valid_to is None or not q.n_bars:
                continue
            if q.valid_to > p.valid_from:
                continue
            # The taker's old name is already tied to q by an edge (a round trip ORIC -> 1ORIC
            # -> ORIC, or the same rename filed twice): q is the security's own history, and
            # the old name's identical bars are the relabelled copy, not the other way round.
            if (q.end is not None and q.end.new == p.start.old) or (
                q.start is not None and q.start.old == p.start.old
            ):
                continue
            cands.append((q.key, q.symbol, p.start.old, q.valid_from, q.valid_to, q.n_bars))
    if not cands:
        return set()
    con.register(
        "_cands",
        pa.table(
            {
                "key": pa.array([c[0] for c in cands], pa.int64()),
                "symbol": [c[1] for c in cands],
                "other": [c[2] for c in cands],
                "lo": pa.array([c[3] for c in cands], pa.date32()),
                "hi": pa.array([c[4] for c in cands], pa.date32()),
            }
        ),
    )
    rows = con.execute(
        """
        SELECT c.key, count(*)
        FROM _cands c
        JOIN _bars a ON a.symbol = c.symbol AND a.session_date >= c.lo AND a.session_date < c.hi
        JOIN _bars b ON b.symbol = c.other AND b.session_date = a.session_date
         AND b.open = a.open AND b.high = a.high AND b.low = a.low
         AND b.close = a.close AND b.volume = a.volume
        GROUP BY 1
        """
    ).fetchall()
    con.unregister("_cands")
    identical = dict(rows)
    return {
        key
        for key, _s, _o, _lo, _hi, n_bars in cands
        if identical.get(key, 0) >= RELABEL_MIN_IDENTICAL * n_bars
    }


def _closes_around(
    con: duckdb.DuckDBPyConnection, cuts: list[tuple[str, date]]
) -> dict[tuple[str, date], tuple[date | None, float | None, date | None, float | None]]:
    """Last traded (date, close) before each cut and first at or after it. Volume-0 filler bars
    (frozen close) would make any delisted label look continuous with its next holder."""
    if not cuts:
        return {}
    con.register(
        "_cuts",
        pa.table(
            {"symbol": [s for s, _ in cuts], "cut": pa.array([d for _, d in cuts], pa.date32())}
        ),
    )
    rows = con.execute(
        """
        SELECT c.symbol, c.cut,
               max(b.session_date) FILTER (WHERE b.session_date < c.cut),
               arg_max(b.close, b.session_date) FILTER (WHERE b.session_date < c.cut),
               min(b.session_date) FILTER (WHERE b.session_date >= c.cut),
               arg_min(b.close, b.session_date) FILTER (WHERE b.session_date >= c.cut)
        FROM _cuts c JOIN _bars b ON b.symbol = c.symbol AND b.volume > 0
        GROUP BY 1, 2
        """
    ).fetchall()
    con.unregister("_cuts")
    return {(s, cut): (d0, c0, d1, c1) for s, cut, d0, c0, d1, c1 in rows}


def _continuous(around: tuple[date | None, float | None, date | None, float | None]) -> bool:
    d0, c0, d1, c1 = around
    if d0 is None or d1 is None or not c0 or c1 is None:
        return False
    if (d1 - d0).days > MAX_GAP_DAYS:
        return False
    return CONTINUOUS[0] <= c1 / c0 <= CONTINUOUS[1]


def _segment_cik(seg: _Segment, presence: list[tuple[date, int]]) -> int | None:
    inside = [
        cik
        for day, cik in presence
        if (seg.valid_from is None or day >= seg.valid_from)
        and (seg.valid_to is None or day < seg.valid_to)
    ]
    if inside:
        return inside[-1]
    before = [cik for day, cik in presence if seg.valid_from is not None and day < seg.valid_from]
    # A label's opening segment may predate our bars; its SEC rows then sit before valid_from.
    # Only trust them when nothing says the label changed hands since (no start edge).
    if before and seg.start is None and not seg.cik_cut:
        return before[-1]
    return None


def apply_overrides(
    master: pa.Table, overrides: Sequence[Mapping[str, Any]]
) -> pa.Table:  # pragma: no cover - thin wrapper kept for the documented API
    """Overrides act on chains, so they are applied inside ``build_security_master``; this
    re-runs that step on a finished table (merge only)."""
    rows = master.to_pylist()
    for o in overrides:
        if "merge" in o:
            sids = {r["security_id"] for r in rows if r["symbol"] in set(o["merge"])}
            if len(sids) > 1:
                target = min(sids)
                for r in rows:
                    if r["security_id"] in sids:
                        r["security_id"] = target
                        r["evidence"] = "override"
    return pa.Table.from_pylist(rows, schema=MASTER_SCHEMA)


def build_security_master(
    *,
    bars: pa.Table,
    sec_history: pa.Table,
    name_changes: pa.Table,
    overrides: Iterable[Mapping[str, Any]] = (),
    sec_history_start: date | None = None,
) -> tuple[pa.Table, pa.Table]:
    """Build the master and the conflicts it could not settle. Deterministic for equal inputs.

    SEC snapshots dated before ``sec_history_start`` are ignored entirely; ``build_from_store``
    passes ``SEC_HISTORY_START``.
    """
    overrides = list(overrides)
    if sec_history_start is not None:
        snapshot_dates: list[date] = sec_history.column("snapshot_date").to_pylist()
        sec_history = sec_history.filter(
            pa.array([d >= sec_history_start for d in snapshot_dates], pa.bool_())
        )
    con = duckdb.connect()
    try:
        con.execute("SET TimeZone = 'UTC'")
        con.register("_bars", bars)
        sec = drop_table_corrections(normalize_spelling(sec_history))
        presence = _sec_presence(sec)
        ciks_of = {symbol: {cik for _, cik in rows} for symbol, rows in presence.items()}
        stats = _symbol_stats(con)

        edge_rows = merge_edges(
            edges_from_name_changes(name_changes), edges_from_sec(sec)
        ).to_pylist()
        edge_rows = [e for e in edge_rows if e["old_symbol"] in stats or e["new_symbol"] in stats]
        _refine_sec_edges(con, edge_rows)
        _refine_alpaca_edges(con, edge_rows)
        edges = [
            _Edge(
                e["old_symbol"],
                e["new_symbol"],
                e["date"],
                e["evidence"],
                e["available_at"],
                e["old_cusip"],
                e["new_cusip"],
            )
            for e in edge_rows
        ]
        starts: dict[str, list[_Edge]] = {}
        ends: dict[str, list[_Edge]] = {}
        for e in edges:
            starts.setdefault(e.new, []).append(e)
            ends.setdefault(e.old, []).append(e)

        pieces: list[_Segment] = []
        for symbol in sorted(stats):
            first_bar, _ = stats[symbol]
            for p in _pieces_for(
                symbol,
                first_bar,
                starts.get(symbol, []),
                ends.get(symbol, []),
                presence.get(symbol, []),
                ciks_of,
            ):
                p.key = len(pieces)
                pieces.append(p)
        _fill_bar_stats(con, pieces)
        # A piece with no bars attributes nothing; keep it only as a link in a chain.
        pieces = [p for p in pieces if p.n_bars or (p.start is not None and p.end is not None)]
        pieces = [p for p in pieces if p.valid_from is not None]
        relabelled = _relabelled_pieces(con, pieces)
        pieces = [p for p in pieces if p.key not in relabelled]
        for i, p in enumerate(pieces):
            p.key = i
            # Attributes describe how the segment STARTED only. The event that ends it is
            # recorded on the segment it creates; copying it here (its cusip, its evidence) would
            # let a reader who may not yet know the end see that one is coming.
            p.cik = _segment_cik(p, presence.get(p.symbol, []))
            p.cusip = p.start.new_cusip if p.start else None
            if p.start is not None:
                p.evidence = p.start.evidence
            elif p.cik is not None:
                p.evidence = "sec"

        by_start = {(p.symbol, p.valid_from): p for p in pieces if p.start is not None or p.cik_cut}
        by_end = {(p.symbol, p.valid_to): p for p in pieces if p.valid_to is not None}
        uf = _Union(len(pieces))
        conflicts: list[dict[str, Any]] = []
        for e in edges:
            a, b = by_end.get((e.old, e.date)), by_start.get((e.new, e.date))
            # Two edges may start one label on one day (HWM.WI -> HWM and ARNC -> HWM, both
            # 2020-04-01); the piece belongs to the day, whichever edge created it.
            if a is not None and b is not None and a.end is e:
                uf.union(a.key, b.key)

        # CIK switches inside one label and takeovers of a label still in use: bridge or report
        # by looking at the prices on both sides of the cut.
        cik_cuts = [(p.symbol, p.valid_from) for p in pieces if p.cik_cut and p.valid_from]
        takeovers = [
            (p.symbol, p.valid_from)
            for p in pieces
            if p.start is not None and p.valid_from and (p.symbol, p.valid_from) in by_end
        ]
        around = _closes_around(con, cik_cuts + takeovers)
        for symbol, cut in cik_cuts:
            before, after = by_end.get((symbol, cut)), by_start[(symbol, cut)]
            if before is None:
                continue
            if before.end is not None:
                # The previous holder left by renaming (ARNC -> HWM); the label's continuation
                # is another filer however close the prices: never bridged, reported if close.
                if _continuous(around[(symbol, cut)]):
                    d0, c0, d1, c1 = around[(symbol, cut)]
                    conflicts.append(
                        {
                            "kind": "reuse_looks_continuous",
                            "symbols": [symbol],
                            "dates": [d for d in (d0, cut, d1) if d is not None],
                            "detail": f"cik {before.cik} -> {after.cik} after {symbol} -> "
                            f"{before.end.new}, close {c0} -> {c1}: relabelled?",
                        }
                    )
                continue
            if _continuous(around[(symbol, cut)]):
                uf.union(before.key, after.key)
            else:
                d0, c0, d1, c1 = around[(symbol, cut)]
                conflicts.append(
                    {
                        "kind": "cik_switch_discontinuous",
                        "symbols": [symbol],
                        "dates": [d for d in (d0, cut, d1) if d is not None],
                        "detail": f"cik {before.cik} -> {after.cik}, close {c0} -> {c1}",
                    }
                )
        for symbol, cut in takeovers:
            before, after = by_end[(symbol, cut)], by_start[(symbol, cut)]
            if not _continuous(around[(symbol, cut)]):
                continue  # a plain reuse: two securities, nothing to say
            d0, c0, d1, c1 = around[(symbol, cut)]
            if before.cik is not None and before.cik == after.cik:
                uf.union(before.key, after.key)  # temporary ticker, SPAC completion, ...
            else:
                conflicts.append(
                    {
                        "kind": "reuse_looks_continuous",
                        "symbols": [after.start.old if after.start else symbol, symbol],
                        "dates": [d for d in (d0, cut, d1) if d is not None],
                        "detail": f"cik {before.cik} -> {after.cik}, close {c0} -> {c1}: "
                        "relabelled?",
                    }
                )

        overridden: set[int] = set()
        for o in overrides:
            if "merge" in o:
                keys = [p.key for p in pieces if p.symbol in set(o["merge"])]
                for k in keys[1:]:
                    uf.union(keys[0], k)
                overridden.update(keys)
        for p in pieces:
            p.chain = uf.find(p.key)
            if p.key in overridden:
                p.evidence = "override"

        # Fill a segment's CIK from its chain when its own label never reached the SEC table --
        # only from members knowable no later than the segment itself, or a reader would learn
        # the filer from a future rename.
        chains: dict[int, list[_Segment]] = {}
        for p in pieces:
            chains.setdefault(p.chain, []).append(p)
        for members in chains.values():
            for p in members:
                if p.cik is not None:
                    continue
                known = [
                    q
                    for q in members
                    if q.cik is not None
                    and q.available_at is not None
                    and p.available_at is not None
                    and q.available_at <= p.available_at
                ]
                if known:
                    p.cik = min(known, key=lambda q: abs((q.valid_from - p.valid_from).days)).cik  # type: ignore[operator]

        # Stable, content-derived ids: chains ordered by their earliest segment.
        order = sorted(
            chains,
            key=lambda c: min((p.valid_from, p.symbol) for p in chains[c]),  # type: ignore[type-var]
        )
        sid = {chain: f"S{i + 1:06d}" for i, chain in enumerate(order)}

        sids_by_symbol: dict[str, set[str]] = {}
        for p in pieces:
            sids_by_symbol.setdefault(p.symbol, set()).add(sid[p.chain])
        for r in ohlcv_duplicates(bars, con).to_pylist():
            a, b = r["symbol_a"], r["symbol_b"]
            if sids_by_symbol.get(a, set()) & sids_by_symbol.get(b, set()):
                continue
            conflicts.append(
                {
                    "kind": "dup_unlinked",
                    "symbols": [a, b],
                    "dates": [],
                    "detail": f"{r['days']} days of identical OHLCV, no rename links them",
                }
            )
    finally:
        con.close()

    rows = sorted(
        (
            {
                "security_id": sid[p.chain],
                "symbol": p.symbol,
                "valid_from": p.valid_from,
                "valid_to": p.valid_to,
                "cik": p.cik,
                "cusip": p.cusip,
                "evidence": p.evidence,
                "available_at": p.available_at,
                "end_available_at": p.end_available_at,
            }
            for p in pieces
        ),
        key=lambda r: (r["security_id"], r["valid_from"], r["symbol"]),
    )
    conflicts.sort(key=lambda c: (c["kind"], c["symbols"]))
    return (
        pa.Table.from_pylist(rows, schema=MASTER_SCHEMA),
        pa.Table.from_pylist(conflicts, schema=CONFLICT_SCHEMA),
    )


# -- store integration ---------------------------------------------------------------------------

MASTER_FILE = "symbols/security_master.parquet"
CONFLICTS_FILE = "symbols/identity_conflicts.parquet"


def load_overrides(path: Path) -> list[dict[str, Any]]:
    """``configs/security_overrides.yaml``: a list of human decisions, each with a reason."""
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text()) or []
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a list of overrides")
    for item in raw:
        if not isinstance(item, dict) or "reason" not in item:
            raise ValueError(f"{path}: every override needs a reason: {item!r}")
    return raw


def build_from_store(
    market_root: Path, overrides: Sequence[Mapping[str, Any]] = ()
) -> tuple[pa.Table, pa.Table]:
    """Build from the daily bars, the SEC history and the Alpaca name changes on disk."""
    con = duckdb.connect()
    try:
        con.execute("SET TimeZone = 'UTC'")
        bars_glob = str(market_root / "bars_1d" / "**" / "*.parquet").replace("'", "''")
        bars = con.execute(
            "SELECT symbol, session_date, open, high, low, close, volume, available_at "
            f"FROM read_parquet('{bars_glob}')"
        ).to_arrow_table()
        sec = pq.read_table(market_root / "symbols" / "sec_history.parquet")
        changes_path = market_root / "corporate_actions" / "name_changes.parquet"
        name_changes = (
            pq.read_table(changes_path)
            if changes_path.exists()
            else pa.Table.from_pylist([], schema=NAME_CHANGE_SCHEMA)
        )
    finally:
        con.close()
    return build_security_master(
        bars=bars,
        sec_history=sec,
        name_changes=name_changes,
        overrides=overrides,
        sec_history_start=SEC_HISTORY_START,
    )


_CHECK_VIEW = "_asof_check_master"

# rule -> query counting its violations over the full, unmasked master (view _CHECK_VIEW)
_MASTER_RULES: dict[str, str] = {
    "null available_at": "SELECT count(*) FROM {v} WHERE available_at IS NULL",
    "valid_to without end_available_at or the reverse": (
        "SELECT count(*) FROM {v} WHERE (valid_to IS NULL) <> (end_available_at IS NULL)"
    ),
    "valid_to not after valid_from": "SELECT count(*) FROM {v} WHERE valid_to <= valid_from",
    "end_available_at before 00:00 New York of valid_to": (
        "SELECT count(*) FROM {v} "
        "WHERE end_available_at < (valid_to::TIMESTAMP AT TIME ZONE 'America/New_York')"
    ),
    "duplicate segment keys": (
        "SELECT count(*) FROM (SELECT security_id, symbol, valid_from FROM {v} "
        "GROUP BY ALL HAVING count(*) > 1)"
    ),
    "overlapping segments of one symbol": (
        "SELECT count(*) FROM {v} a JOIN {v} b ON a.symbol = b.symbol "
        "AND (a.valid_from < b.valid_from "
        "     OR (a.valid_from = b.valid_from AND a.security_id < b.security_id)) "
        "AND (a.valid_to IS NULL OR b.valid_from < a.valid_to)"
    ),
    "adjacent segments of one symbol disagree on the instant": (
        "SELECT count(*) FROM {v} a JOIN {v} b ON a.symbol = b.symbol "
        "AND a.valid_to = b.valid_from "
        "WHERE a.end_available_at IS DISTINCT FROM b.available_at"
    ),
}


def check_master(master: pa.Table) -> None:
    """Refuse a master that ``visible_master`` would trip over at some ``as_of``.

    Checked over the whole table, not the part visible at one instant: an overlap among
    segments nobody can see before 2030 is still an overlap. The last rule is the one a reader
    would only meet at a particular instant: when a label passes from one segment to the next on
    one day (BBBY 2025-08-29), the old end and the new start must become knowable at the same
    moment, or between the two instants the label is owned twice or by nobody (ARNC before the
    "earliest instant of the day" rule). Raises ``ValueError`` naming every broken rule with its
    count.
    """
    if master.num_rows == 0:
        return
    con = duckdb.connect()
    try:
        con.execute("SET TimeZone = 'UTC'")
        con.register(_CHECK_VIEW, master)
        broken = []
        for rule, query in _MASTER_RULES.items():
            row = con.execute(query.format(v=_CHECK_VIEW)).fetchone()
            n = int(row[0]) if row else 0
            if n:
                broken.append(f"{n} {rule}")
    finally:
        con.close()
    if broken:
        raise ValueError("security master refused: " + "; ".join(broken))


def write_master(master: pa.Table, conflicts: pa.Table, market_root: Path) -> None:
    """Write both files atomically, after ``check_master`` has accepted the master."""
    check_master(master)
    for table, name in ((master, MASTER_FILE), (conflicts, CONFLICTS_FILE)):
        path = market_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        try:
            pq.write_table(table, tmp)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
