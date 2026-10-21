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
availability for a symbol's opening segment), and never before its own first bar: a segment
that carries bars is made only from them, so a build at 00:00 of the event cannot yet produce
it (CXAIV -> CXAI is filed for 2023-03-15, CXAI's first bar is knowable at 20:00 that day). A
reader at ``as_of`` filters on it like on any other table. A segment's end is a fact of its own
with its own instant, ``end_available_at`` (the rename, takeover or CIK switch that closed it;
null while the segment is open): a reader in 2022 must see BK as open although the file says it
ends 2026-05-21, so ``visible_master`` masks ``valid_to`` until that instant. A closed segment's
end is knowable at the event; the segment that replaces it is knowable at the same instant or
later, at its first bar. In between a reader sees the old segment closed and the new one not at
all (ABAT: the SEC snapshot of 2023-10-03 shows the label under a new CIK, the new piece's first
bar is knowable that evening). The CIK is a third fact with its own instant,
``cik_available_at``: the SEC table may list a label months after it starts trading (AAAU: bars
from 2020-01-02, listed from 2020-07-10), and ``visible_master`` hides the CIK until then.
A CIK reached through another spelling is not known before the SEC writes the label's own
(BRK.B: ``BRKB`` until 2020-06-01, ``BRK.B`` from 2020-07-10, so 2020-07-10 and not 01-13).
"""

import os
import re
from collections.abc import Collection, Iterable, Mapping, Sequence
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
        ("cik_available_at", TS_UTC),  # when the CIK became knowable; null iff cik is
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
        ("resolution", pa.string()),  # what the build did about it, one of RESOLUTIONS
    ]
)
# new_security: the build kept two securities (a CIK switch with a price break); bridged: the
# build joined them; ignored: a record that is not an event; unresolved: for a human (the
# audit's M6 gates only these)
RESOLUTIONS = ("new_security", "bridged", "ignored", "unresolved", "relabelled")
IDENT_SCHEMA = pa.schema(
    [("symbol_a", pa.string()), ("symbol_b", pa.string()), ("session_date", pa.date32())]
)
DUP_SCHEMA = pa.schema([("symbol_a", pa.string()), ("symbol_b", pa.string()), ("days", pa.int64())])

DUP_MIN_VOLUME = 1000  # thin names can match by accident; identical days above this cannot
DUP_MIN_DAYS = 3  # one or two identical days are a coincidence (SPAC trusts all print 10.00)
CONTINUOUS = (0.8, 1.25)  # close after / close before: a rename or CIK switch, not a new security
MAX_GAP_DAYS = 10  # bars this far apart are not "continuous" whatever the ratio
SEC_HISTORY_START = date(2019, 10, 2)  # first refreshed SEC snapshot; earlier ones are stale
TABLE_CORRECTION_MIN_SWITCHES = 50  # CIK switches in one snapshot pair; normal pairs: <= 14
RELABEL_MIN_IDENTICAL = 0.9  # copy rules: share of a piece's traded days identical to another's
COPY_MIN_DAYS = 3  # fewer identical days never make a copy (one-day coincidences, the H cluster)
RUN_MIN_DAYS = 3  # E: shortest identical run that reads as an unfiled rename
RUN_NEAR_SESSIONS = 10  # E: ender's last own bar and continuer's first bar vs the run start
RUN_TOLERANCE = (3, 0.05)  # E: continuer's differing days before the run end <= max(3, 5% of run)
CONVERSION_MIN_DIFFERING_DAYS = 5  # B: parallel OHLC-differing days that make a conversion
CONVERSION_WINDOW = 5  # B: sessions before the record that must hold one of them
TAIL_GAP_SESSIONS = 10  # F: a copied tail that starts after a gap this long is cut off
SEC_LAG_MAX_DAYS = 183  # an SEC snapshot may record a rename this long after Alpaca's date
HANDOVER_MAX_DAYS = 5  # new name's first traded bar this close after the old name's last: the cut
RESUME_GAP_DAYS = 14  # traded bars of a closed label resuming later than this start a new piece
SEC_GAP_MAX_SNAPSHOTS = 1  # a CIK absent from more snapshots than this was dropped, not renamed
# When-issued symbols: NYSE ".WI" / ".RTWI", Nasdaq fifth letter V. They trade for a few sessions
# around a spin-off or a split and are never the security itself; the instrument-class rule file
# repeats this pattern (``sym_when_issued``) and a test keeps the two spellings equal.
WHEN_ISSUED = re.compile(r".*\.(WI|RTWI)$|[A-Z]{4}V")
BANKRUPT = re.compile(r"[A-Z]{4}Q")  # Nasdaq fifth letter Q: in bankruptcy proceedings


def _midnight(day: date) -> datetime:
    return datetime.combine(day, time(0, 0), tzinfo=ET).astimezone(UTC)


# -- SEC spelling --------------------------------------------------------------------------------


def normalize_spelling(
    sec_history: pa.Table, store_symbols: Collection[str] | None = None
) -> pa.Table:
    """Same CIK, same letters once dots are removed -> one spelling.

    The SEC wrote ``BRKB`` until 2020 and ``BRK.B`` afterwards; treating them as two tickers
    would turn a spelling change into a rename. The spelling kept is the one the bar store
    uses (``store_symbols``): the SEC writes Nasdaq units dotted once and undotted afterwards
    (``KINZ.U`` on 2020-11-03, ``KINZU`` 21 times) while Alpaca has ``KINZU``, and keeping the
    dotted one left ten unit labels without SEC rows. When the store has neither or both
    spellings, the dotted one wins.
    """
    tickers: list[str] = sec_history.column("ticker").to_pylist()
    ciks: list[int] = sec_history.column("cik").to_pylist()
    spellings: dict[tuple[int, str], set[str]] = {}
    for ticker, cik in zip(tickers, ciks, strict=True):
        spellings.setdefault((cik, ticker.replace(".", "")), set()).add(ticker)
    in_store = set(store_symbols) if store_symbols is not None else set()

    def pick(group: set[str]) -> str:
        stored = group & in_store
        if len(stored) == 1:
            return next(iter(stored))
        return sorted(group, key=lambda s: ("." not in s, s))[0]

    canonical = {key: pick(group) for key, group in spellings.items()}
    fixed = [canonical[(cik, t.replace(".", ""))] for t, cik in zip(tickers, ciks, strict=True)]
    return sec_history.set_column(
        sec_history.schema.get_field_index("ticker"), "ticker", pa.array(fixed, pa.string())
    )


def spelling_known_since(raw: pa.Table, normalized: pa.Table) -> dict[tuple[str, int], date]:
    """For each (ticker, cik) that took rows from another spelling: the first snapshot that
    writes the ticker itself. ``normalized`` is ``normalize_spelling(raw, ...)``, row for row.

    The join is only possible from then on. Until 2020-07-10 the SEC wrote only ``BRKB``, and
    letters alone do not prove a match (``LGCU`` is a stale Luna Gold row, the store's ``LGC.U``
    is Legacy Acquisition), so a CIK learned through a renamed row is not known before this day.
    """
    days: list[date] = raw.column("snapshot_date").to_pylist()
    before: list[str] = raw.column("ticker").to_pylist()
    after: list[str] = normalized.column("ticker").to_pylist()
    ciks: list[int] = raw.column("cik").to_pylist()
    renamed: set[tuple[str, int]] = set()
    first: dict[tuple[str, int], date] = {}
    for day, old, new, cik in zip(days, before, after, ciks, strict=True):
        if old != new:
            renamed.add((new, cik))
        elif (new, cik) not in first or day < first[(new, cik)]:
            first[(new, cik)] = day
    return {key: first[key] for key in renamed}


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


def ohlcv_duplicates(
    bars: pa.Table, con: duckdb.DuckDBPyConnection | None = None, owned: pa.Table | None = None
) -> pa.Table:
    """Pairs of symbols with identical open/high/low/close/volume on the same day, counted over
    days with volume above ``DUP_MIN_VOLUME``. With ``owned`` (``symbol, lo, hi`` ranges) only
    bars inside a range count: a copied tail the build has cut off belongs to nobody and has
    nothing left to link (the loss metric lists it instead)."""
    own = con is None
    con = con or duckdb.connect()
    try:
        con.register("_dup_bars", bars)
        inside = ""
        if owned is not None:
            con.register("_dup_owned", owned)
            inside = """
             AND EXISTS (SELECT 1 FROM _dup_owned o WHERE o.symbol = a.symbol
                         AND a.session_date >= o.lo AND (o.hi IS NULL OR a.session_date < o.hi))
             AND EXISTS (SELECT 1 FROM _dup_owned o WHERE o.symbol = b.symbol
                         AND b.session_date >= o.lo AND (o.hi IS NULL OR b.session_date < o.hi))
            """
        table = con.execute(
            f"""
            SELECT a.symbol AS symbol_a, b.symbol AS symbol_b, count(*) AS days
            FROM _dup_bars a JOIN _dup_bars b
              ON a.session_date = b.session_date AND a.symbol < b.symbol
             AND a.open = b.open AND a.high = b.high AND a.low = b.low
             AND a.close = b.close AND a.volume = b.volume
            WHERE a.volume > {DUP_MIN_VOLUME} {inside}
            GROUP BY 1, 2 ORDER BY 3 DESC, 1, 2
            """
        ).to_arrow_table()
        con.unregister("_dup_bars")
        if owned is not None:
            con.unregister("_dup_owned")
    finally:
        if own:
            con.close()
    return table.cast(DUP_SCHEMA) if table.num_rows else DUP_SCHEMA.empty_table()


def identical_days(bars: pa.Table, con: duckdb.DuckDBPyConnection | None = None) -> pa.Table:
    """One row per pair of symbols and day on which both traded (volume > 0) the same
    open/high/low/close, ``symbol_a < symbol_b``. Volume is ignored: Alpaca revises it after
    20:00 New York, so a relabelled copy can differ from its original on volume alone (the E
    cluster count found pairs identical on OHLC and not on OHLCV). Computed once per build and
    read by every copy rule (relabel, E, C, F, Z); the audit's I2 keeps its stricter OHLCV
    test because it gates on it."""
    own = con is None
    con = con or duckdb.connect()
    try:
        con.register("_id_bars", bars)
        table = con.execute(
            """
            SELECT a.symbol AS symbol_a, b.symbol AS symbol_b, a.session_date
            FROM _id_bars a JOIN _id_bars b
              ON a.session_date = b.session_date AND a.open = b.open AND a.high = b.high
             AND a.low = b.low AND a.close = b.close AND a.symbol < b.symbol
            WHERE a.volume > 0 AND b.volume > 0
            ORDER BY 1, 2, 3
            """
        ).to_arrow_table()
        con.unregister("_id_bars")
    finally:
        if own:
            con.close()
    return table.cast(IDENT_SCHEMA) if table.num_rows else IDENT_SCHEMA.empty_table()


def _register_sessions(con: duckdb.DuckDBPyConnection) -> None:
    """``_sessions`` (every weekday from the first traded bar to the last, numbered) and
    ``_traded`` (every traded bar with its session number): the copy rules count gaps in
    sessions, not calendar days. Weekdays rather than traded dates, so that a store holding few
    symbols (a test world, a partial backfill) does not see a whole quiet quarter as one
    session; exchange holidays count as sessions, which only shifts the thresholds by a day
    or two a year."""
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE _sessions AS
        SELECT session_date, row_number() OVER (ORDER BY session_date) AS n
        FROM (
            SELECT unnest(generate_series(lo, hi, INTERVAL 1 DAY))::DATE AS session_date
            FROM (SELECT min(session_date) AS lo, max(session_date) AS hi
                  FROM _bars WHERE volume > 0)
            WHERE lo IS NOT NULL
        )
        WHERE dayofweek(session_date) BETWEEN 1 AND 5
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE _traded AS
        SELECT b.symbol, b.session_date, s.n, b.available_at
        FROM _bars b JOIN _sessions s USING (session_date) WHERE b.volume > 0
        """
    )


def _symbol_components(edges: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """Symbol -> representative of its connected component over the linking edges."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in edges:
        if not e.get("links", True):
            continue
        a, b = find(e["old_symbol"]), find(e["new_symbol"])
        if a != b:
            parent[max(a, b)] = min(a, b)
    return {x: find(x) for x in list(parent)}


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
    links: bool = True  # False: the edge cuts the new label but joins no chain (when-issued)
    starts: bool = True  # False: the edge only ends its old label (a when-issued name's exit)


@dataclass
class _Segment:
    symbol: str
    valid_from: date | None  # None until the first bar is known
    valid_to: date | None
    start: _Edge | None = None  # the rename that created it
    end: _Edge | None = None  # the rename that ended it
    cik_cut: bool = False  # created by a CIK switch inside one symbol
    resumed: bool = False  # opened by bars resuming under a closed label (a new holder)
    cik: int | None = None
    cik_since: datetime | None = None  # the snapshot that first showed the CIK
    cusip: str | None = None
    evidence: str = "bars_only"
    available_at: datetime | None = None
    end_available_at: datetime | None = None  # set exactly when valid_to is
    n_bars: int = 0
    chain: int = -1
    key: int = field(default=-1)
    # Instants decided from another piece (rule R's relabelled history, rule C's clipped end):
    # they follow that piece's final available_at, which the first-bar pass may still raise.
    known_with: "_Segment | None" = field(default=None, compare=False, repr=False)
    end_known_with: "_Segment | None" = field(default=None, compare=False, repr=False)


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


def _first_traded(con: duckdb.DuckDBPyConnection) -> dict[str, date]:
    rows = con.execute(
        "SELECT symbol, min(session_date) FROM _bars WHERE volume > 0 GROUP BY 1"
    ).fetchall()
    return dict(rows)


def _collapse_when_issued(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Make when-issued symbols transparent in rename chains.

    A when-issued symbol trades for a few sessions and is never the security, so no edge may
    chain it to one; but Alpaca routes real renames through it (``RBS -> NWG.WI -> NWG``, an
    ADR ratio change ``GSK -> GSK.WI -> GSK``). An edge into a when-issued name is dropped (the
    regular label was not vacated), an edge out of one is replaced by the direct edge from
    whatever led into it (``RBS -> NWG``, dated and knowable with the edge out; a round trip
    back to the same label is no event at all) -- or, with nothing leading in, kept as the
    record of the regular label changing hands that day (``AAN.WI -> AAN`` on 2020-12-01, the
    spin-off's first regular session) but marked ``links=False`` so it joins no chain.
    """
    into: dict[str, list[dict[str, Any]]] = {}
    for e in edges:
        if WHEN_ISSUED.fullmatch(e["new_symbol"]):
            into.setdefault(e["new_symbol"], []).append(e)
    out: list[dict[str, Any]] = []
    collapsed: list[dict[str, Any]] = []
    for e in edges:
        old, new = e["old_symbol"], e["new_symbol"]
        if WHEN_ISSUED.fullmatch(new):
            continue
        # the edge in that led to this edge out: the latest one filed on or before it (a
        # when-issued name serves several spin-offs over the years)
        led_in = [i for i in into.get(old, []) if i["date"] <= e["date"]]
        if WHEN_ISSUED.fullmatch(old) is None:
            out.append({**e, "links": True, "starts": True})
        elif led_in:
            entry = max(led_in, key=lambda i: i["date"])
            source = entry["old_symbol"]
            if source != new:
                collapsed.append({**e, "old_symbol": source, "old_cusip": entry["old_cusip"]})
            # the when-issued piece still ends here; the regular label is not started by it
            out.append({**e, "links": False, "starts": False})
        else:
            out.append({**e, "links": False, "starts": True})
    # A collapsed edge the SEC also saw (RBS gone, NWG new under one CIK) is one event: the
    # Alpaca date and CUSIPs, evidence ``both``, as merge_edges would have done.
    by_pair = {(e["old_symbol"], e["new_symbol"]): e for e in out if e["evidence"] == "sec"}
    for c in collapsed:
        seen = by_pair.get((c["old_symbol"], c["new_symbol"]))
        if seen is None:
            out.append({**c, "links": True, "starts": True})
        else:
            seen.update({**c, "evidence": "both", "date_lower": None})
    return out


def _drop_takeovers_without_a_taker(
    edges: list[dict[str, Any]], first_traded: Mapping[str, date], stats: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """A rename whose old name never traded here while the new name already did is no event.
    Returns the edges kept and the ones dropped (the build lists the latter as
    ``record_without_taker`` conflicts, resolution ``ignored``).

    CBS -> VIAC was filed 2020-02-13 for a rename that happened 2019-12-05, before the store
    starts; WORXD -> WORX files the return from a temporary OTC suffix that never traded on
    Alpaca (62 such records). Read as a takeover, the record cuts the new label at the filing
    and leaves its earlier bars to nobody (VIAC) or in a second security (WORX). A taker with
    no history cannot have taken anything, so the label simply lives on from its first bar.
    When-issued old names are kept: ``ARNC.WI -> ARNC`` on 2020-04-01 is the only record of
    the ARNC label changing hands that day, and the when-issued symbol itself never trades.
    The reverse edge of a dropped one goes with it: a round trip through a temporary ticker
    that never traded here (``ORIC -> 1ORIC -> ORIC``) left alone would vacate the label. An
    old name that another edge leads into is a link in a chain, not a phantom
    (``CEAD -> VAPE -> BNC``: VAPE never traded here, yet BNC's early bars are CEAD's history
    relabelled and the chain must hold), so only a name no other edge touches qualifies.
    """
    touching: dict[str, set[tuple[str, str]]] = {}
    for e in edges:
        pair = (e["old_symbol"], e["new_symbol"])
        touching.setdefault(e["old_symbol"], set()).add(pair)
        touching.setdefault(e["new_symbol"], set()).add(pair)
    dropped = {
        (e["old_symbol"], e["new_symbol"])
        for e in edges
        if e["old_symbol"] not in stats
        and WHEN_ISSUED.fullmatch(e["old_symbol"]) is None
        # Nasdaq fifth letter Q: the label went through bankruptcy and the equity that comes
        # back is a new issue, so the record is kept and the label is vacated (bars after it,
        # if any, resume as a new security).
        and BANKRUPT.fullmatch(e["old_symbol"]) is None
        and e["new_symbol"] in first_traded
        and first_traded[e["new_symbol"]] < e["date"]
        and touching[e["old_symbol"]]
        <= {(e["old_symbol"], e["new_symbol"]), (e["new_symbol"], e["old_symbol"])}
    }
    kept, gone = [], []
    for e in edges:
        pair = (e["old_symbol"], e["new_symbol"])
        (gone if pair in dropped or pair[::-1] in dropped else kept).append(e)
    return kept, gone


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
    days (NKLA traded from 2020-06-04, the VTIQ -> NKLA record says 06-08) or by weeks (SPAC
    completions are filed six weeks late: SVMK -> MNTV filed 2021-08-09, traded 06-15; 248
    such records). The cut moves to the day after the old name's last traded bar (volume > 0;
    see ``_refine_sec_edges`` on filler bars) when the new name already has bars in between and
    either the record trails that bar by at most ``MAX_GAP_DAYS``, or the hand-over is visible
    in the bars: the new name's FIRST traded bar is strictly after the old name's last and
    within ``HANDOVER_MAX_DAYS`` of it (a label that already carried the new name's relabelled
    history cannot show a hand-over), for records trailing by up to ``SEC_LAG_MAX_DAYS``. The
    edge's ``available_at`` is left alone: the rename was knowable only when Alpaca filed it."""
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
        ), first_new AS (
            SELECT l.*, (SELECT min(n.session_date) FROM _bars n
                         WHERE n.symbol = l.new_symbol AND n.volume > 0) AS first_bar
            FROM last_old l
        )
        SELECT f.old_symbol, f.new_symbol, f.date, f.last_bar
        FROM first_new f
        WHERE (f.date - f.last_bar - 1 BETWEEN 1 AND {MAX_GAP_DAYS}
               OR (f.date - f.last_bar - 1 BETWEEN 1 AND {SEC_LAG_MAX_DAYS}
                   AND f.first_bar > f.last_bar
                   AND f.first_bar - f.last_bar <= {HANDOVER_MAX_DAYS}))
          AND EXISTS (
              SELECT 1 FROM _bars n
              WHERE n.symbol = f.new_symbol
                AND n.session_date > f.last_bar AND n.session_date < f.date
          )
        """
    ).fetchall()
    con.unregister("_alp_edges")
    moved = {(o, n, d): last + timedelta(days=1) for o, n, d, last in rows}
    for e in todo:
        new_date = moved.get((e["old_symbol"], e["new_symbol"], e["date"]))
        if new_date is not None:
            e["date"] = new_date


def _cik_on(presence: list[tuple[date, int]], day: date) -> set[int]:
    return {cik for d, cik in presence if d == day}


def _mark_conversions(
    con: duckdb.DuckDBPyConnection,
    edges: list[dict[str, Any]],
    presence: Mapping[str, list[tuple[date, int]]],
) -> None:
    """Rule B: an Alpaca record whose two names traded side by side right up to the record is a
    conversion -- the old security became the new one's existing shares (CUK -> CCL: Carnival
    plc's ADR into Carnival Corp; GNTY -> GBCI: a merger; CWEN.A -> CWEN: a share-class
    consolidation) -- not a rename. The old label's piece ends at the record; the new label is
    not cut and the two are not joined.

    Three tests, all on the (refined) record date. Parallel days: at least
    ``CONVERSION_MIN_DIFFERING_DAYS`` sessions before the record on which both names traded
    bars that differ on OHLC (identical bars are Alpaca's relabelling, whatever the volume says:
    FFR -> DTRE). Recency: one of them inside the last ``CONVERSION_WINDOW`` sessions before the
    record, or the previous holder of the new label had already stopped and the record is a
    plain rename into a vacated ticker (SCHN -> RDUS: 659 parallel days with Radius Health, a
    year earlier). Same filer: the new label's SEC CIK is known and equal in the last snapshot
    before the record and in the first one at least ``SEC_LAG_MAX_DAYS`` after it (the SEC lists
    the old filer for months: IR -> TT reads unchanged in the first snapshot after the record);
    when the table has no snapshot that late, the latest one after the record. A changed or
    unknown CIK means the new label was taken by a new filer (GOLD -> B: Barnes Group's ticker
    taken by Barrick; COT -> PRMW: the acquirer took the target's ticker) and the record is a
    rename, handled by the takeover rules. Counted on the whole store 2026-10-17: 45 records
    with five or more differing parallel days, 30 recent, 15 of those with one filer.
    """
    todo = [
        e
        for e in edges
        if e["evidence"] != "sec"
        and e.get("links", True)
        and e.get("starts", True)
        and WHEN_ISSUED.fullmatch(e["old_symbol"]) is None
    ]
    if not todo:
        return
    con.register(
        "_conv",
        pa.table(
            {
                "key": pa.array(range(len(todo)), pa.int64()),
                "old": [e["old_symbol"] for e in todo],
                "new": [e["new_symbol"] for e in todo],
                "date": pa.array([e["date"] for e in todo], pa.date32()),
            }
        ),
    )
    rows = con.execute(
        f"""
        WITH ct AS (
            SELECT * FROM _traded
            WHERE symbol IN (SELECT old FROM _conv UNION SELECT new FROM _conv)
        ), since AS (
            SELECT c.key, (SELECT s.session_date FROM _sessions s WHERE s.session_date < c.date
                           ORDER BY s.session_date DESC LIMIT 1 OFFSET {CONVERSION_WINDOW - 1}
                          ) AS since
            FROM _conv c
        ), par AS (
            SELECT c.key, a.session_date,
                   NOT EXISTS (SELECT 1 FROM _ident i WHERE i.session_date = a.session_date
                               AND i.symbol_a = least(c.old, c.new)
                               AND i.symbol_b = greatest(c.old, c.new)) AS differ
            FROM _conv c
            JOIN ct a ON a.symbol = c.old AND a.session_date < c.date
            JOIN ct b ON b.symbol = c.new AND b.session_date = a.session_date
        )
        SELECT p.key, count(*) FILTER (WHERE p.differ) AS differing,
               count(*) FILTER (WHERE p.differ AND p.session_date >= s.since) AS recent
        FROM par p JOIN since s USING (key) GROUP BY 1
        """
    ).fetchall()
    con.unregister("_conv")
    counts = {key: (differing, recent) for key, differing, recent in rows}
    for key, e in enumerate(todo):
        differing, recent = counts.get(key, (0, 0))
        if differing < CONVERSION_MIN_DIFFERING_DAYS or not recent:
            continue
        rows_of = presence.get(e["new_symbol"], [])
        before = [d for d, _ in rows_of if d < e["date"]]
        late = [d for d, _ in rows_of if d >= e["date"] + timedelta(days=SEC_LAG_MAX_DAYS)]
        after = [d for d, _ in rows_of if d >= e["date"]]
        if not before or not after:
            continue
        cik_before = _cik_on(rows_of, before[-1])
        # every snapshot late enough must still show the same filer: a lag longer than
        # SEC_LAG_MAX_DAYS would otherwise read a takeover as a conversion for good
        late_days = set(late) if late else {after[-1]}
        cik_after = {cik for d, cik in rows_of if d in late_days}
        if len(cik_before) == 1 and cik_before == cik_after:
            e["starts"] = False
            e["links"] = False


def _edges_from_bars(
    con: duckdb.DuckDBPyConnection, edges: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Rule E: the rename Alpaca did not file. When one name's bars become identical to
    another's for a run of sessions and then stop while the other trades on, the first name was
    renamed into the second and Alpaca rewrote the new name's history (CYCN -> KRSA: all 1,662
    of Cyclerion's bars sit under KRSA too; XPOA -> QBTS: D-Wave's SPAC). The edge is dated the
    continuer's first traded bar after the run, evidence ``bars``, knowable with that bar.
    "Stops" means no traded bar within ``RUN_NEAR_SESSIONS`` of the run end: a label that comes
    back years later under a new holder (EYES, Second Sight's ticker, trading again in 2026)
    still ended there, and rule O opens the later bars as a new security.

    Conditions, each one separating the 18 renames found on the store (2026-10-16) from the
    shapes next to them: the run is at least ``RUN_MIN_DAYS`` sessions and ends on the ender's
    last traded bar; the ender's own bars (in no identical pair with anyone) end within
    ``RUN_NEAR_SESSIONS`` sessions of the run start, or the run begins on the ender's first bar
    (CYCN; a label whose every bar copies someone and whose run starts later is a relabelled
    ghost, not a predecessor); the continuer's first bar is within ``RUN_NEAR_SESSIONS`` of the
    run start and it had at most max(3, 5%) differing days before the run end (a continuer with
    a life of its own before -- BNZI under PARA, 893 differing days -- is a stranger whose bars
    were copied); no edge of any evidence joins the pair already; the pair is unique both ways.
    A gap between the ender's own bars and the run (AXNX / MIMI) is rule F's shape. Everything
    that fails stays a ``dup_unlinked`` conflict for a human.
    """
    cands = con.execute(
        f"""
        WITH pairs AS (
            SELECT symbol_a, symbol_b, max(session_date) AS last_ident
            FROM _ident GROUP BY 1, 2 HAVING count(*) >= {RUN_MIN_DAYS}
        ), pt AS (
            SELECT * FROM _traded
            WHERE symbol IN (SELECT symbol_a FROM pairs UNION SELECT symbol_b FROM pairs)
        ), nxt AS (  -- each name's first session after the run end; NULL: it never trades again
            SELECT p.*, s.n AS end_n,
                   (SELECT min(t.n) FROM pt t
                    WHERE t.symbol = p.symbol_a AND t.session_date > p.last_ident) AS next_a,
                   (SELECT min(t.n) FROM pt t
                    WHERE t.symbol = p.symbol_b AND t.session_date > p.last_ident) AS next_b
            FROM pairs p JOIN _sessions s ON s.session_date = p.last_ident
        )
        SELECT CASE WHEN stops_a AND goes_on_b THEN symbol_a
                    WHEN stops_b AND goes_on_a THEN symbol_b END AS ender,
               CASE WHEN stops_a AND goes_on_b THEN symbol_b
                    WHEN stops_b AND goes_on_a THEN symbol_a END AS continuer,
               last_ident AS run_end
        FROM (SELECT *,
                     next_a IS NULL OR next_a - end_n > {RUN_NEAR_SESSIONS} AS stops_a,
                     next_b IS NULL OR next_b - end_n > {RUN_NEAR_SESSIONS} AS stops_b,
                     next_a IS NOT NULL AND next_a - end_n <= {RUN_NEAR_SESSIONS} AS goes_on_a,
                     next_b IS NOT NULL AND next_b - end_n <= {RUN_NEAR_SESSIONS} AS goes_on_b
              FROM nxt)
        WHERE ender IS NOT NULL
        """
    ).fetchall()
    if not cands:
        return []
    con.register(
        "_ecand",
        pa.table(
            {
                "key": pa.array(range(len(cands)), pa.int64()),
                "ender": [c[0] for c in cands],
                "continuer": [c[1] for c in cands],
                "run_end": pa.array([c[2] for c in cands], pa.date32()),
            }
        ),
    )
    rows = con.execute(
        """
        WITH et AS (
            SELECT * FROM _traded
            WHERE symbol IN (SELECT ender FROM _ecand UNION SELECT continuer FROM _ecand)
        ), pair_ident AS (  -- identical days of each candidate pair
            SELECT c.key, i.session_date
            FROM _ecand c JOIN _ident i
              ON i.symbol_a = least(c.ender, c.continuer)
             AND i.symbol_b = greatest(c.ender, c.continuer)
        ), any_ident AS (  -- days on which a symbol is identical to anyone
            SELECT symbol_a AS symbol, session_date FROM _ident
            UNION SELECT symbol_b, session_date FROM _ident
        ), run AS (
            SELECT c.key,
                   (SELECT min(t.session_date) FROM et t
                    WHERE t.symbol = c.ender AND t.session_date <= c.run_end
                      AND t.session_date > coalesce(
                          (SELECT max(u.session_date) FROM et u
                           WHERE u.symbol = c.ender AND u.session_date <= c.run_end
                             AND u.session_date NOT IN
                                 (SELECT session_date FROM pair_ident WHERE key = c.key)),
                          DATE '1900-01-01')) AS run_start
            FROM _ecand c
        )
        SELECT c.key, r.run_start,
               (SELECT count(*) FROM et t WHERE t.symbol = c.ender
                 AND t.session_date BETWEEN r.run_start AND c.run_end) AS run_len,
               (SELECT min(t.session_date) FROM et t WHERE t.symbol = c.ender) AS ender_first,
               (SELECT max(t.n) FROM et t WHERE t.symbol = c.ender AND t.session_date < r.run_start
                 AND NOT EXISTS (SELECT 1 FROM any_ident x
                                 WHERE x.symbol = t.symbol AND x.session_date = t.session_date)
               ) AS last_own_n,
               (SELECT n FROM _sessions WHERE session_date = r.run_start) AS run_start_n,
               (SELECT min(t.n) FROM et t WHERE t.symbol = c.continuer) AS cont_first_n,
               (SELECT count(*) FROM et t WHERE t.symbol = c.continuer
                 AND t.session_date <= c.run_end
                 AND t.session_date NOT IN (SELECT session_date FROM pair_ident WHERE key = c.key)
               ) AS cont_differing,
               (SELECT min(t.session_date) FROM et t WHERE t.symbol = c.continuer
                 AND t.session_date > c.run_end) AS cut,
               (SELECT arg_min(t.available_at, t.session_date) FROM et t
                 WHERE t.symbol = c.continuer AND t.session_date > c.run_end) AS cut_available
        FROM _ecand c JOIN run r USING (key)
        """
    ).to_arrow_table()  # arrow, not fetchall: a TIMESTAMPTZ column would need pytz
    con.unregister("_ecand")
    components = _symbol_components(edges)
    paired = {frozenset((e["old_symbol"], e["new_symbol"])) for e in edges}
    found: list[dict[str, Any]] = []
    for row in rows.to_pylist():
        key, run_start, run_len, ender_first, last_own_n, run_start_n = (
            row["key"],
            row["run_start"],
            row["run_len"],
            row["ender_first"],
            row["last_own_n"],
            row["run_start_n"],
        )
        cont_first_n, cont_differing, cut, cut_available = (
            row["cont_first_n"],
            row["cont_differing"],
            row["cut"],
            row["cut_available"],
        )
        ender, continuer, run_end = cands[key]
        if run_len < RUN_MIN_DAYS or cut is None:
            continue
        if WHEN_ISSUED.fullmatch(ender) or WHEN_ISSUED.fullmatch(continuer):
            continue  # a when-issued name prints the regular share's bars: rule A's shape
        if last_own_n is None:
            if run_start != ender_first:
                continue
        elif run_start_n - last_own_n > RUN_NEAR_SESSIONS:
            continue
        if abs(cont_first_n - run_start_n) > RUN_NEAR_SESSIONS:
            continue
        if cont_differing > max(RUN_TOLERANCE[0], RUN_TOLERANCE[1] * run_len):
            continue
        if frozenset((ender, continuer)) in paired:
            continue
        if components.get(ender, ender) == components.get(continuer, continuer):
            continue
        if cut_available.tzinfo is None:
            cut_available = cut_available.replace(tzinfo=UTC)
        found.append(
            {
                "old_symbol": ender,
                "new_symbol": continuer,
                "date": cut,
                "date_lower": None,
                "evidence": "bars",
                "available_at": cut_available,
                "old_cusip": None,
                "new_cusip": None,
            }
        )
    unique = [
        e
        for e in found
        if sum(f["old_symbol"] == e["old_symbol"] for f in found) == 1
        and sum(f["new_symbol"] == e["new_symbol"] for f in found) == 1
    ]
    return sorted(unique, key=lambda e: (e["date"], e["old_symbol"], e["new_symbol"]))


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
        # Alpaca may file the record a few days after the snapshot that already shows the new
        # filer (DPW -> NILE: snapshot 2022-01-03, record 01-04): still the same event.
        explained = any(
            lower < e.date <= upper + timedelta(days=MAX_GAP_DAYS)
            and ciks_of.get(e.old, set()) & now
            for e in starts
        )
        if not explained:
            events.append((upper, 2, None))
    # Same day, same kind: the edge that joins a chain goes last, so it is the one the piece
    # remembers as its start (OLD -> X and X.WI -> X on one day: relabelled copies under X are
    # compared with OLD's bars, not with the when-issued name's three sessions). The piece is
    # knowable at the earliest of the day's records, whichever edge it remembers.
    events.sort(key=lambda t: (t[0], t[1], t[2] is not None and t[2].links))
    starts_known: dict[date, datetime] = {}
    for e in starts:
        starts_known[e.date] = min(starts_known.get(e.date, e.available_at), e.available_at)

    # Renames dated on or before the first bar only settle the label's state when our bars
    # begin; they cannot end a piece that has not started.
    pre = [ev for ev in events if ev[1] != 2 and ev[0] <= first_bar]
    rest = [ev for ev in events if not (ev[1] != 2 and ev[0] <= first_bar)]
    current: _Segment | None
    if pre:
        day, kind, edge = pre[-1]
        if kind == 1:  # taken before our bars: the taker holds it from that day
            current = _Segment(symbol, day, None, start=edge, available_at=starts_known[day])
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
            current = _Segment(symbol, day, None, start=edge, available_at=starts_known[day])
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


def _first_own_bars(
    con: duckdb.DuckDBPyConnection, pieces: Sequence[_Segment]
) -> dict[int, tuple[int, date, datetime]]:
    """key -> (bars, first session, first bar's available_at) over each piece's own days
    ``[valid_from, valid_to)``; pieces without a bar are absent."""
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
    return {r["key"]: tuple(r.values())[1:] for r in rows}  # type: ignore[misc]


def _fill_bar_stats(con: duckdb.DuckDBPyConnection, pieces: list[_Segment]) -> None:
    stats = _first_own_bars(con, pieces)
    for p in pieces:
        n, first, avail = stats.get(p.key, (0, None, None))
        p.n_bars = n
        if p.valid_from is None and first is not None:
            p.valid_from = first
        if p.start is not None:
            p.available_at = p.available_at or p.start.available_at
        elif p.cik_cut and p.valid_from is not None:
            p.available_at = _midnight(p.valid_from)
        elif avail is not None:
            p.available_at = avail.replace(tzinfo=UTC) if avail.tzinfo is None else avail


def _resumed_pieces(con: duckdb.DuckDBPyConnection, pieces: list[_Segment]) -> list[_Segment]:
    """New pieces for traded bars that resume under a closed label after a gap.

    A label vacated by a rename or ended by a takeover can come back with a new holder that no
    record announces: Alpaca filed ``JAMF -> JNMF`` for a pre-IPO placeholder on 2020-01-13 and
    the real Jamf listed under JAMF on 2020-07-22; StoneX left INTL for SNEX in 2020 and an
    ETF the SEC never lists took INTL in 2022 (78 labels, 18,863 bars owned by nobody). For
    every closed piece whose label has no piece right after it, the first traded bar (volume >
    0, so Alpaca's zero-volume filler is not a resumption) between its end and the label's next
    piece opens a new piece when it comes more than ``RESUME_GAP_DAYS`` after the label's
    previous traded bar; bars that run straight through the vacate date are the old holder
    still trading (the audit's N2) and stay unowned. The piece ends where the next piece of
    the label starts, knowable with it; its CIK comes from the SEC rows inside it only.
    """
    by_symbol: dict[str, list[_Segment]] = {}
    for p in pieces:
        by_symbol.setdefault(p.symbol, []).append(p)
    windows: list[tuple[int, str, date, date | None, _Segment | None]] = []
    for symbol, members in by_symbol.items():
        members.sort(key=lambda p: (p.valid_from or date.min, p.valid_to or date.max))
        for p, following in zip(members, [*members[1:], None], strict=True):
            if p.valid_to is None or (following is not None and following.valid_from == p.valid_to):
                continue
            hi = following.valid_from if following is not None else None
            windows.append((len(windows), symbol, p.valid_to, hi, following))
    if not windows:
        return []
    con.register(
        "_windows",
        pa.table(
            {
                "key": pa.array([w[0] for w in windows], pa.int64()),
                "symbol": [w[1] for w in windows],
                "lo": pa.array([w[2] for w in windows], pa.date32()),
                "hi": pa.array([w[3] for w in windows], pa.date32()),
            }
        ),
    )
    rows = con.execute(
        f"""
        WITH resumed AS (
            SELECT w.key, min(b.session_date) AS first_bar
            FROM _windows w JOIN _bars b ON b.symbol = w.symbol AND b.volume > 0
             AND b.session_date >= w.lo AND (w.hi IS NULL OR b.session_date < w.hi)
            GROUP BY 1
        )
        SELECT r.key, r.first_bar
        FROM resumed r JOIN _windows w ON w.key = r.key
        WHERE r.first_bar - coalesce(
            (SELECT max(b.session_date) FROM _bars b
             WHERE b.symbol = w.symbol AND b.volume > 0 AND b.session_date < r.first_bar),
            r.first_bar) > {RESUME_GAP_DAYS}
        """
    ).fetchall()
    con.unregister("_windows")
    new_pieces: list[_Segment] = []
    for key, first_bar in rows:
        _, symbol, _lo, hi, following = windows[key]
        piece = _Segment(symbol, first_bar, hi, resumed=True)
        if following is not None:
            piece.end_available_at = following.available_at
        new_pieces.append(piece)
    return new_pieces


def _chain_symbols_without(edges: Sequence[_Edge], start: str, without: str) -> set[str]:
    """Symbols reachable from ``start`` over linking edges with the node ``without`` removed:
    the taker's whole chain, seen from outside the label it took."""
    adjacent: dict[str, set[str]] = {}
    for e in edges:
        if e.links:
            adjacent.setdefault(e.old, set()).add(e.new)
            adjacent.setdefault(e.new, set()).add(e.old)
    seen, stack = {start}, [start]
    while stack:
        for nxt in adjacent.get(stack.pop(), ()):
            if nxt != without and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def _copy_shares(
    con: duckdb.DuckDBPyConnection, cands: Sequence[tuple[int, str, str, date, date | None]]
) -> dict[int, tuple[int, int]]:
    """For candidates ``(key, symbol, other, lo, hi)``: key -> (traded days of ``symbol`` in
    [lo, hi), days among them identical to any of the key's ``other`` symbols)."""
    if not cands:
        return {}
    con.register(
        "_copy",
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
    traded = dict(
        con.execute(
            """
            SELECT c.key, count(*)
            FROM (SELECT DISTINCT key, symbol, lo, hi FROM _copy) c
            JOIN _traded t ON t.symbol = c.symbol AND t.session_date >= c.lo
             AND (c.hi IS NULL OR t.session_date < c.hi)
            GROUP BY 1
            """
        ).fetchall()
    )
    identical = dict(
        con.execute(
            """
            SELECT c.key, count(DISTINCT i.session_date)
            FROM _copy c JOIN _ident i
              ON i.symbol_a = least(c.symbol, c.other) AND i.symbol_b = greatest(c.symbol, c.other)
             AND i.session_date >= c.lo AND (c.hi IS NULL OR i.session_date < c.hi)
            GROUP BY 1
            """
        ).fetchall()
    )
    con.unregister("_copy")
    return {c[0]: (traded.get(c[0], 0), identical.get(c[0], 0)) for c in cands}


def _is_copy(traded: int, identical: int) -> bool:
    return identical >= COPY_MIN_DAYS and identical >= RELABEL_MIN_IDENTICAL * traded


def _relabelled_pieces(
    con: duckdb.DuckDBPyConnection, pieces: list[_Segment], edges: Sequence[_Edge]
) -> dict[int, int]:
    """Pieces that are the taker's own history filed under a label it took later: key of the
    dropped piece -> key of the taker's piece. A dropped piece that was itself opened by a
    rename record (JCS -> SUNE, found from the bars, while SUNE's label then carried PEGY's
    bars) hands that record on: the name renamed into the copy was renamed into the chain the
    copy belongs to, and the build unions the two.

    For every takeover (a piece opened by a rename) each earlier piece of the same label is
    compared day by day with the taker's WHOLE chain -- every symbol reachable from the taker's
    old name over linking edges, the taken label itself left out of the graph (PARA removed,
    Banzai's chain is {VII, BNZI}; Paramount's PARA piece is compared with both, not with BNZI
    alone, since Alpaca rewrote PARA with the chain's full history). OHLC-identical days on at
    least ``RELABEL_MIN_IDENTICAL`` of the piece's traded days (and ``COPY_MIN_DAYS`` of them)
    mean Alpaca relabelling, not a previous holder. Real: BNY carried BK's bars while the SEC
    listed BNY as a BlackRock fund.
    """
    by_symbol: dict[str, list[_Segment]] = {}
    for p in pieces:
        by_symbol.setdefault(p.symbol, []).append(p)
    cands: list[tuple[int, str, str, date, date | None]] = []
    cand_takers: list[int] = []
    by_construction: dict[int, int] = {}
    for p in pieces:
        if p.start is None or p.valid_from is None:
            continue
        chain = _chain_symbols_without(edges, p.start.old, p.symbol)
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
            if p.start.evidence == "bars" and q.valid_to == p.valid_from and q.start is None:
                # rule E already established that the label's bars up to the cut are the
                # ender's: its opening piece is the copy, whatever its few own days weigh
                by_construction[q.key] = p.key
                continue
            for other in sorted(chain):
                if other != q.symbol:
                    cands.append((q.key, q.symbol, other, q.valid_from, q.valid_to))
                    cand_takers.append(p.key)
    taker_of = {c[0]: p_key for c, p_key in zip(cands, cand_takers, strict=True)}
    shares = _copy_shares(con, cands)
    return by_construction | {
        key: taker_of[key]
        for key, (traded, identical) in shares.items()
        if _is_copy(traded, identical)
    }


def _clip_identical_overlaps(
    con: duckdb.DuckDBPyConnection, pieces: list[_Segment], chain_of: Mapping[int, int]
) -> None:
    """Rule C: two pieces of one chain overlap in time and the earlier label's traded bars
    inside the overlap are copies of the later label's (AZRX / FWBI, 181 days: Alpaca kept
    filing the new name's bars under the old one past the switch) -> the earlier piece ends
    where the later one begins, knowable with it. A non-identical overlap (two names really
    trading side by side) is left alone: that is rule B's shape or a conflict."""
    by_chain: dict[int, list[_Segment]] = {}
    for p in pieces:
        by_chain.setdefault(chain_of[p.key], []).append(p)
    cands: list[tuple[int, str, str, date, date | None]] = []
    overlaps: list[tuple[_Segment, _Segment]] = []  # (earlier, later), same order as cands
    for members in by_chain.values():
        for a in members:
            for b in members:
                if (
                    a is b
                    or a.symbol == b.symbol
                    or a.valid_from is None
                    or b.valid_from is None
                    or not (a.valid_from < b.valid_from < (a.valid_to or date.max))
                ):
                    continue
                hi = min(a.valid_to or date.max, b.valid_to or date.max)
                cands.append(
                    (len(cands), a.symbol, b.symbol, b.valid_from, None if hi == date.max else hi)
                )
                overlaps.append((a, b))
    shares = _copy_shares(con, cands)
    for key, (a, b) in sorted(enumerate(overlaps), key=lambda t: t[1][1].valid_from or date.min):
        if a.valid_to is not None and b.valid_from is not None and a.valid_to <= b.valid_from:
            continue  # already clipped at an earlier overlap
        if _is_copy(*shares.get(key, (0, 0))):
            a.valid_to, a.end, a.end_available_at = b.valid_from, None, b.available_at
            a.end_known_with = b


def _cut_copied_tails(
    con: duckdb.DuckDBPyConnection,
    pieces: list[_Segment],
    presence: Mapping[str, list[tuple[date, int]]],
    snapshots: Sequence[date],
    chain_of: Mapping[int, int],
) -> set[int]:
    """Rule F: a piece whose trailing traded bars are copies of another security's, after the
    holder itself stopped -- the tail begins either after a gap of more than
    ``TAIL_GAP_SESSIONS`` sessions in the label's trading (AXNX: Axonics was bought in 2024-11,
    MIMI's bars sit under the label from 2025-01) or once the piece's filer leaves the SEC table
    (VJET: Voxeljet's own bars stop 2024-03-22, VTAK's copies begin 03-25, the SEC lists VJET
    once more on 04-01 and never again; the tail may start in the filer's last snapshot
    interval). The piece ends on the tail's first day, knowable with that bar; the tail owns
    nothing (the loss metric names it; the other label owns the same bars under its own name).
    A tail identical to a symbol of the piece's own chain is a late-filed rename, not a copy.

    The tail is the earliest traded bar from which the rest of the piece is at least
    ``RELABEL_MIN_IDENTICAL`` identical to one other symbol (``COPY_MIN_DAYS`` at least) and
    whose previous bar is not; a piece that copies from its first bar is rule Z's or the relabel
    rule's, never cut here. The other side is a whole chain (a tail that copies VII and then
    BNZI is Banzai's), never the piece's own chain. Returns the keys of the pieces cut."""
    con.register(
        "_fchain",
        pa.table(
            {
                "symbol": [p.symbol for p in pieces],
                "chain": pa.array([chain_of[p.key] for p in pieces], pa.int64()),
            }
        ),
    )
    con.register(
        "_fp",
        pa.table(
            {
                "key": pa.array([p.key for p in pieces], pa.int64()),
                "symbol": [p.symbol for p in pieces],
                "lo": pa.array([p.valid_from for p in pieces], pa.date32()),
                "hi": pa.array([p.valid_to for p in pieces], pa.date32()),
            }
        ),
    )
    rows = con.execute(
        f"""
        WITH ident_days AS (  -- (piece, other chain, day) identical inside the piece
            SELECT DISTINCT p.key, p.symbol, c.chain AS other, i.session_date
            FROM _fp p JOIN _ident i
              ON (i.symbol_a = p.symbol OR i.symbol_b = p.symbol)
             AND i.session_date >= p.lo AND (p.hi IS NULL OR i.session_date < p.hi)
            JOIN _fchain c
              ON c.symbol = CASE WHEN i.symbol_a = p.symbol THEN i.symbol_b ELSE i.symbol_a END
        ), others AS (  -- (piece, other chain) with enough identical days inside the piece
            SELECT key, symbol, other FROM ident_days
            GROUP BY ALL HAVING count(*) >= {COPY_MIN_DAYS}
        ), bars AS (
            SELECT o.key, o.other, t.session_date, t.n, t.available_at,
                   EXISTS (SELECT 1 FROM ident_days d WHERE d.key = o.key AND d.other = o.other
                           AND d.session_date = t.session_date) AS ident
            FROM others o JOIN _fp p USING (key)
            JOIN _traded t ON t.symbol = p.symbol AND t.session_date >= p.lo
             AND (p.hi IS NULL OR t.session_date < p.hi)
        ), suffix AS (
            SELECT *,
                   sum(ident::INT) OVER w AS suffix_identical, count(*) OVER w AS suffix_n,
                   lag(n) OVER (PARTITION BY key, other ORDER BY session_date) AS prev_n,
                   lag(ident) OVER (PARTITION BY key, other ORDER BY session_date) AS prev_ident
            FROM bars
            WINDOW w AS (PARTITION BY key, other ORDER BY session_date DESC
                         ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
        )
        SELECT key, other, arg_min(session_date, session_date) AS tail_start,
               arg_min(n, session_date) AS tail_n, arg_min(prev_n, session_date) AS prev_n,
               arg_min(available_at, session_date) AS tail_available
        FROM suffix
        WHERE ident AND NOT prev_ident AND prev_n IS NOT NULL
          AND suffix_identical >= {COPY_MIN_DAYS}
          AND suffix_identical >= {RELABEL_MIN_IDENTICAL} * suffix_n
        GROUP BY 1, 2
        """
    ).to_arrow_table()  # arrow, not fetchall: the TIMESTAMPTZ column would need pytz
    con.unregister("_fp")
    con.unregister("_fchain")
    by_key = {p.key: p for p in pieces}
    snapshot_list = sorted(set(snapshots))
    cuts: dict[int, tuple[date, datetime]] = {}
    for row in rows.to_pylist():
        key, other, tail_start = row["key"], row["other"], row["tail_start"]
        tail_n, prev_n, tail_available = row["tail_n"], row["prev_n"], row["tail_available"]
        piece = by_key[key]
        if other == chain_of[key]:
            continue
        gap = tail_n - prev_n > TAIL_GAP_SESSIONS
        left = False
        if piece.cik is not None and snapshot_list:
            seen = [d for d, cik in presence.get(piece.symbol, []) if cik == piece.cik]
            if seen and seen[-1] < snapshot_list[-1]:
                earlier = [d for d in snapshot_list if d < seen[-1]]
                left = tail_start >= (earlier[-1] if earlier else seen[-1])
        if not (gap or left):
            continue
        if key not in cuts or tail_start < cuts[key][0]:
            cuts[key] = (tail_start, tail_available)
    cut_keys: set[int] = set()
    for key, (tail_start, tail_available) in cuts.items():
        piece = by_key[key]
        if piece.valid_from is not None and tail_start <= piece.valid_from:
            continue
        piece.valid_to = tail_start
        piece.end = None
        piece.end_available_at = (
            tail_available.replace(tzinfo=UTC) if tail_available.tzinfo is None else tail_available
        )
        cut_keys.add(key)
    return cut_keys


def _drop_copied_pieces(
    con: duckdb.DuckDBPyConnection, pieces: list[_Segment], chain_of: Mapping[int, int]
) -> set[int]:
    """Rule Z: keys of pieces that are relabelled history of another chain. (a) A label held by
    two or more securities whose earlier holder's traded bars are copies of the latest holder's
    chain (BTLN: the old filer's 1,289 bars are VRAR's and GGRP's, Glimpse's chain, which now
    holds BTLN) -- Alpaca rewrote the label's past with the current holder's history. (b) A
    piece opened by bars resuming under a closed label (rule O) that copies a stranger (the
    VJET ghost): the resumed bars are the stranger's, not a new holder's."""
    by_symbol: dict[str, list[_Segment]] = {}
    for p in pieces:
        by_symbol.setdefault(p.symbol, []).append(p)
    symbols_of_chain: dict[int, set[str]] = {}
    for p in pieces:
        symbols_of_chain.setdefault(chain_of[p.key], set()).add(p.symbol)
    cands: list[tuple[int, str, str, date, date | None]] = []
    for members in by_symbol.values():
        chains = {chain_of[p.key] for p in members}
        if len(chains) < 2:
            continue
        latest = chain_of[max(members, key=lambda p: p.valid_from or date.min).key]
        for q in members:
            if chain_of[q.key] == latest or q.valid_from is None:
                continue
            for other in sorted(symbols_of_chain[latest] - {q.symbol}):
                cands.append((q.key, q.symbol, other, q.valid_from, q.valid_to))
    shares = _copy_shares(con, cands)
    dropped = {key for key, (traded, identical) in shares.items() if _is_copy(traded, identical)}

    resumed = [p for p in pieces if p.resumed and p.valid_from is not None and p.key not in dropped]
    if resumed:
        con.register(
            "_zres",
            pa.table(
                {
                    "key": pa.array([p.key for p in resumed], pa.int64()),
                    "symbol": [p.symbol for p in resumed],
                    "lo": pa.array([p.valid_from for p in resumed], pa.date32()),
                    "hi": pa.array([p.valid_to for p in resumed], pa.date32()),
                }
            ),
        )
        con.register(
            "_zchain",
            pa.table(
                {
                    "key": pa.array(
                        [p.key for p in resumed for _ in symbols_of_chain[chain_of[p.key]]],
                        pa.int64(),
                    ),
                    "symbol": [
                        s for p in resumed for s in sorted(symbols_of_chain[chain_of[p.key]])
                    ],
                }
            ),
        )
        rows = con.execute(
            """
            WITH traded AS (
                SELECT p.key, count(*) AS n
                FROM _zres p JOIN _traded t ON t.symbol = p.symbol AND t.session_date >= p.lo
                 AND (p.hi IS NULL OR t.session_date < p.hi)
                GROUP BY 1
            ), copied AS (
                SELECT p.key, count(DISTINCT i.session_date) AS n
                FROM _zres p JOIN _ident i
                  ON (i.symbol_a = p.symbol OR i.symbol_b = p.symbol)
                 AND i.session_date >= p.lo AND (p.hi IS NULL OR i.session_date < p.hi)
                WHERE (CASE WHEN i.symbol_a = p.symbol THEN i.symbol_b ELSE i.symbol_a END)
                      NOT IN (SELECT symbol FROM _zchain z WHERE z.key = p.key)
                GROUP BY 1
            )
            SELECT t.key, t.n, coalesce(c.n, 0) FROM traded t LEFT JOIN copied c USING (key)
            """
        ).fetchall()
        con.unregister("_zres")
        con.unregister("_zchain")
        dropped |= {key for key, traded, identical in rows if _is_copy(traded, identical)}
    return dropped


def _round_trip(leaver: _Segment, taker: _Segment) -> bool:
    """The label is taken back from where its holder just went (ORIC -> 1ORIC -> ORIC)."""
    return leaver.end is not None and taker.start is not None and taker.start.old == leaver.end.new


def _taker_traded_own_bars(
    con: duckdb.DuckDBPyConnection, before: _Segment, members: Sequence[_Segment], cut: date
) -> bool:
    """Whether a piece of the taker's chain traded inside [before.valid_from, cut) on a day
    whose bar is not ``before``'s bar of that day (OHLC; volumes are revised after 20:00)."""
    spans = [
        (q.symbol, max(q.valid_from, before.valid_from), min(q.valid_to or cut, cut))  # type: ignore[type-var]
        for q in members
        if q.valid_from is not None and before.valid_from is not None
    ]
    spans = [s for s in spans if s[1] < s[2]]
    if not spans:
        return False
    con.register(
        "_taker_spans",
        pa.table(
            {
                "symbol": [s for s, _, _ in spans],
                "lo": pa.array([lo for _, lo, _ in spans], pa.date32()),
                "hi": pa.array([hi for _, _, hi in spans], pa.date32()),
            }
        ),
    )
    try:
        hit = con.execute(
            """
            SELECT 1 FROM _taker_spans w
            JOIN _bars t ON t.symbol = w.symbol AND t.session_date >= w.lo
                 AND t.session_date < w.hi
            LEFT JOIN _bars p ON p.symbol = ? AND p.session_date = t.session_date
            WHERE t.volume > 0
              AND (p.symbol IS NULL OR p.open <> t.open OR p.high <> t.high OR p.low <> t.low
                   OR p.close <> t.close)
            LIMIT 1
            """,
            [before.symbol],
        ).fetchone()
    finally:
        con.unregister("_taker_spans")
    return hit is not None


def _settle_takeovers(
    con: duckdb.DuckDBPyConnection,
    pieces: list[_Segment],
    uf: _Union,
    takeovers: Sequence[tuple[int, _Segment, _Segment, date]],
    conflicts: list[dict[str, Any]],
) -> set[int]:
    """Rule R, after the copy rules: a label taken at a continuous price by a filer other than
    its previous holder. If the taker's chain traded no bar of its own inside the previous
    piece's span, that piece is the taker's history filed under the label (Alpaca rewrote B
    with Barrick's past when GOLD -> B; Barnes Group's own bars are gone): it joins the taker,
    knowable only from the takeover's record, and the taker's pieces inside the span end where
    it begins. Otherwise two securities. The absence of the taker's bars is weak evidence, so a
    merge is written ``relabelled`` and the audit (M6) has a human read it once.

    Shapes the merge cannot handle cleanly stay ``unresolved``: a previous piece that already
    chains to other pieces (merging it would carry that company's earlier history along), a
    taker piece in the span that runs past the cut (clipping would orphan its later bars), and
    a previous piece that follows another on the same label (whose end is knowable when the
    previous piece was, not at the takeover). Returns the keys of taker pieces clipped away.
    """
    dropped: set[int] = set()
    for index, before, after, cut in takeovers:
        root, own = uf.find(after.key), uf.find(before.key)
        if own == root:  # a same-day round trip already chained them (L -> 1L -> L)
            conflicts[index]["resolution"] = "bridged"
            continue
        members = [
            q for q in pieces if q.key not in dropped and q is not before and uf.find(q.key) == root
        ]
        if _taker_traded_own_bars(con, before, members, cut):
            conflicts[index]["resolution"] = "new_security"
            continue
        inside = [
            q
            for q in members
            if q is not after
            and q.valid_from is not None
            and q.valid_from < cut
            and (q.valid_to is None or q.valid_to > before.valid_from)  # type: ignore[operator]
        ]
        if (
            any(q is not before and uf.find(q.key) == own for q in pieces)
            or any(q.valid_to is None or q.valid_to > cut for q in inside)
            or any(q.symbol == before.symbol and q.valid_to == before.valid_from for q in pieces)
        ):
            continue  # stays unresolved
        lost = before.cik
        uf.union(before.key, after.key)
        # the taker's CIK is known no earlier on its relabelled history than on its own piece
        before.cik, before.cik_since, before.evidence = after.cik, after.cik_since, "relabelled"
        # judged on the taker's bars after the cut: knowable no earlier than the taker's piece
        before.available_at = before.end_available_at = after.available_at
        before.known_with = before.end_known_with = after
        for q in inside:
            if q.valid_from >= before.valid_from:  # type: ignore[operator]
                dropped.add(q.key)
            else:
                q.valid_to, q.end_available_at = before.valid_from, after.available_at
                q.end_known_with = after
        conflicts[index]["resolution"] = "relabelled"
        conflicts[index]["detail"] += (
            f"; {before.symbol} {before.valid_from}..{cut} is the taker's history, "
            f"holder cik {lost} has no bars of its own here"
        )
    return dropped


def _closes_around(
    con: duckdb.DuckDBPyConnection, cuts: list[tuple[str, date]]
) -> dict[tuple[str, date], tuple[date | None, float | None, date | None, float | None]]:
    """Last traded (date, close) before each cut and first at or after it. Volume-0 filler bars
    (frozen close) would make any delisted label look continuous with its next holder. Every cut
    gets an answer, all None when its label has no traded bar at all (BMR at 2022-10-21: filler
    only until 2023-02-28)."""
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
    out: dict[tuple[str, date], tuple[date | None, float | None, date | None, float | None]]
    out = dict.fromkeys(cuts, (None, None, None, None))
    out.update({(s, cut): (d0, c0, d1, c1) for s, cut, d0, c0, d1, c1 in rows})
    return out


def _continuous(around: tuple[date | None, float | None, date | None, float | None]) -> bool:
    d0, c0, d1, c1 = around
    if d0 is None or d1 is None or not c0 or c1 is None:
        return False
    if (d1 - d0).days > MAX_GAP_DAYS:
        return False
    return CONTINUOUS[0] <= c1 / c0 <= CONTINUOUS[1]


def _segment_cik(
    seg: _Segment,
    presence: list[tuple[date, int]],
    left: int | None = None,
    spelled: Mapping[int, date] | None = None,
) -> tuple[int | None, datetime | None]:
    """The segment's CIK and the snapshot instant that first showed it (00:00 New York).

    ``left``: the CIK of the holder that vacated this label by a rename on the day this
    segment starts (rule L). Its rows still sit on the label for a snapshot or two of SEC lag
    (XPER under Adeia's 1803696 on 2022-10-03, ARNC under Arconic Inc's 4281 until 07-10) and
    are not the new holder's.

    ``spelled``: per CIK, the first snapshot writing this label's own spelling when some rows
    reached it from another one (``spelling_known_since``); the CIK is not known before it."""
    cik, since = _segment_cik_rows(seg, presence, left)
    if cik is not None and since is not None and spelled and cik in spelled:
        since = max(since, _midnight(spelled[cik]))
    return cik, since


def _segment_cik_rows(
    seg: _Segment, presence: list[tuple[date, int]], left: int | None
) -> tuple[int | None, datetime | None]:
    inside = [
        (day, cik)
        for day, cik in presence
        if (seg.valid_from is None or day >= seg.valid_from)
        and (seg.valid_to is None or day < seg.valid_to)
        and cik != left
    ]
    # Known from the first snapshot inside the segment, even when an earlier one listed the CIK:
    # matching that earlier row to the label may need a later spelling (BRKB until 2020, BRK.B
    # after; normalize_spelling joins them only once both are in the table, and _segment_cik
    # dates the CIK no earlier than that).
    if inside:
        cik = inside[-1][1]
        return cik, _midnight(min(day for day, c in inside if c == cik))
    before = [
        (day, cik) for day, cik in presence if seg.valid_from is not None and day < seg.valid_from
    ]
    # A label's opening segment may predate our bars; its SEC rows then sit before valid_from.
    # Only trust them when nothing says the label changed hands since (no start edge, not a
    # resumption: the ETF that took INTL is not StoneX, whose rows sit before it).
    if before and seg.start is None and not seg.cik_cut and not seg.resumed:
        return before[-1][1], _midnight(before[-1][0])
    return None, None


def _cik_available_at(seg: _Segment) -> datetime | None:
    """Never before the segment itself is known: a CIK listed before the first bar is learned
    with the segment."""
    if seg.cik is None:
        return None
    if seg.cik_since is None or seg.available_at is None:
        return seg.available_at
    return max(seg.cik_since, seg.available_at)


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
        _register_sessions(con)
        con.register("_ident", identical_days(bars, con))
        stats = _symbol_stats(con)
        first_traded = _first_traded(con)
        spelled = normalize_spelling(sec_history, stats)
        spelled_since: dict[str, dict[int, date]] = {}
        for (symbol, cik), day in spelling_known_since(sec_history, spelled).items():
            spelled_since.setdefault(symbol, {})[cik] = day
        sec = drop_table_corrections(spelled)
        presence = _sec_presence(sec)
        snapshots = sorted(set(sec.column("snapshot_date").to_pylist()))
        ciks_of = {symbol: {cik for _, cik in rows} for symbol, rows in presence.items()}

        edge_rows = merge_edges(
            edges_from_name_changes(name_changes), edges_from_sec(sec)
        ).to_pylist()
        edge_rows = [e for e in edge_rows if e["old_symbol"] in stats or e["new_symbol"] in stats]
        edge_rows = _collapse_when_issued(edge_rows)
        edge_rows, no_taker = _drop_takeovers_without_a_taker(edge_rows, first_traded, stats)
        _refine_sec_edges(con, edge_rows)
        _refine_alpaca_edges(con, edge_rows)
        _mark_conversions(con, edge_rows, presence)
        edge_rows += _edges_from_bars(con, edge_rows)
        edges = [
            _Edge(
                e["old_symbol"],
                e["new_symbol"],
                e["date"],
                e["evidence"],
                e["available_at"],
                e["old_cusip"],
                e["new_cusip"],
                links=e.get("links", True),
                starts=e.get("starts", True),
            )
            for e in edge_rows
        ]
        # pairs a when-issued record explains; a linking edge that failed to join its two
        # pieces must still surface as dup_unlinked
        edge_pairs = {frozenset((e.old, e.new)) for e in edges if not e.links and e.starts}
        starts: dict[str, list[_Edge]] = {}
        ends: dict[str, list[_Edge]] = {}
        for e in edges:
            if e.starts:
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
        # Keys must stay unique once resumed pieces are appended at len(pieces): every later step
        # finds pieces by key, and a shared key let rule R drop YEXT along with resumed AACI.
        for i, p in enumerate(pieces):
            p.key = i
        resumed = _resumed_pieces(con, pieces)
        for p in resumed:
            p.key = len(pieces)
            pieces.append(p)
        _fill_bar_stats(con, resumed)
        relabelled = _relabelled_pieces(con, pieces, edges)
        by_key = {p.key: p for p in pieces}
        # A bar-evidence edge that opened a dropped copy leads into the taker's chain: the E
        # edge JCS -> SUNE says JCS's bars continue under SUNE, and SUNE's piece is PEGY's bars,
        # so JCS joins PEGY. Only for edges read off the bars: a filed record into a label whose
        # history a later holder overwrote (VIAC -> PARA, PARA's piece being Banzai's bars) is
        # about the earlier company and must not pull it into the later one's chain.
        handed_on = []
        for q_key, p_key in relabelled.items():
            start = by_key[q_key].start
            if start is None or not start.links or start.evidence != "bars":
                continue
            while p_key in relabelled:  # the taker itself a copy of a later holder's chain
                p_key = relabelled[p_key]
            handed_on.append((start, by_key[p_key]))
        pieces = [p for p in pieces if p.key not in relabelled]
        # L: the holder that renamed off a label takes its CIK along (pieces of one symbol are in
        # time order, so the leaver's CIK is known before the piece that follows it)
        # -- unless the label comes straight back from where the leaver went (ORIC -> 1ORIC ->
        # ORIC on one day: the same filer)
        leavers: dict[tuple[str, date], _Segment] = {}
        for i, p in enumerate(pieces):
            p.key = i
            # Attributes describe how the segment STARTED only. The event that ends it is
            # recorded on the segment it creates; copying it here (its cusip, its evidence) would
            # let a reader who may not yet know the end see that one is coming.
            leaver = leavers.get((p.symbol, p.valid_from)) if p.start is not None else None  # type: ignore[arg-type]
            left = leaver.cik if leaver is not None and not _round_trip(leaver, p) else None
            p.cik, p.cik_since = _segment_cik(
                p, presence.get(p.symbol, []), left, spelled_since.get(p.symbol)
            )
            if p.end is not None and p.valid_to is not None:
                leavers[(p.symbol, p.valid_to)] = p
            p.cusip = p.start.new_cusip if p.start else None
            if p.start is not None:
                p.evidence = p.start.evidence
            elif p.cik is not None:
                p.evidence = "sec"

        by_start = {(p.symbol, p.valid_from): p for p in pieces if p.start is not None or p.cik_cut}
        by_end = {(p.symbol, p.valid_to): p for p in pieces if p.valid_to is not None}
        uf = _Union(len(pieces))
        conflicts: list[dict[str, Any]] = [
            {
                "kind": "record_without_taker",
                "symbols": [e["old_symbol"], e["new_symbol"]],
                "dates": [e["date"]],
                "detail": f"{e['old_symbol']} never traded here while {e['new_symbol']} did: "
                "not an event (rule V)",
                "resolution": "ignored",
            }
            for e in no_taker
        ]
        conflicts += [
            {
                "kind": "bars_edge",
                "symbols": [e.old, e.new],
                "dates": [e.date],
                "detail": f"{e.old} -> {e.new} read off identical bars (rule E), no record",
                "resolution": "bridged",
            }
            for e in edges
            if e.evidence == "bars"
        ]
        # reuse_looks_continuous rows are only true while the previous holder's piece still
        # ends at the cut; the copy rules below may end or drop it. Rule R then settles the
        # takeovers among them (index, previous piece, taker's piece or None, cut).
        reuses: list[tuple[int, _Segment, _Segment | None, date]] = []
        for e in edges:
            a, b = by_end.get((e.old, e.date)), by_start.get((e.new, e.date))
            # Two edges may start one label on one day (HWM.WI -> HWM and ARNC -> HWM, both
            # 2020-04-01); the piece belongs to the day, whichever edge created it.
            if a is not None and b is not None and a.end is e and e.links:
                uf.union(a.key, b.key)
        for edge, taker in handed_on:
            a = by_end.get((edge.old, edge.date))  # type: ignore[union-attr]
            if a is not None and a.end is edge:
                uf.union(a.key, taker.key)

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
                    reuses.append((len(conflicts), before, None, cut))
                    conflicts.append(
                        {
                            "kind": "reuse_looks_continuous",
                            "symbols": [symbol],
                            "dates": [d for d in (d0, cut, d1) if d is not None],
                            "detail": f"cik {before.cik} -> {after.cik} after {symbol} -> "
                            f"{before.end.new}, close {c0} -> {c1}: previous holder renamed away",
                            "resolution": "new_security",
                        }
                    )
                continue
            d0, c0, d1, c1 = around[(symbol, cut)]
            if _continuous(around[(symbol, cut)]):
                uf.union(before.key, after.key)
                conflicts.append(
                    {
                        "kind": "cik_switch_bridged",
                        "symbols": [symbol],
                        "dates": [d for d in (d0, cut, d1) if d is not None],
                        "detail": f"cik {before.cik} -> {after.cik}, close {c0} -> {c1}",
                        "resolution": "bridged",
                    }
                )
            else:
                conflicts.append(
                    {
                        "kind": "cik_switch_discontinuous",
                        "symbols": [symbol],
                        "dates": [d for d in (d0, cut, d1) if d is not None],
                        "detail": f"cik {before.cik} -> {after.cik}, close {c0} -> {c1}",
                        "resolution": "new_security",
                    }
                )
        for symbol, cut in takeovers:
            before, after = by_end[(symbol, cut)], by_start[(symbol, cut)]
            if not _continuous(around[(symbol, cut)]):
                continue  # a plain reuse: two securities, nothing to say
            d0, c0, d1, c1 = around[(symbol, cut)]
            if before.end is not None and not _round_trip(before, after):
                # the previous holder renamed away the day the label was taken (XPER -> ADEA,
                # XPER.WI -> XPER): two filers, however close the prices
                reuses.append((len(conflicts), before, None, cut))
                conflicts.append(
                    {
                        "kind": "reuse_looks_continuous",
                        "symbols": [after.start.old if after.start else symbol, symbol],
                        "dates": [d for d in (d0, cut, d1) if d is not None],
                        "detail": f"cik {before.cik} -> {after.cik}, close {c0} -> {c1}: "
                        f"previous holder renamed to {before.end.new}",
                        "resolution": "new_security",
                    }
                )
            elif before.cik is not None and before.cik == after.cik:
                uf.union(before.key, after.key)  # temporary ticker, SPAC completion, ...
            else:
                reuses.append((len(conflicts), before, after, cut))
                conflicts.append(
                    {
                        "kind": "reuse_looks_continuous",
                        "symbols": [after.start.old if after.start else symbol, symbol],
                        "dates": [d for d in (d0, cut, d1) if d is not None],
                        "detail": f"cik {before.cik} -> {after.cik}, close {c0} -> {c1}",
                        "resolution": "unresolved",  # rule R below settles it
                    }
                )

        # Copy rules on the chained pieces: all three read the identical-day table built once
        # above. C clips a copied overlap inside a chain, F ends a piece where a copied tail
        # begins, Z drops pieces that are another chain's relabelled history.
        chain_of = {p.key: uf.find(p.key) for p in pieces}
        _clip_identical_overlaps(con, pieces, chain_of)
        tails_cut = _cut_copied_tails(con, pieces, presence, snapshots, chain_of)
        gone = _drop_copied_pieces(con, pieces, chain_of)
        pieces = [p for p in pieces if p.key not in gone]
        # C clips inside one chain, so the price comparison at the takeover still stands
        stale = {
            index
            for index, before, _, cut in reuses
            if before.key in gone or (before.key in tails_cut and before.valid_to != cut)
        }
        dropped = _settle_takeovers(
            con,
            pieces,
            uf,
            [(i, b, a, cut) for i, b, a, cut in reuses if a is not None and i not in stale],
            conflicts,
        )
        pieces = [p for p in pieces if p.key not in dropped]
        conflicts = [c for i, c in enumerate(conflicts) if i not in stale]

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

        # A segment is no earlier than its first own bar: the build makes none without one.
        # Before the chain fill below, which compares these instants.
        current = {p.key: p for p in pieces}
        # a piece may follow one dropped later (rule C ran before rules Z / R): raise it too
        followed = {
            r.key: r
            for p in pieces
            for r in (p.known_with, p.end_known_with)
            if r is not None and r.key not in current
        }
        current |= followed
        for key, (_, _, avail) in _first_own_bars(con, list(current.values())).items():
            p = current[key]
            avail = avail.replace(tzinfo=UTC) if avail.tzinfo is None else avail
            if p.available_at is None or p.available_at < avail:
                p.available_at = avail
        settled = False
        while not settled:  # a piece may follow one that follows another
            settled = True
            for p in pieces:
                ref = p.known_with
                if ref is not None and ref.available_at is not None:
                    if p.available_at is None or p.available_at < ref.available_at:
                        p.available_at, settled = ref.available_at, False
                ref = p.end_known_with
                if ref is not None and ref.available_at is not None:
                    if p.end_available_at is not None and p.end_available_at < ref.available_at:
                        p.end_available_at, settled = ref.available_at, False

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
                    donor = min(known, key=lambda q: abs((q.valid_from - p.valid_from).days))  # type: ignore[operator]
                    p.cik, p.cik_since = donor.cik, donor.cik_since

        # Stable, content-derived ids: chains ordered by their earliest segment.
        order = sorted(
            chains,
            key=lambda c: min((p.valid_from, p.symbol) for p in chains[c]),  # type: ignore[type-var]
        )
        sid = {chain: f"S{i + 1:06d}" for i, chain in enumerate(order)}

        sids_by_symbol: dict[str, set[str]] = {}
        for p in pieces:
            sids_by_symbol.setdefault(p.symbol, set()).add(sid[p.chain])
        owned = pa.table(
            {
                "symbol": [p.symbol for p in pieces],
                "lo": pa.array([p.valid_from for p in pieces], pa.date32()),
                "hi": pa.array([p.valid_to for p in pieces], pa.date32()),
            }
        )
        for r in ohlcv_duplicates(bars, con, owned).to_pylist():
            a, b = r["symbol_a"], r["symbol_b"]
            if r["days"] < DUP_MIN_DAYS:
                continue
            if WHEN_ISSUED.fullmatch(a) or WHEN_ISSUED.fullmatch(b):
                continue  # a when-issued name prints the regular share's bars by construction
            if sids_by_symbol.get(a, set()) & sids_by_symbol.get(b, set()):
                continue
            if frozenset((a, b)) in edge_pairs:
                continue  # a record explains the identical days (HONAV -> HONA): nothing to settle
            conflicts.append(
                {
                    "kind": "dup_unlinked",
                    "symbols": [a, b],
                    "dates": [],
                    "detail": f"{r['days']} days of identical OHLCV, no rename links them",
                    "resolution": "unresolved",
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
                "cik_available_at": _cik_available_at(p),
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
        if item.get("date") is None:
            raise ValueError(f"{path}: every override needs the date it was decided: {item!r}")
    return raw


@dataclass(frozen=True)
class StoreInputs:
    """What a build reads from the store: daily bars (imports applied), the SEC ticker history
    and Alpaca's name changes. Loaded once (about a second on the real store), shared by every
    build of a truncation run."""

    bars: pa.Table
    sec_history: pa.Table
    name_changes: pa.Table


def load_store_inputs(market_root: Path) -> StoreInputs:
    """The build inputs on disk. Imported bars override the vendor's row of the same (symbol,
    t), exactly as the read path serves them (PARA 2022-02..2025-08 is Paramount from Yahoo,
    not Alpaca's Banzai copies): a master built on the vendor's rows alone would judge
    Paramount's piece on Banzai's bars."""
    from asof.store.market import bar_files, bars_source_sql  # noqa: PLC0415  (market imports us)

    alpaca_files, ext_files = bar_files(market_root, "1Day")
    con = duckdb.connect()
    try:
        con.execute("SET TimeZone = 'UTC'")
        columns = "symbol, session_date, open, high, low, close, volume, available_at"
        if alpaca_files or ext_files:
            bars = con.execute(
                f"SELECT {columns} FROM ({bars_source_sql(alpaca_files, ext_files)})"
            ).to_arrow_table()
        else:
            bars = pa.Table.from_pylist(
                [],
                schema=pa.schema(
                    [
                        ("symbol", pa.string()),
                        ("session_date", pa.date32()),
                        ("open", pa.float64()),
                        ("high", pa.float64()),
                        ("low", pa.float64()),
                        ("close", pa.float64()),
                        ("volume", pa.int64()),
                        ("available_at", TS_UTC),
                    ]
                ),
            )
        sec = pq.read_table(market_root / "symbols" / "sec_history.parquet")
        changes_path = market_root / "corporate_actions" / "name_changes.parquet"
        name_changes = (
            pq.read_table(changes_path)
            if changes_path.exists()
            else pa.Table.from_pylist([], schema=NAME_CHANGE_SCHEMA)
        )
    finally:
        con.close()
    return StoreInputs(bars=bars, sec_history=sec, name_changes=name_changes)


def build_from_inputs(
    inputs: StoreInputs,
    overrides: Sequence[Mapping[str, Any]] = (),
    *,
    as_of: datetime | None = None,
) -> tuple[pa.Table, pa.Table]:
    """Build from loaded store inputs, which are never changed.

    With ``as_of``, only the inputs knowable then (``master_truncation.truncate_inputs``): the
    master a build at that instant would have produced, for the truncation test."""
    if as_of is not None and as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    bars, sec, name_changes = inputs.bars, inputs.sec_history, inputs.name_changes
    if as_of is not None:
        from asof.ingest.master_truncation import truncate_inputs  # noqa: PLC0415  (imports us)

        bars, sec, name_changes, overrides = truncate_inputs(
            bars=bars, sec_history=sec, name_changes=name_changes, overrides=overrides, as_of=as_of
        )
    return build_security_master(
        bars=bars,
        sec_history=sec,
        name_changes=name_changes,
        overrides=overrides,
        sec_history_start=SEC_HISTORY_START,
    )


def build_from_store(
    market_root: Path,
    overrides: Sequence[Mapping[str, Any]] = (),
    *,
    as_of: datetime | None = None,
) -> tuple[pa.Table, pa.Table]:
    """Build from the daily bars, the SEC history and the Alpaca name changes on disk."""
    if as_of is not None and as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    return build_from_inputs(load_store_inputs(market_root), overrides, as_of=as_of)


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
    "adjacent segment knowable before the previous one's end": (
        "SELECT count(*) FROM {v} a JOIN {v} b ON a.symbol = b.symbol "
        "AND a.valid_to <= b.valid_from "
        "WHERE b.available_at < a.end_available_at"
    ),
    "cik without cik_available_at or the reverse": (
        "SELECT count(*) FROM {v} WHERE (cik IS NULL) <> (cik_available_at IS NULL)"
    ),
    "cik_available_at before available_at": (
        "SELECT count(*) FROM {v} WHERE cik_available_at < available_at"
    ),
}


def check_master(master: pa.Table) -> None:
    """Refuse a master that ``visible_master`` would trip over at some ``as_of``.

    Checked over the whole table, not the part visible at one instant: an overlap among
    segments nobody can see before 2030 is still an overlap. The last rule is the one a reader
    would only meet at a particular instant: when a label passes from one segment to the next on
    one day (BBBY 2025-08-29), the new start must not become knowable before the old end, or
    between the two instants the label is owned twice (ARNC before the "earliest instant of the
    day" rule). The other way round is a gap, not an error: the new segment waits for its first
    bar while the old end is known at the event. Raises ``ValueError`` naming every broken rule
    with its count.
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
