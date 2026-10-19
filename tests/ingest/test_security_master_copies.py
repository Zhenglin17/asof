"""Guards against the master attributing a security's relabelled copies to the wrong holder, or
splitting one security in two, in the six shapes counted on the whole store on 2026-10-16/17
(notes: step6b-counts.md, sections B, "B recount", C/D, E, F, G, Z, "Order of application"):

  B  a rename record whose two names traded in parallel (CUK -> CCL: the plc ADR converted
     into the Corp share) is a conversion, not a rename: two securities, the old label ends at
     the record. The same record shape with the new label changing filer (GOLD -> B) or with
     the previous holder long gone (SCHN -> RDUS) stays a rename;
  E  an unfiled rename: one name's bars become identical to another's and then stop while the
     other trades on (CYCN -> KRSA, XPOA -> QBTS). Today the pair is two securities and a
     dup_unlinked conflict;
  relabel  a label's earlier piece that copies the taker's WHOLE chain (PARA before Banzai:
     VII's and BNZI's bars) is not a previous holder, although it copies no single symbol of the
     chain on 90% of its days;
  C  two pieces of one chain overlap because the old label kept carrying the new label's bars
     after the switch (AZRX / FWBI, 181 days): the old label ends where the new one begins;
  F  a dead ticker whose bars resume, after a gap or after its filer left the SEC table, as a
     copy of another security's bars (AXNX <- MIMI, VJET <- VTAK): the tail is nobody's;
  Z  an old-holder piece that is 90% copies of the label's latest holder's chain (BTLN, TE), and
     a resumed piece (rule O) that is 90% copies of a stranger (the VJET ghost), are dropped;
  G  every conflict row says what the build did about it (``resolution``), and the build reads
     the imported bars (``bars_1d_ext``) like the read path does.

Proposed API (``build_security_master``'s contract and these names stay; internals may move):

    IDENT_SCHEMA = pa.schema([symbol_a, symbol_b, session_date])
    identical_days(bars, con=None) -> pa.Table   one row per (pair, day), symbol_a < symbol_b,
                                   both volume > 0, open/high/low/close equal, volume ignored
                                   (Alpaca revises volume after 20:00 ET)
    RELABEL_MIN_IDENTICAL = 0.9    threshold of every copy rule (relabel, C, F, Z)
    COPY_MIN_DAYS = 3              fewer identical days never count as a copy
    RUN_MIN_DAYS = 3               E: shortest identical run that is a rename
    RUN_NEAR_SESSIONS = 10         E: ender's last own bar / continuer's first bar vs run start
    RUN_TOLERANCE = (3, 0.05)      E: continuer's differing days before the run end
    CONVERSION_MIN_DIFFERING_DAYS = 5, CONVERSION_WINDOW = 5     B1
    TAIL_GAP_SESSIONS = 10         F: gap in the label's traded bars before a copied tail
    RESOLUTIONS = ("new_security", "bridged", "ignored", "unresolved")
    CONFLICT_SCHEMA ends with ("resolution", string); new kinds ``cik_switch_bridged`` (symbols
    [label], dates [d0, cut, d1]) and ``record_without_taker`` (symbols [old, new], dates
    [record]); segments created by an E edge carry evidence "bars".

Sessions: ``S[i]`` is the i-th weekday from 2020-01-02 (a Thursday; ``S[i]`` is a Thursday iff
``i % 5 == 0``, which the E worlds use so that "run end + 1 calendar day" and "the continuer's
first bar after the run" are the same Friday).

Fake data provenance (store probe 2026-10-17 unless noted):
- CUK -> CCL: Alpaca record 2026-05-08, CUK's last bar 2026-05-06 (close 27.47), CCL 27.52 on
  05-06 and 27.00 on 05-07, 1,594 parallel days all OHLC-different; SEC CCL under 815097 in
  every snapshot through 2026-09-08 (the last one; none >= record + 183 d yet), CUK under
  1125259 to 2026-04-21 (real). Levels approximate.
- GOLD -> B 2025-05-09: B's SEC CIK 9984 (Barnes Group) before the record, 756894 (Barrick)
  >= 183 days after; closes 19.40 -> 19.46 (real, from the B recount). The snapshot dates are
  constructed; B's bars right before the record are modelled as Barnes's own at that level,
  while in the store they are imperfect copies of GOLD (SIMPLIFIED: both count as "differing").
- SCHN -> RDUS 2023-09-01 (real record; 659 parallel differing days, none in the last 5
  sessions): the previous RDUS holder (Radius Health) stopped trading in 2022-08 (ASSUMED
  2022-08-10; the exact day does not matter to the rule).
- FFR -> DTRE shape (OLDV / NEWV, dates constructed): under OHLCV identity 689 of DTRE's copies
  of FFR counted as differing because Alpaca revised the volume; under OHLC they are copies
  (real mechanism, Z table of the note).
- TMKRW -> PMGMW 2023-08-22 (real): a warrant renamed with its company, PMGMW carrying TMKRW's
  history relabelled.
- E worlds (ENDR / CONR): constructed; the real anchors are read on the store by
  tests/store/test_golden_cases.py (CYCN / KRSA). The CYCN shape itself (both labels start on
  the store's first day, every bar identical) is a fake world here too.
- AXNX <- MIMI: AXNX's last own bar 2024-11-14 (close 70.98), no traded bar until 2025-01-10,
  245 bars from 2025-01-10 identical to MIMI (close 4.11 that day), MIMI listed by the SEC under
  1998560 from 2024-11-28 (real). AXNX kept in the SEC table here so that only the gap triggers
  (SIMPLIFIED: the real AXNX rows end 2024-12-07).
- VJET <- VTAK: VJET's own bars to 2024-03-22 (close 0.6401), VTAK copies from 2024-03-25
  (0.501), VJET's SEC rows under 1582581 end 2024-04-01 and the next snapshot is 2024-05-02;
  VTAK under 1716621 from 2023-09-02 (real). The "safe" variant moves the copies to 2024-05-06,
  after that next snapshot.
- PARA shape: VIAC -> PARA 2022-02-17, VII -> BNZI 2023-12-19 (VII's last bar 12-14), BNZI ->
  PARA 2026-08-07; SEC PARA under 813828 2022-03-01..2025-08-19, VII / BNZI / PARA under
  1826011 (real). The copy shares (37% VII, 59% BNZI, 4% own) are CONSTRUCTED so that the
  whole-chain share clears 90% while no single symbol does; the store's PARA piece is 84%
  copies and is handled by F, not by this rule (see the assumptions list).
- BTLN: VRAR 2021-07-01..2026-02-18 (1,163 bars), VRAR -> GGRP 2026-02-19 (126 bars), GGRP ->
  BTLN 2026-08-20, BTLN's 1,289 bars before the takeover are exact copies of the two; SEC BTLN
  under 1425289 2021-06-03..2022-01-03, VRAR / GGRP / BTLN under 1854445 (real). Levels
  constructed.
- C (OLDC / NEWC), Z(b) (RSM / RSN / UNR), G (DSW, BRG, CBSX -> VIAX, LBL, DUPA / DUPB) and the
  import world (TAKR -> IMP): constructed; the AZRX / FWBI overlap is real (181 days), the
  CBS -> VIAC record is real (2020-02-13).

Assumptions that need a real-data check (audit or golden case) because the fake worlds cannot
tell whether they are right:
 1. E, "own bars": CYCN has NO OHLC-differing day against KRSA in the store (1,662 of 1,662
    traded days identical, both labels start 2020-01-02). The brief's condition "the ender has
    traded non-identical bars of its own" cannot hold for it; the fake CYCN-shape test and the
    golden case both assume the condition is waived when the run begins on the ender's first
    traded bar, and the "ender is all copies" negative assumes "own" means "in no identical
    pair at all". Check: the golden CYCN / KRSA case, and the audit's I2 / M6 counts for the
    18 pairs of the note's E table after the rebuild.
 2. E boundaries: "within RUN_NEAR_SESSIONS" read as <= 10 for the continuer's first bar; the
    tolerance read as count(differing) <= max(3, 0.05 * run); the cut dated the continuer's
    first bar after the run. Check: I3 (double days) and I1 (orphans of the continuer's label)
    on the 18 real pairs.
 3. F, CIK-leaves: VJET's copies start 2024-03-25, BEFORE its CIK's last snapshot (04-01); the
    brief's wording ("starts after the last snapshot + one interval") would not cut it. The
    VJET-exact fake and the golden case follow the note (the copies begin when the filer leaves
    the table). PARA <- BNZI has the same shape (copies from 2025-08-07, last snapshot 08-19).
 4. Relabel, whole chain: the store's Paramount PARA piece is 84% Banzai-chain copies, under the
    90% threshold, so it survives this rule; it is F that must end it at 2025-08-07. Check: the
    golden PARA cases and M2 after the rebuild.
 5. Z(a) is a backstop: in the BTLN / TE / FFR shapes the whole-chain relabel already drops the
    piece. Check: M7 on the real store lists no segment at share >= 0.9 after the rebuild.
 6. B1 on COT -> PRMW: ``drop_table_corrections`` removes old Primo's PRMW rows (CIK 1365101
    vanished in the 2020-07-10 correction), so PRMW's CIK before the record is UNKNOWN to the
    build, not "different"; the outcome (rename) is the same. Check: golden COT / PRMW.
 7. B1 on IIVIP -> IIVI 2023-02-23: the IIVI bars from that day (close 195) are the preferred's
    own trading under the vacated common ticker (the common had become COHR at ~42); IIVIP had
    stopped the day before, so no recent differing day and the record is a plain rename. A
    suffix-shape rule ("W/R/U/P old name into a plain name is a conversion") was considered and
    dropped for exactly this case; FATBW -> FAT and GSRTR -> NKLR are conversions by B1 alone
    (same CIK on the new label). Check: golden IIVIP / IIVI (one security).
"""

from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from asof.ingest import security_master as sm
from asof.ingest.corporate_actions import NAME_CHANGE_SCHEMA
from asof.ingest.sec_ticker_history import HISTORY_SCHEMA
from asof.ingest.security_master import build_security_master
from asof.store.market import market_root
from tests.ingest.fakes import daily
from tests.ingest.test_security_master import (
    Life,
    Rename,
    assert_end_knowability_invariant,
    assert_nothing_known_before_its_valid_from,
    bar_available_at,
    bar_rows,
    bars_table,
    business_days,
    conflicts_of,
    covering,
    et_midnight,
    filler_rows,
    name_changes_table,
    only_sid,
    relabel,
    sec_table,
    segment_at,
    segments,
    sids,
    visible_at,
)
from tests.store.test_market import store_daily
from tests.store.test_market_ext import ext_row, write_ext

MICRO = timedelta(microseconds=1)
S = business_days(date(2020, 1, 2), date(2026, 12, 31))  # S[i] is a Thursday iff i % 5 == 0
STORE_START = S[0]


# =============================================================================================
# helpers
# =============================================================================================


def build(
    bars: Sequence[dict[str, Any]],
    snapshots: Sequence[date] = (),
    lives: Sequence[Life] = (),
    renames: Sequence[Rename] = (),
) -> tuple[pa.Table, pa.Table]:
    return build_security_master(
        bars=bars_table(bars),
        sec_history=sec_table(snapshots, lives),
        name_changes=name_changes_table(renames),
    )


def between(rows: Sequence[dict[str, Any]], lo: date, hi: date) -> list[dict[str, Any]]:
    """Rows of a bar list with ``lo <= session_date <= hi``."""
    return [r for r in rows if lo <= r["session_date"] <= hi]


def revolume(rows: Sequence[dict[str, Any]], delta: int) -> list[dict[str, Any]]:
    """Alpaca's late volume revision: the same OHLC, the volume off by ``delta``."""
    return [{**r, "volume": r["volume"] + delta} for r in rows]


def one(master: pa.Table, symbol: str) -> dict[str, Any]:
    found = segments(master, symbol)
    assert len(found) == 1, (symbol, [(r["valid_from"], r["valid_to"]) for r in found])
    return found[0]


def span(row: dict[str, Any]) -> tuple[date, date | None]:
    return row["valid_from"], row["valid_to"]


def dup_pairs(conflicts: pa.Table) -> set[frozenset[str]]:
    return {frozenset(r["symbols"]) for r in conflicts_of(conflicts, "dup_unlinked")}


def bars_evidence(master: pa.Table) -> list[tuple[str, date]]:
    return [(r["symbol"], r["valid_from"]) for r in master.to_pylist() if r["evidence"] == "bars"]


def assert_no_overlap_within_a_chain(master: pa.Table) -> None:
    """Two segments of one security never cover the same day (the audit's M2)."""
    rows = master.to_pylist()
    for a in rows:
        for b in rows:
            if a is b or a["security_id"] != b["security_id"]:
                continue
            a_end = a["valid_to"] or date.max
            b_end = b["valid_to"] or date.max
            assert not (a["valid_from"] < b_end and b["valid_from"] < a_end), (a, b)


def monthly(start: date, end: date) -> list[date]:
    """First weekday on or after the 1st of every month in [start, end]: fake SEC snapshots."""
    out = []
    year, month = start.year, start.month
    while date(year, month, 1) <= end:
        day = date(year, month, 1)
        while day.weekday() >= 5:
            day += timedelta(days=1)
        if start <= day <= end:
            out.append(day)
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


# =============================================================================================
# constants and identical_days
# =============================================================================================


def test_copy_rule_constants() -> None:
    assert sm.RELABEL_MIN_IDENTICAL == 0.9
    assert sm.COPY_MIN_DAYS == 3
    assert sm.RUN_MIN_DAYS == 3
    assert sm.RUN_NEAR_SESSIONS == 10
    assert sm.RUN_TOLERANCE == (3, 0.05)
    assert sm.CONVERSION_MIN_DIFFERING_DAYS == 5
    assert sm.CONVERSION_WINDOW == 5
    assert sm.TAIL_GAP_SESSIONS == 10
    # 6-c rule R adds "relabelled" (tests/ingest/test_security_master_takeovers.py)
    assert set(sm.RESOLUTIONS) == {"new_security", "bridged", "ignored", "unresolved", "relabelled"}
    assert len(sm.RESOLUTIONS) == 5
    assert sm.CONFLICT_SCHEMA.names == ["kind", "symbols", "dates", "detail", "resolution"]
    assert sm.CONFLICT_SCHEMA.field("resolution").type == pa.string()
    assert sm.IDENT_SCHEMA.names == ["symbol_a", "symbol_b", "session_date"]
    assert sm.IDENT_SCHEMA.field("session_date").type == pa.date32()


def test_identical_days_pairs_ordered_volume_ignored_untraded_excluded() -> None:
    # ZED trades S[0..4]; ALF copies ZED on S[0..2]: S[0] exact, S[1] with the volume revised
    # (still identical: Alpaca revises volume after 20:00 ET), S[2] with volume 0 (untraded, not
    # a copy). ALF's own bar on S[3] differs. The pair comes back as (ALF, ZED) whatever the
    # input order, days S[0] and S[1] only.
    zed = bar_rows("ZED", S[0], S[4], 10.0)
    alf = [
        {**zed[0], "symbol": "ALF"},
        {**zed[1], "symbol": "ALF", "volume": zed[1]["volume"] + 999},
        {**zed[2], "symbol": "ALF", "volume": 0},
        *bar_rows("ALF", S[3], S[3], 11.0),
    ]

    table = sm.identical_days(bars_table([*zed, *alf]))

    assert table.schema.equals(sm.IDENT_SCHEMA)
    rows = sorted(table.to_pylist(), key=lambda r: r["session_date"])
    assert [(r["symbol_a"], r["symbol_b"], r["session_date"]) for r in rows] == [
        ("ALF", "ZED", S[0]),
        ("ALF", "ZED", S[1]),
    ]
    reversed_input = sm.identical_days(bars_table([*alf, *zed]))
    assert all(r["symbol_a"] < r["symbol_b"] for r in reversed_input.to_pylist())
    assert reversed_input.num_rows == 2


def test_identical_days_accepts_a_connection_and_an_empty_table() -> None:
    con = duckdb.connect()
    try:
        empty = sm.identical_days(bars_table([]), con)
        assert empty.num_rows == 0
        assert empty.schema.equals(sm.IDENT_SCHEMA)
        zed = bar_rows("ZED", S[0], S[2], 10.0)
        again = sm.identical_days(bars_table([*zed, *relabel(zed, "ALF")]), con)
        assert again.num_rows == 3
    finally:
        con.close()


# =============================================================================================
# B: a record whose two names traded in parallel
# =============================================================================================

B_CUK_RECORD = date(2026, 5, 8)  # real: Alpaca's process_date
B_CUK_LAST = date(2026, 5, 6)  # real: CUK's last bar
B_CUK_CUT = date(2026, 5, 7)  # the refined cut: the day after CUK's last bar (D rule)
B_CCL_CIK, B_CUK_CIK = 815097, 1125259  # real
B_CUK_SNAPSHOTS = [
    *monthly(date(2020, 1, 1), date(2026, 4, 30)),
    date(2026, 7, 16),
    date(2026, 9, 8),
]
B_CUK_LIVES: list[Life] = [
    ("CCL", B_CCL_CIK, B_CUK_SNAPSHOTS[0], None),  # real: listed in every snapshot
    ("CUK", B_CUK_CIK, B_CUK_SNAPSHOTS[0], date(2026, 4, 21)),  # real: last listed 2026-04-21
]


@pytest.fixture(scope="module")
def b_cuk_built() -> tuple[pa.Table, pa.Table]:
    return build(
        bar_rows("CUK", STORE_START, B_CUK_LAST, 27.0) + bar_rows("CCL", STORE_START, S[-1], 27.1),
        B_CUK_SNAPSHOTS,
        B_CUK_LIVES,
        [("CUK", "CCL", B_CUK_RECORD, "14365C103", "143658300")],  # CUSIPs constructed
    )


def test_b1_parallel_names_with_one_filer_on_the_new_label_are_a_conversion(
    b_cuk_built: tuple[pa.Table, pa.Table],
) -> None:
    # B1. Today (real master): CUK [2020-01-02, 2026-05-07) and CCL cut in two at 05-07, all
    # one security S001248. CUK and CCL differ on all 1,594 parallel days including the last
    # five sessions, and CCL's CIK is 815097 before and after: the plc ADR converted into the
    # Corp share, two securities. The CCL label is untouched (one segment, not cut at the
    # record and not clipped at CUK's first bar: the "non-identical overlap" of rule C is
    # exactly this shape and must be left alone).
    master, conflicts = b_cuk_built

    cuk = one(master, "CUK")
    ccl = one(master, "CCL")
    assert span(cuk) == (STORE_START, B_CUK_CUT)
    assert span(ccl) == (STORE_START, None)
    assert cuk["security_id"] != ccl["security_id"]
    assert (cuk["cik"], ccl["cik"]) == (B_CUK_CIK, B_CCL_CIK)
    assert ccl["evidence"] == "sec"
    assert segment_at(master, "CCL", B_CUK_CUT) == ccl
    assert not any({"CUK", "CCL"} & set(r["symbols"]) for r in conflicts.to_pylist())
    assert_no_overlap_within_a_chain(master)


def test_b1_conversion_end_is_knowable_at_the_record(
    b_cuk_built: tuple[pa.Table, pa.Table],
) -> None:
    # Leakage: the cut is 05-07 (CUK's last bar + 1) but Alpaca filed the record on 05-08; a
    # reader on 05-07 sees CUK open. Nothing about CCL changes at any instant.
    master, _ = b_cuk_built

    cuk = one(master, "CUK")
    assert cuk["end_available_at"] == et_midnight(B_CUK_RECORD)
    [seen] = [
        r for r in visible_at(master, et_midnight(B_CUK_RECORD) - MICRO) if r["symbol"] == "CUK"
    ]
    assert seen["end_available_at"] > et_midnight(B_CUK_CUT)
    assert one(master, "CCL")["available_at"] == bar_available_at(STORE_START)
    assert_nothing_known_before_its_valid_from(master)
    assert_end_knowability_invariant(master)


B_GOLD_RECORD = date(2025, 5, 9)  # real
B_BARRICK, B_BARNES = 756894, 9984  # real CIKs
B_GOLD_SNAPSHOTS = [
    *monthly(date(2020, 1, 1), date(2025, 5, 1)),  # 2025-05-01: last before the record
    date(2025, 6, 2),  # first after the record: still Barnes (the SEC lists the old filer)
    date(2025, 8, 1),
    date(2025, 11, 10),  # first >= record + 183 d (2025-11-08): Barrick
    date(2026, 1, 5),
]


def b_gold_world(*, sec_lag: bool) -> tuple[pa.Table, pa.Table]:
    """GOLD (Barrick) and B (Barnes) traded in parallel until the record; B then continues at
    Barrick's level. ``sec_lag``: the SEC still lists Barnes under B in the two snapshots after
    the record (real), or switches B to Barrick in the first one (constructed)."""
    switch = date(2025, 11, 10) if sec_lag else date(2025, 6, 2)
    lives: list[Life] = [
        ("GOLD", B_BARRICK, B_GOLD_SNAPSHOTS[0], date(2025, 5, 1)),
        ("B", B_BARNES, B_GOLD_SNAPSHOTS[0], B_GOLD_SNAPSHOTS[B_GOLD_SNAPSHOTS.index(switch) - 1]),
        ("B", B_BARRICK, switch, None),
    ]
    before = B_GOLD_RECORD - timedelta(days=1)
    barnes = bar_rows("B", STORE_START, before, 19.4)
    # B's close before the record 19.40, after it 19.46 (real): the acquisition was priced at
    # the market, so the label looks continuous although the filer changed
    return build(
        bar_rows("GOLD", STORE_START, before, 19.0)
        + barnes
        + bar_rows("B", B_GOLD_RECORD, date(2026, 1, 30), round(barnes[-1]["close"] * 1.003, 4)),
        B_GOLD_SNAPSHOTS,
        lives,
        [("GOLD", "B", B_GOLD_RECORD, "067901108", "067901108")],
    )


def test_b1_new_label_changing_filer_across_the_record_is_a_rename() -> None:
    # B1 negative (GOLD -> B). The parallel days are many and recent, but the B label's CIK is
    # Barnes before the record and Barrick after it: Barrick took the vacated ticker. GOLD and
    # the new B are one security; the old B is another.
    master, conflicts = b_gold_world(sec_lag=False)

    assert len(segments(master, "B")) == 2
    barnes, barrick = segments(master, "B")
    assert span(barnes) == (STORE_START, B_GOLD_RECORD)
    assert span(barrick) == (B_GOLD_RECORD, None)
    assert barrick["security_id"] == only_sid(master, "GOLD")
    assert barnes["security_id"] != barrick["security_id"]
    assert (barnes["cik"], barrick["cik"]) == (B_BARNES, B_BARRICK)
    assert one(master, "GOLD")["valid_to"] == B_GOLD_RECORD
    # the reuse is reported, not bridged; since 6-c rule R decides it: GOLD (the taker's chain)
    # traded bars of its own inside the old B piece's span, so the old B is a previous holder
    [row] = [r for r in conflicts_of(conflicts, "reuse_looks_continuous") if "B" in r["symbols"]]
    assert row["resolution"] == "new_security"


def test_b1_reads_the_new_labels_cik_at_least_183_days_after_the_record() -> None:
    # B1, the SEC lag (real: IR -> TT reads SAME in the first snapshot after the record). With
    # the first snapshot after the record B is still Barnes, so a "before vs first after"
    # comparison would call GOLD -> B a conversion and give GOLD a security of its own; the
    # snapshot >= record + SEC_LAG_MAX_DAYS says Barrick, so it is a rename and GOLD shares B's
    # renamed piece. (The lag also hands that piece Barnes's CIK and lets the same-CIK
    # continuity rule pull the old holder in, and the late switch on 2025-11-10 is bridged by
    # prices: the IR -> TT zombie shape. Only the date of the CIK test is pinned here.)
    master, conflicts = b_gold_world(sec_lag=True)

    gold = only_sid(master, "GOLD")
    assert gold in sids(master, "B")
    assert segment_at(master, "B", date(2025, 6, 2))["security_id"] == gold
    assert segment_at(master, "B", date(2026, 1, 5))["security_id"] == gold
    bridged = [r for r in conflicts_of(conflicts, "cik_switch_bridged") if r["symbols"] == ["B"]]
    assert [r["resolution"] for r in bridged] == ["bridged"]


B_RDUS_RECORD = date(2023, 9, 1)  # real: SCHN -> RDUS
B_RDUS_OLD_LAST = date(2022, 8, 10)  # ASSUMED: Radius Health's last bar under RDUS
B_RDUS_SNAPSHOTS = monthly(date(2020, 1, 1), date(2024, 1, 31))
B_RDUS_LIVES: list[Life] = [
    (
        "SCHN",
        912603,
        B_RDUS_SNAPSHOTS[0],
        date(2023, 8, 1),
    ),  # real CIK (Schnitzer / Radius Recycling)
    ("RDUS", 1428522, B_RDUS_SNAPSHOTS[0], date(2022, 9, 1)),  # real CIK (Radius Health)
    ("RDUS", 912603, date(2023, 10, 2), None),
]


def test_b1_many_parallel_days_but_none_recent_is_a_plain_rename() -> None:
    # B1 negative (SCHN -> RDUS; guard, passes today and must keep passing): 650 parallel
    # differing days with Radius Health, which stopped a year before the record. The old RDUS
    # holder is its own security; SCHN and the new RDUS are one.
    master, _ = build(
        bar_rows("SCHN", STORE_START, B_RDUS_RECORD - timedelta(days=1), 30.0)
        + bar_rows("RDUS", STORE_START, B_RDUS_OLD_LAST, 12.0)
        + bar_rows("RDUS", B_RDUS_RECORD, date(2024, 1, 31), 30.1),
        B_RDUS_SNAPSHOTS,
        B_RDUS_LIVES,
        [("SCHN", "RDUS", B_RDUS_RECORD, "806882106", "806882106")],
    )

    assert len(segments(master, "RDUS")) == 2
    radius_health, radius_recycling = segments(master, "RDUS")
    assert span(radius_health) == (STORE_START, B_RDUS_RECORD)
    assert span(radius_recycling) == (B_RDUS_RECORD, None)
    assert radius_recycling["security_id"] == only_sid(master, "SCHN")
    assert radius_health["security_id"] != radius_recycling["security_id"]
    assert one(master, "SCHN")["valid_to"] == B_RDUS_RECORD


B_V_RECORD = date(2021, 9, 1)  # constructed (real shape FFR -> DTRE 2022-10-03)
B_V_CIK = 7101
B_V_SNAPSHOTS = monthly(date(2020, 1, 1), date(2021, 12, 31))
B_V_LIVES: list[Life] = [
    ("OLDV", B_V_CIK, B_V_SNAPSHOTS[0], date(2021, 8, 2)),
    ("NEWV", B_V_CIK, date(2021, 3, 1), None),  # listed before the record: an opening piece
]


def test_b1_copies_that_differ_only_in_volume_are_not_differing_days() -> None:
    # B1 and relabel on OHLC, not OHLCV. NEWV carries OLDV's whole history with every volume
    # revised by a few shares. Under OHLCV identity those 418 days are "differing", recent and
    # under one CIK -> a conversion, two securities; and NEWV's opening piece survives the
    # relabel check and overlaps OLDV. Under OHLC they are copies: one security, NEWV from the
    # record only.
    oldv = bar_rows("OLDV", STORE_START, B_V_RECORD - timedelta(days=1), 5.5)
    master, conflicts = build(
        oldv
        + revolume(relabel(oldv, "NEWV"), 7)
        + bar_rows("NEWV", B_V_RECORD, date(2021, 12, 31), 5.6),
        B_V_SNAPSHOTS,
        B_V_LIVES,
        [("OLDV", "NEWV", B_V_RECORD, "337364103", "337364103")],
    )

    newv = one(master, "NEWV")
    oldv_seg = one(master, "OLDV")
    assert span(newv) == (B_V_RECORD, None)
    assert span(oldv_seg) == (STORE_START, B_V_RECORD)
    assert newv["security_id"] == oldv_seg["security_id"]
    assert newv["evidence"] == "name_change"
    assert covering(master, "NEWV", date(2021, 3, 1)) == []
    assert conflicts.num_rows == 0
    assert_no_overlap_within_a_chain(master)


def test_b1_renamed_warrant_whose_history_is_relabelled_is_a_plain_rename() -> None:
    # B1 negative (TMKRW -> PMGMW 2023-08-22, real; guard, passes today): a warrant renamed with
    # its company. PMGMW carries TMKRW's history relabelled (no differing parallel day) and its
    # own bars from the record: one security.
    record = date(2023, 8, 22)
    tmkrw = bar_rows("TMKRW", STORE_START, record - timedelta(days=1), 0.3)
    master, _ = build(
        tmkrw + relabel(tmkrw, "PMGMW") + bar_rows("PMGMW", record, date(2023, 12, 29), 0.31),
        renames=[("TMKRW", "PMGMW", record, "87279R115", "72581J117")],
    )

    pmgmw = one(master, "PMGMW")
    assert span(pmgmw) == (record, None)
    assert pmgmw["security_id"] == only_sid(master, "TMKRW")
    assert pmgmw["evidence"] == "name_change"
    assert one(master, "TMKRW")["valid_to"] == record


# =============================================================================================
# E: the rename Alpaca did not file
# =============================================================================================

E_ENDER_CIK, E_CONTINUER_CIK = 8001, 8002


def e_world(
    *, run: int, lead: int, after: int = 240
) -> tuple[tuple[pa.Table, pa.Table], dict[str, Any]]:
    """ENDR trades its own bars from the store's start; CONR lists and starts trading ``lead``
    sessions before the run with its own bars; on the ``run`` sessions the two labels carry the
    same bars (CONR's, filed under ENDR as well); ENDR stops on the run's last day, CONR trades
    on for ``after`` sessions. The run ends on a Thursday so that the Friday after it is both
    "run end + 1 calendar day" and CONR's first bar after the run. No record anywhere."""
    start = 260 + ((1 - run) % 5)
    assert S[start + run - 1].weekday() == 3
    cut = S[start + run]
    conr = bar_rows("CONR", S[start - lead], S[start + run + after - 1], 20.5)
    bars = [
        *bar_rows("ENDR", S[0], S[start - 1], 20.0),
        *relabel(between(conr, S[start], S[start + run - 1]), "ENDR"),
        *conr,
    ]
    snapshots = monthly(S[0], S[start + run + after - 1])
    lives: list[Life] = [
        ("ENDR", E_ENDER_CIK, snapshots[0], None),  # listed throughout: no F trigger
        ("CONR", E_CONTINUER_CIK, S[start - lead], None),
    ]
    facts = {
        "run_start": S[start],
        "run_end": S[start + run - 1],
        "cut": cut,
        "first": S[start - lead],
        "run": run,
    }
    return build(bars, snapshots, lives), facts


@pytest.fixture(scope="module")
def e_built() -> tuple[tuple[pa.Table, pa.Table], dict[str, Any]]:
    return e_world(run=130, lead=0)


def test_e_identical_run_where_one_name_stops_links_the_two_labels(
    e_built: tuple[tuple[pa.Table, pa.Table], dict[str, Any]],
) -> None:
    # E. Today: ENDR and CONR are two securities (CONR 'sec' from its first bar) and a
    # dup_unlinked conflict. Every condition holds: run 130 days, ENDR's own bars run into the
    # run, CONR's first bar is the run's first day, no differing day, no edge, not one security.
    (master, conflicts), facts = e_built

    endr = one(master, "ENDR")
    conr = one(master, "CONR")
    assert span(endr) == (S[0], facts["cut"])
    assert span(conr) == (facts["cut"], None)
    assert endr["security_id"] == conr["security_id"]
    assert conr["evidence"] == "bars"
    assert conr["cik"] == E_CONTINUER_CIK
    assert conr["cusip"] is None
    assert covering(master, "CONR", facts["run_start"]) == []  # the copies own nothing
    assert segment_at(master, "ENDR", facts["run_end"]) == endr
    assert frozenset({"ENDR", "CONR"}) not in dup_pairs(conflicts)
    assert_no_overlap_within_a_chain(master)


def test_e_new_piece_is_knowable_with_the_continuers_first_bar_after_the_run(
    e_built: tuple[tuple[pa.Table, pa.Table], dict[str, Any]],
) -> None:
    # Leakage: nothing but the bars says the rename happened, so the new piece and the end of
    # the old one are knowable when CONR's first post-run bar is (20:00 New York), not at
    # midnight of the cut and not with the run's last identical bar.
    (master, _), facts = e_built

    endr = one(master, "ENDR")
    conr = one(master, "CONR")
    instant = bar_available_at(facts["cut"])
    assert conr["available_at"] == instant
    assert endr["end_available_at"] == instant
    assert instant > et_midnight(facts["cut"])
    before = visible_at(master, instant - MICRO)
    assert "CONR" not in {r["symbol"] for r in before}
    [endr_seen] = [r for r in before if r["symbol"] == "ENDR"]
    assert endr_seen["end_available_at"] > instant - MICRO
    at = visible_at(master, instant)
    assert {r["symbol"] for r in at} == {"ENDR", "CONR"}
    assert_nothing_known_before_its_valid_from(master)
    assert_end_knowability_invariant(master)


def assert_e_negative(master: pa.Table, conflicts: pa.Table, facts: dict[str, Any]) -> None:
    """The pair stays two securities and a dup_unlinked conflict; no bars-evidence piece."""
    assert sids(master, "ENDR").isdisjoint(sids(master, "CONR"))
    assert bars_evidence(master) == []
    assert span(one(master, "CONR")) == (facts["first"], None)
    # fewer than DUP_MIN_DAYS identical days are a coincidence, not a conflict
    reported = frozenset({"ENDR", "CONR"}) in dup_pairs(conflicts)
    assert reported is (facts["run"] >= sm.DUP_MIN_DAYS)


@pytest.mark.parametrize(
    ("run", "lead", "linked"),
    [
        (2, 0, False),  # shorter than RUN_MIN_DAYS
        (3, 0, True),  # the shortest run that counts
        (300, 10, True),  # continuer's first bar 10 sessions before the run: within
        (300, 11, False),  # 11: the continuer had its own life (BNZI / PARA shape)
        (100, 5, True),  # tolerance 5% of 100 = 5 differing days
        (100, 6, False),  # one differing day too many
        (40, 3, True),  # tolerance floor: max(3, 5% of 40 = 2) = 3
        (40, 4, False),
        (10, 2, True),  # linked although the opening piece is only 83% identical (10 of 12)
    ],
    ids=[
        "run_of_two",
        "run_of_three",
        "lead_ten",
        "lead_eleven",
        "five_differing_of_hundred",
        "six_differing_of_hundred",
        "three_differing_floor",
        "four_differing_floor",
        "short_run_two_own_days",
    ],
)
def test_e_run_length_lead_and_tolerance_boundaries(run: int, lead: int, linked: bool) -> None:
    # E boundaries (constructed): the lead sessions are days on which both labels trade their
    # own bars, so "continuer's first bar within RUN_NEAR_SESSIONS of the run start" and
    # "differing days <= max(3, 5% of the run)" are read off the same number.
    (master, conflicts), facts = e_world(run=run, lead=lead)

    if linked:
        conr = one(master, "CONR")
        assert span(conr) == (facts["cut"], None)
        assert conr["evidence"] == "bars"
        assert conr["security_id"] == only_sid(master, "ENDR")
        assert one(master, "ENDR")["valid_to"] == facts["cut"]
    else:
        assert_e_negative(master, conflicts, facts)


def f_gap_world() -> tuple[pa.Table, pa.Table]:
    """AXNX <- MIMI: AXNX's own bars stop 2024-11-14, nothing trades under the label for 57
    calendar days, then 245 bars identical to MIMI's (listed 2024-11-28, trading from 2025-01-10)
    sit under AXNX. No record. AXNX is kept in the SEC table so that only the gap triggers."""
    mimi = bar_rows("MIMI", date(2025, 1, 10), date(2026, 10, 12), 4.11)
    snapshots = [*monthly(date(2020, 1, 1), date(2026, 10, 1)), date(2024, 11, 28)]
    snapshots.sort()
    return build(
        bar_rows("AXNX", STORE_START, date(2024, 11, 14), 70.0)
        + relabel(between(mimi, date(2025, 1, 10), date(2025, 12, 31)), "AXNX")
        + mimi,
        snapshots,
        [("AXNX", 1603756, snapshots[0], None), ("MIMI", 1998560, date(2024, 11, 28), None)],
    )


def test_e_gap_between_the_enders_own_bars_and_the_run_is_not_a_rename() -> None:
    # E negative, "gap before the run" (AXNX / MIMI): MIMI's first bar is the run's first day
    # and it has no differing day, but AXNX's last own bar is 38 sessions before the run. The
    # pair is two securities (F owns the shape, below).
    master, _ = f_gap_world()

    assert sids(master, "AXNX").isdisjoint(sids(master, "MIMI"))
    assert bars_evidence(master) == []
    assert span(one(master, "MIMI")) == (date(2025, 1, 10), None)


def test_e_ender_whose_every_bar_is_a_copy_is_not_a_rename() -> None:
    # E negative, "ender is all copies": ENDC copies THRD for 130 sessions, then CONC for 130
    # sessions, then stops; CONC's first bar is the run's first day and it has no differing day.
    # ENDC never had a bar of its own (one that is in no identical pair), so it is a relabelled
    # ghost, not a renamed security. ASSUMPTION: "own bars" means days in no identical pair at
    # all (the earlier THRD copies differ from CONC but are copies too).
    thrd = bar_rows("THRD", S[0], S[799], 30.0)
    conc = bar_rows("CONC", S[130], S[799], 40.0)
    master, conflicts = build(
        thrd
        + conc
        + relabel(between(thrd, S[0], S[129]), "ENDC")
        + relabel(between(conc, S[130], S[259]), "ENDC"),
    )

    assert len({only_sid(master, "ENDC"), only_sid(master, "CONC"), only_sid(master, "THRD")}) == 3
    assert bars_evidence(master) == []
    assert span(one(master, "CONC")) == (S[130], None)
    assert frozenset({"ENDC", "CONC"}) in dup_pairs(conflicts)
    assert frozenset({"ENDC", "THRD"}) in dup_pairs(conflicts)


def test_e_run_that_begins_on_the_enders_first_bar_needs_no_own_bars() -> None:
    # E, the CYCN / KRSA shape (real: both labels start on the store's first day, all 1,662 of
    # CYCN's traded bars identical to KRSA's, CYCN stops 2026-09-08, KRSA trades on). Nobody can
    # tell from the bars which label had the life before the store began, so the "own bars"
    # condition is waived when the run begins on the ender's first traded bar. ASSUMPTION
    # (header, item 1); the golden case pins the real pair.
    conk = bar_rows("CONK", S[0], S[799], 2.57)
    master, conflicts = build(
        relabel(between(conk, S[0], S[399]), "ENDK") + conk,
        monthly(S[0], S[799]),
        [("ENDK", 1755237, S[0], None)],
    )

    endk = one(master, "ENDK")
    conk_seg = one(master, "CONK")
    assert span(endk) == (S[0], S[400])
    assert span(conk_seg) == (S[400], None)
    assert conk_seg["security_id"] == endk["security_id"]
    assert conk_seg["evidence"] == "bars"
    assert conk_seg["available_at"] == endk["end_available_at"] == bar_available_at(S[400])
    assert frozenset({"ENDK", "CONK"}) not in dup_pairs(conflicts)
    assert_end_knowability_invariant(master)


def test_e_when_issued_name_is_never_an_ender_or_a_continuer() -> None:
    # E negative (real shape SGDVV / RENX 2023-09-28): a when-issued symbol prints the regular
    # share's bars for a few sessions and stops; that is rule A's shape (its own security,
    # never chained), not an unfiled rename. No bars edge, whichever side the V name is on.
    regular = bar_rows("RENX", S[0], S[400], 10.0)
    master, _ = build(
        relabel(between(regular, S[0], S[4]), "SGDVV") + regular,
        monthly(S[0], S[400]),
        [("RENX", 7801, S[0], None)],
    )

    assert bars_evidence(master) == []
    assert sids(master, "SGDVV").isdisjoint(sids(master, "RENX"))


def test_e_pair_already_joined_by_a_record_gets_no_second_edge() -> None:
    # E negative, "already an edge" (guard, passes today): OLDE -> NEWE is filed; NEWE carries
    # OLDE's last six months relabelled. The record owns the cut and the evidence.
    record = S[265]
    olde = bar_rows("OLDE", S[0], S[264], 12.0)
    master, conflicts = build(
        olde
        + relabel(between(olde, S[135], S[264]), "NEWE")
        + bar_rows("NEWE", record, S[500], 12.1),
        renames=[("OLDE", "NEWE", record, "000100100", "000100200")],
    )

    newe = one(master, "NEWE")
    assert span(newe) == (record, None)
    assert newe["evidence"] == "name_change"
    assert newe["security_id"] == only_sid(master, "OLDE")
    assert bars_evidence(master) == []
    assert conflicts.num_rows == 0


def test_e_pair_already_one_security_through_a_third_label_gets_no_edge() -> None:
    # E negative, "already one security" (guard, passes today): A -> M -> B by two records; B
    # carries A's last eight months relabelled, A's last bar is the run's last day, B trades on,
    # no record joins A and B directly. An E edge A -> B would be dated the day of A -> M and
    # start B's label twice.
    a_to_m, m_to_b = S[300], S[385]
    a = bar_rows("AAA1", S[0], S[299], 10.0)
    master, conflicts = build(
        a
        + bar_rows("MMM1", a_to_m, S[384], 10.1)
        + relabel(between(a, S[130], S[299]), "BBB1")
        + bar_rows("BBB1", m_to_b, S[600], 10.2),
        renames=[
            ("AAA1", "MMM1", a_to_m, "000200100", "000200200"),
            ("MMM1", "BBB1", m_to_b, "000200200", "000200300"),
        ],
    )

    bbb = one(master, "BBB1")
    assert span(bbb) == (m_to_b, None)
    assert bbb["evidence"] == "name_change"
    assert only_sid(master, "AAA1") == only_sid(master, "MMM1") == bbb["security_id"]
    assert span(one(master, "AAA1")) == (S[0], a_to_m)
    assert bars_evidence(master) == []
    assert covering(master, "BBB1", S[200]) == []
    assert conflicts.num_rows == 0


# =============================================================================================
# relabel: the taker's whole chain
# =============================================================================================

R_VIAC_TO_PARA = date(2022, 2, 17)  # real
R_VII_TO_BNZI = date(2023, 12, 19)  # real record; VII's last bar 2023-12-14, BNZI from 12-15
R_BNZI_FIRST = date(2023, 12, 15)
R_BNZI_TO_PARA = date(2026, 8, 7)  # real
R_PARAMOUNT, R_BANZAI = 813828, 1826011  # real CIKs
R_SNAPSHOTS = sorted(
    {
        *monthly(date(2020, 1, 1), date(2026, 9, 1)),
        date(2020, 2, 16),
        date(2025, 8, 19),
        date(2026, 9, 8),
    }
)
R_LIVES: list[Life] = [
    ("VIAC", R_PARAMOUNT, date(2020, 2, 16), date(2022, 2, 1)),
    ("PARA", R_PARAMOUNT, date(2022, 3, 1), date(2025, 8, 19)),
    ("VII", R_BANZAI, date(2021, 1, 4), date(2023, 12, 1)),
    ("BNZI", R_BANZAI, date(2024, 1, 2), date(2026, 8, 3)),
    ("PARA", R_BANZAI, date(2026, 9, 8), None),
]
R_RENAMES: list[Rename] = [
    ("VIAC", "PARA", R_VIAC_TO_PARA, "92556H206", "92556H206"),
    ("VII", "BNZI", R_VII_TO_BNZI, "92868C106", "06683L104"),
    ("BNZI", "PARA", R_BNZI_TO_PARA, "06683L104", "06683L104"),
]
R_PARA_OWN_LAST = date(2022, 4, 29)  # constructed: the last of PARA's ~51 own bars


def r_bars() -> list[dict[str, Any]]:
    vii = bar_rows("VII", STORE_START, date(2023, 12, 14), 10.0)
    bnzi = bar_rows("BNZI", R_BNZI_FIRST, date(2026, 8, 6), 1.9)
    return [
        *bar_rows("VIAC", STORE_START, date(2022, 2, 16), 35.0),
        *bar_rows("PARA", R_VIAC_TO_PARA, R_PARA_OWN_LAST, 30.0),  # Paramount's own: ~4%
        *relabel(between(vii, date(2022, 5, 2), date(2023, 12, 14)), "PARA"),  # ~37% VII
        *relabel(bnzi, "PARA"),  # ~59% BNZI
        *bar_rows("PARA", R_BNZI_TO_PARA, date(2026, 10, 16), 1.76),  # Banzai's own
        *vii,
        *bnzi,
    ]


@pytest.fixture(scope="module")
def r_built() -> tuple[pa.Table, pa.Table]:
    return build(r_bars(), R_SNAPSHOTS, R_LIVES, R_RENAMES)


def test_relabel_compares_the_earlier_piece_with_the_takers_whole_chain(
    r_built: tuple[pa.Table, pa.Table],
) -> None:
    # Relabel, whole chain. The PARA piece before Banzai's takeover is 96% copies of VII + BNZI
    # but only 59% of BNZI, the taker's old name. Today the comparison is against BNZI alone and
    # the piece survives as Paramount's; it is Banzai's relabelled history and gets no segment.
    master, conflicts = r_built

    para = one(master, "PARA")
    assert span(para) == (R_BNZI_TO_PARA, None)
    assert para["security_id"] == only_sid(master, "VII") == only_sid(master, "BNZI")
    assert para["cik"] == R_BANZAI
    assert covering(master, "PARA", date(2024, 6, 3)) == []
    assert covering(master, "PARA", date(2022, 3, 1)) == []
    viac = one(master, "VIAC")
    assert span(viac) == (STORE_START, R_VIAC_TO_PARA)
    assert viac["security_id"] != para["security_id"]
    assert conflicts_of(conflicts, "reuse_looks_continuous") == []
    assert frozenset({"PARA", "VII"}) not in dup_pairs(conflicts)


def test_relabel_component_is_built_without_the_taken_label_itself() -> None:
    # Guard (passes today, must keep passing): OLDX -> TMPX -> TMP2 -> OLDX. The taker's chain
    # reaches OLDX's earlier piece only through the label OLDX itself; with the label removed
    # from the graph the component of TMP2 is {TMPX, TMP2} and the earlier piece is compared
    # with their bars, never with its own (100% identical) bars.
    out, back_1, back_2 = date(2021, 3, 1), date(2021, 5, 3), date(2021, 7, 1)
    master, _ = build(
        bar_rows("OLDX", STORE_START, date(2021, 2, 26), 10.0)
        + bar_rows("TMPX", out, date(2021, 4, 30), 10.05)
        + bar_rows("TMP2", back_1, date(2021, 6, 30), 10.1)
        + bar_rows("OLDX", back_2, date(2021, 12, 31), 10.15),
        monthly(date(2020, 1, 1), date(2021, 12, 31)),
        [("OLDX", 7301, date(2020, 1, 1), None)],
        [
            ("OLDX", "TMPX", out, "000300100", "000300100"),
            ("TMPX", "TMP2", back_1, "000300100", "000300100"),
            ("TMP2", "OLDX", back_2, "000300100", "000300100"),
        ],
    )

    assert len(segments(master, "OLDX")) == 2
    early, late = segments(master, "OLDX")
    assert span(early) == (STORE_START, out)
    assert span(late) == (back_2, None)
    assert (
        early["security_id"]
        == late["security_id"]
        == only_sid(master, "TMPX")
        == only_sid(master, "TMP2")
    )
    assert covering(master, "OLDX", date(2020, 6, 1)) == [early]


# =============================================================================================
# C: identical overlap inside one chain
# =============================================================================================

C_RECORD = date(2021, 9, 1)  # constructed (real shape AZRX -> FWBI, 181 overlapping days)
C_NEWC_FIRST = date(2020, 6, 1)
C_OLDC_LAST_OWN = date(2020, 5, 29)
C_COPIES_FROM = date(2021, 3, 1)
C_CIK = 7401
C_SNAPSHOTS = monthly(date(2020, 1, 1), date(2021, 12, 31))
C_LIVES: list[Life] = [
    ("OLDC", C_CIK, C_SNAPSHOTS[0], date(2020, 6, 1)),
    ("NEWC", C_CIK, date(2020, 7, 1), None),
]


def c_bars() -> list[dict[str, Any]]:
    oldc = bar_rows("OLDC", STORE_START, C_OLDC_LAST_OWN, 8.0)
    newc = bar_rows("NEWC", C_NEWC_FIRST, date(2021, 12, 31), 8.2)
    return [
        *oldc,
        # Alpaca's volume-0 filler under the dead label, then NEWC's bars filed under OLDC
        *filler_rows(oldc[-1], C_NEWC_FIRST, date(2021, 2, 26)),
        *relabel(between(newc, C_COPIES_FROM, date(2021, 8, 31)), "OLDC"),
        *newc,
    ]


@pytest.fixture(scope="module")
def c_built() -> tuple[pa.Table, pa.Table]:
    return build(
        c_bars(), C_SNAPSHOTS, C_LIVES, [("OLDC", "NEWC", C_RECORD, "000400100", "000400100")]
    )


def test_c_old_label_whose_bars_in_the_overlap_are_copies_ends_where_the_new_one_begins(
    c_built: tuple[pa.Table, pa.Table],
) -> None:
    # C. NEWC's opening piece (own bars from 2020-06-01, listed from 07-01) is 40% identical to
    # OLDC and survives the relabel check; OLDC's piece runs to the record (its last "traded"
    # bar is a copy on 08-31) and overlaps it for 15 months -- today an M2 row. OLDC's traded
    # bars inside the overlap are 126 copies of NEWC, 100%: OLDC ends where NEWC begins.
    master, conflicts = c_built

    oldc = one(master, "OLDC")
    assert len(segments(master, "NEWC")) == 2
    opening, renamed = segments(master, "NEWC")
    assert span(oldc) == (STORE_START, C_NEWC_FIRST)
    assert span(opening) == (C_NEWC_FIRST, C_RECORD)
    assert span(renamed) == (C_RECORD, None)
    assert oldc["security_id"] == opening["security_id"] == renamed["security_id"]
    assert covering(master, "OLDC", date(2021, 5, 3)) == []  # the copies own nothing
    assert covering(master, "OLDC", date(2020, 11, 2)) == []  # nor does the filler
    assert segment_at(master, "NEWC", date(2021, 5, 3)) == opening
    assert conflicts.num_rows == 0
    assert_no_overlap_within_a_chain(master)


def test_c_clipped_end_is_knowable_with_the_later_piece(c_built: tuple[pa.Table, pa.Table]) -> None:
    # Leakage: OLDC's end is the start of NEWC's opening piece, knowable with NEWC's first bar
    # (20:00 New York of 2020-06-01); a reader a microsecond earlier sees OLDC open.
    master, _ = c_built

    oldc = one(master, "OLDC")
    opening = segments(master, "NEWC")[0]
    assert oldc["end_available_at"] == opening["available_at"] == bar_available_at(C_NEWC_FIRST)
    [seen] = [
        r
        for r in visible_at(master, bar_available_at(C_NEWC_FIRST) - MICRO)
        if r["symbol"] == "OLDC"
    ]
    assert seen["end_available_at"] > bar_available_at(C_NEWC_FIRST) - MICRO
    assert_nothing_known_before_its_valid_from(master)
    assert_end_knowability_invariant(master)


# =============================================================================================
# F: copied tails
# =============================================================================================

F_AXNX_TAIL = date(2025, 1, 10)  # real: first MIMI copy under AXNX


def test_f_tail_after_a_gap_that_copies_another_symbol_is_cut_off() -> None:
    # F, gap variant (AXNX <- MIMI). Today (real master): AXNX [2020-01-02, None) owns the 245
    # MIMI bars and I2 reports the pair. AXNX ends on the tail's first day; the tail owns
    # nothing; MIMI keeps its own security from its first bar.
    master, conflicts = f_gap_world()

    axnx = one(master, "AXNX")
    mimi = one(master, "MIMI")
    assert span(axnx) == (STORE_START, F_AXNX_TAIL)
    assert span(mimi) == (F_AXNX_TAIL, None)
    assert axnx["security_id"] != mimi["security_id"]
    assert covering(master, "AXNX", date(2025, 6, 2)) == []
    assert covering(master, "AXNX", F_AXNX_TAIL) == []
    assert segment_at(master, "AXNX", date(2024, 11, 14)) == axnx
    # the cut tail belongs to nobody, so there is nothing left to link: no dup_unlinked row
    # (the loss metric I1 names the tail; I2 sees no two securities sharing it)
    assert frozenset({"AXNX", "MIMI"}) not in dup_pairs(conflicts)


def test_f_cut_end_is_knowable_with_the_first_tail_bar_and_never_earlier() -> None:
    # Leakage: the tail bar itself is the evidence that the label changed hands; before 20:00
    # New York of 2025-01-10 a reader sees AXNX open, and the end is never earlier than that
    # bar's availability (an end at midnight of the tail day would leak the day's bar).
    master, _ = f_gap_world()

    axnx = one(master, "AXNX")
    instant = bar_available_at(F_AXNX_TAIL)
    assert axnx["end_available_at"] == instant
    assert axnx["end_available_at"] >= bar_available_at(F_AXNX_TAIL)
    [seen] = [r for r in visible_at(master, instant - MICRO) if r["symbol"] == "AXNX"]
    assert seen["end_available_at"] > instant - MICRO
    assert_nothing_known_before_its_valid_from(master)
    assert_end_knowability_invariant(master)


F_VJET_CIK, F_VTAK_CIK = 1582581, 1716621  # real
F_VJET_LAST_SNAPSHOT = date(2024, 4, 1)  # real: last SEC row of VJET
F_NEXT_SNAPSHOT = date(2024, 5, 2)  # real: the next snapshot in the table
F_VJET_SNAPSHOTS = sorted(
    {
        *monthly(date(2020, 1, 1), date(2026, 10, 1)),
        F_VJET_LAST_SNAPSHOT,
        F_NEXT_SNAPSHOT,
        date(2023, 9, 2),
    }
)


def f_cik_world(copies_from: date) -> tuple[pa.Table, pa.Table]:
    """VJET's own bars stop the session before ``copies_from``; from there its bars are VTAK's.
    VJET's filer leaves the SEC table after 2024-04-01; OTHR keeps every snapshot date alive."""
    vtak = bar_rows("VTAK", date(2023, 8, 18), date(2026, 10, 9), 0.5)
    own_last = business_days(STORE_START, copies_from - timedelta(days=1))[-1]
    return build(
        bar_rows("VJET", STORE_START, own_last, 0.6)
        + relabel(between(vtak, copies_from, date(2026, 10, 2)), "VJET")
        + vtak
        + bar_rows("OTHR", STORE_START, date(2026, 10, 9), 100.0),
        F_VJET_SNAPSHOTS,
        [
            ("VJET", F_VJET_CIK, F_VJET_SNAPSHOTS[0], F_VJET_LAST_SNAPSHOT),
            ("VTAK", F_VTAK_CIK, date(2023, 9, 2), None),
            ("OTHR", 7501, F_VJET_SNAPSHOTS[0], None),
        ],
    )


def test_f_tail_copying_another_symbol_after_the_filer_left_the_sec_table_is_cut_off() -> None:
    # F, CIK-leaves variant, safe shape: no gap in VJET's traded bars; the copies begin
    # 2024-05-06, after the filer's last snapshot (04-01) plus one interval (05-02). Today: VJET
    # one piece to the SEC-seen end, the copies owned by Voxeljet.
    copies_from = date(2024, 5, 6)
    master, conflicts = f_cik_world(copies_from)

    vjet = one(master, "VJET")
    vtak = one(master, "VTAK")
    assert span(vjet) == (STORE_START, copies_from)
    assert vjet["end_available_at"] == bar_available_at(copies_from)
    assert vjet["security_id"] != vtak["security_id"]
    assert span(vtak) == (date(2023, 8, 18), None)
    assert covering(master, "VJET", date(2025, 1, 2)) == []
    assert bars_evidence(master) == []  # VTAK had its own life: no E edge
    assert frozenset({"VJET", "VTAK"}) not in dup_pairs(conflicts)  # the tail owns nothing
    assert_end_knowability_invariant(master)


def test_f_copies_that_begin_just_before_the_filers_last_snapshot_are_the_vjet_shape() -> None:
    # F, CIK-leaves variant, the real VJET dates: Voxeljet's own bars stop 2024-03-22, VTAK's
    # copies start 03-25, the SEC lists VJET once more on 04-01 and never again. The note's
    # reading ("the copies begin when the holder leaves the table") cuts at 03-25; the brief's
    # wording ("starts after the last snapshot + one interval") would not. Golden case
    # test_vjet_ends_on_2024_03_25... pins the same date on the store (header, item 3).
    copies_from = date(2024, 3, 25)
    master, _ = f_cik_world(copies_from)

    vjet = one(master, "VJET")
    assert span(vjet) == (STORE_START, copies_from)
    assert vjet["end_available_at"] == bar_available_at(copies_from)
    assert covering(master, "VJET", date(2024, 6, 3)) == []


def test_f_tail_identical_to_a_symbol_of_the_same_chain_is_not_cut() -> None:
    # F negative (guard, passes today): OLDT -> NEWT is filed late; OLDT's own bars stop
    # 2021-01-29, nothing for 15 sessions, then NEWT's bars sit under OLDT until the record. The
    # tail starts after a gap and is 100% identical -- to NEWT, the same security. OLDT keeps
    # running to the record; NEWT's copies of the same days are the relabelled side and own
    # nothing.
    record = date(2021, 9, 1)
    newt = bar_rows("NEWT", date(2021, 2, 22), date(2021, 12, 31), 15.0)
    master, conflicts = build(
        bar_rows("OLDT", STORE_START, date(2021, 1, 29), 14.9)
        + relabel(between(newt, date(2021, 2, 22), date(2021, 8, 31)), "OLDT")
        + newt,
        monthly(date(2020, 1, 1), date(2021, 12, 31)),
        [
            ("OLDT", 7601, date(2020, 1, 1), date(2021, 2, 1)),
            ("NEWT", 7601, date(2021, 3, 1), None),
        ],
        [("OLDT", "NEWT", record, "000500100", "000500100")],
    )

    oldt = one(master, "OLDT")
    newt_seg = one(master, "NEWT")
    assert span(oldt) == (STORE_START, record)
    assert span(newt_seg) == (record, None)
    assert oldt["security_id"] == newt_seg["security_id"]
    assert covering(master, "OLDT", date(2021, 5, 3)) == [oldt]
    assert covering(master, "NEWT", date(2021, 5, 3)) == []
    assert conflicts.num_rows == 0


# =============================================================================================
# Z: old-holder pieces and resumed pieces that are copies
# =============================================================================================

Z_VRAR_TO_GGRP = date(2026, 2, 19)  # real
Z_GGRP_TO_BTLN = date(2026, 8, 20)  # real
Z_GLIMPSE, Z_OLD_BTLN = 1854445, 1425289  # real CIKs
Z_SNAPSHOTS = sorted({*monthly(date(2020, 1, 1), date(2026, 9, 1)), date(2026, 9, 8)})
Z_LIVES: list[Life] = [
    ("BTLN", Z_OLD_BTLN, date(2021, 6, 1), date(2022, 1, 3)),  # real: six snapshots
    ("VRAR", Z_GLIMPSE, date(2021, 5, 3), date(2026, 1, 1)),
    ("GGRP", Z_GLIMPSE, date(2026, 3, 2), date(2026, 8, 3)),
    ("BTLN", Z_GLIMPSE, date(2026, 9, 8), None),
]
Z_RENAMES: list[Rename] = [
    ("VRAR", "GGRP", Z_VRAR_TO_GGRP, "37892C106", "37892C106"),
    ("GGRP", "BTLN", Z_GGRP_TO_BTLN, "37892C106", "37892C106"),
]


def test_z_old_holder_piece_that_copies_the_latest_holders_chain_is_dropped() -> None:
    # Z(a), BTLN shape. The old BTLN holder (CIK 1425289, listed 2021-06..2022-01) has 1,289
    # bars that are exact copies of VRAR (1,163) and GGRP (126), the latest holder's chain.
    # Today (real master): the old piece is kept (10% identical to GGRP alone) and reported as
    # reuse_looks_continuous. The piece goes; BTLN's bars before the takeover own nothing; the
    # same bars under VRAR / GGRP are the chain's own.
    vrar = bar_rows("VRAR", date(2021, 7, 1), date(2026, 2, 18), 17.0)
    ggrp = bar_rows("GGRP", Z_VRAR_TO_GGRP, date(2026, 8, 19), 0.93)
    master, conflicts = build(
        vrar
        + ggrp
        + relabel(vrar, "BTLN")
        + relabel(ggrp, "BTLN")
        + bar_rows("BTLN", Z_GGRP_TO_BTLN, date(2026, 10, 16), 0.93),
        Z_SNAPSHOTS,
        Z_LIVES,
        Z_RENAMES,
    )

    btln = one(master, "BTLN")
    glimpse = only_sid(master, "VRAR")
    assert span(btln) == (Z_GGRP_TO_BTLN, None)
    assert btln["security_id"] == glimpse == only_sid(master, "GGRP")
    assert btln["cik"] == Z_GLIMPSE
    assert covering(master, "BTLN", date(2023, 6, 1)) == []
    assert segment_at(master, "VRAR", date(2023, 6, 1))["security_id"] == glimpse
    assert covering(master, "GGRP", date(2026, 5, 1))[0]["security_id"] == glimpse
    assert conflicts_of(conflicts, "reuse_looks_continuous") == []
    assert_no_overlap_within_a_chain(master)


Z_RSM_VACATED = date(2021, 3, 1)
Z_RSM_RESUMED = date(2021, 6, 1)


def z_resumed_world(copied: int, resumed_days: int = 100) -> tuple[pa.Table, pa.Table]:
    """RSM vacated for RSN (record), its bars resume after a gap: ``copied`` sessions identical
    to the unrelated UNR, then own bars to the end of the ``resumed_days``. UNR trades on."""
    sessions = business_days(Z_RSM_RESUMED, date(2022, 12, 30))[:resumed_days]
    unr = bar_rows("UNR", STORE_START, date(2022, 12, 30), 3.0)
    resumed = [
        *relabel(between(unr, sessions[0], sessions[copied - 1]), "RSM"),
        *bar_rows("RSM", sessions[copied], sessions[-1], 50.0),
    ]
    return build(
        bar_rows("RSM", STORE_START, date(2021, 2, 26), 10.0)
        + bar_rows("RSN", Z_RSM_VACATED, date(2022, 12, 30), 10.1)
        + resumed
        + unr,
        monthly(date(2020, 1, 1), date(2022, 12, 31)),
        [
            ("RSM", 7701, date(2020, 1, 1), date(2021, 2, 1)),
            ("RSN", 7701, date(2021, 3, 1), None),
            ("UNR", 7702, date(2020, 1, 1), None),
        ],
        [("RSM", "RSN", Z_RSM_VACATED, "000600100", "000600200")],
    )


@pytest.mark.parametrize(
    ("copied", "dropped"), [(89, False), (90, True)], ids=["89_percent", "90_percent"]
)
def test_z_resumed_piece_that_copies_a_stranger_is_dropped_at_ninety_percent(
    copied: int, dropped: bool
) -> None:
    # Z(b), the VJET-type ghost: rule O opens a bars_only piece for RSM's resumed bars; when 90
    # of its 100 traded days are UNR's bars the piece is UNR's relabelled history and goes; at
    # 89 it stays (boundary). UNR is never touched. No E edge either way: RSM's own bars end
    # 60 sessions before the copies (gap) and its last bar is its own.
    master, _ = z_resumed_world(copied)

    unr = one(master, "UNR")
    assert span(unr) == (STORE_START, None)
    first = segments(master, "RSM")[0]
    assert span(first) == (STORE_START, Z_RSM_VACATED)
    assert first["security_id"] == only_sid(master, "RSN")
    assert bars_evidence(master) == []
    if dropped:
        assert len(segments(master, "RSM")) == 1
        assert covering(master, "RSM", date(2021, 8, 2)) == []
    else:
        assert len(segments(master, "RSM")) == 2
        resumed = segments(master, "RSM")[1]
        assert span(resumed) == (Z_RSM_RESUMED, None)
        assert resumed["evidence"] == "bars_only"
        assert resumed["security_id"] not in {first["security_id"], unr["security_id"]}


# =============================================================================================
# G: every conflict carries a resolution
# =============================================================================================

G_SWITCH = date(2020, 6, 1)  # the snapshot that shows the new CIK; also the first bar at it
G_SNAPSHOTS = [date(2020, 1, 13), date(2020, 5, 1), G_SWITCH, date(2021, 1, 4)]
G_VIAX_RECORD = date(2020, 2, 13)  # real: CBS -> VIAC
G_LIVES: list[Life] = [
    ("DSW", 9101, date(2020, 1, 13), date(2020, 5, 1)),
    ("DSW", 9102, G_SWITCH, None),
    ("BRG", 9201, date(2020, 1, 13), date(2020, 5, 1)),
    ("BRG", 9202, G_SWITCH, None),
    ("VIAX", 9301, date(2020, 5, 1), None),
    ("LBL", 9401, date(2020, 1, 13), date(2020, 5, 1)),  # the old holder
    ("TAKR", 9402, date(2020, 1, 13), date(2020, 5, 1)),
    ("LBL", 9402, G_SWITCH, None),  # the taker, seen by the SEC
]


def g_bars() -> list[dict[str, Any]]:
    dupa = bar_rows("DUPA", STORE_START, date(2020, 12, 31), 5.0)
    dupb = bar_rows("DUPB", STORE_START, date(2020, 12, 31), 6.0)
    identical = {r["session_date"] for r in between(dupa, date(2020, 3, 2), date(2020, 3, 6))}
    dupb = [
        {**next(a for a in dupa if a["session_date"] == r["session_date"]), "symbol": "DUPB"}
        if r["session_date"] in identical
        else r
        for r in dupb
    ]
    before = G_SWITCH - timedelta(days=1)
    return [
        *bar_rows("DSW", STORE_START, before, 10.0),
        *bar_rows("DSW", G_SWITCH, date(2020, 12, 31), 30.0),  # 3x: not bridged
        # BRG at its own level: at DSW's 10.0 the two would be identical bars (a dup_unlinked
        # pair this test does not want)
        *bar_rows("BRG", STORE_START, before, 12.0),
        *bar_rows("BRG", G_SWITCH, date(2020, 12, 31), 12.06),  # continuous: bridged
        *bar_rows("VIAX", STORE_START, date(2020, 12, 31), 42.0),  # CBSX never traded here
        *bar_rows("LBL", STORE_START, before, 20.0),
        *bar_rows("TAKR", STORE_START, before, 50.0),
        *bar_rows("LBL", G_SWITCH, date(2020, 12, 31), 20.1),  # continuous with the old holder
        *dupa,
        *dupb,
    ]


@pytest.fixture(scope="module")
def g_built() -> tuple[pa.Table, pa.Table]:
    return build(
        g_bars(),
        G_SNAPSHOTS,
        G_LIVES,
        [
            ("CBSX", "VIAX", G_VIAX_RECORD, "124857202", "92556H206"),
            ("TAKR", "LBL", G_SWITCH, "000700100", "000700100"),
        ],
    )


def test_g_every_conflict_kind_carries_its_resolution(g_built: tuple[pa.Table, pa.Table]) -> None:
    # G. One row per value: a discontinuous CIK switch is a new security by construction; a
    # bridged one is now written down (new kind); a record V dropped is written down (new kind);
    # a continuous reuse whose taker traded its own bars in the span is a new security (6-c rule
    # R); an unlinked duplicate stays for a human.
    master, conflicts = g_built

    assert conflicts.schema.equals(sm.CONFLICT_SCHEMA)
    by_kind = {k: conflicts_of(conflicts, k) for k in set(conflicts.column("kind").to_pylist())}
    assert {k: len(v) for k, v in by_kind.items()} == {
        "cik_switch_discontinuous": 1,
        "cik_switch_bridged": 1,
        "record_without_taker": 1,
        "reuse_looks_continuous": 1,
        "dup_unlinked": 1,
    }
    assert {k: v[0]["resolution"] for k, v in by_kind.items()} == {
        "cik_switch_discontinuous": "new_security",
        "cik_switch_bridged": "bridged",
        "record_without_taker": "ignored",
        "reuse_looks_continuous": "new_security",  # 6-c rule R: TAKR traded its own bars
        "dup_unlinked": "unresolved",
    }
    assert set(conflicts.column("resolution").to_pylist()) <= set(sm.RESOLUTIONS)

    [discontinuous] = by_kind["cik_switch_discontinuous"]
    assert discontinuous["symbols"] == ["DSW"]
    assert len(sids(master, "DSW")) == 2
    [bridged] = by_kind["cik_switch_bridged"]
    assert bridged["symbols"] == ["BRG"]
    assert bridged["dates"] == [G_SWITCH - timedelta(days=3), G_SWITCH, G_SWITCH]  # Fri, Mon, Mon
    assert len(sids(master, "BRG")) == 1 and len(segments(master, "BRG")) == 2
    [ignored] = by_kind["record_without_taker"]
    assert (ignored["symbols"], ignored["dates"]) == (["CBSX", "VIAX"], [G_VIAX_RECORD])
    assert span(one(master, "VIAX")) == (STORE_START, None)  # rule V, unchanged
    [reuse] = by_kind["reuse_looks_continuous"]
    assert reuse["symbols"] == ["TAKR", "LBL"]
    assert len(sids(master, "LBL")) == 2
    [dup] = by_kind["dup_unlinked"]
    assert set(dup["symbols"]) == {"DUPA", "DUPB"}
    assert bars_evidence(master) == []  # five identical days, both names trade on: no E


# =============================================================================================
# the build reads the imported bars like the read path does
# =============================================================================================

I_RECORD = date(2024, 7, 1)
I_DAYS = business_days(date(2024, 1, 2), date(2024, 6, 28))
I_SNAPSHOTS = [
    date(2024, 1, 5),
    date(2024, 3, 1),
    date(2024, 6, 6),
    date(2024, 8, 1),
    date(2024, 10, 1),
]


def i_write(root: Path, *, with_ext: bool) -> Path:
    """Alpaca: TAKR's own bars Jan-Jun 2024 and the same bars filed under IMP (Alpaca's
    relabelling), then IMP's own from the record. Ext: IMP's real Jan-Jun history (other
    prices), the recorded decision that Alpaca's IMP rows for those days are wrong."""
    takr = [
        daily(
            "TAKR",
            d,
            open=9.9 + i / 1000,
            high=10.2 + i / 1000,
            low=9.8,
            close=10.0 + i / 1000,
            volume=5_000 + i,
        )
        for i, d in enumerate(I_DAYS)
    ]
    copies = [
        daily("IMP", d, open=b.open, high=b.high, low=b.low, close=b.close, volume=b.volume)
        for d, b in zip(I_DAYS, takr, strict=True)
    ]
    own = [
        daily("IMP", d, open=10.1, high=10.4, low=10.0, close=10.2 + i / 1000, volume=6_000 + i)
        for i, d in enumerate(business_days(I_RECORD, date(2024, 9, 30)))
    ]
    store_daily(root, [*takr, *copies, *own])
    if with_ext:
        write_ext(
            root,
            [
                ext_row("IMP", d, close=3.0 + i / 1000, volume=2_000 + i)
                for i, d in enumerate(I_DAYS)
            ],
        )
    sec = pa.Table.from_pylist(
        [
            {"snapshot_date": s, "ticker": t, "cik": c, "name": f"{t} Inc"}
            for s in I_SNAPSHOTS
            for t, c, lo, hi in (
                ("TAKR", 9501, I_SNAPSHOTS[0], date(2024, 6, 6)),
                ("IMP", 9502, I_SNAPSHOTS[0], date(2024, 6, 6)),
                ("IMP", 9501, date(2024, 8, 1), None),
            )
            if lo <= s and (hi is None or s <= hi)
        ],
        schema=HISTORY_SCHEMA,
    )
    (root / "symbols").mkdir(parents=True, exist_ok=True)
    pq.write_table(sec, root / "symbols" / "sec_history.parquet")
    (root / "corporate_actions").mkdir(parents=True, exist_ok=True)
    pq.write_table(
        name_changes_table([("TAKR", "IMP", I_RECORD, "000800100", "000800200")]).cast(
            NAME_CHANGE_SCHEMA
        ),
        root / "corporate_actions" / "name_changes.parquet",
    )
    return root


def test_build_from_store_reads_imports_over_alpaca_rows_of_the_same_key(tmp_path: Path) -> None:
    # Build. Reading Alpaca alone, IMP's pre-record piece is 100% copies of TAKR and is dropped
    # as relabelled; with the ext rows replacing those keys (as bars_source_sql does for the
    # read path) the piece is IMP's own history and stays. Today build_from_store reads only
    # bars_1d/** and never sees the import.
    alpaca_only = i_write(market_root(tmp_path / "a"), with_ext=False)
    with_import = i_write(market_root(tmp_path / "b"), with_ext=True)

    master_a, _ = sm.build_from_store(alpaca_only)
    master_b, _ = sm.build_from_store(with_import)

    assert [span(r) for r in segments(master_a, "IMP")] == [(I_RECORD, None)]
    assert [span(r) for r in segments(master_b, "IMP")] == [(I_DAYS[0], I_RECORD), (I_RECORD, None)]
    old_holder, renamed = segments(master_b, "IMP")
    assert old_holder["security_id"] != renamed["security_id"]
    assert renamed["security_id"] == only_sid(master_b, "TAKR")
    assert old_holder["cik"] == 9502
