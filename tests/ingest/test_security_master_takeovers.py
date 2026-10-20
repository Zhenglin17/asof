"""Guards against a label that changes hands at a continuous price being given to the wrong
company: the leaver's CIK must not follow its old label to the next holder (rule L), and a
previous holder's piece that is only the taker's own history filed under the label must join
the taker instead of standing as a separate security with someone else's CIK (rule R).

  L  CIK leaves with a rename. Label L is vacated by a rename on day d (the piece ending at d
     has an ``end`` edge, XPER -> ADEA on 2022-10-03) and a new piece of L starts on d (the
     when-issued record XPER.WI -> XPER, links=False). SEC rows of the leaver's CIK are not
     used for the new piece's CIK, even when a snapshot dated d (or later, by SEC lag) still
     lists L under it. Today the new XPER piece reads 1803696 off the 2022-10-03 snapshot,
     shares Adeia's CIK, and the same-CIK takeover rule bridges Xperi Inc into Adeia.
  R  Takeover of a label still in use, previous holder's piece P NOT ended by a rename,
     continuous prices, different CIKs (today: ``reuse_looks_continuous``, ``unresolved``).
     Decided after the copy rules C / F / Z. If no piece of the taker's chain (minus dropped
     pieces) has a traded bar inside [P.valid_from, cut) that differs (OHLCV) from P's bar of
     that day, P is the taker's relabelled history: P joins the taker's security, takes the
     taker piece's CIK, evidence ``relabelled``, ``available_at`` = the taker piece's
     ``available_at``; taker-chain pieces overlapping [P.valid_from, cut) end at P.valid_from
     (dropped when that leaves nothing), their ``end_available_at`` = the taker piece's
     ``available_at``; the conflict row stays with resolution ``relabelled`` (a new member of
     ``RESOLUTIONS``) and a detail naming the lost holder's CIK. Otherwise two securities and
     resolution ``new_security``.

Proposed API: ``sm.RESOLUTIONS`` gains ``"relabelled"``; master ``evidence`` gains
``"relabelled"``; ``build_security_master``'s contract is unchanged.

Fake data provenance (store probe 2026-10-19 unless noted; bars, SEC rows and Alpaca records
read off /data/asof/market, master rows quoted are the 2026-10-17 build):
- XPER / ADEA (real): Xperi Holding (CIK 1803696) renamed itself Adeia; the spun-off Xperi Inc
  (CIK 1788999) took XPER. SEC: XPER under 1803696 2020-07-10 .. 2022-10-03 (inclusive),
  XPER.WI under 1788999 on 2022-10-03 only, ADEA under 1803696 and XPER under 1788999 from
  2022-11-05. Records XPER.WI -> XPER and ADEAV -> ADEA, both 2022-10-03, CUSIPs real. XPER
  trades through: 14.14 on 09-30, 14.97 on 10-03 (Xperi Inc); ADEA 7.19 on 10-03; ADEA's 188
  bars 2022-01-03 .. 09-30 are OHLCV copies of XPER's (real). XPER.WI: 9 bars 09-20 .. 09-30,
  2 traded (23.0 / 26.0 real, the dates of the traded pair ASSUMED 09-20 and 09-30); ADEAV one
  bar 09-30 at 16.0 (real). Levels after 10-03 drift slowly (ASSUMPTION).
- ARNC (real mechanism, F2/F4 of test_security_master.py): Arconic Inc (4281) renamed itself
  Howmet 2020-04-01 (HWM.WI -> HWM, ARNC.WI -> ARNC, both that day, CUSIPs real); the SEC still
  lists ARNC under 4281 on 2020-06-01 and switches it to Arconic Corp (1790982) on 07-10. The
  3x level change across 04-01 is an ASSUMPTION that keeps continuity out of the case.
- GOLD / B (real): Barrick (756894) GOLD -> B 2025-05-09 (CUSIPs real); B was Barnes Group
  (9984, SEC 2019-10..2025-01-11), and Alpaca files Barrick's whole history under B (B 18.86 on
  05-08 -> 19.46 on 05-09, one series). A-Mark (1591588) AMRK -> GOLD 2025-12-02; the GOLD bars
  2020-01-02 .. 2025-12-01 are OHLCV copies of AMRK's (1,487 days). SEC: GOLD under 756894 to
  2025-04-29, B under 756894 from 2025-06-21, GOLD under 1591588 from 2025-12-06. SIMPLIFIED:
  the real file also lists the stale ABX under 756894 throughout; left out. Levels drift
  (ASSUMPTION; real B ~17 in 2023, AMRK ~24-34).
- INXB / INBX (real, values exact): Inhibrx (1739614) was bought; Inhibrx Biosciences
  (2007919) traded INXBV 05-29, INXB 05-30, INBX from 05-31 (records INXBV -> INXB 05-30,
  INXB -> INBX 05-31, CUSIPs real). The INBX label's own bars start 2024-05-29; its 05-29 bar
  is identical to INXBV's, its 05-30 bar to INXB's. INBXV (old CIK's when-issued line) 05-28 /
  05-29 at ~31, and 1739614 lists INBXV on 2024-06-06 (real). SEC: INBX under 1739614 to
  2024-05-02, INXBV / INXB / INBX under 2007919 on 06-06, INBX alone after. The real records
  INBX -> INBX of 06-04 / 06-05 (CUSIP-only) are left out. Bars after 06-03 drift (ASSUMPTION).
- FGMC / ICCT (real): FG Merger Corp (1906133) merged with iCoreConnect; ICCT trades from
  2023-08-28 (closes and volumes 08-28 .. 10-05 real), FGMC's last traded bar is 08-25 and
  volume-0 filler at 4.82 follows (real). SEC: FGMC under 1906133 to 2023-09-02, ICCT under
  1408057 (old iCoreConnect) to 09-02 and under 1906133 from 10-03. SIMPLIFIED: 1408057's
  later ICNP / ICCRW rows, FGMCU / FGMCW, and FGMC's 2025 reuse are left out. FGMC levels
  before 08-25 ASSUMED (a SPAC near 10).
- DISCA / WBD (real): WBDWV -> WBD 2022-04-11 (CUSIP real); WBD's bars 04-04 .. 04-08 are
  identical to WBDWV's and its own from 04-11 (24.47 -> 24.78), DISCA traded its own bars on
  the same days (25.80, 24.42, 23.92, 23.07, 24.43). SEC: a stale WBD row under 1166718
  2019-10-02 .. 2020-06-01, DISCA under 1437107 to 2022-04-01, WBD under 1437107 from 05-03.
  DISCK / DISCB under 1437107 left out. DISCA levels before 04-04 ASSUMED.
- BBU / BBUC (real): Brookfield Business Corp BBUC (1871130, BBUC.WI -> BBUC 2022-03-15,
  CUSIP real) and Brookfield Business Partners BBU (1654795) traded side by side until BBU's
  last bar 2026-03-30; the SEC lists BBUC under 1654795 from 2026-04-21 (BBUC 31.65 -> 31.64
  across 03-31). BBUC.WI's two traded bars and all levels ASSUMED.

Assumptions that need a real-data check (golden case or audit) because the fake worlds
cannot tell whether they are right:
 1. R's "identical" is read as OHLCV here because every copy in these worlds is exact; the
    build's identical-day table compares OHLC only (volume is revised after 20:00). If the
    implementation uses OHLC, a real copy whose volume was revised still counts as identical;
    if OHLCV, it does not -> golden INBX (the INXB 05-30 copy) decides on real data.
 2. The four real shapes are the only ones the rule was designed on; after the rebuild the
    audit's M6 must show no ``reuse_looks_continuous`` left ``unresolved`` in the liquid tier,
    and M8 must list GOLD/B, INXB/INBX, FGMC/ICCT as ``relabelled`` and DISCA/WBD, BBU/BBUC as
    ``new_security`` -> ``asof market audit`` (manual) + golden GOLD/B, INBX.
 3. L on the whole store: every piece started on the day its label was vacated by a rename now
    carries no CIK of the leaver -> golden ADEA/XPER and the ARNC golden case (whose pinned
    "known imprecision" 4281 row this rule removes).
 4. A relabelled P becomes knowable only at the takeover record: readers before it see no
    segment for the label's past (Barnes's B, old ICCT, old INBX vanish from history). Accepted
    by the spec; the loss metric (I1) counts the bars a reader before the record cannot place.
"""

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pytest

from asof.ingest import security_master as sm
from asof.ingest.security_master import build_security_master, write_master
from asof.store.market import market_root, visible_master
from tests.ingest.test_security_master import (
    Life,
    Rename,
    assert_end_knowability_invariant,
    assert_nothing_known_before_its_valid_from,
    bar_available_at,
    bar_rows,
    bars_table,
    conflicts_of,
    et_midnight,
    filler_rows,
    name_changes_table,
    relabel,
    sec_table,
    segment_at,
    segments,
    unresolved,
    visible_at,
)

MICRO = timedelta(microseconds=1)
Built = tuple[pa.Table, pa.Table]


def ohlcv(
    symbol: str, day: date, o: float, h: float, low: float, c: float, v: int
) -> dict[str, Any]:
    """One bar with exact values (the real ones where the provenance says so)."""
    return {
        "symbol": symbol,
        "session_date": day,
        "open": o,
        "high": h,
        "low": low,
        "close": c,
        "volume": v,
        "available_at": bar_available_at(day),
    }


def closes(symbol: str, rows: list[tuple[date, float, int]]) -> list[dict[str, Any]]:
    """Bars from (day, close, volume) with OHLC derived from the close like ``bar_rows``."""
    return [
        ohlcv(symbol, d, round(c * 0.995, 6), round(c * 1.01, 6), round(c * 0.99, 6), c, v)
        for d, c, v in rows
    ]


def touching(conflicts: pa.Table, symbol: str, day: date) -> list[dict[str, Any]]:
    return [r for r in conflicts.to_pylist() if symbol in r["symbols"] and day in r["dates"]]


def overlaps_within_securities(master: pa.Table) -> list[tuple[dict, dict]]:
    """Pairs of segments of one security that overlap in time (any symbols)."""
    rows = master.to_pylist()
    out = []
    for i, a in enumerate(rows):
        for b in rows[i + 1 :]:
            if a["security_id"] != b["security_id"]:
                continue
            a_end, b_end = a["valid_to"] or date.max, b["valid_to"] or date.max
            if a["valid_from"] < b_end and b["valid_from"] < a_end:
                out.append((a, b))
    return out


def test_relabelled_is_a_resolution() -> None:
    assert "relabelled" in sm.RESOLUTIONS
    assert {"new_security", "bridged", "ignored", "unresolved"} <= set(sm.RESOLUTIONS)


# =============================================================================================
# L: the leaver's CIK does not follow its old label
# =============================================================================================

XPER_CUT = date(2022, 10, 3)
XPER_SNAPSHOTS = [
    date(2020, 7, 10),
    date(2021, 1, 4),
    date(2022, 1, 3),
    date(2022, 8, 1),
    date(2022, 9, 4),
    XPER_CUT,  # real: still lists XPER under Adeia's 1803696
    date(2022, 11, 5),
    date(2022, 12, 4),
]
XPER_ADEIA, XPER_XPERI_INC = 1803696, 1788999
XPER_LIVES: list[Life] = [
    ("XPER", XPER_ADEIA, date(2020, 7, 10), XPER_CUT),
    ("XPER.WI", XPER_XPERI_INC, XPER_CUT, XPER_CUT),
    ("ADEA", XPER_ADEIA, date(2022, 11, 5), None),
    ("XPER", XPER_XPERI_INC, date(2022, 11, 5), None),
]
XPER_RENAMES: list[Rename] = [
    ("XPER.WI", "XPER", XPER_CUT, "98423J101", "98423J101"),
    ("ADEAV", "ADEA", XPER_CUT, "00676P107", "00676P107"),
]


def xper_bars() -> list[dict[str, Any]]:
    xper = bar_rows("XPER", date(2020, 1, 2), date(2022, 12, 30), 13.5)  # one series: continuous
    copies = relabel([r for r in xper if date(2022, 1, 3) <= r["session_date"] < XPER_CUT], "ADEA")
    wi_first = ohlcv("XPER.WI", date(2022, 9, 20), 23.0, 23.0, 23.0, 23.0, 5_000)
    wi = [
        wi_first,
        *filler_rows(wi_first, date(2022, 9, 21), date(2022, 9, 29)),
        ohlcv("XPER.WI", date(2022, 9, 30), 25.5, 26.5, 25.0, 26.0, 246_390),
    ]
    return [
        *xper,
        *copies,
        *bar_rows("ADEA", XPER_CUT, date(2022, 12, 30), 7.19),
        *wi,
        ohlcv("ADEAV", date(2022, 9, 30), 16.0, 16.2, 15.8, 16.0, 35_372),
    ]


@pytest.fixture(scope="module")
def xper_built() -> Built:
    return build_security_master(
        bars=bars_table(xper_bars()),
        sec_history=sec_table(XPER_SNAPSHOTS, XPER_LIVES),
        name_changes=name_changes_table(XPER_RENAMES),
    )


def test_l_xper_world_is_shaped_like_the_store() -> None:
    # fixture sanity: the closes are continuous across the cut on XPER and jump on ADEA
    rows = {(r["symbol"], r["session_date"]): r["close"] for r in xper_bars()}
    before, after = rows[("XPER", date(2022, 9, 30))], rows[("XPER", XPER_CUT)]
    assert sm.CONTINUOUS[0] <= after / before <= sm.CONTINUOUS[1]
    assert rows[("ADEA", XPER_CUT)] / before < sm.CONTINUOUS[0]
    assert rows[("ADEA", date(2022, 9, 30))] == before  # the relabelled copy


def test_l_xper_after_the_rename_is_a_different_security_from_adeia(xper_built: Built) -> None:
    # L. Today: XPER [2022-10-03, 11-05) reads 1803696 off the 10-03 snapshot, so the takeover
    # rule sees one CIK on both sides and bridges Xperi Inc into Adeia (master S008468 holds
    # XPER 2020.., ADEA, and XPER 2022-10-03.. alike).
    master, _ = xper_built

    old = segment_at(master, "XPER", date(2022, 9, 30))
    adea = segment_at(master, "ADEA", date(2022, 10, 4))
    assert old["valid_to"] == XPER_CUT
    assert old["security_id"] == adea["security_id"]
    assert old["cik"] == XPER_ADEIA
    for day in (XPER_CUT, date(2022, 10, 4), date(2022, 11, 7), date(2022, 12, 1)):
        assert segment_at(master, "XPER", day)["security_id"] != old["security_id"], day


def test_l_new_xper_piece_does_not_carry_the_leavers_cik(xper_built: Built) -> None:
    master, _ = xper_built

    new = segment_at(master, "XPER", XPER_CUT)
    assert new["valid_from"] == XPER_CUT
    assert new["cik"] != XPER_ADEIA
    # Xperi Inc stays one security across the SEC's 2022-11-05 switch to its own CIK (guard:
    # today's bridge across that cut must survive the fix)
    later = segment_at(master, "XPER", date(2022, 12, 1))
    assert later["cik"] == XPER_XPERI_INC
    assert later["security_id"] == new["security_id"]


def test_l_xper_takeover_is_not_left_for_a_human(xper_built: Built) -> None:
    # A row about the 10-03 hand-over may be emitted, but the build has decided it: never
    # "unresolved" (the previous holder left by renaming; the label's next holder is another
    # filer however close the prices).
    _, conflicts = xper_built

    assert not [r for r in unresolved(conflicts) if "XPER" in r["symbols"]]
    for row in touching(conflicts, "XPER", XPER_CUT):
        assert row["resolution"] in sm.RESOLUTIONS
        assert row["resolution"] != "unresolved"


def test_l_xper_nothing_is_known_early(xper_built: Built) -> None:
    # Leakage: a reader the day before the rename sees old XPER open and nothing that starts
    # on 10-03; what starts on 10-03 is known from its first bar, 20:00 New York of 10-03.
    master, _ = xper_built

    eve = [r for r in visible_at(master, et_midnight(XPER_CUT) - MICRO) if r["symbol"] == "XPER"]
    assert [r["valid_from"] for r in eve] == [date(2020, 1, 2)]
    # first-bar rule: pieces starting 10-03 wait for their first bar
    on_day = {r["valid_from"] for r in visible_at(master, bar_available_at(XPER_CUT))}
    assert XPER_CUT in on_day
    assert_nothing_known_before_its_valid_from(master)
    assert_end_knowability_invariant(master)


ARNC_DAY = date(2020, 4, 1)
ARNC_SNAPSHOTS = [date(2020, 1, 13), date(2020, 6, 1), date(2020, 7, 10), date(2021, 1, 4)]
ARNC_LIVES: list[Life] = [
    ("ARNC", 4281, date(2020, 1, 13), date(2020, 6, 1)),  # real: one snapshot of SEC lag
    ("HWM", 4281, date(2020, 7, 10), None),
    ("ARNC", 1790982, date(2020, 7, 10), None),
]
ARNC_RENAMES: list[Rename] = [
    ("ARNC.WI", "ARNC", ARNC_DAY, "03966V107", "03966V107"),
    ("HWM.WI", "HWM", ARNC_DAY, "443201108", "443201108"),
]


@pytest.fixture(scope="module")
def arnc_built() -> Built:
    return build_security_master(
        bars=bars_table(
            bar_rows("ARNC", date(2020, 1, 2), date(2020, 3, 31), 25.0)
            + bar_rows("ARNC", ARNC_DAY, date(2020, 12, 31), 75.0)
            + bar_rows("HWM", ARNC_DAY, date(2020, 12, 31), 18.0)
        ),
        sec_history=sec_table(ARNC_SNAPSHOTS, ARNC_LIVES),
        name_changes=name_changes_table(ARNC_RENAMES),
    )


def test_l_arnc_snapshot_lag_does_not_give_arconic_corp_the_old_cik(arnc_built: Built) -> None:
    # L, later-snapshot variant. Today: ARNC [2020-04-01, 07-10) cik 4281 (the master row the
    # golden case pins as a known imprecision) -- the 06-01 snapshot is SEC lag, not Arconic
    # Corp's filing.
    master, _ = arnc_built

    inc = segment_at(master, "ARNC", date(2020, 3, 2))
    corp = segment_at(master, "ARNC", date(2020, 5, 1))
    assert inc["cik"] == 4281 and inc["valid_to"] == ARNC_DAY
    assert inc["security_id"] == segment_at(master, "HWM", date(2020, 5, 1))["security_id"]
    assert corp["valid_from"] == ARNC_DAY
    assert corp["cik"] != 4281
    assert segment_at(master, "ARNC", date(2020, 9, 1))["cik"] == 1790982
    assert_nothing_known_before_its_valid_from(master)


# =============================================================================================
# R: a takeover of a label whose earlier piece is the taker's own history
# =============================================================================================

# --- GOLD / B: Barrick's history under Barnes's old label --------------------------------------

GB_CUT = date(2025, 5, 9)
GB_AMRK_DAY = date(2025, 12, 2)
GB_BARNES, GB_BARRICK, GB_AMARK = 9984, 756894, 1591588
GB_SNAPSHOTS = [
    date(2019, 10, 2),
    date(2020, 7, 10),
    date(2021, 1, 4),
    date(2022, 1, 3),
    date(2023, 1, 3),
    date(2024, 1, 2),
    date(2025, 1, 11),  # real: last snapshot listing B under Barnes
    date(2025, 4, 29),  # real: last snapshot listing GOLD under Barrick
    date(2025, 6, 21),  # real: first listing B under Barrick
    date(2025, 12, 6),
    date(2026, 1, 5),
]
GB_LIVES: list[Life] = [
    ("B", GB_BARNES, date(2019, 10, 2), date(2025, 1, 11)),
    ("GOLD", GB_BARRICK, date(2019, 10, 2), date(2025, 4, 29)),
    ("B", GB_BARRICK, date(2025, 6, 21), None),
    ("AMRK", GB_AMARK, date(2019, 10, 2), date(2025, 12, 6)),
    ("GOLD", GB_AMARK, date(2025, 12, 6), None),
]
GB_RENAMES: list[Rename] = [
    ("GOLD", "B", GB_CUT, "067901108", "06849F108"),
    ("AMRK", "GOLD", GB_AMRK_DAY, "00181T107", "00181T107"),
]


def gb_bars() -> list[dict[str, Any]]:
    amrk = bar_rows("AMRK", date(2020, 1, 2), GB_AMRK_DAY - timedelta(days=1), 24.0)
    return [
        *bar_rows("B", date(2020, 1, 2), date(2026, 6, 30), 17.0),  # Barrick's, one series
        *amrk,
        *relabel(amrk, "GOLD"),  # Alpaca rewrote GOLD with A-Mark's history
        *bar_rows("GOLD", GB_AMRK_DAY, date(2026, 6, 30), 29.8),
    ]


@pytest.fixture(scope="module")
def gb_built() -> Built:
    return build_security_master(
        bars=bars_table(gb_bars()),
        sec_history=sec_table(GB_SNAPSHOTS, GB_LIVES),
        name_changes=name_changes_table(GB_RENAMES),
    )


def test_r_gold_b_barricks_history_under_b_joins_barrick(gb_built: Built) -> None:
    # R. Today (real master): B [2020-01-02, 2025-05-09) S000702 cik 9984 'sec' and B
    # [2025-05-09, None) S016469 cik 756894, conflict reuse_looks_continuous GOLD/B unresolved.
    # The taker's chain minus the dropped GOLD piece (A-Mark's copies, rule Z / relabel) has
    # no bar before the cut, so nothing contradicts "B's past is Barrick's".
    master, _ = gb_built

    past = segment_at(master, "B", date(2023, 6, 1))
    now = segment_at(master, "B", date(2025, 6, 2))
    assert (past["valid_from"], past["valid_to"]) == (date(2020, 1, 2), GB_CUT)
    assert now["valid_from"] == GB_CUT
    assert past["security_id"] == now["security_id"]
    assert past["cik"] == now["cik"] == GB_BARRICK
    assert past["evidence"] == "relabelled"
    # first-bar rule: the relabelled piece is judged on the taker's bars after the cut, so both
    # are knowable at the taker's first bar
    assert past["available_at"] == past["end_available_at"] == bar_available_at(GB_CUT)
    assert now["available_at"] == bar_available_at(GB_CUT)
    # A-Mark's GOLD from 2025-12-02 is A-Mark, not Barrick
    gold = segment_at(master, "GOLD", date(2026, 1, 5))
    assert gold["security_id"] == segment_at(master, "AMRK", date(2023, 6, 1))["security_id"]
    assert gold["security_id"] != now["security_id"]
    assert overlaps_within_securities(master) == []


def test_r_gold_b_conflict_is_kept_as_relabelled_naming_barnes(gb_built: Built) -> None:
    _, conflicts = gb_built

    rows = touching(conflicts, "B", GB_CUT)
    assert len(rows) == 1
    [row] = rows
    assert row["kind"] == "reuse_looks_continuous"
    assert row["resolution"] == "relabelled"
    assert str(GB_BARNES) in row["detail"]
    assert not [r for r in unresolved(conflicts) if "B" in r["symbols"]]


def test_r_gold_b_relabelled_piece_is_unknown_before_the_takeover_record(
    gb_built: Built, tmp_path: Path
) -> None:
    # Leakage, on the read path strategies use. The B bars of 2020..2025-05 sit on disk all
    # along, but that they are Barrick's is only known from the GOLD -> B record: a reader on
    # 2023-06-02 must see no segment placing them, and none with Barrick's CIK; from 00:00 New
    # York of 2025-05-09 nothing yet; with Barrick's first bar under B that evening the piece is
    # there with its original valid_from.
    master, conflicts = gb_built
    root = market_root(tmp_path)
    write_master(master, conflicts, root)
    barrick = segment_at(master, "B", date(2025, 6, 2))["security_id"]
    con = duckdb.connect()
    try:
        early = visible_master(con, root, datetime(2023, 6, 2, 12, tzinfo=UTC)).to_pylist()
        assert not [r for r in early if r["security_id"] == barrick]
        assert not [r for r in early if r["symbol"] == "B" and r["cik"] == GB_BARRICK]
        eve = visible_master(con, root, et_midnight(GB_CUT) - MICRO).to_pylist()
        assert not [r for r in eve if r["symbol"] == "B"]
        at_record = visible_master(con, root, et_midnight(GB_CUT)).to_pylist()
        # first-bar rule: the relabelling waits for the taker's first bar
        assert not [r for r in at_record if r["symbol"] == "B"]
        known = visible_master(con, root, bar_available_at(GB_CUT)).to_pylist()
        assert [(r["valid_from"], r["valid_to"]) for r in known if r["symbol"] == "B"] == [
            (date(2020, 1, 2), GB_CUT),
            (GB_CUT, None),
        ]
    finally:
        con.close()
    assert_nothing_known_before_its_valid_from(master)
    assert_end_knowability_invariant(master)


# --- INXB / INBX: the taker's only traded bar in the span is a copy ---------------------------

IN_RECORD_INXB, IN_CUT = date(2024, 5, 30), date(2024, 5, 31)
IN_OLD, IN_NEW = 1739614, 2007919
IN_SNAPSHOTS = [
    date(2023, 1, 3),
    date(2024, 4, 1),
    date(2024, 5, 2),
    date(2024, 6, 6),
    date(2024, 8, 2),
]
IN_LIVES: list[Life] = [
    ("INBX", IN_OLD, date(2023, 1, 3), date(2024, 5, 2)),
    ("INBXV", IN_OLD, date(2024, 6, 6), date(2024, 6, 6)),
    ("INXBV", IN_NEW, date(2024, 6, 6), date(2024, 6, 6)),
    ("INXB", IN_NEW, date(2024, 6, 6), date(2024, 6, 6)),
    ("INBX", IN_NEW, date(2024, 6, 6), None),
]
IN_RENAMES: list[Rename] = [
    ("INXBV", "INXB", IN_RECORD_INXB, "45720N103", "45720N103"),
    ("INXB", "INBX", IN_CUT, "45720N103", "45720N103"),
]


def in_bars(*, inxb_copy: bool = True) -> list[dict[str, Any]]:
    d29, d30 = date(2024, 5, 29), date(2024, 5, 30)
    bar29 = (13.0, 13.95, 12.95, 13.92, 260_650)
    bar30 = (13.85, 15.99, 13.85, 15.86, 725_625)
    # constructed variant: INXB's 05-30 differs from the label's by one cent on the close
    inxb30 = bar30 if inxb_copy else (13.85, 15.99, 13.85, 15.87, 725_625)
    return [
        ohlcv("INBXV", date(2024, 5, 28), 31.01, 31.35, 31.0, 31.25, 7_028),
        ohlcv("INBXV", d29, 31.15, 31.35, 30.78, 30.85, 308_292),
        ohlcv("INXBV", d29, *bar29),
        ohlcv("INBX", d29, *bar29),
        ohlcv("INBX", d30, *bar30),
        ohlcv("INXB", d30, *inxb30),
        ohlcv("INBX", IN_CUT, 15.97, 16.5, 15.25, 16.25, 544_085),
        ohlcv("INBX", date(2024, 6, 3), 16.5, 18.1899, 16.31, 18.02, 541_254),
        *bar_rows("INBX", date(2024, 6, 4), date(2024, 12, 31), 17.2),
    ]


def in_build(*, inxb_copy: bool = True) -> Built:
    return build_security_master(
        bars=bars_table(in_bars(inxb_copy=inxb_copy)),
        sec_history=sec_table(IN_SNAPSHOTS, IN_LIVES),
        name_changes=name_changes_table(IN_RENAMES),
    )


@pytest.fixture(scope="module")
def in_built() -> Built:
    return in_build()


def test_r_inbx_old_label_days_are_inhibrx_biosciences(in_built: Built) -> None:
    # R. Today (real master): INBX [05-29, 05-31) S015203 cik 1739614, INXB [05-30, 05-31) and
    # INBX [05-31, None) S015208 cik 2007919, reuse_looks_continuous INXB/INBX unresolved.
    master, conflicts = in_built

    p = segment_at(master, "INBX", date(2024, 5, 29))
    new = segment_at(master, "INBX", date(2024, 6, 3))
    assert (p["valid_from"], p["valid_to"]) == (date(2024, 5, 29), IN_CUT)
    assert p["security_id"] == new["security_id"]
    assert p["cik"] == new["cik"] == IN_NEW
    assert p["evidence"] == "relabelled"
    # first-bar rule: the relabelled piece is knowable with the taker, at its first bar
    assert p["available_at"] == bar_available_at(IN_CUT)
    assert new["available_at"] == bar_available_at(IN_CUT)
    [row] = touching(conflicts, "INBX", IN_CUT)
    assert row["resolution"] == "relabelled"
    assert str(IN_OLD) in row["detail"]


def test_r_inbx_taker_piece_inside_the_span_is_clipped_away(in_built: Built) -> None:
    # INXB [05-30, 05-31) lies inside [P.valid_from, cut): clipped to end at 05-29 it is empty
    # and goes; no two segments of the security overlap.
    master, _ = in_built

    sid = segment_at(master, "INBX", date(2024, 6, 3))["security_id"]
    assert not [r for r in segments(master, "INXB") if r["security_id"] == sid]
    assert overlaps_within_securities(master) == []
    assert_end_knowability_invariant(master)


def test_r_one_differing_taker_bar_in_the_span_keeps_two_securities() -> None:
    # R boundary (constructed): the same world with INXB's 05-30 close one cent off the
    # label's. The taker traded a bar of its own inside the span, so P is a previous holder.
    master, conflicts = in_build(inxb_copy=False)

    p = segment_at(master, "INBX", date(2024, 5, 29))
    new = segment_at(master, "INBX", date(2024, 6, 3))
    assert p["security_id"] != new["security_id"]
    assert p["cik"] == IN_OLD
    assert p["evidence"] != "relabelled"
    [row] = touching(conflicts, "INBX", IN_CUT)
    assert row["resolution"] == "new_security"


# --- FGMC / ICCT: the taker only has volume-0 filler in the span ------------------------------

FI_CUT = date(2023, 10, 3)
FI_P_START = date(2023, 8, 28)
FI_SPAC, FI_OLD = 1906133, 1408057
FI_SNAPSHOTS = [
    date(2022, 3, 1),
    date(2022, 5, 3),
    date(2023, 1, 3),
    date(2023, 4, 2),
    date(2023, 9, 2),
    FI_CUT,
    date(2023, 11, 2),
    date(2024, 1, 2),
]
FI_LIVES: list[Life] = [
    ("FGMC", FI_SPAC, date(2022, 3, 1), date(2023, 9, 2)),
    ("ICCT", FI_OLD, date(2022, 3, 1), date(2023, 9, 2)),
    ("ICCT", FI_SPAC, FI_CUT, None),
]
# real ICCT closes and volumes, 2023-08-28 .. 10-05
FI_ICCT_REAL = [
    (date(2023, 8, 28), 1.59, 11_072_257),
    (date(2023, 8, 29), 5.64, 44_691_219),
    (date(2023, 8, 30), 12.28, 72_002_213),
    (date(2023, 8, 31), 11.96, 29_769_303),
    (date(2023, 9, 1), 8.74, 5_771_301),
    (date(2023, 9, 5), 6.73, 2_693_512),
    (date(2023, 9, 6), 7.83, 28_260_922),
    (date(2023, 9, 7), 6.5, 1_542_386),
    (date(2023, 9, 8), 6.16, 1_291_133),
    (date(2023, 9, 11), 5.39, 1_027_865),
    (date(2023, 9, 12), 5.57, 736_180),
    (date(2023, 9, 13), 5.83, 659_471),
    (date(2023, 9, 14), 5.62, 364_077),
    (date(2023, 9, 15), 5.16, 173_845),
    (date(2023, 9, 18), 4.53, 404_807),
    (date(2023, 9, 19), 3.85, 417_703),
    (date(2023, 9, 20), 3.93, 547_236),
    (date(2023, 9, 21), 4.06, 614_520),
    (date(2023, 9, 22), 3.99, 218_666),
    (date(2023, 9, 25), 4.01, 176_624),
    (date(2023, 9, 26), 3.68, 127_485),
    (date(2023, 9, 27), 3.83, 167_619),
    (date(2023, 9, 28), 3.96, 57_291),
    (date(2023, 9, 29), 3.98, 70_422),
    (date(2023, 10, 2), 4.01, 48_046),
    (FI_CUT, 3.81, 76_010),
    (date(2023, 10, 4), 3.41, 174_692),
    (date(2023, 10, 5), 3.23, 104_914),
]


def fi_bars() -> list[dict[str, Any]]:
    fgmc = bar_rows("FGMC", date(2022, 4, 19), date(2023, 8, 25), 10.0)
    return [
        *fgmc,
        *filler_rows(fgmc[-1], FI_P_START, date(2023, 12, 29)),  # real: frozen, volume 0
        *closes("ICCT", FI_ICCT_REAL),
        *bar_rows("ICCT", date(2023, 10, 6), date(2023, 12, 29), 3.2),
    ]


@pytest.fixture(scope="module")
def fi_built() -> Built:
    return build_security_master(
        bars=bars_table(fi_bars()),
        sec_history=sec_table(FI_SNAPSHOTS, FI_LIVES),
        name_changes=name_changes_table([]),
    )


def test_r_fgmc_icct_listing_before_the_sec_saw_it_is_the_merged_company(
    fi_built: Built,
) -> None:
    # R. Today (real master): ICCT [08-28, 10-03) S014515 cik 1408057, FGMC [2022-04-19,
    # 10-03) and ICCT [10-03, None) S013493 cik 1906133, conflict FGMC/ICCT unresolved. FGMC's
    # traded bars all lie before 08-28 (outside the span) and its bars inside are filler.
    master, conflicts = fi_built

    p = segment_at(master, "ICCT", date(2023, 9, 1))
    new = segment_at(master, "ICCT", date(2023, 10, 4))
    spac = segment_at(master, "FGMC", date(2023, 1, 3))
    assert (p["valid_from"], p["valid_to"]) == (FI_P_START, FI_CUT)
    assert p["security_id"] == new["security_id"] == spac["security_id"]
    assert p["cik"] == new["cik"] == FI_SPAC
    assert p["evidence"] == "relabelled"
    # first-bar rule: the relabelled piece is knowable with the new piece, at its first bar
    assert p["available_at"] == bar_available_at(FI_CUT)
    assert new["available_at"] == bar_available_at(FI_CUT)
    [row] = touching(conflicts, "ICCT", FI_CUT)
    assert row["resolution"] == "relabelled"
    assert str(FI_OLD) in row["detail"]


def test_r_fgmc_ends_where_the_relabelled_piece_begins(fi_built: Built) -> None:
    master, _ = fi_built

    [fgmc] = segments(master, "FGMC")
    assert fgmc["valid_to"] == FI_P_START
    assert overlaps_within_securities(master) == []


def test_r_fgmc_clipped_end_is_not_knowable_before_the_takeover(fi_built: Built) -> None:
    # Leakage. The FGMC filler after 08-25 is on disk all along; that FGMC ended on 08-28 is
    # only known with the SEC snapshot of 10-03. A reader on 2023-09-15 sees FGMC open and no
    # ICCT segment at all.
    master, _ = fi_built

    [fgmc] = segments(master, "FGMC")
    # first-bar rule: the clip is part of the relabelling, knowable at the new piece's first bar
    assert fgmc["end_available_at"] == bar_available_at(FI_CUT)
    assert fgmc["end_available_at"] >= segment_at(master, "ICCT", FI_P_START)["available_at"]
    mid = visible_at(master, datetime(2023, 9, 15, 12, tzinfo=UTC))
    assert [r["symbol"] for r in mid if r["symbol"] in ("FGMC", "ICCT")] == ["FGMC"]
    [seen] = [r for r in mid if r["symbol"] == "FGMC"]
    assert seen["end_available_at"] > datetime(2023, 9, 15, 12, tzinfo=UTC)
    assert_nothing_known_before_its_valid_from(master)
    assert_end_knowability_invariant(master)


# --- DISCA / WBD and BBU / BBUC: the taker traded its own bars in the span --------------------

DW_CUT = date(2022, 4, 11)
DW_SNAPSHOTS = [
    date(2019, 10, 2),
    date(2020, 1, 13),
    date(2020, 6, 1),
    date(2021, 1, 4),
    date(2022, 1, 3),
    date(2022, 4, 1),
    date(2022, 5, 3),
    date(2022, 6, 5),
    date(2023, 1, 3),
]
DW_LIVES: list[Life] = [
    ("WBD", 1166718, date(2019, 10, 2), date(2020, 6, 1)),  # real: a stale row
    ("DISCA", 1437107, date(2019, 10, 2), date(2022, 4, 1)),
    ("WBD", 1437107, date(2022, 5, 3), None),
]
DW_WI_DAYS = [(date(2022, 4, d), c, v) for d, c, v in [
    (4, 25.5, 61_860), (5, 24.41, 93_571), (6, 23.98, 63_434), (7, 23.12, 272_978),
    (8, 24.47, 646_531),
]]  # fmt: skip
DW_DISCA_DAYS = [(date(2022, 4, d), c, v) for d, c, v in [
    (4, 25.8, 8_855_783), (5, 24.42, 11_986_503), (6, 23.92, 11_278_379),
    (7, 23.07, 16_822_112), (8, 24.43, 22_396_034),
]]  # fmt: skip


@pytest.fixture(scope="module")
def dw_built() -> Built:
    wbd_wi = closes("WBD", DW_WI_DAYS)
    return build_security_master(
        bars=bars_table(
            bar_rows("DISCA", date(2020, 1, 2), date(2022, 4, 1), 20.0)
            + closes("DISCA", DW_DISCA_DAYS)
            + wbd_wi
            + relabel(wbd_wi, "WBDWV")
            + bar_rows("WBD", DW_CUT, date(2022, 12, 30), 24.78)
        ),
        sec_history=sec_table(DW_SNAPSHOTS, DW_LIVES),
        name_changes=name_changes_table([("WBDWV", "WBD", DW_CUT, "934423104", "934423104")]),
    )


def test_r_disca_traded_its_own_bars_so_old_wbd_stays_apart(dw_built: Built) -> None:
    # R, "yes" branch. Today: the two securities are already apart but the conflict reads
    # unresolved. DISCA (the taker's chain) has five traded bars in [04-04, 04-11), none equal
    # to WBD's.
    master, conflicts = dw_built

    old = segment_at(master, "WBD", date(2022, 4, 6))
    new = segment_at(master, "WBD", date(2022, 4, 12))
    assert old["security_id"] != new["security_id"]
    assert new["security_id"] == segment_at(master, "DISCA", date(2022, 4, 6))["security_id"]
    assert old["evidence"] != "relabelled"
    [row] = touching(conflicts, "WBD", DW_CUT)
    assert row["resolution"] == "new_security"
    assert not [r for r in unresolved(conflicts) if "WBD" in r["symbols"]]


BB_CUT = date(2026, 3, 31)
BB_SNAPSHOTS = [
    date(2021, 9, 3),
    date(2022, 1, 3),
    date(2023, 1, 3),
    date(2024, 1, 2),
    date(2025, 1, 2),
    date(2026, 1, 5),
    date(2026, 3, 29),
    date(2026, 4, 21),
    date(2026, 6, 1),
]
BB_LIVES: list[Life] = [
    ("BBU", 1654795, date(2021, 9, 3), date(2026, 3, 29)),
    ("BBUC", 1871130, date(2021, 9, 3), date(2026, 3, 29)),
    ("BBUC", 1654795, date(2026, 4, 21), None),
]


@pytest.fixture(scope="module")
def bb_built() -> Built:
    return build_security_master(
        bars=bars_table(
            bar_rows("BBU", date(2020, 1, 2), date(2026, 3, 30), 25.0)
            + closes("BBUC.WI", [(date(2022, 3, 4), 30.0, 5_000), (date(2022, 3, 9), 30.5, 4_000)])
            + bar_rows("BBUC", date(2022, 3, 15), date(2026, 6, 30), 28.0)
        ),
        sec_history=sec_table(BB_SNAPSHOTS, BB_LIVES),
        name_changes=name_changes_table(
            [("BBUC.WI", "BBUC", date(2022, 3, 15), "11259V106", "11259V106")]
        ),
    )


def test_r_bbu_traded_beside_bbuc_so_they_stay_two_securities(bb_built: Built) -> None:
    master, conflicts = bb_built

    old = segment_at(master, "BBUC", date(2024, 1, 3))
    new = segment_at(master, "BBUC", date(2026, 4, 1))
    assert new["valid_from"] == BB_CUT
    assert old["security_id"] != new["security_id"]
    assert new["security_id"] == segment_at(master, "BBU", date(2024, 1, 3))["security_id"]
    assert old["cik"] == 1871130
    rows = [r for r in conflicts_of(conflicts, "reuse_looks_continuous") if "BBUC" in r["symbols"]]
    assert [r["resolution"] for r in rows] == ["new_security"]


def test_r_worlds_are_continuous_at_the_cut_with_different_ciks() -> None:
    # fixture sanity: every R world is the shape the rule applies to (continuous closes across
    # the cut on the label itself, last bar before and first bar after within MAX_GAP_DAYS)
    cases = [
        (gb_bars(), "B", GB_CUT),
        (in_bars(), "INBX", IN_CUT),
        (fi_bars(), "ICCT", FI_CUT),
    ]
    for rows, symbol, cut in cases:
        traded = sorted(
            (r["session_date"], r["close"])
            for r in rows
            if r["symbol"] == symbol and r["volume"] > 0
        )
        d0, c0 = max(t for t in traded if t[0] < cut)
        d1, c1 = min(t for t in traded if t[0] >= cut)
        assert (d1 - d0).days <= sm.MAX_GAP_DAYS, symbol
        assert sm.CONTINUOUS[0] <= c1 / c0 <= sm.CONTINUOUS[1], symbol


# --- guards on R (reviewer, 2026-10-19): shapes the merge must leave alone --------------------
# Constructed world: no real case of this shape is in the store today (the 6-c rebuild merged
# only B, ICCT and INBX); it pins what the rule does when one appears. The round-trip guard
# (previous piece already in the taker's chain) has no test: a label without SEC rows does not
# keep its pre-trip piece at all (_pieces_for), and one with SEC rows has a CIK on both sides
# and is chained before R runs.

CH_RENAME, CH_CUT = date(2021, 3, 1), date(2022, 6, 1)
CH_Y, CH_X = 7001, 7002


def test_r_previous_piece_that_chains_to_another_name_is_left_for_a_human() -> None:
    # YOLD renamed itself LBL in 2021; in 2022 the SPAC XSPC (silent since 2020) takes LBL at a
    # continuous price. Merging LBL's 2021-2022 piece into XSPC would carry YOLD's 2020 history
    # along with it, so R must not decide: the row stays unresolved and LBL's earlier piece
    # stays with YOLD.
    snapshots = [date(2020, 1, 13), date(2021, 1, 4), date(2021, 6, 1), date(2022, 9, 1)]
    lives: list[Life] = [
        ("YOLD", CH_Y, date(2020, 1, 13), date(2021, 1, 4)),
        ("LBL", CH_Y, date(2021, 6, 1), date(2021, 6, 1)),
        ("XSPC", CH_X, date(2020, 1, 13), date(2021, 6, 1)),
        ("LBL", CH_X, date(2022, 9, 1), None),
    ]
    master, conflicts = build_security_master(
        bars=bars_table(
            bar_rows("YOLD", date(2020, 1, 2), date(2021, 2, 26), 20.0)
            + bar_rows("LBL", CH_RENAME, date(2022, 12, 30), 20.5)
            + bar_rows("XSPC", date(2020, 1, 2), date(2020, 6, 30), 10.0)
        ),
        sec_history=sec_table(snapshots, lives),
        name_changes=name_changes_table(
            [
                ("YOLD", "LBL", CH_RENAME, "000000BB1", "000000BB2"),
                ("XSPC", "LBL", CH_CUT, "000000CC1", "000000CC2"),
            ]
        ),
    )

    old = segment_at(master, "LBL", date(2021, 9, 1))
    assert old["security_id"] == segment_at(master, "YOLD", date(2020, 6, 1))["security_id"]
    assert old["evidence"] != "relabelled"
    assert old["cik"] == CH_Y
    [row] = touching(conflicts, "LBL", CH_CUT)
    assert row["resolution"] == "unresolved"


# --- guards on R called directly: shapes the full build does not produce today ---------------
# Constructed pieces (no real case in the store on 2026-10-19). The taker trades nothing in the
# span, so without the guard R would merge; each test pins that the row stays unresolved and
# nothing is moved, clipped or dropped.

GD_V, GD_CUT = date(2022, 3, 1), date(2022, 6, 1)
GD_KNOWN = et_midnight(GD_CUT)


def _guard_case(extra: list[sm._Segment]) -> tuple[list[sm._Segment], sm._Union, list[dict], set]:
    before = sm._Segment("LBL", GD_V, GD_CUT, cik=8001, available_at=et_midnight(GD_V))
    after = sm._Segment("LBL", GD_CUT, None, cik=8002, available_at=GD_KNOWN)
    pieces = [before, after, *extra]
    for i, p in enumerate(pieces):
        p.key = i
    uf = sm._Union(len(pieces))
    for p in extra:
        if p.symbol != "LBL":
            uf.union(after.key, p.key)  # a parallel name of the taker's security
    conflicts = [{"detail": "constructed", "resolution": "unresolved"}]
    con = duckdb.connect()
    try:
        con.register("_bars", bars_table(bar_rows("LBL", GD_V, GD_CUT - timedelta(days=1), 20.0)))
        dropped = sm._settle_takeovers(con, pieces, uf, [(0, before, after, GD_CUT)], conflicts)
    finally:
        con.close()
    return pieces, uf, conflicts, dropped


def test_r_taker_piece_running_past_the_cut_is_not_clipped() -> None:
    # A piece of the taker's chain starts inside the span and runs on past the cut (an open
    # parallel name, no traded bar in the span). Clipping it to the span's start would orphan
    # its bars after the cut, so R must not merge.
    parallel = sm._Segment("TKRB", date(2022, 4, 1), None, cik=8002, available_at=GD_KNOWN)
    pieces, uf, conflicts, dropped = _guard_case([parallel])
    before, after = pieces[0], pieces[1]

    assert conflicts[0]["resolution"] == "unresolved"
    assert dropped == set()
    assert (parallel.valid_from, parallel.valid_to) == (date(2022, 4, 1), None)
    assert uf.find(before.key) != uf.find(after.key)
    assert (before.cik, before.evidence) == (8001, "bars_only")


def test_r_previous_piece_following_another_on_the_label_is_not_moved() -> None:
    # The label had an earlier piece ending where the previous holder's begins. That piece's
    # end is knowable when the previous holder's start was; moving the previous holder's
    # available_at to the takeover would break the pair (check_master refuses it), so R must
    # not merge.
    earlier = sm._Segment(
        "LBL", date(2020, 1, 2), GD_V, cik=8000, available_at=et_midnight(date(2020, 1, 2))
    )
    earlier.end_available_at = et_midnight(GD_V)
    pieces, uf, conflicts, dropped = _guard_case([earlier])
    before, after = pieces[0], pieces[1]

    assert conflicts[0]["resolution"] == "unresolved"
    assert dropped == set()
    assert before.available_at == et_midnight(GD_V)
    assert uf.find(before.key) != uf.find(after.key)
