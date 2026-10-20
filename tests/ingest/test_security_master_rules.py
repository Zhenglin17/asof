"""Guards against five ways the master attributes bars to the wrong security or to nobody, each
measured on the whole store on 2026-10-16 (notes: step6b-counts.md, sections N, D, V, O, A):

  N  the SEC spells a unit dotted once and undotted afterwards (KINZ.U then KINZU); keeping the
     dotted spelling leaves the label the store has (KINZU) without SEC rows -> no CIK;
  D  Alpaca files a rename weeks late (SVMK -> MNTV filed 2021-08-09, traded 06-15); the cut
     sits on the record and the new label gets a spurious opening piece or orphans;
  V  a rename whose old name never traded here while the new one already did (CBS -> VIAC,
     WORXD -> WORX): the new label is started at the record, its earlier bars are orphans or
     a second security;
  O  a label vacated by a rename whose bars resume after a gap (JAMF placeholder -> JNMF, then
     the real Jamf IPO; StoneX INTL -> SNEX, then an ETF under INTL): the resumed bars own
     nothing;
  A  when-issued symbols (HONAV, AAN.WI) chained into the regular security.

Proposed API (the implementation may regroup internals; ``build_security_master``'s contract
and these names stay):

    normalize_spelling(sec_history, store_symbols=None)
        Same CIK, same letters once dots are removed -> the spelling in ``store_symbols``;
        when none or both are there, the dotted one (today's rule). ``build_security_master``
        passes the bar store's symbols.
    HANDOVER_MAX_DAYS = 5          _refine_alpaca_edges: also move the cut when the new name's
                                   first traded bar is strictly after the old name's last and
                                   within this many calendar days of it, for records trailing
                                   by up to SEC_LAG_MAX_DAYS (183). ``available_at`` stays the
                                   record's.
    RESUME_GAP_DAYS = 14           O: traded bars of a closed label that resume more than this
                                   many calendar days after the label's previous traded bar
                                   open a new piece (CIK from SEC rows inside it, evidence
                                   ``sec``; else ``bars_only``; ``available_at`` = that bar's).
    WHEN_ISSUED = re.compile(r".*\\.(WI|RTWI)$|[A-Z]{4}V")
                                   A: an edge whose old name matches still cuts the new label
                                   at the record but never unions the two pieces.

Every world below is self-contained. Fake data provenance (store probe 2026-10-16 unless noted):
- KINZU: SEC KINZ.U on 2020-11-03 once, KINZU 21 times 2021-07-05..2023-03-01, CIK 1820875;
  bars 2021-01-04..2023-02-28 under KINZU only (real). BRKB / BRK.B: real mechanism (shared
  world of test_security_master.py). NONEU / NONE.U: constructed, neither has bars.
- SVMK -> MNTV: record 2021-08-09; SVMK's last traded bar 2021-06-14, MNTV's first 06-15; SEC
  CIK 1739936 lists SVMK to 2021-08-02 and MNTV from 2021-07-05 (both for a while, so no SEC
  edge) (real). Price levels ~21 real; CUSIPs not checked.
- CBS -> VIAC: record 2020-02-13, VIAC bars from 2020-01-02, CBS none, SEC VIAC under 813828
  from 2020-02-16 (real). CBS's own SEC rows are left out: in the real file the SEC edge
  CBS -> VIAC is dropped by the F1 flicker rule, so Alpaca's record is the only edge.
- WORXD -> WORX: record 2026-09-01, WORX bars 2020-01-02..2026-10, WORXD none, SEC WORX under
  1674227 throughout and WORXD once on 2026-09-08 (real).
- ARNC.WI -> ARNC 2020-04-01: ARNC.WI has no bars, ARNC trades through (real); Arconic Corp's
  3x level and the SEC rows as in F3 of test_security_master.py (ASSUMPTION on levels).
- JAMF -> JNMF 2020-01-13 (real): JAMF has ONE traded bar (2020-01-02, 500 shares, 26.73) and
  volume-0 filler at 26.73 to 2020-07-21, real bars from 2020-07-22 (the IPO); JNMF 18 traded
  bars 2020-01-13..05-27 (modelled as a short run, levels ASSUMED); SEC JAMF under 1721947
  from 2020-08-13 (real).
- INTL -> SNEX: Alpaca record 2020-07-06 (real; the note called it SEC-only, the store has the
  record); INTL traded to 2020-07-02, SNEX carries INTL's history relabelled plus its own; SEC
  INTL under 913760 to 2021-01-03 (lag), SNEX from 2020-07-10; INTL's bars resume 2022-12-02
  (an ETF the SEC never lists) (real; levels 52 / 20 real).
- HONAV -> HONA 2026-06-29: HONAV traded 9 days 2026-06-15..06-26 (06-19 holiday), HONA's bars
  on those days are exact copies, HONA's own from 06-29; SEC HONA under 2089271 from
  2026-03-29 (real; levels ~200..230 real).
- AAN.WI -> AAN and PRG.WI -> PRG, both 2020-12-01 (real; Alpaca also filed AAN.WI -> AAN again
  on 12-03, a duplicate left out). AAN 62 -> 18.4 across the spin-off, PRG 55.6, the when-issued
  bars 11-25..11-30 (real). SEC: old AAN under 706688, new AAN under 1821393 from the 12-01
  snapshot (real). SIMPLIFIED: the real file also re-registers the old holder as 1808834 on
  2020-11-03 (a CIK switch bridged by the existing rules); here PRG keeps 706688.
- OLD*/NEW*, TAKR, NONE*: constructed boundaries, see each test.

Assumptions that need a real-data check (audit or golden case) because the fake worlds cannot
tell whether they are right:
 1. N: after the rebuild the 5 undotted-with-bars groups (ACEVU, ERESU, KINZU, QELLU, TMKRU)
    carry a CIK and the 64 dotted ones (BRK.B, CWEN.A) are unchanged -> golden KINZU case;
    "both spellings have bars" is constructed (the note found 0 such groups).
 2. D: the 248 moved records produce no overlap (audit M2) and no orphan before the new cut
    (I1); the 5-vs-6-day boundary and the "relabelled history under the new name" guard are
    constructed -- count real records in the 11-60 d bucket whose new name starts 6-10 days
    after the old name's last bar (left as orphans by this rule).
 3. V: every one of the 62 records' new labels now starts at its first bar; no XD/XQ old name
    has bars (then V does not apply) -> golden VIAC / WORX. test_ffr_and_para_... in
    tests/store/test_golden_cases.py pins VIAC from 2020-02-13 and must be updated with V.
 4. O: the resumed piece must not inherit the previous holder's CIK through the "SEC rows
    before valid_from" fallback (INTL stays bars_only -> golden INTL); the 14-day boundary is
    constructed; the gap is measured on traded bars (JAMF's filler is real). Interplay with F
    (copied tails after a gap: OAS <- CHRD, AXNX <- MIMI) is NOT covered here: O would open a
    bars_only piece for those tails unless F cuts them first -> audit I2 after the rebuild.
 5. A: every 5-letter symbol ending in V is treated as when-issued (longest real one BBIGV, 25
    traded days) -> list 5-letter V symbols with > 25 traded days on the real store; the AAN
    chain on real data (old AAN + PRG one security, AAN.WI / PRG.WI chained to nobody) -> audit
    M2 "A" rows go to 0; HONA -> golden case.
"""

from datetime import date, timedelta
from typing import Any

import pyarrow as pa
import pytest

from asof.ingest import security_master as sm
from asof.ingest.security_master import build_security_master, normalize_spelling
from tests.ingest.test_security_master import (
    Life,
    Rename,
    assert_end_knowability_invariant,
    assert_nothing_known_before_its_valid_from,
    bar_available_at,
    bar_rows,
    bars_table,
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
    unresolved,
    visible_at,
)

MICRO = timedelta(microseconds=1)


def spellings_of(table: pa.Table, cik: int) -> set[str]:
    tickers = table.column("ticker").to_pylist()
    ciks = table.column("cik").to_pylist()
    return {t for t, c in zip(tickers, ciks, strict=True) if c == cik}


# =============================================================================================
# N: the spelling the bar store uses wins over the dotted one
# =============================================================================================

N_SNAPSHOTS = [
    date(2020, 11, 3),
    date(2021, 1, 4),
    date(2021, 7, 5),
    date(2022, 1, 3),
    date(2023, 3, 1),
]
N_KINZ_CIK = 1820875  # real
N_LIVES: list[Life] = [
    ("KINZ.U", N_KINZ_CIK, date(2020, 11, 3), date(2020, 11, 3)),  # real: dotted once
    ("KINZU", N_KINZ_CIK, date(2021, 7, 5), date(2023, 3, 1)),  # real: undotted afterwards
    ("BRKB", 1067983, date(2020, 11, 3), date(2021, 1, 4)),  # real mechanism: old spelling
    ("BRK.B", 1067983, date(2021, 7, 5), None),
    ("NONEU", 7001, date(2020, 11, 3), date(2020, 11, 3)),  # constructed: neither has bars
    ("NONE.U", 7001, date(2021, 1, 4), None),
]
N_FIRST_BAR = date(2021, 1, 4)  # real: KINZU's first bar in the store


def n_sec() -> pa.Table:
    return sec_table(N_SNAPSHOTS, N_LIVES)


def n_bars() -> pa.Table:
    return bars_table(
        bar_rows("KINZU", N_FIRST_BAR, date(2023, 2, 28), 10.0)
        + bar_rows("BRK.B", N_FIRST_BAR, date(2023, 2, 28), 280.0)
    )


def test_n_store_spelling_wins_over_the_dotted_one() -> None:
    # N. Today every KINZ row becomes KINZ.U, the spelling the store never uses.
    out = normalize_spelling(n_sec(), store_symbols={"KINZU", "BRK.B"})

    assert spellings_of(out, N_KINZ_CIK) == {"KINZU"}
    assert spellings_of(out, 1067983) == {"BRK.B"}  # counter-case: dotted one is the store's
    assert out.num_rows == n_sec().num_rows


def test_n_neither_spelling_in_the_store_falls_back_to_dotted() -> None:
    out = normalize_spelling(n_sec(), store_symbols={"KINZU", "BRK.B"})

    assert spellings_of(out, 7001) == {"NONE.U"}


def test_n_both_spellings_in_the_store_fall_back_to_dotted() -> None:
    # Constructed (the note found no such group): the store cannot break the tie, so the old
    # rule decides.
    out = normalize_spelling(n_sec(), store_symbols={"KINZU", "KINZ.U", "BRK.B"})

    assert spellings_of(out, N_KINZ_CIK) == {"KINZ.U"}


def test_n_without_store_symbols_the_dotted_spelling_wins() -> None:
    # Guard (passes today, must keep passing): the old behaviour when no store is given.
    out = normalize_spelling(n_sec())

    assert spellings_of(out, N_KINZ_CIK) == {"KINZ.U"}
    assert spellings_of(out, 1067983) == {"BRK.B"}
    assert spellings_of(out, 7001) == {"NONE.U"}


def test_n_kinzu_segment_carries_the_cik_through_the_build() -> None:
    # N end to end: build_security_master hands the bar symbols to normalize_spelling. Today
    # KINZU is bars_only with cik None (real master row) and KINZ.U is nowhere.
    master, conflicts = build_security_master(
        bars=n_bars(), sec_history=n_sec(), name_changes=name_changes_table([])
    )

    assert len(segments(master, "KINZU")) == 1
    [kinzu] = segments(master, "KINZU")
    assert (kinzu["valid_from"], kinzu["valid_to"]) == (N_FIRST_BAR, None)
    assert kinzu["cik"] == N_KINZ_CIK
    assert kinzu["evidence"] == "sec"
    # the spelling fix adds no event: the segment is knowable with its first bar, as before
    assert kinzu["available_at"] == bar_available_at(N_FIRST_BAR)
    symbols = set(master.column("symbol").to_pylist())
    assert "KINZ.U" not in symbols and "BRKB" not in symbols
    [brkb] = segments(master, "BRK.B")
    assert brkb["cik"] == 1067983
    assert conflicts.num_rows == 0
    assert_nothing_known_before_its_valid_from(master)


# =============================================================================================
# D: a late Alpaca record whose hand-over is visible in the bars
# =============================================================================================

D_SNAPSHOTS = [
    date(2021, 1, 4),
    date(2021, 6, 1),
    date(2021, 7, 5),  # real: first snapshot listing MNTV
    date(2021, 8, 2),  # real: last snapshot listing SVMK
    date(2021, 9, 1),
    date(2022, 1, 3),
]
D_RECORD = date(2021, 8, 9)  # real: Alpaca's process_date
D_SVMK_LAST = date(2021, 6, 14)  # real: SVMK's last traded bar
D_MNTV_FIRST = date(2021, 6, 15)  # real: MNTV's first bar; also the cut (last + 1)
D_CIK = 1739936  # real
D_LIVES: list[Life] = [
    ("SVMK", D_CIK, date(2021, 1, 4), date(2021, 8, 2)),
    ("MNTV", D_CIK, date(2021, 7, 5), None),
]
D_RENAMES: list[Rename] = [("SVMK", "MNTV", D_RECORD, "869500100", "609500100")]  # CUSIPs
#   constructed


def d_bars() -> list[dict[str, Any]]:
    return [
        *bar_rows("SVMK", date(2020, 1, 2), D_SVMK_LAST, 21.0),
        *bar_rows("MNTV", D_MNTV_FIRST, date(2023, 5, 31), 21.02),
    ]


@pytest.fixture(scope="module")
def d_built() -> tuple[pa.Table, pa.Table]:
    return build_security_master(
        bars=bars_table(d_bars()),
        sec_history=sec_table(D_SNAPSHOTS, D_LIVES),
        name_changes=name_changes_table(D_RENAMES),
    )


def test_d_constants() -> None:
    assert sm.MAX_GAP_DAYS == 10
    assert sm.SEC_LAG_MAX_DAYS == 183
    assert sm.HANDOVER_MAX_DAYS == 5


def test_d_late_record_with_a_visible_handover_cuts_at_the_new_names_first_bar(
    d_built: tuple[pa.Table, pa.Table],
) -> None:
    # D. Today (real master): MNTV [06-15, 08-09) 'sec' + [08-09, None) 'name_change', SVMK
    # runs to 08-09 and overlaps the first MNTV piece. The record trails SVMK's last bar by 55
    # days (> MAX_GAP_DAYS) but MNTV's first bar is the very next day: the hand-over is in the
    # bars, so the cut is 06-15.
    master, conflicts = d_built

    [svmk] = segments(master, "SVMK")
    [mntv] = segments(master, "MNTV")
    assert (svmk["valid_from"], svmk["valid_to"]) == (date(2020, 1, 2), D_MNTV_FIRST)
    assert (mntv["valid_from"], mntv["valid_to"]) == (D_MNTV_FIRST, None)
    assert svmk["security_id"] == mntv["security_id"]
    assert svmk["cik"] == mntv["cik"] == D_CIK
    assert mntv["evidence"] == "name_change"
    assert mntv["cusip"] == "609500100"
    for day in (D_MNTV_FIRST, date(2021, 7, 1), D_RECORD, date(2022, 3, 1)):
        assert segment_at(master, "MNTV", day) == mntv, day
    assert covering(master, "SVMK", date(2021, 7, 1)) == []
    assert not any("SVMK" in r["symbols"] or "MNTV" in r["symbols"] for r in conflicts.to_pylist())


def test_d_moved_cut_is_knowable_at_the_record_not_at_the_first_bar(
    d_built: tuple[pa.Table, pa.Table],
) -> None:
    # Leakage. The rename was filed 08-09: a reader on 2021-07-01 sees SVMK (whose end it must
    # treat as open) and no MNTV at all; at midnight 08-09 both facts appear together. The
    # available_at is therefore later than midnight of valid_from, as in F9.
    master, _ = d_built

    [svmk] = segments(master, "SVMK")
    [mntv] = segments(master, "MNTV")
    assert mntv["available_at"] == et_midnight(D_RECORD)
    assert svmk["end_available_at"] == et_midnight(D_RECORD)
    assert mntv["available_at"] > et_midnight(mntv["valid_from"])

    july = visible_at(master, et_midnight(date(2021, 7, 1)))
    assert "MNTV" not in {r["symbol"] for r in july}
    [svmk_seen] = [r for r in july if r["symbol"] == "SVMK"]
    assert svmk_seen["end_available_at"] > et_midnight(date(2021, 7, 1))
    on_record = [r for r in visible_at(master, et_midnight(D_RECORD)) if r["symbol"] == "MNTV"]
    assert [(r["valid_from"], r["valid_to"]) for r in on_record] == [(D_MNTV_FIRST, None)]
    assert_nothing_known_before_its_valid_from(master)
    assert_end_knowability_invariant(master)


D_OLD_LAST = date(2021, 6, 10)  # a Thursday


def d_handover_world(gap_days: int) -> tuple[pa.Table, pa.Table]:
    """OLD's last traded bar on a Thursday, NEW's first ``gap_days`` later, the record on
    D_RECORD (60 days after). No SEC rows: the rule reads the bars and the record only."""
    new_first = D_OLD_LAST + timedelta(days=gap_days)
    return build_security_master(
        bars=bars_table(
            bar_rows("OLD", date(2020, 1, 2), D_OLD_LAST, 10.0)
            + bar_rows("NEW", new_first, date(2021, 12, 31), 10.1)
        ),
        sec_history=sec_table([], []),
        name_changes=name_changes_table([("OLD", "NEW", D_RECORD, "000100100", "000100200")]),
    )


@pytest.mark.parametrize(
    ("gap_days", "moved"),
    [(5, True), (6, False)],
    ids=["five_days_after_the_old_bar", "six_days_after_the_old_bar"],
)
def test_d_handover_within_five_calendar_days_moves_the_cut(gap_days: int, moved: bool) -> None:
    # D boundary (constructed): "within HANDOVER_MAX_DAYS" means first_new - last_old <= 5. The
    # cut is the day after OLD's last traded bar (06-11), not NEW's first bar. Beyond the
    # boundary the record keeps its date and NEW's early bars stay uncovered.
    master, _ = d_handover_world(gap_days)

    [old] = segments(master, "OLD")
    [new] = segments(master, "NEW")
    cut = D_OLD_LAST + timedelta(days=1) if moved else D_RECORD
    assert new["valid_from"] == old["valid_to"] == cut
    assert new["security_id"] == old["security_id"]
    # first-bar rule: unmoved, NEW's first own bar is on the record day
    assert new["available_at"] == (et_midnight(D_RECORD) if moved else bar_available_at(D_RECORD))
    assert old["end_available_at"] == et_midnight(D_RECORD)
    assert_end_knowability_invariant(master)


def test_d_relabelled_history_under_the_new_name_keeps_the_record_date() -> None:
    # D guard (passes today, must keep passing): NEWA carries OLDA's whole history (Alpaca's
    # relabelling, real mechanism BK / BNY), so NEWA's FIRST traded bar is 2020-01-02, before
    # OLDA's last -- the hand-over cannot be read off the bars and only the <= 10-day rule
    # applies; a 51-day lag is not moved. NEWA's 06-15..08-03 bars stay uncovered (accepted,
    # ASSUMPTION flagged in the file header).
    olda = bar_rows("OLDA", date(2020, 1, 2), date(2021, 6, 14), 40.0)
    record = date(2021, 8, 4)
    master, _ = build_security_master(
        bars=bars_table(
            olda
            + relabel(olda, "NEWA")
            + bar_rows("NEWA", date(2021, 6, 15), date(2021, 12, 31), 40.5)
        ),
        sec_history=sec_table([], []),
        name_changes=name_changes_table([("OLDA", "NEWA", record, "000200100", "000200200")]),
    )

    [olda_seg] = segments(master, "OLDA")
    [newa] = segments(master, "NEWA")
    assert newa["valid_from"] == olda_seg["valid_to"] == record
    assert newa["security_id"] == olda_seg["security_id"]
    assert covering(master, "NEWA", date(2021, 7, 1)) == []


def test_d_record_trailing_by_more_than_sec_lag_max_days_is_not_moved() -> None:
    # D guard (passes today, must keep passing): OLDB stops 2020-06-30, NEWB starts 07-01, the
    # record comes 213 days later -- outside SEC_LAG_MAX_DAYS, whatever the bars show.
    record = date(2021, 1, 29)
    master, _ = build_security_master(
        bars=bars_table(
            bar_rows("OLDB", date(2020, 1, 2), date(2020, 6, 30), 15.0)
            + bar_rows("NEWB", date(2020, 7, 1), date(2021, 12, 31), 15.1)
        ),
        sec_history=sec_table([], []),
        name_changes=name_changes_table([("OLDB", "NEWB", record, "000300100", "000300200")]),
    )

    [oldb] = segments(master, "OLDB")
    [newb] = segments(master, "NEWB")
    assert (record - date(2020, 6, 30)).days > sm.SEC_LAG_MAX_DAYS
    assert newb["valid_from"] == oldb["valid_to"] == record
    assert newb["security_id"] == oldb["security_id"]


# =============================================================================================
# V: a rename whose old name has no bars while the new name already traded
# =============================================================================================

V_VIAC_RECORD = date(2020, 2, 13)  # real
V_VIAC_FIRST_BAR = date(2020, 1, 2)  # real
V_VIAC_CIK = 813828  # real
V_VIAC_SNAPSHOTS = [
    date(2020, 1, 13),
    date(2020, 2, 16),  # real: first snapshot listing VIAC
    date(2020, 6, 1),
    date(2021, 1, 4),
    date(2022, 1, 3),
]
V_VIAC_RENAMES: list[Rename] = [("CBS", "VIAC", V_VIAC_RECORD, "124857202", "92556H206")]


@pytest.fixture(scope="module")
def v_viac_built() -> tuple[pa.Table, pa.Table]:
    return build_security_master(
        bars=bars_table(bar_rows("VIAC", V_VIAC_FIRST_BAR, date(2022, 2, 15), 42.0)),
        sec_history=sec_table(V_VIAC_SNAPSHOTS, [("VIAC", V_VIAC_CIK, date(2020, 2, 16), None)]),
        name_changes=name_changes_table(V_VIAC_RENAMES),
    )


def test_v_viac_segment_starts_at_its_first_bar_not_at_the_record(
    v_viac_built: tuple[pa.Table, pa.Table],
) -> None:
    # V (a). Today (real master): VIAC [2020-02-13, ...) and the 29 bars before it belong to
    # nobody. CBS has no bars, so the record cannot be a takeover: it is not an event.
    master, conflicts = v_viac_built

    assert len(segments(master, "VIAC")) == 1
    [viac] = segments(master, "VIAC")
    assert (viac["valid_from"], viac["valid_to"]) == (V_VIAC_FIRST_BAR, None)
    assert viac["cik"] == V_VIAC_CIK
    assert viac["evidence"] == "sec"
    assert viac["cusip"] is None
    for day in (V_VIAC_FIRST_BAR, date(2020, 1, 15), V_VIAC_RECORD, date(2021, 6, 1)):
        assert segment_at(master, "VIAC", day) == viac, day
    assert "CBS" not in set(master.column("symbol").to_pylist())
    assert not any("VIAC" in r["symbols"] for r in unresolved(conflicts))


def test_v_viac_segment_is_knowable_with_its_first_bar(
    v_viac_built: tuple[pa.Table, pa.Table],
) -> None:
    # Leakage (DECISIONS 2026-10-16): the day's trading is the public evidence that the label
    # exists, so the segment is knowable at the first bar's availability (20:00 ET) -- not a
    # microsecond earlier and not only at the record's midnight; nothing starts on 02-13.
    master, _ = v_viac_built

    [viac] = segments(master, "VIAC")
    assert viac["available_at"] == bar_available_at(V_VIAC_FIRST_BAR)
    assert "VIAC" not in {
        r["symbol"] for r in visible_at(master, bar_available_at(V_VIAC_FIRST_BAR) - MICRO)
    }
    assert "VIAC" in {r["symbol"] for r in visible_at(master, bar_available_at(V_VIAC_FIRST_BAR))}
    at_record = [r for r in visible_at(master, et_midnight(V_VIAC_RECORD)) if r["symbol"] == "VIAC"]
    assert [r["valid_from"] for r in at_record] == [V_VIAC_FIRST_BAR]
    assert_nothing_known_before_its_valid_from(master)


V_WORX_RECORD = date(2026, 9, 1)  # real
V_WORX_CIK = 1674227  # real
V_WORX_SNAPSHOTS = [
    date(2020, 1, 13),
    date(2021, 1, 4),
    date(2022, 1, 3),
    date(2023, 1, 3),
    date(2024, 1, 2),
    date(2025, 1, 2),
    date(2026, 1, 5),
    date(2026, 9, 8),
]
V_WORX_LIVES: list[Life] = [
    ("WORX", V_WORX_CIK, date(2020, 1, 13), None),
    ("WORXD", V_WORX_CIK, date(2026, 9, 8), date(2026, 9, 8)),  # real: listed once, no bars
]


def test_v_worx_with_an_sec_life_since_2020_stays_one_security_in_one_segment() -> None:
    # V (b). Today (real master): WORX [2020-01-02, 2026-09-01) + [2026-09-01, None), two
    # securities. WORXD (a temporary OTC suffix) never traded here, so the record is no event.
    master, conflicts = build_security_master(
        bars=bars_table(bar_rows("WORX", date(2020, 1, 2), date(2026, 10, 5), 3.0)),
        sec_history=sec_table(V_WORX_SNAPSHOTS, V_WORX_LIVES),
        name_changes=name_changes_table(
            [("WORXD", "WORX", V_WORX_RECORD, "784000100", "784000100")]
        ),
    )

    assert len(segments(master, "WORX")) == 1
    [worx] = segments(master, "WORX")
    assert (worx["valid_from"], worx["valid_to"]) == (date(2020, 1, 2), None)
    assert worx["cik"] == V_WORX_CIK
    assert worx["evidence"] == "sec"
    assert worx["available_at"] == bar_available_at(date(2020, 1, 2))
    assert worx["end_available_at"] is None
    assert segment_at(master, "WORX", date(2026, 9, 15)) == worx
    assert "WORXD" not in set(master.column("symbol").to_pylist())
    assert not any("WORX" in r["symbols"] for r in unresolved(conflicts))
    # a reader before the record and one after see the same single open row
    before = [r for r in visible_at(master, et_midnight(date(2026, 8, 3))) if r["symbol"] == "WORX"]
    assert [(r["valid_from"], r["valid_to"]) for r in before] == [(date(2020, 1, 2), None)]
    assert_nothing_known_before_its_valid_from(master)


V_ARNC_DAY = date(2020, 4, 1)  # real: ARNC.WI -> ARNC process_date
V_ARNC_SNAPSHOTS = [date(2020, 1, 13), date(2020, 6, 1), date(2020, 7, 10), date(2021, 1, 4)]
V_ARNC_LIVES: list[Life] = [
    ("ARNC", 4281, date(2020, 1, 13), date(2020, 6, 1)),  # real: Arconic Inc (SEC lag)
    ("ARNC", 1790982, date(2020, 7, 10), None),  # real: Arconic Corp
]


@pytest.mark.parametrize("old", ["ARNC.WI", "ARNCV"], ids=["dot_wi", "fifth_letter_v"])
def test_v_when_issued_old_name_without_bars_still_cuts_the_label(old: str) -> None:
    # V exception (passes today, must keep passing): the when-issued record is the only trace
    # of ARNC changing hands on 04-01. ARNC.WI has no bars and ARNC traded before the record,
    # so V's conditions hold; the exception keeps the edge. ARNCV: the same record under the
    # Nasdaq fifth-letter form (constructed; real analogue CXAIV -> CXAI 2023-03-15).
    master, _ = build_security_master(
        bars=bars_table(
            bar_rows("ARNC", date(2020, 1, 2), date(2020, 3, 31), 25.0)
            + bar_rows("ARNC", V_ARNC_DAY, date(2020, 12, 31), 75.0)  # 3x: a plain reuse
        ),
        sec_history=sec_table(V_ARNC_SNAPSHOTS, V_ARNC_LIVES),
        name_changes=name_changes_table([(old, "ARNC", V_ARNC_DAY, "03966V107", "03966V107")]),
    )

    before = segment_at(master, "ARNC", date(2020, 3, 2))
    after = segment_at(master, "ARNC", date(2020, 5, 1))
    assert before["valid_to"] == after["valid_from"] == V_ARNC_DAY
    assert before["security_id"] != after["security_id"]
    # first-bar rule: the end stays at the record, the new holder waits for its first bar
    assert before["end_available_at"] == et_midnight(V_ARNC_DAY)
    assert after["available_at"] == bar_available_at(V_ARNC_DAY)
    assert old not in set(master.column("symbol").to_pylist())
    assert_end_knowability_invariant(master)


@pytest.mark.parametrize(
    "first_bar", [date(2021, 3, 1), date(2021, 3, 8)], ids=["on_the_record", "after_the_record"]
)
def test_v_old_name_without_bars_and_new_name_not_traded_before_is_an_ordinary_start(
    first_bar: date,
) -> None:
    # V boundary (passes today, must keep passing): nothing traded under NEWC before the
    # record, so the record is where the label starts (F9's CBS -> VIAC shape).
    record = date(2021, 3, 1)
    master, _ = build_security_master(
        bars=bars_table(bar_rows("NEWC", first_bar, date(2021, 12, 31), 7.0)),
        sec_history=sec_table([], []),
        name_changes=name_changes_table([("OLDC", "NEWC", record, "000400100", "000400200")]),
    )

    [newc] = segments(master, "NEWC")
    assert (newc["valid_from"], newc["valid_to"]) == (record, None)
    assert newc["evidence"] == "name_change"
    assert newc["cusip"] == "000400200"
    # first-bar rule: not before NEWC's first own bar (FSLF shape when after it)
    assert newc["available_at"] == bar_available_at(first_bar)
    assert "OLDC" not in set(master.column("symbol").to_pylist())


# =============================================================================================
# O: a closed label whose traded bars resume after a gap
# =============================================================================================

O_SNAPSHOTS = [
    date(2020, 1, 13),
    date(2020, 6, 1),
    date(2020, 7, 10),
    date(2020, 8, 13),  # real: first snapshot listing JAMF under 1721947
    date(2021, 1, 4),
    date(2021, 6, 1),
    date(2022, 1, 3),
    date(2023, 1, 3),
    date(2024, 1, 2),
]
O_JAMF_CIK = 1721947  # real
O_JAMF_VACATED = date(2020, 1, 13)  # real: JAMF -> JNMF process_date
O_JAMF_IPO = date(2020, 7, 22)  # real: first traded bar after the gap
O_SNEX_RECORD = date(2020, 7, 6)  # real: INTL -> SNEX process_date
O_INTL_LAST = date(2020, 7, 2)  # real: StoneX's last bar under INTL
O_INTL_RESUMED = date(2022, 12, 2)  # real: the ETF's first bar
O_LIVES: list[Life] = [
    ("JAMF", O_JAMF_CIK, date(2020, 8, 13), None),  # real
    ("INTL", 913760, date(2020, 1, 13), date(2021, 1, 4)),  # real: StoneX, SEC lag to 2021
    ("SNEX", 913760, date(2020, 7, 10), None),  # real
]
O_RENAMES: list[Rename] = [
    ("JAMF", "JNMF", O_JAMF_VACATED, "47074L105", "47074L105"),  # CUSIPs not checked
    ("INTL", "SNEX", O_SNEX_RECORD, "45884A102", "861896108"),  # CUSIPs not checked
]


def o_bars() -> list[dict[str, Any]]:
    placeholder = bar_rows("JAMF", date(2020, 1, 2), date(2020, 1, 2), 26.73, volume=500)
    intl = bar_rows("INTL", date(2020, 1, 2), O_INTL_LAST, 52.0)
    return [
        # JAMF (real shape): one traded bar, filler frozen at 26.73 up to the IPO, real bars
        # from 07-22 at ~39; JNMF: a short traded run from the record day (levels ASSUMED).
        *placeholder,
        *filler_rows(placeholder[0], date(2020, 1, 3), O_JAMF_IPO - timedelta(days=1)),
        *bar_rows("JAMF", O_JAMF_IPO, date(2021, 12, 31), 39.2),
        *bar_rows("JNMF", O_JAMF_VACATED, date(2020, 2, 5), 26.0),
        # INTL / SNEX (real shape): SNEX carries INTL's history relabelled plus its own; INTL
        # resumes 2022-12-02 as an ETF at ~20.
        *intl,
        *relabel(intl, "SNEX"),
        *bar_rows("SNEX", O_SNEX_RECORD, date(2023, 12, 29), 55.0),
        *bar_rows("INTL", O_INTL_RESUMED, date(2023, 12, 29), 20.2),
    ]


@pytest.fixture(scope="module")
def o_built() -> tuple[pa.Table, pa.Table]:
    return build_security_master(
        bars=bars_table(o_bars()),
        sec_history=sec_table(O_SNAPSHOTS, O_LIVES),
        name_changes=name_changes_table(O_RENAMES),
    )


def test_o_constant() -> None:
    assert sm.RESUME_GAP_DAYS == 14


def test_o_jamf_resumed_bars_open_a_second_security_with_the_sec_cik(
    o_built: tuple[pa.Table, pa.Table],
) -> None:
    # O, evidence 'sec'. Today (real master): JAMF [01-02, 01-13) bars_only, JNMF [01-13, None)
    # in the same security, and the real Jamf's 1,388 bars from 07-22 belong to nobody. The
    # gap is measured on traded bars: 01-02 -> 07-22 (the filler in between has volume 0).
    master, conflicts = o_built

    assert len(segments(master, "JAMF")) == 2
    placeholder, jamf = segments(master, "JAMF")
    assert (placeholder["valid_from"], placeholder["valid_to"]) == (
        date(2020, 1, 2),
        O_JAMF_VACATED,
    )
    assert placeholder["security_id"] == only_sid(master, "JNMF")
    assert (jamf["valid_from"], jamf["valid_to"]) == (O_JAMF_IPO, None)
    assert jamf["cik"] == O_JAMF_CIK
    assert jamf["evidence"] == "sec"
    assert jamf["security_id"] != placeholder["security_id"]
    for day in (O_JAMF_IPO, date(2020, 8, 3), date(2021, 1, 4), date(2021, 12, 31)):
        assert segment_at(master, "JAMF", day) == jamf, day
    assert covering(master, "JAMF", date(2020, 4, 1)) == []  # the filler owns nothing
    assert not any("JAMF" in r["symbols"] for r in conflicts.to_pylist())


def test_o_intl_resumed_bars_open_a_bars_only_security_not_the_old_holders(
    o_built: tuple[pa.Table, pa.Table],
) -> None:
    # O, evidence 'bars_only'. The ETF under INTL never reaches the SEC table; the StoneX rows
    # (to 2021-01-04) sit BEFORE the resumed piece and must not be inherited through the
    # "SEC rows before valid_from" fallback of _segment_cik, which today would hand the ETF
    # StoneX's CIK 913760.
    master, conflicts = o_built

    assert len(segments(master, "INTL")) == 2
    stonex, etf = segments(master, "INTL")
    assert (stonex["valid_from"], stonex["valid_to"]) == (date(2020, 1, 2), O_SNEX_RECORD)
    assert stonex["cik"] == 913760
    assert stonex["security_id"] == only_sid(master, "SNEX")
    assert (etf["valid_from"], etf["valid_to"]) == (O_INTL_RESUMED, None)
    assert etf["cik"] is None
    assert etf["evidence"] == "bars_only"
    assert etf["security_id"] != stonex["security_id"]
    assert segment_at(master, "INTL", date(2023, 1, 3)) == etf
    assert covering(master, "INTL", date(2021, 6, 1)) == []
    assert covering(master, "SNEX", date(2020, 3, 2)) == []  # relabelled history, as before
    assert not any(
        set(r["symbols"]) == {"INTL", "SNEX"} for r in conflicts_of(conflicts, "dup_unlinked")
    )


def test_o_resumed_segment_is_not_knowable_before_its_first_bar(
    o_built: tuple[pa.Table, pa.Table],
) -> None:
    # Leakage: the resumed piece exists because of its first bar, so it is knowable at that
    # bar's availability and not a microsecond earlier; the placeholder's end stays knowable
    # at the vacate record (01-13), untouched by the later piece.
    master, _ = o_built

    placeholder, jamf = segments(master, "JAMF")
    assert jamf["available_at"] == bar_available_at(O_JAMF_IPO)
    assert placeholder["end_available_at"] == et_midnight(O_JAMF_VACATED)
    just_before = bar_available_at(O_JAMF_IPO) - MICRO
    before = [r for r in visible_at(master, just_before) if r["symbol"] == "JAMF"]
    assert [r["valid_from"] for r in before] == [date(2020, 1, 2)]
    at_bar = [r for r in visible_at(master, bar_available_at(O_JAMF_IPO)) if r["symbol"] == "JAMF"]
    assert sorted(r["valid_from"] for r in at_bar) == [date(2020, 1, 2), O_JAMF_IPO]
    _, etf = segments(master, "INTL")
    assert etf["available_at"] == bar_available_at(O_INTL_RESUMED)
    assert_nothing_known_before_its_valid_from(master)
    assert_end_knowability_invariant(master)


def test_o_bars_that_continue_without_a_gap_after_the_vacate_open_nothing() -> None:
    # O boundary (passes today, must keep passing; the audit's N2 shape): OLDN keeps trading
    # right through its rename to NEWN. No gap, no resumed piece: the later bars stay orphans.
    record = date(2021, 3, 1)
    master, _ = build_security_master(
        bars=bars_table(
            bar_rows("OLDN", date(2020, 1, 2), date(2021, 12, 31), 8.0)
            + bar_rows("NEWN", record, date(2021, 12, 31), 8.1)
        ),
        sec_history=sec_table([], []),
        name_changes=name_changes_table([("OLDN", "NEWN", record, "000500100", "000500200")]),
    )

    assert len(segments(master, "OLDN")) == 1
    [oldn] = segments(master, "OLDN")
    assert (oldn["valid_from"], oldn["valid_to"]) == (date(2020, 1, 2), record)
    assert oldn["security_id"] == only_sid(master, "NEWN")
    assert covering(master, "OLDN", date(2021, 6, 1)) == []


O_GAP_LAST = date(2021, 2, 25)  # a Thursday; the record (03-01) comes four days later


@pytest.mark.parametrize(
    ("gap_days", "resumes"),
    [(14, False), (15, True)],
    ids=["fourteen_days", "fifteen_days"],
)
def test_o_resume_gap_boundary(gap_days: int, resumes: bool) -> None:
    # O boundary (constructed): "more than RESUME_GAP_DAYS" means first_resumed - last_traded
    # > 14; at exactly 14 the bars are a continuation (orphans), at 15 a new bars_only piece.
    record = date(2021, 3, 1)
    resumed = O_GAP_LAST + timedelta(days=gap_days)
    master, _ = build_security_master(
        bars=bars_table(
            bar_rows("OLDG", date(2020, 1, 2), O_GAP_LAST, 12.0)
            + bar_rows("OLDG", resumed, date(2021, 12, 31), 12.2)
            + bar_rows("NEWG", record, date(2021, 12, 31), 12.1)
        ),
        sec_history=sec_table([], []),
        name_changes=name_changes_table([("OLDG", "NEWG", record, "000600100", "000600200")]),
    )

    first = segments(master, "OLDG")[0]
    assert (first["valid_from"], first["valid_to"]) == (date(2020, 1, 2), record)
    if not resumes:
        assert len(segments(master, "OLDG")) == 1
        assert covering(master, "OLDG", resumed) == []
    else:
        assert len(segments(master, "OLDG")) == 2
        second = segments(master, "OLDG")[1]
        assert (second["valid_from"], second["valid_to"]) == (resumed, None)
        assert second["evidence"] == "bars_only" and second["cik"] is None
        assert second["security_id"] != first["security_id"]
        assert second["available_at"] == bar_available_at(resumed)


def test_o_resumed_piece_ends_where_the_next_piece_of_the_label_starts() -> None:
    # O, the closed variant: OLDR vacated 2020-06-01 (-> NEWR), its bars resume 08-03 at another
    # level, and TAKR takes the label on 2021-06-01. The resumed piece runs to the takeover,
    # its end knowable with it; three securities touch the label.
    vacated, taken = date(2020, 6, 1), date(2021, 6, 1)
    resumed = date(2020, 8, 3)
    master, conflicts = build_security_master(
        bars=bars_table(
            bar_rows("OLDR", date(2020, 1, 2), date(2020, 5, 29), 10.0)
            + bar_rows("NEWR", vacated, date(2021, 12, 31), 10.1)
            + bar_rows("OLDR", resumed, date(2021, 5, 28), 4.0)  # distinct level: no continuity
            + bar_rows("TAKR", date(2020, 1, 2), date(2021, 5, 28), 30.0)
            + bar_rows("OLDR", taken, date(2021, 12, 31), 30.5)
        ),
        sec_history=sec_table([], []),
        name_changes=name_changes_table(
            [
                ("OLDR", "NEWR", vacated, "000700100", "000700200"),
                ("TAKR", "OLDR", taken, "000800100", "000800100"),
            ]
        ),
    )

    assert len(segments(master, "OLDR")) == 3
    first, middle, last = segments(master, "OLDR")
    assert (first["valid_from"], first["valid_to"]) == (date(2020, 1, 2), vacated)
    assert first["security_id"] == only_sid(master, "NEWR")
    assert (middle["valid_from"], middle["valid_to"]) == (resumed, taken)
    assert middle["evidence"] == "bars_only"
    assert middle["available_at"] == bar_available_at(resumed)
    # first-bar rule: the end stays at the takeover, the taker waits for its first bar
    assert middle["end_available_at"] == et_midnight(taken)
    assert last["available_at"] == bar_available_at(taken)
    assert (last["valid_from"], last["valid_to"]) == (taken, None)
    assert last["security_id"] == only_sid(master, "TAKR")
    assert len({first["security_id"], middle["security_id"], last["security_id"]}) == 3
    assert not any(
        "OLDR" in r["symbols"] for r in conflicts_of(conflicts, "reuse_looks_continuous")
    )
    assert_end_knowability_invariant(master)
    assert_nothing_known_before_its_valid_from(master)


# =============================================================================================
# A: when-issued edges cut but never link
# =============================================================================================


@pytest.mark.parametrize("symbol", ["HONAV", "PRG.WI", "RIV.RTWI", "ARNC.WI", "CXAIV"])
def test_a_when_issued_pattern_matches(symbol: str) -> None:
    assert sm.WHEN_ISSUED.fullmatch(symbol), symbol


@pytest.mark.parametrize("symbol", ["HONA", "GOOGL", "BRK.B", "ARNC", "VTIQ"])
def test_a_when_issued_pattern_leaves_regular_symbols_alone(symbol: str) -> None:
    assert sm.WHEN_ISSUED.fullmatch(symbol) is None, symbol


A_HONA_DAY = date(2026, 6, 29)  # real: HONAV -> HONA process_date and HONA's first own bar
A_HONAV_FIRST, A_HONAV_LAST = date(2026, 6, 15), date(2026, 6, 26)  # real
A_HONA_CIK = 2089271  # real
A_HONA_SNAPSHOTS = [date(2026, 3, 29), date(2026, 6, 4), date(2026, 9, 8)]
A_HONA_LIVES: list[Life] = [("HONA", A_HONA_CIK, date(2026, 3, 29), None)]  # real: listed
#   months before the when-issued days, so an opening HONA piece exists and must be dropped as
#   relabelled (its bars are HONAV's copies)
A_HONA_RENAMES: list[Rename] = [("HONAV", "HONA", A_HONA_DAY, "438100100", "438100100")]  # CUSIPs
#   constructed


def a_hona_bars() -> list[dict[str, Any]]:
    honav = [
        r
        for r in bar_rows("HONAV", A_HONAV_FIRST, A_HONAV_LAST, 200.0)
        if r["session_date"] != date(2026, 6, 19)  # Juneteenth: 9 bars, as in the store
    ]
    own = bar_rows("HONA", A_HONA_DAY, date(2026, 10, 5), 230.0)
    return [*honav, *relabel(honav, "HONA"), *own]


@pytest.fixture(scope="module")
def a_hona_built() -> tuple[pa.Table, pa.Table]:
    return build_security_master(
        bars=bars_table(a_hona_bars()),
        sec_history=sec_table(A_HONA_SNAPSHOTS, A_HONA_LIVES),
        name_changes=name_changes_table(A_HONA_RENAMES),
    )


def test_a_when_issued_symbol_is_its_own_security(a_hona_built: tuple[pa.Table, pa.Table]) -> None:
    # A. Today (real master): HONAV [06-15, 06-29) and HONA [06-29, None) share one security.
    # The edge still ends HONAV's piece and starts HONA's on the record day; HONA's copies of
    # the when-issued days are dropped as relabelled, as before.
    master, _ = a_hona_built

    assert len(segments(master, "HONA")) == 1
    [hona] = segments(master, "HONA")
    assert (hona["valid_from"], hona["valid_to"]) == (A_HONA_DAY, None)
    assert hona["cik"] == A_HONA_CIK
    assert hona["evidence"] == "name_change"
    assert len(segments(master, "HONAV")) == 1
    [honav] = segments(master, "HONAV")
    assert (honav["valid_from"], honav["valid_to"]) == (A_HONAV_FIRST, A_HONA_DAY)
    assert honav["security_id"] != hona["security_id"]
    assert covering(master, "HONA", date(2026, 6, 22)) == []  # the copies own nothing
    assert covering(master, "HONAV", date(2026, 6, 22)) == [honav]


def test_a_when_issued_pair_is_not_reported_as_an_unlinked_duplicate(
    a_hona_built: tuple[pa.Table, pa.Table],
) -> None:
    # Consequence of A (not in the rule's wording; drop if the author disagrees): the 9
    # identical days are explained by the record, so there is nothing for a human to decide.
    # Today the pair shares a security and is not reported; with A it would become
    # dup_unlinked unless the conflict check also treats the when-issued edge as the link.
    _, conflicts = a_hona_built

    assert not any(
        set(r["symbols"]) == {"HONA", "HONAV"} for r in conflicts_of(conflicts, "dup_unlinked")
    )


def test_a_hona_segment_is_knowable_at_the_record_and_honav_ends_then(
    a_hona_built: tuple[pa.Table, pa.Table],
) -> None:
    # Leakage: a reader on 06-20 sees HONAV open and no HONA; at midnight 06-29 HONAV's end
    # becomes knowable, HONA at its first bar that evening (first-bar rule, 2026-10-19).
    master, _ = a_hona_built

    [hona] = segments(master, "HONA")
    [honav] = segments(master, "HONAV")
    # first-bar rule: the end stays at the record, HONA waits for its first bar
    assert honav["end_available_at"] == et_midnight(A_HONA_DAY)
    assert hona["available_at"] == bar_available_at(A_HONA_DAY)
    seen = visible_at(master, et_midnight(date(2026, 6, 20)))
    assert "HONA" not in {r["symbol"] for r in seen}
    [honav_seen] = [r for r in seen if r["symbol"] == "HONAV"]
    assert honav_seen["end_available_at"] > et_midnight(date(2026, 6, 20))
    # first-bar rule: HONA appears with its first bar, not at the record's midnight
    assert "HONA" not in {r["symbol"] for r in visible_at(master, et_midnight(A_HONA_DAY))}
    assert "HONA" in {r["symbol"] for r in visible_at(master, bar_available_at(A_HONA_DAY))}
    assert_nothing_known_before_its_valid_from(master)
    assert_end_knowability_invariant(master)


A_AAN_DAY = date(2020, 12, 1)  # real: both records' process_date and the spin-off day
A_AAN_SNAPSHOTS = [date(2020, 10, 6), date(2020, 11, 2), A_AAN_DAY, date(2021, 1, 4)]
A_OLD_AAN_CIK, A_NEW_AAN_CIK = 706688, 1821393  # real: Aaron's Holdings, The Aaron's Company
A_AAN_LIVES: list[Life] = [
    ("AAN", A_OLD_AAN_CIK, date(2020, 10, 6), date(2020, 11, 2)),
    ("PRG", A_OLD_AAN_CIK, A_AAN_DAY, None),  # SIMPLIFIED: real PRG sits under 1808834
    ("AAN", A_NEW_AAN_CIK, A_AAN_DAY, None),
]
A_AAN_RENAMES: list[Rename] = [
    ("AAN.WI", "AAN", A_AAN_DAY, "002535300", "002535300"),  # CUSIPs constructed
    ("PRG.WI", "PRG", A_AAN_DAY, "74319R101", "74319R101"),
]
A_WI_DAYS = [date(2020, 11, 25), date(2020, 11, 27), date(2020, 11, 30)]  # real: 3 bars each


def a_aan_bars() -> list[dict[str, Any]]:
    def when_issued(symbol: str, base: float) -> list[dict[str, Any]]:
        rows = bar_rows(symbol, A_WI_DAYS[0], A_WI_DAYS[-1], base)
        return [r for r in rows if r["session_date"] in A_WI_DAYS]  # 11-26: Thanksgiving

    return [
        *bar_rows("AAN", date(2020, 1, 2), date(2020, 11, 30), 62.0),  # Aaron's Holdings
        *bar_rows("AAN", A_AAN_DAY, date(2021, 12, 31), 18.4),  # The Aaron's Company (3.4x drop)
        *bar_rows("PRG", A_AAN_DAY, date(2021, 12, 31), 55.6),  # PROG Holdings
        *when_issued("AAN.WI", 22.0),
        *when_issued("PRG.WI", 53.0),
    ]


@pytest.fixture(scope="module")
def a_aan_built() -> tuple[pa.Table, pa.Table]:
    return build_security_master(
        bars=bars_table(a_aan_bars()),
        sec_history=sec_table(A_AAN_SNAPSHOTS, A_AAN_LIVES),
        name_changes=name_changes_table(A_AAN_RENAMES),
    )


def test_a_aan_spin_off_cuts_the_label_but_chains_neither_when_issued_symbol(
    a_aan_built: tuple[pa.Table, pa.Table],
) -> None:
    # A, the two-sided real case. The old holder leaves AAN for PRG (SEC edge, dated by the
    # same-day Alpaca records) while AAN.WI -> AAN starts the new holder's piece. Today (real
    # master): AAN.WI sits in the new AAN's security and PRG.WI in PRG's.
    master, conflicts = a_aan_built

    old = segment_at(master, "AAN", date(2020, 6, 1))
    new = segment_at(master, "AAN", date(2021, 3, 1))
    assert (old["valid_to"], old["cik"]) == (A_AAN_DAY, A_OLD_AAN_CIK)
    assert old["security_id"] == only_sid(master, "PRG")
    assert new["valid_from"] == A_AAN_DAY
    assert new["cik"] == A_NEW_AAN_CIK
    assert new["security_id"] != old["security_id"]
    assert segment_at(master, "AAN", A_AAN_DAY)["security_id"] == new["security_id"]
    # first-bar rule: the end stays at the records, the new holder waits for a bar
    assert old["end_available_at"] == et_midnight(A_AAN_DAY)
    assert new["available_at"] == bar_available_at(A_AAN_DAY)

    [aan_wi] = segments(master, "AAN.WI")
    [prg_wi] = segments(master, "PRG.WI")
    assert (aan_wi["valid_from"], aan_wi["valid_to"]) == (A_WI_DAYS[0], A_AAN_DAY)
    assert (prg_wi["valid_from"], prg_wi["valid_to"]) == (A_WI_DAYS[0], A_AAN_DAY)
    assert aan_wi["security_id"] not in {old["security_id"], new["security_id"]}
    assert prg_wi["security_id"] not in {old["security_id"], new["security_id"]}
    assert aan_wi["security_id"] != prg_wi["security_id"]
    assert sids(master, "AAN") == {old["security_id"], new["security_id"]}
    assert conflicts_of(conflicts, "cik_switch_discontinuous") == []
    assert_end_knowability_invariant(master)
    assert_nothing_known_before_its_valid_from(master)


# ---------------------------------------------------------------------------------------------
# A, the chain shapes found on the first real rebuild (2026-10-17): Alpaca routes real renames
# THROUGH a when-issued name. "Cut but never link" would split RBS from NWG and GSK from itself;
# the when-issued name has to be transparent instead.
# ---------------------------------------------------------------------------------------------

A_NWG_IN, A_NWG_OUT = date(2022, 8, 30), date(2022, 9, 1)  # real: RBS -> NWG.WI, NWG.WI -> NWG
A_NWG_SNAPSHOTS = [date(2022, 1, 3), date(2022, 8, 1), date(2022, 10, 3), date(2023, 1, 3)]
A_NWG_LIVES: list[Life] = [
    ("RBS", 844150, date(2022, 1, 3), date(2022, 8, 1)),  # real CIK (NatWest Group plc)
    ("NWG", 844150, date(2022, 10, 3), None),
]


def test_a_rename_routed_through_a_when_issued_name_is_one_security() -> None:
    # real shape: NatWest's ADR went RBS -> NWG.WI (08-30) -> NWG (09-01); NWG.WI traded two
    # sessions (levels ASSUMED). RBS and NWG are one security; NWG.WI is its own, chained to
    # nobody; RBS is not vacated on 08-30 (its last bar is 08-31).
    master, conflicts = build_security_master(
        bars=bars_table(
            bar_rows("RBS", date(2022, 1, 3), date(2022, 8, 31), 6.0)
            + bar_rows("NWG.WI", A_NWG_IN, date(2022, 8, 31), 6.0)
            + bar_rows("NWG", A_NWG_OUT, date(2022, 12, 30), 6.05)
        ),
        sec_history=sec_table(A_NWG_SNAPSHOTS, A_NWG_LIVES),
        name_changes=name_changes_table(
            [
                ("RBS", "NWG.WI", A_NWG_IN, "780097838", "780097838"),
                ("NWG.WI", "NWG", A_NWG_OUT, "780097838", "639081108"),
            ]
        ),
    )

    [rbs] = segments(master, "RBS")
    [nwg] = segments(master, "NWG")
    [wi] = segments(master, "NWG.WI")
    assert rbs["security_id"] == nwg["security_id"]
    assert rbs["valid_to"] == nwg["valid_from"] == A_NWG_OUT
    assert nwg["evidence"] == "both" and nwg["cusip"] == "639081108"  # the SEC saw it too
    # first-bar rule: the end stays at the record, NWG waits for its first bar
    assert rbs["end_available_at"] == et_midnight(A_NWG_OUT)
    assert nwg["available_at"] == bar_available_at(A_NWG_OUT)
    assert segment_at(master, "RBS", date(2022, 8, 31)) == rbs  # not vacated by the edge in
    assert (wi["valid_from"], wi["valid_to"]) == (A_NWG_IN, A_NWG_OUT)
    assert wi["security_id"] != rbs["security_id"]
    assert conflicts.num_rows == 0
    assert_end_knowability_invariant(master)
    assert_nothing_known_before_its_valid_from(master)


def test_a_round_trip_through_a_when_issued_name_leaves_the_label_continuous() -> None:
    # real shape: GSK's ADR ratio change, GSK -> GSK.WI (2022-07-19) -> GSK (07-22); GSK.WI
    # traded three sessions (levels ASSUMED). GSK is one open segment from its first bar.
    master, conflicts = build_security_master(
        bars=bars_table(
            bar_rows("GSK", date(2022, 1, 3), date(2022, 12, 30), 40.0)
            + bar_rows("GSK.WI", date(2022, 7, 19), date(2022, 7, 21), 33.0)
        ),
        sec_history=sec_table(
            [date(2022, 1, 3), date(2023, 1, 3)], [("GSK", 1131399, date(2022, 1, 3), None)]
        ),
        name_changes=name_changes_table(
            [
                ("GSK", "GSK.WI", date(2022, 7, 19), "37733W105", "37733W105"),
                ("GSK.WI", "GSK", date(2022, 7, 22), "37733W105", "37733W204"),
            ]
        ),
    )

    [gsk] = segments(master, "GSK")
    [wi] = segments(master, "GSK.WI")
    assert (gsk["valid_from"], gsk["valid_to"]) == (date(2022, 1, 3), None)
    assert gsk["cik"] == 1131399 and gsk["evidence"] == "sec"
    assert (wi["valid_from"], wi["valid_to"]) == (date(2022, 7, 19), date(2022, 7, 22))
    assert wi["security_id"] != gsk["security_id"]
    assert conflicts.num_rows == 0


def test_v_bankruptcy_suffix_record_is_kept_and_the_label_is_vacated() -> None:
    # V exception (reviewer 2026-10-17, no real instance yet: the six real XQ -> X records have
    # no bars after the record). ASSUMPTION on the shape: OLDQ never traded here, the label
    # resumes 90 days later with new equity at another level, same CIK kept through the
    # bankruptcy. The record stays an event: the old equity ends there and the resumed bars
    # are a new security (rule O), not a continuation.
    record, resumed = date(2021, 3, 1), date(2021, 6, 1)
    master, _ = build_security_master(
        bars=bars_table(
            bar_rows("BKRP", date(2020, 1, 2), date(2020, 11, 30), 9.0)
            + bar_rows("BKRP", resumed, date(2021, 12, 31), 25.0)
        ),
        sec_history=sec_table(
            [date(2020, 1, 13), date(2021, 1, 4), date(2021, 7, 5)],
            [("BKRP", 5005, date(2020, 1, 13), None)],
        ),
        name_changes=name_changes_table([("BKRPQ", "BKRP", record, "000900100", "000900200")]),
    )

    assert sm.BANKRUPT.fullmatch("BKRPQ") and sm.BANKRUPT.fullmatch("BKRP") is None
    assert len(sids(master, "BKRP")) == 2
    old = segment_at(master, "BKRP", date(2020, 6, 1))
    new = segment_at(master, "BKRP", date(2021, 9, 1))
    assert old["security_id"] != new["security_id"]
    assert new["valid_from"] == record and new["evidence"] == "name_change"
