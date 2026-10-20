"""Guards against a security-master segment that is knowable before the build could have
produced it: a segment exists only if it has a bar of its own in [valid_from, valid_to), and a
daily bar is knowable only at 20:00 New York of its session, so a segment stamped knowable at the
instant of the event that created it (00:00 New York of a rename's day or of the SEC snapshot
showing a new CIK) is shown to readers up to a session too early -- or months too early when the
first bar comes long after the record.

Rule these tests pin (decided 2026-10-19):

* every output segment: ``available_at = max(event instant as today, available_at of the
  segment's first own bar in [valid_from, valid_to))``, applied to the final segments (after
  takeovers, relabelling, clipping, resumed pieces, overrides) and BEFORE a segment's CIK is
  filled from its chain, so a chain member knowable after the segment is no donor;
* ``cik_available_at`` stays ``max(cik_since, available_at)`` and so moves with it;
* the previous segment's ``end_available_at`` stays at the event instant. Between the event and
  the successor's first bar a reader sees the old segment closed and the new one not at all
  (a knowability gap, decided 2026-10-19). ``check_master``'s adjacent-segment rule becomes:
  a violation iff the successor is knowable BEFORE the predecessor's end.

Fake data provenance:
- Real (found by ``asof market truncation`` on the real store): CXAIV -> CXAI record effective
  2023-03-15 (available 2023-03-15 04:00 UTC), CXAI's 2023-03-15 bar available 2023-03-16
  00:00 UTC; ABAT listed under a new CIK by the SEC snapshot of 2023-10-03 (old CIK's piece
  2023-09-21 .. 2023-10-03), the new piece's first bar 2023-10-03 available 2023-10-04 00:00
  UTC; AACG listed by the SEC from snapshot 2019-12-03, first bar 2020-01-02 (available
  2020-01-03 01:00 UTC); TSLF -> FSLF record 2020-06-08, first FSLF bar 2021-01-04. The worlds
  below copy those dates; tickers, CIKs, CUSIPs and prices are constructed (OLDN / NEWN stand
  for CXAIV / CXAI without the when-issued spelling, which other rules treat specially; ATAI as
  AACG's earlier ticker is constructed).
- A daily bar is knowable from 20:00 New York of its session (ingest.bars, real: the 2020-01-02
  bar has available_at 2020-01-03 01:00 UTC); an Alpaca name change from 00:00 New York of its
  process_date (real, probe 2026-10-06).
- ASSUMPTION (as in tests/ingest/test_master_truncation.py): an SEC ticker snapshot is knowable
  from 00:00 New York of its snapshot_date; a human override from 00:00 New York of its date.
- The chain-fill world (PNEW / QNEW renamed on one day and merged by a dated override) is
  constructed: no real pair of this shape is in hand.

Real-data checks for the assumptions (audit command, then the standard cases):
- ``asof market truncation`` on the real store: the only_in_full leaks CXAI:2023-03-15,
  ABAT:2023-10-03, AACG:2019-12-03 and FSLF:2020-06-08 no longer occur.
- Standard cases: ABAT seen at 2023-10-03 12:00 UTC shows the old-CIK piece closed at 2023-10-03
  and no new piece; seen at 2023-10-04 00:00 UTC both. CXAI seen at 2023-03-15 12:00 UTC has no
  row; at 2023-03-16 00:00 UTC it has one.
"""

import bisect
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from functools import cache
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pytest

from asof.ingest import security_master as sm
from asof.ingest.master_truncation import compare, mask_master
from asof.ingest.security_master import build_security_master, write_master
from asof.store.market import market_root, visible_master
from tests.ingest import test_security_master as base
from tests.ingest.master_scenarios import SCENARIOS, Inputs
from tests.ingest.test_master_truncation import build, build_truncated
from tests.ingest.test_security_master import (
    Life,
    Rename,
    bar_available_at,
    bar_rows,
    bars_table,
    et_midnight,
    master_table,
    name_changes_table,
    sec_table,
    seg_row,
    segments,
)

MICRO = timedelta(microseconds=1)


def inputs(
    bars: list[dict[str, Any]],
    snapshots: Sequence[date] = (),
    lives: Sequence[Life] = (),
    renames: Sequence[Rename] = (),
) -> Inputs:
    return Inputs(bars_table(bars), sec_table(snapshots, lives), name_changes_table(renames))


def rows_of(table: pa.Table, symbol: str) -> list[dict[str, Any]]:
    return sorted(
        (r for r in table.to_pylist() if r["symbol"] == symbol), key=lambda r: r["valid_from"]
    )


def leaks(full: pa.Table, inp: Inputs, as_of: datetime) -> set[tuple[str, str, date]]:
    """Leak rows of the truncation comparison at ``as_of``: what the full master shows then and
    a build from the inputs knowable then cannot."""
    diff = compare(full, build_truncated(inp, as_of), as_of)
    return {
        (r["reason"], r["symbol"], r["valid_from"]) for r in diff.to_pylist() if r["kind"] == "leak"
    }


# =============================================================================================
# worlds
# =============================================================================================

# CXAI shape: a rename record dated D, the new name's first bar on D itself.
RN_DAY = date(2023, 3, 15)  # real CXAIV -> CXAI effective day (EDT: midnight is 04:00 UTC)
RN_FIRST_KNOWN = datetime(2023, 3, 16, 0, 0, tzinfo=UTC)  # real: CXAI's first bar's available_at


def rename_world() -> Inputs:
    return inputs(
        bar_rows("OLDN", date(2023, 1, 3), date(2023, 3, 14), 4.0)
        + bar_rows("NEWN", RN_DAY, date(2023, 6, 30), 4.01),
        renames=[("OLDN", "NEWN", RN_DAY, "000900100", "000900100")],
    )


# ABAT shape: the SEC snapshot of D lists the ticker under a new CIK; prices continuous, so the
# two pieces are one security (bridged) -- the rule is about knowability, not grouping.
CS_DAY = date(2023, 10, 3)  # real snapshot listing ABAT under the new CIK
CS_FIRST_KNOWN = datetime(2023, 10, 4, 0, 0, tzinfo=UTC)  # real: first new-piece bar's available_at
CS_SNAPSHOTS = [date(2023, 7, 3), date(2023, 9, 21), CS_DAY, date(2023, 11, 1)]
CS_LIVES: list[Life] = [
    ("ABAT", 1000101, date(2023, 7, 3), date(2023, 9, 21)),
    ("ABAT", 1000102, CS_DAY, None),
]


def cik_switch_world() -> Inputs:
    return inputs(
        bar_rows("ABAT", date(2023, 7, 3), date(2023, 12, 29), 4.0), CS_SNAPSHOTS, CS_LIVES
    )


# AACG shape: an SEC edge (constructed old ticker ATAI, same CIK) starts AACG at the 2019-12-03
# snapshot, a month before the store's first bar.
AA_SNAPSHOT = date(2019, 12, 3)  # real: first snapshot listing AACG
AA_FIRST_BAR = date(2020, 1, 2)  # real: first AACG bar (the store's first session)
AA_SNAPSHOTS = [date(2019, 11, 1), AA_SNAPSHOT, date(2020, 1, 13)]
AA_LIVES: list[Life] = [
    ("ATAI", 1000201, date(2019, 11, 1), date(2019, 11, 1)),
    ("AACG", 1000201, AA_SNAPSHOT, None),
]


def aacg_world() -> Inputs:
    return inputs(bar_rows("AACG", AA_FIRST_BAR, date(2020, 6, 30), 2.0), AA_SNAPSHOTS, AA_LIVES)


# FSLF shape: the new name's first bar comes seven months after the record.
FS_RECORD = date(2020, 6, 8)  # real TSLF -> FSLF process_date
FS_FIRST_BAR = date(2021, 1, 4)  # real first FSLF bar


def fslf_world() -> Inputs:
    return inputs(
        bar_rows("TSLF", date(2020, 1, 2), date(2020, 6, 5), 9.0)
        + bar_rows("FSLF", FS_FIRST_BAR, date(2021, 6, 30), 9.5),
        renames=[("TSLF", "FSLF", FS_RECORD, "000910100", "000910100")],
    )


# Chain fill (constructed): XOLD -> PNEW and ROLD -> QNEW on one day, a dated human merge joins
# PNEW and QNEW. Only QNEW reaches the SEC table (CIK 9 from the same day's snapshot). PNEW trades
# on the record day, QNEW only from the next session, so QNEW is knowable one session after PNEW
# and must not lend PNEW its CIK in between.
CH_DAY = date(2021, 3, 1)  # a Monday
CH_Q_FIRST = date(2021, 3, 2)
CH_CIK = 9
CH_MERGE = [{"merge": ["PNEW", "QNEW"], "reason": "same company", "date": CH_DAY}]


def chain_world() -> Inputs:
    return inputs(
        bar_rows("XOLD", date(2020, 6, 1), date(2021, 2, 26), 3.0)
        + bar_rows("ROLD", date(2020, 6, 1), date(2021, 2, 26), 8.0)
        + bar_rows("PNEW", CH_DAY, date(2021, 6, 30), 3.01)
        + bar_rows("QNEW", CH_Q_FIRST, date(2021, 6, 30), 8.01),
        [date(2021, 1, 4), CH_DAY, date(2021, 4, 1)],
        [("QNEW", CH_CIK, CH_DAY, None)],
        [
            ("XOLD", "PNEW", CH_DAY, "000920100", "000920100"),
            ("ROLD", "QNEW", CH_DAY, "000930100", "000930100"),
        ],
    )


def built(world: Callable[[], Inputs], overrides: Sequence[Mapping[str, Any]] = ()) -> pa.Table:
    return build(world(), overrides)


# =============================================================================================
# normal path
# =============================================================================================


def test_rename_segment_is_knowable_at_its_first_bar_and_the_old_end_at_the_record() -> None:
    master = built(rename_world)
    [old] = segments(master, "OLDN")
    [new] = segments(master, "NEWN")

    assert old["valid_to"] == new["valid_from"] == RN_DAY  # the dates do not move
    assert new["available_at"] == bar_available_at(RN_DAY) == RN_FIRST_KNOWN
    assert old["end_available_at"] == et_midnight(RN_DAY) == datetime(2023, 3, 15, 4, tzinfo=UTC)
    assert new["available_at"] > old["end_available_at"]  # the allowed gap


def test_cik_switch_segment_is_knowable_at_its_first_bar_and_the_old_end_at_the_snapshot() -> None:
    master = built(cik_switch_world)
    old, new = segments(master, "ABAT")

    assert old["valid_to"] == new["valid_from"] == CS_DAY
    assert (old["cik"], new["cik"]) == (1000101, 1000102)
    assert new["available_at"] == bar_available_at(CS_DAY) == CS_FIRST_KNOWN
    assert old["end_available_at"] == et_midnight(CS_DAY)
    assert new["cik_available_at"] == new["available_at"]  # moves with it (snapshot was earlier)


def test_world_rename_and_cik_switch_rows_follow_the_rule() -> None:
    # The scenario world's BK -> BNY rename (2026-05-21) and the GOOG / DISC CIK switches: the
    # new piece at its first bar, the old piece's end still at the event's midnight.
    master = build(SCENARIOS["world"]())

    [bk] = segments(master, "BK")
    [bny] = segments(master, "BNY")
    assert bny["available_at"] == bar_available_at(date(2026, 5, 21))
    assert bk["end_available_at"] == et_midnight(date(2026, 5, 21))
    for symbol, day in (("GOOG", date(2019, 7, 1)), ("DISC", date(2024, 6, 3))):
        before, after = segments(master, symbol)
        assert after["valid_from"] == day, symbol
        assert after["available_at"] == bar_available_at(day), symbol
        assert before["end_available_at"] == et_midnight(day), symbol


def test_event_later_than_the_first_bar_keeps_the_event_instant() -> None:
    # max, not replace: VTIQ -> NKLA (F9, real shape) starts at NKLA's first bar 06-04, but the
    # record is dated 06-08 -- the segment stays knowable at the record's midnight.
    master = build(SCENARIOS["f9"]())

    [nkla] = segments(master, "NKLA")
    assert nkla["valid_from"] == base.F9_NKLA_FIRST_BAR
    assert nkla["available_at"] == et_midnight(base.F9_ALPACA_DAY)
    assert nkla["available_at"] > bar_available_at(base.F9_NKLA_FIRST_BAR)


def test_opening_segment_without_an_event_is_knowable_at_its_first_bar() -> None:
    master = built(rename_world)
    [old] = segments(master, "OLDN")
    assert old["available_at"] == bar_available_at(date(2023, 1, 3))

    world = build(SCENARIOS["world"]())
    [spy] = segments(world, "SPY")
    assert spy["available_at"] == bar_available_at(date(2020, 1, 2))


# =============================================================================================
# boundaries
# =============================================================================================


def test_sec_listing_before_the_first_bar_is_knowable_at_the_first_bar() -> None:
    # AACG: the segment still starts at the snapshot (the rule moves knowability, not dates), but
    # nobody can see it before its first bar is final.
    master = built(aacg_world)

    [aacg] = segments(master, "AACG")
    assert aacg["valid_from"] == AA_SNAPSHOT
    assert aacg["available_at"] == bar_available_at(AA_FIRST_BAR)
    assert aacg["available_at"] == datetime(2020, 1, 3, 1, 0, tzinfo=UTC)
    assert aacg["cik"] == 1000201
    assert aacg["cik_available_at"] == aacg["available_at"]


def test_first_bar_months_after_the_record_moves_availability_by_months() -> None:
    master = built(fslf_world)

    [tslf] = segments(master, "TSLF")
    [fslf] = segments(master, "FSLF")
    assert tslf["valid_to"] == fslf["valid_from"] == FS_RECORD
    assert tslf["end_available_at"] == et_midnight(FS_RECORD)
    assert fslf["available_at"] == bar_available_at(FS_FIRST_BAR)
    assert fslf["available_at"] == datetime(2021, 1, 5, 1, 0, tzinfo=UTC)


@cache
def scenario_master(name: str) -> tuple[Inputs, pa.Table]:
    inp = SCENARIOS[name]()
    return inp, build(inp)


def first_own_bar_known(bars: pa.Table) -> Callable[[Mapping[str, Any]], datetime | None]:
    """available_at of a segment's first bar in [valid_from, valid_to), or None."""
    by_symbol: dict[str, list[tuple[date, datetime]]] = {}
    for b in bars.select(["symbol", "session_date", "available_at"]).to_pylist():
        by_symbol.setdefault(b["symbol"], []).append((b["session_date"], b["available_at"]))
    days: dict[str, list[date]] = {}
    for symbol, rows in by_symbol.items():
        rows.sort()
        days[symbol] = [d for d, _ in rows]

    def first(row: Mapping[str, Any]) -> datetime | None:
        rows = by_symbol.get(row["symbol"], [])
        i = bisect.bisect_left(days.get(row["symbol"], []), row["valid_from"])
        if i == len(rows) or (row["valid_to"] is not None and rows[i][0] >= row["valid_to"]):
            return None
        return rows[i][1]

    return first


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_every_row_is_knowable_no_earlier_than_its_first_own_bar(scenario: str) -> None:
    # Property over every scenario world, relabelled / takeover-settled / clipped / resumed
    # pieces included: the build cannot produce a segment before one of its own bars is final.
    inp, master = scenario_master(scenario)
    first = first_own_bar_known(inp.bars)

    early = []
    for row in master.to_pylist():
        known = first(row)
        if known is not None and row["available_at"] < known:
            early.append((row["symbol"], row["valid_from"], row["available_at"], known))
    assert master.num_rows > 0
    assert early == [], early


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_cik_is_never_known_before_its_segment(scenario: str) -> None:
    _, master = scenario_master(scenario)
    for row in master.to_pylist():
        assert (row["cik"] is None) == (row["cik_available_at"] is None), row
        if row["cik"] is not None:
            assert row["cik_available_at"] >= row["available_at"], row


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_no_successor_is_knowable_before_its_predecessors_end(scenario: str) -> None:
    # Option 1: a gap in knowability between adjacent pieces of one label is allowed, the
    # reverse (the next owner shown while the previous one is still open) never; the built
    # master always passes check_master.
    _, master = scenario_master(scenario)
    rows = master.to_pylist()
    for a in rows:
        for b in rows:
            if a["symbol"] == b["symbol"] and a["valid_to"] is not None:
                if a["valid_to"] == b["valid_from"]:
                    assert b["available_at"] >= a["end_available_at"], (a, b)
    assert sm.check_master(master) is None


# =============================================================================================
# leakage: the future segment exists in the full master but is invisible until its first bar
# =============================================================================================


def test_reader_between_the_record_and_the_first_bar_sees_the_old_name_closed_and_no_new_name() -> (
    None
):
    master = built(rename_world)
    midnight = et_midnight(RN_DAY)

    for t in (midnight, midnight + timedelta(hours=8), RN_FIRST_KNOWN - MICRO):
        seen = mask_master(master, t)
        old = rows_of(seen, "OLDN")
        assert [(r["valid_from"], r["valid_to"]) for r in old] == [(date(2023, 1, 3), RN_DAY)], t
        assert rows_of(seen, "NEWN") == [], t
    for t in (RN_FIRST_KNOWN, RN_FIRST_KNOWN + MICRO):
        new = rows_of(mask_master(master, t), "NEWN")
        assert [(r["valid_from"], r["valid_to"]) for r in new] == [(RN_DAY, None)], t


def test_reader_between_the_snapshot_and_the_first_bar_sees_one_closed_abat_piece(
    tmp_path: Path,
) -> None:
    # On the read path strategies use (visible_master on the written file).
    inp = cik_switch_world()
    master, conflicts = build_security_master(
        bars=inp.bars, sec_history=inp.sec_history, name_changes=inp.name_changes
    )
    root = market_root(tmp_path)
    write_master(master, conflicts, root)
    con = duckdb.connect()
    try:
        between = visible_master(con, root, datetime(2023, 10, 3, 12, tzinfo=UTC)).to_pylist()
        abat = [(r["valid_from"], r["valid_to"]) for r in between if r["symbol"] == "ABAT"]
        assert abat == [(date(2023, 7, 3), CS_DAY)]
        eve = visible_master(con, root, CS_FIRST_KNOWN - MICRO).to_pylist()
        assert [r["valid_from"] for r in eve if r["symbol"] == "ABAT"] == [date(2023, 7, 3)]
        after = visible_master(con, root, CS_FIRST_KNOWN).to_pylist()
        abat = [(r["valid_from"], r["valid_to"]) for r in after if r["symbol"] == "ABAT"]
        assert abat == [(date(2023, 7, 3), CS_DAY), (CS_DAY, None)]
    finally:
        con.close()


@pytest.mark.parametrize(
    ("world", "event_day", "first_known"),
    [
        (rename_world, RN_DAY, RN_FIRST_KNOWN),
        (cik_switch_world, CS_DAY, CS_FIRST_KNOWN),
        (aacg_world, AA_SNAPSHOT, bar_available_at(AA_FIRST_BAR)),
        (fslf_world, FS_RECORD, bar_available_at(FS_FIRST_BAR)),
    ],
    ids=["rename_cxai", "cik_switch_abat", "sec_listing_aacg", "late_first_bar_fslf"],
)
def test_build_from_knowable_inputs_shows_nothing_the_full_master_shows_early(
    world: Callable[[], Inputs], event_day: date, first_known: datetime
) -> None:
    # The defect as the truncation test found it on real data: between the event's midnight and
    # the first bar's availability the full master showed a segment no build could produce yet.
    inp = world()
    full = build(inp)
    midnight = et_midnight(event_day)
    for t in (midnight, midnight + timedelta(hours=8), first_known - MICRO, first_known):
        assert leaks(full, inp, t) == set(), t


def test_chain_cik_fill_takes_no_donor_knowable_after_the_segment() -> None:
    # Before the rule both new pieces were knowable at the record's midnight, so QNEW (CIK 9)
    # lent PNEW its CIK. Moved to their first bars, QNEW is knowable a session after PNEW: a
    # reader on 03-02 at 02:00 UTC sees PNEW and must not learn CIK 9 from a piece it cannot see.
    master = build(chain_world(), CH_MERGE)
    [p] = segments(master, "PNEW")
    [q] = segments(master, "QNEW")

    assert p["security_id"] == q["security_id"]  # the merge joins them
    assert p["available_at"] == bar_available_at(CH_DAY)
    assert q["available_at"] == bar_available_at(CH_Q_FIRST)
    assert q["cik"] == CH_CIK
    assert p["cik"] is None or p["cik_available_at"] >= q["available_at"], p

    t = bar_available_at(CH_DAY) + timedelta(hours=1)
    [p_seen] = rows_of(mask_master(master, t), "PNEW")
    assert p_seen["cik"] is None
    assert rows_of(mask_master(master, t), "QNEW") == []


# =============================================================================================
# check_master: the adjacent rule allows the gap, refuses the reverse
# =============================================================================================

CUT = date(2023, 10, 3)


def test_check_master_accepts_a_successor_knowable_after_the_predecessors_end() -> None:
    # ABAT after the rule: old piece's end known at the snapshot's midnight, new piece at its
    # first bar (20:00 New York of the cut day)
    rows = [
        seg_row("S1", "ABAT", date(2023, 7, 3), CUT, cik=1000101),
        seg_row("S1", "ABAT", CUT, available_at=bar_available_at(CUT), cik=1000102),
    ]
    assert rows[0]["end_available_at"] == et_midnight(CUT)
    assert sm.check_master(master_table(rows)) is None
    # a rename's successor appearing months later (FSLF) is the same shape on one label
    rows = [
        seg_row("S2", "FFRX", date(2020, 1, 2), FS_RECORD),
        seg_row("S3", "FFRX", FS_RECORD, available_at=bar_available_at(FS_FIRST_BAR)),
    ]
    assert sm.check_master(master_table(rows)) is None


def test_check_master_rejects_a_successor_knowable_before_the_predecessors_end() -> None:
    # the next owner shown while the previous one still looks open: the label owned twice
    rows = [
        seg_row("S1", "ABAT", date(2023, 7, 3), CUT, end_available_at=bar_available_at(CUT)),
        seg_row("S2", "ABAT", CUT, available_at=bar_available_at(CUT) - MICRO),
    ]
    with pytest.raises(ValueError, match=r"instant|adjacent"):
        sm.check_master(master_table(rows))
