"""Guards against the truncation gate passing a leak that changes what a backtest reads, or failing
on churn that changes nothing: the gate now ratchets the backtest impact of each leak (the price
cells and symbol CIKs a reader at ``as_of`` gets differently from the full master), attributed to
the diff keys found at the same instant, instead of the diff keys themselves.

Target API (implementation pending), ``asof.ingest.truncation_gate``:

    @dataclass(frozen=True, order=True) ImpactCount(price: int, cik: int)
    @dataclass(frozen=True) UnmatchedCell(as_of: datetime, what: 'price' | 'cik', owner: str,
        day: date, segments: tuple[str, ...])
        owner: a price cell's security name, a cik cell's symbol; segments: the sorted distinct
        full / truncated segments of the leak rows covering the cell
    @dataclass(frozen=True) Attribution(counts: dict[str, ImpactCount],
        unmatched: list[UnmatchedCell])
    attribute(results: Sequence[truncation_impact.InstantResult]) -> Attribution
        per instant: the keys are diff_key(r) of the diff rows with kind 'leak' at that instant;
        a key's segment is SYM:YYYY-MM-DD. Only impact rows with kind 'leak' count. A price cell
        is (security, day) for each day of a price row's sessions, a cik cell (symbol, day) of a
        cik row. A row touches a key if its full_segment or trunc_segment is the key's segment;
        a cell is covered by every key any of its rows touches. A key's count at an instant =
        distinct price cells / cik cells it covers; counts = per-component sum over instants,
        (0, 0) left out. unmatched = every leak cell no key covers, sorted (as_of, what, owner,
        day).
    load_impact_baseline(path) -> dict[str, ImpactCount]
        configs/truncation_impact_baseline.txt: ``KEY price=N cik=M`` per line, ``#`` comments,
        blank lines; KEY a leak diff_key (master_truncation._KEY); N, M >= 0, not both 0. A
        missing file is {}; anything else malformed, or a duplicated key, is a ValueError naming
        the file.
    prune_impact_baseline(path, *, remove, tighten) -> (removed, tightened)
        header comments before the first entry kept, entries sorted by key, atomic replace;
        refuses (ValueError, file untouched) a key not in the file, a tightened value above the
        current one in either component, a tightened (0, 0), a key both removed and tightened.
    @dataclass(frozen=True) GateResult(new, grew, shrank, fixed: list[str], unmatched: int)
        ok = not new and not grew and not fixed and unmatched == 0
    gate(found, *, baseline, unmatched, full_run) -> GateResult, lists sorted
        new: found keys not in the baseline; grew: found above the baseline in either component
        (any run); shrank: below in some component and not grew, fixed: baseline keys not found
        (both on a full run only).
    asof market truncation [--as-of ISO]... [--overrides PATH] [--baseline PATH]
        [--prune-baseline] [--instants PATH] [--computed-instants] [--jobs N] [--data-dir DIR]
        diffs -> <market>/audit/truncation/<UTC instant>.parquet (compare, as before)
        impact -> <market>/audit/truncation_impact/<UTC instant>.parquet (impact_at, sessions)
        exit 0 when the gate is ok (shrank keys printed as a notice), 2 when not; a full run
        removes earlier instant files in both directories first; --prune-baseline (full run
        only, refused while anything is new, grew or unmatched) removes fixed keys and tightens
        shrank ones. Refused before any build: a missing overrides, master, class or split file,
        a malformed baseline; refused before writing: an on-disk master other than the one the
        inputs build. `asof market truncation-impact` and --exceptions no longer exist.

Fake data provenance:
- The InstantResult tables of the attribute and gate cases are hand-made DIFF_SCHEMA and
  IMPACT_SCHEMA rows. Their shapes follow impact_at's output (measured on the real store,
  2026-10-20: SUNE and DFNS bars the build filed into JCS's and BRLI's securities give leak
  rows whose own segments match no diff key, while the full-side rows of the same cells name
  the PEGY and NUKK grouping keys). A few cases (two rows of one cell touching one key) are
  constructed to pin the counting rule, not modelled on any output.
- The end-to-end attribute cases run compare and impact_at on the hand-made masters of
  tests/ingest/test_truncation_impact.py (its provenance notes apply).
- The CLI store is test_truncation_impact.write_liquid_store: the CYCN / KRSA world (identical
  bars, a dated merge override) with CYCN's merged security in the liquid tier. Measured on it
  with the current compare and impact_at (2026-10-21): at 2024-03-01 00:00 New York the only
  leak diff key is KRSA:2024-01-02:grouping_differs, and the impact has two leak rows, both on
  CYCN@2024-01-02's cells over all 31 sessions (through CYCN and through KRSA); at 2024-06-03
  neither has a row. So the run's attribution is {KRSA key: price=31 cik=0}, nothing unmatched.

Real-data checks for the assumptions:
- Audit command: ``asof market truncation`` on the real store at the fixed instants
  (configs/truncation_instants.txt) against configs/truncation_impact_baseline.txt: exit 0,
  0 unmatched cells; every baseline line is read by a human when it is added.
- Standard cases (tests/ingest/test_truncation_impact.py, realdata): COHR / IIVI at 2022-09-04
  shows no price leak on IIVI@2020-01-02 (a relabelling costs nothing); Barnes / Barrick (B)
  one microsecond before 2025-07-25 shows the CIK leak and the price leak.
- ASSUMPTION to check on the first real run: no leak cell is left unmatched by the cell-level
  rule (the 12 SUNE / DFNS rows of 2026-10-20 are covered through the PEGY and NUKK keys). If
  some are, they are a finding to read, not a rule to loosen.
"""

import dataclasses
import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from asof import cli
from asof.cli import app
from asof.ingest import security_master as sm
from asof.ingest.instrument_class import CLASS_FILE
from asof.ingest.master_truncation import (
    DIFF_SCHEMA,
    compare,
    default_instants,
    diff_key,
    load_instants,
)
from asof.ingest.truncation_gate import (
    Attribution,
    GateResult,
    ImpactCount,
    UnmatchedCell,
    attribute,
    baseline_fingerprint,
    gate,
    instants_fingerprint,
    load_impact_baseline,
    prune_impact_baseline,
    rebase_impact_baseline,
)
from asof.ingest.truncation_impact import IMPACT_SCHEMA, InstantResult, impact_at, run_instants
from asof.store.market import market_root
from asof.store.universe import tier_union
from tests.ingest.test_master_truncation import (
    CLI_DAYS,
    CLI_DECIDED,
    CLI_INSTANTS,
    LEAK_EXIT,
    LEAK_KEYS,
    MERGE,
    OTHER_KEY,
    stamp,
    write_yaml,
)
from tests.ingest.test_security_master import master_table
from tests.ingest.test_truncation_impact import (
    AS_OF,
    DAYS,
    copy_as,
    cycn_id,
    impact,
    joined_world,
    relabel_world,
    series,
    splits_table,
    write_liquid_store,
)

C = ImpactCount
T1 = datetime(2024, 3, 1, 5, tzinfo=UTC)
T2 = datetime(2024, 6, 3, 4, tzinfo=UTC)
D1, D2, D3, D4, D5 = DAYS  # 2020-01-02, 01-03, 01-06, 01-07, 01-08

KEY_A = "A:2020-01-02:grouping_differs"
KEY_B = "B:2020-01-06:only_in_full"
KEY_C = "C:2020-01-02:fields_differ:cik+valid_to"  # the fields part takes no part in matching
SEG_A, SEG_B, SEG_C = "A:2020-01-02", "B:2020-01-06", "C:2020-01-02"


def diff_row(key: str, kind: str = "leak") -> dict[str, Any]:
    """The DIFF_SCHEMA row whose diff_key is ``key``."""
    symbol, day, reason, *fields = key.split(":")
    return {"kind": kind, "reason": reason, "symbol": symbol,
            "valid_from": date.fromisoformat(day), "fields": fields[0] if fields else "",
            "detail": ""}  # fmt: skip


def price(
    security: str,
    symbol: str,
    full: str | None,
    trunc: str | None,
    days: Iterable[date],
    kind: str = "leak",
) -> dict[str, Any]:
    sessions = sorted(set(days))
    return {"kind": kind, "what": "price", "security": security, "symbol": symbol,
            "full_segment": full, "trunc_segment": trunc, "first_day": sessions[0],
            "last_day": sessions[-1], "days": len(sessions), "detail": "",
            "sessions": sessions}  # fmt: skip


def cik(
    symbol: str, full: str | None, trunc: str | None, days: Iterable[date], kind: str = "leak"
) -> dict[str, Any]:
    return {**price("unused", symbol, full, trunc, days, kind), "what": "cik", "security": None}


def at(
    t: datetime, diff: Sequence[str | dict[str, Any]] = (), rows: Sequence[dict[str, Any]] = ()
) -> InstantResult:
    """An instant's result: ``diff`` as leak keys (or whole DIFF_SCHEMA rows), ``rows`` as
    IMPACT_SCHEMA rows."""
    diff_rows = [diff_row(d) if isinstance(d, str) else d for d in diff]
    return InstantResult(
        as_of=t,
        diff=pa.Table.from_pylist(diff_rows, schema=DIFF_SCHEMA),
        impact=pa.Table.from_pylist(list(rows), schema=IMPACT_SCHEMA),
    )


def test_diff_row_helper_round_trips_through_diff_key() -> None:
    for key in (KEY_A, KEY_B, KEY_C):
        assert diff_key(diff_row(key)) == key


# =============================================================================================
# dataclasses
# =============================================================================================


def test_impact_count_is_a_frozen_ordered_pair_of_price_and_cik() -> None:
    count = C(price=3, cik=1)
    assert [f.name for f in dataclasses.fields(count)] == ["price", "cik"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        count.price = 4  # type: ignore[misc]
    assert sorted([C(1, 5), C(0, 9), C(1, 0)]) == [C(0, 9), C(1, 0), C(1, 5)]
    assert C(2, 0) == C(price=2, cik=0)


def test_unmatched_cell_attribution_and_gate_result_fields() -> None:
    assert [f.name for f in dataclasses.fields(UnmatchedCell)] == [
        "as_of", "what", "owner", "day", "segments",
    ]  # fmt: skip
    assert [f.name for f in dataclasses.fields(Attribution)] == ["counts", "unmatched"]
    assert [f.name for f in dataclasses.fields(GateResult)] == [
        "new", "grew", "shrank", "fixed", "unmatched",
    ]  # fmt: skip


# =============================================================================================
# attribute: normal
# =============================================================================================


def test_one_key_covering_a_price_row_counts_its_sessions() -> None:
    # A's security reads three days differently (the sessions, not the span D1..D4)
    results = [at(T1, [KEY_A], [price("A@2020-01-02", "A", SEG_A, SEG_A, [D1, D2, D4])])]
    assert attribute(results) == Attribution(counts={KEY_A: C(3, 0)}, unmatched=[])


@pytest.mark.parametrize("side", ["full", "trunc"])
def test_a_row_touches_a_key_through_either_segment(side: str) -> None:
    full, trunc = (SEG_A, None) if side == "full" else (None, SEG_A)
    results = [at(T1, [KEY_A], [price("A@2020-01-02", "A", full, trunc, [D1, D2])])]
    assert attribute(results).counts == {KEY_A: C(2, 0)}


def test_cik_rows_count_per_symbol_day() -> None:
    # a cik cell is (symbol, day): A's CIK differs on 4 sessions, B's on 2
    results = [
        at(T1, [KEY_A, KEY_B], [
            cik("A", SEG_A, SEG_A, [D1, D2, D3, D4]),
            cik("B", SEG_B, None, [D3, D5]),
            price("A@2020-01-02", "A", SEG_A, SEG_A, [D2]),
        ])
    ]  # fmt: skip
    assert attribute(results) == Attribution(counts={KEY_A: C(1, 4), KEY_B: C(0, 2)}, unmatched=[])


def test_the_fields_part_of_a_key_takes_no_part_in_matching() -> None:
    results = [at(T1, [KEY_C], [cik("C", SEG_C, SEG_C, [D1, D2])])]
    assert attribute(results).counts == {KEY_C: C(0, 2)}


def test_every_key_of_a_touched_segment_is_counted() -> None:
    # two leak reasons on one segment: the cells are each key's impact (a key is a line to fix)
    other = "A:2020-01-02:fields_differ:cik"
    results = [at(T1, [KEY_A, other], [price("A@2020-01-02", "A", SEG_A, SEG_A, [D1, D2])])]
    assert attribute(results).counts == {KEY_A: C(2, 0), other: C(2, 0)}


def test_a_key_is_the_sum_over_instants_per_component() -> None:
    # Price: 3 cells at T1 + 1 at T2; CIK: 0 + 2. A cell recurring at a later instant counts
    # again there: the instants are a fixed list, so the sum is stable and any growth shows.
    results = [
        at(T1, [KEY_A], [price("A@2020-01-02", "A", SEG_A, SEG_A, [D1, D2, D3])]),
        at(T2, [KEY_A], [
            price("A@2020-01-02", "A", SEG_A, SEG_A, [D1]),
            cik("A", SEG_A, SEG_A, [D1, D2]),
        ]),
    ]  # fmt: skip
    assert attribute(results).counts == {KEY_A: C(4, 2)}


def test_growth_at_an_instant_other_than_the_largest_one_grows() -> None:
    # Reviewer H1 (2026-10-21). KEY_A covers 1 cell at T1 and 3 at T2; the baseline is their
    # sum, 4. New data makes it cover 2 at T1: under "the largest instant" the number stays 3
    # and a backtest at T1 silently reads one more wrong day; as a sum it is 5 > 4.
    before = [
        at(T1, [KEY_A], [price("A@2020-01-02", "A", SEG_A, SEG_A, [D1])]),
        at(T2, [KEY_A], [price("A@2020-01-02", "A", SEG_A, SEG_A, [D1, D2, D3])]),
    ]
    after = [
        at(T1, [KEY_A], [price("A@2020-01-02", "A", SEG_A, SEG_A, [D1, D2])]),
        before[1],
    ]
    baseline = attribute(before).counts
    assert baseline == {KEY_A: C(4, 0)}

    result = gate(attribute(after).counts, baseline=baseline, unmatched=0, full_run=True)

    assert result.grew == [KEY_A]
    assert not result.ok


def test_a_cell_is_covered_through_the_other_rows_of_the_same_security_day() -> None:
    # SUNE-like. Cell (P@2020-01-02, d) differs because the build filed S's bars into P's
    # security: its full-side row (through P) touches P's grouping key, its truncated-side row
    # (through S) names only S's segment, which has no key. The cell is covered, once.
    key = "P:2020-01-02:grouping_differs"
    results = [
        at(T1, [key], [
            price("P@2020-01-02", "P", "P:2020-01-02", "P:2020-01-02", [D1, D2, D3]),
            price("P@2020-01-02", "S", "S:2020-01-02", "S:2020-01-02", [D1, D2, D3]),
        ])
    ]  # fmt: skip
    assert attribute(results) == Attribution(counts={key: C(3, 0)}, unmatched=[])


def test_a_joined_symbol_end_to_end_counts_each_cell_once() -> None:
    # H1 of test_truncation_impact.py end to end. compare finds one leak: B:2020-01-02 is named
    # B@2020-01-02 in full and A@2020-01-02 truncated. impact_at gives three leak rows over the 5
    # days: (A@, through A, segments A:2020-01-02), (A@, through B, B:2020-01-02) and (B@, through
    # B, B:2020-01-02). Cells: (A@, d) x 5, covered by the B key through the via-B row although
    # the via-A row names no key; (B@, d) x 5. So B's key covers 10 price cells, none unmatched.
    full, truncated = joined_world(b_in_full=True)
    bars = [*series("A", DAYS), *series("B", DAYS, 30.0)]
    diff = compare(master_table(full), master_table(truncated), AS_OF)
    assert [diff_key(r) for r in diff.to_pylist() if r["kind"] == "leak"] == [
        "B:2020-01-02:grouping_differs"
    ]

    got = attribute([InstantResult(as_of=AS_OF, diff=diff, impact=impact(full, truncated, bars))])

    assert got == Attribution(counts={"B:2020-01-02:grouping_differs": C(10, 0)}, unmatched=[])


def test_a_relabelled_bar_end_to_end_is_named_by_the_full_sides_key() -> None:
    # relabel_world: the full master keeps A open, the build ends A at CUT and continues A's
    # security through Z, whose 01-07 bar differs. compare: A's segment differs in its end (a
    # leak key), Z's segment is only in the build on days nobody owns in full (a delay, no key).
    # impact_at: cell (A@2020-01-02, 01-07) through A (A:2020-01-02, none) and through Z (none,
    # Z:2020-01-06). The cell is covered by A's key; the Z row alone would be unmatched.
    full, truncated = relabel_world()
    a = series("A", DAYS)
    z = [{**r, "close": 99.0} if r["session_date"] == D4 else r for r in copy_as(a, "Z")]
    z = [r for r in z if r["session_date"] >= D3]
    diff = compare(master_table(full), master_table(truncated), AS_OF)
    keys = [diff_key(r) for r in diff.to_pylist() if r["kind"] == "leak"]
    assert keys == ["A:2020-01-02:fields_differ:end_available_at+valid_to"]

    got = attribute(
        [InstantResult(as_of=AS_OF, diff=diff, impact=impact(full, truncated, [*a, *z]))]
    )

    assert got == Attribution(counts={keys[0]: C(1, 0)}, unmatched=[])


# =============================================================================================
# attribute: edges
# =============================================================================================


def test_no_results_is_an_empty_attribution() -> None:
    assert attribute([]) == Attribution(counts={}, unmatched=[])
    assert attribute([at(T1), at(T2)]) == Attribution(counts={}, unmatched=[])


def test_delay_impact_rows_never_count() -> None:
    # a delay is a cell the build then had and the full master withholds: nothing seen early,
    # no impact to ratchet and no cell to match
    results = [
        at(T1, [KEY_A], [
            price("A@2020-01-06", "A", None, SEG_A, [D3, D4], kind="delay"),
            cik("A", None, SEG_A, [D1, D2], kind="delay"),
        ])
    ]  # fmt: skip
    assert attribute(results) == Attribution(counts={}, unmatched=[])


def test_delay_and_info_diff_rows_name_nothing() -> None:
    # only leak diff rows are keys: a delay or info row on A's segment covers none of its cells
    results = [
        at(T1, [diff_row("A:2020-01-02:only_in_truncated", kind="delay"),
                diff_row("A:2020-01-02:cik_withheld", kind="delay"),
                diff_row("A:2020-01-02:evidence_differs", kind="info")],
           [price("A@2020-01-02", "A", SEG_A, SEG_A, [D1, D2])])
    ]  # fmt: skip
    got = attribute(results)
    assert got.counts == {}
    assert got.unmatched == [
        UnmatchedCell(T1, "price", "A@2020-01-02", D1, (SEG_A,)),
        UnmatchedCell(T1, "price", "A@2020-01-02", D2, (SEG_A,)),
    ]


def test_two_rows_of_one_cell_touching_one_key_count_the_cell_once() -> None:
    # The old per-key sum added each row's days: two rows of the same (security, day) touching
    # the key counted that cell twice. A count is of cells. Constructed: one row touches through
    # its full segment, the other through its truncated one, sharing D2.
    results = [
        at(T1, [KEY_A], [
            price("A@2020-01-02", "A", SEG_A, None, [D1, D2]),
            price("A@2020-01-02", "Z", None, SEG_A, [D2, D3]),
        ])
    ]  # fmt: skip
    assert attribute(results).counts == {KEY_A: C(3, 0)}


def test_a_cell_whose_other_row_is_on_another_day_is_unmatched() -> None:
    # The SUNE-like case with the truncated-side row one day later: (P@, D4) has only the S row,
    # which touches no key. It is reported with the segments its rows name.
    key = "P:2020-01-02:grouping_differs"
    results = [
        at(T1, [key], [
            price("P@2020-01-02", "P", "P:2020-01-02", "P:2020-01-02", [D1, D2, D3]),
            price("P@2020-01-02", "S", None, "S:2020-01-02", [D3, D4]),
        ])
    ]  # fmt: skip
    got = attribute(results)
    assert got.counts == {key: C(3, 0)}
    assert got.unmatched == [UnmatchedCell(T1, "price", "P@2020-01-02", D4, ("S:2020-01-02",))]


def test_a_cik_rows_key_does_not_cover_a_price_cell() -> None:
    # price cells and cik cells are separate: A's CIK key covers (A, D1), not A's security's
    # price cell of the same day read through Z
    results = [
        at(T1, [KEY_A], [
            cik("A", SEG_A, SEG_A, [D1]),
            price("A@2020-01-02", "Z", None, "Z:2020-01-02", [D1]),
        ])
    ]  # fmt: skip
    got = attribute(results)
    assert got.counts == {KEY_A: C(0, 1)}
    assert got.unmatched == [UnmatchedCell(T1, "price", "A@2020-01-02", D1, ("Z:2020-01-02",))]


def test_a_key_covering_no_cell_is_left_out() -> None:
    # B's leak changes nothing a backtest reads (COHR / IIVI-like): no count, not (0, 0)
    results = [at(T1, [KEY_A, KEY_B], [price("A@2020-01-02", "A", SEG_A, SEG_A, [D1])])]
    assert attribute(results).counts == {KEY_A: C(1, 0)}


def test_unmatched_cells_are_sorted_with_their_segments() -> None:
    # rows given in no order; cells sorted by (as_of, what, owner, day); a cell's segments are
    # the distinct non-null segments of all its leak rows, sorted
    results = [
        at(T1, [], [
            price("Q@2020-01-02", "Y", "Y:2020-01-02", None, [D2]),
            price("Q@2020-01-02", "X", "X:2020-01-02", "X:2020-01-03", [D2, D1]),
            cik("X", "X:2020-01-02", "X:2020-01-03", [D3]),
            price("E@2020-01-06", "E", "E:2020-01-06", None, [D3]),
        ]),
        at(T2, [], [cik("B", SEG_B, None, [D1])]),
    ]  # fmt: skip
    assert attribute(results).unmatched == [
        UnmatchedCell(T1, "cik", "X", D3, ("X:2020-01-02", "X:2020-01-03")),
        UnmatchedCell(T1, "price", "E@2020-01-06", D3, ("E:2020-01-06",)),
        UnmatchedCell(T1, "price", "Q@2020-01-02", D1, ("X:2020-01-02", "X:2020-01-03")),
        UnmatchedCell(T1, "price", "Q@2020-01-02", D2,
                      ("X:2020-01-02", "X:2020-01-03", "Y:2020-01-02")),
        UnmatchedCell(T2, "cik", "B", D1, (SEG_B,)),
    ]  # fmt: skip


# =============================================================================================
# attribute: leakage -- a key never covers cells of an instant it was not found at
# =============================================================================================


def test_a_key_covers_only_cells_of_its_own_instant() -> None:
    # A's leak is in the diff at T1 only; at T2 the diff no longer has it, yet a row there names
    # A's segment. The T2 cells are unmatched: a key found once must not absorb what happens
    # at other instants.
    results = [
        at(T1, [KEY_A], [price("A@2020-01-02", "A", SEG_A, SEG_A, [D1])]),
        at(T2, [], [price("A@2020-01-02", "A", SEG_A, SEG_A, [D1, D2])]),
    ]
    got = attribute(results)
    assert got.counts == {KEY_A: C(1, 0)}
    assert got.unmatched == [
        UnmatchedCell(T2, "price", "A@2020-01-02", D1, (SEG_A,)),
        UnmatchedCell(T2, "price", "A@2020-01-02", D2, (SEG_A,)),
    ]


def test_a_cell_is_never_attributed_to_a_diff_row_found_only_later() -> None:
    # Leakage. At T1 the cell exists but no key does; the key appears in the T2 diff only (and
    # covers nothing there). Attributing T1's cell to it would read T2's diff at T1.
    results = [
        at(T1, [], [price("A@2020-01-02", "A", SEG_A, SEG_A, [D1])]),
        at(T2, [KEY_A], []),
    ]
    got = attribute(results)
    assert got.counts == {}
    assert got.unmatched == [UnmatchedCell(T1, "price", "A@2020-01-02", D1, (SEG_A,))]


@pytest.mark.parametrize(
    "segment",
    ["AB:2020-01-02", "A:2020-01-03", "B:2020-01-02"],
    ids=["symbol_prefix", "other_valid_from", "other_symbol"],
)
def test_a_key_touches_its_own_segment_only(segment: str) -> None:
    results = [at(T1, [KEY_A], [price("Q@2020-01-02", "Q", segment, segment, [D1])])]
    got = attribute(results)
    assert got.counts == {}
    assert got.unmatched == [UnmatchedCell(T1, "price", "Q@2020-01-02", D1, (segment,))]


# =============================================================================================
# gate
# =============================================================================================

BASE = {KEY_A: C(5, 0), KEY_B: C(0, 3), KEY_C: C(2, 2)}


def verdict(found: Mapping[str, ImpactCount], *, full_run: bool, unmatched: int = 0) -> GateResult:
    return gate(found, baseline=BASE, unmatched=unmatched, full_run=full_run)


@pytest.mark.parametrize("full_run", [True, False], ids=["full", "partial"])
def test_gate_found_equal_to_the_baseline_is_ok(full_run: bool) -> None:
    result = verdict(dict(BASE), full_run=full_run)
    assert result == GateResult(new=[], grew=[], shrank=[], fixed=[], unmatched=0)
    assert result.ok


@pytest.mark.parametrize("full_run", [True, False], ids=["full", "partial"])
def test_gate_a_key_not_in_the_baseline_is_new(full_run: bool) -> None:
    new = "Z:2021-03-04:replaced"
    result = verdict({**BASE, new: C(1, 0)}, full_run=full_run)
    assert result.new == [new]
    assert not result.ok


@pytest.mark.parametrize("full_run", [True, False], ids=["full", "partial"])
@pytest.mark.parametrize(
    "found", [C(6, 0), C(5, 1)], ids=["price_up", "cik_appears"]
)  # fmt: skip
def test_gate_a_key_above_its_baseline_grew_in_any_run(full_run: bool, found: ImpactCount) -> None:
    # Leakage. A key whose price stays but whose CIK starts leaking (0 -> 1) reads a company a
    # backtest could not have known: growth, whatever the price does. A partial run's sum over
    # fewer instants above the baseline is real growth too.
    result = verdict({**BASE, KEY_A: found}, full_run=full_run)
    assert result.grew == [KEY_A]
    assert result.shrank == []
    assert not result.ok


def test_gate_a_key_below_its_baseline_shrank_on_a_full_run_and_stays_ok() -> None:
    result = verdict({**BASE, KEY_A: C(4, 0), KEY_C: C(2, 1)}, full_run=True)
    assert result.shrank == [KEY_A, KEY_C]
    assert (result.new, result.grew, result.fixed) == ([], [], [])
    assert result.ok


def test_gate_a_partial_run_proves_nothing_shrank() -> None:
    # the sum over a subset of the instants leaves out the others the baseline number added
    result = verdict({**BASE, KEY_A: C(4, 0)}, full_run=False)
    assert result.shrank == []
    assert result.ok


def test_gate_a_baseline_key_not_found_on_a_full_run_is_fixed_and_fails() -> None:
    # the ratchet: a fix must also delete its baseline line
    found = {KEY_A: BASE[KEY_A], KEY_B: BASE[KEY_B]}
    result = verdict(found, full_run=True)
    assert result.fixed == [KEY_C]
    assert not result.ok


def test_gate_a_partial_run_reports_nothing_fixed() -> None:
    result = verdict({KEY_A: BASE[KEY_A]}, full_run=False)
    assert result.fixed == []
    assert result.ok


@pytest.mark.parametrize("full_run", [True, False], ids=["full", "partial"])
def test_gate_up_in_one_component_and_down_in_the_other_grew_not_shrank(full_run: bool) -> None:
    # (3, 3) against (2, 2) grows in both; (1, 3) grows in cik although price fell; (3, 1)
    # grows in price although cik fell. Neither is "shrank", and the lexicographic order of
    # ImpactCount (3, 1) < ... plays no part.
    for found in (C(1, 3), C(3, 1)):
        result = verdict({**BASE, KEY_C: found}, full_run=full_run)
        assert result.grew == [KEY_C], found
        assert result.shrank == [], found
        assert not result.ok, found


@pytest.mark.parametrize(
    "found", [C(6, 0), C(5, 1), C(4, 0)], ids=["price_up", "cik_up", "down"]
)  # fmt: skip
def test_gate_a_run_on_other_instants_judges_neither_growth_nor_shrinkage(
    found: ImpactCount,
) -> None:
    # Sums over instants the baseline was not summed over compare with nothing: no grew, no
    # shrank, no fixed. New keys and unmatched cells still fail.
    result = gate({**BASE, KEY_A: found, "Z:2021-03-04:replaced": C(1, 0)}, baseline=BASE,
                  unmatched=2, full_run=False, comparable=False)  # fmt: skip
    assert (result.grew, result.shrank, result.fixed) == ([], [], [])
    assert result.new == ["Z:2021-03-04:replaced"]
    assert result.unmatched == 2
    assert gate({KEY_A: found}, baseline=BASE, unmatched=0, full_run=False, comparable=False).ok


def test_gate_a_full_run_is_always_comparable() -> None:
    result = gate({**BASE, KEY_A: C(6, 0)}, baseline=BASE, unmatched=0, full_run=True,
                  comparable=False)  # fmt: skip
    assert result.grew == [KEY_A]


def test_gate_unmatched_cells_fail() -> None:
    result = verdict(dict(BASE), full_run=True, unmatched=3)
    assert result.unmatched == 3
    assert not result.ok
    assert not verdict(dict(BASE), full_run=False, unmatched=1).ok


def test_gate_result_ok_ignores_shrank_only() -> None:
    assert GateResult(new=[], grew=[], shrank=[KEY_A], fixed=[], unmatched=0).ok
    assert not GateResult(new=[KEY_A], grew=[], shrank=[], fixed=[], unmatched=0).ok
    assert not GateResult(new=[], grew=[KEY_A], shrank=[], fixed=[], unmatched=0).ok
    assert not GateResult(new=[], grew=[], shrank=[], fixed=[KEY_A], unmatched=0).ok
    assert not GateResult(new=[], grew=[], shrank=[], fixed=[], unmatched=1).ok


def test_gate_nothing_found_nothing_known_is_ok() -> None:
    result = gate({}, baseline={}, unmatched=0, full_run=True)
    assert result == GateResult(new=[], grew=[], shrank=[], fixed=[], unmatched=0)
    assert result.ok


def test_gate_outputs_are_sorted_whatever_the_input_order() -> None:
    baseline = {
        "Y:2020-01-02:replaced": C(5, 0), "B:2020-01-02:replaced": C(5, 0),  # grow
        "X:2020-01-02:replaced": C(5, 0), "C:2020-01-02:replaced": C(5, 0),  # shrink
        "W:2020-01-02:replaced": C(5, 0), "D:2020-01-02:replaced": C(5, 0),  # fixed
    }  # fmt: skip
    found = {
        "Z:2021-03-04:replaced": C(1, 0), "A:2021-03-04:replaced": C(1, 0),  # new
        "Y:2020-01-02:replaced": C(6, 0), "B:2020-01-02:replaced": C(5, 1),
        "X:2020-01-02:replaced": C(4, 0), "C:2020-01-02:replaced": C(1, 0),
    }  # fmt: skip
    result = gate(found, baseline=baseline, unmatched=0, full_run=True)
    assert result.new == ["A:2021-03-04:replaced", "Z:2021-03-04:replaced"]
    assert result.grew == ["B:2020-01-02:replaced", "Y:2020-01-02:replaced"]
    assert result.shrank == ["C:2020-01-02:replaced", "X:2020-01-02:replaced"]
    assert result.fixed == ["D:2020-01-02:replaced", "W:2020-01-02:replaced"]


# --- leakage: a new leak must never hide behind a baseline line ---


@pytest.mark.parametrize(
    "found_key",
    [
        "X:2020-01-02:fields_differ:cik+valid_to",  # one more field
        "X:2020-01-02:grouping_differs",  # another reason, same segment
        "X:2020-01-03:fields_differ:valid_to",  # another valid_from
    ],
    ids=["extra_field", "other_reason", "other_valid_from"],
)
def test_gate_a_baseline_key_covers_its_own_string_only(found_key: str) -> None:
    known = "X:2020-01-02:fields_differ:valid_to"
    result = gate({found_key: C(3, 0)}, baseline={known: C(3, 0)}, unmatched=0, full_run=True)
    assert result.new == [found_key]
    assert result.fixed == [known]
    assert not result.ok


# =============================================================================================
# the impact baseline file: load_impact_baseline / prune_impact_baseline
# =============================================================================================

HEADER = "# Leak impact per diff key, summed over the fixed instants; may only shrink.\n#\n"
KX = "X:2020-01-02:only_in_full"


def write_baseline(
    path: Path, entries: Mapping[str, tuple[int, int]], header: str = HEADER
) -> Path:
    path.write_text(header + "".join(f"{k} price={p} cik={c}\n" for k, (p, c) in entries.items()))
    return path


def test_load_impact_baseline_reads_entries_skipping_comments_and_blank_lines(
    tmp_path: Path,
) -> None:
    path = tmp_path / "impact_baseline.txt"
    path.write_text(f"{HEADER}{KEY_C} price=2 cik=2\n\n# a note\n  {KEY_A} price=5 cik=0  \n"
                    f"\t{KEY_B} price=0 cik=3\n   \n")  # fmt: skip
    assert load_impact_baseline(path) == BASE


def test_load_impact_baseline_missing_file_is_empty(tmp_path: Path) -> None:
    assert load_impact_baseline(tmp_path / "absent.txt") == {}


def test_a_baseline_written_from_an_attribution_loads_and_gates_clean(tmp_path: Path) -> None:
    results = [
        at(T1, [KEY_A, KEY_B], [
            price("A@2020-01-02", "A", SEG_A, SEG_A, [D1, D2]),
            cik("B", SEG_B, None, [D3]),
        ]),
        at(T2, [KEY_C], [cik("C", SEG_C, SEG_C, [D1, D2, D3])]),
    ]  # fmt: skip
    counts = attribute(results).counts
    path = write_baseline(
        tmp_path / "impact_baseline.txt", {k: (c.price, c.cik) for k, c in counts.items()}
    )

    loaded = load_impact_baseline(path)

    assert loaded == counts == {KEY_A: C(2, 0), KEY_B: C(0, 1), KEY_C: C(0, 3)}
    assert gate(counts, baseline=loaded, unmatched=0, full_run=True).ok


@pytest.mark.parametrize(
    "line",
    [
        "X:2020-13-02:only_in_full price=1 cik=0",  # date does not parse
        "X:20200102:only_in_full price=1 cik=0",  # not YYYY-MM-DD
        "X:2020-01-02:went_missing price=1 cik=0",  # unknown reason
        "X:2020-01-02:only_in_truncated price=1 cik=0",  # a delay, never a leak key
        "X:2020-01-02:cik_withheld price=1 cik=0",  # a delay
        "X:2020-01-02:evidence_differs price=1 cik=0",  # info
        "X:2020-01-02:fields_differ: price=1 cik=0",  # trailing empty fields
        ":2020-01-02:only_in_full price=1 cik=0",  # no symbol
        f"{KX}",  # no numbers: an old-style key-only line
        f"{KX} price=1",  # no cik
        f"{KX} cik=1",  # no price
        f"{KX} price=1 cik=0 days=4",  # an extra field
        f"{KX} price=1 price=2",  # a field twice, cik missing
        f"{KX} price=one cik=0",
        f"{KX} price=1.5 cik=0",
        f"{KX} price= cik=0",
        f"{KX} price:1 cik=0",
        f"{KX} prices=1 cik=0",
        f"{KX} price=1cik=0",
        f"{KX} price=-1 cik=0",  # negative
        f"{KX} price=0 cik=-2",
        f"{KX} price=0 cik=0",  # a key with no impact belongs nowhere
    ],
)
def test_load_impact_baseline_refuses_a_malformed_line(tmp_path: Path, line: str) -> None:
    path = tmp_path / "impact_baseline.txt"
    path.write_text(f"{HEADER}{KEY_A} price=5 cik=0\n{line}\n")
    with pytest.raises(ValueError, match="impact_baseline.txt"):
        load_impact_baseline(path)


def test_load_impact_baseline_refuses_a_duplicated_key(tmp_path: Path) -> None:
    # even with other numbers: which one would the gate believe?
    path = tmp_path / "impact_baseline.txt"
    path.write_text(f"{KEY_A} price=5 cik=0\n{KEY_B} price=0 cik=3\n{KEY_A} price=4 cik=0\n")
    with pytest.raises(ValueError, match="impact_baseline.txt") as raised:
        load_impact_baseline(path)
    assert KEY_A in str(raised.value)


def test_prune_removes_tightens_keeps_the_header_and_sorts(tmp_path: Path) -> None:
    path = write_baseline(
        tmp_path / "impact_baseline.txt",
        {"Z:2021-03-04:replaced": (5, 1), KEY_B: (0, 3), KEY_A: (5, 0)},
    )

    counts = prune_impact_baseline(path, remove=[KEY_B], tighten={"Z:2021-03-04:replaced": C(4, 1)})

    assert counts == (1, 1)
    assert (
        path.read_text() == f"{HEADER}{KEY_A} price=5 cik=0\nZ:2021-03-04:replaced price=4 cik=1\n"
    )


def test_prune_may_tighten_one_component_only(tmp_path: Path) -> None:
    path = write_baseline(tmp_path / "impact_baseline.txt", {KEY_C: (2, 2)})
    assert prune_impact_baseline(path, remove=[], tighten={KEY_C: C(2, 1)}) == (0, 1)
    assert load_impact_baseline(path) == {KEY_C: C(2, 1)}


def test_prune_then_load_round_trips_through_the_gate(tmp_path: Path) -> None:
    path = write_baseline(
        tmp_path / "impact_baseline.txt", {k: (c.price, c.cik) for k, c in BASE.items()}
    )
    found = {KEY_A: C(3, 0), KEY_B: C(0, 3)}  # A shrank, C fixed
    result = gate(found, baseline=load_impact_baseline(path), unmatched=0, full_run=True)
    assert (result.shrank, result.fixed) == ([KEY_A], [KEY_C])

    assert prune_impact_baseline(
        path, remove=result.fixed, tighten={k: found[k] for k in result.shrank}
    ) == (1, 1)  # fmt: skip

    assert load_impact_baseline(path) == found
    again = gate(found, baseline=load_impact_baseline(path), unmatched=0, full_run=True)
    assert again.ok and again.shrank == []


def test_prune_of_every_key_leaves_the_header(tmp_path: Path) -> None:
    path = write_baseline(tmp_path / "impact_baseline.txt", {KEY_A: (5, 0)})
    assert prune_impact_baseline(path, remove=[KEY_A], tighten={}) == (1, 0)
    assert load_impact_baseline(path) == {}
    assert path.read_text().startswith(HEADER)


@pytest.mark.parametrize(
    ("remove", "tighten", "named"),
    [
        (["Z:2021-03-04:replaced"], {}, "Z:2021-03-04:replaced"),  # never adds
        ([], {"Z:2021-03-04:replaced": C(1, 0)}, "Z:2021-03-04:replaced"),
        ([], {KEY_A: C(6, 0)}, KEY_A),  # a tightening that loosens the price
        ([], {KEY_C: C(1, 3)}, KEY_C),  # ... or the cik, even with the price lower
        ([], {KEY_A: C(0, 0)}, KEY_A),  # zero is a removal, made as one
        ([KEY_A], {KEY_A: C(4, 0)}, KEY_A),  # both at once
    ],
    ids=["remove_unknown", "tighten_unknown", "price_up", "cik_up", "to_zero", "both"],
)
def test_prune_refuses_and_writes_nothing(
    tmp_path: Path, remove: list[str], tighten: dict[str, ImpactCount], named: str
) -> None:
    path = write_baseline(
        tmp_path / "impact_baseline.txt", {k: (c.price, c.cik) for k, c in BASE.items()}
    )
    text = path.read_text()
    with pytest.raises(ValueError, match=re.escape(named)):
        prune_impact_baseline(path, remove=remove, tighten=tighten)
    assert path.read_text() == text
    assert not [p for p in tmp_path.iterdir() if p.name != path.name]  # no temp file left


def test_instants_fingerprint_names_the_sorted_list() -> None:
    assert re.fullmatch(r"[0-9a-f]{12}", instants_fingerprint([T1, T2]))
    assert instants_fingerprint([T2, T1]) == instants_fingerprint([T1, T2])
    assert instants_fingerprint([T1]) != instants_fingerprint([T1, T2])


def test_baseline_fingerprint_reads_the_header_line(tmp_path: Path) -> None:
    fp = instants_fingerprint([T1, T2])
    path = write_baseline(tmp_path / "impact_baseline.txt", {KEY_A: (5, 0)},
                          header=f"{HEADER}# instants fingerprint: {fp}\n")  # fmt: skip
    assert baseline_fingerprint(path) == fp
    assert load_impact_baseline(path) == {KEY_A: C(5, 0)}
    assert baseline_fingerprint(write_baseline(tmp_path / "plain.txt", {KEY_A: (5, 0)})) is None
    assert baseline_fingerprint(tmp_path / "absent.txt") is None


def test_prune_writes_the_fingerprint_replacing_an_earlier_one(tmp_path: Path) -> None:
    old, new = instants_fingerprint([T1]), instants_fingerprint([T1, T2])
    path = write_baseline(tmp_path / "impact_baseline.txt", {KEY_A: (5, 0), KEY_B: (0, 3)},
                          header=f"{HEADER}# instants fingerprint: {old}\n")  # fmt: skip

    assert prune_impact_baseline(path, remove=[KEY_B], tighten={}, fingerprint=new) == (1, 0)

    assert baseline_fingerprint(path) == new
    assert path.read_text() == f"{HEADER}# instants fingerprint: {new}\n{KEY_A} price=5 cik=0\n"


def test_prune_refuses_a_malformed_fingerprint(tmp_path: Path) -> None:
    path = write_baseline(tmp_path / "impact_baseline.txt", {KEY_A: (5, 0)})
    text = path.read_text()
    with pytest.raises(ValueError, match="fingerprint"):
        prune_impact_baseline(path, remove=[KEY_A], tighten={}, fingerprint="xyz")
    assert path.read_text() == text


def test_rebase_takes_every_listed_keys_new_sum_and_drops_the_rest(tmp_path: Path) -> None:
    old, new = instants_fingerprint([T1]), instants_fingerprint([T1, T2])
    path = write_baseline(tmp_path / "impact_baseline.txt",
                          {KEY_A: (5, 0), KEY_B: (0, 3), KEY_C: (2, 2)},
                          header=f"{HEADER}# instants fingerprint: {old}\n")  # fmt: skip

    changes = rebase_impact_baseline(
        path, counts={KEY_A: C(9, 0), KEY_C: C(2, 2)}, fingerprint=new
    )  # fmt: skip

    assert changes == [(KEY_A, C(5, 0), C(9, 0)), (KEY_B, C(0, 3), None)]
    assert load_impact_baseline(path) == {KEY_A: C(9, 0), KEY_C: C(2, 2)}
    assert baseline_fingerprint(path) == new
    assert path.read_text().startswith(HEADER)


@pytest.mark.parametrize("fingerprint", ["xyz", instants_fingerprint([T1])])
def test_rebase_refuses_a_new_key_or_a_bad_fingerprint(tmp_path: Path, fingerprint: str) -> None:
    path = write_baseline(tmp_path / "impact_baseline.txt", {KEY_A: (5, 0)})
    text = path.read_text()
    counts = {KEY_A: C(5, 0), KEY_B: C(0, 1)} if fingerprint != "xyz" else {KEY_A: C(5, 0)}
    with pytest.raises(ValueError, match=KEY_B if fingerprint != "xyz" else "fingerprint"):
        rebase_impact_baseline(path, counts=counts, fingerprint=fingerprint)
    assert path.read_text() == text


def test_prune_refuses_a_malformed_file(tmp_path: Path) -> None:
    path = tmp_path / "impact_baseline.txt"
    path.write_text(f"{KEY_A} price=5\n")
    with pytest.raises(ValueError, match="impact_baseline.txt"):
        prune_impact_baseline(path, remove=[KEY_A], tighten={})
    assert path.read_text() == f"{KEY_A} price=5\n"


# =============================================================================================
# CLI: asof market truncation
# =============================================================================================

N = len(CLI_DAYS)  # 31 sessions: the merge leak's cells at 2024-03-01 (module docstring)
KEY = LEAK_KEYS[0]  # KRSA:2024-01-02:grouping_differs
EXACT = {KEY: (N, 0)}


def run_cli(
    tmp_path: Path,
    *,
    as_of: Sequence[str] = (),
    baseline: Mapping[str, tuple[int, int]] | None = None,
    baseline_file: Path | None = None,
    instants: Sequence[str] = CLI_INSTANTS,
    computed: bool = False,
    prune: bool = False,
    jobs: str | None = None,
    overrides: Sequence[Mapping[str, Any]] | None = (MERGE,),
    extra: Sequence[str] = (),
) -> Any:
    """``baseline``: entries written to a file of the test's own (never the repo's configs);
    ``baseline_file``: passed as it is instead. ``overrides`` None: a path that does not
    exist. The instants file is always the test's own (``instants``)."""
    if overrides is None:
        overrides_path = tmp_path / "no_such_overrides.yaml"
    else:
        overrides_path = write_yaml(tmp_path / "overrides.yaml", list(overrides))
    if baseline_file is None:
        listed = instants_fingerprint(datetime.fromisoformat(t).astimezone(UTC) for t in instants)
        baseline_file = fingerprinted(tmp_path, baseline or {}, listed)
    instants_path = tmp_path / "instants.txt"
    instants_path.write_text("".join(f"{t}\n" for t in instants))
    args = ["market", "truncation"]
    for value in as_of:
        args += ["--as-of", value]
    args += [
        "--overrides", str(overrides_path),
        "--baseline", str(baseline_file),
        "--instants", str(instants_path),
        *(["--computed-instants"] if computed else []),
        *(["--prune-baseline"] if prune else []),
        *(["--jobs", jobs] if jobs is not None else []),
        *extra,
        "--data-dir", str(tmp_path),
    ]  # fmt: skip
    return CliRunner().invoke(app, args)


def diff_dir(tmp_path: Path) -> Path:
    return market_root(tmp_path) / "audit" / "truncation"


def impact_dir(tmp_path: Path) -> Path:
    return market_root(tmp_path) / "audit" / "truncation_impact"


def written(directory: Path) -> dict[str, pa.Table]:
    return {p.name: pq.read_table(p) for p in sorted(directory.glob("*.parquet"))}


def instant_names(instants: Iterable[datetime]) -> list[str]:
    return sorted(f"{stamp(t)}.parquet" for t in instants)


FIXED = [datetime.fromisoformat(t) for t in CLI_INSTANTS]


def numbers(line: str) -> list[int]:
    return [int(n.replace(",", "")) for n in re.findall(r"\d[\d,]*", line)]


def line_naming(output: str, key: str, *values: int) -> str:
    """The first output line naming ``key`` and every one of ``values`` as a number."""
    for line in output.splitlines():
        if key in line and all(v in numbers(line) for v in values):
            return line
    raise AssertionError(f"no line names {key} with {values}:\n{output}")


def baseline_of(tmp_path: Path) -> dict[str, ImpactCount]:
    return load_impact_baseline(tmp_path / "baseline.txt")


def no_build(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("built although a precondition failed")

    monkeypatch.setattr(sm, "build_security_master", refuse)
    monkeypatch.setattr(cli, "build_from_inputs", refuse, raising=False)


def nothing_written(tmp_path: Path) -> bool:
    return all(
        not d.exists() or not list(d.glob("*")) for d in (diff_dir(tmp_path), impact_dir(tmp_path))
    )


def refused_cleanly(tmp_path: Path, result: Any, name: str) -> None:
    assert result.exit_code not in (0, LEAK_EXIT), result.output
    assert result.exception is None or isinstance(result.exception, SystemExit), result.output
    assert name in result.output, result.output
    assert nothing_written(tmp_path)


# --- normal ---


def test_cli_the_runs_own_attribution_as_baseline_passes(tmp_path: Path) -> None:
    root = write_liquid_store(market_root(tmp_path))
    inputs = sm.load_store_inputs(root)
    full, _ = sm.build_from_inputs(inputs, [MERGE])
    results = run_instants(
        full, inputs, [MERGE], splits_table(), FIXED, liquid={cycn_id(tmp_path)}
    )  # fmt: skip
    found = attribute(results)
    # by hand (module docstring): CYCN@2024-01-02's 31 cells, through KRSA's key
    assert found == Attribution(counts={KEY: C(N, 0)}, unmatched=[])

    result = run_cli(tmp_path, baseline={k: (c.price, c.cik) for k, c in found.counts.items()})

    assert result.exit_code == 0, result.output


def test_cli_writes_the_diff_and_the_impact_of_each_instant(tmp_path: Path) -> None:
    root = write_liquid_store(market_root(tmp_path))

    result = run_cli(tmp_path, baseline=EXACT)

    assert result.exit_code == 0, result.output
    diffs, impacts = written(diff_dir(tmp_path)), written(impact_dir(tmp_path))
    assert list(diffs) == list(impacts) == instant_names(FIXED)
    assert instant_names(FIXED)[0] == "20240301T050000000000.parquet"  # 00:00 New York, as UTC
    inputs = sm.load_store_inputs(root)
    full, _ = sm.build_from_inputs(inputs, [MERGE])
    for t in FIXED:
        truncated = sm.build_from_inputs(inputs, [MERGE], as_of=t)[0]
        diff, rows = diffs[f"{stamp(t)}.parquet"], impacts[f"{stamp(t)}.parquet"]
        assert diff.schema.equals(DIFF_SCHEMA), t
        assert diff.equals(compare(full, truncated, t)), t
        assert rows.schema.equals(IMPACT_SCHEMA), t
        expected = impact_at(
            full, truncated, inputs.bars, splits_table(), t, liquid={cycn_id(tmp_path)}
        )  # fmt: skip
        assert rows.equals(expected), t
    early = impacts[f"{stamp(FIXED[0])}.parquet"].to_pylist()
    assert {(r["kind"], r["what"], r["security"]) for r in early} == {
        ("leak", "price", "CYCN@2024-01-02")
    }
    assert all(r["sessions"] == CLI_DAYS for r in early)
    assert impacts[f"{stamp(FIXED[1])}.parquet"].num_rows == 0


def test_cli_an_empty_baseline_fails_naming_the_new_key_with_its_numbers(tmp_path: Path) -> None:
    write_liquid_store(market_root(tmp_path))
    result = run_cli(tmp_path, baseline={})
    assert result.exit_code == LEAK_EXIT, result.output
    line_naming(result.output, KEY, N)
    # the reports are written whatever the verdict
    assert list(written(impact_dir(tmp_path))) == instant_names(FIXED)


def test_cli_a_baseline_below_the_found_numbers_fails_as_grown(tmp_path: Path) -> None:
    write_liquid_store(market_root(tmp_path))
    result = run_cli(tmp_path, baseline={KEY: (N - 1, 0)})
    assert result.exit_code == LEAK_EXIT, result.output
    line_naming(result.output, KEY, N, N - 1)  # found vs baseline


def test_cli_a_full_run_below_the_baseline_passes_with_a_shrank_notice(tmp_path: Path) -> None:
    write_liquid_store(market_root(tmp_path))
    result = run_cli(tmp_path, baseline={KEY: (N + 5, 0)})
    assert result.exit_code == 0, result.output
    line_naming(result.output, KEY, N, N + 5)
    assert baseline_of(tmp_path) == {KEY: C(N + 5, 0)}  # a notice, not a rewrite


def test_cli_a_full_run_fails_on_a_baseline_key_that_no_longer_occurs(tmp_path: Path) -> None:
    write_liquid_store(market_root(tmp_path))
    result = run_cli(tmp_path, baseline={**EXACT, OTHER_KEY: (3, 0)})
    assert result.exit_code == LEAK_EXIT, result.output
    assert OTHER_KEY in result.output


@pytest.mark.parametrize(
    ("baseline", "exit_code"),
    [
        ({KEY: (N + 5, 0), OTHER_KEY: (3, 0)}, 0),  # neither shrank nor fixed: unprovable
        ({}, LEAK_EXIT),  # still new
        ({KEY: (N - 1, 0)}, LEAK_EXIT),  # still grew
    ],
    ids=["no_fixed_no_shrank", "new", "grew"],
)
def test_cli_a_partial_run_flags_new_and_grew_only(
    tmp_path: Path, baseline: dict[str, tuple[int, int]], exit_code: int
) -> None:
    write_liquid_store(market_root(tmp_path))
    result = run_cli(tmp_path, as_of=["2024-03-01"], baseline=baseline)
    assert result.exit_code == exit_code, result.output
    if exit_code:
        assert KEY in result.output
    assert list(written(diff_dir(tmp_path))) == ["20240301T050000000000.parquet"]
    assert list(written(impact_dir(tmp_path))) == ["20240301T050000000000.parquet"]


def test_cli_computed_instants_are_a_partial_run(tmp_path: Path) -> None:
    # The computed spread is not the fixed list: its sum for KRSA's key is over other instants
    # than the baseline's 31 cells, and OTHER_KEY is not found. Neither may count as grew,
    # shrank or fixed; the files are those of the computed instants.
    root = write_liquid_store(market_root(tmp_path))
    full, _ = sm.build_from_store(root, [MERGE])

    result = run_cli(tmp_path, computed=True, baseline={**EXACT, OTHER_KEY: (3, 0)})

    assert result.exit_code == 0, result.output
    expected = instant_names(default_instants(full))
    assert list(written(diff_dir(tmp_path))) == list(written(impact_dir(tmp_path))) == expected
    assert run_cli(tmp_path, computed=True, baseline={}).exit_code == LEAK_EXIT


def test_cli_a_default_run_checks_exactly_the_listed_instants(tmp_path: Path) -> None:
    root = write_liquid_store(market_root(tmp_path))
    full, _ = sm.build_from_store(root, [MERGE])
    result = run_cli(tmp_path, baseline=EXACT)
    assert result.exit_code == 0, result.output
    assert list(written(diff_dir(tmp_path))) == instant_names(FIXED)
    # the computed spread is reported, not checked
    assert set(default_instants(full)) - {t.astimezone(UTC) for t in FIXED}
    assert "computed instants not in" in result.output


# --- prune ---


def test_cli_prune_removes_fixed_tightens_shrank_and_the_rerun_passes(tmp_path: Path) -> None:
    write_liquid_store(market_root(tmp_path))
    result = run_cli(tmp_path, baseline={KEY: (N + 5, 0), OTHER_KEY: (3, 0)}, prune=True)
    assert result.exit_code == 0, result.output
    assert baseline_of(tmp_path) == {KEY: C(N, 0)}
    assert (tmp_path / "baseline.txt").read_text().startswith(HEADER)

    again = run_cli(tmp_path, baseline_file=tmp_path / "baseline.txt")

    assert again.exit_code == 0, again.output
    assert not [line for line in again.output.splitlines() if N + 5 in numbers(line)]


@pytest.mark.parametrize(
    "baseline",
    [{OTHER_KEY: (3, 0)}, {KEY: (N - 1, 0), OTHER_KEY: (3, 0)}],
    ids=["new_open", "grew_open"],
)
def test_cli_prune_refuses_while_a_key_is_new_or_grew(
    tmp_path: Path, baseline: dict[str, tuple[int, int]]
) -> None:
    # deleting OTHER_KEY must wait for reviewing KRSA's leak
    write_liquid_store(market_root(tmp_path))
    path = fingerprinted(tmp_path, baseline)
    text = path.read_text()
    result = run_cli(tmp_path, baseline_file=path, prune=True)
    assert result.exit_code != 0, result.output
    assert KEY in result.output  # listed, not only counted
    assert path.read_text() == text


@pytest.mark.parametrize("partial", ["as_of", "computed"])
def test_cli_prune_refuses_a_partial_run(tmp_path: Path, partial: str) -> None:
    write_liquid_store(market_root(tmp_path))
    path = fingerprinted(tmp_path, {KEY: (N + 5, 0), OTHER_KEY: (3, 0)})
    text = path.read_text()
    result = run_cli(
        tmp_path, baseline_file=path, prune=True,
        as_of=["2024-03-01"] if partial == "as_of" else (), computed=partial == "computed",
    )  # fmt: skip
    assert result.exit_code != 0, result.output
    assert "--prune-baseline" in result.output
    assert path.read_text() == text


def planted_unmatched(monkeypatch: pytest.MonkeyPatch) -> UnmatchedCell:
    """``cli.attribute`` with one unmatched cell added: the CYCN / KRSA world has none, and a
    world whose leak rows name no diff key at all is not one compare and impact_at produce on
    a store this small."""
    cell = UnmatchedCell(FIXED[0].astimezone(UTC), "price", "ZZZ@2024-01-02", CLI_DAYS[0],
                         ("ZZZ:2024-01-02",))  # fmt: skip
    real = cli.attribute

    def planted(results: Sequence[InstantResult]) -> Attribution:
        got = real(results)
        return dataclasses.replace(got, unmatched=[*got.unmatched, cell])

    monkeypatch.setattr(cli, "attribute", planted)
    return cell


def test_cli_an_unmatched_cell_fails_and_is_shown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_liquid_store(market_root(tmp_path))
    cell = planted_unmatched(monkeypatch)
    result = run_cli(tmp_path, baseline=EXACT)
    assert result.exit_code == LEAK_EXIT, result.output
    assert cell.owner in result.output  # an example, not only the count


def test_cli_prune_refuses_while_a_cell_is_unmatched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_liquid_store(market_root(tmp_path))
    planted_unmatched(monkeypatch)
    path = fingerprinted(tmp_path, {KEY: (N + 5, 0), OTHER_KEY: (3, 0)})
    text = path.read_text()
    result = run_cli(tmp_path, baseline_file=path, prune=True)
    assert result.exit_code != 0, result.output
    assert path.read_text() == text


FIXED_PRINT = instants_fingerprint(t.astimezone(UTC) for t in FIXED)


def fingerprinted(
    tmp_path: Path, entries: Mapping[str, tuple[int, int]], fp: str = FIXED_PRINT
) -> Path:
    """The test's baseline file, its header naming ``fp`` (default: the fixed CLI instants)."""
    header = f"{HEADER}# instants fingerprint: {fp}\n"
    return write_baseline(tmp_path / "baseline.txt", entries, header=header)


def test_cli_prune_keeps_the_fingerprint_of_the_fixed_instants(tmp_path: Path) -> None:
    write_liquid_store(market_root(tmp_path))
    result = run_cli(tmp_path, baseline={KEY: (N + 5, 0)}, prune=True)
    assert result.exit_code == 0, result.output
    assert baseline_fingerprint(tmp_path / "baseline.txt") == FIXED_PRINT


def test_cli_prune_refuses_a_baseline_without_a_fingerprint_before_any_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Reviewer N2: without the line, nothing tells which list the numbers were summed over
    write_liquid_store(market_root(tmp_path))
    path = write_baseline(tmp_path / "baseline.txt", {KEY: (N + 5, 0), OTHER_KEY: (3, 0)})
    text = path.read_text()
    no_build(monkeypatch)
    result = run_cli(tmp_path, baseline_file=path, prune=True)
    refused_cleanly(tmp_path, result, "fingerprint")
    assert path.read_text() == text


def test_cli_prune_refuses_before_any_build_when_the_instants_list_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Reviewer M1. The baseline was summed over FIXED; the instants file now lists only its
    # first instant. A prune would drop OTHER_KEY although the instant it may occur at was
    # merely taken out of the list.
    write_liquid_store(market_root(tmp_path))
    path = fingerprinted(tmp_path, {**EXACT, OTHER_KEY: (3, 0)})
    text = path.read_text()
    no_build(monkeypatch)
    result = run_cli(tmp_path, baseline_file=path, instants=CLI_INSTANTS[:1], prune=True)
    refused_cleanly(tmp_path, result, "fingerprint")
    assert "--rebase-baseline" in result.output
    assert path.read_text() == text


@pytest.mark.parametrize(
    "entries", [{KEY: (N + 99, 0), OTHER_KEY: (3, 0)}, {KEY: (N - 1, 0)}, {OTHER_KEY: (3, 0)}],
    ids=["above", "below", "new_key"],
)  # fmt: skip
def test_cli_a_changed_instants_list_fails_every_default_run(
    tmp_path: Path, entries: dict[str, tuple[int, int]]
) -> None:
    # Reviewer N1. The sums of another list compare with nothing, so nothing may pass on them:
    # a run after the list changed fails until --rebase-baseline re-sums the numbers.
    write_liquid_store(market_root(tmp_path))
    stale = instants_fingerprint([datetime(2030, 1, 1, tzinfo=UTC)])
    path = fingerprinted(tmp_path, entries, stale)
    text = path.read_text()
    result = run_cli(tmp_path, baseline_file=path)
    assert result.exit_code == LEAK_EXIT, result.output
    assert "--rebase-baseline" in result.output
    assert path.read_text() == text


def test_cli_rebase_re_sums_the_baseline_over_the_new_list_and_the_rerun_passes(
    tmp_path: Path,
) -> None:
    # The list loses its second instant (where nothing differs): KRSA's key keeps its 31 cells
    # of 2024-03-01, whatever the old number; OTHER_KEY has no impact on the new list and goes.
    write_liquid_store(market_root(tmp_path))
    path = fingerprinted(tmp_path, {KEY: (N + 7, 0), OTHER_KEY: (3, 0)})
    one = CLI_INSTANTS[:1]

    result = run_cli(tmp_path, baseline_file=path, instants=one, extra=["--rebase-baseline"])

    assert result.exit_code == 0, result.output
    assert load_impact_baseline(path) == {KEY: C(N, 0)}
    new_print = instants_fingerprint([datetime.fromisoformat(one[0]).astimezone(UTC)])
    assert baseline_fingerprint(path) == new_print
    assert path.read_text().startswith(HEADER)
    line_naming(result.output, KEY, N + 7, N)  # every change printed, old -> new
    assert OTHER_KEY in result.output
    assert run_cli(tmp_path, baseline_file=path, instants=one).exit_code == 0


def test_cli_rebase_may_raise_a_number_only_after_the_list_changed(tmp_path: Path) -> None:
    # adding an instant adds its cells: the re-summed number may be higher, printed as such
    write_liquid_store(market_root(tmp_path))
    old = instants_fingerprint([datetime.fromisoformat(CLI_INSTANTS[1]).astimezone(UTC)])
    path = fingerprinted(tmp_path, {KEY: (1, 0)}, old)
    result = run_cli(tmp_path, baseline_file=path, extra=["--rebase-baseline"])
    assert result.exit_code == 0, result.output
    assert load_impact_baseline(path) == {KEY: C(N, 0)}
    assert baseline_fingerprint(path) == FIXED_PRINT
    assert line_naming(result.output, KEY, 1, N).lstrip().startswith("rose:")  # set apart


def test_cli_a_baseline_without_a_fingerprint_fails_a_default_run(tmp_path: Path) -> None:
    # Reviewer L-a: a non-empty baseline naming no instant list compares with nothing either
    write_liquid_store(market_root(tmp_path))
    path = write_baseline(tmp_path / "baseline.txt", EXACT)
    result = run_cli(tmp_path, baseline_file=path)
    assert result.exit_code == LEAK_EXIT, result.output
    assert "--rebase-baseline" in result.output
    rebased = run_cli(tmp_path, baseline_file=path, extra=["--rebase-baseline"])
    assert rebased.exit_code == 0, rebased.output
    assert baseline_fingerprint(path) == FIXED_PRINT
    assert run_cli(tmp_path, baseline_file=path).exit_code == 0


def test_cli_rebase_refuses_a_list_that_did_not_change_before_any_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Leakage: re-summing over the same list could only lift a number past what the gate allows
    write_liquid_store(market_root(tmp_path))
    path = fingerprinted(tmp_path, {KEY: (N - 1, 0)})
    text = path.read_text()
    no_build(monkeypatch)
    result = run_cli(tmp_path, baseline_file=path, extra=["--rebase-baseline"])
    refused_cleanly(tmp_path, result, "nothing to rebase")
    assert path.read_text() == text


def test_cli_rebase_refuses_a_new_key(tmp_path: Path) -> None:
    write_liquid_store(market_root(tmp_path))
    stale = instants_fingerprint([datetime(2030, 1, 1, tzinfo=UTC)])
    path = fingerprinted(tmp_path, {OTHER_KEY: (3, 0)}, stale)
    text = path.read_text()
    result = run_cli(tmp_path, baseline_file=path, extra=["--rebase-baseline"])
    assert result.exit_code not in (0, LEAK_EXIT), result.output
    assert KEY in result.output  # listed, not only counted
    assert path.read_text() == text


@pytest.mark.parametrize("other", ["prune", "as_of", "computed"])
def test_cli_rebase_refuses_with_prune_or_a_partial_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, other: str
) -> None:
    write_liquid_store(market_root(tmp_path))
    stale = instants_fingerprint([datetime(2030, 1, 1, tzinfo=UTC)])
    path = fingerprinted(tmp_path, EXACT, stale)
    no_build(monkeypatch)
    result = run_cli(
        tmp_path, baseline_file=path, extra=["--rebase-baseline"], prune=other == "prune",
        as_of=["2024-03-01"] if other == "as_of" else (), computed=other == "computed",
    )  # fmt: skip
    refused_cleanly(tmp_path, result, "--rebase-baseline")


def test_cli_an_instant_outside_the_fixed_list_judges_no_growth(tmp_path: Path) -> None:
    # 2024-04-01 is in no --instants line: its sum is over another instant than the baseline's
    write_liquid_store(market_root(tmp_path))
    result = run_cli(tmp_path, as_of=["2024-04-01"], baseline={KEY: (1, 0)})
    assert result.exit_code == 0, result.output
    assert "outside the fixed list" in result.output
    assert run_cli(tmp_path, as_of=["2024-04-01"], baseline={}).exit_code == LEAK_EXIT


# --- report directories ---


@pytest.mark.parametrize("run", ["full", "partial"])
def test_cli_only_a_full_run_removes_instant_files_of_earlier_runs(
    tmp_path: Path, run: str
) -> None:
    # an instant dropped from --instants must not linger beside the run's own files, where a
    # reader of the directory would take it for part of this run; a partial run adds its own
    write_liquid_store(market_root(tmp_path))
    stale = "20190101T000000000000.parquet"
    for directory, schema in (
        (diff_dir(tmp_path), DIFF_SCHEMA),
        (impact_dir(tmp_path), IMPACT_SCHEMA),
    ):
        directory.mkdir(parents=True)
        pq.write_table(schema.empty_table(), directory / stale)

    result = run_cli(tmp_path, baseline=EXACT, as_of=["2024-03-01"] if run == "partial" else ())

    assert result.exit_code == 0, result.output
    own = instant_names(FIXED) if run == "full" else ["20240301T050000000000.parquet"]
    expected = own if run == "full" else sorted([stale, *own])
    assert list(written(diff_dir(tmp_path))) == expected
    assert list(written(impact_dir(tmp_path))) == expected


# --- the liquid tier and the splits ---


def test_cli_liquid_is_the_tier_union_from_2020_to_the_last_instant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_liquid_store(market_root(tmp_path))
    spans: list[tuple[date, date]] = []
    real_union = cli.tier_union

    def union_spy(con: Any, root: Path, start: date, end: date, **kwargs: Any) -> pa.Table:
        spans.append((start, end))
        return real_union(con, root, start, end, **kwargs)

    used: list[Any] = []
    real_run = cli.run_instants

    def run_spy(*args: Any, **kwargs: Any) -> Any:
        used.append(kwargs.get("liquid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(cli, "tier_union", union_spy)
    monkeypatch.setattr(cli, "run_instants", run_spy)

    result = run_cli(tmp_path, baseline=EXACT)

    assert result.exit_code == 0, result.output
    [(start, end)] = spans
    assert start == date(2020, 1, 1)
    assert end == CLI_DECIDED  # the New York date of the last instant
    # the fixture guard: the real tier over these months is CYCN's merged security
    con = duckdb.connect()
    try:
        tier = tier_union(con, market_root(tmp_path), start, end)
    finally:
        con.close()
    assert set(tier.column("security_id").to_pylist()) == {cycn_id(tmp_path)}
    [liquid] = used
    assert set(liquid) == {cycn_id(tmp_path)}


def test_cli_liquid_union_ends_at_the_last_fixed_instant_also_for_as_of_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The liquid set must not depend on which instants a run picks: a partial run at 2024-03-01
    # reads the same tier as the full run, through the last instant of --instants (2024-06-03).
    write_liquid_store(market_root(tmp_path))
    spans: list[tuple[date, date]] = []
    real_union = cli.tier_union

    def union_spy(con: Any, root: Path, start: date, end: date, **kwargs: Any) -> pa.Table:
        spans.append((start, end))
        return real_union(con, root, start, end, **kwargs)

    monkeypatch.setattr(cli, "tier_union", union_spy)

    result = run_cli(tmp_path, as_of=["2024-03-01"], baseline=EXACT)

    assert result.exit_code == 0, result.output
    assert spans == [(date(2020, 1, 1), CLI_DECIDED)]


def test_cli_splits_are_those_visible_at_the_last_instant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # impact_at drops splits filed after each instant itself; the store read happens once
    write_liquid_store(market_root(tmp_path))
    seen: list[datetime] = []
    real = cli.visible_splits

    def spy(con: Any, root: Path, as_of: datetime, *args: Any, **kwargs: Any) -> pa.Table:
        seen.append(as_of)
        return real(con, root, as_of, *args, **kwargs)

    monkeypatch.setattr(cli, "visible_splits", spy)
    result = run_cli(tmp_path, baseline=EXACT)
    assert result.exit_code == 0, result.output
    assert seen == [FIXED[-1]]


# --- refusals ---


def test_cli_refuses_a_missing_overrides_file_before_any_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_liquid_store(market_root(tmp_path))
    no_build(monkeypatch)
    result = run_cli(tmp_path, overrides=None, baseline=EXACT)
    refused_cleanly(tmp_path, result, "no_such_overrides.yaml")


@pytest.mark.parametrize(
    "missing",
    [sm.MASTER_FILE, CLASS_FILE, "corporate_actions/splits.parquet"],
    ids=["master", "class", "splits"],
)
def test_cli_refuses_a_missing_store_file_before_any_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    # the liquid tier reads the master and the classes; without the split file every split
    # would read as "none" and a price leak through one would vanish
    root = write_liquid_store(market_root(tmp_path))
    (root / missing).unlink()
    no_build(monkeypatch)
    result = run_cli(tmp_path, baseline=EXACT)
    refused_cleanly(tmp_path, result, Path(missing).name)


def test_cli_refuses_a_malformed_baseline_before_any_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_liquid_store(market_root(tmp_path))
    path = tmp_path / "baseline.txt"
    path.write_text(f"{KEY} price=31\n")  # no cik
    no_build(monkeypatch)
    result = run_cli(tmp_path, baseline_file=path)
    refused_cleanly(tmp_path, result, "baseline.txt")


def test_cli_refuses_an_on_disk_master_other_than_the_one_the_inputs_build(tmp_path: Path) -> None:
    # The liquid tier names the ids of the master on disk (built with the merge); without the
    # merge override the inputs build another master, whose ids the tier would misname.
    write_liquid_store(market_root(tmp_path))
    result = run_cli(tmp_path, overrides=[], baseline=EXACT)
    refused_cleanly(tmp_path, result, Path(sm.MASTER_FILE).name)


def test_cli_a_missing_baseline_file_is_an_empty_baseline(tmp_path: Path) -> None:
    # a mistyped --baseline fails the run (every key new), never passes it
    write_liquid_store(market_root(tmp_path))
    result = run_cli(tmp_path, baseline_file=tmp_path / "no_such_baseline.txt")
    assert result.exit_code == LEAK_EXIT, result.output
    line_naming(result.output, KEY, N)


def test_cli_refuses_as_of_with_computed_instants(tmp_path: Path) -> None:
    write_liquid_store(market_root(tmp_path))
    result = run_cli(tmp_path, as_of=["2024-03-01"], computed=True, baseline=EXACT)
    assert result.exit_code not in (0, LEAK_EXIT), result.output
    assert "--computed-instants" in result.output
    assert nothing_written(tmp_path)


def test_cli_without_instants_refuses_to_guess(tmp_path: Path) -> None:
    write_liquid_store(market_root(tmp_path))
    result = run_cli(tmp_path, instants=[], baseline=EXACT)
    assert result.exit_code not in (0, LEAK_EXIT), result.output
    assert "no instants" in result.output
    assert nothing_written(tmp_path)


def test_cli_the_truncation_impact_command_is_gone(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["market", "truncation-impact", "--data-dir", str(tmp_path)])
    assert result.exit_code != 0
    assert "No such command" in result.output


def test_cli_the_exceptions_option_is_gone(tmp_path: Path) -> None:
    write_liquid_store(market_root(tmp_path))
    exceptions = write_yaml(tmp_path / "exceptions.yaml", [])
    result = run_cli(tmp_path, baseline=EXACT, extra=["--exceptions", str(exceptions)])
    assert result.exit_code != 0
    assert "No such option" in result.output
    assert nothing_written(tmp_path)


# --- --jobs ---


def verdict_lines(output: str) -> list[str]:
    """The lines naming a key: the verdict, without per-instant timings."""
    return [line for line in output.splitlines() if KEY in line or OTHER_KEY in line]


@pytest.mark.parametrize(
    "baseline",
    [EXACT, {}, {**EXACT, OTHER_KEY: (3, 0)}, {KEY: (N + 5, 0)}],
    ids=["exact", "new", "fixed", "shrank"],
)
def test_cli_reports_and_verdict_do_not_depend_on_jobs(
    tmp_path: Path, baseline: dict[str, tuple[int, int]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "_cores", lambda: 4)  # --jobs 2 must not depend on this machine
    runs = {}
    for jobs in ("1", "2", None):  # None: the default, min(4, cores)
        here = tmp_path / f"jobs_{jobs}"
        here.mkdir()
        write_liquid_store(market_root(here))
        result = run_cli(here, baseline=baseline, jobs=jobs)
        output = result.output.replace(str(here), "<run>")
        runs[jobs] = (result.exit_code, verdict_lines(output), written(diff_dir(here)),
                      written(impact_dir(here)), (here / "baseline.txt").read_text())  # fmt: skip

    serial = runs["1"]
    assert list(serial[2]) == list(serial[3]) == instant_names(FIXED)
    for jobs in ("2", None):
        code, lines, diffs, impacts, kept = runs[jobs]
        assert code == serial[0], jobs
        assert lines == serial[1], jobs
        for got, expected in ((diffs, serial[2]), (impacts, serial[3])):
            assert got.keys() == expected.keys(), jobs
            for name, table in got.items():
                assert table.equals(expected[name]), (jobs, name)
        assert kept == serial[4], jobs


def test_cli_prune_with_jobs_prunes_as_serial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "_cores", lambda: 4)
    write_liquid_store(market_root(tmp_path))
    result = run_cli(
        tmp_path, baseline={KEY: (N + 5, 0), OTHER_KEY: (3, 0)}, prune=True, jobs="2"
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert baseline_of(tmp_path) == {KEY: C(N, 0)}


@pytest.mark.parametrize(("cores", "expected"), [(2, 2), (32, 4)])
def test_cli_default_jobs_is_the_smaller_of_4_and_the_cores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cores: int, expected: int
) -> None:
    # 2026-10-20 on the real store: 6 workers ran out of a 31 GB machine, 4 peaked at 22 GB; a
    # CI runner has 2-4 cores and must not be refused
    monkeypatch.setattr(cli, "_cores", lambda: cores)
    used: list[int] = []
    real = cli.run_instants

    def spy(*args: Any, **kwargs: Any) -> Any:
        used.append(kwargs["jobs"])
        return real(*args, **kwargs)

    monkeypatch.setattr(cli, "run_instants", spy)
    write_liquid_store(market_root(tmp_path))
    result = run_cli(tmp_path, baseline=EXACT)
    assert result.exit_code == 0, result.output
    assert used == [expected]


def test_cli_refuses_more_jobs_than_cores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_cores", lambda: 2)
    write_liquid_store(market_root(tmp_path))
    result = run_cli(tmp_path, baseline=EXACT, jobs="3")
    assert result.exit_code not in (0, LEAK_EXIT), result.output
    assert "between 1 and 2" in result.output


@pytest.mark.parametrize("jobs", ["0", "-2"])
def test_cli_refuses_fewer_than_one_job_before_any_build(
    tmp_path: Path, jobs: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_liquid_store(market_root(tmp_path))
    no_build(monkeypatch)
    result = run_cli(tmp_path, baseline=EXACT, jobs=jobs)
    refused_cleanly(tmp_path, result, "jobs")


# =============================================================================================
# the repo's files
# =============================================================================================

REPO_CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def test_repo_impact_baseline_and_instants_load() -> None:
    # read only: a hand-added line that is malformed or duplicated fails CI, not 26 minutes into
    # a real run
    path = REPO_CONFIGS / "truncation_impact_baseline.txt"
    if not path.exists():
        pytest.skip(f"{path} not written yet (created by hand from the first real run)")
    assert load_impact_baseline(path)
    instants = load_instants(REPO_CONFIGS / "truncation_instants.txt")
    assert instants
    # reviewer N2: the numbers are sums over exactly this list; a changed list must be rebased
    # (`asof market truncation --rebase-baseline`) in the same change
    assert baseline_fingerprint(path) == instants_fingerprint(instants)


def test_repo_has_no_diff_key_ratchet_files_left() -> None:
    # the old key ratchet and its exceptions would mislead a reader into thinking they gate
    assert not (REPO_CONFIGS / "truncation_baseline.txt").exists()
    assert not (REPO_CONFIGS / "truncation_exceptions.yaml").exists()
