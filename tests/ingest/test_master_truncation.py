"""Guards against the security master leaking the future into its past: a segment, a grouping or
a field that a reader at ``as_of`` sees in the full master, but that the build could not have
produced from the inputs knowable at ``as_of``. Some build rules read data after an event (rule
B reads SEC snapshots up to 183 days after a rename record; the copy rules read the bars on both
sides of a takeover), and a row stamped knowable at the event may carry what they found.

The test rebuilds the master from the inputs knowable at ``as_of`` (the "truncated" master) and
compares it with the full master as seen at ``as_of``. A row the full master shows and the
truncated one cannot is a leak; a row only the truncated master shows, on days the full master
leaves to nobody, is a delay (the full master learned later that the row was wrong and withdrew
it; nothing is seen early).

Target API (implementation pending, acceptance point 6-d):

    asof.ingest.security_master.build_from_store(market_root, overrides=(), *, as_of=None)
        None: today's behaviour; an aware ``as_of``: build from the inputs knowable then.
    asof.ingest.master_truncation:
        truncate_inputs(*, bars, sec_history, name_changes, overrides, as_of)
            -> (bars, sec_history, name_changes, overrides)
        mask_master(master, as_of) -> pa.Table          visible_master's masking, on a table
        DIFF_SCHEMA = kind (leak | delay | info), reason, symbol, valid_from, fields, detail
        diff_key(row) = SYMBOL:YYYY-MM-DD:reason[:field+field...]
        compare(full, truncated, as_of) -> pa.Table     DIFF_SCHEMA rows
        default_instants(full_master) -> list[datetime]
    asof.ingest.security_master (acceptance point 6-f, parallel instants):
        load_store_inputs(market_root) -> StoreInputs     frozen: bars, sec_history, name_changes
        build_from_inputs(inputs, overrides=(), *, as_of=None) -> (master, conflicts)
            build_from_store(root, ...) == build_from_inputs(load_store_inputs(root), ...)
    asof.ingest.master_truncation.run_instants(full, inputs, overrides, instants, *, jobs=1)
        -> [(instant UTC, compare(full, truncated at instant, instant))], sorted, one per distinct
        instant; jobs > 1 = a fork process pool, results identical to jobs=1; jobs < 1 refused.
    asof market truncation [--as-of ISO]... [--exceptions PATH] [--overrides PATH] [--data-dir]
        [--jobs N]  (default min(6, cores); outside 1..cores refused before any build;
                     output identical whatever N)
        diffs -> <market>/audit/truncation/<YYYYMMDDTHHMMSSffffff of the UTC instant>.parquet;
        exit 2 on a leak whose key ``SYMBOL:YYYY-MM-DD:reason`` is not excepted, 0 otherwise.

Fake data provenance:
- The scenario worlds are the ones of tests/ingest/test_security_master*.py, read through
  tests/ingest/master_scenarios.py; their provenance notes apply.
- A daily bar is knowable from 20:00 New York of its session (ingest.bars, real: the 2020-01-02
  bar has available_at 2020-01-03 01:00 UTC); an Alpaca name change from its ``available_at``
  (00:00 New York of process_date when null, as ``edges_from_name_changes`` reads it).
- ASSUMPTION: an SEC ticker snapshot is knowable from 00:00 New York of its snapshot_date (the
  build already dates SEC-created segments that way); a human override from 00:00 New York of
  its ``date``.
- The CLI store is constructed: CYCN / KRSA carry identical bars (the real pair has 1,609
  identical days and no record anywhere) and a dated merge override joins them.

Real-data checks for the assumptions (audit command, then the standard cases):
- ``asof market truncation`` on the real store at the default instants: every leak row is read
  by a human; none may remain outside configs/truncation_exceptions.yaml.
- Standard cases: BK / BNY seen at 2022-06-01 shows BK open and no BNY (no diff); BBBY seen at
  2025-08-07 shows Bed Bath & Beyond's closed piece and no Beyond Inc piece yet (no diff).
"""

import dataclasses
import multiprocessing
import os
import signal
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from typer.testing import CliRunner

from asof import cli
from asof.cli import app
from asof.ingest import master_truncation as mt
from asof.ingest import security_master as sm
from asof.ingest.corporate_actions import NAME_CHANGE_SCHEMA
from asof.ingest.master_truncation import (
    DIFF_SCHEMA,
    compare,
    default_instants,
    diff_key,
    load_baseline,
    load_truncation_exceptions,
    mask_master,
    truncate_inputs,
)
from asof.ingest.sec_ticker_history import HISTORY_SCHEMA
from asof.ingest.security_master import MASTER_SCHEMA, build_security_master
from asof.store.market import market_root, visible_master
from tests.ingest.fakes import daily
from tests.ingest.master_scenarios import SCENARIOS, Inputs
from tests.ingest.test_security_master import (
    bar_rows,
    bars_table,
    business_days,
    et_midnight,
    master_table,
    name_changes_table,
    sec_table,
    seg_row,
)
from tests.store.test_market import store_daily
from tests.store.test_market_ext import ext_row, write_ext

MICRO = timedelta(microseconds=1)
T = et_midnight(date(2020, 6, 1))  # the as_of of the truncate_inputs cases
NAMED_DAYS = [
    date(2020, 2, 13),
    date(2020, 6, 8),
    date(2025, 8, 7),
    date(2025, 7, 25),
    date(2026, 10, 5),
]
Diff = tuple[str, str, str, date]  # kind, reason, symbol, valid_from


def diffs(table: pa.Table) -> set[Diff]:
    return {(r["kind"], r["reason"], r["symbol"], r["valid_from"]) for r in table.to_pylist()}


def build(inputs: Inputs, overrides: Sequence[Mapping[str, Any]] = ()) -> pa.Table:
    master, _ = build_security_master(
        bars=inputs.bars,
        sec_history=inputs.sec_history,
        name_changes=inputs.name_changes,
        overrides=overrides,
        sec_history_start=inputs.sec_history_start,
    )
    return master


def build_truncated(
    inputs: Inputs, as_of: datetime, overrides: Sequence[Mapping[str, Any]] = ()
) -> pa.Table:
    bars, sec, changes, kept = truncate_inputs(
        bars=inputs.bars,
        sec_history=inputs.sec_history,
        name_changes=inputs.name_changes,
        overrides=overrides,
        as_of=as_of,
    )
    return build(Inputs(bars, sec, changes, inputs.sec_history_start), kept)


# =============================================================================================
# truncate_inputs
# =============================================================================================


def changes_table(rows: Sequence[tuple[str, str, date, datetime | None]]) -> pa.Table:
    """Alpaca name changes with an explicit (possibly null, possibly late) available_at."""
    return pa.Table.from_pylist(
        [
            {
                "old_symbol": old,
                "new_symbol": new,
                "process_date": day,
                "old_cusip": None,
                "new_cusip": None,
                "available_at": known,
            }
            for old, new, day, known in rows
        ],
        schema=NAME_CHANGE_SCHEMA,
    )


def history_table(rows: Sequence[tuple[date, str, int]]) -> pa.Table:
    return pa.Table.from_pylist(
        [{"snapshot_date": d, "ticker": t, "cik": c, "name": f"{t} CO"} for d, t, c in rows],
        schema=HISTORY_SCHEMA,
    )


def merge(day: date | str, *symbols: str) -> dict[str, Any]:
    return {"merge": list(symbols), "reason": "identical history", "date": day}


def truncate(
    *,
    bars: pa.Table | None = None,
    sec: pa.Table | None = None,
    changes: pa.Table | None = None,
    overrides: Sequence[Mapping[str, Any]] = (),
    as_of: datetime = T,
) -> tuple[pa.Table, pa.Table, pa.Table, Sequence[Mapping[str, Any]]]:
    return truncate_inputs(
        bars=bars_table([]) if bars is None else bars,
        sec_history=history_table([]) if sec is None else sec,
        name_changes=changes_table([]) if changes is None else changes,
        overrides=overrides,
        as_of=as_of,
    )


def test_truncate_inputs_keeps_what_was_knowable_at_as_of() -> None:
    bars = bars_table(bar_rows("AAA", date(2020, 5, 27), date(2020, 6, 3), 10.0))
    sec = history_table(
        [(date(2020, 5, 1), "AAA", 1), (date(2020, 6, 1), "AAA", 1), (date(2020, 7, 1), "AAA", 1)]
    )
    changes = changes_table(
        [
            ("OLD1", "NEW1", date(2020, 5, 15), et_midnight(date(2020, 5, 15))),
            ("OLD2", "NEW2", date(2020, 6, 2), et_midnight(date(2020, 6, 2))),
        ]
    )
    early, late = merge(date(2020, 5, 20), "AAA", "BBB"), merge("2020-07-01", "CCC", "DDD")

    out_bars, out_sec, out_changes, out_overrides = truncate(
        bars=bars, sec=sec, changes=changes, overrides=[early, late]
    )

    # 05-29 (a Friday) is visible from 05-29 20:00; the 06-01 bar only from 06-01 20:00
    assert out_bars.column("session_date").to_pylist() == [
        date(2020, 5, 27),
        date(2020, 5, 28),
        date(2020, 5, 29),
    ]
    assert out_sec.column("snapshot_date").to_pylist() == [date(2020, 5, 1), date(2020, 6, 1)]
    assert out_changes.column("old_symbol").to_pylist() == ["OLD1"]
    assert list(out_overrides) == [early]


@pytest.mark.parametrize("knowable", ["at_as_of", "one_microsecond_later"])
@pytest.mark.parametrize("source", ["bars", "sec", "changes", "overrides"])
def test_truncate_inputs_keeps_a_row_knowable_exactly_at_as_of(source: str, knowable: str) -> None:
    # Every input row below becomes knowable at exactly T (2020-06-01 00:00 New York); at T it
    # is kept, at T - 1us it is not.
    as_of = T if knowable == "at_as_of" else T - MICRO
    kept = knowable == "at_as_of"
    if source == "bars":
        row = {**bar_rows("AAA", date(2020, 5, 29), date(2020, 5, 29), 10.0)[0], "available_at": T}
        out = truncate(bars=bars_table([row]), as_of=as_of)[0]
    elif source == "sec":
        out = truncate(sec=history_table([(date(2020, 6, 1), "AAA", 1)]), as_of=as_of)[1]
    elif source == "changes":
        out = truncate(changes=changes_table([("O", "N", date(2020, 6, 1), T)]), as_of=as_of)[2]
    else:
        overrides = truncate(overrides=[merge(date(2020, 6, 1), "A", "B")], as_of=as_of)[3]
        assert len(list(overrides)) == int(kept)
        return
    assert out.num_rows == int(kept)


def test_truncate_inputs_override_date_may_be_an_iso_string() -> None:
    as_date, as_text = merge(date(2020, 6, 1), "A", "B"), merge("2020-06-01", "C", "D")
    assert list(truncate(overrides=[as_date, as_text])[3]) == [as_date, as_text]
    assert list(truncate(overrides=[as_date, as_text], as_of=T - MICRO)[3]) == []


def test_truncate_inputs_late_filed_rename_is_dropped_although_its_date_has_passed() -> None:
    # Leakage. D rule shape (real: SVMK -> MNTV traded from 2021-06-15, filed 2021-08-09): the
    # record's process_date is before as_of but nobody could read it until it was filed.
    changes = changes_table([("SVMK", "MNTV", date(2020, 5, 15), et_midnight(date(2020, 7, 6)))])
    assert truncate(changes=changes)[2].num_rows == 0
    assert truncate(changes=changes, as_of=et_midnight(date(2020, 7, 6)))[2].num_rows == 1


def test_truncate_inputs_bar_of_the_as_of_session_is_not_visible_before_the_evening() -> None:
    # Leakage. The session of 2020-06-01 is on disk; at 16:00 New York its bar is not knowable.
    bars = bars_table(bar_rows("AAA", date(2020, 6, 1), date(2020, 6, 1), 10.0))
    afternoon = T + timedelta(hours=16)
    assert truncate(bars=bars, as_of=afternoon)[0].num_rows == 0
    assert truncate(bars=bars, as_of=T + timedelta(hours=20))[0].num_rows == 1


def test_truncate_inputs_null_available_at_is_midnight_of_the_process_date() -> None:
    changes = changes_table(
        [("OLD1", "NEW1", date(2020, 6, 1), None), ("OLD2", "NEW2", date(2020, 6, 2), None)]
    )
    assert truncate(changes=changes)[2].column("old_symbol").to_pylist() == ["OLD1"]
    assert truncate(changes=changes, as_of=T - MICRO)[2].num_rows == 0


def test_truncate_inputs_keeps_schemas_even_when_everything_is_dropped() -> None:
    bars = bars_table(bar_rows("AAA", date(2020, 5, 27), date(2020, 6, 3), 10.0))
    sec = history_table([(date(2020, 5, 1), "AAA", 1)])
    changes = changes_table([("O", "N", date(2020, 5, 15), None)])
    for as_of in (T, datetime(2000, 1, 1, tzinfo=UTC)):
        out_bars, out_sec, out_changes, _ = truncate(
            bars=bars, sec=sec, changes=changes, as_of=as_of
        )
        assert out_bars.schema.equals(bars.schema)
        assert out_sec.schema.equals(sec.schema)
        assert out_changes.schema.equals(changes.schema)
    assert truncate(bars=bars, as_of=datetime(2000, 1, 1, tzinfo=UTC))[0].num_rows == 0


def test_truncate_inputs_refuses_an_override_without_a_date() -> None:
    with pytest.raises(ValueError, match="date"):
        truncate(overrides=[{"merge": ["A", "B"], "reason": "identical history"}])


def test_truncate_inputs_refuses_a_naive_as_of() -> None:
    with pytest.raises(ValueError):
        truncate(as_of=datetime(2020, 6, 1))


# =============================================================================================
# mask_master: visible_master's masking on a table
# =============================================================================================

D0, CUT, LATER = date(2020, 1, 2), date(2020, 6, 1), date(2020, 9, 1)
AS_OF = et_midnight(date(2021, 1, 4))  # every fixture row below is knowable


def test_mask_master_hides_unknown_rows_and_unknown_ends() -> None:
    master = master_table(
        [
            seg_row("S2", "B", D0, CUT),  # its end knowable at 00:00 of CUT
            seg_row("S1", "A", D0),
            seg_row("S3", "C", LATER),  # not knowable before LATER
        ]
    )
    before_cut = mask_master(master, et_midnight(CUT) - MICRO)

    assert before_cut.schema.equals(MASTER_SCHEMA)
    assert [(r["security_id"], r["symbol"], r["valid_to"]) for r in before_cut.to_pylist()] == [
        ("S1", "A", None),
        ("S2", "B", None),  # BK in 2022: listed open, the end not yet known
    ]
    assert before_cut.column("end_available_at").to_pylist() == [None, None]
    at_cut = mask_master(master, et_midnight(CUT))
    assert at_cut.to_pylist()[1]["valid_to"] == CUT
    assert at_cut.to_pylist()[1]["end_available_at"] == et_midnight(CUT)
    assert [r["symbol"] for r in mask_master(master, et_midnight(LATER)).to_pylist()] == [
        "A",
        "B",
        "C",
    ]
    assert mask_master(master, et_midnight(D0) - MICRO).num_rows == 0
    assert mask_master(master_table([]), AS_OF).schema.equals(MASTER_SCHEMA)


def test_mask_master_hides_a_cik_until_it_is_knowable() -> None:
    # Leakage. AAAU trades from 2020-01-02; the SEC table first lists it on 2020-07-10.
    listed = et_midnight(date(2020, 7, 10))
    master = master_table([seg_row("S1", "AAAU", D0, cik=1708646, cik_available_at=listed)])
    [before] = mask_master(master, listed - MICRO).to_pylist()
    assert before["cik"] is None and before["cik_available_at"] is None
    [at] = mask_master(master, listed).to_pylist()
    assert (at["cik"], at["cik_available_at"]) == (1708646, listed)


def test_mask_master_orders_by_security_valid_from_symbol() -> None:
    master = master_table(
        [seg_row("S2", "Z", D0), seg_row("S1", "B", CUT), seg_row("S1", "A", CUT)]
    )
    assert [(r["security_id"], r["symbol"]) for r in mask_master(master, AS_OF).to_pylist()] == [
        ("S1", "A"),
        ("S1", "B"),
        ("S2", "Z"),
    ]


def test_mask_master_refuses_a_naive_as_of() -> None:
    with pytest.raises(ValueError):
        mask_master(master_table([seg_row("S1", "A", D0)]), datetime(2021, 1, 4))


def test_mask_master_matches_visible_master_on_a_written_master(tmp_path: Path) -> None:
    world = SCENARIOS["world"]()
    master, conflicts = build_security_master(
        bars=world.bars, sec_history=world.sec_history, name_changes=world.name_changes
    )
    root = market_root(tmp_path)
    sm.write_master(master, conflicts, root)
    instants = {
        t
        for t in master.column("available_at").to_pylist()
        + master.column("end_available_at").to_pylist()
        if t is not None
    }
    instants |= {t - MICRO for t in instants}
    instants |= {datetime(2019, 1, 1, tzinfo=UTC), datetime(2030, 1, 1, tzinfo=UTC)}
    con = duckdb.connect()
    try:
        for t in sorted(instants):
            assert mask_master(master, t).equals(visible_master(con, root, t)), t
    finally:
        con.close()


# =============================================================================================
# compare
# =============================================================================================


def rows_with(rows: Sequence[dict[str, Any]], symbol: str, **fields: Any) -> list[dict[str, Any]]:
    return [{**r, **fields} if r["symbol"] == symbol else r for r in rows]


def good_rows() -> list[dict[str, Any]]:
    return [
        seg_row("S1", "A", D0, cik=1),
        seg_row("S2", "B", D0, CUT, cik=2),
        seg_row("S2", "C", CUT, cik=2),  # B renamed to C
        seg_row("S3", "D", D0, LATER),
        seg_row("S4", "D", LATER),  # D vacated and retaken the same day
    ]


def test_diff_schema() -> None:
    assert DIFF_SCHEMA.names == ["kind", "reason", "symbol", "valid_from", "fields", "detail"]
    assert DIFF_SCHEMA.field("fields").type == pa.string()
    assert DIFF_SCHEMA.field("kind").type == pa.string()
    assert DIFF_SCHEMA.field("reason").type == pa.string()
    assert DIFF_SCHEMA.field("symbol").type == pa.string()
    assert DIFF_SCHEMA.field("valid_from").type == pa.date32()
    assert DIFF_SCHEMA.field("detail").type == pa.string()


def test_identical_masters_have_no_diff() -> None:
    table = master_table(good_rows())
    out = compare(table, table, AS_OF)
    assert out.schema.equals(DIFF_SCHEMA)
    assert out.num_rows == 0


def test_security_id_numbering_is_not_a_diff() -> None:
    # ids are positional (S000001...): a truncated build numbers its securities afresh
    renumber = {"S1": "S9", "S2": "S7", "S3": "S8", "S4": "S1"}
    truncated = [{**r, "security_id": renumber[r["security_id"]]} for r in good_rows()]
    assert compare(master_table(good_rows()), master_table(truncated), AS_OF).num_rows == 0


def test_row_only_in_full_is_a_leak() -> None:
    full = [*good_rows(), seg_row("S5", "E", CUT)]
    out = compare(master_table(full), master_table(good_rows()), AS_OF)
    assert diffs(out) == {("leak", "only_in_full", "E", CUT)}


@pytest.mark.parametrize(
    ("symbol", "fields"),
    [
        ("A", {"cik": 99}),
        ("A", {"cusip": "000000AA1"}),
        ("A", {"available_at": et_midnight(D0) + timedelta(hours=1)}),
        ("B", {"end_available_at": et_midnight(CUT) + timedelta(days=1)}),
        ("B", {"valid_to": LATER, "end_available_at": et_midnight(LATER)}),
        ("A", {"cik": 99, "evidence": "override"}),
    ],
    ids=["cik", "cusip", "available_at", "end_available_at", "valid_to", "two"],
)
def test_same_key_with_different_fields_is_one_leak_naming_them(
    symbol: str, fields: dict[str, Any]
) -> None:
    full = rows_with(good_rows(), symbol, **fields)
    out = compare(master_table(full), master_table(good_rows()), AS_OF)
    [row] = out.to_pylist()
    assert (row["kind"], row["reason"], row["symbol"], row["valid_from"]) == (
        "leak",
        "fields_differ",
        symbol,
        D0,
    )
    for name in fields:
        assert name in row["detail"], (name, row["detail"])
    # evidence is reported in the detail but never gates, so it is not part of the key
    assert row["fields"] == "+".join(sorted(n for n in fields if n != "evidence"))
    assert diff_key(row) == f"{symbol}:{D0.isoformat()}:fields_differ:{row['fields']}"


def test_a_withheld_cik_does_not_hide_a_grouping_leak_on_the_same_segment() -> None:
    # Leakage. C joins B's security in the full master (a merge known early) while its CIK is
    # still withheld; the truncated build knows C's CIK but not the merge. The CIK is a delay,
    # the grouping is a leak, and both are reported.
    listed = et_midnight(date(2020, 7, 10))
    full = [
        {**r, "security_id": "S2"} if r["symbol"] == "C" else r
        for r in rows_with(good_rows(), "C", cik=2, cik_available_at=listed)
    ]
    truncated = [
        {**r, "security_id": "S9"} if r["symbol"] == "C" else r
        for r in rows_with(good_rows(), "C", cik=2)
    ]
    out = compare(master_table(full), master_table(truncated), listed - MICRO)
    assert diffs(out) == {
        ("delay", "cik_withheld", "C", CUT),
        ("leak", "grouping_differs", "C", CUT),
    }


def test_same_cik_known_later_in_the_full_master_is_a_delay() -> None:
    # the full master shows the CIK later than a build then could have: it withholds, it does not
    # reveal
    full = rows_with(good_rows(), "A", cik_available_at=et_midnight(D0) + timedelta(days=30))
    at = et_midnight(D0) + timedelta(days=60)
    [row] = compare(master_table(full), master_table(good_rows()), at).to_pylist()
    assert (row["kind"], row["reason"], row["fields"]) == (
        "delay",
        "cik_withheld",
        "cik_available_at",
    )


def test_evidence_alone_is_reported_but_not_a_leak() -> None:
    # evidence says how the build learned of a segment (a record, then the SEC confirming it:
    # name_change -> both); no reader decides anything on it
    full = rows_with(good_rows(), "A", evidence="both")
    [row] = compare(master_table(full), master_table(good_rows()), AS_OF).to_pylist()
    assert (row["kind"], row["reason"], row["symbol"]) == ("info", "evidence_differs", "A")
    assert "evidence" in row["detail"]


def test_cik_the_full_master_still_withholds_is_a_delay() -> None:
    # AFC before the SEC's 2020-07-10 correction: a build then reads the zombie CIK 3906; the
    # full master knows the corrected 1287750 only from the correction on and shows nothing
    corrected = et_midnight(date(2020, 7, 10))
    full = rows_with(good_rows(), "A", cik=1287750, cik_available_at=corrected)
    truncated = rows_with(good_rows(), "A", cik=3906)
    at = corrected - MICRO
    [row] = compare(master_table(full), master_table(truncated), at).to_pylist()
    assert (row["kind"], row["reason"], row["symbol"]) == ("delay", "cik_withheld", "A")
    out = compare(master_table(full), master_table(truncated), corrected)
    assert diffs(out) == {("leak", "fields_differ", "A", D0)}


@pytest.mark.parametrize("joined", ["full", "truncated"])
def test_same_rows_grouped_differently_is_a_leak(joined: str) -> None:
    # B -> C as one security on one side, two on the other: every field equal, only the
    # grouping differs (canonical id = the security's earliest visible (valid_from, symbol))
    one = good_rows()
    two = [{**r, "security_id": "S9"} if r["symbol"] == "C" else r for r in good_rows()]
    full, truncated = (one, two) if joined == "full" else (two, one)
    out = compare(master_table(full), master_table(truncated), AS_OF)
    assert diffs(out) == {("leak", "grouping_differs", "C", CUT)}


def test_row_only_in_truncated_on_days_nobody_owns_is_a_delay() -> None:
    truncated = [*good_rows(), seg_row("S5", "E", CUT)]
    out = compare(master_table(good_rows()), master_table(truncated), AS_OF)
    assert diffs(out) == {("delay", "only_in_truncated", "E", CUT)}


def test_truncated_row_starting_where_the_full_segment_ends_is_a_delay() -> None:
    # boundary: [D0, CUT) and [CUT, ...) do not share a day
    full = [seg_row("S1", "E", D0, CUT)]
    truncated = [seg_row("S1", "E", D0, CUT), seg_row("S2", "E", CUT)]
    out = compare(master_table(full), master_table(truncated), AS_OF)
    assert diffs(out) == {("delay", "only_in_truncated", "E", CUT)}


def test_truncated_row_on_days_the_full_master_gives_to_another_row_is_a_leak() -> None:
    # the full master shows E as one open segment from D0; the truncated build started E at
    # LATER: what a reader at as_of saw for E before LATER came from the future
    full = [seg_row("S1", "E", D0)]
    truncated = [seg_row("S1", "E", LATER)]
    out = compare(master_table(full), master_table(truncated), AS_OF)
    assert diffs(out) == {("leak", "only_in_full", "E", D0), ("leak", "replaced", "E", LATER)}


def test_truncated_row_one_day_inside_the_full_segment_is_replaced() -> None:
    # boundary partner of the delay case above: one shared day is enough
    before_cut = CUT - timedelta(days=1)
    full = [seg_row("S1", "E", D0, CUT)]
    truncated = [seg_row("S1", "E", D0, before_cut), seg_row("S2", "E", before_cut)]
    out = compare(master_table(full), master_table(truncated), AS_OF)
    assert diffs(out) == {
        ("leak", "fields_differ", "E", D0),
        ("leak", "replaced", "E", before_cut),
    }


def test_end_masked_at_as_of_matches_the_truncated_open_segment() -> None:
    # BK in 2022: the full file says BK ends 2026-05-21, the truncated build has BK open
    full = [seg_row("S1", "BK", D0, CUT)]
    truncated = [seg_row("S1", "BK", D0)]
    before = et_midnight(CUT) - MICRO
    assert compare(master_table(full), master_table(truncated), before).num_rows == 0
    # from the end's own instant on the full master shows it; a truncated build that still has
    # BK open disagrees on valid_to
    at = et_midnight(CUT)
    out = compare(master_table(full), master_table(truncated), at)
    [row] = out.to_pylist()
    assert (row["kind"], row["reason"], row["symbol"]) == ("leak", "fields_differ", "BK")
    assert "valid_to" in row["detail"]


def test_rows_not_yet_knowable_on_either_side_are_not_compared() -> None:
    # Leakage. The full master has a segment from the future (knowable at LATER); at CUT it is
    # on disk but not visible, so it is not a diff -- and a truncated row not yet knowable is
    # not a delay either
    future = seg_row("S5", "E", LATER)
    full = master_table([*good_rows(), future])
    truncated = master_table([*good_rows(), {**seg_row("S6", "F", LATER)}])
    at_cut = et_midnight(CUT)
    assert compare(full, truncated, at_cut).num_rows == 0
    assert diffs(compare(full, truncated, et_midnight(LATER))) == {
        ("leak", "only_in_full", "E", LATER),
        ("delay", "only_in_truncated", "F", LATER),
    }


def test_compare_refuses_a_naive_as_of() -> None:
    table = master_table(good_rows())
    with pytest.raises(ValueError):
        compare(table, table, datetime(2021, 1, 4))


# =============================================================================================
# default_instants
# =============================================================================================


def test_default_instants_are_the_named_days_and_each_evidences_spread() -> None:
    def at(day: date, hours: int = 0) -> datetime:
        return et_midnight(day) + timedelta(hours=hours)

    name_change = [at(date(2021, 3, 1)), at(date(2021, 5, 3)), at(date(2022, 1, 3))]
    sec = [at(date(2020, 1, 13)), at(date(2020, 7, 10)), at(date(2023, 1, 3)), at(date(2024, 1, 2))]
    bars = [at(date(2020, 2, 13))]  # equal to a named instant: listed once
    rows = [
        seg_row(f"S{i}", f"N{i}", date(2020, 1, 2), available_at=t, cik=None)
        for i, t in enumerate(name_change)
    ]
    rows = [{**r, "evidence": "name_change"} for r in rows]
    rows += [
        {**seg_row(f"S{10 + i}", f"X{i}", date(2020, 1, 2), available_at=t), "evidence": "sec"}
        for i, t in enumerate(sec)
    ]
    rows += [{**seg_row("S20", "Y", date(2020, 1, 2), available_at=bars[0]), "evidence": "bars"}]

    instants = default_instants(master_table(rows))

    # earliest, lower median, latest per evidence: name_change 3 values -> the middle one;
    # sec 4 values -> the second; bars 1 value -> it, three times
    picked = {at(d) for d in NAMED_DAYS} | {
        name_change[0],
        name_change[1],
        name_change[2],
        sec[0],
        sec[1],
        sec[3],
        bars[0],
    }
    expected = sorted(picked | {t - MICRO for t in picked})
    assert instants == expected
    assert sec[2] not in instants
    assert all(t.tzinfo is not None and t.utcoffset() == timedelta(0) for t in instants)


def test_default_instants_include_the_spread_of_cik_available_at() -> None:
    # a leak may start at a CIK's own instant; the default run samples them like an evidence
    # kind (earliest, lower median, latest) -- every one would be a build per SEC snapshot
    days = [date(2020, 7, 10), date(2020, 8, 3), date(2020, 9, 1), date(2021, 2, 1)]
    master = master_table(
        [
            seg_row(f"S{i}", f"C{i}", D0, cik=i, cik_available_at=et_midnight(d))
            for i, d in enumerate(days)
        ]
    )
    instants = default_instants(master)
    for d in (days[0], days[1], days[3]):
        assert et_midnight(d) in instants and et_midnight(d) - MICRO in instants
    assert et_midnight(days[2]) not in instants


def test_default_instants_of_an_empty_master_are_the_named_ones() -> None:
    named = {et_midnight(d) for d in NAMED_DAYS}
    assert default_instants(master_table([])) == sorted(named | {t - MICRO for t in named})


# =============================================================================================
# the property: no scenario world leaks at any of its own instants
# =============================================================================================


def instants_of(master: pa.Table, inputs: Inputs | None = None) -> list[datetime]:
    """Every instant at which the full master changes, plus the instants of the SEC snapshots
    and the rename records: a leak may start at one of those (a snapshot that flips a CIK)
    without the master changing then. Bar instants are left out (one per session; the rules
    that read bars change the master at the instants they stamp)."""
    found = {
        t
        for name in ("available_at", "end_available_at", "cik_available_at")
        for t in master.column(name).to_pylist()
        if t is not None
    }
    if inputs is not None:
        found |= {et_midnight(d) for d in inputs.sec_history.column("snapshot_date").to_pylist()}
        found |= {
            t if t is not None else et_midnight(d)
            for d, t in zip(
                inputs.name_changes.column("process_date").to_pylist(),
                inputs.name_changes.column("available_at").to_pylist(),
                strict=True,
            )
        }
    return sorted(found | {t - MICRO for t in found})


# Known leaks, one line per (scenario, instant, diff_key -- the differing fields included, so a
# known row that starts differing in one more field is a new line). The first run
# (2026-10-19) found leak classes the build rules still have; they are fixed class by class, and
# each fix removes its lines. The test fails on a leak not listed (a new one) and on a listed
# line that no longer occurs (a fix landed: delete the line). Regenerate after a deliberate
# change with ASOF_TRUNCATION_BASELINE=write; read the diff before committing it.
BASELINE = Path(__file__).with_name("truncation_baseline.txt")


def leak_lines(scenario: str) -> set[str]:
    inputs = SCENARIOS[scenario]()
    full = build(inputs)
    assert full.num_rows > 0
    return {
        f"{scenario}|{as_of.isoformat()}|{diff_key(r)}"
        for as_of in instants_of(full, inputs)
        for r in compare(full, build_truncated(inputs, as_of), as_of).to_pylist()
        if r["kind"] == "leak"
    }


def baseline_lines(scenario: str) -> set[str]:
    if not BASELINE.exists():
        return set()
    return {
        line
        for line in BASELINE.read_text().splitlines()
        if line and not line.startswith("#") and line.split("|", 1)[0] == scenario
    }


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_scenario_world_rebuilt_from_knowable_inputs_leaks_only_the_known_leaks(
    scenario: str,
) -> None:
    # For every instant at which the full master changes (and one microsecond before it), the
    # master built from the inputs knowable then is compared with the full master as seen then.
    # Delays are allowed; leaks must be exactly the baseline's. A new leak is a finding to
    # triage, never a line to add without reading it.
    found, known = leak_lines(scenario), baseline_lines(scenario)
    if os.environ.get("ASOF_TRUNCATION_BASELINE") == "write":
        others = (
            {line for line in BASELINE.read_text().splitlines()} if BASELINE.exists() else set()
        )
        others = {line for line in others if line and line.split("|", 1)[0] != scenario}
        header = "# scenario|instant|SYMBOL:valid_from:reason[:fields] -- see this module\n"
        body = sorted(line for line in others | found if not line.startswith("#"))
        BASELINE.write_text(header + "".join(f"{line}\n" for line in body))
        return
    new, fixed = sorted(found - known), sorted(known - found)
    assert not new, f"{scenario}: {len(new)} new leak rows\n" + "\n".join(new)
    assert not fixed, f"{scenario}: {len(fixed)} baseline rows no longer occur, delete them\n" + (
        "\n".join(fixed)
    )


def test_a_dated_override_is_a_leak_before_its_date() -> None:
    # The merge of CYCN and KRSA (a human decision dated 2026-10-07) rewrites history: a reader
    # on 2026-10-06 sees them as one security in the full master, but nothing knowable then
    # said so. Everything else in the world is knowable by 2026-10-06 (the last bar is
    # 2026-10-05), so the merge is the whole diff: the label that joins the earlier one's
    # security is grouped differently (a leak), the earlier one only changes its evidence to
    # "override" (info); from the decision's own midnight on there is no diff.
    inputs = SCENARIOS["world"]()
    overrides = [merge("2026-10-07", "CYCN", "KRSA")]
    full = build(inputs, overrides)
    decided = et_midnight(date(2026, 10, 7))
    starts = {
        r["symbol"]: r["valid_from"] for r in full.to_pylist() if r["symbol"] in ("CYCN", "KRSA")
    }

    first, joiner = sorted(starts, key=lambda s: (starts[s], s))
    before = compare(full, build_truncated(inputs, decided - MICRO, overrides), decided - MICRO)
    assert diffs(before) == {
        ("info", "evidence_differs", first, starts[first]),
        ("info", "evidence_differs", joiner, starts[joiner]),
        ("leak", "grouping_differs", joiner, starts[joiner]),
    }
    assert compare(full, build_truncated(inputs, decided, overrides), decided).num_rows == 0


# =============================================================================================
# build_from_store(as_of=...)
# =============================================================================================

STORE_RECORD = date(2024, 4, 1)  # TAKR -> IMP
STORE_DAYS = business_days(date(2024, 1, 2), date(2024, 3, 29))
STORE_SNAPSHOTS = [date(2024, 1, 5), date(2024, 3, 1), date(2024, 5, 1)]


def write_store(root: Path, as_of: datetime | None = None) -> Path:
    """The I-world shape of tests/ingest/test_security_master_copies.py, smaller: TAKR's own
    bars Jan-Mar 2024, the same bars filed under IMP (Alpaca's relabelling), IMP's own bars from
    the TAKR -> IMP record, and an imported IMP history for Jan-Mar. With ``as_of``, only the
    rows knowable then are written: the store a reader at ``as_of`` would have had."""

    def knowable(t: datetime) -> bool:
        return as_of is None or t <= as_of

    def bar_knowable(day: date) -> bool:
        return knowable(et_midnight(day) + timedelta(hours=20))  # 20:00 New York

    takr = {
        d: daily("TAKR", d, open=9.9 + i / 1000, high=10.2 + i / 1000, low=9.8,
                 close=10.0 + i / 1000, volume=5_000 + i)
        for i, d in enumerate(STORE_DAYS)
    }  # fmt: skip
    copies = {
        d: daily("IMP", d, open=b.open, high=b.high, low=b.low, close=b.close, volume=b.volume)
        for d, b in takr.items()
    }
    own = {
        d: daily("IMP", d, open=10.1, high=10.4, low=10.0, close=10.2 + i / 1000, volume=6_000 + i)
        for i, d in enumerate(business_days(STORE_RECORD, date(2024, 5, 31)))
    }
    store_daily(
        root,
        [b for group in (takr, copies, own) for d, b in group.items() if bar_knowable(d)],
    )
    ext = [
        ext_row("IMP", d, close=3.0 + i / 1000, volume=2_000 + i)
        for i, d in enumerate(STORE_DAYS)
        if bar_knowable(d)
    ]
    if ext:
        write_ext(root, ext)
    lives: list[tuple[str, int, date, date | None]] = [
        ("TAKR", 9501, STORE_SNAPSHOTS[0], date(2024, 3, 1)),
        ("IMP", 9502, STORE_SNAPSHOTS[0], date(2024, 3, 1)),
        ("IMP", 9501, date(2024, 5, 1), None),
    ]
    sec = sec_table([s for s in STORE_SNAPSHOTS if knowable(et_midnight(s))], lives)
    (root / "symbols").mkdir(parents=True, exist_ok=True)
    pq.write_table(sec.cast(HISTORY_SCHEMA), root / "symbols" / "sec_history.parquet")
    if knowable(et_midnight(STORE_RECORD)):
        (root / "corporate_actions").mkdir(parents=True, exist_ok=True)
        pq.write_table(
            name_changes_table([("TAKR", "IMP", STORE_RECORD, "000800100", "000800200")]).cast(
                NAME_CHANGE_SCHEMA
            ),
            root / "corporate_actions" / "name_changes.parquet",
        )
    return root


def test_build_from_store_without_as_of_is_unchanged(tmp_path: Path) -> None:
    root = write_store(market_root(tmp_path))
    master, conflicts = sm.build_from_store(root)
    again, again_conflicts = sm.build_from_store(root, as_of=None)
    assert master.num_rows > 0
    assert again.equals(master) and again_conflicts.equals(conflicts)


@pytest.mark.parametrize(
    "as_of",
    [
        et_midnight(date(2024, 3, 1)),  # a snapshot's own midnight
        et_midnight(STORE_RECORD) - MICRO,  # one microsecond before the record
        et_midnight(STORE_RECORD),  # the record, before IMP's first own bar is visible
        et_midnight(date(2024, 4, 15)) + timedelta(hours=20),
        datetime(2030, 1, 1, tzinfo=UTC),  # everything knowable
    ],
    ids=["snapshot", "record_minus_1us", "record", "mid_april", "everything"],
)
def test_build_from_store_at_as_of_is_the_build_of_the_store_as_it_was(
    tmp_path: Path, as_of: datetime
) -> None:
    # The store on disk holds rows from after as_of (later bars, imported rows, the 05-01
    # snapshot, the record); building at as_of must equal building a store that never had them.
    full_root = write_store(market_root(tmp_path / "full"))
    then_root = write_store(market_root(tmp_path / "then"), as_of)

    master, conflicts = sm.build_from_store(full_root, as_of=as_of)
    expected, expected_conflicts = sm.build_from_store(then_root)

    assert master.equals(expected)
    assert conflicts.equals(expected_conflicts)


def test_build_from_store_refuses_a_naive_as_of(tmp_path: Path) -> None:
    root = write_store(market_root(tmp_path))
    with pytest.raises(ValueError):
        sm.build_from_store(root, as_of=datetime(2024, 4, 1))


# =============================================================================================
# CLI: asof market truncation
# =============================================================================================

CLI_DAYS = business_days(date(2024, 1, 2), date(2024, 2, 13))
CLI_SNAPSHOTS = [date(2024, 1, 5), date(2024, 3, 1), date(2024, 5, 1)]
CLI_DECIDED = date(2024, 6, 3)  # the merge override's date
LEAK_EXIT = 2  # as asof market audit: 1 is "could not run", 2 is "ran and found work"


def write_cli_store(root: Path) -> Path:
    """CYCN and KRSA trade identical bars (the real pair: 1,609 identical days, no record);
    AAA trades its own. Each label is listed under its own CIK in every snapshot."""
    cycn = [
        daily("CYCN", d, open=6.9, high=7.1, low=6.8, close=7.0 + i / 1000, volume=5_000 + i)
        for i, d in enumerate(CLI_DAYS)
    ]
    krsa = [
        daily("KRSA", d, open=b.open, high=b.high, low=b.low, close=b.close, volume=b.volume)
        for d, b in zip(CLI_DAYS, cycn, strict=True)
    ]
    aaa = [
        daily("AAA", d, open=49.5, high=50.5, low=49.0, close=50.0 + i / 100, volume=9_000 + i)
        for i, d in enumerate(business_days(date(2024, 1, 2), date(2024, 6, 28)))
    ]
    store_daily(root, [*cycn, *krsa, *aaa])
    sec = history_table(
        [
            (s, t, c)
            for s in CLI_SNAPSHOTS
            for t, c in (("CYCN", 1004), ("KRSA", 1005), ("AAA", 1006))
        ]
    )
    (root / "symbols").mkdir(parents=True, exist_ok=True)
    pq.write_table(sec, root / "symbols" / "sec_history.parquet")
    return root


def write_yaml(path: Path, content: Any) -> Path:
    path.write_text(yaml.safe_dump(content))
    return path


def stamp(t: datetime) -> str:
    return t.astimezone(UTC).strftime("%Y%m%dT%H%M%S%f")


def run_truncation(
    tmp_path: Path,
    *,
    as_of: Sequence[str] = (),
    overrides: Sequence[Mapping[str, Any]] = (),
    exceptions: Any = (),
    baseline: Sequence[str] = (),
    prune: bool = False,
    instants: Sequence[str] | None = None,
    jobs: str | None = None,
) -> Any:
    overrides_path = write_yaml(tmp_path / "overrides.yaml", list(overrides))
    exceptions_path = write_yaml(tmp_path / "exceptions.yaml", list(exceptions))
    # always a file of the test's own: never the repo's configs/truncation_baseline.txt
    baseline_path = tmp_path / "baseline.txt"
    baseline_path.write_text("".join(f"{k}\n" for k in baseline))
    # likewise never the repo's configs/truncation_instants.txt; None = --computed-instants
    instants_path = tmp_path / "instants.txt"
    if instants is not None:
        instants_path.write_text("".join(f"{t}\n" for t in instants))
    args = ["market", "truncation"]
    for value in as_of:
        args += ["--as-of", value]
    args += [
        "--overrides",
        str(overrides_path),
        "--exceptions",
        str(exceptions_path),
        "--baseline",
        str(baseline_path),
        "--instants",
        str(instants_path),
        *(["--computed-instants"] if instants is None and not as_of else []),
        *(["--prune-baseline"] if prune else []),
        *(["--jobs", jobs] if jobs is not None else []),
        "--data-dir",
        str(tmp_path),
    ]
    return CliRunner().invoke(app, args)


MERGE = {"merge": ["CYCN", "KRSA"], "reason": "identical history", "date": CLI_DECIDED.isoformat()}
# a fixed instant list for default runs: before the merge decision (the leak) and at it
CLI_INSTANTS = [et_midnight(date(2024, 3, 1)).isoformat(), et_midnight(CLI_DECIDED).isoformat()]
# the merge's one leak: KRSA joins CYCN's security (CYCN itself only gets an evidence note)
LEAK_KEYS = [f"KRSA:{CLI_DAYS[0].isoformat()}:grouping_differs"]
OTHER_KEY = f"CYCN:{CLI_DAYS[0].isoformat()}:grouping_differs"  # matches no row


def report_dir(tmp_path: Path) -> Path:
    return market_root(tmp_path) / "audit" / "truncation"


def test_cli_writes_one_diff_file_per_instant_and_fails_on_a_leak(tmp_path: Path) -> None:
    write_cli_store(market_root(tmp_path))

    result = run_truncation(
        tmp_path, as_of=["2024-03-01", CLI_DECIDED.isoformat()], overrides=[MERGE]
    )

    assert result.exit_code == LEAK_EXIT, result.output
    assert "leak" in result.output
    before, decided = et_midnight(date(2024, 3, 1)), et_midnight(CLI_DECIDED)
    files = sorted(p.name for p in report_dir(tmp_path).glob("*.parquet"))
    assert files == [f"{stamp(before)}.parquet", f"{stamp(decided)}.parquet"]
    assert stamp(before) == "20240301T050000000000"  # 00:00 New York in winter, as UTC
    early = pq.read_table(report_dir(tmp_path) / f"{stamp(before)}.parquet")
    assert early.schema.equals(DIFF_SCHEMA)
    assert {
        f"{r['symbol']}:{r['valid_from'].isoformat()}:{r['reason']}"
        for r in early.to_pylist()
        if r["kind"] == "leak"
    } == set(LEAK_KEYS)
    late = pq.read_table(report_dir(tmp_path) / f"{stamp(decided)}.parquet")
    assert late.schema.equals(DIFF_SCHEMA) and late.num_rows == 0


def test_cli_excepted_leaks_do_not_fail(tmp_path: Path) -> None:
    write_cli_store(market_root(tmp_path))
    reviewed = [{"key": k, "reason": "dated human merge", "date": "2026-10-19"} for k in LEAK_KEYS]

    result = run_truncation(tmp_path, as_of=["2024-03-01"], overrides=[MERGE], exceptions=reviewed)

    assert result.exit_code == 0, result.output
    # the rows are still written: an exception silences the gate, not the report (the leak
    # and the two evidence notes)
    assert pq.read_table(report_dir(tmp_path) / "20240301T050000000000.parquet").num_rows == 3


def test_cli_one_leak_left_unexcepted_still_fails(tmp_path: Path) -> None:
    write_cli_store(market_root(tmp_path))
    # an exception covers its own key only
    reviewed = [{"key": OTHER_KEY, "reason": "dated human merge", "date": "2026-10-19"}]
    result = run_truncation(tmp_path, as_of=["2024-03-01"], overrides=[MERGE], exceptions=reviewed)
    assert result.exit_code == LEAK_EXIT, result.output
    # a partial run cannot tell an exception unused: OTHER_KEY may occur at another instant
    assert "stale exception" not in result.output


def test_cli_baseline_leaks_do_not_fail(tmp_path: Path) -> None:
    write_cli_store(market_root(tmp_path))
    result = run_truncation(tmp_path, as_of=["2024-03-01"], overrides=[MERGE], baseline=LEAK_KEYS)
    assert result.exit_code == 0, result.output
    assert "0 new" in result.output


def test_cli_default_run_fails_on_a_baseline_key_that_no_longer_occurs(tmp_path: Path) -> None:
    write_cli_store(market_root(tmp_path))
    # the ratchet: OTHER_KEY matches no row, so its line must go
    result = run_truncation(
        tmp_path, overrides=[MERGE], baseline=[*LEAK_KEYS, OTHER_KEY], instants=CLI_INSTANTS
    )
    assert result.exit_code == LEAK_EXIT, result.output
    assert f"(--prune-baseline): {OTHER_KEY}" in result.output


def test_cli_prune_baseline_refuses_while_a_new_leak_is_open(tmp_path: Path) -> None:
    write_cli_store(market_root(tmp_path))
    # KRSA's leak is new and OTHER_KEY is gone: deleting one must wait for reviewing the other
    result = run_truncation(
        tmp_path, overrides=[MERGE], baseline=[OTHER_KEY], prune=True, instants=CLI_INSTANTS
    )
    assert result.exit_code != 0
    assert f"new leak: {LEAK_KEYS[0]}" in result.output  # listed, not only counted
    assert load_baseline(tmp_path / "baseline.txt") == [OTHER_KEY]


def test_repo_baseline_and_exceptions_load_and_do_not_overlap() -> None:
    # read only: a hand-added line that is malformed, duplicated or also excepted fails CI,
    # not 26 minutes into a real run
    repo = Path(__file__).resolve().parents[2] / "configs"
    known = load_baseline(repo / "truncation_baseline.txt")
    reviewed = load_truncation_exceptions(repo / "truncation_exceptions.yaml")
    assert known
    assert mt.gate([], baseline=known, exceptions=reviewed, full_run=False).ok
    assert mt.load_instants(repo / "truncation_instants.txt")


def test_cli_prune_baseline_deletes_the_fixed_keys_and_passes(tmp_path: Path) -> None:
    write_cli_store(market_root(tmp_path))
    result = run_truncation(
        tmp_path,
        overrides=[MERGE],
        baseline=[*LEAK_KEYS, OTHER_KEY],
        prune=True,
        instants=CLI_INSTANTS,
    )
    assert result.exit_code == 0, result.output
    assert load_baseline(tmp_path / "baseline.txt") == LEAK_KEYS


def test_cli_prune_baseline_refuses_the_computed_instants(tmp_path: Path) -> None:
    write_cli_store(market_root(tmp_path))
    # the computed spread drifts with the master: a key gone from it may still occur
    result = run_truncation(tmp_path, overrides=[MERGE], baseline=[OTHER_KEY], prune=True)
    assert result.exit_code != 0
    assert "--prune-baseline" in result.output
    assert load_baseline(tmp_path / "baseline.txt") == [OTHER_KEY]


def test_cli_without_an_instants_file_refuses_to_guess(tmp_path: Path) -> None:
    write_cli_store(market_root(tmp_path))
    result = run_truncation(tmp_path, overrides=[MERGE], instants=[])  # header-less, empty
    assert result.exit_code != 0
    assert "no instants" in result.output
    assert not report_dir(tmp_path).exists() or not list(report_dir(tmp_path).glob("*"))


def test_cli_prune_baseline_refuses_a_partial_run(tmp_path: Path) -> None:
    write_cli_store(market_root(tmp_path))
    result = run_truncation(
        tmp_path, as_of=["2024-03-01"], overrides=[MERGE], baseline=[OTHER_KEY], prune=True
    )
    assert result.exit_code != 0
    assert "--prune-baseline" in result.output
    assert load_baseline(tmp_path / "baseline.txt") == [OTHER_KEY]


def test_cli_default_run_checks_exactly_the_listed_instants(tmp_path: Path) -> None:
    root = write_cli_store(market_root(tmp_path))
    full, _ = sm.build_from_store(root, [MERGE])
    listed = [et_midnight(date(2024, 3, 1)), et_midnight(CLI_DECIDED)]
    result = run_truncation(
        tmp_path, overrides=[MERGE], baseline=LEAK_KEYS, instants=[t.isoformat() for t in listed]
    )
    assert {p.name for p in report_dir(tmp_path).glob("*.parquet")} == {
        f"{stamp(t)}.parquet" for t in listed
    }
    # a full run: the listed instants stand for every instant, so fixed keys are checked
    assert result.exit_code == 0, result.output
    assert "0 no longer occurring" in result.output
    # the computed spread is reported, not checked
    assert len(set(default_instants(full)) - set(listed)) > 0
    assert "computed instants not in" in result.output


@pytest.mark.parametrize(
    "line",
    [
        "2024-03-01T05:00:00",  # no offset
        "2024-03-01",  # a date has no offset either
        "yesterday",
    ],
)
def test_load_instants_refuses_a_bad_line(tmp_path: Path, line: str) -> None:
    path = tmp_path / "instants.txt"
    path.write_text(f"# header\n2024-03-01T05:00:00+00:00\n{line}\n")
    with pytest.raises(ValueError, match="instants.txt"):
        mt.load_instants(path)


def test_load_instants_reads_utc_sorted_and_refuses_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "instants.txt"
    path.write_text("# h\n\n2024-06-03T00:00:00-04:00\n2024-03-01T05:00:00.000000+00:00\n")
    assert mt.load_instants(path) == [
        datetime(2024, 3, 1, 5, tzinfo=UTC),
        datetime(2024, 6, 3, 4, tzinfo=UTC),
    ]
    assert mt.load_instants(tmp_path / "absent.txt") == []
    path.write_text("2024-03-01T05:00:00+00:00\n2024-03-01T00:00:00-05:00\n")
    with pytest.raises(ValueError, match="duplicated"):
        mt.load_instants(path)


@pytest.mark.parametrize("missing", ["reason", "date"])
def test_cli_rejects_an_exception_without_reason_or_date(tmp_path: Path, missing: str) -> None:
    write_cli_store(market_root(tmp_path))
    entry = {"key": LEAK_KEYS[0], "reason": "dated human merge", "date": "2026-10-19"}
    del entry[missing]

    result = run_truncation(tmp_path, as_of=["2024-03-01"], overrides=[MERGE], exceptions=[entry])

    assert result.exit_code != 0
    assert missing in result.output


def test_cli_clean_store_exits_zero(tmp_path: Path) -> None:
    write_cli_store(market_root(tmp_path))
    result = run_truncation(tmp_path, as_of=["2024-03-01", "2024-06-03"])
    assert result.exit_code == 0, result.output
    for name in ("20240301T050000000000", "20240603T040000000000"):
        table = pq.read_table(report_dir(tmp_path) / f"{name}.parquet")
        assert table.schema.equals(DIFF_SCHEMA)
        assert not [r for r in table.to_pylist() if r["kind"] == "leak"]


def test_cli_without_as_of_runs_the_default_instants(tmp_path: Path) -> None:
    root = write_cli_store(market_root(tmp_path))
    full, _ = sm.build_from_store(root, [MERGE])

    result = run_truncation(tmp_path, overrides=[MERGE])

    expected = {f"{stamp(t)}.parquet" for t in default_instants(full)}
    assert {p.name for p in report_dir(tmp_path).glob("*.parquet")} == expected
    # the override evidence's earliest instant (CYCN's first bar) is before the decision
    assert result.exit_code == LEAK_EXIT, result.output


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_a_relabelled_piece_never_shows_the_takers_cik_before_the_taker_does(
    scenario: str,
) -> None:
    # Leakage. Rule R hands the taker's CIK to the piece that was its relabelled history (INBX
    # 2024-05-29 under Inhibrx Biosciences' 2007919). That CIK is knowable no earlier than on
    # the taker's own segments.
    master = build(SCENARIOS[scenario]())
    rows = master.to_pylist()
    for r in rows:
        if r["evidence"] != "relabelled" or r["cik"] is None:
            continue
        # the taker's own piece: the same label from the relabelled piece's end
        [taker] = [
            q for q in rows if q["symbol"] == r["symbol"] and q["valid_from"] == r["valid_to"]
        ]
        assert taker["cik"] == r["cik"], (r["symbol"], r["valid_from"])
        assert r["cik_available_at"] == taker["cik_available_at"], (r["symbol"], r["valid_from"])


# =============================================================================================
# the ratchet gate: configs/truncation_baseline.txt (known leaks, count may only shrink)
# =============================================================================================
# API:
#     mt.GateResult(new, fixed, stale_exceptions), .ok = not new and not fixed
#     mt.gate(found, *, baseline, exceptions, full_run) -> GateResult
#     mt.load_baseline(path) -> list[str]
#     mt.prune_baseline(path, fixed) -> int
# The baseline holds known-but-unfixed leaks; the exceptions file holds reviewed-and-accepted
# ones. A key may live in one of them, never both. ``fixed`` and ``stale_exceptions`` are only
# computed on a full run: a partial ``--as-of`` run does not visit every instant, so a key it
# did not find may still occur elsewhere.

BASE_A = "AAAU:2020-01-02:fields_differ:cik_available_at"
BASE_B = "CCL:2020-01-02:fields_differ:end_available_at+valid_to"
EXC_C = "GOOG:2019-07-01:only_in_full"
REVIEWED = {EXC_C: "reviewed share-class split"}
HEADER = "# Known leaks, one key per line; the count may only shrink.\n#\n"


def write_baseline(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def test_gate_all_found_keys_known_on_a_full_run_is_ok() -> None:
    result = mt.gate(
        [BASE_A, BASE_B, EXC_C], baseline=[BASE_A, BASE_B], exceptions=REVIEWED, full_run=True
    )
    assert result.ok
    assert (result.new, result.fixed, result.stale_exceptions) == ([], [], [])


def test_gate_baseline_key_not_found_on_a_full_run_is_fixed_and_fails() -> None:
    # the ratchet: a fix must also delete its baseline line, so the count shrinks for good
    result = mt.gate([BASE_A, EXC_C], baseline=[BASE_A, BASE_B], exceptions=REVIEWED, full_run=True)
    assert result.fixed == [BASE_B]
    assert result.new == []
    assert not result.ok


def test_gate_exception_not_found_on_a_full_run_is_stale_but_ok() -> None:
    result = mt.gate([BASE_A], baseline=[BASE_A], exceptions=REVIEWED, full_run=True)
    assert result.stale_exceptions == [EXC_C]
    assert result.ok


def test_gate_partial_run_reports_no_fixed_and_no_stale() -> None:
    result = mt.gate([], baseline=[BASE_A, BASE_B], exceptions=REVIEWED, full_run=False)
    assert (result.fixed, result.stale_exceptions) == ([], [])
    assert result.ok


def test_gate_partial_run_still_reports_new_leaks() -> None:
    new = "ZZZ:2021-03-04:replaced"
    result = mt.gate([new], baseline=[BASE_A], exceptions=REVIEWED, full_run=False)
    assert result.new == [new]
    assert not result.ok


def test_gate_nothing_found_nothing_known_is_ok() -> None:
    result = mt.gate([], baseline=[], exceptions={}, full_run=True)
    assert result.ok
    assert (result.new, result.fixed, result.stale_exceptions) == ([], [], [])


def test_gate_a_key_found_at_several_instants_counts_once() -> None:
    new = "ZZZ:2021-03-04:replaced"
    result = mt.gate([new, BASE_A, new, BASE_A], baseline=[BASE_A], exceptions={}, full_run=True)
    assert result.new == [new]


def test_gate_outputs_are_sorted_whatever_the_input_order() -> None:
    found = ["ZZZ:2021-03-04:replaced", "BBB:2020-05-01:only_in_full", "MMM:2020-05-01:replaced"]
    baseline = ["YYY:2020-01-02:replaced", "AAA:2020-01-02:replaced"]
    exceptions = {"XXX:2020-01-02:replaced": "r", "CCC:2020-01-02:replaced": "r"}
    result = mt.gate(found, baseline=baseline, exceptions=exceptions, full_run=True)
    assert result.new == sorted(found)
    assert result.fixed == sorted(baseline)
    assert result.stale_exceptions == sorted(exceptions)


def test_gate_refuses_a_key_both_in_baseline_and_exceptions() -> None:
    with pytest.raises(ValueError, match="GOOG:2019-07-01:only_in_full"):
        mt.gate([EXC_C], baseline=[BASE_A, EXC_C], exceptions=REVIEWED, full_run=True)


def test_gate_result_ok_ignores_stale_exceptions() -> None:
    assert mt.GateResult(new=[], fixed=[], stale_exceptions=[EXC_C]).ok
    assert not mt.GateResult(new=[BASE_A], fixed=[], stale_exceptions=[]).ok
    assert not mt.GateResult(new=[], fixed=[BASE_A], stale_exceptions=[]).ok


# --- leakage: a new leak must never slip through the gate ---


def test_gate_baseline_key_does_not_cover_the_same_row_with_one_more_field() -> None:
    known = "X:2020-01-02:fields_differ:valid_to"
    found = "X:2020-01-02:fields_differ:cik+valid_to"
    result = mt.gate([found], baseline=[known], exceptions={}, full_run=True)
    assert result.new == [found]
    assert result.fixed == [known]
    assert not result.ok


def test_gate_same_symbol_and_date_with_another_reason_is_new() -> None:
    found = "AAAU:2020-01-02:grouping_differs"
    result = mt.gate([found], baseline=[BASE_A], exceptions={}, full_run=False)
    assert result.new == [found]


def test_gate_same_symbol_with_another_valid_from_is_new() -> None:
    found = "AAAU:2020-01-03:fields_differ:cik_available_at"
    result = mt.gate([found], baseline=[BASE_A], exceptions={}, full_run=False)
    assert result.new == [found]


def test_gate_a_key_covered_by_exceptions_is_not_new() -> None:
    result = mt.gate([EXC_C], baseline=[], exceptions=REVIEWED, full_run=False)
    assert result.new == []
    assert result.ok


def test_baseline_written_from_diff_key_rows_loads_and_gates_clean(tmp_path: Path) -> None:
    rows = pa.Table.from_pylist(
        [
            {
                "kind": "leak",
                "reason": "fields_differ",
                "symbol": "CCL",
                "valid_from": date(2020, 1, 2),
                "fields": "end_available_at+valid_to",
                "detail": "",
            },
            {
                "kind": "leak",
                "reason": "only_in_full",
                "symbol": "GOOG",
                "valid_from": date(2019, 7, 1),
                "fields": None,
                "detail": "",
            },
            {
                "kind": "leak",
                "reason": "grouping_differs",
                "symbol": "KRSA",
                "valid_from": date(2024, 1, 2),
                "fields": "",
                "detail": "",
            },
        ],
        schema=DIFF_SCHEMA,
    ).to_pylist()
    keys = [diff_key(r) for r in rows]
    path = write_baseline(tmp_path / "baseline.txt", HEADER + "".join(f"{k}\n" for k in keys))

    loaded = mt.load_baseline(path)

    assert loaded == keys
    assert mt.gate(keys, baseline=loaded, exceptions={}, full_run=True).ok


# --- load_baseline ---


def test_load_baseline_reads_keys_in_file_order_skipping_comments(tmp_path: Path) -> None:
    path = write_baseline(tmp_path / "baseline.txt", f"{HEADER}{BASE_B}\n{BASE_A}\n")
    assert mt.load_baseline(path) == [BASE_B, BASE_A]


def test_load_baseline_missing_file_is_empty(tmp_path: Path) -> None:
    assert mt.load_baseline(tmp_path / "absent.txt") == []


def test_load_baseline_ignores_blank_lines_and_strips_whitespace(tmp_path: Path) -> None:
    path = write_baseline(tmp_path / "baseline.txt", f"\n  {BASE_A}  \n\n\t{BASE_B}\n   \n")
    assert mt.load_baseline(path) == [BASE_A, BASE_B]


def test_load_baseline_refuses_a_duplicated_key(tmp_path: Path) -> None:
    path = write_baseline(tmp_path / "baseline.txt", f"{BASE_A}\n{BASE_B}\n{BASE_A}\n")
    with pytest.raises(ValueError, match="baseline.txt") as raised:
        mt.load_baseline(path)
    assert BASE_A in str(raised.value)


@pytest.mark.parametrize(
    "line",
    [
        "X:2020-13-02:only_in_full",  # date does not parse
        "X:20200102:only_in_full",  # not YYYY-MM-DD
        "X:2020-01-02:went_missing",  # unknown reason
        "X:2020-01-02:only_in_truncated",  # a delay, never a baseline leak
        "X:2020-01-02:cik_withheld",  # a delay
        "X:2020-01-02:evidence_differs",  # info
        "X:2020-01-02:fields_differ:",  # trailing empty fields
        "X:2020-01-02",  # no reason
        ":2020-01-02:only_in_full",  # no symbol
    ],
)
def test_load_baseline_refuses_a_malformed_line(tmp_path: Path, line: str) -> None:
    path = write_baseline(tmp_path / "baseline.txt", f"{HEADER}{BASE_A}\n{line}\n")
    with pytest.raises(ValueError, match="baseline.txt") as raised:
        mt.load_baseline(path)
    assert line in str(raised.value)


# --- prune_baseline ---


def test_prune_baseline_removes_keys_keeps_header_and_sorts(tmp_path: Path) -> None:
    other = "ZZZ:2021-03-04:replaced"
    path = write_baseline(tmp_path / "baseline.txt", f"{HEADER}{other}\n{BASE_B}\n{BASE_A}\n")

    removed = mt.prune_baseline(path, [BASE_B])

    assert removed == 1
    assert path.read_text() == f"{HEADER}{BASE_A}\n{other}\n"


def test_prune_then_load_round_trips(tmp_path: Path) -> None:
    path = write_baseline(tmp_path / "baseline.txt", f"{HEADER}{BASE_B}\n{BASE_A}\n{EXC_C}\n")
    fixed = mt.gate([BASE_A], baseline=mt.load_baseline(path), exceptions={}, full_run=True).fixed

    assert mt.prune_baseline(path, fixed) == 2
    assert mt.load_baseline(path) == [BASE_A]
    assert mt.gate([BASE_A], baseline=mt.load_baseline(path), exceptions={}, full_run=True).ok


def test_prune_baseline_of_every_key_leaves_the_header(tmp_path: Path) -> None:
    path = write_baseline(tmp_path / "baseline.txt", f"{HEADER}{BASE_A}\n")
    assert mt.prune_baseline(path, [BASE_A]) == 1
    assert mt.load_baseline(path) == []
    assert path.read_text().startswith(HEADER)


def test_prune_baseline_refuses_an_unknown_key_and_writes_nothing(tmp_path: Path) -> None:
    text = f"{HEADER}{BASE_B}\n{BASE_A}\n"
    path = write_baseline(tmp_path / "baseline.txt", text)
    with pytest.raises(ValueError, match="ZZZ:2021-03-04:replaced"):
        mt.prune_baseline(path, [BASE_A, "ZZZ:2021-03-04:replaced"])
    assert path.read_text() == text


# =============================================================================================
# parallel instants: load_store_inputs / build_from_inputs / run_instants / --jobs
# =============================================================================================
# The real run loaded the store once per instant (1.1 s) and built serially (33 s x 66 instants
# = 40 min). The store is now read once, and the instants run in a fork-context process pool
# whose workers inherit the inputs. Whatever the number of workers, the diffs must be exactly
# those of the serial run: parallelism may change how long the run takes, never what it finds.

DiffList = list[tuple[datetime, pa.Table]]


@contextmanager
def within(seconds: float) -> Iterator[None]:
    """Fail instead of hanging: a pool that waits forever on a dead worker raises here."""

    def expire(signum: int, frame: Any) -> None:
        raise TimeoutError(f"no answer within {seconds} s")

    previous = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def no_children_left(seconds: float = 10.0) -> bool:
    deadline = time.monotonic() + seconds
    while multiprocessing.active_children():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


def store_inputs_of(world: Inputs) -> Any:
    """A scenario world as the store loader's output. build_from_inputs applies
    SEC_HISTORY_START as build_from_store does; the worlds below keep their leaks with it
    (measured 2026-10-20: r_gold_b 12, l_arnc 8, r4 8 leak rows over instants_of)."""
    return sm.StoreInputs(
        bars=world.bars, sec_history=world.sec_history, name_changes=world.name_changes
    )


def same_diffs(left: DiffList, right: DiffList) -> None:
    assert [t for t, _ in left] == [t for t, _ in right]
    for (t, a), (_, b) in zip(left, right, strict=True):
        assert a.schema.equals(DIFF_SCHEMA), t
        assert a.equals(b), t


def hand_loop(
    full: pa.Table, inputs: Any, overrides: Any, instants: Sequence[datetime]
) -> DiffList:
    return [
        (t, compare(full, sm.build_from_inputs(inputs, overrides, as_of=t)[0], t))
        for t in sorted({t.astimezone(UTC) for t in instants})
    ]


def cli_world(tmp_path: Path) -> tuple[Path, Any, pa.Table]:
    root = write_cli_store(market_root(tmp_path))
    inputs = sm.load_store_inputs(root)
    full, _ = sm.build_from_inputs(inputs, [MERGE])
    return root, inputs, full


def cli_instants(full: pa.Table) -> list[datetime]:
    named = [datetime.fromisoformat(t) for t in CLI_INSTANTS]
    return sorted({*(t.astimezone(UTC) for t in named), *default_instants(full)[:4]})


# --- load_store_inputs / build_from_inputs ---


def test_store_inputs_is_a_frozen_dataclass_of_the_three_tables(tmp_path: Path) -> None:
    inputs = sm.load_store_inputs(write_cli_store(market_root(tmp_path)))
    assert [f.name for f in dataclasses.fields(inputs)] == [
        "bars",
        "sec_history",
        "name_changes",
    ]
    assert all(isinstance(getattr(inputs, n), pa.Table) for n in ("bars", "sec_history"))
    with pytest.raises(dataclasses.FrozenInstanceError):
        inputs.bars = inputs.sec_history  # type: ignore[misc]


@pytest.mark.parametrize(
    "as_of",
    [
        None,
        et_midnight(date(2024, 1, 5)) - MICRO,  # before the first snapshot
        et_midnight(date(2024, 3, 1)),  # a snapshot's own midnight, the merge not yet decided
        et_midnight(CLI_DECIDED) - MICRO,
        et_midnight(CLI_DECIDED),
        datetime(2030, 1, 1, tzinfo=UTC),
    ],
    ids=["none", "pre_snapshot", "snapshot", "decided_minus_1us", "decided", "everything"],
)
@pytest.mark.parametrize("with_merge", [False, True], ids=["plain", "merge"])
def test_build_from_inputs_of_the_loaded_store_equals_build_from_store(
    tmp_path: Path, as_of: datetime | None, with_merge: bool
) -> None:
    root = write_cli_store(market_root(tmp_path))
    overrides = [MERGE] if with_merge else []

    master, conflicts = sm.build_from_inputs(sm.load_store_inputs(root), overrides, as_of=as_of)
    expected, expected_conflicts = sm.build_from_store(root, overrides, as_of=as_of)

    assert master.equals(expected)
    assert conflicts.equals(expected_conflicts)


def test_build_from_inputs_matches_the_store_with_records_and_imported_bars(
    tmp_path: Path,
) -> None:
    # the I-world store: an imported history and a name-change file, which the CLI store lacks
    root = write_store(market_root(tmp_path))
    inputs = sm.load_store_inputs(root)
    assert inputs.name_changes.num_rows == 1
    for as_of in (None, et_midnight(STORE_RECORD) - MICRO, et_midnight(STORE_RECORD)):
        master, conflicts = sm.build_from_inputs(inputs, as_of=as_of)
        expected, expected_conflicts = sm.build_from_store(root, as_of=as_of)
        assert master.equals(expected), as_of
        assert conflicts.equals(expected_conflicts), as_of


def test_load_store_inputs_without_a_name_change_file_is_an_empty_table(tmp_path: Path) -> None:
    inputs = sm.load_store_inputs(write_cli_store(market_root(tmp_path)))
    assert inputs.name_changes.num_rows == 0
    assert inputs.name_changes.schema.equals(NAME_CHANGE_SCHEMA)
    assert inputs.bars.num_rows > 0


def test_build_from_inputs_refuses_a_naive_as_of(tmp_path: Path) -> None:
    inputs = sm.load_store_inputs(write_cli_store(market_root(tmp_path)))
    with pytest.raises(ValueError):
        sm.build_from_inputs(inputs, as_of=datetime(2024, 3, 1))


def test_build_from_inputs_at_an_instant_does_not_change_the_inputs(tmp_path: Path) -> None:
    # the workers share one inputs object: a build at one instant must not truncate it for
    # the next
    inputs = sm.load_store_inputs(write_cli_store(market_root(tmp_path)))
    rows = (inputs.bars.num_rows, inputs.sec_history.num_rows, inputs.name_changes.num_rows)
    sm.build_from_inputs(inputs, [MERGE], as_of=et_midnight(date(2024, 1, 5)))
    assert (inputs.bars.num_rows, inputs.sec_history.num_rows, inputs.name_changes.num_rows) == rows


# --- run_instants: normal ---


def test_run_instants_serial_is_the_hand_loop(tmp_path: Path) -> None:
    _, inputs, full = cli_world(tmp_path)
    instants = cli_instants(full)
    assert len(instants) >= 4

    got = mt.run_instants(full, inputs, [MERGE], instants, jobs=1)

    same_diffs(got, hand_loop(full, inputs, [MERGE], instants))


def test_run_instants_default_jobs_is_serial(tmp_path: Path) -> None:
    _, inputs, full = cli_world(tmp_path)
    instants = cli_instants(full)
    same_diffs(
        mt.run_instants(full, inputs, [MERGE], instants),
        mt.run_instants(full, inputs, [MERGE], instants, jobs=1),
    )


@pytest.mark.parametrize("jobs", [2, 3])
def test_run_instants_in_a_pool_equals_serial_on_the_cli_store(tmp_path: Path, jobs: int) -> None:
    _, inputs, full = cli_world(tmp_path)
    instants = cli_instants(full)
    serial = mt.run_instants(full, inputs, [MERGE], instants, jobs=1)
    with within(120):
        pooled = mt.run_instants(full, inputs, [MERGE], instants, jobs=jobs)
    same_diffs(pooled, serial)
    assert any(table.num_rows for _, table in serial)  # the comparison is not of empty tables


@pytest.mark.parametrize("scenario", ["r_gold_b", "l_arnc", "r4"])
def test_run_instants_in_a_pool_equals_serial_on_leaking_worlds(scenario: str) -> None:
    inputs = store_inputs_of(SCENARIOS[scenario]())
    full, _ = sm.build_from_inputs(inputs)
    instants = instants_of(full, SCENARIOS[scenario]())
    serial = mt.run_instants(full, inputs, (), instants, jobs=1)
    with within(120):
        pooled = mt.run_instants(full, inputs, (), instants, jobs=2)
    same_diffs(pooled, serial)
    leaks = [r for _, table in serial for r in table.to_pylist() if r["kind"] == "leak"]
    assert leaks, f"{scenario}: no leak rows; pick a world that leaks"


# --- run_instants: edges ---


def test_run_instants_sorts_and_deduplicates_the_instants(tmp_path: Path) -> None:
    _, inputs, full = cli_world(tmp_path)
    decided = et_midnight(CLI_DECIDED)
    early = et_midnight(date(2024, 3, 1))
    # the same instant twice, once in New York time and once in UTC, and out of order
    given = [decided, early, decided.astimezone(UTC), early - MICRO]
    expected = sorted({t.astimezone(UTC) for t in given})
    assert len(expected) == 3

    for jobs in (1, 2):
        with within(120):
            got = mt.run_instants(full, inputs, [MERGE], given, jobs=jobs)
        assert [t for t, _ in got] == expected, jobs
        assert all(t.utcoffset() == timedelta(0) for t, _ in got), jobs
        same_diffs(got, hand_loop(full, inputs, [MERGE], given))


@pytest.mark.parametrize("jobs", [1, 2])
def test_run_instants_of_no_instants_is_empty(tmp_path: Path, jobs: int) -> None:
    _, inputs, full = cli_world(tmp_path)
    with within(60):
        assert mt.run_instants(full, inputs, [MERGE], [], jobs=jobs) == []


def test_run_instants_with_more_jobs_than_instants(tmp_path: Path) -> None:
    _, inputs, full = cli_world(tmp_path)
    instants = [et_midnight(date(2024, 3, 1)), et_midnight(CLI_DECIDED)]
    serial = mt.run_instants(full, inputs, [MERGE], instants, jobs=1)
    with within(120):
        same_diffs(mt.run_instants(full, inputs, [MERGE], instants, jobs=8), serial)
        same_diffs(mt.run_instants(full, inputs, [MERGE], instants[:1], jobs=8), serial[:1])
    assert no_children_left()


@pytest.mark.parametrize("jobs", [0, -1])
def test_run_instants_refuses_fewer_than_one_job(tmp_path: Path, jobs: int) -> None:
    _, inputs, full = cli_world(tmp_path)
    with pytest.raises(ValueError):
        mt.run_instants(full, inputs, [MERGE], [et_midnight(CLI_DECIDED)], jobs=jobs)
    with pytest.raises(ValueError):  # refused even with nothing to run
        mt.run_instants(full, inputs, [MERGE], [], jobs=jobs)


@pytest.mark.parametrize("jobs", [1, 2])
def test_run_instants_refuses_a_naive_instant(tmp_path: Path, jobs: int) -> None:
    _, inputs, full = cli_world(tmp_path)
    with within(60), pytest.raises(ValueError):
        mt.run_instants(full, inputs, [MERGE], [datetime(2024, 3, 1)], jobs=jobs)
    assert no_children_left()


def failing_at_two_snapshots(real: Callable[..., Any]) -> Callable[..., Any]:
    """build_security_master raising when exactly two of the three CLI snapshots are given:
    the instants from 2024-03-01 to before 2024-05-01. Installed in the parent, so a forked
    worker inherits it."""

    def build(**kwargs: Any) -> Any:
        snapshots = set(kwargs["sec_history"].column("snapshot_date").to_pylist())
        if len(snapshots) == 2:
            raise ValueError("planted failure at the 2024-03-01 instant")
        return real(**kwargs)

    return build


@pytest.mark.parametrize("jobs", [1, 2, 3])
def test_run_instants_an_error_at_one_instant_reaches_the_caller(
    tmp_path: Path, jobs: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, inputs, full = cli_world(tmp_path)  # the full build, before the failure is planted
    monkeypatch.setattr(
        sm, "build_security_master", failing_at_two_snapshots(build_security_master)
    )
    instants = [
        et_midnight(date(2024, 1, 5)),
        et_midnight(date(2024, 3, 1)),  # the failing one
        et_midnight(CLI_DECIDED),
        datetime(2030, 1, 1, tzinfo=UTC),
    ]
    with within(120), pytest.raises(ValueError, match="planted failure"):
        mt.run_instants(full, inputs, [MERGE], instants, jobs=jobs)
    assert no_children_left()


def test_run_instants_an_error_does_not_wait_on_a_stuck_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # one instant fails, a later one never returns: the error must arrive and the stuck worker
    # must be gone, not waited on (a hung build would otherwise hang the whole run)
    _, inputs, full = cli_world(tmp_path)
    real = build_security_master

    def build(**kwargs: Any) -> Any:
        snapshots = set(kwargs["sec_history"].column("snapshot_date").to_pylist())
        if len(snapshots) == 2:
            raise ValueError("planted failure at the 2024-03-01 instant")
        if len(snapshots) == 3:
            time.sleep(1e6)
        return real(**kwargs)

    monkeypatch.setattr(sm, "build_security_master", build)
    instants = [et_midnight(date(2024, 3, 1)), et_midnight(CLI_DECIDED)]
    with within(60), pytest.raises(ValueError, match="planted failure"):
        mt.run_instants(full, inputs, [MERGE], instants, jobs=2)
    assert no_children_left()


# --- run_instants: leakage ---


@pytest.mark.parametrize("jobs", [1, 2])
def test_run_instants_each_worker_truncates_to_its_own_instant(tmp_path: Path, jobs: int) -> None:
    # Leakage. The merge decided on 2024-06-03 is in the overrides every worker inherits; a
    # worker must still build the master of 2024-06-02 without it (the KRSA leak shows), and
    # only the 06-03 build may use it (no diff).
    _, inputs, full = cli_world(tmp_path)
    decided = et_midnight(CLI_DECIDED)
    with within(120):
        (before, early), (at, late) = mt.run_instants(
            full, inputs, [MERGE], [decided, decided - MICRO], jobs=jobs
        )
    assert (before, at) == (decided - MICRO, decided)
    assert {
        f"{r['symbol']}:{r['valid_from'].isoformat()}:{r['reason']}"
        for r in early.to_pylist()
        if r["kind"] == "leak"
    } == set(LEAK_KEYS)
    assert late.num_rows == 0


# --- CLI --jobs ---


def summary_lines(output: str) -> list[str]:
    """The run's output without the per-instant timings and the total duration."""
    return [
        line
        for line in output.splitlines()
        if not line.startswith("  20") and not line.startswith("rows -> ")
    ]


def reports(tmp_path: Path) -> dict[str, pa.Table]:
    return {p.name: pq.read_table(p) for p in sorted(report_dir(tmp_path).glob("*.parquet"))}


@pytest.mark.parametrize(
    "baseline", [[], LEAK_KEYS, [*LEAK_KEYS, OTHER_KEY]], ids=["new", "known", "fixed"]
)
def test_cli_reports_and_verdict_do_not_depend_on_jobs(
    tmp_path: Path, baseline: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "_cores", lambda: 4)  # --jobs 2 must not depend on this machine
    runs = {}
    for jobs in ("1", "2", None):  # None: the default, min(4, cores)
        here = tmp_path / f"jobs_{jobs}"
        here.mkdir()
        write_cli_store(market_root(here))
        result = run_truncation(
            here, overrides=[MERGE], baseline=baseline, instants=CLI_INSTANTS, jobs=jobs
        )
        # each run has its own directory, printed in the output: not a difference of behaviour
        output = result.output.replace(str(here), "<run>")
        runs[jobs] = (result, output, reports(here), (here / "baseline.txt").read_text())

    serial, serial_output, serial_reports, serial_baseline = runs["1"]
    assert len(serial_reports) == len(CLI_INSTANTS)
    for jobs in ("2", None):
        result, output, written, kept = runs[jobs]
        assert result.exit_code == serial.exit_code, result.output
        assert summary_lines(output) == summary_lines(serial_output)
        assert output.splitlines()[-1] == serial_output.splitlines()[-1]
        assert written.keys() == serial_reports.keys()
        for name, table in written.items():
            assert table.equals(serial_reports[name]), (jobs, name)
        assert kept == serial_baseline


def test_cli_prune_with_jobs_prunes_as_serial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "_cores", lambda: 4)  # --jobs 2 must not depend on this machine
    write_cli_store(market_root(tmp_path))
    result = run_truncation(
        tmp_path,
        overrides=[MERGE],
        baseline=[*LEAK_KEYS, OTHER_KEY],
        prune=True,
        instants=CLI_INSTANTS,
        jobs="2",
    )
    assert result.exit_code == 0, result.output
    assert load_baseline(tmp_path / "baseline.txt") == LEAK_KEYS


def test_cli_default_jobs_fit_a_small_machine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # a CI runner has 2-4 cores: the default must shrink to them, not be refused
    monkeypatch.setattr(cli, "_cores", lambda: 2)
    used: list[int] = []
    real = cli.run_instants

    def spy(*args: Any, **kwargs: Any) -> Any:
        used.append(kwargs["jobs"])
        return real(*args, **kwargs)

    monkeypatch.setattr(cli, "run_instants", spy)
    write_cli_store(market_root(tmp_path))
    result = run_truncation(tmp_path, overrides=[MERGE], baseline=LEAK_KEYS, instants=CLI_INSTANTS)
    assert result.exit_code == 0, result.output
    assert used == [2]


def test_cli_default_jobs_are_bounded_by_memory_not_cores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 2026-10-20 on the real store: 6 workers ran out of a 31 GB machine (the kernel killed one
    # at the 2026 instants, ~6 GB each); 4 peaked at 22 GB in all. More cores add no memory.
    monkeypatch.setattr(cli, "_cores", lambda: 32)
    used: list[int] = []
    real = cli.run_instants

    def spy(*args: Any, **kwargs: Any) -> Any:
        used.append(kwargs["jobs"])
        return real(*args, **kwargs)

    monkeypatch.setattr(cli, "run_instants", spy)
    write_cli_store(market_root(tmp_path))
    result = run_truncation(tmp_path, overrides=[MERGE], baseline=LEAK_KEYS, instants=CLI_INSTANTS)
    assert result.exit_code == 0, result.output
    assert used == [4]


def test_cli_refuses_more_jobs_than_cores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_cores", lambda: 2)
    write_cli_store(market_root(tmp_path))
    result = run_truncation(tmp_path, overrides=[MERGE], instants=CLI_INSTANTS, jobs="3")
    assert result.exit_code not in (0, LEAK_EXIT), result.output
    assert "between 1 and 2" in result.output


@pytest.mark.parametrize("jobs", ["0", "-2"])
def test_cli_refuses_fewer_than_one_job_before_any_build(
    tmp_path: Path, jobs: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_cli_store(market_root(tmp_path))

    def no_build(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("built although --jobs was refused")

    monkeypatch.setattr(sm, "build_security_master", no_build)
    result = run_truncation(tmp_path, overrides=[MERGE], instants=CLI_INSTANTS, jobs=jobs)

    assert result.exit_code not in (0, LEAK_EXIT), result.output
    assert result.exception is None or isinstance(result.exception, SystemExit), result.output
    assert "jobs" in result.output
    assert not report_dir(tmp_path).exists() or not list(report_dir(tmp_path).glob("*"))
