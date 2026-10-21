"""Guards against triaging truncation leaks by the wrong yardstick: a difference between the full
security master and one rebuilt from the inputs knowable at ``as_of`` matters only when it
changes what a backtest at ``as_of`` reads -- (1) the daily price row a security shows on a day,
(2) the CIK a symbol shows on a day. A measurement that counts a pure relabelling (the same
prices read through another symbol, COHR / IIVI 2022-08) as a leak sends us fixing harmless keys;
one that reads bars, splits or master rows not yet knowable at ``as_of`` invents impact that no
reader could have seen.

Target API (implementation pending), ``asof.ingest.truncation_impact``:

    IMPACT_SCHEMA = kind (leak | delay), what (price | cik), security, symbol, full_segment,
        trunc_segment, first_day, last_day, days, detail, sessions
        sessions: list<date32>, the sorted distinct session dates the row covers;
        days == len(sessions), first_day == sessions[0], last_day == sessions[-1]
    impact_at(full, truncated, bars, splits, as_of, *, liquid=None, exhaustive=False) -> pa.Table
        a price cell present on both sides and differing is reported through the bars of both
        sides (rows with equal grouping columns merge); exhaustive=True compares every
        (security, day) cell instead of only the candidate cells, a verification mode whose
        output equals the default's
    @dataclass(frozen=True) InstantResult(as_of: datetime UTC, diff: pa.Table, impact: pa.Table)
        diff = master_truncation.compare(full, truncated, as_of) (DIFF_SCHEMA)
        impact = impact_at(full, truncated, inputs.bars, splits, as_of, liquid=liquid)
    run_instants(full, inputs, overrides, splits, instants, *, liquid=None, jobs=1, progress=None)
        -> list[InstantResult]: ONE truncated build per distinct instant feeds both; sorted by
        instant, deduplicated; a naive instant or jobs < 1 -> ValueError; jobs > 1 = a fork
        process pool (parallel_map), results identical to jobs=1; progress(result) in instant
        order
    removed: run_impact, summarize, KEY_IMPACT_SCHEMA, unmatched_leaks (the per-key attribution
    is asof.ingest.truncation_gate.attribute, tests/ingest/test_truncation_gate.py, which also
    tests the `asof market truncation` command; `asof market truncation-impact` is gone)

Fake data provenance:
- Masters are hand-made MASTER_SCHEMA rows (``seg_row``: a segment knowable from 00:00 New York
  of its first day, an end from 00:00 of its last, a CIK with its segment unless said later),
  the same conventions as tests/ingest/test_master_truncation.py.
- A daily bar is knowable from 20:00 New York of its session (measured: the real 2020-01-02 bar
  has available_at 2020-01-03 01:00 UTC; ``bar_available_at``).
- Volume-0 bars: Alpaca writes placeholder bars with volume 0 after a rename (measured: 1.18M of
  them in the real store, see asof.store.universe); a reader drops them.
- A split record is knowable from 00:00 New York of its ex_date (documented:
  ``corporate_actions.available_at``); the leak cases set a later ``available_at`` explicitly.
- Relabelled copies with identical OHLCV under a new symbol: measured on the real store (COHR's
  bars from 2022-08-01, closes 52.92, 52.80, 53.23, read by the full master through COHR and by
  the build at 2022-09-04 through IIVI). The real-store standard case at the bottom checks it.
- Random worlds (exhaustive mode): constructed, not modelled on any source; they only have to
  exercise segments, groupings, CIKs, copies, placeholders, splits and late instants at once.
- The CLI store is the CYCN / KRSA world of test_master_truncation.py with volumes raised to
  ~10M shares a day (~$70M at $7), constructed so that the merged security is in the liquid tier;
  every segment is classed "common" (constructed).

Real-data checks for the assumptions (no as_of-free assumption here is left unchecked):
- Standard case below: at 2022-09-04 00:00 New York the COHR / IIVI relabelling shows no price
  leak on IIVI@2020-01-02 in 2022-08-01..2022-09-02 and no CIK leak on COHR or IIVI.
- Standard case below (positive): one microsecond before 2025-07-25 00:00 New York the full
  master shows Barrick's CIK 756894 on Barnes Group's B history (2020-01-02..2025-05-08) and
  continues Barnes's security with Barrick's B bars from 2025-05-09: a CIK leak and a price leak.
- Exhaustive mode equals the default on the real store at 2022-09-04 (COHR, IIVI, B, YPF, AAPL).
- Audit command: ``asof market truncation`` on the real store at the fixed instants; every leak
  cell is attributed to a diff key of its instant (none unmatched) and the per-key counts stand
  in configs/truncation_impact_baseline.txt, read by a human.
"""

import dataclasses
import random
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from asof.ingest import security_master as sm
from asof.ingest.corporate_actions import SPLIT_SCHEMA
from asof.ingest.corporate_actions import available_at as split_known
from asof.ingest.instrument_class import CLASS_FILE, CLASS_SCHEMA
from asof.ingest.master_truncation import DIFF_SCHEMA, compare, diff_key
from asof.ingest.security_master import build_security_master
from asof.ingest.truncation_impact import (
    IMPACT_SCHEMA,
    InstantResult,
    impact_at,
    run_instants,
)
from asof.store.db import default_data_dir
from asof.store.market import market_root
from tests.ingest.fakes import daily
from tests.ingest.master_scenarios import SCENARIOS
from tests.ingest.test_master_truncation import (
    CLI_DAYS,
    CLI_DECIDED,
    CLI_SNAPSHOTS,
    LEAK_KEYS,
    MERGE,
    MICRO,
    cli_instants,
    cli_world,
    history_table,
    instants_of,
    no_children_left,
    store_inputs_of,
    within,
)
from tests.ingest.test_security_master import (
    bar_available_at,
    bars_table,
    business_days,
    et_midnight,
    master_table,
    seg_row,
)
from tests.store.test_market import store_daily

ET = ZoneInfo("America/New_York")
D0, CUT = date(2020, 1, 2), date(2020, 1, 6)
DAYS = business_days(D0, date(2020, 1, 8))  # 01-02, 01-03, 01-06, 01-07, 01-08
BEFORE = [d for d in DAYS if d < CUT]  # 01-02, 01-03
AFTER = [d for d in DAYS if d >= CUT]  # 01-06, 01-07, 01-08
AS_OF = et_midnight(date(2020, 2, 3))  # every bar, segment, end and split above is knowable
LATE = et_midnight(date(2020, 3, 2))  # what the leak cases make knowable only later

# (kind, what, security, symbol, full_segment, trunc_segment, first_day, last_day, days)
Fact = tuple[str, str, str | None, str, str | None, str | None, date, date, int]


def facts(table: pa.Table) -> list[Fact]:
    """The rows in table order, without the free-form detail."""
    return [
        (r["kind"], r["what"], r["security"], r["symbol"], r["full_segment"], r["trunc_segment"],
         r["first_day"], r["last_day"], r["days"])
        for r in table.to_pylist()
    ]  # fmt: skip


def check_sessions(table: pa.Table) -> None:
    """The ``sessions`` invariant on every row: sorted, distinct, ``days`` of them, from
    ``first_day`` to ``last_day``."""
    for r in table.to_pylist():
        sessions = r["sessions"]
        assert sessions == sorted(set(sessions)), r
        assert len(sessions) == r["days"], r
        assert (sessions[0], sessions[-1]) == (r["first_day"], r["last_day"]), r


def bar(
    symbol: str,
    day: date,
    close: float = 10.0,
    *,
    volume: int = 1_000,
    known: datetime | None = None,
    **fields: Any,
) -> dict[str, Any]:
    row = {
        "symbol": symbol,
        "session_date": day,
        "open": close - 0.1,
        "high": close + 0.2,
        "low": close - 0.2,
        "close": close,
        "volume": volume,
        "available_at": bar_available_at(day) if known is None else known,
    }
    return {**row, **fields}


def series(symbol: str, days: Sequence[date], base: float = 10.0) -> list[dict[str, Any]]:
    """One bar a day with a drifting close and volume, so no two days are equal."""
    return [bar(symbol, d, base + i / 100, volume=1_000 + i) for i, d in enumerate(days)]


def copy_as(rows: Sequence[dict[str, Any]], symbol: str) -> list[dict[str, Any]]:
    """Alpaca's relabelling: the same bars filed under another symbol."""
    return [{**r, "symbol": symbol} for r in rows]


def splits_table(rows: Sequence[tuple[str, date, float, datetime | None]] = ()) -> pa.Table:
    return pa.Table.from_pylist(
        [
            {"symbol": s, "ex_date": d, "old_rate": 1.0, "new_rate": f, "factor": f,
             "kind": "forward_split", "old_cusip": None, "new_cusip": None,
             "available_at": split_known(d) if known is None else known}
            for s, d, f, known in rows
        ],
        schema=SPLIT_SCHEMA,
    )  # fmt: skip


def impact(
    full: Sequence[dict[str, Any]],
    truncated: Sequence[dict[str, Any]],
    bars: Sequence[dict[str, Any]],
    splits: pa.Table | None = None,
    as_of: datetime = AS_OF,
    liquid: Collection[str] | None = None,
) -> pa.Table:
    out = impact_at(
        master_table(full),
        master_table(truncated),
        bars_table(list(bars)),
        splits_table() if splits is None else splits,
        as_of,
        liquid=liquid,
    )
    assert out.schema.equals(IMPACT_SCHEMA)
    check_sessions(out)
    return out


# =============================================================================================
# schemas
# =============================================================================================


def test_impact_schema() -> None:
    assert IMPACT_SCHEMA.names == [
        "kind", "what", "security", "symbol", "full_segment", "trunc_segment",
        "first_day", "last_day", "days", "detail", "sessions",
    ]  # fmt: skip
    for name in ("kind", "what", "security", "symbol", "full_segment", "trunc_segment", "detail"):
        assert IMPACT_SCHEMA.field(name).type == pa.string(), name
    assert IMPACT_SCHEMA.field("first_day").type == pa.date32()
    assert IMPACT_SCHEMA.field("last_day").type == pa.date32()
    assert IMPACT_SCHEMA.field("days").type == pa.int64()
    # the cells a row covers, for the gate's per-cell attribution (asof.ingest.truncation_gate)
    assert IMPACT_SCHEMA.field("sessions").type == pa.list_(pa.date32())


# =============================================================================================
# impact_at: normal
# =============================================================================================


def renamed_world() -> list[dict[str, Any]]:
    return [
        seg_row("S1", "A", D0, CUT, cik=1111),
        seg_row("S1", "X", CUT, cik=1111),  # A renamed to X
        seg_row("S2", "B", D0),
    ]


def test_a_master_compared_with_itself_has_no_impact() -> None:
    bars = [*series("A", BEFORE), *series("X", AFTER, 20.0), *series("B", DAYS, 30.0)]
    splits = splits_table([("X", date(2020, 1, 7), 2.0, None)])
    out = impact(renamed_world(), renamed_world(), bars, splits)
    assert out.num_rows == 0


def test_security_id_numbering_is_not_an_impact() -> None:
    # ids are positional: a truncated build numbers its securities afresh; names are compared
    renumber = {"S1": "S7", "S2": "S1"}
    truncated = [{**r, "security_id": renumber[r["security_id"]]} for r in renamed_world()]
    bars = [*series("A", BEFORE), *series("X", AFTER, 20.0), *series("B", DAYS, 30.0)]
    assert impact(renamed_world(), truncated, bars).num_rows == 0


def test_cli_world_compared_with_itself_has_no_impact(tmp_path: Path) -> None:
    _, inputs, full = cli_world(tmp_path)
    for t in cli_instants(full):
        out = impact_at(full, full, inputs.bars, splits_table(), t)
        assert out.schema.equals(IMPACT_SCHEMA), t
        assert out.num_rows == 0, t
        assert out.column("sessions").type == pa.list_(pa.date32()), t


@pytest.mark.parametrize("with_cik", [False, True], ids=["no_cik", "cik"])
def test_the_same_prices_through_another_symbol_are_no_price_impact(with_cik: bool) -> None:
    # COHR-like. The full master gives A's security the days from CUT through symbol X; the
    # truncated build gives the same security (same name A@2020-01-02) those days through Y,
    # whose bars are Alpaca's identical copies. A backtest reads the same price rows: nothing.
    # A CIK on X is another matter: a reader asking "which company is X" gets an answer from the
    # full master and none from the truncated one -- reported.
    cik = 1111 if with_cik else None
    full = [seg_row("S1", "A", D0, CUT), seg_row("S1", "X", CUT, cik=cik)]
    truncated = [seg_row("S1", "A", D0, CUT), seg_row("S1", "Y", CUT, cik=cik)]
    x = series("X", AFTER, 20.0)
    bars = [*series("A", BEFORE), *x, *copy_as(x, "Y")]

    out = impact(full, truncated, bars)

    assert [f for f in facts(out) if f[1] == "price"] == []
    expected: list[Fact] = []
    if with_cik:
        expected = [
            ("delay", "cik", None, "Y", None, "Y:2020-01-06", AFTER[0], AFTER[-1], 3),
            ("leak", "cik", None, "X", "X:2020-01-06", None, AFTER[0], AFTER[-1], 3),
        ]
    assert facts(out) == expected


def test_the_same_days_grouped_into_another_security_are_a_leak_and_a_delay() -> None:
    # Equal prices, different owner: the full master puts Y's days into A's security, the
    # truncated build into a security of their own. A backtest on A reads 3 rows it could not
    # have read; one on Y's security loses them.
    full = [seg_row("S1", "A", D0, CUT), seg_row("S1", "Y", CUT)]
    truncated = [seg_row("S1", "A", D0, CUT), seg_row("S2", "Y", CUT)]
    bars = [*series("A", BEFORE), *series("Y", AFTER, 20.0)]

    out = impact(full, truncated, bars)

    assert facts(out) == [
        ("delay", "price", "Y@2020-01-06", "Y", "Y:2020-01-06", "Y:2020-01-06",
         AFTER[0], AFTER[-1], 3),
        ("leak", "price", "A@2020-01-02", "Y", "Y:2020-01-06", "Y:2020-01-06",
         AFTER[0], AFTER[-1], 3),
    ]  # fmt: skip


def joined_world(b_in_full: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Truncated: B filed into A's security (one id, named A@2020-01-02 by its earliest segment,
    A before B on the same day). Full: B a security of its own, or unknown."""
    full = [seg_row("S1", "A", D0), *([seg_row("S2", "B", D0)] if b_in_full else [])]
    truncated = [seg_row("S1", "A", D0), seg_row("S1", "B", D0)]
    return full, truncated


H1_A: Fact = ("leak", "price", "A@2020-01-02", "A", "A:2020-01-02", "A:2020-01-02",
              DAYS[0], DAYS[-1], 5)  # fmt: skip


@pytest.mark.parametrize("b_in_full", [True, False], ids=["b_own_security", "b_unknown_to_full"])
def test_a_differing_cell_reports_the_bars_of_both_sides(b_in_full: bool) -> None:
    # H1. A and B trade every day at different prices. Cell (A@2020-01-02, d) is {A's bar} in
    # full and {A's bar, B's bar} truncated: it differs on all 5 days. Its leak is evidenced by
    # the bars of both sides:
    # - full side: A's bar -> H1_A
    # - truncated side: A's bar (the same grouping columns as H1_A: merged into it, still 5
    #   days) and B's bar -> (A@2020-01-02, B, full owner of (B, d), B:2020-01-02), 5 days.
    #   The full owner is B:2020-01-02 (b_own_security) or nobody (b_unknown_to_full).
    # Before H1 the B row was missing: the bars that made A's cell differ were named nowhere.
    # b_own_security also has the full-only cell (B@2020-01-02, d): a leak through B's bar, its
    # truncated owner B:2020-01-02 (as before). No CIKs: no cik rows. The truncated cell
    # (A@2020-01-02, d) exists in full: no delay.
    full, truncated = joined_world(b_in_full)
    bars = [*series("A", DAYS), *series("B", DAYS, 30.0)]

    out = impact(full, truncated, bars)

    b_full = "B:2020-01-02" if b_in_full else None
    via_b: Fact = ("leak", "price", "A@2020-01-02", "B", b_full, "B:2020-01-02",
                   DAYS[0], DAYS[-1], 5)  # fmt: skip
    b_own: Fact = ("leak", "price", "B@2020-01-02", "B", "B:2020-01-02", "B:2020-01-02",
                   DAYS[0], DAYS[-1], 5)  # fmt: skip
    # sorted by symbol, then security (A@ < B@)
    assert facts(out) == [H1_A, via_b, *([b_own] if b_in_full else [])]
    # the liquid filter goes by the cell's security: S1 (A's id in full) keeps both rows of
    # A@2020-01-02's cells, the truncated side's included; S2 only B's own cells
    assert facts(impact(full, truncated, bars, liquid={"S1"})) == [H1_A, via_b]
    if b_in_full:
        assert facts(impact(full, truncated, bars, liquid={"S2"})) == [b_own]


def test_a_cell_missing_from_the_full_master_gets_no_truncated_side_leak_rows() -> None:
    # The truncated build ends A's first segment at CUT and starts a new security A@2020-01-06;
    # the full master keeps A open in A@2020-01-02. From CUT, cell (A@2020-01-02, d) is in full
    # only (a leak through A's bar, truncated owner A:2020-01-06) and (A@2020-01-06, d) in
    # truncated only (a delay). No cell is on both sides and differing, so H1 adds nothing: a
    # cell the full master lacks is a delay, never a leak.
    full = [seg_row("S1", "A", D0)]
    truncated = [seg_row("S1", "A", D0, CUT), seg_row("S2", "A", CUT)]

    out = impact(full, truncated, series("A", DAYS))

    assert facts(out) == [
        ("delay", "price", "A@2020-01-06", "A", "A:2020-01-02", "A:2020-01-06",
         AFTER[0], AFTER[-1], 3),
        ("leak", "price", "A@2020-01-02", "A", "A:2020-01-02", "A:2020-01-06",
         AFTER[0], AFTER[-1], 3),
    ]  # fmt: skip


def relabel_world() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Full: A open from D0. Truncated: A ends at CUT, Z continues A's security."""
    full = [seg_row("S1", "A", D0)]
    truncated = [seg_row("S1", "A", D0, CUT), seg_row("S1", "Z", CUT)]
    return full, truncated


@pytest.mark.parametrize(
    "change",
    [{"close": 99.0}, {"volume": 1}, {"high": 99.0}, {}],
    ids=["close", "volume", "high", "identical"],
)
def test_a_different_bar_on_the_same_security_day_is_a_leak(change: dict[str, Any]) -> None:
    # The truncated build reads A's security through Z from CUT; Z copies A's bars except on
    # 01-07, where one field differs. Only that day is read differently.
    full, truncated = relabel_world()
    a = series("A", DAYS)
    z = [{**r, **change} if r["session_date"] == date(2020, 1, 7) else r for r in copy_as(a, "Z")]
    z = [r for r in z if r["session_date"] >= CUT]

    out = impact(full, truncated, [*a, *z])

    day = date(2020, 1, 7)
    # the cell is on both sides: the symbol and segments of the full side's bar; (A, 01-07) has
    # no owner in the truncated master (A ends at CUT)
    expected = [
        ("leak", "price", "A@2020-01-02", "A", "A:2020-01-02", None, day, day, 1),
        # H1: the truncated side reads the cell through Z, named too
        ("leak", "price", "A@2020-01-02", "Z", None, "Z:2020-01-06", day, day, 1),
    ]
    assert facts(out) == (expected if change else [])


@pytest.mark.parametrize(
    ("split_on", "differs"),
    [(("A",), True), (("Z",), True), (("A", "Z"), False), (("Q",), False)],
    ids=["full_symbol", "truncated_symbol", "both", "unrelated"],
)
def test_a_split_recorded_for_one_sides_symbol_only_is_a_leak(
    split_on: tuple[str, ...], differs: bool
) -> None:
    # A reader adjusting prices reads the split record of the bar's symbol: equal OHLCV with a
    # 2:1 split on one side only is a different price history.
    full, truncated = relabel_world()
    a = series("A", DAYS)
    z = [r for r in copy_as(a, "Z") if r["session_date"] >= CUT]
    day = date(2020, 1, 7)
    splits = splits_table([(s, day, 2.0, None) for s in split_on])

    out = impact(full, truncated, [*a, *z], splits)

    expected = [
        ("leak", "price", "A@2020-01-02", "A", "A:2020-01-02", None, day, day, 1),
        # H1: the truncated side reads the cell through Z, named too
        ("leak", "price", "A@2020-01-02", "Z", None, "Z:2020-01-06", day, day, 1),
    ]
    assert facts(out) == (expected if differs else [])


@pytest.mark.parametrize("only_in", ["full", "truncated"])
def test_a_security_on_one_side_only_is_a_leak_or_a_delay(only_in: str) -> None:
    base = [seg_row("S1", "A", D0)]
    more = [*base, seg_row("S2", "E", CUT)]
    full, truncated = (more, base) if only_in == "full" else (base, more)
    bars = [*series("A", DAYS), *series("E", AFTER, 40.0)]

    out = impact(full, truncated, bars)

    if only_in == "full":
        expected = ("leak", "price", "E@2020-01-06", "E", "E:2020-01-06", None)
    else:
        expected = ("delay", "price", "E@2020-01-06", "E", None, "E:2020-01-06")
    assert facts(out) == [(*expected, AFTER[0], AFTER[-1], 3)]


@pytest.mark.parametrize(
    ("full_cik", "trunc_cik", "kind"),
    [
        (1111, 2222, "leak"),  # another company
        (1111, None, "leak"),  # the full master names a company the build could not
        (None, 2222, "delay"),  # the full master withholds what a build then had
        (1111, 1111, None),
        (None, None, None),
    ],
    ids=["differ", "full_only", "truncated_only", "equal", "neither"],
)
def test_cik_per_symbol_day(full_cik: int | None, trunc_cik: int | None, kind: str | None) -> None:
    full = [seg_row("S1", "A", D0, cik=full_cik)]
    truncated = [seg_row("S1", "A", D0, cik=trunc_cik)]

    out = impact(full, truncated, series("A", DAYS))

    expected: list[Fact] = []
    if kind is not None:
        expected = [(kind, "cik", None, "A", "A:2020-01-02", "A:2020-01-02", DAYS[0], DAYS[-1], 5)]
    assert facts(out) == expected


def test_cik_leak_detail_names_both_ciks() -> None:
    full = [seg_row("S1", "A", D0, cik=1111)]
    truncated = [seg_row("S1", "A", D0, cik=2222)]
    [row] = impact(full, truncated, series("A", DAYS)).to_pylist()
    assert "1111" in row["detail"] and "2222" in row["detail"]


def test_a_cik_symbol_day_without_any_owner_on_either_side_is_nothing() -> None:
    # bars of a symbol neither master knows are read by nobody
    out = impact([seg_row("S1", "A", D0)], [seg_row("S1", "A", D0)], series("NOBODY", DAYS))
    assert out.num_rows == 0


# =============================================================================================
# impact_at: edges
# =============================================================================================


def test_untraded_bars_are_not_read() -> None:
    full = [seg_row("S1", "A", D0), seg_row("S2", "E", CUT)]
    truncated = [seg_row("S1", "A", D0)]
    e = [bar("E", d, 40.0, volume=0 if d != date(2020, 1, 7) else 500) for d in AFTER]

    out = impact(full, truncated, [*series("A", DAYS), *e])

    day = date(2020, 1, 7)
    assert facts(out) == [("leak", "price", "E@2020-01-06", "E", "E:2020-01-06", None, day, day, 1)]
    only_placeholders = [{**r, "volume": 0} for r in e]
    assert impact(full, truncated, [*series("A", DAYS), *only_placeholders]).num_rows == 0


def test_an_untraded_bar_does_not_make_two_equal_cells_differ() -> None:
    # the truncated side reads Z's copies plus a volume-0 placeholder of A on the same days; the
    # placeholder is dropped before the cells are compared
    full, truncated = relabel_world()
    a = series("A", DAYS)
    z = [r for r in copy_as(a, "Z") if r["session_date"] >= CUT]
    truncated = [*truncated, seg_row("S1", "P", CUT)]
    placeholders = [{**r, "symbol": "P", "volume": 0} for r in z]
    assert impact(full, truncated, [*a, *z, *placeholders]).num_rows == 0


def test_no_bars_is_an_empty_table_with_the_schema() -> None:
    full = [seg_row("S1", "A", D0, cik=1111), seg_row("S2", "E", CUT)]
    truncated = [seg_row("S1", "A", D0, cik=2222)]
    out = impact(full, truncated, [])
    assert out.num_rows == 0
    assert out.schema.equals(IMPACT_SCHEMA)


def test_empty_masters_are_an_empty_table() -> None:
    assert impact([], [], series("A", DAYS)).num_rows == 0


def test_days_count_distinct_sessions_and_rows_split_by_segment() -> None:
    # E trades on 01-02, 01-06 and 01-08 only; its security has two segments (a CIK change at
    # CUT, same symbol). One row per (security, symbol, full_segment, trunc_segment): the days
    # are counted, not spanned.
    full = [
        seg_row("S1", "A", D0),
        seg_row("S2", "E", D0, CUT),
        seg_row("S2", "E", CUT),
    ]
    truncated = [seg_row("S1", "A", D0)]
    e = [bar("E", d, 40.0) for d in (date(2020, 1, 2), date(2020, 1, 6), date(2020, 1, 8))]

    out = impact(full, truncated, [*series("A", DAYS), *e])

    assert facts(out) == [
        ("leak", "price", "E@2020-01-02", "E", "E:2020-01-02", None,
         date(2020, 1, 2), date(2020, 1, 2), 1),
        ("leak", "price", "E@2020-01-02", "E", "E:2020-01-06", None,
         date(2020, 1, 6), date(2020, 1, 8), 2),
    ]  # fmt: skip
    # the sessions themselves: 01-07 is inside the span but has no bar of E
    assert out.column("sessions").to_pylist() == [
        [date(2020, 1, 2)],
        [date(2020, 1, 6), date(2020, 1, 8)],
    ]


def test_rows_are_sorted_by_what_kind_symbol_first_day() -> None:
    full = [seg_row("S1", "A", D0, cik=1111), seg_row("S2", "E", CUT)]
    truncated = [seg_row("S1", "A", D0, cik=2222), seg_row("S3", "F", D0, cik=3333)]
    bars = [*series("A", DAYS), *series("E", AFTER, 40.0), *series("F", DAYS, 50.0)]

    out = impact(full, truncated, bars)

    # cik < price, delay < leak
    assert facts(out) == [
        ("delay", "cik", None, "F", None, "F:2020-01-02", DAYS[0], DAYS[-1], 5),
        ("leak", "cik", None, "A", "A:2020-01-02", "A:2020-01-02", DAYS[0], DAYS[-1], 5),
        ("delay", "price", "F@2020-01-02", "F", None, "F:2020-01-02", DAYS[0], DAYS[-1], 5),
        ("leak", "price", "E@2020-01-06", "E", "E:2020-01-06", None, AFTER[0], AFTER[-1], 3),
    ]
    assert all(isinstance(r["detail"], str) for r in out.to_pylist())


def test_impact_at_refuses_a_naive_as_of() -> None:
    table = master_table([seg_row("S1", "A", D0)])
    with pytest.raises(ValueError):
        impact_at(table, table, bars_table(series("A", DAYS)), splits_table(), datetime(2020, 2, 3))


# --- the liquid filter ---


def liquid_world() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """A's end is a delay (truncated keeps A open), E is a leak (full only)."""
    full = [seg_row("S1", "A", D0, CUT), seg_row("S2", "E", CUT)]
    truncated = [seg_row("S1", "A", D0)]
    return full, truncated, [*series("A", DAYS), *series("E", AFTER, 40.0)]


A_DELAY: Fact = (
    "delay", "price", "A@2020-01-02", "A", None, "A:2020-01-02", AFTER[0], AFTER[-1], 3
)  # fmt: skip
E_LEAK: Fact = ("leak", "price", "E@2020-01-06", "E", "E:2020-01-06", None, AFTER[0], AFTER[-1], 3)


@pytest.mark.parametrize(
    ("liquid", "expected"),
    [
        (None, [A_DELAY, E_LEAK]),
        ({"S1", "S2"}, [A_DELAY, E_LEAK]),
        # a cell only the truncated side has is kept: its name is a liquid security of the full
        # master
        ({"S1"}, [A_DELAY]),
        ({"S2"}, [E_LEAK]),
        (set(), []),
        ({"S9"}, []),  # an id the full master does not have
    ],
    ids=["none", "both", "a_only", "e_only", "empty", "unknown"],
)
def test_liquid_keeps_cells_of_liquid_securities(
    liquid: set[str] | None, expected: list[Fact]
) -> None:
    full, truncated, bars = liquid_world()
    assert facts(impact(full, truncated, bars, liquid=liquid)) == expected


def test_liquid_ids_are_full_master_ids_not_truncated_ones() -> None:
    # the truncated build numbers A's security S7; the filter speaks the full master's ids
    full, truncated, bars = liquid_world()
    truncated = [{**r, "security_id": "S7"} for r in truncated]
    assert facts(impact(full, truncated, bars, liquid={"S1"})) == [A_DELAY]
    assert facts(impact(full, truncated, bars, liquid={"S7"})) == []


@pytest.mark.parametrize(
    ("liquid", "kept"),
    [({"S1"}, True), ({"S2"}, True), ({"S3"}, False)],
    ids=["truncated_owner_liquid", "full_owner_liquid", "neither"],
)
def test_liquid_keeps_a_cik_row_whose_owner_is_liquid_on_either_side(
    liquid: set[str], kept: bool
) -> None:
    # X is a security of its own in the full master (S2, CIK 5555) and part of A's security in
    # the truncated build (CIK 6666): its owner's name is X@2020-01-02 in full and A@2020-01-02
    # in truncated
    full = [seg_row("S1", "A", D0), seg_row("S2", "X", D0, cik=5555), seg_row("S3", "B", D0)]
    truncated = [seg_row("S1", "A", D0), seg_row("S1", "X", D0, cik=6666), seg_row("S3", "B", D0)]
    bars = [*series("A", DAYS), *series("X", DAYS, 20.0), *series("B", DAYS, 30.0)]

    out = impact(full, truncated, bars, liquid=liquid)

    cik_rows = [f for f in facts(out) if f[1] == "cik"]
    expected = [("leak", "cik", None, "X", "X:2020-01-02", "X:2020-01-02", DAYS[0], DAYS[-1], 5)]
    assert cik_rows == (expected if kept else [])


# --- exhaustive mode: every cell compared, the same output ---

EX_DAYS = business_days(date(2020, 1, 2), date(2020, 1, 15))  # 10 sessions
EX_SYMBOLS = ("A", "B", "C", "D")
EX_IDS = ("S1", "S2", "S3", "S4")


def random_master(rng: random.Random, symbols: Sequence[str]) -> list[dict[str, Any]]:
    """Per symbol 1-3 consecutive non-overlapping segments (some dropped: gaps), each in a random
    security, with a random CIK; some segments, ends and CIKs knowable only at LATE."""
    rows = []
    for s in symbols:
        cuts = sorted(rng.sample(EX_DAYS[1:], rng.randint(0, 2)))
        starts = [EX_DAYS[0], *cuts]
        last_end = rng.choice([None, None, EX_DAYS[-2]])
        if last_end is not None and last_end <= starts[-1]:
            last_end = None
        ends = [*cuts, last_end]
        for start, end in zip(starts, ends, strict=True):
            if rng.random() < 0.15:
                continue
            cik = rng.choice([None, 1111, 2222])
            rows.append(
                seg_row(
                    rng.choice(EX_IDS[:3]), s, start, end,
                    available_at=rng.choice([None, None, None, LATE]),
                    end_available_at=None if end is None else rng.choice([None, None, LATE]),
                    cik=cik,
                    cik_available_at=None if cik is None else rng.choice([None, None, LATE]),
                )
            )  # fmt: skip
    return rows


def perturbed(rng: random.Random, rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The full master with 0-3 changes a truncated build may show: another grouping, another
    CIK, a segment missing, a segment knowable only later."""
    out = [dict(r) for r in rows]
    for _ in range(rng.randint(0, 3)):
        if not out:
            break
        i = rng.randrange(len(out))
        change = rng.choice(["regroup", "cik", "drop", "late"])
        if change == "regroup":
            out[i]["security_id"] = rng.choice(EX_IDS)
        elif change == "cik":
            cik = rng.choice([None, 1111, 2222, 3333])
            known = out[i]["cik_available_at"] or out[i]["available_at"]
            out[i] = {**out[i], "cik": cik, "cik_available_at": None if cik is None else known}
        elif change == "drop":
            del out[i]
        else:
            out[i]["available_at"] = LATE
    return out


def random_bars(rng: random.Random, symbols: Sequence[str]) -> list[dict[str, Any]]:
    """At most one bar per (symbol, day), closes and volumes from small sets (equal bars across
    symbols happen), some explicit copies of another symbol's bar, some volume-0 placeholders,
    some filed only at LATE; Q is a symbol no master knows."""
    by_key: dict[tuple[str, date], dict[str, Any]] = {}
    for s in (*symbols, "Q"):
        for d in EX_DAYS:
            if rng.random() < 0.2:
                continue
            by_key[(s, d)] = bar(
                s, d, rng.choice([10.0, 10.5, 11.0]),
                volume=rng.choice([0, 1_000, 1_000, 2_000]),
                known=LATE if rng.random() < 0.1 else None,
            )  # fmt: skip
    for _ in range(rng.randint(0, 4)):
        if not by_key:
            break
        (s, d), row = rng.choice(sorted(by_key.items(), key=lambda kv: kv[0]))
        target = rng.choice([x for x in symbols if x != s] or [s])
        by_key.setdefault((target, d), {**row, "symbol": target})
    return [by_key[k] for k in sorted(by_key)]


def random_world(seed: int) -> tuple[pa.Table, pa.Table, pa.Table, pa.Table, datetime, set[str]]:
    rng = random.Random(seed)
    symbols = rng.sample(EX_SYMBOLS, rng.randint(2, 4))
    full = random_master(rng, symbols)
    truncated = random_master(rng, symbols) if rng.random() < 0.3 else perturbed(rng, full)
    bars = random_bars(rng, symbols)
    splits = sorted({(rng.choice(symbols), rng.choice(EX_DAYS)) for _ in range(rng.randint(0, 2))})
    split_rows = [(s, d, 2.0, rng.choice([None, LATE])) for s, d in splits]
    as_of = rng.choice([AS_OF, et_midnight(date(2020, 1, 9)), LATE])
    ids = sorted({r["security_id"] for r in full})
    liquid = set(rng.sample(ids, rng.randint(0, len(ids))))
    if rng.random() < 0.2:
        liquid.add("S9")  # an id the full master does not have
    return (master_table(full), master_table(truncated), bars_table(bars),
            splits_table(split_rows), as_of, liquid)  # fmt: skip


def test_exhaustive_equals_the_candidate_cells_on_random_worlds() -> None:
    # The default compares only cells holding a bar owned differently on the two sides; the
    # exhaustive mode compares every (security, day) cell. Any cell the candidate step misses
    # shows up as a difference here.
    seen: set[tuple[str, str]] = set()
    nonempty = 0
    for seed in range(30):
        full, truncated, bars, splits, as_of, liquid = random_world(seed)
        for scope in (None, liquid):
            default = impact_at(full, truncated, bars, splits, as_of, liquid=scope)
            every = impact_at(full, truncated, bars, splits, as_of, liquid=scope, exhaustive=True)
            assert every.schema.equals(IMPACT_SCHEMA), seed
            # equals compares the sessions lists too
            assert every.equals(default), (seed, scope, facts(every), facts(default))
            check_sessions(default)
            check_sessions(every)
            nonempty += default.num_rows > 0
            seen |= {(r["kind"], r["what"]) for r in default.to_pylist()}
    # not a comparison of empty tables: most worlds differ, and in every kind of way
    assert nonempty >= 20, nonempty
    assert seen == {("leak", "price"), ("delay", "price"), ("leak", "cik"), ("delay", "cik")}


def test_exhaustive_is_keyword_only_and_defaults_to_false() -> None:
    full, truncated, bars, splits, as_of, _ = random_world(0)
    default = impact_at(full, truncated, bars, splits, as_of)
    assert impact_at(full, truncated, bars, splits, as_of, exhaustive=False).equals(default)
    with pytest.raises(TypeError):
        impact_at(full, truncated, bars, splits, as_of, None, True)  # type: ignore[misc]


# =============================================================================================
# impact_at: leakage -- future rows on disk, invisible at as_of
# =============================================================================================


def test_a_bar_not_yet_knowable_is_not_read() -> None:
    # Leakage. Z's 01-09 bar differs from A's; both are on disk, knowable from 20:00 of 01-09.
    # At 16:00 that day (and one microsecond before 20:00) nobody could read either.
    full, truncated = relabel_world()
    days = [*DAYS, date(2020, 1, 9)]
    a = series("A", days)
    z = [
        {**r, "close": 99.0} if r["session_date"] == date(2020, 1, 9) else r
        for r in copy_as(a, "Z")
        if r["session_date"] >= CUT
    ]
    evening = bar_available_at(date(2020, 1, 9))
    for as_of in (et_midnight(date(2020, 1, 9)) + timedelta(hours=16), evening - MICRO):
        assert impact(full, truncated, [*a, *z], as_of=as_of).num_rows == 0, as_of
    day = date(2020, 1, 9)
    assert facts(impact(full, truncated, [*a, *z], as_of=evening)) == [
        ("leak", "price", "A@2020-01-02", "A", "A:2020-01-02", None, day, day, 1),
        # H1: the truncated side reads the cell through Z, named too
        ("leak", "price", "A@2020-01-02", "Z", None, "Z:2020-01-06", day, day, 1),
    ]


def test_a_late_filed_bar_of_an_earlier_session_is_not_read_before_it_was_filed() -> None:
    # Leakage. E's 01-07 bar is on disk but knowable only from LATE (a late import): at AS_OF
    # the full-only E shows two days, not three.
    full = [seg_row("S1", "A", D0), seg_row("S2", "E", CUT)]
    truncated = [seg_row("S1", "A", D0)]
    e = [bar("E", d, 40.0, known=LATE if d == date(2020, 1, 7) else None) for d in AFTER]
    bars = [*series("A", DAYS), *e]

    [early] = facts(impact(full, truncated, bars))
    assert early[6:] == (AFTER[0], AFTER[-1], 2)
    # the cells a reader at AS_OF has: not the late-filed day, although it lies inside the span
    assert impact(full, truncated, bars).column("sessions").to_pylist() == [
        [date(2020, 1, 6), date(2020, 1, 8)]
    ]
    [late] = facts(impact(full, truncated, bars, as_of=LATE))
    assert late[6:] == (AFTER[0], AFTER[-1], 3)


def test_a_split_not_yet_knowable_is_not_read() -> None:
    # Leakage. A's 2:1 split on 01-07 is on disk, filed at LATE: before that the cells are equal.
    full, truncated = relabel_world()
    a = series("A", DAYS)
    z = [r for r in copy_as(a, "Z") if r["session_date"] >= CUT]
    day = date(2020, 1, 7)
    splits = splits_table([("A", day, 2.0, LATE)])

    assert impact(full, truncated, [*a, *z], splits, as_of=LATE - MICRO).num_rows == 0
    assert facts(impact(full, truncated, [*a, *z], splits, as_of=LATE)) == [
        ("leak", "price", "A@2020-01-02", "A", "A:2020-01-02", None, day, day, 1),
        # H1: the truncated side reads the cell through Z, named too
        ("leak", "price", "A@2020-01-02", "Z", None, "Z:2020-01-06", day, day, 1),
    ]


def test_segments_not_yet_knowable_are_masked_on_both_sides() -> None:
    # Leakage. The full master has E and the truncated one F, each knowable only at LATE; their
    # bars are visible long before. Read early, E would be a leak and F a delay.
    full = [seg_row("S1", "A", D0), seg_row("S2", "E", CUT, available_at=LATE)]
    truncated = [seg_row("S1", "A", D0), seg_row("S3", "F", CUT, available_at=LATE)]
    bars = [*series("A", DAYS), *series("E", AFTER, 40.0), *series("F", AFTER, 50.0)]

    assert impact(full, truncated, bars, as_of=LATE - MICRO).num_rows == 0
    assert facts(impact(full, truncated, bars, as_of=LATE)) == [
        ("delay", "price", "F@2020-01-06", "F", None, "F:2020-01-06", AFTER[0], AFTER[-1], 3),
        ("leak", "price", "E@2020-01-06", "E", "E:2020-01-06", None, AFTER[0], AFTER[-1], 3),
    ]


def test_an_end_known_only_later_is_masked() -> None:
    # Leakage. BK-like: the full master ends A at CUT, an end knowable only at LATE. Before LATE
    # a reader sees A open, as the truncated build has it; from LATE A's days after CUT are a
    # delay (the full master withdrew them).
    full = [seg_row("S1", "A", D0, CUT, end_available_at=LATE)]
    truncated = [seg_row("S1", "A", D0)]
    bars = series("A", DAYS)

    assert impact(full, truncated, bars, as_of=LATE - MICRO).num_rows == 0
    assert facts(impact(full, truncated, bars, as_of=LATE)) == [
        ("delay", "price", "A@2020-01-02", "A", None, "A:2020-01-02", AFTER[0], AFTER[-1], 3)
    ]


@pytest.mark.parametrize(
    ("trunc_cik", "before"),
    [(None, None), (2222, "delay")],
    ids=["truncated_none", "truncated_other"],
)
def test_a_cik_the_full_master_still_withholds_counts_as_none(
    trunc_cik: int | None, before: str | None
) -> None:
    # Leakage. The full master's CIK 1111 for A is knowable only at LATE. Before that it reads as
    # no CIK: never a leak; at most a delay against a CIK the build then had.
    full = [seg_row("S1", "A", D0, cik=1111, cik_available_at=LATE)]
    truncated = [seg_row("S1", "A", D0, cik=trunc_cik)]
    bars = series("A", DAYS)
    row = ("cik", None, "A", "A:2020-01-02", "A:2020-01-02", DAYS[0], DAYS[-1], 5)

    early = facts(impact(full, truncated, bars, as_of=LATE - MICRO))
    assert early == ([] if before is None else [(before, *row)])
    assert facts(impact(full, truncated, bars, as_of=LATE)) == [("leak", *row)]


# =============================================================================================
# run_instants: one truncated build per instant, its diff and its impact
# =============================================================================================
# The real run loaded the store once per instant and built serially; the instants now run in a
# fork-context process pool whose workers inherit the inputs. The diff and the impact of an
# instant come from the same truncated build (one build per instant, not one per command), and
# whatever the number of workers the results must be exactly those of the serial run.

NO_SPLITS = splits_table()


def hand_loop(
    full: pa.Table,
    inputs: Any,
    overrides: Sequence[Mapping[str, Any]],
    instants: Sequence[datetime],
    splits: pa.Table = NO_SPLITS,
    liquid: Any = None,
) -> list[tuple[datetime, pa.Table, pa.Table]]:
    """(instant, compare, impact_at) of one truncated build per distinct instant, sorted."""
    out = []
    for t in sorted({t.astimezone(UTC) for t in instants}):
        truncated = sm.build_from_inputs(inputs, overrides, as_of=t)[0]
        out.append(
            (t, compare(full, truncated, t),
             impact_at(full, truncated, inputs.bars, splits, t, liquid=liquid))
        )  # fmt: skip
    return out


def same_results(got: Sequence[InstantResult], expected: Sequence[Any]) -> None:
    """``expected``: InstantResults or (instant, diff, impact) triples."""
    triples = [(e.as_of, e.diff, e.impact) if isinstance(e, InstantResult) else e for e in expected]
    assert [r.as_of for r in got] == [t for t, _, _ in triples]
    for r, (t, diff, rows) in zip(got, triples, strict=True):
        assert isinstance(r, InstantResult), t
        assert r.diff.schema.equals(DIFF_SCHEMA), t
        assert r.impact.schema.equals(IMPACT_SCHEMA), t
        assert r.diff.equals(diff), t
        assert r.impact.equals(rows), t
        check_sessions(r.impact)


def leak_keys(diff: pa.Table) -> set[str]:
    return {diff_key(r) for r in diff.to_pylist() if r["kind"] == "leak"}


# --- normal ---


def test_the_cli_world_merge_shows_as_price_impact_before_the_decision(tmp_path: Path) -> None:
    # The fixture the run_instants and CLI tests rely on. Before the merge is decided, the full
    # master reads KRSA's bars inside CYCN's security (CYCN@2024-01-02 then holds two equal bars a
    # day, the build then one) and the build reads them as KRSA@2024-01-02, which the full master
    # does not name: a leak on CYCN's cells, a delay on KRSA's.
    _, inputs, full = cli_world(tmp_path)
    early = et_midnight(date(2024, 3, 1))
    truncated = sm.build_from_inputs(inputs, [MERGE], as_of=early)[0]
    assert set(LEAK_KEYS) <= {diff_key(r) for r in compare(full, truncated, early).to_pylist()}

    table = impact_at(full, truncated, inputs.bars, NO_SPLITS, early)
    out = facts(table)

    n = len(CLI_DAYS)
    assert ("delay", "price", "KRSA@2024-01-02", "KRSA", "KRSA:2024-01-02", "KRSA:2024-01-02",
            CLI_DAYS[0], CLI_DAYS[-1], n) in out  # fmt: skip
    leaks = [f for f in out if f[0] == "leak"]
    assert leaks
    for f in leaks:
        assert f[1:3] == ("price", "CYCN@2024-01-02")
        assert f[6:] == (CLI_DAYS[0], CLI_DAYS[-1], n)
    assert not [f for f in out if f[1] == "cik"]
    check_sessions(table)
    for r in table.to_pylist():
        assert r["sessions"] == CLI_DAYS, r
    decided = et_midnight(CLI_DECIDED)
    truncated = sm.build_from_inputs(inputs, [MERGE], as_of=decided)[0]
    assert impact_at(full, truncated, inputs.bars, NO_SPLITS, decided).num_rows == 0


def test_instant_result_is_a_frozen_dataclass_of_instant_diff_and_impact() -> None:
    result = InstantResult(
        as_of=AS_OF, diff=DIFF_SCHEMA.empty_table(), impact=IMPACT_SCHEMA.empty_table()
    )
    assert [f.name for f in dataclasses.fields(result)] == ["as_of", "diff", "impact"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.as_of = LATE  # type: ignore[misc]


def test_run_instants_serial_is_compare_and_impact_at_of_the_same_build(tmp_path: Path) -> None:
    _, inputs, full = cli_world(tmp_path)
    instants = cli_instants(full)
    assert len(instants) >= 4

    got = run_instants(full, inputs, [MERGE], NO_SPLITS, instants, jobs=1)

    same_results(got, hand_loop(full, inputs, [MERGE], instants))
    # the fixture is not a comparison of empty tables: at 2024-03-01 both carry the merge leak
    early = et_midnight(date(2024, 3, 1))
    [at_early] = [r for r in got if r.as_of == early]
    assert leak_keys(at_early.diff) == set(LEAK_KEYS)
    assert [r for r in at_early.impact.to_pylist() if r["kind"] == "leak"]


def test_run_instants_builds_each_instant_once_for_both_diff_and_impact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # one truncated build feeds compare and impact_at: two builds per instant would double the
    # run (33 s per real build) and could let the diff and the impact see different masters
    _, inputs, full = cli_world(tmp_path)
    calls: list[int] = []

    def counting(**kwargs: Any) -> Any:
        calls.append(1)
        return build_security_master(**kwargs)

    monkeypatch.setattr(sm, "build_security_master", counting)
    decided, early = et_midnight(CLI_DECIDED), et_midnight(date(2024, 3, 1))
    run_instants(full, inputs, [MERGE], NO_SPLITS, [decided, early, decided], jobs=1)
    assert len(calls) == 2


def test_run_instants_default_jobs_is_serial(tmp_path: Path) -> None:
    _, inputs, full = cli_world(tmp_path)
    instants = cli_instants(full)
    same_results(
        run_instants(full, inputs, [MERGE], NO_SPLITS, instants),
        run_instants(full, inputs, [MERGE], NO_SPLITS, instants, jobs=1),
    )


@pytest.mark.parametrize("jobs", [2, 3])
def test_run_instants_in_a_pool_equals_serial_on_the_cli_store(tmp_path: Path, jobs: int) -> None:
    _, inputs, full = cli_world(tmp_path)
    instants = cli_instants(full)
    serial = run_instants(full, inputs, [MERGE], NO_SPLITS, instants, jobs=1)
    with within(120):
        pooled = run_instants(full, inputs, [MERGE], NO_SPLITS, instants, jobs=jobs)
    same_results(pooled, serial)
    # not a comparison of empty tables: some instant has both a diff and an impact
    assert any(r.diff.num_rows and r.impact.num_rows for r in serial)
    assert no_children_left()


@pytest.mark.parametrize("scenario", ["r_gold_b", "l_arnc", "r4"])
def test_run_instants_in_a_pool_equals_serial_on_leaking_worlds(scenario: str) -> None:
    inputs = store_inputs_of(SCENARIOS[scenario]())
    full, _ = sm.build_from_inputs(inputs)
    instants = instants_of(full, SCENARIOS[scenario]())
    serial = run_instants(full, inputs, (), NO_SPLITS, instants, jobs=1)
    with within(120):
        pooled = run_instants(full, inputs, (), NO_SPLITS, instants, jobs=2)
    same_results(pooled, serial)
    assert any(leak_keys(r.diff) for r in serial), f"{scenario}: no leak rows; pick a leaking world"


@pytest.mark.parametrize("jobs", [1, 2])
def test_run_instants_passes_liquid_and_splits_to_every_instant(tmp_path: Path, jobs: int) -> None:
    _, inputs, full = cli_world(tmp_path)
    [merged] = {r["security_id"] for r in full.to_pylist() if r["symbol"] == "CYCN"}
    instants = [et_midnight(date(2024, 3, 1)), et_midnight(CLI_DECIDED)]
    # a 2:1 split on CYCN only, knowable before the first instant: a different price history
    # for CYCN's bars than for KRSA's identical ones (impact_at reads the splits it is given)
    splits = splits_table([("CYCN", CLI_DAYS[3], 2.0, None)])
    with within(120):
        got = run_instants(
            full, inputs, [MERGE], splits, instants, liquid={merged}, jobs=jobs
        )  # fmt: skip
    same_results(got, hand_loop(full, inputs, [MERGE], instants, splits, liquid={merged}))
    # KRSA@2024-01-02 is no name of the liquid security: its delay is filtered out
    assert all(r["security"] == "CYCN@2024-01-02" for res in got for r in res.impact.to_pylist())
    assert got[0].impact.num_rows > 0
    # the diff is the master comparison alone: liquid and splits do not filter it
    assert got[0].diff.equals(hand_loop(full, inputs, [MERGE], instants[:1])[0][1])


@pytest.mark.parametrize("jobs", [1, 2])
def test_run_instants_progress_sees_each_result_in_instant_order(tmp_path: Path, jobs: int) -> None:
    _, inputs, full = cli_world(tmp_path)
    instants = cli_instants(full)
    seen: list[InstantResult] = []
    with within(120):
        got = run_instants(
            full, inputs, [MERGE], NO_SPLITS, list(reversed(instants)), jobs=jobs,
            progress=seen.append,
        )  # fmt: skip
    assert [r.as_of for r in seen] == sorted(instants)
    same_results(seen, got)


# --- edges ---


def test_run_instants_sorts_and_deduplicates_the_instants(tmp_path: Path) -> None:
    _, inputs, full = cli_world(tmp_path)
    decided, early = et_midnight(CLI_DECIDED), et_midnight(date(2024, 3, 1))
    # the same instant twice (New York and UTC), out of order
    given = [decided, early, decided.astimezone(UTC), early - MICRO]
    expected = sorted({t.astimezone(UTC) for t in given})
    assert len(expected) == 3

    for jobs in (1, 2):
        with within(120):
            got = run_instants(full, inputs, [MERGE], NO_SPLITS, given, jobs=jobs)
        assert [r.as_of for r in got] == expected, jobs
        assert all(r.as_of.utcoffset() == timedelta(0) for r in got), jobs
        same_results(got, hand_loop(full, inputs, [MERGE], given))


@pytest.mark.parametrize("jobs", [1, 2])
def test_run_instants_of_no_instants_is_empty(tmp_path: Path, jobs: int) -> None:
    _, inputs, full = cli_world(tmp_path)
    with within(60):
        assert run_instants(full, inputs, [MERGE], NO_SPLITS, [], jobs=jobs) == []


def test_run_instants_with_more_jobs_than_instants(tmp_path: Path) -> None:
    _, inputs, full = cli_world(tmp_path)
    instants = [et_midnight(date(2024, 3, 1)), et_midnight(CLI_DECIDED)]
    serial = run_instants(full, inputs, [MERGE], NO_SPLITS, instants, jobs=1)
    with within(120):
        same_results(run_instants(full, inputs, [MERGE], NO_SPLITS, instants, jobs=8), serial)
        same_results(
            run_instants(full, inputs, [MERGE], NO_SPLITS, instants[:1], jobs=8), serial[:1]
        )
    assert no_children_left()


@pytest.mark.parametrize("jobs", [0, -1])
def test_run_instants_refuses_fewer_than_one_job(tmp_path: Path, jobs: int) -> None:
    _, inputs, full = cli_world(tmp_path)
    with pytest.raises(ValueError):
        run_instants(full, inputs, [MERGE], NO_SPLITS, [et_midnight(CLI_DECIDED)], jobs=jobs)
    with pytest.raises(ValueError):  # refused even with nothing to run
        run_instants(full, inputs, [MERGE], NO_SPLITS, [], jobs=jobs)


@pytest.mark.parametrize("jobs", [1, 2])
def test_run_instants_refuses_a_naive_instant(tmp_path: Path, jobs: int) -> None:
    _, inputs, full = cli_world(tmp_path)
    with within(60), pytest.raises(ValueError):
        run_instants(full, inputs, [MERGE], NO_SPLITS, [datetime(2024, 3, 1)], jobs=jobs)
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
        run_instants(full, inputs, [MERGE], NO_SPLITS, instants, jobs=jobs)
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
        run_instants(full, inputs, [MERGE], NO_SPLITS, instants, jobs=2)
    assert no_children_left()


# --- leakage ---


@pytest.mark.parametrize("jobs", [1, 2])
def test_run_instants_each_worker_truncates_to_its_own_instant(tmp_path: Path, jobs: int) -> None:
    # Leakage. The merge decided on 2024-06-03 is in the overrides every worker inherits; a
    # worker must still build the master of one microsecond before without it (the KRSA leak in
    # the diff, CYCN's cells in the impact), and only the 06-03 build may use it (neither).
    _, inputs, full = cli_world(tmp_path)
    decided = et_midnight(CLI_DECIDED)
    with within(120):
        early, late = run_instants(
            full, inputs, [MERGE], NO_SPLITS, [decided, decided - MICRO], jobs=jobs
        )
    assert (early.as_of, late.as_of) == (decided - MICRO, decided)
    assert leak_keys(early.diff) == set(LEAK_KEYS)
    assert [r for r in early.impact.to_pylist() if r["kind"] == "leak"]
    assert late.diff.num_rows == 0
    assert late.impact.num_rows == 0


@pytest.mark.parametrize("jobs", [1, 2])
def test_run_instants_impact_never_reads_bars_later_than_its_instant(
    tmp_path: Path, jobs: int
) -> None:
    # Leakage. Every worker inherits all bars (through 2024-06-28); at 2024-01-05 00:00 New York
    # only the 01-02, 01-03 and 01-04 sessions are knowable (01-04's bar from 20:00 that
    # evening; measured on this store: the leak rows cover 3 sessions there, 31 at 2024-03-01).
    # The impact's sessions must stop before 01-05.
    _, inputs, full = cli_world(tmp_path)
    t = et_midnight(date(2024, 1, 5))
    with within(120):
        [result] = run_instants(full, inputs, [MERGE], NO_SPLITS, [t], jobs=jobs)
    sessions = {d for r in result.impact.to_pylist() for d in r["sessions"]}
    assert sessions
    assert max(sessions) < date(2024, 1, 5)


# =============================================================================================
# the CYCN / KRSA store with a liquid tier (the `asof market truncation` tests in
# tests/ingest/test_truncation_gate.py run on it)
# =============================================================================================


def write_liquid_store(root: Path) -> Path:
    """The CYCN / KRSA world of test_master_truncation.write_cli_store with ~10M shares a day
    at a vwap of $7 ($70M a symbol, $140M for the merged security: above the $50M liquid
    threshold, which reads vwap * volume), AAA illiquid ($450k), plus what tier_union
    reads: the master built with the merge, a class row per segment ("common"), and an empty
    split file."""
    cycn = [
        daily("CYCN", d, open=6.9, high=7.1, low=6.8, close=7.0 + i / 1000,
              volume=10_000_000 + i, vwap=7.0)
        for i, d in enumerate(CLI_DAYS)
    ]  # fmt: skip
    krsa = [
        daily(
            "KRSA",
            d,
            open=b.open,
            high=b.high,
            low=b.low,
            close=b.close,
            volume=b.volume,
            vwap=b.vwap,
        )
        for d, b in zip(CLI_DAYS, cycn, strict=True)
    ]
    aaa = [
        daily("AAA", d, open=49.5, high=50.5, low=49.0, close=50.0 + i / 100, volume=9_000 + i,
              vwap=50.0)
        for i, d in enumerate(business_days(date(2024, 1, 2), date(2024, 6, 28)))
    ]  # fmt: skip
    store_daily(root, [*cycn, *krsa, *aaa])
    sec = history_table(
        [(s, t, c) for s in CLI_SNAPSHOTS for t, c in (("CYCN", 1004), ("KRSA", 1005),
                                                       ("AAA", 1006))]
    )  # fmt: skip
    (root / "symbols").mkdir(parents=True, exist_ok=True)
    pq.write_table(sec, root / "symbols" / "sec_history.parquet")
    master, conflicts = sm.build_from_store(root, [MERGE])
    sm.write_master(master, conflicts, root)
    classes = [
        {"security_id": r["security_id"], "symbol": r["symbol"], "valid_from": r["valid_from"],
         "valid_to": r["valid_to"], "name": f"{r['symbol']} Inc", "name_source": "test",
         "class": "common", "rule": "test"}
        for r in master.to_pylist()
    ]  # fmt: skip
    (root / CLASS_FILE).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(classes, schema=CLASS_SCHEMA), root / CLASS_FILE)
    (root / "corporate_actions").mkdir(parents=True, exist_ok=True)
    pq.write_table(splits_table(), root / "corporate_actions" / "splits.parquet")
    return root


def cycn_id(tmp_path: Path) -> str:
    master = pq.read_table(market_root(tmp_path) / sm.MASTER_FILE)
    [found] = {r["security_id"] for r in master.to_pylist() if r["symbol"] == "CYCN"}
    return found


# =============================================================================================
# standard case on the real store
# =============================================================================================

REAL_ROOT = market_root(default_data_dir())
REPO = Path(__file__).resolve().parents[2]


real_store = pytest.mark.skipif(
    not (REAL_ROOT / "symbols" / "sec_history.parquet").exists()
    or not (REAL_ROOT / "corporate_actions" / "splits.parquet").exists(),
    reason=f"no real store under {REAL_ROOT}",
)


class RealStore:
    """The real inputs, overrides, splits and full master, loaded once per module; truncated
    builds once per instant."""

    def __init__(self) -> None:
        self.inputs = sm.load_store_inputs(REAL_ROOT)
        self.overrides = sm.load_overrides(REPO / "configs" / "security_overrides.yaml")
        self.full, _ = sm.build_from_inputs(self.inputs, self.overrides)
        self.splits = pq.read_table(REAL_ROOT / "corporate_actions" / "splits.parquet")
        self._truncated: dict[datetime, pa.Table] = {}

    def truncated(self, t: datetime) -> pa.Table:
        key = t.astimezone(UTC)
        if key not in self._truncated:
            self._truncated[key] = sm.build_from_inputs(self.inputs, self.overrides, as_of=t)[0]
        return self._truncated[key]

    def impact(self, t: datetime, **kwargs: Any) -> pa.Table:
        out = impact_at(self.full, self.truncated(t), self.inputs.bars, self.splits, t, **kwargs)
        check_sessions(out)
        return out


@pytest.fixture(scope="module")
def real() -> RealStore:
    return RealStore()


@pytest.mark.realdata
@real_store
def test_real_store_cohr_iivi_relabelling_is_no_leak_at_2022_09_04(real: RealStore) -> None:
    # Measured 2026-10-20. II-VI renamed itself Coherent and took the ticker COHR on 2022-08-01
    # (CIK 820318). The full master: IIVI@2020-01-02 = IIVI until 2022-08-01, then COHR from
    # 2022-08-01 (CIK 820318, knowable 2022-08-02). The build at 2022-09-04 00:00 New York reads
    # the same Aug bars through IIVI under IIVI@2020-01-02, and COHR's bars under a security it
    # names COHR@2020-01-02 (CIK 820318), a name the full master does not have. So: IIVI's cells
    # are equal on both sides (no price row on them in Aug), COHR@2020-01-02's Aug cells are a
    # delay, and no CIK differs on COHR or IIVI.
    t = datetime(2022, 9, 4, tzinfo=ET)

    rows = real.impact(t).to_pylist()

    aug_start, aug_end = date(2022, 8, 1), date(2022, 9, 2)  # 09-02: the last session before t
    iivi_aug = [
        r for r in rows
        if r["what"] == "price" and r["security"] == "IIVI@2020-01-02"
        and r["first_day"] <= aug_end and r["last_day"] >= aug_start
    ]  # fmt: skip
    assert iivi_aug == []
    assert [r for r in rows if r["what"] == "cik" and r["kind"] == "leak"
            and r["symbol"] in ("COHR", "IIVI")] == []  # fmt: skip
    cohr = [
        r for r in rows
        if (r["kind"], r["what"], r["security"], r["symbol"], r["full_segment"])
        == ("delay", "price", "COHR@2020-01-02", "COHR", "COHR:2022-08-01")
    ]  # fmt: skip
    assert cohr
    assert min(r["first_day"] for r in cohr) == aug_start
    assert max(r["last_day"] for r in cohr) == aug_end


@pytest.mark.realdata
@real_store
def test_real_store_exhaustive_equals_default_at_2022_09_04(real: RealStore) -> None:
    # The candidate cells miss nothing on real data around the COHR / IIVI relabelling and a few
    # other histories (B: Barnes then Barrick; YPF; AAPL as a plain control). liquid keeps the
    # exhaustive pass small: only these securities' symbols are joined.
    t = datetime(2022, 9, 4, tzinfo=ET)
    names = {"COHR", "IIVI", "B", "YPF", "AAPL"}
    liquid = {r["security_id"] for r in real.full.to_pylist() if r["symbol"] in names}
    assert liquid

    default = real.impact(t, liquid=liquid)
    every = real.impact(t, liquid=liquid, exhaustive=True)

    assert every.schema.equals(IMPACT_SCHEMA)
    assert every.equals(default), (facts(every), facts(default))


@pytest.mark.realdata
@real_store
def test_real_store_barnes_history_shows_barricks_cik_and_bars_before_2025_07_25(
    real: RealStore,
) -> None:
    # Measured 2026-10-20 on the real run at this instant (one microsecond before 2025-07-25
    # 00:00 New York). The ticker B belonged to Barnes Group (CIK 9984) until its take-private
    # (last session 2025-05-08); Barrick took B from 2025-05-09 (CIK 756894). The full master
    # (as built today) puts Barrick's CIK on Barnes's 2020-2025 segment B:2020-01-02 and
    # continues Barnes's security B@2020-01-02 with Barrick's B segment from 2025-05-09; a build
    # then could know neither. Expected among B's rows:
    # - CIK leak on B:2020-01-02 over 2020-01-02..2025-05-08, 1345 sessions, full 756894 vs
    #   truncated 9984
    # - price leak on B@2020-01-02 read through B:2025-05-09 over 2025-05-09..2025-07-24, 52
    #   sessions
    # With H1 more rows for B may appear (the truncated side's bars of those cells): these two
    # must be present, not the only ones.
    t = datetime(2025, 7, 25, 3, 59, 59, 999999, tzinfo=UTC)

    rows = [r for r in real.impact(t).to_pylist() if r["symbol"] == "B"]

    got = facts(pa.Table.from_pylist(rows, schema=IMPACT_SCHEMA))
    cik: Fact = ("leak", "cik", None, "B", "B:2020-01-02", "B:2020-01-02",
                 date(2020, 1, 2), date(2025, 5, 8), 1345)  # fmt: skip
    price: Fact = ("leak", "price", "B@2020-01-02", "B", "B:2025-05-09", "B:2025-05-09",
                   date(2025, 5, 9), date(2025, 7, 24), 52)  # fmt: skip
    assert cik in got, got
    assert price in got, got
    [detail] = [r["detail"] for r, f in zip(rows, got, strict=True) if f == cik]
    assert "756894" in detail and "9984" in detail, detail
