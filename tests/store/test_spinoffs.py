"""Guards against a spin-off rescaling a parent's history wrongly or too early: a factor taken
from a child price nobody could see yet (look-ahead), a factor applied when the child never
traded or traded months later (an invented continuity), several children of one day treated as
several events, a stored duplicate applied twice, and a decision that changes after it was made.

Target API (asof.store.spinoffs, new module):

    visible_spinoffs(con, root, as_of, symbols=None, *, rejected=None) -> pa.Table
        spin_off rows of <root>/corporate_actions/others.parquet with available_at <= as_of;
        columns at least parent_symbol (= source_symbol), child_symbol (= new_symbol), ex_date,
        ratio (= rate), available_at. ``symbols`` filters on the parent. Rows equal in (parent,
        child, ex_date, ratio) come back once, the earliest knowable copy. Every row of a
        rejected (parent, ex_date) is left out, whatever the as_of. Ordered by parent, ex_date,
        child. No file -> empty table. Naive as_of -> ValueError.
    spinoff_events(con, root, as_of, parents=None, *, rejected=None, max_child_delay=20)
        -> pa.Table, one row per (parent security, ex_date), columns EVENT_COLUMNS below.
        Parent security: the security of the last visible bar before ex_date carrying the
        parent symbol. Parent sessions: that security's visible sessions on/after ex_date, the
        first being number 0. Child price: close of the child symbol's first visible bar on/after
        ex_date. child_session: the latest such first bar over the event's children.
        factor = (prev_close - sum(child_close * ratio)) / prev_close.
        total_return = (parent close on its first session on/after ex_date
                        + sum(child_close * ratio)) / prev_close - 1.
        status / reason / known_on, in order of precedence:
          parent == a child symbol        restart  same_symbol          ex_date
          no parent bar before ex_date    restart  no_parent_price      ex_date
          all children priced, child_session within max_child_delay parent sessions:
              0 < factor < 1              adjust   None                 child_session
              otherwise                   restart  factor_out_of_range  child_session
          parent session number max_child_delay visible, not all children priced in time:
              a child priced later        restart  child_too_late       that session
              else                        restart  no_child_price       that session
          otherwise (decision not possible at as_of)
                                          pending  None                 None (factor None)
        factor / total_return are null unless adjust.

Fake data provenance:
- GE -> GEV ex 2024-04-02, ratio 0.25: GE close 2024-04-01 175.36, 2024-04-02 136.47; GEV's
  first bar 2024-04-02 close 140.00 (real, bar store probe 2026-10-27) -> factor 0.8004.
- OPEN -> OPENW / OPENZ / OPENL ex 2025-11-18, ratio 0.033333333 each; OPEN close 11-17 7.84,
  11-18 7.52; the three children's first bars 2025-11-24 (4 parent sessions later) at 1.82 /
  0.6506 / 1.08 (real). OPEN -> OPENW is stored twice, the copies differing in id and
  process_date (real raw_json; same for XRX -> XRXDW 2026-02-09, QNTM -> 747CVR033 2025-10-27).
- LEN -> MRP ex 2025-01-21: MRP's first bar 2025-02-07, 13 parent sessions later (real).
- MPTI -> MPTI.WS ex 2025-03-10: first child bar 36 sessions later (real).
- NVS -> SDZNY ex 2023-10-04: no SDZNY bar at all (real; an OTC ADR).
- CR -> CR ex 2023-04-04: parent and child carry one symbol (real; 13 such records).
- 17 of 237 real records name a parent with no bar at all; 5 have a parent first traded on or
  after the ex-date (store probe 2026-10-27) -> no_parent_price.
- available_at of a record is 00:00 New York of the ex-date (ingest rule); daily bars become
  visible at 20:00 New York of their session (ingest rule; audit P6 gates every exception).
- Prices other than the GE / OPEN numbers are constructed round numbers.
- ASSUMPTION: the delay is counted in the parent's own sessions (the spec's words); a halted
  parent therefore gives a lagging child more calendar time. Not observed on real data.

Real-data checks: tests/store/test_golden_spinoffs.py pins GE / OPEN / LEN / MPTI / NVS / CR
on the real store; the audit's spin-off check reports large total returns of adjust events.
"""

import json
from collections import defaultdict
from collections.abc import Iterator, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from asof.ingest.bars import bucket_of
from asof.ingest.corporate_actions import OTHER_SCHEMA
from asof.store.market import market_root
from asof.store.spinoffs import spinoff_events, visible_spinoffs
from tests.store.audit_world import bar as raw_bar
from tests.store.test_market import Seg, avail, known, write_master
from tests.store.test_universe import store_raw, trading_days

ET = ZoneInfo("America/New_York")
LATE = datetime(2026, 10, 27, tzinfo=UTC)  # every fixture fact is knowable
DAYS = trading_days(date(2024, 1, 2), date(2024, 6, 28))
EX_I = 40
EX = DAYS[EX_I]  # 2024-02-27
EVENT_COLUMNS = {
    "security_id",
    "parent_symbol",
    "child_symbols",
    "ex_date",
    "ratio",
    "parent_prev_session",
    "parent_prev_close",
    "child_session",
    "child_closes",
    "factor",
    "total_return",
    "status",
    "reason",
    "known_on",
}
SPINOFF_COLUMNS = {"parent_symbol", "child_symbol", "ex_date", "ratio", "available_at"}


# =============================================================================================
# builders (shared with tests/store/test_spinoff_rejections.py, the audit and the feature tests)
# =============================================================================================


def spin_row(
    parent: str,
    child: str,
    ex_date: date,
    ratio: float | None = 0.5,
    *,
    available_at: datetime | None = None,
    process_date: date | None = None,
    uid: str = "a",
) -> dict[str, Any]:
    """One stored spin_off row of others.parquet as the ingest writes it: symbol NULL, the
    parent in source_symbol, the child in new_symbol, rate = new_rate / source_rate."""
    processed = process_date or ex_date
    raw = {
        "ex_date": ex_date.isoformat(),
        "id": f"{parent}-{child}-{ex_date.isoformat()}-{uid}",
        "new_rate": ratio,
        "new_symbol": child,
        "process_date": processed.isoformat(),
        "source_rate": 1,
        "source_symbol": parent,
        "_kind": "spin_offs",
    }
    return {
        "kind": "spin_off",
        "symbol": None,
        "source_symbol": parent,
        "new_symbol": child,
        "ex_date": ex_date,
        "process_date": processed,
        "rate": ratio,
        "raw_json": json.dumps(raw),
        "available_at": known(ex_date) if available_at is None else available_at,
    }


def stock_dividend_row(symbol: str, ex_date: date, rate: float) -> dict[str, Any]:
    """Another kind stored in the same file (real: ACSAY 2020-07-02 rate 1.0625)."""
    return {
        "kind": "stock_dividend",
        "symbol": symbol,
        "source_symbol": None,
        "new_symbol": None,
        "ex_date": ex_date,
        "process_date": ex_date,
        "rate": rate,
        "raw_json": "{}",
        "available_at": known(ex_date),
    }


def write_other_rows(root: Path, rows: Sequence[dict[str, Any]]) -> Path:
    """Write ``rows`` as the store's others.parquet (replacing any earlier file)."""
    path = root / "corporate_actions" / "others.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(list(rows), schema=OTHER_SCHEMA), path)
    return path


def px(
    symbol: str, day: date, close: float, *, available_at: datetime | None = None
) -> dict[str, Any]:
    """A raw daily bar with open = high = low = close (flat, so a test reads one number)."""
    return raw_bar(
        symbol,
        day,
        close=close,
        volume=1_000,
        open=close,
        high=close,
        low=close,
        vwap=close,
        available_at=available_at,
    )


def flat(symbol: str, days: Sequence[date], close: float) -> list[dict[str, Any]]:
    return [px(symbol, d, close) for d in days]


def write_bars(root: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Every bar at once, one file per (year, bucket) as the backfill files them."""
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["session_date"].year, bucket_of(row["symbol"]))].append(row)
    for (year, bucket), group in grouped.items():
        store_raw(root, group, year, bucket)


Child = tuple[
    str, int | None, float, float
]  # symbol, first-bar offset from EX (None: never), close, ratio


def build(
    root: Path,
    *,
    children: Sequence[Child] = (("KID", 0, 40.0, 0.5),),
    prev: float = 100.0,
    after: float = 80.0,
    parent: str = "PAR",
    days: Sequence[date] = DAYS,
    ex_i: int = EX_I,
    records: Sequence[dict[str, Any]] | None = None,
    parent_bars: Sequence[dict[str, Any]] | None = None,
    extra_bars: Sequence[dict[str, Any]] = (),
    segs: Sequence[Seg] | None = None,
) -> None:
    """A store with one parent trading at ``prev`` before ``days[ex_i]`` and ``after`` from it,
    each child trading flat from ``days[ex_i + offset]`` on, a master owning every symbol and
    one spin-off record per child on the ex-date."""
    ex = days[ex_i]
    bars = (
        list(parent_bars)
        if parent_bars is not None
        else [px(parent, d, prev if d < ex else after) for d in days]
    )
    master = list(segs) if segs is not None else [Seg("S_" + parent, parent)]
    for symbol, offset, close, _ in children:
        if offset is None or symbol == parent:
            continue
        first = days[ex_i + offset]
        bars += flat(symbol, [d for d in days if d >= first], close)
        master.append(Seg("S_" + symbol, symbol, valid_from=first, available_at=known(first)))
    write_bars(root, [*bars, *extra_bars])
    write_master(root, master)
    if records is None:
        records = [spin_row(parent, symbol, ex, ratio) for symbol, _, _, ratio in children]
    write_other_rows(root, records)


def events(
    con: duckdb.DuckDBPyConnection, root: Path, as_of: datetime = LATE, **kwargs: Any
) -> list[dict[str, Any]]:
    kwargs.setdefault("rejected", ())
    table = spinoff_events(con, root, as_of, **kwargs)
    return sorted(table.to_pylist(), key=lambda r: (r["parent_symbol"], r["ex_date"]))


def one(
    con: duckdb.DuckDBPyConnection, root: Path, as_of: datetime = LATE, **kwargs: Any
) -> dict[str, Any]:
    [row] = events(con, root, as_of, **kwargs)
    return row


def decision(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row["status"], row["reason"], row["factor"], row["known_on"])


def ny(day: date, hour: int) -> datetime:
    return datetime(day.year, day.month, day.day, hour, tzinfo=ET)


@pytest.fixture
def con() -> Iterator[duckdb.DuckDBPyConnection]:
    connection = duckdb.connect()
    yield connection
    connection.close()


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return market_root(tmp_path)


# =============================================================================================
# visible_spinoffs: normal path
# =============================================================================================


def test_visible_spinoffs_reads_spin_off_rows_only(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    write_other_rows(
        root,
        [
            spin_row("GE", "GEV", date(2024, 4, 2), 0.25),
            stock_dividend_row("ACSAY", date(2020, 7, 2), 1.0625),
        ],
    )

    table = visible_spinoffs(con, root, LATE, rejected=())

    assert SPINOFF_COLUMNS <= set(table.schema.names)
    [row] = table.to_pylist()
    assert (row["parent_symbol"], row["child_symbol"], row["ex_date"], row["ratio"]) == (
        "GE",
        "GEV",
        date(2024, 4, 2),
        0.25,
    )
    assert row["available_at"] == known(date(2024, 4, 2))


def test_visible_spinoffs_are_ordered_by_parent_ex_date_child(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    day = date(2025, 11, 18)
    write_other_rows(
        root,
        [
            spin_row("OPEN", "OPENZ", day, 0.033333333),
            spin_row("GE", "GEV", date(2024, 4, 2), 0.25),
            spin_row("OPEN", "OPENL", day, 0.033333333),
            spin_row("GE", "GEHC", date(2023, 1, 4), 0.33333),
            spin_row("OPEN", "OPENW", day, 0.033333333),
        ],
    )

    table = visible_spinoffs(con, root, LATE, rejected=())

    got = [(r["parent_symbol"], r["ex_date"], r["child_symbol"]) for r in table.to_pylist()]
    assert got == [
        ("GE", date(2023, 1, 4), "GEHC"),
        ("GE", date(2024, 4, 2), "GEV"),
        ("OPEN", day, "OPENL"),
        ("OPEN", day, "OPENW"),
        ("OPEN", day, "OPENZ"),
    ]


def test_open_to_openw_stored_twice_comes_back_once(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # real: the two copies differ in id and process_date (11-18 vs 11-21), nothing else
    day = date(2025, 11, 18)
    first = spin_row("OPEN", "OPENW", day, 0.033333333, uid="5bf3")
    second = spin_row(
        "OPEN", "OPENW", day, 0.033333333, uid="dbc4", process_date=date(2025, 11, 21)
    )
    assert first["raw_json"] != second["raw_json"]
    write_other_rows(root, [first, second, spin_row("OPEN", "OPENZ", day, 0.033333333)])

    table = visible_spinoffs(con, root, LATE, rejected=())

    assert table.column("child_symbol").to_pylist() == ["OPENW", "OPENZ"]


def test_visible_spinoffs_symbols_filter_on_the_parent(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    write_other_rows(
        root,
        [spin_row("GE", "GEV", date(2024, 4, 2), 0.25), spin_row("LEN", "MRP", date(2025, 1, 21))],
    )

    assert visible_spinoffs(con, root, LATE, ["GE"], rejected=()).num_rows == 1
    assert visible_spinoffs(con, root, LATE, ["GEV"], rejected=()).num_rows == 0  # a child
    assert visible_spinoffs(con, root, LATE, ["GE", "LEN"], rejected=()).num_rows == 2


# =============================================================================================
# visible_spinoffs: boundaries
# =============================================================================================


def test_same_pair_with_another_ratio_is_another_row(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    day = date(2024, 4, 2)
    write_other_rows(root, [spin_row("GE", "GEV", day, 0.25), spin_row("GE", "GEV", day, 0.5)])

    table = visible_spinoffs(con, root, LATE, rejected=())

    assert sorted(table.column("ratio").to_pylist()) == [0.25, 0.5]


def test_no_others_file_is_an_empty_table(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    table = visible_spinoffs(con, root, LATE, rejected=())
    assert table.num_rows == 0
    assert SPINOFF_COLUMNS <= set(table.schema.names)


def test_naive_as_of_is_refused(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    write_other_rows(root, [spin_row("GE", "GEV", date(2024, 4, 2), 0.25)])
    with pytest.raises(ValueError):
        visible_spinoffs(con, root, datetime(2026, 10, 27), rejected=())


# =============================================================================================
# visible_spinoffs: leakage
# =============================================================================================


def test_record_is_invisible_before_its_available_at(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    day = date(2024, 4, 2)
    write_other_rows(root, [spin_row("GE", "GEV", day, 0.25)])

    assert (
        visible_spinoffs(con, root, known(day) - timedelta(microseconds=1), rejected=()).num_rows
        == 0
    )
    assert visible_spinoffs(con, root, known(day), rejected=()).num_rows == 1


def test_a_copy_known_later_neither_appears_early_nor_replaces_the_first(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    day = date(2025, 11, 18)
    t1 = known(day)
    t2 = t1 + timedelta(days=3)  # a re-ingested copy (constructed; real copies share t1)
    early = spin_row("OPEN", "OPENW", day, 0.033333333, uid="early")
    late = spin_row("OPEN", "OPENW", day, 0.033333333, uid="late", available_at=t2)
    write_other_rows(root, [late, early])

    between = visible_spinoffs(con, root, t1 + timedelta(days=1), rejected=())
    after = visible_spinoffs(con, root, t2 + timedelta(days=1), rejected=())

    assert between.column("available_at").to_pylist() == [t1]
    assert after.column("available_at").to_pylist() == [t1]


@pytest.mark.parametrize("hours", [-1, 0, 30, 80, 200])
def test_no_record_with_available_at_after_as_of_is_returned(
    con: duckdb.DuckDBPyConnection, root: Path, hours: int
) -> None:
    base = known(date(2024, 4, 1))
    rows = [
        spin_row(
            "P" + str(i),
            "C" + str(i),
            date(2024, 4, 1) + timedelta(days=i),
            0.5,
            available_at=base + timedelta(hours=24 * i),
        )
        for i in range(6)
    ]
    write_other_rows(root, rows)
    as_of = base + timedelta(hours=hours)

    got = visible_spinoffs(con, root, as_of, rejected=()).to_pylist()

    assert all(r["available_at"] <= as_of for r in got)
    assert len(got) == sum(1 for r in rows if r["available_at"] <= as_of)


# =============================================================================================
# spinoff_events: normal path
# =============================================================================================


def test_single_child_priced_on_the_ex_date_is_an_adjust_event(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    build(root)  # PAR 100 -> 80 on EX; KID at 40 from EX, 0.5 per parent share

    table = spinoff_events(con, root, LATE, rejected=())
    assert EVENT_COLUMNS <= set(table.schema.names)
    [row] = table.to_pylist()

    assert row["security_id"] == "S_PAR"
    assert row["parent_symbol"] == "PAR"
    assert list(row["child_symbols"]) == ["KID"]
    assert list(row["ratio"]) == [0.5]
    assert row["ex_date"] == EX
    assert (row["parent_prev_session"], row["parent_prev_close"]) == (DAYS[EX_I - 1], 100.0)
    assert (row["child_session"], list(row["child_closes"])) == (EX, [40.0])
    assert row["factor"] == pytest.approx(0.8)  # (100 - 40 * 0.5) / 100
    assert row["total_return"] == pytest.approx(0.0)  # (80 + 20) / 100 - 1
    assert (row["status"], row["reason"], row["known_on"]) == ("adjust", None, EX)


def test_list_columns_are_lists(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    build(root)
    schema = spinoff_events(con, root, LATE, rejected=()).schema
    for name in ("child_symbols", "ratio", "child_closes"):
        assert pa.types.is_list(schema.field(name).type) or pa.types.is_large_list(
            schema.field(name).type
        ), name


def test_ge_gev_numbers(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    # real: GE 175.36 -> 136.47, GEV 140.00, 1 GEV per 4 GE
    build(root, parent="GE", prev=175.36, after=136.47, children=[("GEV", 0, 140.0, 0.25)])

    row = one(con, root)

    assert row["factor"] == pytest.approx((175.36 - 35.0) / 175.36)
    assert row["factor"] == pytest.approx(0.8004, abs=1e-4)
    assert row["total_return"] == pytest.approx((136.47 + 35.0) / 175.36 - 1)
    assert row["status"] == "adjust" and row["known_on"] == EX


def test_open_three_children_on_one_day_are_one_event(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    ratio = 0.033333333
    kids = [("OPENW", 4, 1.82, ratio), ("OPENZ", 4, 0.6506, ratio), ("OPENL", 4, 1.08, ratio)]
    build(root, parent="OPEN", prev=7.84, after=7.52, children=kids)

    row = one(con, root)

    paid = ratio * (1.82 + 0.6506 + 1.08)
    assert list(row["child_symbols"]) == ["OPENL", "OPENW", "OPENZ"]
    assert list(row["child_closes"]) == pytest.approx([1.08, 1.82, 0.6506])
    assert list(row["ratio"]) == pytest.approx([ratio] * 3)
    assert row["factor"] == pytest.approx((7.84 - paid) / 7.84)
    assert row["factor"] == pytest.approx(0.9849, abs=1e-4)
    assert row["total_return"] == pytest.approx((7.52 + paid) / 7.84 - 1)
    assert row["child_session"] == DAYS[EX_I + 4]
    assert row["known_on"] == DAYS[EX_I + 4]
    assert row["status"] == "adjust"


def test_children_first_priced_on_different_days_wait_for_the_last(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    build(root, children=[("KIDA", 1, 20.0, 0.5), ("KIDB", 3, 10.0, 1.0)])

    row = one(con, root)

    assert list(row["child_symbols"]) == ["KIDA", "KIDB"]
    assert list(row["child_closes"]) == [20.0, 10.0]
    assert row["child_session"] == DAYS[EX_I + 3]
    assert row["known_on"] == DAYS[EX_I + 3]
    assert row["factor"] == pytest.approx((100 - 10 - 10) / 100)


def test_len_mrp_child_trading_13_sessions_late_still_adjusts(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    build(root, children=[("MRP", 13, 26.74, 0.5)], prev=141.54, after=133.14)

    row = one(con, root)

    assert row["status"] == "adjust"
    assert row["known_on"] == row["child_session"] == DAYS[EX_I + 13]
    assert row["factor"] == pytest.approx((141.54 - 13.37) / 141.54)


def test_child_close_is_its_first_bar_not_a_later_one(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    first = DAYS[EX_I + 2]
    kid = [px("KID", d, 40.0 if d == first else 60.0) for d in DAYS if d >= first]
    build(
        root,
        children=[("KID", None, 0.0, 0.5)],
        extra_bars=kid,
        segs=[
            Seg("S_PAR", "PAR"),
            Seg("S_KID", "KID", valid_from=first, available_at=known(first)),
        ],
    )

    row = one(con, root)

    assert list(row["child_closes"]) == [40.0]
    assert row["child_session"] == first


def test_child_bars_before_the_ex_date_are_ignored(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # when-issued style trading under the child's symbol before the ex-date does not count
    early = [px("KID", d, 99.0) for d in DAYS[EX_I - 3 : EX_I]]
    later = flat("KID", DAYS[EX_I + 1 :], 40.0)
    build(
        root,
        children=[("KID", None, 0.0, 0.5)],
        extra_bars=early + later,
        segs=[Seg("S_PAR", "PAR"), Seg("S_KID", "KID", valid_from=DAYS[EX_I - 3])],
    )

    row = one(con, root)

    assert row["child_session"] == DAYS[EX_I + 1]
    assert list(row["child_closes"]) == [40.0]


def test_parents_filter(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    bars = flat("PAR", DAYS, 100.0) + flat("OTH", DAYS, 50.0) + flat("KID", DAYS[EX_I:], 10.0)
    write_bars(root, bars)
    write_master(
        root, [Seg("S_PAR", "PAR"), Seg("S_OTH", "OTH"), Seg("S_KID", "KID", valid_from=EX)]
    )
    write_other_rows(root, [spin_row("PAR", "KID", EX, 1.0), spin_row("OTH", "KID", EX, 1.0)])

    assert [r["parent_symbol"] for r in events(con, root)] == ["OTH", "PAR"]
    assert [r["parent_symbol"] for r in events(con, root, parents=["PAR"])] == ["PAR"]
    assert events(con, root, parents=["KID"]) == []


# =============================================================================================
# spinoff_events: boundaries
# =============================================================================================


@pytest.mark.parametrize(
    ("offset", "status", "reason"),
    [(0, "adjust", None), (20, "adjust", None), (21, "restart", "child_too_late")],
    ids=["delay_0", "delay_20", "delay_21"],
)
def test_child_delay_counts_parent_sessions_with_the_ex_date_as_zero(
    con: duckdb.DuckDBPyConnection, root: Path, offset: int, status: str, reason: str | None
) -> None:
    build(root, children=[("KID", offset, 40.0, 0.5)])

    row = one(con, root)

    assert (row["status"], row["reason"]) == (status, reason)
    if status == "adjust":
        assert row["known_on"] == DAYS[EX_I + offset]
        assert row["factor"] == pytest.approx(0.8)
    else:
        assert row["known_on"] == DAYS[EX_I + 20]  # the 20th parent session after EX
        assert row["factor"] is None and row["total_return"] is None


def test_max_child_delay_is_a_parameter(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    build(root, children=[("KID", 4, 40.0, 0.5)])

    assert one(con, root, max_child_delay=4)["status"] == "adjust"
    late = one(con, root, max_child_delay=3)
    assert (late["status"], late["reason"], late["known_on"]) == (
        "restart",
        "child_too_late",
        DAYS[EX_I + 3],
    )


def test_child_that_never_trades_is_no_child_price(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # NVS -> SDZNY shape
    build(root, children=[("SDZNY", None, 0.0, 0.2)])

    row = one(con, root)

    assert (row["status"], row["reason"], row["known_on"]) == (
        "restart",
        "no_child_price",
        DAYS[EX_I + 20],
    )
    assert row["factor"] is None and row["total_return"] is None


def test_one_unpriced_child_sinks_the_whole_event(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    build(root, children=[("KIDA", 0, 20.0, 0.5), ("KIDB", None, 0.0, 0.5)])

    row = one(con, root)

    assert (row["status"], row["reason"], row["known_on"]) == (
        "restart",
        "no_child_price",
        DAYS[EX_I + 20],
    )


def test_delay_counts_the_parents_sessions_not_the_stores(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # ASSUMPTION (module docstring): the parent is halted for 5 sessions after EX; the child's
    # first bar 24 store sessions after EX is the parent's 19th session -> adjust
    halted = set(DAYS[EX_I + 1 : EX_I + 6])
    parent = [px("PAR", d, 100.0 if d < EX else 80.0) for d in DAYS if d not in halted]
    build(root, parent_bars=parent, children=[("KID", 24, 40.0, 0.5)])

    row = one(con, root)

    assert row["status"] == "adjust"
    assert row["known_on"] == DAYS[EX_I + 24]


def test_store_ending_before_the_window_closes_is_pending(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    days = DAYS[: EX_I + 11]  # only 10 parent sessions after EX exist anywhere
    build(root, days=days, children=[("KID", None, 0.0, 0.5)])

    row = one(con, root)

    assert row["status"] == "pending"
    assert row["factor"] is None and row["known_on"] is None and row["total_return"] is None


def test_parent_and_child_one_symbol_is_same_symbol(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # CR -> CR 2023-04-04
    build(root, parent="CR", children=[("CR", 0, 79.0, 1.0)])

    row = one(con, root)

    assert (row["status"], row["reason"], row["known_on"]) == ("restart", "same_symbol", EX)
    assert row["factor"] is None


def test_parent_without_any_bar_is_no_parent_price(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    build(
        root,
        parent="GONE",
        parent_bars=flat("OTH", DAYS, 10.0),
        segs=[Seg("S_OTH", "OTH")],
        children=[("KID", 0, 40.0, 0.5)],
    )

    row = one(con, root)

    assert (row["status"], row["reason"], row["known_on"]) == ("restart", "no_parent_price", EX)


def test_parent_first_traded_on_the_ex_date_is_no_parent_price(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    build(
        root,
        parent_bars=flat("PAR", DAYS[EX_I:], 80.0),
        segs=[Seg("S_PAR", "PAR", valid_from=EX, available_at=known(EX))],
    )

    row = one(con, root)

    assert (row["status"], row["reason"], row["known_on"]) == ("restart", "no_parent_price", EX)


@pytest.mark.parametrize(
    ("close", "ratio"),
    [(250.0, 0.5), (200.0, 0.5), (40.0, 0.0)],
    ids=["factor_negative", "factor_zero", "factor_one"],
)
def test_factor_outside_zero_one_is_restart(
    con: duckdb.DuckDBPyConnection, root: Path, close: float, ratio: float
) -> None:
    build(root, children=[("KID", 2, close, ratio)])

    row = one(con, root)

    assert (row["status"], row["reason"]) == ("restart", "factor_out_of_range")
    assert row["known_on"] == DAYS[EX_I + 2]
    assert row["factor"] is None and row["total_return"] is None


def test_parent_renamed_on_the_ex_date_is_still_one_security(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # the record is filed under the old symbol; from EX the same security trades as NEWP
    parent = [px("OLDP", d, 100.0) for d in DAYS[:EX_I]] + [
        px("NEWP", d, 70.0) for d in DAYS[EX_I:]
    ]
    segs = [
        Seg("S_P", "OLDP", valid_to=EX, end_available_at=known(EX)),
        Seg("S_P", "NEWP", valid_from=EX, available_at=known(EX)),
    ]
    build(root, parent="OLDP", parent_bars=parent, segs=segs, children=[("KID", 1, 40.0, 0.5)])

    row = one(con, root)

    assert row["security_id"] == "S_P"
    assert row["parent_symbol"] == "OLDP"
    assert row["status"] == "adjust" and row["factor"] == pytest.approx(0.8)
    assert row["total_return"] == pytest.approx((70.0 + 20.0) / 100.0 - 1)  # NEWP's EX close


def test_no_others_file_gives_an_empty_event_table(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    write_bars(root, flat("PAR", DAYS, 100.0))
    write_master(root, [Seg("S_PAR", "PAR")])

    table = spinoff_events(con, root, LATE, rejected=())

    assert table.num_rows == 0 and EVENT_COLUMNS <= set(table.schema.names)


def test_events_of_one_parent_on_two_days_are_two_rows(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # GE: GEHC on one day, GEV on another
    ex2 = DAYS[EX_I + 50]
    parent = [px("GE", d, 100.0 if d < EX else (80.0 if d < ex2 else 60.0)) for d in DAYS]
    build(
        root,
        parent="GE",
        parent_bars=parent,
        children=[("GEHC", 0, 40.0, 0.5)],
        extra_bars=flat("GEV", DAYS[EX_I + 50 :], 80.0),
        segs=[
            Seg("S_GE", "GE"),  # build() adds GEHC's segment from ``children``
            Seg("S_GEV", "GEV", valid_from=ex2, available_at=known(ex2)),
        ],
        records=[spin_row("GE", "GEHC", EX, 0.5), spin_row("GE", "GEV", ex2, 0.25)],
    )

    rows = events(con, root)

    assert [(r["ex_date"], r["status"]) for r in rows] == [(EX, "adjust"), (ex2, "adjust")]
    assert rows[0]["factor"] == pytest.approx(0.8)
    assert rows[1]["parent_prev_close"] == 80.0
    assert rows[1]["factor"] == pytest.approx((80.0 - 20.0) / 80.0)


def test_a_stored_duplicate_does_not_double_the_payout(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    build(
        root,
        records=[
            spin_row("PAR", "KID", EX, 0.5, uid="first"),
            spin_row("PAR", "KID", EX, 0.5, uid="second", process_date=EX + timedelta(days=3)),
        ],
    )

    row = one(con, root)

    assert list(row["child_symbols"]) == ["KID"]
    assert row["factor"] == pytest.approx(0.8)


# =============================================================================================
# leakage
# =============================================================================================


def test_before_the_childs_first_bar_is_visible_the_event_is_pending(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # (a) the child first trades on EX+3; the parent's bars of EX..EX+2 are visible earlier
    build(root, children=[("KID", 3, 40.0, 0.5)])
    first = DAYS[EX_I + 3]

    for as_of in (
        ny(EX, 10),  # the ex-date morning: record visible, no bar on/after EX yet
        avail(EX),
        avail(DAYS[EX_I + 2]),
        avail(first) - timedelta(microseconds=1),
    ):
        row = one(con, root, as_of)
        assert row["status"] == "pending", as_of
        assert row["factor"] is None and row["known_on"] is None, as_of

    decided = one(con, root, avail(first))
    assert (decided["status"], decided["known_on"]) == ("adjust", first)
    assert decided["factor"] == pytest.approx(0.8)


def test_record_not_yet_available_gives_no_event(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # (b) bars of EX-1 are visible, the record of EX is on disk but only knowable from 00:00
    build(root)

    assert events(con, root, known(EX) - timedelta(microseconds=1)) == []
    assert one(con, root, known(EX))["status"] == "pending"


def test_a_child_bar_on_disk_but_not_yet_available_is_not_used(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # (c) noon New York on the ex-date: the child's EX bar is stored, visible only at 20:00
    build(root, children=[("KID", 0, 40.0, 0.5)])

    assert one(con, root, ny(EX, 12))["status"] == "pending"
    assert one(con, root, avail(EX))["status"] == "adjust"


def test_a_child_bar_imported_late_is_not_used_before_its_available_at(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # (c) a child bar whose available_at lies days after its session (constructed: the store's
    # rule is 20:00 of the session, audit P6 gates exceptions). Until it is knowable the event
    # must not be decided from it.
    first = DAYS[EX_I + 1]
    late_at = avail(DAYS[EX_I + 4])
    kid = [px("KID", first, 40.0, available_at=late_at)] + flat("KID", DAYS[EX_I + 5 :], 40.0)
    build(
        root,
        children=[("KID", None, 0.0, 0.5)],
        extra_bars=kid,
        segs=[
            Seg("S_PAR", "PAR"),
            Seg("S_KID", "KID", valid_from=first, available_at=known(first)),
        ],
    )

    for as_of in (avail(first), avail(DAYS[EX_I + 3]), late_at - timedelta(microseconds=1)):
        row = one(con, root, as_of)
        assert row["status"] == "pending", as_of
        assert row["factor"] is None, as_of
    assert one(con, root, late_at)["status"] == "adjust"


def test_future_parent_and_child_bars_do_not_change_an_event_decided_earlier(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # the child's later closes and the parent's later sessions exist on disk; the decision made
    # at the child's first evening is the one seen at LATE
    build(root, children=[("KID", 2, 40.0, 0.5)])
    at_decision = one(con, root, avail(DAYS[EX_I + 2]))
    at_late = one(con, root, LATE)

    assert decision(at_decision) == decision(at_late)
    assert at_decision["child_closes"] == at_late["child_closes"]
    assert at_decision["total_return"] == pytest.approx(at_late["total_return"])


SCENARIOS: dict[str, dict[str, Any]] = {
    "adjust_on_ex": {"children": [("KID", 0, 40.0, 0.5)]},
    "adjust_lagging": {"children": [("KID", 7, 40.0, 0.5)]},
    "adjust_at_window_edge": {"children": [("KID", 20, 40.0, 0.5)]},
    "child_too_late": {"children": [("KID", 21, 40.0, 0.5)]},
    "no_child_price": {"children": [("KID", None, 40.0, 0.5)]},
    "factor_out_of_range": {"children": [("KID", 2, 250.0, 0.5)]},
    "same_symbol": {"parent": "CR", "children": [("CR", 0, 79.0, 1.0)]},
    "two_children": {"children": [("KIDA", 1, 20.0, 0.5), ("KIDB", 5, 10.0, 1.0)]},
}


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_every_decision_is_pending_until_known_on_and_final_from_it(
    con: duckdb.DuckDBPyConnection, root: Path, name: str
) -> None:
    # Day by day from the ex-date morning: the event is pending while the as_of session is
    # before the final known_on, and from the evening of known_on it reads exactly as at LATE
    # (a child bar seen later turns no_child_price into child_too_late at LATE: the reason may
    # differ, the status, factor and known_on may not).
    build(root, **SCENARIOS[name])
    final = one(con, root, LATE)
    assert final["status"] != "pending"
    known_on = final["known_on"]

    for day in DAYS[EX_I : EX_I + 25]:
        row = one(con, root, avail(day))
        if day < known_on:
            assert row["status"] == "pending", (day, row)
            assert row["factor"] is None and row["known_on"] is None, day
        else:
            assert row["status"] == final["status"], day
            assert row["known_on"] == known_on, day
            assert row["factor"] == final["factor"], day


# =============================================================================================
# review follow-ups: one event per security, conflicting ratios, every input's visibility
# =============================================================================================


def test_old_and_new_name_of_a_rename_day_are_one_event(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # constructed: the feed files the same distribution under both names of a rename day (the
    # split feed did, 7b9a257); two rows would multiply the factor in twice (0.64, not 0.8)
    parent = [px("OLDP", d, 100.0) for d in DAYS[:EX_I]] + [
        px("NEWP", d, 80.0) for d in DAYS[EX_I:]
    ]
    segs = [
        Seg("S_P", "OLDP", valid_to=EX, end_available_at=known(EX)),
        Seg("S_P", "NEWP", valid_from=EX, available_at=known(EX)),
    ]
    build(
        root,
        parent="OLDP",
        parent_bars=parent,
        segs=segs,
        records=[spin_row("OLDP", "KID", EX, 0.5), spin_row("NEWP", "KID", EX, 0.5)],
    )

    row = one(con, root)

    assert row["security_id"] == "S_P" and row["parent_symbol"] == "NEWP"
    assert list(row["child_symbols"]) == ["KID"]
    assert row["factor"] == pytest.approx(0.8)


def test_one_child_at_two_ratios_is_a_restart(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    # constructed: a revised ratio stored next to the original; adding both would pay the child
    # out 1.5 times, and picking the later copy could use a record visible only after known_on
    build(root, records=[spin_row("PAR", "KID", EX, 0.5), spin_row("PAR", "KID", EX, 1.0)])

    row = one(con, root)

    assert (row["status"], row["reason"], row["known_on"]) == ("restart", "conflicting_ratio", EX)
    assert row["factor"] is None


def test_known_on_waits_for_a_late_record_and_a_late_previous_close(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # constructed (the store's rules make both 00:00 / 20:00 today): every input of the factor
    # must be visible on known_on, not only the child's first bar
    late_record = spin_row("PAR", "KID", EX, 0.5, available_at=avail(DAYS[EX_I + 1]))
    build(root, records=[late_record])
    assert one(con, root)["known_on"] == DAYS[EX_I + 1]

    prev_day = DAYS[EX_I - 1]
    parent = [
        px("PAR", d, 100.0 if d < EX else 80.0, available_at=avail(DAYS[EX_I + 2]))
        if d == prev_day
        else px("PAR", d, 100.0 if d < EX else 80.0)
        for d in DAYS
    ]
    build(root, parent_bars=parent)
    assert one(con, root)["known_on"] == DAYS[EX_I + 2]


@pytest.mark.parametrize(
    ("after", "status"),
    [(74.0, "adjust"), (76.0, "restart"), (46.0, "adjust"), (44.0, "restart")],
    ids=["plus_14", "plus_16", "minus_14", "minus_16"],
)
def test_a_total_return_beyond_15_percent_disproves_the_factor(
    con: duckdb.DuckDBPyConnection, root: Path, after: float, status: str
) -> None:
    # prev 100, one child worth 40 per share -> total return (after + 40) / 100 - 1
    build(root, after=after, children=[("KID", 0, 40.0, 1.0)])

    row = one(con, root)

    assert row["status"] == status
    assert row["total_return"] == pytest.approx((after + 40.0) / 100.0 - 1)
    if status == "restart":
        assert (row["reason"], row["factor"], row["known_on"]) == ("factor_disproved", None, EX)
    else:
        assert row["factor"] == pytest.approx(0.6)


def test_max_total_return_is_a_parameter(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    build(root, after=76.0, children=[("KID", 0, 40.0, 1.0)])  # +16%

    assert one(con, root, max_total_return=0.2)["status"] == "adjust"
    assert one(con, root)["reason"] == "factor_disproved"


def test_the_check_waits_for_the_parents_ex_session_bar(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # the child trades on EX; the parent's bars from EX on are visible only from EX+2's evening
    # (constructed; the store's rule is 20:00 of the session). Until then the event is pending:
    # deciding "adjust" first and "factor_disproved" later would change a decision
    late = avail(DAYS[EX_I + 2])
    parent = [
        px("PAR", d, 100.0 if d < EX else 76.0, available_at=late if d >= EX else None)
        for d in DAYS
    ]
    build(root, parent_bars=parent, children=[("KID", 0, 40.0, 1.0)])

    early = one(con, root, avail(DAYS[EX_I + 1]))
    assert (early["status"], early["factor"], early["known_on"]) == ("pending", None, None)
    decided = one(con, root, late)
    assert (decided["status"], decided["reason"], decided["known_on"]) == (
        "restart",
        "factor_disproved",
        DAYS[EX_I + 2],
    )
    assert decided == one(con, root)


def test_a_parent_that_never_trades_again_stays_pending(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # real shape: FAT, LBTYA, STAR continue as another security on their ex-dates
    build(root, parent_bars=[px("PAR", d, 100.0) for d in DAYS[EX_I - 3 : EX_I]])

    row = one(con, root)

    assert (row["status"], row["factor"], row["known_on"]) == ("pending", None, None)
