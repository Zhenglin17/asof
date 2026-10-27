"""Guards against one split applied twice because the feed stores it as several rows: every
reader (replay, null, scan, benchmark, audit) goes through ``splits_as_of``, so a repeated
record there multiplies a position by the factor once per copy. Also guards the opposite
mistake (two genuine events on one day merged into one) and leakage through the duplicate rule
(a copy not yet knowable at ``as_of`` must neither appear nor change what is returned).

Target API (asof.store.market):

    splits_as_of(con, relation, as_of, symbols=None) -> pa.Table   (visible_splits delegates)
        Rows with equal (symbol, ex_date, factor to 9 decimals, old_cusip, new_cusip) are ONE
        event and come back once. NULL cusips compare equal to NULL (IS NOT DISTINCT FROM).
        ``kind``, ``old_rate`` / ``new_rate`` and ``available_at`` do not take part. Among the
        copies knowable at ``as_of`` the one with the earliest ``available_at`` is returned, a
        non-``unit_split`` copy first. Schema
        stays SPLIT_SCHEMA; order stays symbol, ex_date.

Fake data provenance -- real rows of /data/asof/market/corporate_actions/splits.parquet
(store probe 2026-10-27; prices from the bar store, quoted by the design review):
- SNEX 2023-11-27: two rows identical in every column (2 -> 3, factor 1.5, forward_split,
  old_cusip 861896108, new_cusip NULL). Price 93.99 -> 61.45 confirms 1.5 once.
- NLSP 2025-10-31: reverse_split 10 -> 1 and unit_split 1 -> 0.1, both factor 0.1, both
  H57830137 -> H5835A109. One 1-for-10 event (NLSP became NCEL that day, same security).
- MDRR 2024-07-03: reverse_split 10 -> 1 (58403P303 -> 58403P402) and forward_split 1 -> 5
  (58403P402 -> NULL). Two events; product 0.5 matches the price 5.99 -> 11.79.
- GWACU 2020-12-07: reverse_split 1 -> 0.5 (cusips NULL) and unit_split 1 -> 1
  (38216X206 -> 38216X107). Two events; no bars afterwards (audit S5 reports it).
  SAQNU 2020-10-15 has the same shape and is not repeated here.
- available_at is 00:00 New York of the ex-date (ingest rule, docs in
  asof.ingest.corporate_actions). Copies with different available_at do not occur in the real
  table today (ASSUMPTION for the leak tests: such copies could appear after a re-ingest); the
  rule for them is the design's, not observed behaviour.

Real-data checks (the fixture shapes above are only as good as these):
- ``asof market audit`` S5 lists exactly MDRR:2024-07-03, GWACU:2020-12-07, SAQNU:2020-10-15
  (and nothing for SNEX / NLSP) on the store of 2026-10-27; anything else is a new shape.
- Golden case: ``visible_splits`` on the real store returns one SNEX 2023-11-27 row (1.5) and
  one NLSP 2025-10-31 row (0.1), two MDRR 2024-07-03 rows whose factors multiply to 0.5.
"""

from collections.abc import Iterator, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from asof.ingest.corporate_actions import SPLIT_SCHEMA
from asof.store.market import market_root, splits_as_of, visible_splits
from tests.store.test_market import known

SNEX_DAY = date(2023, 11, 27)
NLSP_DAY = date(2025, 10, 31)
MDRR_DAY = date(2024, 7, 3)
GWACU_DAY = date(2020, 12, 7)
LATE = datetime(2026, 10, 27, tzinfo=UTC)  # every fixture record is knowable


# =============================================================================================
# builders (shared with tests/replay and tests/evolve)
# =============================================================================================


def split_row(
    symbol: str,
    ex_date: date,
    old_rate: float,
    new_rate: float,
    *,
    factor: float | None = None,
    kind: str = "forward_split",
    old_cusip: str | None = None,
    new_cusip: str | None = None,
    available_at: datetime | None = None,
) -> dict[str, Any]:
    """One stored split record; factor = new / old and available_at per the ingest rule unless
    given."""
    return {
        "symbol": symbol,
        "ex_date": ex_date,
        "old_rate": old_rate,
        "new_rate": new_rate,
        "factor": new_rate / old_rate if factor is None else factor,
        "kind": kind,
        "old_cusip": old_cusip,
        "new_cusip": new_cusip,
        "available_at": known(ex_date) if available_at is None else available_at,
    }


def write_split_rows(root: Path, rows: Sequence[dict[str, Any]]) -> Path:
    """Write ``rows`` as the store's splits.parquet (replacing any earlier file)."""
    path = root / "corporate_actions" / "splits.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(list(rows), schema=SPLIT_SCHEMA), path)
    return path


def snex_rows(symbol: str = "SNEX", day: date = SNEX_DAY) -> list[dict[str, Any]]:
    """Real: the same 3-for-2 stored twice, identical in every column."""
    row = split_row(symbol, day, 2.0, 3.0, kind="forward_split", old_cusip="861896108")
    return [row, dict(row)]


def nlsp_rows(symbol: str = "NLSP", day: date = NLSP_DAY) -> list[dict[str, Any]]:
    """Real: one 1-for-10 stored as reverse_split 10 -> 1 and unit_split 1 -> 0.1."""
    old, new = "H57830137", "H5835A109"
    return [
        split_row(symbol, day, 10.0, 1.0, kind="reverse_split", old_cusip=old, new_cusip=new),
        split_row(symbol, day, 1.0, 0.1, kind="unit_split", old_cusip=old, new_cusip=new),
    ]


def mdrr_rows(symbol: str = "MDRR", day: date = MDRR_DAY) -> list[dict[str, Any]]:
    """Real: 1-for-10 then 5-for-1 on the same day, chained through CUSIP 58403P402."""
    return [
        split_row(
            symbol,
            day,
            10.0,
            1.0,
            kind="reverse_split",
            old_cusip="58403P303",
            new_cusip="58403P402",
        ),
        split_row(symbol, day, 1.0, 5.0, kind="forward_split", old_cusip="58403P402"),
    ]


def gwacu_rows(symbol: str = "GWACU", day: date = GWACU_DAY) -> list[dict[str, Any]]:
    """Real: a 1-for-2 without CUSIPs plus a 1:1 unit split with a CUSIP change."""
    return [
        split_row(symbol, day, 1.0, 0.5, kind="reverse_split"),
        split_row(
            symbol,
            day,
            1.0,
            1.0,
            kind="unit_split",
            old_cusip="38216X206",
            new_cusip="38216X107",
        ),
    ]


# =============================================================================================
# fixtures and helpers
# =============================================================================================


@pytest.fixture
def con() -> Iterator[duckdb.DuckDBPyConnection]:
    connection = duckdb.connect()
    yield connection
    connection.close()


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return market_root(tmp_path)


def visible(
    con: duckdb.DuckDBPyConnection,
    root: Path,
    rows: Sequence[dict[str, Any]],
    as_of: datetime = LATE,
    symbols: Sequence[str] | None = None,
) -> pa.Table:
    write_split_rows(root, rows)
    return visible_splits(con, root, as_of, symbols)


def factors(table: pa.Table) -> list[float]:
    return sorted(table.column("factor").to_pylist())


def event_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row["symbol"], row["ex_date"], row["factor"], row["old_cusip"], row["new_cusip"])


# =============================================================================================
# normal path: the four real shapes
# =============================================================================================


def test_snex_identical_copies_are_one_event(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    table = visible(con, root, snex_rows())

    assert table.schema == SPLIT_SCHEMA
    assert table.to_pylist() == [snex_rows()[0]]


def test_nlsp_same_factor_and_cusips_different_kind_and_rates_is_one_event(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    table = visible(con, root, nlsp_rows())

    [only] = table.to_pylist()
    # the reverse_split copy: the audit's split checks (S1, S2) skip unit splits
    assert only == nlsp_rows()[0]


def test_mdrr_two_events_on_one_day_both_stay(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    table = visible(con, root, mdrr_rows())

    assert table.num_rows == 2
    assert factors(table) == pytest.approx([0.1, 5.0])
    assert sorted(table.to_pylist(), key=lambda r: r["factor"]) == mdrr_rows()


def test_gwacu_reverse_split_plus_unit_split_both_stay(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    table = visible(con, root, gwacu_rows())

    assert table.num_rows == 2
    assert factors(table) == pytest.approx([0.5, 1.0])


def test_mixed_store_keeps_order_by_symbol_then_ex_date(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    later_snex = snex_rows(day=date(2024, 6, 3))
    rows = later_snex + nlsp_rows() + snex_rows() + mdrr_rows() + gwacu_rows()

    table = visible(con, root, rows)

    symbols, days = table.column("symbol").to_pylist(), table.column("ex_date").to_pylist()
    pairs = list(zip(symbols, days, strict=True))
    assert pairs == [
        ("GWACU", GWACU_DAY),
        ("GWACU", GWACU_DAY),
        ("MDRR", MDRR_DAY),
        ("MDRR", MDRR_DAY),
        ("NLSP", NLSP_DAY),
        ("SNEX", SNEX_DAY),
        ("SNEX", date(2024, 6, 3)),
    ]


def test_the_cached_relation_path_applies_the_same_rule(con: duckdb.DuckDBPyConnection) -> None:
    # the replay's bar cache hands splits_as_of a registered table instead of a file
    con.register("cached_splits", pa.Table.from_pylist(snex_rows() + nlsp_rows(), SPLIT_SCHEMA))

    table = splits_as_of(con, "cached_splits", LATE)

    assert table.schema == SPLIT_SCHEMA
    assert [(r["symbol"], r["factor"]) for r in table.to_pylist()] == [
        ("NLSP", pytest.approx(0.1)),
        ("SNEX", 1.5),
    ]


# =============================================================================================
# boundaries: what makes two rows one event
# =============================================================================================


def test_single_row_is_returned_untouched(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    row = split_row("AAPL", date(2020, 8, 31), 1.0, 4.0)  # real: 4-for-1

    assert visible(con, root, [row]).to_pylist() == [row]


def test_same_factor_different_new_cusip_is_two_events(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    a, b = nlsp_rows()
    b = {**b, "new_cusip": "H5835A999"}

    assert visible(con, root, [a, b]).num_rows == 2


def test_same_factor_different_old_cusip_is_two_events(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    a, b = nlsp_rows()
    b = {**b, "old_cusip": "H57830999"}

    assert visible(con, root, [a, b]).num_rows == 2


def test_same_cusips_different_factor_is_two_events(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    a, _ = nlsp_rows()
    b = {**a, "new_rate": 0.5, "factor": 0.5}

    table = visible(con, root, [a, b])

    assert factors(table) == pytest.approx([0.1, 0.5])


def test_unit_split_copy_loses_even_when_listed_first(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    reverse, unit = nlsp_rows()

    assert visible(con, root, [unit, reverse]).to_pylist() == [reverse]


def test_factors_equal_to_nine_decimals_are_one_event(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # a 1-for-15 stored as 15 -> 1 and as 1 -> 0.0666666667 (rounded by the feed)
    a = split_row("ZZZ", date(2022, 3, 1), 15.0, 1.0, kind="reverse_split")
    b = split_row("ZZZ", date(2022, 3, 1), 1.0, 0.0666666667, kind="unit_split")
    assert a["factor"] != b["factor"]

    assert visible(con, root, [a, b]).to_pylist() == [a]


def test_null_cusips_compare_equal_to_null(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    a = split_row("ZZZ", date(2022, 3, 1), 1.0, 2.0)  # both cusips NULL
    b = {**a, "kind": "unit_split"}

    assert visible(con, root, [a, b]).num_rows == 1


@pytest.mark.parametrize("column", ["old_cusip", "new_cusip"])
def test_null_cusip_versus_a_value_is_two_events(
    con: duckdb.DuckDBPyConnection, root: Path, column: str
) -> None:
    a = split_row("ZZZ", date(2022, 3, 1), 1.0, 2.0)
    b = {**a, column: "123456789"}

    assert visible(con, root, [a, b]).num_rows == 2


def test_same_event_on_another_day_or_symbol_is_not_merged(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    [a, _] = snex_rows()
    rows = [a, {**a, "symbol": "SNEY"}, {**a, "ex_date": date(2023, 11, 28)}]

    assert visible(con, root, rows).num_rows == 3


def test_symbols_filter_still_works_with_duplicates(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    rows = snex_rows() + nlsp_rows() + mdrr_rows()

    snex = visible(con, root, rows, symbols=["SNEX"])
    both = visible(con, root, rows, symbols=["SNEX", "MDRR"])
    none = visible(con, root, rows, symbols=["AAPL"])

    assert snex.column("symbol").to_pylist() == ["SNEX"]
    assert both.column("symbol").to_pylist() == ["MDRR", "MDRR", "SNEX"]
    assert none.num_rows == 0 and none.schema == SPLIT_SCHEMA


# =============================================================================================
# leakage: copies not yet knowable at as_of
# =============================================================================================


def test_a_future_duplicate_is_invisible_and_changes_nothing(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    [a, _] = snex_rows()
    as_of = known(SNEX_DAY) + timedelta(hours=12)
    future = {**a, "kind": "unit_split", "available_at": as_of + timedelta(microseconds=1)}

    alone = visible(con, root, [a], as_of=as_of).to_pylist()
    with_future = visible(con, root, [a, future], as_of=as_of).to_pylist()

    assert with_future == alone == [a]


def test_between_two_copies_available_ats_exactly_one_row_the_visible_one(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    a, b = nlsp_rows()
    t1 = known(NLSP_DAY)
    t2 = t1 + timedelta(days=3)  # a re-ingested copy, known later
    early, late = {**b, "available_at": t1}, {**a, "available_at": t2}

    before = visible(con, root, [late, early], as_of=t1 - timedelta(microseconds=1))
    between = visible(con, root, [late, early], as_of=t1 + timedelta(days=1))
    after = visible(con, root, [late, early], as_of=t2 + timedelta(days=1))

    assert before.num_rows == 0
    assert between.to_pylist() == [early]
    # once both are knowable the event is still the earliest copy: what was seen does not change
    assert after.num_rows == 1
    assert after.column("available_at").to_pylist() == [t1]


def test_a_future_second_event_on_the_same_day_is_invisible(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    first, second = mdrr_rows()
    as_of = known(MDRR_DAY)
    second = {**second, "available_at": as_of + timedelta(microseconds=1)}

    assert visible(con, root, [first, second], as_of=as_of).to_pylist() == [first]


@pytest.mark.parametrize("hours", [-1, 0, 6, 24, 72, 100, 200])
def test_no_row_with_available_at_after_as_of_is_ever_returned(
    con: duckdb.DuckDBPyConnection, root: Path, hours: int
) -> None:
    base = known(NLSP_DAY)
    rows: list[dict[str, Any]] = []
    for i, row in enumerate(nlsp_rows() + mdrr_rows() + snex_rows(day=NLSP_DAY)):
        rows.append({**row, "available_at": base + timedelta(hours=24 * i)})
    as_of = base + timedelta(hours=hours)

    table = visible(con, root, rows, as_of=as_of)

    got = table.to_pylist()
    assert all(r["available_at"] <= as_of for r in got)
    # one row per event among the copies knowable at as_of
    expected = {event_key(r) for r in rows if r["available_at"] <= as_of}
    assert sorted(map(event_key, got), key=str) == sorted(expected, key=str)
