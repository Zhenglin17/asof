"""Guards against a vendor split record that the prices disprove still reaching a reader: every
reader (replay, null, scan, benchmark, audit, features) multiplies prices or positions by what
``splits_as_of`` returns, so one false record halves (or doubles) a whole history. Also guards
the opposite mistake: a rejection that drops more than its own (symbol, ex_date), or a genuine
split with a large move (TTOO / TAOP / RGC) being rejected.

Target API (asof.store.market):

    SPLIT_REJECTIONS_FILE: Path        <repo>/configs/split_rejections.yaml
    load_split_rejections(path) -> frozenset[tuple[str, date]]
        a YAML list of ``key / reason / added / by`` (same shape and checks as
        configs/audit_exceptions.yaml minus ``check``). key is ``SYMBOL:YYYY-MM-DD`` (SYMBOL as
        the audit's [A-Z0-9.]+, the date a real ISO date); every field present and non-empty,
        nothing extra; added a YYYY-MM-DD date; a key listed twice is an error. A missing file
        or an empty list is the empty set (the loader); a missing SPLIT_REJECTIONS_FILE is an
        error for readers that use the default.
    splits_as_of(con, relation, as_of, symbols=None, *, rejected=None)
    visible_splits(con, root, as_of, symbols=None, *, rejected=None)
        rejected=None reads SPLIT_REJECTIONS_FILE; an explicit collection (``()`` included)
        replaces it. Every row whose (symbol, ex_date) is rejected is left out, whatever its
        kind or factor; nothing else changes (de-duplication, order, schema, as_of cut).

Fake data provenance:
- HON 2026-06-29: reverse_split 2 -> 1 (factor 0.5); opens 1.04, 1.00, 0.98 of the previous
  close over three sessions (real, audit S2 of 2026-10-19).
- AIIO 2026-06-03: reverse_split 10 -> 1 (factor 0.1); opens 0.95, 0.98, 1.08 (real, same).
- TTOO 2023-10-13 (100:1 reverse), TAOP 2020-07-30 (6:1 reverse), RGC 2025-06-16 (38:1
  forward): genuine splits whose opens moved by a different multiple (real, same audit run);
  they stay audit exceptions and are never rejected.
- available_at is 00:00 New York of the ex-date (ingest rule).

Real-data checks: tests/store/test_golden_cases.py (``visible_splits`` at NOW has no HON
2026-06-29 or AIIO 2026-06-03 row) and ``asof market audit`` (no stale S2 exception, no S2 row
for either key).

Not covered here: the rejection is not time-scoped -- a replay on 2026-06-29 does not see the
false record either, although nobody had disproved it yet. It is a data correction (like an
imported bar), not knowledge a strategy acts on.
"""

from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pytest
import yaml

from asof.ingest.corporate_actions import SPLIT_SCHEMA
from asof.store import market
from asof.store.market import (
    SPLIT_REJECTIONS_FILE,
    load_split_rejections,
    market_root,
    splits_as_of,
    visible_splits,
)
from tests.store.test_split_events import mdrr_rows, split_row, write_split_rows

REPO = Path(__file__).resolve().parents[2]
HON_DAY = date(2026, 6, 29)
AIIO_DAY = date(2026, 6, 3)
LATE = datetime(2026, 10, 27, tzinfo=UTC)
SHIPPED = {("HON", HON_DAY), ("AIIO", AIIO_DAY)}


def hon() -> dict[str, Any]:
    return split_row("HON", HON_DAY, 2.0, 1.0, kind="reverse_split")


def aiio() -> dict[str, Any]:
    return split_row("AIIO", AIIO_DAY, 10.0, 1.0, kind="reverse_split")


def genuine() -> list[dict[str, Any]]:
    return [
        split_row("TTOO", date(2023, 10, 13), 100.0, 1.0, kind="reverse_split"),
        split_row("TAOP", date(2020, 7, 30), 6.0, 1.0, kind="reverse_split"),
        split_row("RGC", date(2025, 6, 16), 1.0, 38.0, kind="forward_split"),
    ]


@pytest.fixture
def con() -> Iterator[duckdb.DuckDBPyConnection]:
    connection = duckdb.connect()
    yield connection
    connection.close()


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return market_root(tmp_path)


def pairs(table: pa.Table) -> list[tuple[str, date]]:
    return list(zip(table["symbol"].to_pylist(), table["ex_date"].to_pylist(), strict=True))


# =============================================================================================
# normal path
# =============================================================================================


def test_rejected_row_is_absent_and_the_rest_untouched(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    rows = [hon(), aiio(), *genuine()]
    write_split_rows(root, rows)

    table = visible_splits(con, root, LATE, rejected={("HON", HON_DAY)})

    assert table.schema == SPLIT_SCHEMA
    assert pairs(table) == [
        ("AIIO", AIIO_DAY),
        ("RGC", date(2025, 6, 16)),
        ("TAOP", date(2020, 7, 30)),
        ("TTOO", date(2023, 10, 13)),
    ]


def test_default_reads_the_shipped_file(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    write_split_rows(root, [hon(), aiio(), *genuine()])

    default = visible_splits(con, root, LATE)
    unrejected = visible_splits(con, root, LATE, rejected=())

    assert ("HON", HON_DAY) not in pairs(default) and ("AIIO", AIIO_DAY) not in pairs(default)
    assert len(pairs(default)) == 3
    assert len(pairs(unrejected)) == 5


def test_the_cached_relation_path_rejects_too(con: duckdb.DuckDBPyConnection) -> None:
    # the replay's bar cache hands splits_as_of a registered table instead of a file
    con.register("cached_splits", pa.Table.from_pylist([hon(), *genuine()], SPLIT_SCHEMA))

    default = splits_as_of(con, "cached_splits", LATE)
    explicit = splits_as_of(con, "cached_splits", LATE, rejected={("TTOO", date(2023, 10, 13))})

    assert ("HON", HON_DAY) not in pairs(default) and len(pairs(default)) == 3
    assert ("TTOO", date(2023, 10, 13)) not in pairs(explicit)
    assert ("HON", HON_DAY) in pairs(explicit)  # an explicit set replaces the shipped one


def test_loader_reads_every_entry(tmp_path: Path) -> None:
    path = tmp_path / "split_rejections.yaml"
    path.write_text(
        '- key: "HON:2026-06-29"\n'
        "  reason: no move\n"
        "  added: 2026-10-27\n"
        "  by: greyson\n"
        '- key: "BRK.B:2020-01-02"\n'
        "  reason: made up\n"
        '  added: "2026-10-27"\n'
        "  by: greyson\n"
    )

    assert load_split_rejections(path) == frozenset({("HON", HON_DAY), ("BRK.B", date(2020, 1, 2))})


def test_shipped_rejections_are_exactly_hon_and_aiio() -> None:
    assert SPLIT_REJECTIONS_FILE == REPO / "configs" / "split_rejections.yaml"
    assert load_split_rejections(SPLIT_REJECTIONS_FILE) == frozenset(SHIPPED)
    entries = yaml.safe_load(SPLIT_REJECTIONS_FILE.read_text())
    assert all(str(e["added"]) == "2026-10-27" and e["by"] == "greyson" for e in entries)


def test_shipped_audit_exceptions_no_longer_list_the_rejected_records() -> None:
    entries = yaml.safe_load((REPO / "configs" / "audit_exceptions.yaml").read_text())
    s2 = {e["key"] for e in entries if e["check"] == "S2"}

    assert not {"HON:2026-06-29", "AIIO:2026-06-03"} & {e["key"] for e in entries}
    # the genuine large-move splits stay excepted, not rejected
    assert {"TTOO:2023-10-13", "TAOP:2020-07-30", "RGC:2025-06-16"} <= s2


# =============================================================================================
# boundaries
# =============================================================================================


def test_every_row_of_a_rejected_day_goes(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    # two distinct events on one day (the MDRR shape): a rejection is per (symbol, ex_date)
    write_split_rows(root, mdrr_rows(symbol="HON", day=HON_DAY))

    assert visible_splits(con, root, LATE, rejected=()).num_rows == 2
    assert visible_splits(con, root, LATE).num_rows == 0


def test_same_symbol_other_day_and_other_symbol_same_day_stay(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    other_day = split_row("HON", date(2018, 10, 1), 1.0, 2.0)
    other_symbol = split_row("HONE", HON_DAY, 2.0, 1.0, kind="reverse_split")
    write_split_rows(root, [hon(), other_day, other_symbol])

    assert pairs(visible_splits(con, root, LATE)) == [
        ("HON", date(2018, 10, 1)),
        ("HONE", HON_DAY),
    ]


def test_genuine_large_move_splits_are_never_rejected(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    write_split_rows(root, genuine())
    assert visible_splits(con, root, LATE).num_rows == 3


def test_symbols_filter_and_rejection_together(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    write_split_rows(root, [hon(), aiio(), *genuine()])

    assert visible_splits(con, root, LATE, ["HON"]).num_rows == 0
    assert pairs(visible_splits(con, root, LATE, ["HON", "RGC"])) == [("RGC", date(2025, 6, 16))]


def test_empty_store_still_returns_the_schema(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    table = visible_splits(con, root, LATE, rejected=SHIPPED)
    assert table.num_rows == 0 and table.schema == SPLIT_SCHEMA


def test_a_missing_shipped_file_is_an_error(
    con: duckdb.DuckDBPyConnection, root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # e.g. a non-editable install where the path no longer reaches the repo: HON's false 2:1
    # would otherwise be applied silently
    write_split_rows(root, [hon()])
    monkeypatch.setattr(market, "SPLIT_REJECTIONS_FILE", tmp_path / "absent.yaml")

    with pytest.raises(FileNotFoundError):
        visible_splits(con, root, LATE)
    assert len(pairs(visible_splits(con, root, LATE, rejected=()))) == 1


def test_loader_missing_file_and_empty_list(tmp_path: Path) -> None:
    assert load_split_rejections(tmp_path / "absent.yaml") == frozenset()
    empty = tmp_path / "empty.yaml"
    empty.write_text("[]\n")
    assert load_split_rejections(empty) == frozenset()


def entry(**overrides: Any) -> str:
    fields = {"key": '"HON:2026-06-29"', "reason": "no move", "added": "2026-10-27", "by": "g"}
    fields.update(overrides)
    lines = [f"  {k}: {v}" for k, v in fields.items() if v is not None]
    lines[0] = "-" + lines[0][1:]
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("key: HON:2026-06-29\n", r"list"),
        ("- HON:2026-06-29\n", r"HON:2026-06-29"),
        (entry(by=None), r"HON:2026-06-29"),
        (entry(reason=None), r"HON:2026-06-29"),
        (entry(check="S2"), r"HON:2026-06-29"),
        (entry(reason='""'), r"HON:2026-06-29"),
        (entry(key='"HON"'), r"HON"),
        (entry(key='"hon:2026-06-29"'), r"hon:2026-06-29"),
        (entry(key='"HON:2026-02-30"'), r"HON:2026-02-30"),
        (entry(added="yesterday"), r"yesterday"),
        (entry() + entry(), r"HON:2026-06-29"),
    ],
    ids=[
        "not_a_list",
        "item_not_a_mapping",
        "missing_by",
        "missing_reason",
        "extra_field",
        "empty_field",
        "key_without_date",
        "key_lowercase",
        "key_impossible_date",
        "added_not_a_date",
        "duplicate_key",
    ],
)
def test_loader_rejects_bad_entries(tmp_path: Path, text: str, match: str) -> None:
    path = tmp_path / "split_rejections.yaml"
    path.write_text(text)
    with pytest.raises(ValueError, match=match):
        load_split_rejections(path)


# =============================================================================================
# leakage: the rejection must not disturb the as_of cut
# =============================================================================================


def test_rejection_does_not_reveal_or_hide_anything_else_across_as_of(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # a later (future) record of another symbol exists on disk; at an as_of before it the
    # rejection must neither surface it nor hide the visible genuine records
    future = split_row("ZZZ", date(2026, 11, 2), 1.0, 2.0)
    write_split_rows(root, [hon(), *genuine(), future])
    as_of = datetime(2026, 7, 1, tzinfo=UTC)

    table = visible_splits(con, root, as_of)

    assert all(a <= as_of for a in table["available_at"].to_pylist())
    assert ("ZZZ", date(2026, 11, 2)) not in pairs(table)
    assert pairs(table) == pairs(visible_splits(con, root, as_of, rejected=SHIPPED))
    assert len(pairs(table)) == 3
