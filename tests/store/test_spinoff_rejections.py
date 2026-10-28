"""Guards against a vendor spin-off record that the prices disprove still reaching a reader: a
false record rescales the parent's whole history by its factor (HON 2026-06-29 would multiply
every earlier HON price by 0.53). Also guards the opposite mistake: a rejection that drops more
than its own (parent, ex_date), such as HON's genuine Solstice spin-off of 2025-10-30.

Target API (asof.store.spinoffs):

    SPINOFF_REJECTIONS_FILE: Path      <repo>/configs/spinoff_rejections.yaml
        same format as configs/split_rejections.yaml (``key / reason / added / by``, key
        ``PARENT:YYYY-MM-DD``) and read by the same loader,
        asof.store.market.load_split_rejections.
    visible_spinoffs(con, root, as_of, symbols=None, *, rejected=None)
    spinoff_events(con, root, as_of, parents=None, *, rejected=None, max_child_delay=20)
        rejected=None reads SPINOFF_REJECTIONS_FILE (looked up at call time, so a test can point
        it elsewhere); a missing file is FileNotFoundError; an explicit collection (``()``
        included) replaces it. Every record of a rejected (parent, ex_date) is left out, the
        ex-date itself included: the rejection is a data correction, not dated.

Fake data provenance:
- HON -> HONA 2026-06-29, ratio 0.5: HON closed 227.80 against 232.21 the session before and
  HONA's first bar closed 220.19 the same day, a total return of +45.5% (real, bar store probe
  2026-10-27). The same day carries the rejected false HON 2:1 split record.
- HON -> SOLS 2025-10-30, ratio 0.25: a genuine spin-off (real record; never rejected).
- available_at of a record is 00:00 New York of the ex-date (ingest rule).

Real-data checks: tests/store/test_golden_spinoffs.py (HON 2026-06-29 absent at NOW, present
with ``rejected=()``; HON 2025-10-30 present) and the audit's spin-off check (no row for the
rejected key).
"""

from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import duckdb
import pyarrow as pa
import pytest
import yaml

from asof.store import spinoffs
from asof.store.market import load_split_rejections, market_root
from asof.store.spinoffs import SPINOFF_REJECTIONS_FILE, spinoff_events, visible_spinoffs
from tests.store.test_market import Seg, avail, known, write_master
from tests.store.test_spinoffs import flat, px, spin_row, write_bars, write_other_rows
from tests.store.test_universe import trading_days

REPO = Path(__file__).resolve().parents[2]
HON_DAY = date(2026, 6, 29)
SOLS_DAY = date(2025, 10, 30)
LATE = datetime(2026, 10, 27, tzinfo=UTC)
SHIPPED = frozenset({("HON", HON_DAY)})


def hona() -> dict:
    return spin_row("HON", "HONA", HON_DAY, 0.5)


def sols() -> dict:
    return spin_row("HON", "SOLS", SOLS_DAY, 0.25)


def pairs(table: pa.Table) -> list[tuple[str, date]]:
    return list(zip(table["parent_symbol"].to_pylist(), table["ex_date"].to_pylist(), strict=True))


@pytest.fixture
def con() -> Iterator[duckdb.DuckDBPyConnection]:
    connection = duckdb.connect()
    yield connection
    connection.close()


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return market_root(tmp_path)


def hon_world(root: Path) -> None:
    """HON around 2026-06-29 with the real closes, HONA from that day, the two records."""
    days = trading_days(date(2026, 6, 1), date(2026, 7, 31))
    bars = [px("HON", d, 232.21 if d < HON_DAY else 227.80) for d in days]
    bars += flat("HONA", [d for d in days if d >= HON_DAY], 220.19)
    write_bars(root, bars)
    write_master(
        root,
        [
            Seg("S_HON", "HON"),
            Seg("S_HONA", "HONA", valid_from=HON_DAY, available_at=known(HON_DAY)),
        ],
    )
    write_other_rows(root, [hona()])


# =============================================================================================
# normal path
# =============================================================================================


def test_shipped_file_lists_exactly_hon_2026_06_29() -> None:
    assert SPINOFF_REJECTIONS_FILE == REPO / "configs" / "spinoff_rejections.yaml"
    assert load_split_rejections(SPINOFF_REJECTIONS_FILE) == SHIPPED
    entries = yaml.safe_load(SPINOFF_REJECTIONS_FILE.read_text())
    assert [e["key"] for e in entries] == ["HON:2026-06-29"]
    assert all(str(e["reason"]).strip() and str(e["by"]).strip() for e in entries)


def test_default_reads_the_shipped_file(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    write_other_rows(root, [hona(), sols(), spin_row("GE", "GEV", date(2024, 4, 2), 0.25)])

    default = visible_spinoffs(con, root, LATE)
    unrejected = visible_spinoffs(con, root, LATE, rejected=())

    assert ("HON", HON_DAY) not in pairs(default)
    assert ("HON", SOLS_DAY) in pairs(default) and ("GE", date(2024, 4, 2)) in pairs(default)
    assert len(pairs(unrejected)) == 3


def test_explicit_rejections_replace_the_shipped_ones(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    write_other_rows(root, [hona(), sols()])

    table = visible_spinoffs(con, root, LATE, rejected={("HON", SOLS_DAY)})

    assert pairs(table) == [("HON", HON_DAY)]


def test_rejected_event_is_not_an_event(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    hon_world(root)

    assert spinoff_events(con, root, LATE).num_rows == 0
    [row] = spinoff_events(con, root, LATE, rejected=()).to_pylist()
    # unrejected, the prices still disprove it: a restart, never an adjust (user rule 2026-10-27)
    assert (row["status"], row["reason"]) == ("restart", "factor_disproved")
    assert row["total_return"] == pytest.approx((227.80 + 0.5 * 220.19) / 232.21 - 1)
    assert row["total_return"] == pytest.approx(0.455, abs=1e-3)


# =============================================================================================
# boundaries
# =============================================================================================


def test_every_child_of_a_rejected_day_goes(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    write_other_rows(root, [hona(), spin_row("HON", "HONB", HON_DAY, 0.1), sols()])

    assert pairs(visible_spinoffs(con, root, LATE)) == [("HON", SOLS_DAY)]


def test_same_parent_other_day_and_other_parent_same_day_stay(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    write_other_rows(root, [hona(), sols(), spin_row("HONE", "HX", HON_DAY, 0.5)])

    assert pairs(visible_spinoffs(con, root, LATE)) == [("HON", SOLS_DAY), ("HONE", HON_DAY)]


def test_symbols_filter_and_rejection_together(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    write_other_rows(root, [hona(), sols(), spin_row("GE", "GEV", date(2024, 4, 2), 0.25)])

    assert pairs(visible_spinoffs(con, root, LATE, ["HON"])) == [("HON", SOLS_DAY)]
    assert pairs(visible_spinoffs(con, root, LATE, ["GE"])) == [("GE", date(2024, 4, 2))]


def test_a_missing_shipped_file_is_an_error(
    con: duckdb.DuckDBPyConnection, root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hon_world(root)
    monkeypatch.setattr(spinoffs, "SPINOFF_REJECTIONS_FILE", tmp_path / "absent.yaml")

    with pytest.raises(FileNotFoundError):
        visible_spinoffs(con, root, LATE)
    with pytest.raises(FileNotFoundError):
        spinoff_events(con, root, LATE)
    assert visible_spinoffs(con, root, LATE, rejected=()).num_rows == 1


def test_the_default_follows_the_module_attribute(
    con: duckdb.DuckDBPyConnection, root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # a reviewed file rejecting SOLS instead (constructed): the default now drops SOLS, not HONA
    path = tmp_path / "spinoff_rejections.yaml"
    path.write_text(
        '- key: "HON:2025-10-30"\n  reason: constructed\n  added: 2026-10-27\n  by: greyson\n'
    )
    write_other_rows(root, [hona(), sols()])
    monkeypatch.setattr(spinoffs, "SPINOFF_REJECTIONS_FILE", path)

    assert pairs(visible_spinoffs(con, root, LATE)) == [("HON", HON_DAY)]


# =============================================================================================
# leakage: the rejection is not dated, and it does not disturb the as_of cut
# =============================================================================================


@pytest.mark.parametrize("hours", [0, 12, 20, 72])
def test_rejected_record_is_absent_from_its_ex_date_on(
    con: duckdb.DuckDBPyConnection, root: Path, hours: int
) -> None:
    # from 00:00 New York of the ex-date (the record becomes knowable) a reader sees nothing
    hon_world(root)
    as_of = known(HON_DAY) + timedelta(hours=hours)

    assert visible_spinoffs(con, root, as_of).num_rows == 0
    assert spinoff_events(con, root, as_of).num_rows == 0
    assert visible_spinoffs(con, root, as_of, rejected=()).num_rows == 1


def test_rejection_does_not_reveal_or_hide_anything_else_across_as_of(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    future = spin_row("ZZZ", "ZZC", date(2026, 11, 2), 0.5)
    write_other_rows(root, [hona(), sols(), future])
    as_of = avail(HON_DAY)

    table = visible_spinoffs(con, root, as_of)

    assert all(a <= as_of for a in table["available_at"].to_pylist())
    assert pairs(table) == [("HON", SOLS_DAY)]
    assert pairs(table) == pairs(visible_spinoffs(con, root, as_of, rejected=SHIPPED))
