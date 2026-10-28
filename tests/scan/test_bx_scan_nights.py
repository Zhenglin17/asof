# Guards the per-night scan the B-Xtrender replay reads its entries and exit flags from against
# disagreeing with the tested range scan, dropping the panel row of a held security that has
# left tonight's tier (its exit flags would go missing), skipping a night the market filter
# closes, and reading bars after its window's end.
"""API under test (asof.scan.bx_trender):

    scan_nights(con, root, start, end, cfg) -> list[tuple[date, list[BxCandidate], pd.DataFrame]]
        every session start..end in order, with that night's ranked candidates (an empty list on
        a night without one or one the market filter fails) and that night's panel rows: a row
        for every security that was in the tier on any night of the window, with the same
        late-recorded-master recompute as ``scan_range``.
        scan_range(...) == {d: c for d, c, _ in scan_nights(...) if c}

World: tests/scan/test_bx_trender.range_world (provenance in tests/scan/bx_world.py).

Assumption to confirm on the real store (a standard case): a liquid security that leaves the
tier while held keeps a defined row on the nights after; e.g. run scan_nights over a dev month
and, for each security in the tier on its first night, count nights without a row while it
still has a bar (expected 0).
"""

from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import pytest
import yaml

from asof.scan.bx_trender import bx_panel, load_scan_config, scan_nights, scan_range
from asof.store.market import market_root
from tests.features.test_bx import BX_COLUMNS, NEW_COLUMNS, assert_columns_close
from tests.scan.bx_world import CAL, close_of
from tests.scan.test_bx_trender import (
    RANGE_END,
    RANGE_START,
    TEST_SCAN,
    assert_same_candidates,
    panel_rows,
    range_world,
    tier_at,
)

WINDOW = [d for d in CAL if RANGE_START <= d <= RANGE_END]
LOOKBACK = "universe.lookback_sessions=5"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return market_root(tmp_path)


@pytest.fixture
def cfg_file(tmp_path: Path) -> Path:
    path = tmp_path / "scan.yaml"
    path.write_text(yaml.safe_dump({**TEST_SCAN, "market_filter": None}))
    return path


def by_day(nights: list[tuple[date, list, pd.DataFrame]]) -> dict[date, pd.DataFrame]:
    return {d: rows for d, _, rows in nights}


# =============================================================================================
# normal path
# =============================================================================================


def test_scan_nights_gives_every_session_and_agrees_with_scan_range(
    con: duckdb.DuckDBPyConnection, root: Path, cfg_file: Path
) -> None:
    range_world(root)
    cfg = load_scan_config(cfg_file, [LOOKBACK])

    nights = scan_nights(con, root, RANGE_START, RANGE_END, cfg)
    ranged = scan_range(con, root, RANGE_START, RANGE_END, cfg)

    assert [d for d, _, _ in nights] == WINDOW
    assert sum(len(c) for _, c, _ in nights) > 0  # fixture: the window has candidates
    assert {d: c for d, c, _ in nights if c}.keys() == ranged.keys()
    for d, found, _ in nights:
        assert_same_candidates(found, ranged.get(d, []))


def test_rows_keep_a_security_that_has_left_the_tier(
    con: duckdb.DuckDBPyConnection, root: Path, cfg_file: Path
) -> None:
    # INOUT is liquid only on CAL[300 .. 306]: in the tier for part of the window
    range_world(root)
    cfg = load_scan_config(cfg_file, [LOOKBACK])
    tiers = {d: {r["security_id"] for r in tier_at(con, root, d, cfg).to_pylist()} for d in WINDOW}
    ever = set().union(*tiers.values())
    out_nights = [d for d in WINDOW if "S_INOUT" not in tiers[d]]
    assert "S_INOUT" in ever and out_nights  # fixture guard

    rows = by_day(scan_nights(con, root, RANGE_START, RANGE_END, cfg))

    for d in WINDOW:
        assert not rows[d].duplicated(["security_id", "session_date"]).any()
        assert set(rows[d]["session_date"]) <= {d}
        assert set(rows[d]["security_id"]) >= ever - {"S_KID2"}, d  # every one trades each night
    last_out = out_nights[-1]
    inout = panel_rows(rows[last_out], "S_INOUT")
    nightly = panel_rows(bx_panel(con, root, close_of(last_out), ["S_INOUT"]), "S_INOUT", last_out)
    assert_columns_close(inout, nightly.tail(1).reset_index(drop=True), BX_COLUMNS + NEW_COLUMNS)


# =============================================================================================
# boundaries
# =============================================================================================


def test_a_closed_market_filter_keeps_the_night_with_no_candidate(
    con: duckdb.DuckDBPyConnection, root: Path, cfg_file: Path
) -> None:
    # an index symbol with no bars reads as a failed filter (market_ok): no entries, but the
    # nights and their rows are still there for the positions already held
    range_world(root)
    cfg = load_scan_config(cfg_file, [LOOKBACK, "market_filter={symbol: NOIDX, sma_sessions: 5}"])

    nights = scan_nights(con, root, RANGE_START, RANGE_END, cfg)

    assert [d for d, _, _ in nights] == WINDOW
    assert all(found == [] for _, found, _ in nights)
    assert all(not rows.empty for _, _, rows in nights)
    assert scan_range(con, root, RANGE_START, RANGE_END, cfg) == {}


def test_a_one_session_window(con: duckdb.DuckDBPyConnection, root: Path, cfg_file: Path) -> None:
    range_world(root)
    cfg = load_scan_config(cfg_file, [LOOKBACK])
    [(d, found, rows)] = scan_nights(con, root, RANGE_END, RANGE_END, cfg)
    assert d == RANGE_END
    assert_same_candidates(found, scan_range(con, root, RANGE_END, RANGE_END, cfg).get(d, []))
    assert set(rows["session_date"]) == {RANGE_END}


# =============================================================================================
# leakage
# =============================================================================================


def test_scan_nights_does_not_read_past_its_end(
    con: duckdb.DuckDBPyConnection, tmp_path: Path, cfg_file: Path
) -> None:
    full, cut = market_root(tmp_path / "full"), market_root(tmp_path / "cut")
    range_world(full)
    range_world(cut, cut=RANGE_END)
    cfg = load_scan_config(cfg_file, [LOOKBACK])

    a = scan_nights(con, full, RANGE_START, RANGE_END, cfg)
    b = scan_nights(con, cut, RANGE_START, RANGE_END, cfg)

    assert [d for d, _, _ in a] == [d for d, _, _ in b]
    for (_, ca, ra), (_, cb, rb) in zip(a, b, strict=True):
        assert_same_candidates(ca, cb)
        for sid in sorted(set(ra["security_id"]) | set(rb["security_id"])):
            assert_columns_close(
                panel_rows(ra, sid), panel_rows(rb, sid), BX_COLUMNS + NEW_COLUMNS, 0.0
            )
