# Guards the B-Xtrender THT entry filter against look-ahead -- a later bar or a split record
# not yet on file deciding tonight's colour or candidates -- and against the THT condition
# leaking into the plain random-entry pool.
"""Target API (asof.scan.bx_trender, acceptance point v2-b-2; defaults keep today's behaviour):

    ThtFilter(_Section): length: int, min_crossings: int, median_window: int | None
    BxScanConfig.tht_filter: ThtFilter | None = None
    load_scan_config: a key whose value is null takes a YAML flow mapping, e.g.
        tht_filter={length: 33, min_crossings: 20, median_window: null}
    bx_panel(..., tht: ThtFilter | None = None): with tht, a ``tht_color`` column from
        asof.features.tht.tht_features on the same split-adjusted bars and spin-off events,
        same rows and order; without it, no such column.
    signals: with cfg.tht_filter, a candidate also needs tht_color == "green" that session.
    pool_range(con, root, start, end, cfg, *, pool_tht=False): pool_tht=True also requires
        green; the default is the plain pool whatever cfg.tht_filter says.

Interface points this file pins down where the spec left a choice:
- Candidates removed by a filter leave no gap in the ranks: the survivors rank 1..n in the
  plain order (the filters act before ranking, as daily_min does).
- ``pool`` (one night) and ``scan_and_pool``'s pool stay the plain pool with tht_filter set.

Fake data:
- Stocks: tests/scan/bx_world.py (walk prices, NYSE calendar, availability rules; provenance
  there). Seeds 101 / 202 / 303, volumes 5M shares a day (~$250M, inside the $50M tier).
- THT reference colours come from asof.features.tht.tht_features run directly on the walk
  prices (the feature has its own tests and golden cases; here only the wiring is under test).

Assumptions a real-data check must confirm (the fixtures cannot):
- THT on real daily bars reproduces the chart colour: tests/features/test_golden_tht.py.
"""

import dataclasses
from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd
import pyarrow as pa
import pytest
import yaml

from asof.features.bx import adjust_for_splits
from asof.features.tht import tht_features
from asof.scan.bx_trender import (
    BxCandidate,
    BxScanConfig,
    bx_panel,
    load_scan_config,
    pool,
    pool_range,
    scan,
    scan_range,
    signals,
)
from asof.store.market import market_root, visible_bars, visible_splits
from asof.store.spinoffs import spinoff_events
from asof.store.universe import liquid_securities
from tests.scan.bx_world import CAL, Sec, after, build, close_of, spin_row, split_row
from tests.scan.test_bx_trender import (
    AAA,
    BBB,
    REAL_SCAN_YAML,
    TEST_SCAN,
    assert_same_candidates,
    ids,
    panel_rows,
)

ET = ZoneInfo("America/New_York")
THT: dict[str, Any] = {"length": 33, "min_crossings": 20, "median_window": None}
THT_FLOW = "tht_filter={length: 33, min_crossings: 20, median_window: null}"

CCC = Sec("S_CCC", "CCC", seed=303)
THT_SECS = [AAA, BBB, CCC]


def cfg_of(**extra: Any) -> BxScanConfig:
    return BxScanConfig.model_validate({**TEST_SCAN, **extra})


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return market_root(tmp_path)


def as_day(value: Any) -> date:
    return value if type(value) is date else date.fromisoformat(str(value)[:10])


def reranked(found: Sequence[BxCandidate], keep: set[str]) -> list[BxCandidate]:
    """The plain candidates kept by a filter, ranked 1..n in their plain order."""
    kept = [c for c in found if c.security_id in keep]
    return [dataclasses.replace(c, rank=i) for i, c in enumerate(kept, start=1)]


# =============================================================================================
# config
# =============================================================================================


def test_the_tht_filter_defaults_to_none(tmp_path: Path) -> None:
    path = tmp_path / "scan.yaml"
    path.write_text(yaml.safe_dump(TEST_SCAN))

    cfg = load_scan_config(path)

    assert cfg.tht_filter is None


def test_a_null_tht_filter_takes_a_flow_mapping_from_overrides(tmp_path: Path) -> None:
    path = tmp_path / "scan.yaml"
    path.write_text(yaml.safe_dump({**TEST_SCAN, "tht_filter": None}))

    cfg = load_scan_config(path, [THT_FLOW])

    assert cfg.tht_filter is not None
    assert cfg.tht_filter.model_dump() == THT
    # a window is an int when given
    cfg = load_scan_config(path, ["tht_filter={length: 20, min_crossings: 5, median_window: 9}"])
    assert cfg.tht_filter is not None and cfg.tht_filter.median_window == 9


def test_a_setting_under_a_null_tht_filter_cannot_be_set_alone(tmp_path: Path) -> None:
    path = tmp_path / "scan.yaml"
    path.write_text(yaml.safe_dump({**TEST_SCAN, "tht_filter": None}))

    with pytest.raises(ValueError):
        load_scan_config(path, ["tht_filter.length=33"])


@pytest.mark.parametrize(
    "override",
    [
        "tht_filter={length: 33, min_crossings: 20}",  # a missing setting
        "tht_filter={length: 33, min_crossings: 20, median_window: null, extra: 1}",
        "tht_filter=33",  # not a mapping
        "tht_filter={length: 33, min_crossings: 20.5, median_window: null}",  # not an int
    ],
)
def test_an_incomplete_or_unknown_filter_mapping_is_refused(tmp_path: Path, override: str) -> None:
    path = tmp_path / "scan.yaml"
    path.write_text(yaml.safe_dump({**TEST_SCAN, "tht_filter": None}))

    with pytest.raises(ValueError):
        load_scan_config(path, [override])


def test_the_strategy_scan_yaml_names_the_tht_filter_null_and_still_loads_to_s1() -> None:
    raw = yaml.safe_load(REAL_SCAN_YAML.read_text())
    assert "tht_filter" in raw and raw["tht_filter"] is None

    cfg = load_scan_config(REAL_SCAN_YAML)

    # S1 as pre-registered (the comment block of that file), with the THT filter off
    assert cfg.model_dump(mode="json") == {
        "variant": "mid",
        "universe": {"min_adv_usd": 50_000_000.0, "lookback_sessions": 20, "classes": ["common"]},
        "bx": {"l1": 5, "l2": 20, "l3": 5},
        "warmup": {"daily": 40, "weekly": 44, "monthly": 36},
        "daily_min": 0.0,
        "require_monthly_up": False,
        "market_filter": None,
        "tht_filter": None,
    }
    # and it can be switched on from there
    on = load_scan_config(REAL_SCAN_YAML, [THT_FLOW])
    assert on.tht_filter is not None


def plain_tier(con: duckdb.DuckDBPyConnection, root: Path, as_of: datetime) -> pa.Table:
    u = cfg_of().universe
    return liquid_securities(
        con, root, as_of, min_adv_usd=u.min_adv_usd, lookback_sessions=u.lookback_sessions,
        classes=u.classes,
    )  # fmt: skip


# =============================================================================================
# THT: bx_panel
# =============================================================================================


def norm(value: Any) -> str | None:
    """A colour, or None for a missing one (None or NaN)."""
    return value if isinstance(value, str) else None


def colours_of(frame: pd.DataFrame) -> list[str | None]:
    return [norm(v) for v in frame["tht_color"]]


def reference_colours(secs: Sequence[Sec], **params: Any) -> dict[tuple[str, date], str | None]:
    """(security_id, session) -> colour of tht_features on the walk prices (adjusted)."""
    p = {**THT, **params}
    frame = tht_features(
        pd.concat([s.prices() for s in secs], ignore_index=True),
        length=p["length"], min_crossings=p["min_crossings"], median_window=p["median_window"],
    )  # fmt: skip
    return {
        (str(r["security_id"]), as_day(r["session_date"])): norm(r["tht_color"])
        for r in frame.to_dict("records")
    }


def tht_cfg(**params: Any):  # noqa: ANN201 - the model class is under test
    from asof.scan.bx_trender import ThtFilter

    return ThtFilter(**{**THT, **params})


SPLIT_EX = CAL[230]
SPL = Sec("S_SPL", "SPL", seed=101, raw_scale=((None, SPLIT_EX, 2.0),))  # a 2:1 in raw prices


def test_bx_panel_without_tht_has_no_colour_column(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    build(root, THT_SECS)
    panel = bx_panel(con, root, close_of(CAL[240]), ["S_AAA", "S_BBB"])
    assert "tht_color" not in panel.columns
    panel = bx_panel(con, root, close_of(CAL[240]), ["S_AAA", "S_BBB"], tht=None)
    assert "tht_color" not in panel.columns


@pytest.mark.parametrize("params", [{}, {"length": 20, "min_crossings": 10, "median_window": 15}])
def test_bx_panel_adds_the_tht_colour_of_the_split_adjusted_bars(
    con: duckdb.DuckDBPyConnection, root: Path, params: dict[str, Any]
) -> None:
    build(root, [AAA, SPL], splits=[split_row("SPL", SPLIT_EX, 1.0, 2.0)])
    as_of = close_of(CAL[300])
    plain = bx_panel(con, root, as_of, ["S_AAA", "S_SPL"])

    got = bx_panel(con, root, as_of, ["S_AAA", "S_SPL"], tht=tht_cfg(**params))

    # same rows in the same order, every other column unchanged
    pd.testing.assert_frame_equal(got.drop(columns=["tht_color"]), plain)
    ref = reference_colours([AAA, SPL], **params)
    colours = [
        ref[(str(s), as_day(d))]
        for s, d in zip(got["security_id"], got["session_date"], strict=True)
    ]
    assert colours_of(got) == colours
    assert {"green", "red"} <= set(colours)  # fixture: both colours occur


def tht_of_store(con: duckdb.DuckDBPyConnection, root: Path, as_of: datetime, symbols: list[str],
                 sid: str, *, events: bool) -> list[str | None]:  # fmt: skip
    """The colour written out from the store as bx_panel reads it (spec)."""
    bars = visible_bars(con, root, "1Day", as_of, symbols=symbols).to_pandas()
    daily = adjust_for_splits(bars, visible_splits(con, root, as_of, symbols).to_pandas())
    ev = spinoff_events(con, root, as_of, symbols).to_pandas()
    frame = tht_features(daily, spinoffs=ev if events else None, **THT)
    return colours_of(pd.DataFrame(frame[frame["security_id"] == sid]))


def test_bx_panel_tht_reads_the_spin_off_events_as_bx_does(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    ex = CAL[230]
    prev = AAA.prices().set_index("session_date")["close"][after(ex, -1)]
    parent = dataclasses.replace(AAA, raw_scale=((ex, None, 0.8),))
    kid = Sec("S_KID", "KID", seed=909, start=0.4 * prev, first=CAL.index(ex))
    build(root, [parent, BBB, kid], spinoffs=[spin_row("AAA", "KID", ex, 0.5)])
    as_of = close_of(CAL[-1])
    with_events = tht_of_store(con, root, as_of, ["AAA", "KID"], "S_AAA", events=True)
    raw = tht_of_store(con, root, as_of, ["AAA", "KID"], "S_AAA", events=False)
    assert with_events != raw  # fixture: the adjustment moves some colour

    got = panel_rows(bx_panel(con, root, as_of, ["S_AAA"], tht=tht_cfg()), "S_AAA")

    assert colours_of(got) == with_events


def test_the_tht_colour_ignores_later_bars(con: duckdb.DuckDBPyConnection, tmp_path: Path) -> None:
    full, cut = market_root(tmp_path / "full"), market_root(tmp_path / "cut")
    night = CAL[240]
    build(full, THT_SECS)
    build(cut, THT_SECS, cut=night)
    ids_ = [s.security_id for s in THT_SECS]

    a = bx_panel(con, full, close_of(night), ids_, tht=tht_cfg())
    b = bx_panel(con, cut, close_of(night), ids_, tht=tht_cfg())

    assert as_day(a["session_date"].max()) == night
    assert colours_of(a) == colours_of(b)
    # and the full-history colour of those sessions is the same (it runs forward only)
    later = bx_panel(con, full, close_of(CAL[-1]), ids_, tht=tht_cfg())
    later = later[[as_day(d) <= night for d in later["session_date"]]]
    assert colours_of(later) == colours_of(a)


def test_a_split_record_not_yet_visible_leaves_the_raw_colours(
    con: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    # the prices halved on SPLIT_EX; the record goes on file only 15 sessions later
    night, late = CAL[240], close_of(CAL[245])
    with_rec, without = market_root(tmp_path / "with"), market_root(tmp_path / "without")
    build(with_rec, [AAA, SPL], splits=[split_row("SPL", SPLIT_EX, 1.0, 2.0, available_at=late)])
    build(without, [AAA, SPL])

    a = panel_rows(bx_panel(con, with_rec, close_of(night), ["S_SPL"], tht=tht_cfg()), "S_SPL")
    b = panel_rows(bx_panel(con, without, close_of(night), ["S_SPL"], tht=tht_cfg()), "S_SPL")

    assert colours_of(a) == colours_of(b)
    # once on file the record does change those sessions' colours (the test can see it)
    c = panel_rows(bx_panel(con, with_rec, late, ["S_SPL"], tht=tht_cfg()), "S_SPL", hi=night)
    assert colours_of(c) != colours_of(a)


# =============================================================================================
# THT: signals and the pools
# =============================================================================================

T_START, T_END = CAL[210], CAL[250]


@pytest.fixture
def tht_root(root: Path) -> Path:
    build(root, [*THT_SECS, SPL], splits=[split_row("SPL", SPLIT_EX, 1.0, 2.0)])
    return root


def window() -> list[date]:
    return [d for d in CAL if T_START <= d <= T_END]


def plain_candidate_night(
    con: duckdb.DuckDBPyConnection, root: Path, colour: str | None
) -> tuple[date, str]:
    """A night and a plain candidate whose reference colour that night is ``colour``."""
    ref = reference_colours([*THT_SECS, SPL])
    found = scan_range(con, root, T_START, T_END, cfg_of())
    for d in window():
        for c in found.get(d, []):
            if ref[(c.security_id, d)] == colour:
                return d, c.security_id
    raise AssertionError(f"fixture: no plain candidate is {colour} on a night of the window")


def rows_of(con: duckdb.DuckDBPyConnection, root: Path, day: date) -> tuple[pd.DataFrame, pa.Table]:
    """The plain panel rows of ``day`` (no tht_color column) and that night's plain tier."""
    tier = plain_tier(con, root, close_of(day))
    panel = bx_panel(con, root, close_of(day), tier.column("security_id").to_pylist())
    return pd.DataFrame(panel[[as_day(d) == day for d in panel["session_date"]]]), tier


@pytest.mark.parametrize("colour", ["green", "red", None])
def test_a_candidate_needs_green_that_session(
    con: duckdb.DuckDBPyConnection, tht_root: Path, colour: str | None
) -> None:
    day, sid = plain_candidate_night(con, tht_root, "green")
    rows, tier = rows_of(con, tht_root, day)
    plain = signals(rows, tier, cfg_of())
    assert sid in ids(plain)  # fixture
    # every other row green: only this security's colour differs between the cases
    rows["tht_color"] = [colour if s == sid else "green" for s in rows["security_id"]]

    got = signals(rows, tier, cfg_of(tht_filter=THT))

    keep = set(ids(plain)) - (set() if colour == "green" else {sid})
    assert_same_candidates(got, reranked(plain, keep))


def test_without_the_filter_the_colour_is_ignored(
    con: duckdb.DuckDBPyConnection, tht_root: Path
) -> None:
    day, sid = plain_candidate_night(con, tht_root, "green")
    rows, tier = rows_of(con, tht_root, day)
    plain = signals(rows, tier, cfg_of())
    rows["tht_color"] = "red"

    got = signals(rows, tier, cfg_of())

    assert sid in ids(got)
    assert_same_candidates(got, plain)


def test_the_tht_scan_is_the_plain_scan_keeping_green(
    con: duckdb.DuckDBPyConnection, tht_root: Path
) -> None:
    for colour in ("green", "red", None):  # fixture: every case occurs among plain candidates
        plain_candidate_night(con, tht_root, colour)
    ref = reference_colours([*THT_SECS, SPL])
    plain = scan_range(con, tht_root, T_START, T_END, cfg_of())
    cfg = cfg_of(tht_filter=THT)

    got = scan_range(con, tht_root, T_START, T_END, cfg)

    for d in window():
        green = {sid for (sid, day), c in ref.items() if day == d and c == "green"}
        expected = reranked(plain.get(d, []), green)
        assert_same_candidates(got.get(d, []), expected)
        assert_same_candidates(scan(con, tht_root, close_of(d), cfg), expected)


def test_too_few_crossings_signal_nothing_but_leave_the_plain_pool(
    con: duckdb.DuckDBPyConnection, tht_root: Path
) -> None:
    never = cfg_of(tht_filter={**THT, "min_crossings": 10_000})  # no colour is ever defined

    assert scan_range(con, tht_root, T_START, T_END, never) == {}
    assert pool_range(con, tht_root, T_START, T_END, never) == pool_range(
        con, tht_root, T_START, T_END, cfg_of()
    )
    assert all(
        not p for p in pool_range(con, tht_root, T_START, T_END, never, pool_tht=True).values()
    )


def test_the_default_pool_ignores_the_tht_filter(
    con: duckdb.DuckDBPyConnection, tht_root: Path
) -> None:
    cfg = cfg_of(tht_filter=THT)
    plain = pool_range(con, tht_root, T_START, T_END, cfg_of())

    assert pool_range(con, tht_root, T_START, T_END, cfg) == plain
    assert pool_range(con, tht_root, T_START, T_END, cfg, pool_tht=False) == plain
    for d in (T_START, CAL[230], T_END):
        assert pool(con, tht_root, close_of(d), cfg) == plain[d]


def test_the_tht_pool_keeps_the_green_members_of_the_plain_pool(
    con: duckdb.DuckDBPyConnection, tht_root: Path
) -> None:
    cfg = cfg_of(tht_filter=THT)
    ref = reference_colours([*THT_SECS, SPL])
    plain = pool_range(con, tht_root, T_START, T_END, cfg)

    got = pool_range(con, tht_root, T_START, T_END, cfg, pool_tht=True)

    smaller = 0
    for d in window():
        assert got[d] == frozenset(s for s in plain[d] if ref[(s, d)] == "green"), d
        smaller += got[d] != plain[d]
    assert smaller > 0 and any(got.values())  # fixture: both sides of the condition occur


def test_every_tht_candidate_is_in_both_pools(
    con: duckdb.DuckDBPyConnection, tht_root: Path
) -> None:
    cfg = cfg_of(tht_filter=THT)
    found = scan_range(con, tht_root, T_START, T_END, cfg)
    plain = pool_range(con, tht_root, T_START, T_END, cfg)
    green = pool_range(con, tht_root, T_START, T_END, cfg, pool_tht=True)
    assert found  # fixture

    for d, cands in found.items():
        assert set(ids(cands)) <= plain[d] and set(ids(cands)) <= green[d], d


def test_the_tht_pool_of_a_range_is_each_nights_own(
    con: duckdb.DuckDBPyConnection, tht_root: Path
) -> None:
    # the range is read at T_END's close, with the split (CAL[230]) on file and bars after each
    # night visible: every night must still be what that night alone saw
    cfg = cfg_of(tht_filter=THT)
    ranged = pool_range(con, tht_root, T_START, T_END, cfg, pool_tht=True)
    found = scan_range(con, tht_root, T_START, T_END, cfg)
    for d in window():
        assert ranged[d] == pool_range(con, tht_root, d, d, cfg, pool_tht=True)[d], d
        assert_same_candidates(found.get(d, []), scan(con, tht_root, close_of(d), cfg))


def test_the_tht_candidates_ignore_later_bars(
    con: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    full, cut = market_root(tmp_path / "full"), market_root(tmp_path / "cut")
    build(full, THT_SECS)
    cfg = cfg_of(tht_filter=THT)
    night = next(d for d in window() if scan_range(con, full, d, d, cfg))  # a night with one
    build(cut, THT_SECS, cut=night)

    assert_same_candidates(
        scan(con, full, close_of(night), cfg), scan(con, cut, close_of(night), cfg)
    )
    assert pool_range(con, full, night, night, cfg, pool_tht=True) == pool_range(
        con, cut, night, night, cfg, pool_tht=True
    )


def test_fixture_the_reference_colours_are_green_red_or_missing() -> None:
    # tht_features stores a missing colour as NaN in a str column (pandas 3); norm() maps it
    # to None, and the implementation must treat NaN and None alike as "no colour"
    ref = reference_colours(THT_SECS)
    assert set(ref.values()) == {"green", "red", None}
