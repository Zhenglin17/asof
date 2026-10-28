# Guards the B-Xtrender entry scan against look-ahead (bars, splits or spin-off records not yet
# visible, a running week or month used as finished, a one-shot range scan disagreeing with the
# nightly one) and against signalling outside its universe or before its warm-up.
"""Target API (asof.scan.bx_trender, new module):

    BxScanConfig            pydantic, extra=forbid, frozen: variant ("mid" | "long"),
                            universe {min_adv_usd, lookback_sessions, classes},
                            bx {l1, l2, l3}, warmup {daily, weekly, monthly}
    load_scan_config(path, overrides=()) -> BxScanConfig
                            ``path`` is the scan.yaml file; overrides "key=value" or
                            "section.key=value", the value read as YAML.
    bx_panel(con, root, as_of, security_ids) -> pd.DataFrame
    BxCandidate             frozen dataclass (fields below)
    signals(rows, tier, cfg) -> list[BxCandidate]
    scan(con, root, as_of, cfg) -> list[BxCandidate]
    pool(con, root, as_of, cfg) -> frozenset[str]
    scan_range(con, root, start, end, cfg) -> dict[date, list[BxCandidate]]

Signal on session s (mid: state = weekly, long: state = monthly; the trigger is daily): in that
night's liquid tier with the config's classes, state_bx > state_bx_prev, bx_daily >
bx_daily_prev, daily_bars >= warmup.daily, <state>_bars >= warmup.<state>; any NaN among the four
values -> no signal and not in the pool. Rank: state_bx - state_bx_prev descending, then
security_id.

Expected values come from ``tests.features.test_bx.reference`` (period_bars + bx_histogram, no
store, no scanner code), over the walk prices each fixture security is built from.

Interface points this file pins down where the spec left a choice:
- ``bx_panel`` uses the default BX lengths 5 / 20 / 5; every config here keeps bx at 5 / 20 / 5.
- ``signals`` drops a tier row whose class is not in ``cfg.universe.classes`` even when the
  caller passed a wider tier.
- ``BxCandidate.symbol`` is the tier's symbol (the one in use at as_of).
- Ties in the rise take consecutive ranks (1, 2), ordered by security_id.
- ``scan_range`` may leave out sessions with no candidate; a missing key reads as [].

Fake data: tests/scan/bx_world.py (provenance there). No network.

Assumptions a real-data check must confirm:
- The fake calendar is weekdays minus full NYSE holidays, with no emergency closures; on the
  real store the calendar audit (I12) is what keeps a missing session from silently moving a
  weekly bar's complete_on.
- The tier, the panel and the candidate symbol agree on a real renamed liquid security: a
  standard case on the real store (a liquid rename with a pre-rename spin-off or split record)
  should be added to tests/store/test_golden_spinoffs.py when 2-c lands.
"""

import dataclasses
import math
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow as pa
import pytest
import yaml

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
from asof.store.market import market_root, visible_splits
from asof.store.spinoffs import spinoff_events
from asof.store.universe import liquid_securities
from tests.features.test_bx import BX_COLUMNS, NEW_COLUMNS, assert_columns_close, reference
from tests.scan.bx_world import (
    CAL,
    STATE,
    THIN_VOLUME,
    Sec,
    after,
    build,
    close_of,
    find_session,
    known,
    previous_friday,
    ref_defined,
    ref_frame,
    ref_row,
    spin_row,
    split_row,
)
from tests.store.test_universe import Seg, write_classes, write_master

REPO = Path(__file__).resolve().parents[2]
REAL_SCAN_YAML = REPO / "strategies" / "bx_trender" / "scan.yaml"
FLOAT_FIELDS = ("state_bx", "state_bx_prev", "daily_bx", "daily_bx_prev", "adv_usd")
TEST_SCAN: dict[str, Any] = {
    "variant": "mid",
    "universe": {"min_adv_usd": 50_000_000, "lookback_sessions": 20, "classes": ["common"]},
    "bx": {"l1": 5, "l2": 20, "l3": 5},
    "warmup": {"daily": 1, "weekly": 1, "monthly": 1},  # tests set warm-ups via overrides
}

AAA = Sec("S_AAA", "AAA", seed=101)
BBB = Sec("S_BBB", "BBB", seed=202)
BASE = [AAA, BBB]
BASE_REF = ref_frame(BASE)
MID_DAY = find_session(BASE_REF, {("S_AAA", "mid"): True, ("S_BBB", "mid"): False})
LONG_DAY = find_session(
    BASE_REF,
    {("S_AAA", "long"): True, ("S_AAA", "mid"): False, ("S_BBB", "long"): False},
)
WED_DAY = find_session(BASE_REF, {("S_AAA", "mid"): True}, weekday=2)


# =============================================================================================
# builders
# =============================================================================================


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return market_root(tmp_path)


@pytest.fixture
def cfg_file(tmp_path: Path) -> Path:
    path = tmp_path / "scan.yaml"
    path.write_text(yaml.safe_dump(TEST_SCAN))
    return path


def tier_at(con: duckdb.DuckDBPyConnection, root: Path, day: date, cfg: BxScanConfig) -> pa.Table:
    return liquid_securities(
        con,
        root,
        close_of(day),
        min_adv_usd=cfg.universe.min_adv_usd,
        lookback_sessions=cfg.universe.lookback_sessions,
        classes=cfg.universe.classes,
    )


def adv_of(tier: pa.Table, security_id: str) -> float:
    [row] = [r for r in tier.to_pylist() if r["security_id"] == security_id]
    return float(row["adv_usd"])


def ids(found: Sequence[BxCandidate]) -> list[str]:
    return [c.security_id for c in found]


def assert_candidate(
    c: BxCandidate,
    ref: pd.DataFrame,
    *,
    security_id: str,
    symbol: str,
    day: date,
    variant: str,
    rank: int,
    adv_usd: float,
    bx_since: date,
) -> None:
    r = ref_row(ref, security_id, day)
    state = STATE[variant]
    assert c.security_id == security_id
    assert c.symbol == symbol
    assert c.session == day
    assert c.variant == variant
    assert c.state_bx == pytest.approx(r[f"bx_{state}"], rel=1e-9, abs=1e-9)
    assert c.state_bx_prev == pytest.approx(r[f"bx_{state}_prev"], rel=1e-9, abs=1e-9)
    assert c.state_complete_on == r[f"{state}_complete_on"]
    assert c.daily_bx == pytest.approx(r["bx_daily"], rel=1e-9, abs=1e-9)
    assert c.daily_bx_prev == pytest.approx(r["bx_daily_prev"], rel=1e-9, abs=1e-9)
    assert c.rank == rank
    assert c.bx_since == bx_since
    assert c.state_bars == int(r[f"{state}_bars"])
    assert isinstance(c.state_bars, int)
    assert c.adv_usd == pytest.approx(adv_usd, rel=1e-12)


def assert_same_candidates(got: Sequence[BxCandidate], expected: Sequence[BxCandidate]) -> None:
    """Field by field; floats within 1e-9."""
    assert ids(got) == ids(expected)
    for g, e in zip(got, expected, strict=True):
        for f in dataclasses.fields(BxCandidate):
            a, b = getattr(g, f.name), getattr(e, f.name)
            if f.name in FLOAT_FIELDS:
                assert a == pytest.approx(b, rel=1e-9, abs=1e-9), (g.security_id, f.name)
            else:
                assert a == b, (g.security_id, f.name, a, b)


def panel_rows(panel: pd.DataFrame, security_id: str, hi: date | None = None) -> pd.DataFrame:
    mask = panel["security_id"] == security_id
    if hi is not None:
        mask &= panel["session_date"] <= hi
    part = pd.DataFrame(panel.loc[mask])
    return part.sort_values("session_date").reset_index(drop=True)


def as_day(value: Any) -> date:
    return value if type(value) is date else date.fromisoformat(str(value)[:10])


# =============================================================================================
# config
# =============================================================================================


def test_the_real_scan_yaml_loads_with_the_agreed_values() -> None:
    cfg = load_scan_config(REAL_SCAN_YAML)

    assert isinstance(cfg, BxScanConfig)
    assert cfg.variant == "mid"
    assert cfg.universe.min_adv_usd == 50_000_000
    assert cfg.universe.lookback_sessions == 20
    assert tuple(cfg.universe.classes) == ("common",)
    assert (cfg.bx.l1, cfg.bx.l2, cfg.bx.l3) == (5, 20, 5)
    assert (cfg.warmup.daily, cfg.warmup.weekly, cfg.warmup.monthly) == (40, 44, 36)
    assert load_scan_config(REAL_SCAN_YAML, ["variant=long"]).variant == "long"


def test_overrides_reach_nested_settings(cfg_file: Path) -> None:
    cfg = load_scan_config(cfg_file, ["warmup.daily=7", "universe.classes=[common, etf]"])
    assert cfg.warmup.daily == 7
    assert tuple(cfg.universe.classes) == ("common", "etf")


def test_the_config_is_frozen(cfg_file: Path) -> None:
    cfg = load_scan_config(cfg_file)
    with pytest.raises((TypeError, ValueError)):
        cfg.variant = "long"  # type: ignore[misc]


@pytest.mark.parametrize(
    "edit",
    [
        lambda raw: raw.update(colour="red"),
        lambda raw: raw["warmup"].update(hourly=3),
        lambda raw: raw.update(variant="short"),
    ],
    ids=["unknown_key", "unknown_nested_key", "unknown_variant"],
)
def test_unknown_keys_and_variants_are_rejected(tmp_path: Path, edit: Any) -> None:
    raw = yaml.safe_load(yaml.safe_dump(TEST_SCAN))
    edit(raw)
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        load_scan_config(path)


@pytest.mark.parametrize("override", ["variant=short", "colour=red", "warmup.hourly=3"])
def test_unknown_overrides_are_rejected(cfg_file: Path, override: str) -> None:
    with pytest.raises(ValueError):
        load_scan_config(cfg_file, [override])


# =============================================================================================
# normal path
# =============================================================================================


def test_panel_is_bx_features_from_each_securitys_first_bar(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    build(root, BASE)
    panel = bx_panel(con, root, close_of(MID_DAY), ["S_AAA", "S_BBB"])

    assert not panel.duplicated(["security_id", "session_date"]).any()
    assert set(BX_COLUMNS + NEW_COLUMNS + ["bx_since"]) <= set(panel.columns)
    for sid in ("S_AAA", "S_BBB"):
        got = panel_rows(panel, sid)
        assert got["session_date"].iloc[0] == CAL[0]  # from the first bar, not a window
        assert got["session_date"].iloc[-1] == MID_DAY
        mask = (BASE_REF["security_id"] == sid) & (BASE_REF["session_date"] <= MID_DAY)
        expected = pd.DataFrame(BASE_REF.loc[mask]).reset_index(drop=True)
        assert_columns_close(got, expected, BX_COLUMNS + NEW_COLUMNS)


def test_panel_holds_only_the_requested_securities(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    build(root, BASE)
    panel = bx_panel(con, root, close_of(MID_DAY), ["S_AAA"])
    assert set(panel["security_id"]) == {"S_AAA"}


def test_mid_scan_returns_the_one_signalling_security(
    con: duckdb.DuckDBPyConnection, root: Path, cfg_file: Path
) -> None:
    build(root, BASE)
    cfg = load_scan_config(cfg_file)
    tier = tier_at(con, root, MID_DAY, cfg)

    found = scan(con, root, close_of(MID_DAY), cfg)

    assert len(found) == 1
    assert_candidate(
        found[0],
        BASE_REF,
        security_id="S_AAA",
        symbol="AAA",
        day=MID_DAY,
        variant="mid",
        rank=1,
        adv_usd=adv_of(tier, "S_AAA"),
        bx_since=CAL[0],
    )


def test_long_variant_reads_the_monthly_state(
    con: duckdb.DuckDBPyConnection, root: Path, cfg_file: Path
) -> None:
    # on LONG_DAY AAA's monthly bar turned up but its weekly one did not
    build(root, BASE)
    cfg = load_scan_config(cfg_file, ["variant=long"])
    tier = tier_at(con, root, LONG_DAY, cfg)

    found = scan(con, root, close_of(LONG_DAY), cfg)

    assert len(found) == 1
    assert_candidate(
        found[0],
        BASE_REF,
        security_id="S_AAA",
        symbol="AAA",
        day=LONG_DAY,
        variant="long",
        rank=1,
        adv_usd=adv_of(tier, "S_AAA"),
        bx_since=CAL[0],
    )
    assert "S_AAA" not in ids(scan(con, root, close_of(LONG_DAY), load_scan_config(cfg_file)))


TIE1 = Sec("S_TIE1", "TIEA", seed=101)
TIE2 = Sec("S_TIE2", "TIEB", seed=101)  # the same prices: the same BX to the last bit
OTHER = Sec("S_OTH", "OTH", seed=202)
RANK_REF = ref_frame([TIE1, TIE2, OTHER])
RANK_DAY = find_session(
    RANK_REF, {("S_TIE1", "mid"): True, ("S_TIE2", "mid"): True, ("S_OTH", "mid"): True}
)


def rise(security_id: str, day: date) -> float:
    r = ref_row(RANK_REF, security_id, day)
    return float(r["bx_weekly"] - r["bx_weekly_prev"])


def test_rank_by_state_rise_then_security_id(
    con: duckdb.DuckDBPyConnection, root: Path, cfg_file: Path
) -> None:
    # write the tie's securities in reverse so the store's order is not the expected one
    build(root, [OTHER, TIE2, TIE1])
    cfg = load_scan_config(cfg_file)
    tie, other = rise("S_TIE1", RANK_DAY), rise("S_OTH", RANK_DAY)
    assert tie == rise("S_TIE2", RANK_DAY)
    assert tie != other

    found = scan(con, root, close_of(RANK_DAY), cfg)

    expected = ["S_TIE1", "S_TIE2", "S_OTH"] if tie > other else ["S_OTH", "S_TIE1", "S_TIE2"]
    assert ids(found) == expected
    assert [c.rank for c in found] == [1, 2, 3]
    assert [c.symbol for c in found] == [
        {"S_TIE1": "TIEA", "S_TIE2": "TIEB", "S_OTH": "OTH"}[s] for s in expected
    ]


def test_pool_is_every_tier_security_with_defined_values(
    con: duckdb.DuckDBPyConnection, root: Path, cfg_file: Path
) -> None:
    # BBB does not signal on MID_DAY but is warm and defined: in the pool
    build(root, BASE)
    cfg = load_scan_config(cfg_file)
    assert ref_defined(ref_row(BASE_REF, "S_BBB", MID_DAY), "mid")

    assert pool(con, root, close_of(MID_DAY), cfg) == frozenset({"S_AAA", "S_BBB"})


def test_signals_on_hand_set_values(
    con: duckdb.DuckDBPyConnection, root: Path, cfg_file: Path
) -> None:
    build(root, BASE)
    cfg = load_scan_config(cfg_file, ["warmup.daily=100", "warmup.weekly=20"])
    panel = bx_panel(con, root, close_of(MID_DAY), ["S_AAA", "S_BBB"])
    rows = pd.DataFrame(panel[panel["session_date"] == MID_DAY]).copy()
    tier = tier_at(con, root, MID_DAY, cfg)
    hand = {
        "bx_weekly": 10.0,
        "bx_weekly_prev": 5.0,
        "bx_daily": 3.0,
        "bx_daily_prev": 2.0,
        "daily_bars": 100,
        "weekly_bars": 20,
    }
    for column, value in hand.items():
        rows[column] = value
    # BBB stands higher (12 vs 10) but rose less (1 vs 5): the rise ranks, not the level
    is_bbb = rows["security_id"] == "S_BBB"
    rows.loc[is_bbb, "bx_weekly"] = 12.0
    rows.loc[is_bbb, "bx_weekly_prev"] = 11.0

    found = signals(rows, tier, cfg)

    assert ids(found) == ["S_AAA", "S_BBB"]
    aaa, bbb = found
    assert (bbb.state_bx, bbb.state_bx_prev, bbb.daily_bx, bbb.daily_bx_prev) == (12, 11, 3, 2)
    assert (aaa.state_bx, aaa.rank, aaa.state_bars) == (10, 1, 20)
    assert bbb.rank == 2
    assert aaa.adv_usd == pytest.approx(adv_of(tier, "S_AAA"))
    # a DataFrame tier answers the same
    assert ids(signals(rows, tier.to_pandas(), cfg)) == ["S_AAA", "S_BBB"]
    # equal rises: security_id decides
    rows.loc[is_bbb, "bx_weekly_prev"] = 7.0
    assert ids(signals(rows, tier, cfg)) == ["S_AAA", "S_BBB"]


def test_signals_equals_scan_on_the_panel_rows(
    con: duckdb.DuckDBPyConnection, root: Path, cfg_file: Path
) -> None:
    build(root, BASE)
    cfg = load_scan_config(cfg_file)
    panel = bx_panel(con, root, close_of(MID_DAY), ["S_AAA", "S_BBB"])
    rows = pd.DataFrame(panel[panel["session_date"] == MID_DAY])

    assert_same_candidates(
        signals(rows, tier_at(con, root, MID_DAY, cfg), cfg),
        scan(con, root, close_of(MID_DAY), cfg),
    )


# =============================================================================================
# boundaries
# =============================================================================================


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("bx_weekly", 5.0),  # equal to prev: not a turn up
        ("bx_daily", 2.0),  # equal to prev
        ("bx_weekly_prev", float("nan")),
        ("bx_daily_prev", float("nan")),
        ("bx_weekly", float("nan")),
        ("bx_daily", float("nan")),
    ],
)
def test_signals_need_strict_rises_and_four_defined_values(
    con: duckdb.DuckDBPyConnection, root: Path, cfg_file: Path, column: str, value: float
) -> None:
    build(root, BASE)
    cfg = load_scan_config(cfg_file)
    panel = bx_panel(con, root, close_of(MID_DAY), ["S_AAA"])
    rows = pd.DataFrame(panel[panel["session_date"] == MID_DAY]).copy()
    rows[["bx_weekly", "bx_weekly_prev", "bx_daily", "bx_daily_prev"]] = [10.0, 5.0, 3.0, 2.0]
    tier = tier_at(con, root, MID_DAY, cfg)
    assert ids(signals(rows, tier, cfg)) == ["S_AAA"]  # the hand values signal

    rows[column] = value
    assert signals(rows, tier, cfg) == []


def test_signals_drop_a_tier_row_of_another_class(
    con: duckdb.DuckDBPyConnection, root: Path, cfg_file: Path
) -> None:
    build(root, BASE)
    cfg = load_scan_config(cfg_file)
    panel = bx_panel(con, root, close_of(MID_DAY), ["S_AAA"])
    rows = pd.DataFrame(panel[panel["session_date"] == MID_DAY])
    tier = tier_at(con, root, MID_DAY, cfg).to_pandas()
    tier["class"] = "etf"

    assert signals(rows, tier, cfg) == []


def test_a_running_week_is_not_the_state(
    con: duckdb.DuckDBPyConnection, root: Path, cfg_file: Path
) -> None:
    # WED_DAY: AAA signals on a Wednesday; its state is the week that completed before Monday
    build(root, BASE)
    cfg = load_scan_config(cfg_file)

    [c] = [c for c in scan(con, root, close_of(WED_DAY), cfg) if c.security_id == "S_AAA"]

    assert c.state_complete_on == previous_friday(WED_DAY)
    monday = WED_DAY - timedelta(days=2)
    assert c.state_bx == pytest.approx(ref_row(BASE_REF, "S_AAA", monday)["bx_weekly"])


def test_editing_the_running_wednesday_moves_no_state_value(
    con: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    # the same store cut at WED_DAY twice, once with Wednesday's close and high raised by half
    plain, edited = market_root(tmp_path / "plain"), market_root(tmp_path / "edited")
    build(plain, BASE, cut=WED_DAY)
    raised = dataclasses.replace(AAA, raw_scale=((WED_DAY, None, 1.5),))
    build(edited, [raised, BBB], cut=WED_DAY)

    a = panel_rows(bx_panel(con, plain, close_of(WED_DAY), ["S_AAA"]), "S_AAA")
    b = panel_rows(bx_panel(con, edited, close_of(WED_DAY), ["S_AAA"]), "S_AAA")

    for column in ("bx_weekly", "bx_weekly_prev", "weekly_bars", "bx_monthly", "monthly_bars"):
        pd.testing.assert_series_equal(a[column], b[column])
    assert a["bx_daily"].iloc[-1] != b["bx_daily"].iloc[-1]


def test_warmup_one_short_and_exactly_enough(
    con: duckdb.DuckDBPyConnection, root: Path, cfg_file: Path
) -> None:
    build(root, BASE)
    r = ref_row(BASE_REF, "S_AAA", MID_DAY)
    d, w = int(r["daily_bars"]), int(r["weekly_bars"])

    def signalled(*overrides: str) -> bool:
        cfg = load_scan_config(cfg_file, list(overrides))
        return "S_AAA" in ids(scan(con, root, close_of(MID_DAY), cfg))

    assert signalled(f"warmup.daily={d}", f"warmup.weekly={w}")
    assert not signalled(f"warmup.daily={d + 1}", f"warmup.weekly={w}")
    assert not signalled(f"warmup.daily={d}", f"warmup.weekly={w + 1}")
    # the pool applies the same warm-up
    cfg = load_scan_config(cfg_file, [f"warmup.daily={d + 1}"])
    assert "S_AAA" not in pool(con, root, close_of(MID_DAY), cfg)


def test_long_warmup_reads_monthly_bars_and_ignores_the_weekly_one(
    con: duckdb.DuckDBPyConnection, root: Path, cfg_file: Path
) -> None:
    build(root, BASE)
    m = int(ref_row(BASE_REF, "S_AAA", LONG_DAY)["monthly_bars"])

    def signalled(*overrides: str) -> bool:
        cfg = load_scan_config(cfg_file, ["variant=long", *overrides])
        return "S_AAA" in ids(scan(con, root, close_of(LONG_DAY), cfg))

    assert signalled(f"warmup.monthly={m}", "warmup.weekly=100000")
    assert not signalled(f"warmup.monthly={m + 1}")


def test_a_security_outside_tonights_tier_never_signals(
    con: duckdb.DuckDBPyConnection, root: Path, cfg_file: Path
) -> None:
    thin = Sec("S_THIN", "THIN", seed=101, volume=THIN_VOLUME)  # AAA's prices, $50k a day
    build(root, [AAA, thin])

    found = scan(con, root, close_of(MID_DAY), load_scan_config(cfg_file))
    assert ids(found) == ["S_AAA"]
    assert "S_THIN" not in pool(con, root, close_of(MID_DAY), load_scan_config(cfg_file))
    # the same security does signal once the threshold lets it in (the fixture is right)
    loose = load_scan_config(cfg_file, ["universe.min_adv_usd=1"])
    assert set(ids(scan(con, root, close_of(MID_DAY), loose))) == {"S_AAA", "S_THIN"}


def test_an_etf_is_left_out_when_classes_are_common_only(
    con: duckdb.DuckDBPyConnection, root: Path, cfg_file: Path
) -> None:
    etf = Sec("S_ETFX", "ETFX", seed=101, cls="etf")  # AAA's prices
    build(root, [AAA, etf])

    assert ids(scan(con, root, close_of(MID_DAY), load_scan_config(cfg_file))) == ["S_AAA"]
    wide = load_scan_config(cfg_file, ["universe.classes=[common, etf]"])
    assert set(ids(scan(con, root, close_of(MID_DAY), wide))) == {"S_AAA", "S_ETFX"}


REN_EX = CAL[200]
REN_DAY = CAL[250]  # the rename
REN = Sec("S_REN", "NEWN", seed=505, old_symbol=("OLDN", REN_DAY))


def test_a_renamed_securitys_spinoff_under_its_old_symbol_is_applied(
    con: duckdb.DuckDBPyConnection, root: Path, cfg_file: Path
) -> None:
    # the record names OLDN; the child never trades: restart, decided 20 parent sessions later
    build(root, [REN, BBB], spinoffs=[spin_row("OLDN", "GONE", REN_EX, 0.5)])
    as_of = close_of(CAL[300])
    events = spinoff_events(con, root, as_of).to_pylist()
    assert [(e["security_id"], e["status"], e["known_on"]) for e in events] == [
        ("S_REN", "restart", after(REN_EX, 20))
    ]
    assert spinoff_events(con, root, as_of, parents=["NEWN"]).num_rows == 0  # the trap

    rows = panel_rows(bx_panel(con, root, as_of, ["S_REN"]), "S_REN")

    later = rows[rows["session_date"] >= after(REN_EX, 20)]
    assert {as_day(v) for v in later["bx_since"]} == {REN_EX}
    blank = rows[(rows["session_date"] >= REN_EX) & (rows["session_date"] < after(REN_EX, 20))]
    assert len(blank) == 20 and pd.DataFrame(blank[BX_COLUMNS]).isna().to_numpy().all()
    # the restarted series is warm and defined by as_of, so the scan reads it; whatever it
    # returns for S_REN carries the series start and today's symbol
    cfg = load_scan_config(cfg_file)
    assert "S_REN" in pool(con, root, as_of, cfg)
    for c in scan(con, root, as_of, cfg):
        if c.security_id == "S_REN":
            assert c.symbol == "NEWN" and c.bx_since == REN_EX


RST_EX = CAL[300]
RST = Sec("S_RST", "RST", seed=303)
RST_RECORD = spin_row("RST", "RST", RST_EX, 1.0)  # parent and child on one symbol: restart


def test_a_restart_on_its_ex_date_gives_no_signal_and_restarts_the_warmup(
    con: duckdb.DuckDBPyConnection, root: Path, cfg_file: Path
) -> None:
    build(root, [RST, BBB], spinoffs=[RST_RECORD])
    [e] = spinoff_events(con, root, close_of(CAL[-1])).to_pylist()
    assert (e["status"], e["reason"], e["known_on"]) == ("restart", "same_symbol", RST_EX)
    cfg = load_scan_config(cfg_file)

    # the evening before: warm and defined on the whole series
    assert "S_RST" in pool(con, root, close_of(after(RST_EX, -1)), cfg)
    # the ex-date: one bar into the new series, no previous value
    panel = bx_panel(con, root, close_of(RST_EX), ["S_RST"])
    first = panel_rows(panel, "S_RST").iloc[-1]
    assert first["session_date"] == RST_EX and first["daily_bars"] == 1
    assert math.isnan(first["bx_daily_prev"])
    assert "S_RST" not in pool(con, root, close_of(RST_EX), cfg)
    assert "S_RST" not in ids(scan(con, root, close_of(RST_EX), cfg))

    # the warm-up counts from the ex-date: n bars of the new series are needed, n - 1 are not
    prices = RST.prices()
    restarted = reference(pd.DataFrame(prices.loc[prices["session_date"] >= RST_EX]))
    defined = [i for i in range(len(restarted)) if ref_defined(restarted.iloc[i].to_dict(), "mid")]
    n = defined[0] + 3  # values defined at both sessions checked below
    warm = load_scan_config(cfg_file, [f"warmup.daily={n}"])
    assert "S_RST" not in pool(con, root, close_of(after(RST_EX, n - 2)), warm)
    assert "S_RST" in pool(con, root, close_of(after(RST_EX, n - 1)), warm)


LATE_LISTING = Sec("S_NEW", "NEWL", seed=606, first=len(CAL) - 20)  # four weeks: no weekly BX
PEND_EX = CAL[-3]
PEND = Sec("S_PEND", "PEND", seed=707)
PEND_RECORD = spin_row("PEND", "PKID", PEND_EX, 0.5)  # the child has not traded yet: pending


def test_nan_anywhere_keeps_a_security_out_of_the_pool(
    con: duckdb.DuckDBPyConnection, root: Path, cfg_file: Path
) -> None:
    build(root, [AAA, LATE_LISTING, PEND], spinoffs=[PEND_RECORD])
    cfg = load_scan_config(cfg_file, ["universe.lookback_sessions=5"])
    as_of = close_of(CAL[-1])
    tier = {r["security_id"] for r in tier_at(con, root, CAL[-1], cfg).to_pylist()}
    assert {"S_AAA", "S_NEW", "S_PEND"} <= tier
    [e] = spinoff_events(con, root, as_of).to_pylist()
    assert e["status"] == "pending"

    got = pool(con, root, as_of, cfg)

    assert got == frozenset({"S_AAA"})  # NEWL: no weekly BX yet; PEND: blank from its ex-date
    assert "S_PEND" in pool(con, root, close_of(after(PEND_EX, -1)), cfg)
    assert not {"S_NEW", "S_PEND"} & set(ids(scan(con, root, as_of, cfg)))


def test_no_session_final_yet_gives_nothing(
    con: duckdb.DuckDBPyConnection, root: Path, cfg_file: Path
) -> None:
    build(root, BASE)
    cfg = load_scan_config(cfg_file)
    before = close_of(CAL[0]) - timedelta(seconds=1)
    assert scan(con, root, before, cfg) == []
    assert pool(con, root, before, cfg) == frozenset()


def test_one_second_before_the_close_reads_the_previous_session(
    con: duckdb.DuckDBPyConnection, root: Path, cfg_file: Path
) -> None:
    build(root, BASE)
    cfg = load_scan_config(cfg_file)
    prev = after(MID_DAY, -1)
    early = scan(con, root, close_of(MID_DAY) - timedelta(seconds=1), cfg)
    assert_same_candidates(early, scan(con, root, close_of(prev), cfg))
    assert all(c.session == prev for c in early)


# =============================================================================================
# leakage
# =============================================================================================


def test_deleting_every_bar_after_as_of_changes_nothing(
    con: duckdb.DuckDBPyConnection, tmp_path: Path, cfg_file: Path
) -> None:
    full, cut = market_root(tmp_path / "full"), market_root(tmp_path / "cut")
    build(full, BASE)
    build(cut, BASE, cut=MID_DAY)
    as_of = close_of(MID_DAY)
    for variant in ("mid", "long"):
        cfg = load_scan_config(cfg_file, [f"variant={variant}"])
        assert_same_candidates(scan(con, full, as_of, cfg), scan(con, cut, as_of, cfg))
        assert pool(con, full, as_of, cfg) == pool(con, cut, as_of, cfg)
    a = bx_panel(con, full, as_of, ["S_AAA", "S_BBB"])
    b = bx_panel(con, cut, as_of, ["S_AAA", "S_BBB"])
    for sid in ("S_AAA", "S_BBB"):
        assert_columns_close(panel_rows(a, sid), panel_rows(b, sid), BX_COLUMNS + NEW_COLUMNS, 0.0)


LEAK_I = 280
LEAK_DAY = CAL[LEAK_I]
LATE_FEED = close_of(LEAK_DAY) + timedelta(hours=1)
SPL_EX = CAL[LEAK_I - 10]
SPL = Sec("S_SPL", "SPL", seed=808, raw_scale=((None, SPL_EX, 2.0),))  # a real 2:1 in prices


def test_a_split_record_not_yet_visible_changes_nothing(
    con: duckdb.DuckDBPyConnection, tmp_path: Path, cfg_file: Path
) -> None:
    # the record of a split that already happened in the prices arrives an hour after as_of
    with_rec, without = market_root(tmp_path / "with"), market_root(tmp_path / "without")
    build(with_rec, [AAA, SPL], splits=[split_row("SPL", SPL_EX, 1.0, 2.0, available_at=LATE_FEED)])
    build(without, [AAA, SPL])
    cfg = load_scan_config(cfg_file)
    as_of = close_of(LEAK_DAY)
    assert visible_splits(con, with_rec, as_of, rejected=()).num_rows == 0

    assert_same_candidates(scan(con, with_rec, as_of, cfg), scan(con, without, as_of, cfg))
    assert pool(con, with_rec, as_of, cfg) == pool(con, without, as_of, cfg)
    a = panel_rows(bx_panel(con, with_rec, as_of, ["S_SPL"]), "S_SPL")
    b = panel_rows(bx_panel(con, without, as_of, ["S_SPL"]), "S_SPL")
    assert_columns_close(a, b, BX_COLUMNS + NEW_COLUMNS, 0.0)
    # once visible, the record does change the series (the test can see it)
    later = close_of(after(LEAK_DAY, 1))
    c = panel_rows(bx_panel(con, with_rec, later, ["S_SPL"]), "S_SPL", hi=LEAK_DAY)
    assert not (c["bx_daily"].fillna(0) == a["bx_daily"].fillna(0)).all()


SPIN_EX = CAL[LEAK_I - 5]


def spin_world(root: Path, record: dict[str, Any] | None) -> None:
    """AAA hands out KID on SPIN_EX: AAA's stored prices drop by 0.8 from SPIN_EX, KID trades
    from SPIN_EX at 0.4 x AAA's previous close (ratio 0.5: factor ~0.8)."""
    prev = AAA.prices().set_index("session_date")["close"][after(SPIN_EX, -1)]
    parent = dataclasses.replace(AAA, raw_scale=((SPIN_EX, None, 0.8),))
    kid = Sec("S_KID", "KID", seed=909, start=0.4 * prev, first=CAL.index(SPIN_EX))
    build(root, [parent, BBB, kid], spinoffs=[record] if record else [])


def test_a_spinoff_record_not_yet_visible_changes_nothing(
    con: duckdb.DuckDBPyConnection, tmp_path: Path, cfg_file: Path
) -> None:
    with_rec, without = market_root(tmp_path / "with"), market_root(tmp_path / "without")
    spin_world(with_rec, spin_row("AAA", "KID", SPIN_EX, 0.5, available_at=LATE_FEED))
    spin_world(without, None)
    cfg = load_scan_config(cfg_file)
    as_of = close_of(LEAK_DAY)
    assert spinoff_events(con, with_rec, as_of).num_rows == 0

    assert_same_candidates(scan(con, with_rec, as_of, cfg), scan(con, without, as_of, cfg))
    assert pool(con, with_rec, as_of, cfg) == pool(con, without, as_of, cfg)
    a = panel_rows(bx_panel(con, with_rec, as_of, ["S_AAA"]), "S_AAA")
    b = panel_rows(bx_panel(con, without, as_of, ["S_AAA"]), "S_AAA")
    assert_columns_close(a, b, BX_COLUMNS + NEW_COLUMNS, 0.0)
    # once visible it is an adjust event that rewrites the sessions from its ex-date
    later = close_of(after(LEAK_DAY, 1))
    [e] = spinoff_events(con, with_rec, later).to_pylist()
    assert e["status"] == "adjust"


EARLY_EX = CAL[LEAK_I + 2]
EARLY = Sec("S_ERL", "ERL", seed=808, raw_scale=((None, EARLY_EX, 2.0),))


def test_a_visible_split_with_a_later_ex_date_moves_no_bx_value(
    con: duckdb.DuckDBPyConnection, tmp_path: Path, cfg_file: Path
) -> None:
    # announced five sessions before as_of, effective two sessions after it (constructed)
    record = split_row("ERL", EARLY_EX, 1.0, 2.0, available_at=known(CAL[LEAK_I - 5]))
    with_rec, without = market_root(tmp_path / "with"), market_root(tmp_path / "without")
    build(with_rec, [AAA, EARLY], splits=[record])
    build(without, [AAA, EARLY])
    cfg = load_scan_config(cfg_file)
    as_of = close_of(LEAK_DAY)
    assert visible_splits(con, with_rec, as_of).num_rows == 1

    a = panel_rows(bx_panel(con, with_rec, as_of, ["S_ERL"]), "S_ERL")
    b = panel_rows(bx_panel(con, without, as_of, ["S_ERL"]), "S_ERL")
    assert a["session_date"].iloc[-1] == LEAK_DAY
    assert_columns_close(a, b, BX_COLUMNS + NEW_COLUMNS, 1e-9)
    assert_same_candidates(scan(con, with_rec, as_of, cfg), scan(con, without, as_of, cfg))


RANGE_START, RANGE_END = CAL[290], CAL[318]
RANGE_SPLIT = CAL[300]
RANGE_SPIN = CAL[296]
RANGE_KID_FIRST = CAL[298]  # two sessions after the ex-date: known_on after it


def range_world(root: Path, cut: date | None = None) -> None:
    """Five tier candidates over CAL[290 .. 318]: AAA and BBB plain; SPL splitting 2:1 on
    CAL[300]; PAR handing out KID2 on CAL[296], KID2 first trading on CAL[298] (an adjust
    decided two sessions after its ex-date); INOUT liquid only on CAL[300 .. 306] (lookback 5:
    in the tier from about CAL[302], out again by about CAL[310])."""
    par_plain = Sec("S_PAR", "PAR", seed=111)
    prev = par_plain.prices().set_index("session_date")["close"][after(RANGE_SPIN, -1)]
    par = dataclasses.replace(par_plain, raw_scale=((RANGE_SPIN, None, 0.8),))
    kid = Sec(
        "S_KID2",
        "KIDB",
        seed=222,
        start=0.4 * prev,
        first=CAL.index(RANGE_KID_FIRST),
        volume=THIN_VOLUME,
    )
    spl = Sec("S_SPL", "SPL", seed=808, raw_scale=((None, RANGE_SPLIT, 2.0),))
    inout = Sec(
        "S_INOUT",
        "INOUT",
        seed=333,
        volume=THIN_VOLUME,
        volumes={d: 10_000_000 for d in CAL[300:307]},
    )
    build(
        root,
        [AAA, BBB, spl, par, kid, inout],
        splits=[split_row("SPL", RANGE_SPLIT, 1.0, 2.0)],
        spinoffs=[spin_row("PAR", "KIDB", RANGE_SPIN, 0.5)],
        cut=cut,
    )


def test_scan_range_equals_the_nightly_scan(
    con: duckdb.DuckDBPyConnection, root: Path, cfg_file: Path
) -> None:
    range_world(root)
    cfg = load_scan_config(cfg_file, ["universe.lookback_sessions=5"])
    window = [d for d in CAL if RANGE_START <= d <= RANGE_END]
    [e] = spinoff_events(con, root, close_of(RANGE_END)).to_pylist()
    assert (e["status"], e["known_on"]) == ("adjust", RANGE_KID_FIRST)
    tiers = {d: {r["security_id"] for r in tier_at(con, root, d, cfg).to_pylist()} for d in window}
    assert any("S_INOUT" in t for t in tiers.values())
    assert not all("S_INOUT" in t for t in tiers.values())

    ranged = scan_range(con, root, RANGE_START, RANGE_END, cfg)

    assert set(ranged) <= set(window)
    nightly = {d: scan(con, root, close_of(d), cfg) for d in window}
    assert sum(len(v) for v in nightly.values()) > 0  # fixture: the window has candidates
    for d in window:
        assert_same_candidates(ranged.get(d, []), nightly[d])
        assert {c.security_id for c in nightly[d]} <= tiers[d]
    # the long variant over the same window
    long_cfg = load_scan_config(cfg_file, ["universe.lookback_sessions=5", "variant=long"])
    ranged_long = scan_range(con, root, RANGE_START, RANGE_END, long_cfg)
    for d in window:
        assert_same_candidates(ranged_long.get(d, []), scan(con, root, close_of(d), long_cfg))


def test_scan_range_does_not_read_past_its_end(
    con: duckdb.DuckDBPyConnection, tmp_path: Path, cfg_file: Path
) -> None:
    full, cut = market_root(tmp_path / "full"), market_root(tmp_path / "cut")
    range_world(full)
    range_world(cut, cut=RANGE_END)
    cfg = load_scan_config(cfg_file, ["universe.lookback_sessions=5"])

    a = scan_range(con, full, RANGE_START, RANGE_END, cfg)
    b = scan_range(con, cut, RANGE_START, RANGE_END, cfg)

    assert set(a) == set(b)
    for d in a:
        assert_same_candidates(a[d], b[d])


def test_an_empty_tier_gives_nothing(
    con: duckdb.DuckDBPyConnection, root: Path, cfg_file: Path
) -> None:
    # sessions exist but nothing clears the threshold: no candidate, an empty pool, no error
    build(root, [Sec("S_THIN", "THIN", seed=101, volume=THIN_VOLUME)])
    cfg = load_scan_config(cfg_file)
    assert scan(con, root, close_of(MID_DAY), cfg) == []
    assert pool(con, root, close_of(MID_DAY), cfg) == frozenset()
    assert scan_range(con, root, after(MID_DAY, -3), MID_DAY, cfg) == {}


# =============================================================================================
# the master as known each night (scan_range must not attribute bars by the master of `end`)
# =============================================================================================
#
# Real shapes (asof.store.market docstring): NKLA's segment starts 2020-06-04 but is published
# 06-08; LAC is renamed 10-04 and the end of the old segment is recorded 11-02. Inside such a
# window the nightly reader attributes the bars by the master it can see then.

TAKE_DAY = CAL[300]  # S_OLD's symbol LAC passes to S_TAKER
TAKE_LAG = 10  # sessions until the hand-over is recorded
OLD = Sec("S_OLD", "LAC", seed=101, last=CAL.index(TAKE_DAY) - 1)
TAKER = Sec("S_TAKER", "LAC", seed=202, first=CAL.index(TAKE_DAY))


def lagged_takeover(root: Path) -> None:
    """S_OLD trades LAC until TAKE_DAY; S_TAKER trades LAC from it. The master learns both the
    end of S_OLD's segment and S_TAKER's start TAKE_LAG sessions late."""
    build(root, [OLD, TAKER, BBB])
    recorded = known(after(TAKE_DAY, TAKE_LAG))
    segs = [
        Seg("S_OLD", "LAC", valid_to=TAKE_DAY, end_available_at=recorded),
        Seg("S_TAKER", "LAC", valid_from=TAKE_DAY, available_at=recorded),
        *BBB.segs(),
    ]
    write_master(root, segs)
    write_classes(root, segs)


def test_scan_range_reads_a_late_recorded_takeover_as_each_night_saw_it(
    con: duckdb.DuckDBPyConnection, root: Path, cfg_file: Path
) -> None:
    lagged_takeover(root)
    cfg = load_scan_config(cfg_file)
    start, end = after(TAKE_DAY, -3), after(TAKE_DAY, TAKE_LAG + 3)
    window = [d for d in CAL if start <= d <= end]
    # fixture guard: inside the lag S_OLD is warm and defined each night (TAKER's bars under
    # its name), while the master of `end` gives it no row there at all
    inside = [d for d in window if TAKE_DAY <= d < after(TAKE_DAY, TAKE_LAG)]
    assert all("S_OLD" in pool(con, root, close_of(d), cfg) for d in inside)
    at_end = bx_panel(con, root, close_of(end), ["S_OLD"])
    assert not (at_end["session_date"] >= TAKE_DAY).any()

    pools = pool_range(con, root, start, end, cfg)
    ranged = scan_range(con, root, start, end, cfg)

    for d in window:
        assert pools.get(d, frozenset()) == pool(con, root, close_of(d), cfg), d
        assert_same_candidates(ranged.get(d, []), scan(con, root, close_of(d), cfg))


KID_LAG = 5


def test_scan_range_reads_a_late_recorded_child_as_each_night_saw_it(
    con: duckdb.DuckDBPyConnection, root: Path, cfg_file: Path
) -> None:
    # KID trades from SPIN_EX but its segment is recorded KID_LAG sessions later: until then the
    # nightly reader cannot see a child bar, the event is pending and AAA is blank
    spin_world(root, spin_row("AAA", "KID", SPIN_EX, 0.5))
    kid = Sec("S_KID", "KID", seed=909, first=CAL.index(SPIN_EX))
    segs = [
        *AAA.segs(),
        *BBB.segs(),
        Seg("S_KID", "KID", valid_from=SPIN_EX, available_at=known(after(SPIN_EX, KID_LAG))),
    ]
    write_master(root, segs)
    write_classes(root, segs)
    cfg = load_scan_config(cfg_file)
    start, end = after(SPIN_EX, -2), after(SPIN_EX, KID_LAG + 3)
    window = [d for d in CAL if start <= d <= end]
    assert kid.first == CAL.index(SPIN_EX)
    # fixture guard: blank inside the lag night by night, an adjust known on SPIN_EX at `end`
    assert "S_AAA" not in pool(con, root, close_of(SPIN_EX), cfg)
    [e] = spinoff_events(con, root, close_of(end)).to_pylist()
    assert (e["status"], e["known_on"]) == ("adjust", SPIN_EX)

    pools = pool_range(con, root, start, end, cfg)
    ranged = scan_range(con, root, start, end, cfg)

    for d in window:
        assert pools.get(d, frozenset()) == pool(con, root, close_of(d), cfg), d
        assert_same_candidates(ranged.get(d, []), scan(con, root, close_of(d), cfg))
