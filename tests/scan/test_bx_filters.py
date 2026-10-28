# Guards the three B-Xtrender signal filters (daily_min, require_monthly_up, market_filter)
# against look-ahead -- an index bar, a split record or a monthly bar not yet final at as_of
# deciding tonight -- and against leaking into the random-entry null pool the conditions that
# are part of the signal itself.
"""Target API (asof.scan.bx_trender, additions to 2-c; defaults keep 2-c behaviour):

    BxScanConfig.daily_min: float | None = None
        a candidate also needs bx_daily >= daily_min. A signal condition: ``pool`` ignores it.
    BxScanConfig.require_monthly_up: bool = False
        a candidate also needs bx_monthly > bx_monthly_prev, both defined, and monthly_bars >=
        warmup.monthly. When on, definedness and warmth of the monthly values join the pool's
        eligibility; the rise itself does not.
    BxScanConfig.market_filter: MarketFilter | None = None; MarketFilter(symbol, sma_sessions)
    market_ok(con, root, as_of, flt) -> bool
        the symbol's split-adjusted daily closes visible at as_of whose 20:00 New York close is
        not after as_of; True iff the last close > the mean of the last sma_sessions closes
        (tonight's included); fewer closes than sma_sessions -> False.
        When False on a night: ``scan`` -> [], ``pool`` -> frozenset(); ``scan_range`` has no
        entry for that night and ``pool_range`` an empty set.

Fake data: tests/scan/bx_world.py (provenance there) for the stocks. The index is a made-up
"SPY" (class etf, so never in the common tier) whose closes are written here directly so the
SMA test is hand-checkable: RISING 100 + 0.1 i, FALLING 200 - 0.1 i, PEAK rising to CAL[300]
then falling 1.0 a session, FLAT 100 (i = index in CAL). OHLC around the close and the 20:00
New York availability come from tests.store.audit_world.bar (the ingest rule). A bar or split
record with another available_at is constructed (a late or early feed). No network.

Assumptions a real-data check must confirm:
- The real index ETF chosen for S3 (SPY) is in the bar store under that symbol with no gap in
  the last ``sma_sessions`` sessions of every replay night: audit by counting, for each replay
  night, SPY's visible closes over the previous 200 calendar sessions (expected 200 each) on
  the real store.
- SPY's real splits are none since 2016 (the store's history); the split-leakage case here is
  a construction. Check: visible_splits(con, root, now, ["SPY"]) on the real store is empty.
- The monthly-bar completion on the real calendar (a month ending on a holiday) is the
  feature's (tests/features/test_bx.py) and the calendar audit's job (I12), not re-checked here.
"""

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pytest

from asof.scan.bx_trender import (
    BxScanConfig,
    MarketFilter,
    bx_panel,
    market_ok,
    pool,
    pool_range,
    scan,
    scan_range,
    signals,
)
from asof.store.market import market_root
from tests.scan.bx_world import (
    CAL,
    LIQUID_VOLUME,
    Sec,
    after,
    close_of,
    find_session,
    ref_row,
    split_row,
)
from tests.scan.test_bx_trender import (
    AAA,
    BASE,
    BASE_REF,
    BBB,
    MID_DAY,
    TEST_SCAN,
    as_day,
    assert_same_candidates,
    ids,
    panel_rows,
    tier_at,
)
from tests.store.audit_world import bar as raw_bar
from tests.store.test_spinoffs import write_bars
from tests.store.test_split_events import write_split_rows
from tests.store.test_universe import Seg, write_classes, write_master

SPY = "SPY"
N = 200
FILTER = {"symbol": SPY, "sma_sessions": N}
PEAK = 300
WINDOW_START, WINDOW_END = CAL[290], CAL[318]


def rising(i: int) -> float:
    return round(100 + 0.1 * i, 4)


def falling(i: int) -> float:
    return round(200 - 0.1 * i, 4)


def peak(i: int) -> float:
    return rising(i) if i <= PEAK else round(rising(PEAK) - 1.0 * (i - PEAK), 4)


def flat(i: int) -> float:
    return 100.0


Series = Callable[[int], float]


def ref_ok(series: Series, last: int, n: int = N) -> bool:
    """The filter by hand: closes CAL[0 .. last]; last close > mean of the last n."""
    closes = [series(i) for i in range(last + 1)]
    if len(closes) < n:
        return False
    return closes[-1] > sum(closes[-n:]) / n


def spy_bars(
    series: Series,
    *,
    cut: date | None = None,
    scale_before: tuple[date, float] | None = None,
    edits: Mapping[date, float] | None = None,
    available: Mapping[date, datetime] | None = None,
) -> list[dict[str, Any]]:
    """The index's raw bars. ``scale_before`` (ex, k): closes before ex stored k times higher
    (a k-for-1 split in the raw prices). ``edits``: a replaced close on a day. ``available``:
    a constructed available_at on a day."""
    out = []
    for i, d in enumerate(CAL):
        if cut is not None and d > cut:
            continue
        c = (edits or {}).get(d, series(i))
        if scale_before is not None and d < scale_before[0]:
            c = round(c * scale_before[1], 4)
        out.append(
            raw_bar(SPY, d, close=c, volume=LIQUID_VOLUME, available_at=(available or {}).get(d))
        )
    return out


def world(
    root: Path,
    secs: Sequence[Sec],
    series: Series | None,
    *,
    cut: date | None = None,
    splits: Sequence[dict[str, Any]] = (),
    **spy: Any,
) -> None:
    """The stocks of bx_world plus the index (left out when ``series`` is None)."""
    rows = [b for s in secs for b in s.bars(cut)]
    segs = [seg for s in secs for seg in s.segs()]
    if series is not None:
        rows += spy_bars(series, cut=cut, **spy)
        segs.append(Seg("S_SPY", SPY, cls="etf"))
    write_bars(root, rows)
    write_master(root, segs)
    write_classes(root, segs)
    if splits:
        write_split_rows(root, splits)


def cfg_of(**changes: Any) -> BxScanConfig:
    return BxScanConfig.model_validate({**TEST_SCAN, **changes})


def flt(n: int = N, symbol: str = SPY) -> MarketFilter:
    return cfg_of(market_filter={"symbol": symbol, "sma_sessions": n}).market_filter  # type: ignore[return-value]


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return market_root(tmp_path)


def day_rows(
    con: duckdb.DuckDBPyConnection, root: Path, day: date, sids: list[str]
) -> pd.DataFrame:
    panel = bx_panel(con, root, close_of(day), sids)
    return pd.DataFrame(panel[panel["session_date"] == day]).copy()


# =============================================================================================
# defaults
# =============================================================================================


def test_the_new_fields_default_to_the_2c_scan() -> None:
    cfg = cfg_of()
    assert cfg.daily_min is None
    assert cfg.require_monthly_up is False
    assert cfg.market_filter is None
    f = flt()
    assert (f.symbol, f.sma_sessions) == (SPY, N)


def test_defaults_scan_exactly_as_before(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    world(root, BASE, rising)
    explicit = cfg_of(daily_min=None, require_monthly_up=False, market_filter=None)
    as_of = close_of(MID_DAY)
    assert_same_candidates(scan(con, root, as_of, explicit), scan(con, root, as_of, cfg_of()))
    assert "S_AAA" in ids(scan(con, root, as_of, explicit))


# =============================================================================================
# market_filter: market_ok
# =============================================================================================


def test_market_ok_is_true_on_a_rising_index_and_false_on_a_falling_one(
    con: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    up, down = market_root(tmp_path / "up"), market_root(tmp_path / "down")
    world(up, [AAA], rising)
    world(down, [AAA], falling)
    # CAL[250]: rising close 125.0 > mean of CAL[51..250] = 100 + 0.1 x 150.5 = 115.05
    assert market_ok(con, up, close_of(CAL[250]), flt()) is True
    # falling close 175.0 < mean 184.95
    assert market_ok(con, down, close_of(CAL[250]), flt()) is False


def test_market_ok_matches_the_hand_sma_around_the_turn(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # PEAK: 130.0 at CAL[300], then 129, 128, ... The 200-session mean is about 120.7 there,
    # so the filter turns off roughly ten sessions after the peak.
    world(root, [AAA], peak)
    window = [i for i, d in enumerate(CAL) if WINDOW_START <= d <= WINDOW_END]
    expected = {i: ref_ok(peak, i) for i in window}
    assert any(expected.values()) and not all(expected.values())  # fixture: it turns

    for i in window:
        assert market_ok(con, root, close_of(CAL[i]), flt()) is expected[i], CAL[i]


def test_fewer_closes_than_the_window_is_false_and_exactly_enough_is_read(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    world(root, [AAA], rising)
    # CAL[198] has 199 closes; CAL[199] the 200th, 119.9 > mean 109.95
    assert market_ok(con, root, close_of(CAL[N - 2]), flt()) is False
    assert market_ok(con, root, close_of(CAL[N - 1]), flt()) is True
    assert ref_ok(rising, N - 1) and not ref_ok(rising, N - 2)


def test_a_close_equal_to_its_mean_is_not_above_it(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    world(root, [AAA], flat)  # 100.0 every session: last == mean exactly
    assert market_ok(con, root, close_of(CAL[250]), flt()) is False
    assert market_ok(con, root, close_of(CAL[250]), flt(n=3)) is False


def test_an_index_with_no_bars_is_false(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    world(root, [AAA], None)
    assert market_ok(con, root, close_of(CAL[250]), flt()) is False
    assert market_ok(con, root, close_of(CAL[250]), flt(symbol="QQQ")) is False


def test_a_visible_split_of_the_index_is_adjusted_for(
    con: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    # The raw closes before CAL[270] are doubled (a 2-for-1). Adjusted, the series is RISING
    # (True); read raw, the post-ex closes ~127 sit far under a mean pulled up to ~200 (False).
    ex, day = CAL[270], CAL[280]
    adjusted, raw = market_root(tmp_path / "adj"), market_root(tmp_path / "raw")
    world(adjusted, [AAA], rising, scale_before=(ex, 2.0), splits=[split_row(SPY, ex, 1.0, 2.0)])
    world(raw, [AAA], rising, scale_before=(ex, 2.0))

    assert market_ok(con, adjusted, close_of(day), flt()) is True
    assert market_ok(con, raw, close_of(day), flt()) is False


# =============================================================================================
# market_filter: leakage
# =============================================================================================


def test_bars_after_as_of_change_nothing_tonight(
    con: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    # The next session's index bar crashes to 1.0: visible at the next close, not tonight.
    day = CAL[250]
    full, cut = market_root(tmp_path / "full"), market_root(tmp_path / "cut")
    world(full, [AAA], rising, edits={after(day, 1): 1.0})
    world(cut, [AAA], rising, cut=day)

    assert market_ok(con, full, close_of(day), flt()) is True
    assert market_ok(con, cut, close_of(day), flt()) is True
    assert market_ok(con, full, close_of(after(day, 1)), flt()) is False  # the test can see it


def test_tonights_bar_is_read_only_when_final_and_available(
    con: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    # Tonight's close is 1.0 (it would turn the filter off). Late: published an hour after the
    # 20:00 close, so at 20:00 the last close read is yesterday's. Early: published at 16:05,
    # before the session is final; at 19:00 it must still not be read.
    day = CAL[250]
    late_root, early_root = market_root(tmp_path / "late"), market_root(tmp_path / "early")
    world(
        late_root,
        [AAA],
        rising,
        edits={day: 1.0},
        available={day: close_of(day) + timedelta(hours=1)},
    )
    early_at = close_of(day) - timedelta(hours=3, minutes=55)
    world(early_root, [AAA], rising, edits={day: 1.0}, available={day: early_at})

    assert market_ok(con, late_root, close_of(day), flt()) is True
    assert market_ok(con, late_root, close_of(day) + timedelta(hours=2), flt()) is False
    assert market_ok(con, early_root, close_of(day) - timedelta(hours=1), flt()) is True
    assert market_ok(con, early_root, close_of(day), flt()) is False


def test_an_index_split_recorded_after_as_of_is_not_applied(
    con: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    # The same doubled raw prices as above; the record arrives an hour after tonight's close.
    ex, day = CAL[270], CAL[280]
    late = close_of(day) + timedelta(hours=1)
    with_rec, without = market_root(tmp_path / "with"), market_root(tmp_path / "without")
    world(
        with_rec,
        [AAA],
        rising,
        scale_before=(ex, 2.0),
        splits=[split_row(SPY, ex, 1.0, 2.0, available_at=late)],
    )
    world(without, [AAA], rising, scale_before=(ex, 2.0))

    assert market_ok(con, with_rec, close_of(day), flt()) is False
    assert market_ok(con, without, close_of(day), flt()) is False
    assert market_ok(con, with_rec, close_of(after(day, 1)), flt()) is True  # visible next night


# =============================================================================================
# market_filter: scan, pool and the range readers
# =============================================================================================


def test_an_index_above_its_sma_lets_the_scan_through(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    world(root, BASE, rising)
    as_of = close_of(MID_DAY)
    filtered = cfg_of(market_filter=FILTER)

    plain = scan(con, root, as_of, cfg_of())
    assert "S_AAA" in ids(plain)
    assert_same_candidates(scan(con, root, as_of, filtered), plain)
    assert pool(con, root, as_of, filtered) == pool(con, root, as_of, cfg_of())


def test_an_index_below_its_sma_empties_scan_and_pool(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    world(root, BASE, falling)
    as_of = close_of(MID_DAY)
    filtered = cfg_of(market_filter=FILTER)
    assert "S_AAA" in ids(scan(con, root, as_of, cfg_of()))  # fixture: a signal to block

    assert scan(con, root, as_of, filtered) == []
    assert pool(con, root, as_of, filtered) == frozenset()


def test_the_range_readers_drop_exactly_the_nights_the_index_fails(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    world(root, BASE, peak)
    filtered, plain = cfg_of(market_filter=FILTER), cfg_of()
    nights = [(i, d) for i, d in enumerate(CAL) if WINDOW_START <= d <= WINDOW_END]
    passing = {d for i, d in nights if ref_ok(peak, i)}

    ranged = scan_range(con, root, WINDOW_START, WINDOW_END, filtered)
    unfiltered = scan_range(con, root, WINDOW_START, WINDOW_END, plain)
    pools = pool_range(con, root, WINDOW_START, WINDOW_END, filtered)
    plain_pools = pool_range(con, root, WINDOW_START, WINDOW_END, plain)

    # fixture: a failing night with a candidate the filter must remove
    assert any(unfiltered.get(d) for _, d in nights if d not in passing)
    assert set(ranged) <= passing
    for _, d in nights:
        expected = unfiltered.get(d, []) if d in passing else []
        assert_same_candidates(ranged.get(d, []), expected)
        assert_same_candidates(ranged.get(d, []), scan(con, root, close_of(d), filtered))
        assert pools[d] == (plain_pools[d] if d in passing else frozenset()), d


def test_the_range_reader_with_a_filter_does_not_read_past_its_end(
    con: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    # The index crashes right after the window; the window's nights must not notice.
    full, cut = market_root(tmp_path / "full"), market_root(tmp_path / "cut")
    world(full, BASE, peak, edits={after(WINDOW_END, 1): 1.0})
    world(cut, BASE, peak, cut=WINDOW_END)
    filtered = cfg_of(market_filter=FILTER)

    a = scan_range(con, full, WINDOW_START, WINDOW_END, filtered)
    b = scan_range(con, cut, WINDOW_START, WINDOW_END, filtered)
    assert set(a) == set(b)
    for d in a:
        assert_same_candidates(a[d], b[d])
    assert pool_range(con, full, WINDOW_START, WINDOW_END, filtered) == pool_range(
        con, cut, WINDOW_START, WINDOW_END, filtered
    )


# =============================================================================================
# daily_min
# =============================================================================================


HAND = ["bx_weekly", "bx_weekly_prev", "bx_daily", "bx_daily_prev"]


def test_daily_min_needs_the_daily_value_at_or_above_it(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    world(root, BASE, None)
    rows = day_rows(con, root, MID_DAY, ["S_AAA"])
    tier = tier_at(con, root, MID_DAY, cfg_of())
    gate = cfg_of(daily_min=0.0)

    rows[HAND] = [10.0, 5.0, -1.0, -2.0]  # rising, still red
    assert ids(signals(rows, tier, cfg_of())) == ["S_AAA"]
    assert signals(rows, tier, gate) == []

    rows[HAND] = [10.0, 5.0, 0.0, -2.0]  # exactly 0: at the minimum, in
    assert ids(signals(rows, tier, gate)) == ["S_AAA"]

    rows[HAND] = [10.0, 5.0, 0.5, 0.2]
    assert signals(rows, tier, cfg_of(daily_min=0.6)) == []
    assert ids(signals(rows, tier, cfg_of(daily_min=-50.0))) == ["S_AAA"]


def test_daily_min_on_the_scan_and_not_on_the_pool(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    world(root, BASE, None)
    as_of = close_of(MID_DAY)
    plain = scan(con, root, as_of, cfg_of())
    gated = scan(con, root, as_of, cfg_of(daily_min=0.0))

    assert_same_candidates(gated, [c for c in plain if c.daily_bx >= 0.0])
    # a minimum nothing reaches empties the scan but leaves the pool whole
    assert scan(con, root, as_of, cfg_of(daily_min=1000.0)) == []
    assert pool(con, root, as_of, cfg_of(daily_min=1000.0)) == pool(con, root, as_of, cfg_of())
    assert pool(con, root, as_of, cfg_of()) != frozenset()


# =============================================================================================
# require_monthly_up
# =============================================================================================


MONTHLY_UP_DAY = find_session(BASE_REF, {("S_AAA", "mid"): True, ("S_AAA", "long"): True})
# mid signals (daily rose) but long does not: the monthly bar did not rise
MONTHLY_DOWN_DAY = find_session(BASE_REF, {("S_AAA", "mid"): True, ("S_AAA", "long"): False})
MONTHLY = ["bx_monthly", "bx_monthly_prev"]


def test_monthly_up_on_hand_values(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    world(root, BASE, None)
    rows = day_rows(con, root, MID_DAY, ["S_AAA"])
    tier = tier_at(con, root, MID_DAY, cfg_of())
    on = cfg_of(require_monthly_up=True, warmup={"daily": 1, "weekly": 1, "monthly": 10})
    rows[HAND] = [10.0, 5.0, 3.0, 2.0]
    rows["monthly_bars"] = 10

    rows[MONTHLY] = [4.0, 1.0]
    assert ids(signals(rows, tier, on)) == ["S_AAA"]
    rows[MONTHLY] = [1.0, 1.0]  # equal: not a rise
    assert signals(rows, tier, on) == []
    assert ids(signals(rows, tier, cfg_of())) == ["S_AAA"]  # off: the monthly bar is not read
    rows[MONTHLY] = [4.0, float("nan")]
    assert signals(rows, tier, on) == []
    rows[MONTHLY] = [float("nan"), 1.0]
    assert signals(rows, tier, on) == []
    assert ids(signals(rows, tier, cfg_of())) == ["S_AAA"]
    rows[MONTHLY] = [4.0, 1.0]
    rows["monthly_bars"] = 9  # one short of warmup.monthly
    assert signals(rows, tier, on) == []


def test_monthly_up_keeps_a_rising_month_and_drops_a_falling_one(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    world(root, BASE, None)
    on = cfg_of(require_monthly_up=True)
    up = ref_row(BASE_REF, "S_AAA", MONTHLY_UP_DAY)
    down = ref_row(BASE_REF, "S_AAA", MONTHLY_DOWN_DAY)
    assert up["bx_monthly"] > up["bx_monthly_prev"]
    assert not down["bx_monthly"] > down["bx_monthly_prev"]

    assert "S_AAA" in ids(scan(con, root, close_of(MONTHLY_UP_DAY), on))
    assert "S_AAA" not in ids(scan(con, root, close_of(MONTHLY_DOWN_DAY), on))
    assert "S_AAA" in ids(scan(con, root, close_of(MONTHLY_DOWN_DAY), cfg_of()))
    # the rise is a signal condition, not an eligibility one: AAA stays in the pool
    assert "S_AAA" in pool(con, root, close_of(MONTHLY_DOWN_DAY), on)


def test_monthly_warmup_counts_only_when_the_monthly_bar_is_required(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    world(root, BASE, None)
    as_of = close_of(MONTHLY_UP_DAY)
    m = int(ref_row(BASE_REF, "S_AAA", MONTHLY_UP_DAY)["monthly_bars"])

    def warm(monthly: int, required: bool) -> BxScanConfig:
        return cfg_of(
            require_monthly_up=required, warmup={"daily": 1, "weekly": 1, "monthly": monthly}
        )

    assert "S_AAA" in ids(scan(con, root, as_of, warm(m, True)))
    assert "S_AAA" not in ids(scan(con, root, as_of, warm(m + 1, True)))
    assert "S_AAA" not in pool(con, root, as_of, warm(m + 1, True))
    # off (mid): the monthly warm-up is not read, as in 2-c
    assert "S_AAA" in ids(scan(con, root, as_of, warm(10**6, False)))
    assert "S_AAA" in pool(con, root, as_of, warm(10**6, False))


YOUNG = Sec("S_YNG", "YNG", seed=404, first=len(CAL) - 85)  # ~17 weeks, ~4 months listed


def test_undefined_monthly_values_leave_the_pool_only_when_required(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    world(root, [AAA, YOUNG], None)
    day = CAL[-1]
    rows = day_rows(con, root, day, ["S_YNG"])
    [young] = rows.to_dict("records")
    # fixture: weekly and daily defined, monthly not (too few completed months)
    assert not any(pd.isna(young[c]) for c in HAND)
    assert pd.isna(young["bx_monthly"]) or pd.isna(young["bx_monthly_prev"])

    assert "S_YNG" in pool(con, root, close_of(day), cfg_of())
    assert "S_YNG" not in pool(con, root, close_of(day), cfg_of(require_monthly_up=True))
    assert "S_YNG" not in ids(scan(con, root, close_of(day), cfg_of(require_monthly_up=True)))


def test_the_month_still_forming_is_never_read(
    con: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    # D is mid-month. The edited store raises AAA's raw prices by half from the month's first
    # session through D: tonight's daily BX moves, but the monthly values read on D come from
    # months completed before D's month, so the monthly condition must not move.
    day = next(d for d in CAL[250:] if d.day >= 12 and d.weekday() == 2)
    month_start = next(d for d in CAL if (d.year, d.month) == (day.year, day.month))
    plain_root, edited_root = market_root(tmp_path / "plain"), market_root(tmp_path / "edited")
    world(plain_root, BASE, None, cut=day)
    raised = dataclasses.replace(AAA, raw_scale=((month_start, None, 1.5),))
    world(edited_root, [raised, BBB], None, cut=day)

    a = panel_rows(bx_panel(con, plain_root, close_of(day), ["S_AAA"]), "S_AAA")
    b = panel_rows(bx_panel(con, edited_root, close_of(day), ["S_AAA"]), "S_AAA")
    for column in ("bx_monthly", "bx_monthly_prev", "monthly_bars", "monthly_complete_on"):
        pd.testing.assert_series_equal(a[column], b[column])
    assert a["bx_daily"].iloc[-1] != b["bx_daily"].iloc[-1]  # the edit took effect
    assert as_day(a["monthly_complete_on"].iloc[-1]) < month_start

    on = cfg_of(require_monthly_up=True)
    plain_up = {
        sid
        for sid in ("S_AAA", "S_BBB")
        if ref_row(BASE_REF, sid, day)["bx_monthly"]
        > ref_row(BASE_REF, sid, day)["bx_monthly_prev"]
    }
    expected = [
        c for c in scan(con, edited_root, close_of(day), cfg_of()) if c.security_id in plain_up
    ]
    assert_same_candidates(scan(con, edited_root, close_of(day), on), expected)
