"""Guards against a THT Fair Value Bands rule that passes on hand-made numbers and disagrees with
the TradingView chart it was reverse-engineered from: a wrong window, wrong thresholds or wrong
carry-over shows up as a colour flip on the wrong day, a wrong price basis as an fv off by
cents, a missing split adjustment as a jump in NVDA's fv on its ex-date.

Runs on the real store under ``$ASOF_DATA_DIR/market`` (default /data/asof) and skips when it is
absent, so CI does not run it (the convention of tests/store/test_golden_cases.py). Bars are read
through the strategies' path: visible_bars as of an evening, adjust_for_splits with the splits
visible then, spinoff_events visible then.

Facts (hand-checked 2026-10-29 against the THT Fair Value Bands chart on TradingView, MSFT daily,
settings OHLC4 SMA 33 Cross; the store read the same day matches them):
- colour flips, each on the exact session: 2025-08-21 red (08-20 green), 2025-10-03 green (10-02
  red), 2025-11-06 red (11-05 green), 2026-04-15 green (04-14 red), 2026-06-09 red (06-08 green),
  2026-07-21 green (07-17 red; 07-20 is a session in the store and is red too).
- lower band (fv) 514.82 on 2025-08-20, 508.28 on 2025-10-03, 518.37 on 2025-11-06.
- NVDA 10-for-1 split, ex 2024-06-10 (public record); adjusted fv barely moves across it.
"""

from datetime import date, datetime
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd
import pytest

from asof.features.bx import adjust_for_splits
from asof.features.tht import tht_features
from asof.ingest.security_master import MASTER_FILE
from asof.store.db import default_data_dir
from asof.store.market import market_root, visible_bars, visible_splits
from asof.store.spinoffs import spinoff_events

ET = ZoneInfo("America/New_York")
ROOT = market_root(default_data_dir())

pytestmark = [
    pytest.mark.realdata,
    pytest.mark.skipif(not (ROOT / MASTER_FILE).exists(), reason=f"no real store under {ROOT}"),
]


@pytest.fixture(scope="module")
def con() -> duckdb.DuckDBPyConnection:
    return duckdb.connect()


def evening(day: date) -> datetime:
    """21:00 New York: the session's daily bar is visible."""
    return datetime(day.year, day.month, day.day, 21, tzinfo=ET)


def tht_as_of(con: duckdb.DuckDBPyConnection, symbol: str, as_of: datetime) -> pd.DataFrame:
    """THT of the one security carrying ``symbol``, from everything visible at ``as_of``,
    indexed by session."""
    bars = visible_bars(con, ROOT, "1Day", as_of, symbols=[symbol]).to_pandas()
    assert bars["security_id"].nunique() == 1, bars["security_id"].unique()
    daily = adjust_for_splits(bars, visible_splits(con, ROOT, as_of, [symbol]).to_pandas())
    events = spinoff_events(con, ROOT, as_of, [symbol]).to_pandas()
    return tht_features(daily, spinoffs=events).set_index("session_date")


@pytest.fixture(scope="module")
def msft(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    out = tht_as_of(con, "MSFT", evening(date(2026, 7, 22)))
    assert max(out.index) >= date(2026, 7, 21), "store stops before the last flip"
    return out


def color(out: pd.DataFrame, day: date) -> str | None:
    v = out.loc[day, "tht_color"]
    return None if v is None or (isinstance(v, float) and v != v) else str(v)


FLIPS = [
    (date(2025, 8, 20), "green", date(2025, 8, 21), "red"),
    (date(2025, 10, 2), "red", date(2025, 10, 3), "green"),
    (date(2025, 11, 5), "green", date(2025, 11, 6), "red"),
    (date(2026, 4, 14), "red", date(2026, 4, 15), "green"),
    (date(2026, 6, 8), "green", date(2026, 6, 9), "red"),
    (date(2026, 7, 17), "red", date(2026, 7, 21), "green"),
]


@pytest.mark.parametrize(("before", "was", "day", "now"), FLIPS, ids=[str(f[2]) for f in FLIPS])
def test_msft_daily_flips_on_the_charts_day(
    msft: pd.DataFrame, before: date, was: str, day: date, now: str
) -> None:
    sessions = sorted(msft.index)
    previous = sessions[sessions.index(day) - 1]
    assert color(msft, before) == was
    assert color(msft, previous) == was  # 2026-07-20 between 07-17 and 07-21
    assert color(msft, day) == now


def test_msft_colour_changes_only_on_the_charts_days(msft: pd.DataFrame) -> None:
    window = [d for d in sorted(msft.index) if date(2025, 8, 20) <= d <= date(2026, 7, 21)]
    flips = [
        b for a, b in zip(window, window[1:], strict=False) if color(msft, a) != color(msft, b)
    ]
    assert flips == [f[2] for f in FLIPS]


@pytest.mark.parametrize(
    ("day", "fv"),
    [(date(2025, 8, 20), 514.82), (date(2025, 10, 3), 508.28), (date(2025, 11, 6), 518.37)],
)
def test_msft_fv_matches_the_chart_to_the_cent(msft: pd.DataFrame, day: date, fv: float) -> None:
    assert abs(float(msft.loc[day, "tht_fv"]) - fv) < 0.005


def test_nvda_fv_and_colour_continue_across_the_split(con: duckdb.DuckDBPyConnection) -> None:
    before, ex = date(2024, 6, 7), date(2024, 6, 10)
    out = tht_as_of(con, "NVDA", evening(ex))
    assert abs(float(out.loc[ex, "tht_fv"]) / float(out.loc[before, "tht_fv"]) - 1) < 0.03
    # 98.94 vs 121.79 on 2024-06-10; unadjusted, fv would sit near 950
    assert 0.5 < float(out.loc[ex, "tht_fv"]) / float(out.loc[ex, "close"]) < 2
    assert color(out, before) is not None
    assert color(out, ex) == color(out, before)


def test_nvda_split_is_not_seen_the_evening_before(con: duckdb.DuckDBPyConnection) -> None:
    # point in time on real data: on 06-07's evening the split is not visible, so the night's
    # fv is on the pre-split scale, ten times the one the 06-10 evening computes for 06-07
    before, ex = date(2024, 6, 7), date(2024, 6, 10)
    then = tht_as_of(con, "NVDA", evening(before))
    later = tht_as_of(con, "NVDA", evening(ex))
    assert max(then.index) == before
    ratio = float(then.loc[before, "tht_fv"]) / float(later.loc[before, "tht_fv"])
    assert ratio == pytest.approx(10.0, rel=1e-9)
    assert color(then, before) == color(later, before)
