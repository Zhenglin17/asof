"""Guards against share counts that look right on fake data and are wrong on the real store: a
cover count already split-adjusted by the filer (so the split would be applied twice), a 20-F
count in ordinary shares read against an ADS price, a 40-F filer wrongly treated as unknown.

Runs on the real store under ``$ASOF_DATA_DIR/market`` (default /data/asof) and skips when the
master or the shares file is absent, so CI does not run it (the convention of
tests/store/test_golden_cases.py).

Facts used (split ratios and ex-dates are public record; raw closes read off the real store
2026-10-29 with visible_bars and matching the public record):
- AAPL 4-for-1 split, ex-date 2020-08-31. Close 2020-08-28 499.23, 2020-09-01 134.18 (x4 =
  536.72, +7.5%): a correct count gives caps within 10% across the split.
- NVDA 10-for-1 split, ex-date 2024-06-10. Close 2024-06-07 1208.88, 2024-06-10 121.79 (x10 =
  1217.90, +0.7%).
- TSM files 20-F; one ADS is five ordinary shares, so a count in ordinary shares (about 25.9e9)
  times the ADS close (173.81 on 2024-06-28) overstates the cap about fivefold; it must still
  clear 5e9.
- TECK (Teck Resources, class B; close 47.90 on 2024-06-28) files 40-F.

Assumed and checked here: that the AAPL and NVDA cover counts filed before the splits are
pre-split numbers (AAPL 10-Q 2020-07-31, NVDA 10-Q 2024-05-29). If either filer reported the
post-split count, the caps would differ by the split ratio and these cases fail.
"""

from datetime import date, datetime
from zoneinfo import ZoneInfo

import duckdb
import pytest

from asof.ingest.security_master import MASTER_FILE
from asof.ingest.shares import SHARES_FILE
from asof.store.db import default_data_dir
from asof.store.market import market_root, visible_master
from asof.store.shares import cap_passes, market_caps
from asof.store.universe import liquid_securities

ET = ZoneInfo("America/New_York")
ROOT = market_root(default_data_dir())
MAX_AGE_DAYS = 400  # a 20-F's year-end count is up to ~6 months old by June

pytestmark = [
    pytest.mark.realdata,
    pytest.mark.skipif(
        not (ROOT / MASTER_FILE).exists() or not (ROOT / SHARES_FILE).exists(),
        reason=f"no real store or no share counts under {ROOT}",
    ),
]


@pytest.fixture(scope="module")
def con() -> duckdb.DuckDBPyConnection:
    return duckdb.connect()


def evening(day: date) -> datetime:
    """21:00 New York: the session's daily bar is visible."""
    return datetime(day.year, day.month, day.day, 21, tzinfo=ET)


def sid(con: duckdb.DuckDBPyConnection, symbol: str, day: date) -> str:
    """The security holding ``symbol`` on ``day``, as the master was known that evening."""
    found = {
        r["security_id"]
        for r in visible_master(con, ROOT, evening(day)).to_pylist()
        if r["symbol"] == symbol
        and r["valid_from"] <= day
        and (r["valid_to"] is None or day < r["valid_to"])
    }
    assert len(found) == 1, (symbol, day, found)
    return found.pop()


def cap(con: duckdb.DuckDBPyConnection, symbol: str, day: date) -> dict:
    security = sid(con, symbol, day)
    rows = market_caps(con, ROOT, evening(day), [security], max_age_days=MAX_AGE_DAYS).to_pylist()
    assert len(rows) == 1, (symbol, day, rows)
    return rows[0]


@pytest.mark.parametrize(
    ("symbol", "before", "after", "ratio"),
    [
        ("AAPL", date(2020, 8, 28), date(2020, 9, 1), 4.0),
        ("NVDA", date(2024, 6, 7), date(2024, 6, 10), 10.0),
    ],
)
def test_market_cap_is_continuous_across_a_split(
    con: duckdb.DuckDBPyConnection, symbol: str, before: date, after: date, ratio: float
) -> None:
    a, b = cap(con, symbol, before), cap(con, symbol, after)

    assert a["security_id"] == b["security_id"]
    assert b["split_factor"] == pytest.approx(a["split_factor"] * ratio)
    assert 0.9 <= b["raw_cap"] / a["raw_cap"] <= 1.1, (a, b)


def test_tsm_is_ratio_unknown_and_passes_the_20f_bar(con: duckdb.DuckDBPyConnection) -> None:
    row = cap(con, "TSM", date(2024, 6, 28))

    assert row["ratio_unknown"] is True
    night = evening(date(2024, 6, 28))
    tier = liquid_securities(
        con, ROOT, night, min_adv_usd=5e7, lookback_sessions=20, classes=("common",)
    )
    adv = {r["security_id"]: r["adv_usd"] for r in tier.to_pylist()}
    passed = cap_passes(
        market_caps(con, ROOT, night, [row["security_id"]], max_age_days=MAX_AGE_DAYS),
        adv_usd={row["security_id"]: adv[row["security_id"]]},
        min_cap=5e8,
        min_raw_cap_ratio_unknown=5e9,
        max_turnover=1.0,
        max_cap=5e12,
    )
    assert row["security_id"] in passed


def test_teck_files_40f_and_its_ratio_is_known(con: duckdb.DuckDBPyConnection) -> None:
    row = cap(con, "TECK", date(2024, 6, 28))
    assert row["ratio_unknown"] is False
