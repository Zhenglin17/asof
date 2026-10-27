"""Guards against a B-Xtrender histogram that does not match the chart it is copied from (wrong
EMA / RMA seeding, wrong RSI edge cases), against split-unadjusted prices faking a crash on the
ex-date, against weekly / monthly bars that leak the rest of their period (a running week used as
if it were finished, a holiday week emitted before it is known to be over), and against values
that change when later rows or later-knowable splits are added.

Target API (asof.features.bx):

    pine_ema(src: pd.Series, length: int) -> pd.Series
        alpha = 2 / (length + 1); the first value IS src[0]; then alpha*src + (1-alpha)*prev.
    pine_rma(src: pd.Series, length: int) -> pd.Series
        alpha = 1 / length; NaN until the first window of ``length`` defined values, whose plain
        mean is the first value; then alpha*src + (1-alpha)*prev. Leading NaN in src push the
        first window back (u / d of an RSI start one row late).
    pine_rsi(x: pd.Series, length: int) -> pd.Series
        u = max(x - x[1], 0), d = max(x[1] - x, 0) (undefined on the first row);
        up = rma(u), down = rma(d); down == 0 -> 100, else up == 0 -> 0,
        else 100 - 100 / (1 + up / down). First defined at row ``length``.
    bx_histogram(close: pd.Series, l1=5, l2=20, l3=5) -> pd.Series
        rsi(ema(close, l1) - ema(close, l2), l3) - 50; first defined at row l3.
    adjust_for_splits(bars: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame
        bars: security_id, symbol, session_date, open, high, low, close (raw). splits: rows of
        visible_splits(...).to_pandas() (symbol, ex_date, factor, ...), already cut at as_of by
        the caller. Returns bars sorted by security_id, session_date with open/high/low/close of
        every session BEFORE ex_date divided by factor (factor = new shares per old share);
        several splits multiply. Other columns pass through unchanged.
        Matching rule: a split (S, d) applies to security X when X carries S at the boundary d,
        i.e. X's last bar before d has symbol S or X's first bar on/after d has symbol S.
    period_bars(daily: pd.DataFrame, period: "W" | "M") -> pd.DataFrame
        per security and calendar week (Monday-Sunday) or calendar month with at least one
        session: security_id, period_end (last session in it), complete_on, open (first open),
        high (max), low (min), close (last close); sorted by security_id, period_end.
        complete_on: the period's last weekday (Mon-Fri) if the security has a session that day;
        otherwise the security's first session after the period. A period that is complete by
        neither (still running at the end of the input) is not returned. The calendar is the
        security's own sessions (there is no exchange calendar in the store).
    bx_features(daily, *, l1=5, l2=20, l3=5) -> pd.DataFrame
        one row per input (security_id, session_date), sorted, with bx_daily (histogram over
        the daily closes), bx_weekly and bx_monthly (histogram over the completed weekly /
        monthly closes, taken from the latest period whose complete_on <= session_date; NaN
        before the first such period or while the period histogram is undefined).

Pine semantics: TradingView's Pine v4 reference pseudo-code for ema / rma / rsi (documentation,
not observed output). The indicator is "B-Xtrender @Puppytherapy", shortTermXtrender with the
user's lengths 5 / 20 / 5. The seeding encoded here is what the acceptance comparison against
TradingView numbers will test; if TradingView's built-in ema seeds differently (e.g. with an SMA)
the difference shows up there, on the first ~100 bars of a series.

Fake data: hand-made numbers, except the NVDA rows marked "real" (bar store, also pinned in
tests/store/test_golden_cases.py: close 2024-06-07 1208.88, 2024-06-10 open 120.37 close 121.79,
10-for-1 split ex 2024-06-10). Holidays used: Good Friday 2024-03-29, Memorial Day 2024-05-27,
Independence Day 2024-07-04 (NYSE calendar, documentation). Split rows go through the real
visible_splits on a tmp store in the leakage tests.

Assumptions a real-data check must confirm (tests/store/test_golden_cases.py):
- the matching rule attributes NVDA's 2024-06-10 split to NVDA and adjusts 2024-06-07 to 120.888.
- the rejected HON / AIIO records never reach the adjustment on the real store.
"""

import math
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pytest

from asof.features.bx import (
    adjust_for_splits,
    bx_features,
    bx_histogram,
    period_bars,
    pine_ema,
    pine_rma,
    pine_rsi,
)
from asof.store.market import market_root, visible_splits
from tests.store.test_market import known
from tests.store.test_split_events import split_row, write_split_rows

NAN = float("nan")
GOOD_FRIDAY = date(2024, 3, 29)
HOLIDAYS_2024 = {GOOD_FRIDAY, date(2024, 5, 27), date(2024, 7, 4), date(2024, 1, 15)}
PERIOD_COLUMNS = ["security_id", "period_end", "complete_on", "open", "high", "low", "close"]
BX_COLUMNS = ["bx_daily", "bx_weekly", "bx_monthly"]


# =============================================================================================
# builders
# =============================================================================================


def s(values: list[float]) -> pd.Series:
    return pd.Series(values, dtype=float)


def assert_series(got: pd.Series, expected: list[float]) -> None:
    assert len(got) == len(expected)
    for g, e in zip(got.tolist(), expected, strict=True):
        if math.isnan(e):
            assert math.isnan(g), (got.tolist(), expected)
        else:
            assert g == pytest.approx(e, rel=1e-12, abs=1e-12), (got.tolist(), expected)


def sessions(start: str, end: str, holidays: set[date] = HOLIDAYS_2024) -> list[date]:
    return [d.date() for d in pd.bdate_range(start, end) if d.date() not in holidays]


def bar(
    security_id: str,
    day: date,
    o: float,
    h: float,
    lo: float,
    c: float,
    symbol: str | None = None,
) -> dict[str, Any]:
    return {
        "security_id": security_id,
        "symbol": symbol or security_id,
        "session_date": day,
        "open": o,
        "high": h,
        "low": lo,
        "close": c,
        "volume": 1000,
    }


def four(price: float) -> tuple[float, float, float, float]:
    """open, high, low, close all at ``price``."""
    return (price, price, price, price)


def flat_bars(
    security_id: str, days: list[date], price: float, symbol: str | None = None
) -> list[dict[str, Any]]:
    return [bar(security_id, d, price, price, price, price, symbol) for d in days]


def walk(security_id: str, days: list[date], seed: int, start: float = 50.0) -> list[dict]:
    """A deterministic random walk with consistent OHLC (made-up numbers)."""
    rng = np.random.default_rng(seed)
    rows, close = [], start
    for d in days:
        o = close * (1 + rng.normal(0, 0.01))
        close = max(1.0, o * (1 + rng.normal(0, 0.02)))
        h = max(o, close) * (1 + abs(rng.normal(0, 0.005)))
        lo = min(o, close) * (1 - abs(rng.normal(0, 0.005)))
        rows.append(bar(security_id, d, o, h, lo, close))
    return rows


def frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame.from_records(rows)


def splits_frame(rows: list[dict]) -> pd.DataFrame:
    columns = ["symbol", "ex_date", "old_rate", "new_rate", "factor", "kind"]
    return pd.DataFrame.from_records(rows, columns=columns + ["old_cusip", "new_cusip"])


def split(symbol: str, ex_date: date, factor: float) -> dict[str, Any]:
    return split_row(symbol, ex_date, 1.0, factor)


def where(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Rows of ``df`` where ``mask`` holds, re-indexed from 0."""
    return pd.DataFrame(df.loc[mask]).reset_index(drop=True)


def col(df: pd.DataFrame, name: str) -> pd.Series:
    return pd.Series(df[name])


def of(df: pd.DataFrame, security_id: str) -> pd.DataFrame:
    return where(df, df["security_id"] == security_id)


def by_day(df: pd.DataFrame, security_id: str, column: str) -> dict[date, float]:
    part = of(df, security_id)
    return dict(zip(part["session_date"], part[column], strict=True))


def periods(df: pd.DataFrame) -> list[tuple]:
    return [tuple(r) for r in df[PERIOD_COLUMNS].itertuples(index=False)]


# NVDA around its 10-for-1 split. Real: 06-07 close 1208.88, 06-10 open 120.37 / close 121.79.
# Constructed: every other number.
NVDA_EX = date(2024, 6, 10)
NVDA_ROWS = [
    bar("S_NVDA", date(2024, 6, 5), 1190.0, 1225.0, 1185.0, 1224.40, "NVDA"),
    bar("S_NVDA", date(2024, 6, 6), 1224.0, 1226.0, 1190.0, 1209.98, "NVDA"),
    bar("S_NVDA", date(2024, 6, 7), 1197.7, 1216.9, 1180.2, 1208.88, "NVDA"),
    bar("S_NVDA", date(2024, 6, 10), 120.37, 123.10, 117.01, 121.79, "NVDA"),
    bar("S_NVDA", date(2024, 6, 11), 121.77, 122.87, 118.74, 120.91, "NVDA"),
]


# =============================================================================================
# normal path: Pine primitives on hand-computed sequences
# =============================================================================================


def test_pine_ema_starts_at_the_first_value() -> None:
    # length 3 -> alpha = 2 / 4 = 0.5
    # e0 = 1 (seed = src); e1 = 0.5*2 + 0.5*1 = 1.5; e2 = 0.5*3 + 0.5*1.5 = 2.25
    assert_series(pine_ema(s([1, 2, 3]), 3), [1.0, 1.5, 2.25])


def test_pine_rma_seeds_with_the_mean_of_the_first_window() -> None:
    # length 3 -> alpha = 1/3; r2 = mean(1, 2, 3) = 2; r3 = (4 + 2*2) / 3 = 8/3;
    # r4 = (5 + 2*8/3) / 3 = 31/9
    assert_series(pine_rma(s([1, 2, 3, 4, 5]), 3), [NAN, NAN, 2.0, 8 / 3, 31 / 9])


def test_pine_rma_with_a_leading_nan_starts_one_row_later() -> None:
    # u / d of an RSI are undefined on the first row: the first window is rows 1..3
    assert_series(pine_rma(s([NAN, 1, 2, 3, 4]), 3), [NAN, NAN, NAN, 2.0, 8 / 3])


def test_pine_rsi_hand_computed() -> None:
    # x = 1, 2, 4, 3, 5; length 2 (alpha 0.5)
    # u = -, 1, 2, 0, 2     d = -, 0, 0, 1, 0
    # up:   row2 = mean(1, 2) = 1.5; row3 = 0.5*0 + 0.5*1.5 = 0.75; row4 = 0.5*2 + 0.5*0.75 = 1.375
    # down: row2 = mean(0, 0) = 0;   row3 = 0.5*1 + 0 = 0.5;        row4 = 0 + 0.5*0.5 = 0.25
    # rsi:  row2 down == 0 -> 100; row3 rs 1.5 -> 100 - 100/2.5 = 60; row4 rs 5.5 -> 100 - 100/6.5
    assert_series(pine_rsi(s([1, 2, 4, 3, 5]), 2), [NAN, NAN, 100.0, 60.0, 100 - 100 / 6.5])


def test_bx_on_a_constant_series_is_plus_50() -> None:
    # ema of a constant is the constant (100.0 is exact in float for alpha 1/3 and 2/21), so
    # the ema difference is 0, u = d = 0, and Pine's rsi returns 100 when rma(d) == 0
    out = bx_histogram(s([100.0] * 30), 5, 20, 5)
    assert out.iloc[:5].isna().all()
    assert (out.iloc[5:] == 50.0).all()


def test_bx_rising_trend_is_positive_and_falling_trend_negative() -> None:
    rising = bx_histogram(s([10 + 0.5 * i for i in range(60)]), 5, 20, 5).iloc[5:]
    falling = bx_histogram(s([60 - 0.5 * i for i in range(60)]), 5, 20, 5).iloc[5:]
    assert (rising > 0).all()
    assert (falling < 0).all()


def test_bx_takes_the_lengths_as_arguments() -> None:
    close = s([10, 11, 10.5, 12, 11, 13, 12.5, 14, 13, 15, 14.2, 16])
    expected = pine_rsi(pine_ema(close, 3) - pine_ema(close, 6), 2) - 50
    assert_series(bx_histogram(close, 3, 6, 2), expected.tolist())


def test_adjust_for_splits_makes_the_nvda_series_continuous() -> None:
    out = adjust_for_splits(frame(NVDA_ROWS), splits_frame([split("NVDA", NVDA_EX, 10.0)]))

    closes = by_day(out, "S_NVDA", "close")
    assert closes[date(2024, 6, 7)] == pytest.approx(120.888)
    assert closes[date(2024, 6, 10)] == 121.79  # the ex-date itself is already post-split
    opens = by_day(out, "S_NVDA", "open")
    assert opens[NVDA_EX] / closes[date(2024, 6, 7)] == pytest.approx(120.37 / 120.888)
    for column in ("open", "high", "low", "close"):
        pre = by_day(out, "S_NVDA", column)[date(2024, 6, 5)]
        raw = next(r[column] for r in NVDA_ROWS if r["session_date"] == date(2024, 6, 5))
        assert pre == pytest.approx(raw / 10)


def test_adjust_for_splits_passes_other_columns_and_sorts() -> None:
    rows = list(reversed(NVDA_ROWS))
    out = adjust_for_splits(frame(rows), splits_frame([split("NVDA", NVDA_EX, 10.0)]))

    assert list(out["session_date"]) == sorted(r["session_date"] for r in rows)
    assert set(out["symbol"]) == {"NVDA"}
    assert len(out) == len(rows)


def test_adjust_for_splits_without_splits_is_identity() -> None:
    columns = ["security_id", "symbol", "session_date", "open", "high", "low", "close"]
    out = adjust_for_splits(frame(NVDA_ROWS), splits_frame([]))
    pd.testing.assert_frame_equal(out[columns].reset_index(drop=True), frame(NVDA_ROWS)[columns])


def test_period_bars_weekly_ohlc_over_two_weeks() -> None:
    days = sessions("2024-04-08", "2024-04-19")  # two full weeks, Mon-Fri
    rows = [
        bar("A", d, 10 + i, 12 + i + (5 if i == 2 else 0), 9 + i - (4 if i == 7 else 0), 11 + i)
        for i, d in enumerate(days)
    ]

    out = period_bars(frame(rows), "W")

    assert periods(out) == [
        # week 1: open of Mon 10, high Wed 12+2+5 = 19, low Mon 9, close Fri 15
        ("A", date(2024, 4, 12), date(2024, 4, 12), 10.0, 19.0, 9.0, 15.0),
        # week 2: open Mon 15, high Fri 21, low Wed 9+7-4 = 12, close Fri 20
        ("A", date(2024, 4, 19), date(2024, 4, 19), 15.0, 21.0, 12.0, 20.0),
    ]


def test_period_bars_monthly_ohlc() -> None:
    days = sessions("2024-04-01", "2024-05-31")
    rows = [bar("A", d, 100 + i, 101 + i, 99 + i, 100.5 + i) for i, d in enumerate(days)]

    out = period_bars(frame(rows), "M")

    april = [d for d in days if d.month == 4]
    may = [d for d in days if d.month == 5]
    n = len(april)
    assert periods(out) == [
        ("A", april[-1], date(2024, 4, 30), 100.0, 101.0 + n - 1, 99.0, 100.5 + n - 1),
        (
            "A",
            may[-1],
            date(2024, 5, 31),
            100.0 + n,
            101.0 + n + len(may) - 1,
            99.0 + n,
            100.5 + n + len(may) - 1,
        ),
    ]


def test_bx_features_rows_and_columns() -> None:
    days = sessions("2024-01-02", "2024-09-30")
    daily = frame(walk("A", days, seed=1) + walk("B", days, seed=2))

    out = bx_features(daily, l1=5, l2=20, l3=5)

    assert len(out) == len(daily)
    assert set(BX_COLUMNS) <= set(out.columns)
    a = of(out, "A")
    # daily: first defined on the 6th session; weekly / monthly defined once 6 periods completed
    daily_bx = col(a, "bx_daily")
    assert daily_bx.iloc[:5].isna().all() and daily_bx.iloc[5:].notna().all()
    assert col(a, "bx_weekly").notna().any() and col(a, "bx_monthly").notna().any()
    assert col(where(a, a["session_date"] < date(2024, 6, 28)), "bx_monthly").isna().all()
    assert col(where(a, a["session_date"] >= date(2024, 6, 28)), "bx_monthly").notna().all()


# =============================================================================================
# boundaries
# =============================================================================================


def test_good_friday_week_is_complete_at_the_close_of_the_next_monday() -> None:
    days = sessions("2024-03-25", "2024-04-05")  # 03-29 is a holiday
    out = period_bars(frame(flat_bars("A", days, 10.0)), "W")

    week = out[out["period_end"] == date(2024, 3, 28)]
    assert week["complete_on"].tolist() == [date(2024, 4, 1)]


def test_good_friday_week_is_not_emitted_when_the_input_ends_on_thursday() -> None:
    # nothing in the data says Friday is a holiday until Monday's bar exists
    days = sessions("2024-03-18", "2024-03-28")
    out = period_bars(frame(flat_bars("A", days, 10.0)), "W")

    assert out["period_end"].tolist() == [date(2024, 3, 22)]


def test_month_whose_last_weekday_is_a_holiday() -> None:
    # March 2024 ends on Sunday 03-31; its last weekday is Good Friday 03-29
    days = sessions("2024-03-01", "2024-04-03")
    out = period_bars(frame(flat_bars("A", days, 10.0)), "M")

    march = out[out["period_end"] == date(2024, 3, 28)]
    assert march["complete_on"].tolist() == [date(2024, 4, 1)]
    assert date(2024, 4, 3) not in out["period_end"].tolist()  # April still running


def test_month_ending_on_a_weekend_is_complete_on_its_last_friday() -> None:
    # August 2024 ends on Saturday; Friday 08-30 is a session
    days = sessions("2024-08-26", "2024-08-30")
    out = period_bars(frame(flat_bars("A", days, 10.0)), "M")

    assert periods(out) == [("A", date(2024, 8, 30), date(2024, 8, 30), 10.0, 10.0, 10.0, 10.0)]


def test_month_with_a_single_session() -> None:
    # a listing whose first session is the last day of the month
    rows = [bar("A", date(2024, 1, 31), 5.0, 6.0, 4.0, 5.5)] + flat_bars(
        "A", sessions("2024-02-01", "2024-02-29"), 5.5
    )
    out = period_bars(frame(rows), "M")

    assert periods(out)[0] == ("A", date(2024, 1, 31), date(2024, 1, 31), 5.0, 6.0, 4.0, 5.5)


def test_a_week_without_a_session_gets_no_bar() -> None:
    days = sessions("2024-04-01", "2024-04-05") + sessions("2024-04-15", "2024-04-19")
    out = period_bars(frame(flat_bars("A", days, 10.0)), "W")

    assert out["period_end"].tolist() == [date(2024, 4, 5), date(2024, 4, 19)]


def test_a_week_missing_its_friday_bar_completes_at_the_next_session() -> None:
    # the security did not trade on Friday 04-05 (halt); its next bar is Tuesday 04-09
    days = [
        d
        for d in sessions("2024-04-01", "2024-04-12")
        if d not in {date(2024, 4, 5), date(2024, 4, 8)}
    ]
    out = period_bars(frame(flat_bars("A", days, 10.0)), "W")

    assert out["complete_on"].tolist() == [date(2024, 4, 9), date(2024, 4, 12)]


def test_split_ex_date_mid_week_inside_a_weekly_bar() -> None:
    # 2-for-1 effective Wednesday 04-10: raw 100 on Mon / Tue, 50 from Wed on
    days = sessions("2024-04-08", "2024-04-12")
    rows = [bar("A", d, *four(100.0 if d < date(2024, 4, 10) else 50.0)) for d in days]

    adjusted = adjust_for_splits(frame(rows), splits_frame([split("A", date(2024, 4, 10), 2.0)]))
    out = period_bars(adjusted, "W")

    assert periods(out) == [("A", date(2024, 4, 12), date(2024, 4, 12), 50.0, 50.0, 50.0, 50.0)]


def test_two_splits_compound() -> None:
    days = sessions("2024-04-01", "2024-04-05")
    rows = flat_bars("A", days, 60.0)
    splits = splits_frame([split("A", days[2], 2.0), split("A", days[4], 3.0)])

    out = adjust_for_splits(frame(rows), splits)

    assert out["close"].tolist() == pytest.approx([10.0, 10.0, 20.0, 20.0, 60.0])


def test_two_events_on_one_day_multiply() -> None:
    # the MDRR 2024-07-03 shape: 1-for-10 then 5-for-1 the same day, net 0.5
    days = sessions("2024-07-01", "2024-07-03")
    rows = flat_bars("A", days, 6.0)
    splits = splits_frame([split("A", date(2024, 7, 3), 0.1), split("A", date(2024, 7, 3), 5.0)])

    out = adjust_for_splits(frame(rows), splits)

    assert out["close"].tolist() == pytest.approx([12.0, 12.0, 6.0])


def test_split_on_a_non_session_day_applies_to_the_bars_before_it() -> None:
    days = sessions("2024-04-01", "2024-04-12")
    rows = flat_bars("A", days, 10.0)
    out = adjust_for_splits(frame(rows), splits_frame([split("A", date(2024, 4, 6), 2.0)]))

    closes = by_day(out, "A", "close")
    assert closes[date(2024, 4, 5)] == 5.0 and closes[date(2024, 4, 8)] == 10.0


def test_split_known_before_its_first_bar_adjusts_all_earlier_bars() -> None:
    # as_of 10:00 New York on the ex-date: the record is visible (00:00), the bar is not (20:00)
    days = sessions("2024-06-03", "2024-06-07")
    out = adjust_for_splits(
        frame(flat_bars("S_NVDA", days, 1200.0, "NVDA")),
        splits_frame([split("NVDA", NVDA_EX, 10.0)]),
    )

    assert out["close"].tolist() == pytest.approx([120.0] * 5)


def test_split_before_the_first_bar_changes_nothing() -> None:
    days = sessions("2024-04-01", "2024-04-05")
    out = adjust_for_splits(
        frame(flat_bars("A", days, 10.0)), splits_frame([split("A", date(2024, 3, 1), 2.0)])
    )

    assert out["close"].tolist() == [10.0] * 5


def test_split_record_under_the_old_symbol_on_a_rename_day() -> None:
    # NLSP -> NCEL on 2025-10-31, one security; the 1-for-10 record is filed under NLSP while
    # the ex-date bar already carries NCEL (real record shape, prices constructed)
    days = sessions("2025-10-28", "2025-11-04", holidays=set())
    rows = [
        bar(
            "S1",
            d,
            *four(1.0 if d < date(2025, 10, 31) else 10.0),
            symbol="NLSP" if d < date(2025, 10, 31) else "NCEL",
        )
        for d in days
    ]
    out = adjust_for_splits(frame(rows), splits_frame([split("NLSP", date(2025, 10, 31), 0.1)]))

    assert out["close"].tolist() == pytest.approx([10.0] * len(days))


def test_split_record_under_the_new_symbol_on_a_rename_day() -> None:
    days = sessions("2025-10-28", "2025-11-04", holidays=set())
    rows = [
        bar(
            "S1",
            d,
            *four(1.0 if d < date(2025, 10, 31) else 10.0),
            symbol="OLD" if d < date(2025, 10, 31) else "NEW",
        )
        for d in days
    ]
    out = adjust_for_splits(frame(rows), splits_frame([split("NEW", date(2025, 10, 31), 0.1)]))

    assert out["close"].tolist() == pytest.approx([10.0] * len(days))


def test_a_label_reused_by_another_security_adjusts_only_the_current_holder() -> None:
    # A traded as XYZ, renamed to ABC in April; B took XYZ in May and splits 2-for-1 in June
    a_days = sessions("2024-03-01", "2024-07-31")
    rows = [
        bar("A", d, 30.0, 30.0, 30.0, 30.0, "XYZ" if d < date(2024, 4, 15) else "ABC")
        for d in a_days
    ]
    b_days = sessions("2024-05-01", "2024-07-31")
    rows += [
        bar("B", d, *four(80.0 if d < date(2024, 6, 17) else 40.0), symbol="XYZ") for d in b_days
    ]

    out = adjust_for_splits(frame(rows), splits_frame([split("XYZ", date(2024, 6, 17), 2.0)]))

    assert set(by_day(out, "A", "close").values()) == {30.0}
    assert set(by_day(out, "B", "close").values()) == {40.0}


def test_an_earlier_holder_that_stopped_trading_is_not_adjusted() -> None:
    # A traded as XYZ until 2021 and has no bar since; B took XYZ in 2024 and splits 2-for-1
    a_days = sessions("2021-03-01", "2021-03-31", holidays=set())
    rows = flat_bars("A", a_days, 30.0, "XYZ")
    b_days = sessions("2024-05-01", "2024-07-31")
    rows += [
        bar("B", d, *four(80.0 if d < date(2024, 6, 17) else 40.0), symbol="XYZ") for d in b_days
    ]

    out = adjust_for_splits(frame(rows), splits_frame([split("XYZ", date(2024, 6, 17), 2.0)]))

    assert set(by_day(out, "A", "close").values()) == {30.0}
    assert set(by_day(out, "B", "close").values()) == {40.0}


def test_two_rows_for_one_security_and_session_are_refused() -> None:
    # real shape: S010438 read as both BCYP and SABS on 170 days of 2021
    days = sessions("2021-03-01", "2021-03-31", holidays=set())
    rows = flat_bars("S1", days, 10.0, "BCYP") + flat_bars("S1", days[:3], 10.0, "SABS")

    with pytest.raises(ValueError, match="share a security and session"):
        bx_features(frame(rows))


def test_two_securities_never_mix() -> None:
    days = sessions("2024-04-01", "2024-04-30")
    a = flat_bars("A", days, 10.0)
    b = [bar("B", d, 20.0 + i, 21.0 + i, 19.0 + i, 20.5 + i) for i, d in enumerate(days)]

    adjusted = adjust_for_splits(frame(a + b), splits_frame([split("A", days[5], 2.0)]))
    assert by_day(adjusted, "B", "close") == {r["session_date"]: r["close"] for r in b}

    both = period_bars(frame(a + b), "W")
    alone = period_bars(frame(b), "W")
    pd.testing.assert_frame_equal(
        of(both, "B")[PERIOD_COLUMNS],
        alone[PERIOD_COLUMNS].reset_index(drop=True),
    )

    features_both = bx_features(frame(a + b))
    features_alone = bx_features(frame(b))
    pd.testing.assert_frame_equal(
        of(features_both, "B")[BX_COLUMNS],
        features_alone[BX_COLUMNS].reset_index(drop=True),
    )


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1, 2, 3, 4, 5], [NAN, NAN, 100.0, 100.0, 100.0]),  # all up: rma(d) == 0
        ([5, 4, 3, 2, 1], [NAN, NAN, 0.0, 0.0, 0.0]),  # all down: rma(u) == 0
        ([3, 3, 3, 3, 3], [NAN, NAN, 100.0, 100.0, 100.0]),  # flat: rma(d) == 0 is checked first
    ],
    ids=["all_up", "all_down", "flat"],
)
def test_pine_rsi_edge_cases(values: list[float], expected: list[float]) -> None:
    assert_series(pine_rsi(s(values), 2), expected)


def test_first_defined_values() -> None:
    close = s([10 + i % 3 for i in range(12)])
    assert pine_ema(close, 20).notna().all()  # defined from the first row
    assert pine_rma(close, 5).iloc[:4].isna().all() and pine_rma(close, 5).iloc[4:].notna().all()
    assert pine_rsi(close, 5).iloc[:5].isna().all() and pine_rsi(close, 5).iloc[5:].notna().all()
    out = bx_histogram(close, 5, 20, 5)
    assert out.iloc[:5].isna().all() and out.iloc[5:].notna().all()
    assert bx_histogram(close.iloc[:5], 5, 20, 5).isna().all()


def test_series_index_is_kept() -> None:
    close = pd.Series([1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 4.0], index=range(100, 107))
    assert list(bx_histogram(close, 2, 3, 2).index) == list(close.index)


# =============================================================================================
# leakage: later rows, the running period, later-knowable and rejected splits
# =============================================================================================

LEAK_DAYS = sessions("2024-01-02", "2024-09-30")
LEAK_DAILY = frame(walk("A", LEAK_DAYS, seed=11) + walk("B", LEAK_DAYS[40:], seed=12, start=8.0))


@pytest.mark.parametrize(
    "cut",
    [
        date(2024, 3, 28),  # Thursday before Good Friday: that week is not over yet
        date(2024, 4, 1),  # the Monday that completes the Good Friday week and March
        date(2024, 5, 31),  # a Friday that ends a month
        date(2024, 7, 3),  # day before a holiday, mid-week
        date(2024, 8, 14),  # plain Wednesday
        date(2024, 9, 27),
    ],
)
def test_appending_later_rows_changes_nothing_known_at_the_cut(cut: date) -> None:
    prefix = where(LEAK_DAILY, LEAK_DAILY["session_date"] <= cut)

    for p in ("W", "M"):
        full = period_bars(LEAK_DAILY, p)
        part = period_bars(prefix, p)
        assert (part["complete_on"] <= cut).all()
        pd.testing.assert_frame_equal(
            part[PERIOD_COLUMNS].reset_index(drop=True),
            where(full, full["complete_on"] <= cut)[PERIOD_COLUMNS],
        )

    full_bx = bx_features(LEAK_DAILY)
    part_bx = bx_features(prefix)
    pd.testing.assert_frame_equal(
        part_bx[["security_id", "session_date", *BX_COLUMNS]].reset_index(drop=True),
        where(full_bx, full_bx["session_date"] <= cut)[
            ["security_id", "session_date", *BX_COLUMNS]
        ],
    )


def test_the_running_week_and_month_are_not_emitted() -> None:
    # the input ends Thursday 2024-08-15: that week and August are still running
    cut = date(2024, 8, 15)
    base = where(LEAK_DAILY, LEAK_DAILY["session_date"] <= cut)
    edited = base.copy()
    on_cut = edited["session_date"] == cut
    edited.loc[on_cut, "close"] = edited.loc[on_cut, "close"] * 3
    edited.loc[on_cut, "high"] = edited.loc[on_cut, "high"] * 3

    for p in ("W", "M"):
        assert cut not in period_bars(base, p)["period_end"].tolist()
        pd.testing.assert_frame_equal(period_bars(base, p), period_bars(edited, p))
    before, after = bx_features(base), bx_features(edited)
    pd.testing.assert_series_equal(before["bx_weekly"], after["bx_weekly"])
    pd.testing.assert_series_equal(before["bx_monthly"], after["bx_monthly"])
    assert not before["bx_daily"].equals(after["bx_daily"])  # the daily value does move


def test_weekly_value_appears_only_from_complete_on() -> None:
    out = bx_features(LEAK_DAILY)
    weeks = period_bars(LEAK_DAILY, "W")
    weekly = by_day(out, "A", "bx_weekly")
    session_days = sorted(weekly)
    a_weeks = of(weeks, "A")
    hist = bx_histogram(col(a_weeks, "close")).tolist()
    for i, on in enumerate(a_weeks["complete_on"].tolist()):
        if i < 6:
            continue
        assert weekly[on] == pytest.approx(hist[i])
        day_before = session_days[session_days.index(on) - 1]
        assert weekly[day_before] == pytest.approx(hist[i - 1])


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return market_root(tmp_path)


@pytest.fixture
def con() -> Iterator[duckdb.DuckDBPyConnection]:
    connection = duckdb.connect()
    yield connection
    connection.close()


def test_a_split_not_yet_knowable_is_not_applied(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # The record's available_at is moved two days past its ex-date (a late feed; constructed):
    # at 2024-06-11 09:00 New York the 06-10 bar exists but the split does not yet.
    late = known(NVDA_EX) + timedelta(days=2)
    write_split_rows(root, [{**split_row("NVDA", NVDA_EX, 1.0, 10.0), "available_at": late}])
    bars = frame(NVDA_ROWS[:4])

    before = visible_splits(con, root, late - timedelta(hours=15)).to_pandas()
    after = visible_splits(con, root, late).to_pandas()

    unadjusted = adjust_for_splits(bars, before)
    adjusted = adjust_for_splits(bars, after)
    assert by_day(unadjusted, "S_NVDA", "close")[date(2024, 6, 7)] == 1208.88
    assert by_day(adjusted, "S_NVDA", "close")[date(2024, 6, 7)] == pytest.approx(120.888)


def test_a_split_after_the_last_bar_is_invisible_at_that_bars_as_of(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # the 06-07 evening: bars through 06-07 visible, the 06-10 record exists on disk only
    write_split_rows(root, [split_row("NVDA", NVDA_EX, 1.0, 10.0)])
    as_of = datetime(2024, 6, 8, 0, 30, tzinfo=UTC)  # 20:30 New York on 06-07

    splits = visible_splits(con, root, as_of).to_pandas()
    out = adjust_for_splits(frame(NVDA_ROWS[:3]), splits)

    assert splits.empty
    assert by_day(out, "S_NVDA", "close")[date(2024, 6, 7)] == 1208.88


def test_rejected_split_records_never_reach_the_adjustment(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # HON 2026-06-29: a 2:1 reverse record (factor 0.5) the prices disprove (opens 1.04, 1.00,
    # 0.98 of the previous close; real). Prices here are constructed around 220.
    hon_ex = date(2026, 6, 29)
    write_split_rows(root, [split_row("HON", hon_ex, 2.0, 1.0, kind="reverse_split")])
    days = sessions("2026-06-22", "2026-07-02", holidays=set())
    bars = frame(flat_bars("S_HON", days, 220.0, "HON"))
    as_of = datetime(2026, 10, 27, tzinfo=UTC)

    shipped = visible_splits(con, root, as_of).to_pandas()  # default: configs/split_rejections
    explicit = visible_splits(con, root, as_of, rejected={("HON", hon_ex)}).to_pandas()
    unrejected = visible_splits(con, root, as_of, rejected=()).to_pandas()

    assert shipped.empty and explicit.empty and len(unrejected) == 1
    assert adjust_for_splits(bars, shipped)["close"].tolist() == [220.0] * len(days)
    assert adjust_for_splits(bars, explicit)["close"].tolist() == [220.0] * len(days)
