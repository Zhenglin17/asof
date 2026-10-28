"""Guards against a B-Xtrender histogram that does not match the chart it is copied from (wrong
EMA / RMA seeding, wrong RSI edge cases), against split-unadjusted prices faking a crash on the
ex-date, against weekly / monthly bars that leak the rest of their period (a running week used as
if it were finished, a holiday week emitted before it is known to be over), and against values
that change when later rows or later-knowable splits are added; and against a spin-off
faking a crash on its ex-date, or its factor being used before the child's price existed.

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
    bx_features(daily, *, spinoffs=None, l1=5, l2=20, l3=5) -> pd.DataFrame
        ``spinoffs``: asof.store.spinoffs.spinoff_events(...).to_pandas() (security_id, ex_date
        E, factor, status, known_on, ...); a "pending" row blanks the sessions from E on. Adds
        ``bx_since``: the first session of the price series a row's values come from (the
        security's first bar when no event applies). For a session s of a security with an
        event: s < E -> values as without the event; E <= s < known_on -> all three NaN;
        s >= known_on -> "adjust": computed on the series whose open / high / low / close
        before E are multiplied by factor, bx_since unchanged; "restart": computed on the
        series starting at E, bx_since = E. Several events compose (factors multiply, a later
        restart resets bx_since). No events -> exactly the output without ``spinoffs``.

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

Spin-off section: event rows are hand-made (status, factor and known_on chosen per test); the
one-shot vs day-by-day test builds a store (tests/store/test_spinoffs.py builders) and reads it
through the real visible_bars / spinoff_events. BX's invariance to scaling every price by one
constant (EMA linear, RSI a ratio of gains to losses) is what makes "s < E unchanged" hold while
the adjusted series rescales those sessions; it is tested on its own.

Assumptions a real-data check must confirm (tests/store/test_golden_cases.py):
- the matching rule attributes NVDA's 2024-06-10 split to NVDA and adjusts 2024-06-07 to 120.888.
- the rejected HON / AIIO records never reach the adjustment on the real store.
- with GE's two spin-offs applied, GE's BX matches TradingView (weekly 21.50 on 2025-06-27,
  monthly 28.93 / 22.20 / -19.61 for 2025-06 / 2025-12 / 2026-09), and NVS restarts on
  2023-10-04 (tests/store/test_golden_spinoffs.py).
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


# =============================================================================================
# spin-offs: scale invariance, the NaN window, adjust / restart, composition, bx_since
# =============================================================================================

SPIN_DAYS = sessions("2024-01-02", "2024-12-31")  # 252 sessions
E1, K1 = SPIN_DAYS[150], SPIN_DAYS[153]  # an adjust decided three sessions after its ex-date
E2, K2 = SPIN_DAYS[200], SPIN_DAYS[220]  # a later event decided twenty sessions after it
SPIN_DAILY = frame(walk("A", SPIN_DAYS, seed=21) + walk("B", SPIN_DAYS, seed=22, start=30.0))
EVENT_COLUMNS = [
    "security_id",
    "parent_symbol",
    "child_symbols",
    "ex_date",
    "ratio",
    "parent_prev_session",
    "parent_prev_close",
    "child_session",
    "child_closes",
    "factor",
    "total_return",
    "status",
    "reason",
    "known_on",
]
SINCE = "bx_since"


def event(
    security_id: str,
    ex_date: date,
    status: str,
    *,
    factor: float | None = None,
    known_on: date | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    """One spinoff_events row; only security_id, ex_date, status, factor and known_on matter
    to the feature, the rest is filled in the table's shape."""
    return {
        "security_id": security_id,
        "parent_symbol": security_id,
        "child_symbols": ["KID"],
        "ex_date": ex_date,
        "ratio": [0.5],
        "parent_prev_session": None,
        "parent_prev_close": None,
        "child_session": known_on if status == "adjust" else None,
        "child_closes": [10.0] if status == "adjust" else [],
        "factor": factor,
        "total_return": 0.0 if status == "adjust" else None,
        "status": status,
        "reason": reason,
        "known_on": known_on,
    }


def events_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame.from_records(rows, columns=EVENT_COLUMNS)


def scaled_before(daily: pd.DataFrame, security_id: str, day: date, k: float) -> pd.DataFrame:
    """``daily`` with the OHLC of ``security_id``'s sessions before ``day`` multiplied by k."""
    out = daily.copy()
    mask = (out["security_id"] == security_id) & (out["session_date"] < day)
    for column in ("open", "high", "low", "close"):
        out.loc[mask, column] = out.loc[mask, column] * k
    return out


def from_day(daily: pd.DataFrame, security_id: str, day: date) -> pd.DataFrame:
    """``daily`` with ``security_id``'s sessions before ``day`` dropped."""
    keep = (daily["security_id"] != security_id) | (daily["session_date"] >= day)
    return where(daily, keep)


def values(df: pd.DataFrame, security_id: str, lo: date, hi: date | None = None) -> pd.DataFrame:
    """The BX columns of ``security_id`` for lo <= session < hi, indexed by session."""
    part = of(df, security_id)
    mask = part["session_date"] >= lo
    if hi is not None:
        mask &= part["session_date"] < hi
    return pd.DataFrame(part.loc[mask, ["session_date", *BX_COLUMNS]]).set_index("session_date")


def assert_same_values(got: pd.DataFrame, expected: pd.DataFrame) -> None:
    """Same sessions, same NaN, values equal to 1e-9 (an implementation may compute the
    unchanged sessions on the rescaled series: equal up to rounding, by scale invariance)."""
    assert list(got.index) == list(expected.index)
    assert len(got) > 0
    for column in BX_COLUMNS:
        g = got[column].to_numpy(dtype=float)
        e = expected[column].to_numpy(dtype=float)
        assert (np.isnan(g) == np.isnan(e)).all(), column
        ok = ~np.isnan(e)
        assert g[ok] == pytest.approx(e[ok], rel=1e-9, abs=1e-9), column


def as_day(value: Any) -> date:
    return value if type(value) is date else date.fromisoformat(str(value)[:10])


def since(df: pd.DataFrame, security_id: str, lo: date, hi: date | None = None) -> set[date]:
    part = of(df, security_id)
    mask = part["session_date"] >= lo
    if hi is not None:
        mask &= part["session_date"] < hi
    return {as_day(v) for v in part.loc[mask, SINCE]}


# -- normal path -------------------------------------------------------------------------------


def test_bx_is_invariant_to_scaling_every_price() -> None:
    close = col(of(SPIN_DAILY, "A"), "close")
    base = bx_histogram(close)
    for k in (0.37, 0.8004, 12.5):
        got = bx_histogram(close * k)
        assert (got.isna() == base.isna()).all()
        assert got.dropna().tolist() == pytest.approx(base.dropna().tolist(), rel=1e-9, abs=1e-9)

    scaled = scaled_before(SPIN_DAILY, "A", date(2100, 1, 1), 0.25)  # every A price
    assert_same_values(
        values(bx_features(scaled), "A", SPIN_DAYS[0]),
        values(bx_features(SPIN_DAILY), "A", SPIN_DAYS[0]),
    )


def test_scaling_only_a_prefix_does_change_later_values() -> None:
    # why a spin-off needs handling at all: a level shift in the middle of a series is not
    # a scale change of the series
    shifted = bx_features(scaled_before(SPIN_DAILY, "A", E1, 0.6))
    plain = bx_features(SPIN_DAILY)
    after = (of(shifted, "A")["session_date"] >= E1).to_numpy()
    diff = np.abs(
        of(shifted, "A")["bx_daily"].to_numpy()[after]
        - of(plain, "A")["bx_daily"].to_numpy()[after]
    )
    assert np.nanmax(diff) > 1.0


def test_no_events_is_exactly_todays_output_plus_bx_since() -> None:
    plain = bx_features(SPIN_DAILY)
    for spin in (None, events_frame([])):
        out = bx_features(SPIN_DAILY, spinoffs=spin)
        pd.testing.assert_frame_equal(out[BX_COLUMNS], plain[BX_COLUMNS])
        assert since(out, "A", SPIN_DAYS[0]) == {SPIN_DAYS[0]}
        assert since(out, "B", SPIN_DAYS[0]) == {SPIN_DAYS[0]}


def test_bx_since_is_each_securitys_own_first_bar() -> None:
    late_start = frame(walk("A", SPIN_DAYS, seed=21) + walk("C", SPIN_DAYS[40:], seed=23))
    out = bx_features(late_start, spinoffs=events_frame([]))
    assert since(out, "A", SPIN_DAYS[0]) == {SPIN_DAYS[0]}
    assert since(out, "C", SPIN_DAYS[0]) == {SPIN_DAYS[40]}


def test_adjust_event_before_window_after() -> None:
    spin = events_frame([event("A", E1, "adjust", factor=0.8, known_on=K1)])
    out = bx_features(SPIN_DAILY, spinoffs=spin)
    plain = bx_features(SPIN_DAILY)
    adjusted = bx_features(scaled_before(SPIN_DAILY, "A", E1, 0.8))

    # before the ex-date: as if nothing happened
    assert_same_values(values(out, "A", SPIN_DAYS[0], E1), values(plain, "A", SPIN_DAYS[0], E1))
    # from the ex-date until known_on: nothing (the child's price is not known yet)
    window = values(out, "A", E1, K1)
    assert len(window) == 3 and window.isna().all().all()
    # from known_on: the adjusted series
    assert_same_values(values(out, "A", K1), values(adjusted, "A", K1))
    assert since(out, "A", SPIN_DAYS[0], E1) == {SPIN_DAYS[0]}
    assert since(out, "A", K1) == {SPIN_DAYS[0]}


def test_restart_event_before_window_after() -> None:
    spin = events_frame([event("A", E2, "restart", known_on=K2, reason="no_child_price")])
    out = bx_features(SPIN_DAILY, spinoffs=spin)
    plain = bx_features(SPIN_DAILY)
    restarted = bx_features(from_day(SPIN_DAILY, "A", E2))

    assert_same_values(values(out, "A", SPIN_DAYS[0], E2), values(plain, "A", SPIN_DAYS[0], E2))
    window = values(out, "A", E2, K2)
    assert len(window) == 20 and window.isna().all().all()
    assert_same_values(values(out, "A", K2), values(restarted, "A", K2))
    assert not np.isnan(values(out, "A", K2)["bx_daily"].iloc[0])  # 20 rows in: defined
    assert since(out, "A", K2) == {E2}
    assert since(out, "A", SPIN_DAYS[0], E2) == {SPIN_DAYS[0]}


def test_events_of_one_security_leave_another_untouched() -> None:
    spin = events_frame(
        [
            event("A", E1, "adjust", factor=0.5, known_on=K1),
            event("A", E2, "restart", known_on=K2, reason="child_too_late"),
        ]
    )
    out = bx_features(SPIN_DAILY, spinoffs=spin)
    plain = bx_features(SPIN_DAILY)

    pd.testing.assert_frame_equal(of(out, "B")[BX_COLUMNS], of(plain, "B")[BX_COLUMNS])
    assert since(out, "B", SPIN_DAYS[0]) == {SPIN_DAYS[0]}


# -- boundaries --------------------------------------------------------------------------------


def test_adjust_known_on_its_ex_date_has_no_nan_window() -> None:
    # GE -> GEV: the child traded on the ex-date itself
    spin = events_frame([event("A", E1, "adjust", factor=0.8004, known_on=E1)])
    out = bx_features(SPIN_DAILY, spinoffs=spin)
    adjusted = bx_features(scaled_before(SPIN_DAILY, "A", E1, 0.8004))

    assert_same_values(values(out, "A", E1), values(adjusted, "A", E1))
    assert not values(out, "A", E1).isna().any().any()


def test_two_adjusts_compose() -> None:
    spin = events_frame(
        [
            event("A", E1, "adjust", factor=0.8, known_on=K1),
            event("A", E2, "adjust", factor=0.5, known_on=K2),
        ]
    )
    out = bx_features(SPIN_DAILY, spinoffs=spin)
    first_only = bx_features(scaled_before(SPIN_DAILY, "A", E1, 0.8))
    both = bx_features(scaled_before(scaled_before(SPIN_DAILY, "A", E1, 0.8), "A", E2, 0.5))

    assert_same_values(values(out, "A", K1, E2), values(first_only, "A", K1, E2))
    assert values(out, "A", E2, K2).isna().all().all()
    assert_same_values(values(out, "A", K2), values(both, "A", K2))
    assert since(out, "A", K2) == {SPIN_DAYS[0]}


def test_adjust_then_restart_resets_bx_since() -> None:
    spin = events_frame(
        [
            event("A", E1, "adjust", factor=0.8, known_on=K1),
            event("A", E2, "restart", known_on=K2, reason="no_child_price"),
        ]
    )
    out = bx_features(SPIN_DAILY, spinoffs=spin)
    first_only = bx_features(scaled_before(SPIN_DAILY, "A", E1, 0.8))
    restarted = bx_features(from_day(SPIN_DAILY, "A", E2))

    assert_same_values(values(out, "A", K1, E2), values(first_only, "A", K1, E2))
    assert since(out, "A", K1, E2) == {SPIN_DAYS[0]}
    assert_same_values(values(out, "A", K2), values(restarted, "A", K2))
    assert since(out, "A", K2) == {E2}


def test_restart_then_adjust_keeps_the_restart_start() -> None:
    k1 = SPIN_DAYS[170]  # the restart's known_on, before E2
    spin = events_frame(
        [
            event("A", E1, "restart", known_on=k1, reason="no_child_price"),
            event("A", E2, "adjust", factor=0.7, known_on=K2),
        ]
    )
    out = bx_features(SPIN_DAILY, spinoffs=spin)
    restarted = bx_features(from_day(SPIN_DAILY, "A", E1))
    both = bx_features(scaled_before(from_day(SPIN_DAILY, "A", E1), "A", E2, 0.7))

    assert_same_values(values(out, "A", k1, E2), values(restarted, "A", k1, E2))
    assert_same_values(values(out, "A", K2), values(both, "A", K2))
    assert since(out, "A", k1, E2) == {E1}
    assert since(out, "A", K2) == {E1}


def test_pending_event_blanks_from_its_ex_date() -> None:
    # undecided at as_of: the sessions from the ex-date on have no trustworthy series yet, as in
    # the NaN window of a decided event (one-shot == day-by-day needs exactly this)
    spin = events_frame([event("A", E2, "pending")])
    out = bx_features(SPIN_DAILY, spinoffs=spin)
    plain = bx_features(SPIN_DAILY)

    assert_same_values(values(out, "A", SPIN_DAYS[0], E2), values(plain, "A", SPIN_DAYS[0], E2))
    assert values(out, "A", E2).isna().all().all()
    pd.testing.assert_frame_equal(of(out, "B")[BX_COLUMNS], of(plain, "B")[BX_COLUMNS])
    assert since(out, "A", SPIN_DAYS[0]) == {SPIN_DAYS[0]}


def test_an_event_after_the_last_bar_changes_nothing() -> None:
    last = SPIN_DAYS[-1]
    spin = events_frame(
        [event("A", date(2025, 1, 6), "adjust", factor=0.8, known_on=date(2025, 1, 6))]
    )
    out = bx_features(SPIN_DAILY, spinoffs=spin)
    plain = bx_features(SPIN_DAILY)

    assert_same_values(
        values(out, "A", SPIN_DAYS[0], last + timedelta(days=1)),
        values(plain, "A", SPIN_DAYS[0], last + timedelta(days=1)),
    )


# -- leakage -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "other",
    [
        event("A", E1, "adjust", factor=0.3, known_on=K1),
        event("A", E1, "adjust", factor=0.99, known_on=K1),
        event("A", E1, "restart", known_on=K1, reason="factor_out_of_range"),
    ],
    ids=["factor_0.3", "factor_0.99", "restart"],
)
def test_the_decision_cannot_reach_sessions_before_known_on(other: dict[str, Any]) -> None:
    # (d) whatever the event turns out to be, every session before known_on reads the same
    base = bx_features(
        SPIN_DAILY, spinoffs=events_frame([event("A", E1, "adjust", factor=0.8, known_on=K1)])
    )
    alt = bx_features(SPIN_DAILY, spinoffs=events_frame([other]))

    assert_same_values(values(alt, "A", SPIN_DAYS[0], K1), values(base, "A", SPIN_DAYS[0], K1))
    assert since(alt, "A", SPIN_DAYS[0], E1) == since(base, "A", SPIN_DAYS[0], E1)


def test_later_rows_change_nothing_known_at_a_cut_with_events() -> None:
    # the prefix-stability property of the plain feature, with a decided event in the prefix
    spin = events_frame([event("A", E1, "adjust", factor=0.8, known_on=K1)])
    cut = SPIN_DAYS[190]
    full = bx_features(SPIN_DAILY, spinoffs=spin)
    part = bx_features(where(SPIN_DAILY, SPIN_DAILY["session_date"] <= cut), spinoffs=spin)

    for sid in ("A", "B"):
        assert_same_values(
            values(part, sid, SPIN_DAYS[0]),
            values(full, sid, SPIN_DAYS[0], cut + timedelta(days=1)),
        )


def test_one_shot_equals_day_by_day_on_a_store(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    # (e) the feature computed once with every event known at the end equals, for every
    # session, the feature computed from what was visible at that session's 20:00 New York.
    # P1: spin-off on E (index 150), child K1 first trades three sessions later (adjust).
    # P2: spin-off on index 170, child never trades (restart twenty sessions later).
    from asof.store.market import visible_bars
    from asof.store.spinoffs import spinoff_events
    from tests.store.audit_world import bar as raw_bar
    from tests.store.test_market import Seg, avail, write_master
    from tests.store.test_spinoffs import spin_row, write_bars, write_other_rows

    days = sessions("2024-01-02", "2024-10-31")
    e1, e2 = days[150], days[170]
    k1_first = days[153]

    def raw(
        symbol: str, rows: list[dict[str, Any]], scale_from: date | None = None, k: float = 1.0
    ) -> list[dict[str, Any]]:
        out = []
        for r in rows:
            m = k if scale_from is not None and r["session_date"] >= scale_from else 1.0
            out.append(
                raw_bar(
                    symbol,
                    r["session_date"],
                    close=round(r["close"] * m, 4),
                    open=round(r["open"] * m, 4),
                    high=round(r["high"] * m, 4),
                    low=round(r["low"] * m, 4),
                    volume=10_000,
                )
            )
        return out

    bars = (
        raw("P1", walk("x", days, seed=31, start=100.0), e1, 0.7)
        + raw("P2", walk("x", days, seed=32, start=40.0), e2, 0.8)
        + raw("K1", walk("x", [d for d in days if d >= k1_first], seed=33, start=60.0))
    )
    write_bars(root, bars)
    write_master(
        root,
        [
            Seg("S_P1", "P1"),
            Seg("S_P2", "P2"),
            Seg("S_K1", "K1", valid_from=k1_first, available_at=known(k1_first)),
        ],
    )
    write_other_rows(root, [spin_row("P1", "K1", e1, 0.5), spin_row("P2", "K2", e2, 0.2)])

    def features(as_of: datetime) -> pd.DataFrame:
        daily = visible_bars(con, root, "1Day", as_of).to_pandas()
        spin = spinoff_events(con, root, as_of, rejected=()).to_pandas()
        return bx_features(daily, spinoffs=spin)

    end = datetime(2026, 10, 27, tzinfo=UTC)
    final_events = spinoff_events(con, root, end, rejected=()).to_pylist()
    assert sorted((r["security_id"], r["status"]) for r in final_events) == [
        ("S_P1", "adjust"),
        ("S_P2", "restart"),
    ]
    windows = {r["security_id"]: (r["ex_date"], r["known_on"]) for r in final_events}
    assert windows["S_P1"] == (e1, k1_first) and windows["S_P2"] == (e2, days[190])
    once = features(end)

    for day in [*days[148:196], *days[-3:]]:
        seen = features(avail(day))
        for sid in ("S_P1", "S_P2", "S_K1"):
            now = seen.loc[(seen["security_id"] == sid) & (seen["session_date"] == day)]
            then = once.loc[(once["security_id"] == sid) & (once["session_date"] == day)]
            assert len(now) == len(then), (sid, day)
            if now.empty:
                continue
            for column in BX_COLUMNS:
                a, b = float(now[column].iloc[0]), float(then[column].iloc[0])
                assert (math.isnan(a) and math.isnan(b)) or a == pytest.approx(
                    b, rel=1e-9, abs=1e-9
                ), (sid, day, column, a, b)
            ex, known_on = windows.get(sid, (None, None))
            if ex is None or not (ex <= day < known_on):
                assert as_day(now[SINCE].iloc[0]) == as_day(then[SINCE].iloc[0]), (sid, day)


# =============================================================================================
# previous values and bar counts (2-c): bx_*_prev, daily_bars / weekly_bars / monthly_bars
# =============================================================================================
#
# Target (added to bx_features, computed in the same per-series pass as the BX columns):
#   bx_daily_prev    bx_daily of the previous row of the same series; NaN on a series' first
#                    row and wherever that previous value is NaN.
#   bx_weekly_prev   histogram of the completed week before the one bx_weekly reads (same
#   bx_monthly_prev  complete_on as-of join); NaN when the series has no such period.
#   daily_bars       rows of the series from bx_since through the session, inclusive.
#   weekly_bars      completed periods of the series whose complete_on <= session.
#   monthly_bars
# In a spin-off blank stretch (E <= session < known_on, or pending) all six are NaN like the
# BX columns. "Previous row of the same series" on the first session after a blank stretch is
# read here as the previous row of the series that session's values come from (the restarted
# or adjusted series), not the blanked output row before it.

PREV_COLUMNS = ["bx_daily_prev", "bx_weekly_prev", "bx_monthly_prev"]
COUNT_COLUMNS = ["daily_bars", "weekly_bars", "monthly_bars"]
NEW_COLUMNS = PREV_COLUMNS + COUNT_COLUMNS


def reference(daily: pd.DataFrame, l1: int = 5, l2: int = 20, l3: int = 5) -> pd.DataFrame:
    """The BX columns, the six new columns and ``weekly_complete_on`` / ``monthly_complete_on``
    (complete_on of the period the value comes from) of plain daily rows (no events), computed
    per security straight from ``period_bars`` and ``bx_histogram``: the k-th completed period
    is visible from its complete_on, its value is hist[k - 1] and the previous one hist[k - 2]."""
    parts = []
    ordered = daily.sort_values(["security_id", "session_date"])
    for sid, g in ordered.groupby("security_id", sort=True):
        g = g.reset_index(drop=True)
        days = np.array([d.toordinal() for d in g["session_date"]])
        out = pd.DataFrame({"security_id": sid, "session_date": g["session_date"]})
        out["bx_daily"] = bx_histogram(pd.Series(g["close"], dtype=float), l1, l2, l3).to_numpy()
        out["bx_daily_prev"] = out["bx_daily"].shift(1)
        out["daily_bars"] = np.arange(1, len(g) + 1, dtype=float)
        for p, name in (("W", "weekly"), ("M", "monthly")):
            bars = period_bars(g, p)  # type: ignore[arg-type]
            hist = bx_histogram(pd.Series(bars["close"], dtype=float), l1, l2, l3).to_numpy()
            done = np.array([d.toordinal() for d in bars["complete_on"]])
            assert (np.diff(done) > 0).all()  # fixture: one period completes per session at most
            k = np.searchsorted(done, days, side="right")
            out[f"bx_{name}"] = [hist[i - 1] if i >= 1 else NAN for i in k]
            out[f"bx_{name}_prev"] = [hist[i - 2] if i >= 2 else NAN for i in k]
            out[f"{name}_bars"] = k.astype(float)
            out[f"{name}_complete_on"] = [bars["complete_on"].iloc[i - 1] if i else None for i in k]
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def assert_columns_close(
    got: pd.DataFrame, expected: pd.DataFrame, columns: list[str], tol: float = 1e-9
) -> None:
    """Row-aligned frames: same NaN pattern and values within ``tol`` per column."""
    assert len(got) == len(expected) and len(got) > 0
    for column in columns:
        g = pd.Series(got[column]).astype(float).to_numpy(dtype=float)
        e = pd.Series(expected[column]).astype(float).to_numpy(dtype=float)
        assert (np.isnan(g) == np.isnan(e)).all(), (column, g[np.isnan(g) != np.isnan(e)][:5])
        ok = ~np.isnan(e)
        assert g[ok] == pytest.approx(e[ok], rel=tol, abs=tol), column


def rows_of(df: pd.DataFrame, security_id: str, lo: date, hi: date | None = None) -> pd.DataFrame:
    part = of(df, security_id)
    mask = part["session_date"] >= lo
    if hi is not None:
        mask &= part["session_date"] < hi
    return where(part, mask)


def at(df: pd.DataFrame, security_id: str, day: date) -> dict[str, Any]:
    [i] = np.flatnonzero((df["security_id"] == security_id) & (df["session_date"] == day))
    return df.iloc[int(i)].to_dict()


# -- normal path -------------------------------------------------------------------------------


def test_new_columns_exist_and_match_the_reference() -> None:
    out = bx_features(LEAK_DAILY)
    expected = reference(LEAK_DAILY)

    assert set(NEW_COLUMNS) <= set(out.columns)
    assert_columns_close(out, expected, BX_COLUMNS + NEW_COLUMNS)


def test_bx_daily_prev_and_daily_bars_by_hand() -> None:
    # nine sessions; bx_daily is first defined on row 5 (l3 = 5), so its prev from row 6
    days = sessions("2024-01-02", "2024-01-12")
    closes = [10.0, 11.0, 10.5, 12.0, 11.0, 13.0, 12.5, 14.0, 13.0]
    rows = [bar("A", d, c, c, c, c) for d, c in zip(days, closes, strict=True)]
    out = bx_features(frame(rows))

    hist = bx_histogram(s(closes)).tolist()
    assert_series(col(out, "bx_daily_prev"), [NAN, *hist[:-1]])
    assert col(out, "bx_daily_prev").iloc[:6].isna().all()
    assert col(out, "bx_daily_prev").iloc[6] == pytest.approx(hist[5])
    assert col(out, "daily_bars").tolist() == list(range(1, 10))


def test_weekly_prev_is_the_week_before_the_one_in_bx_weekly() -> None:
    out = of(bx_features(LEAK_DAILY), "A")
    weeks = of(period_bars(LEAK_DAILY, "W"), "A")
    hist = bx_histogram(col(weeks, "close")).tolist()
    on = weeks["complete_on"].tolist()
    by = dict(zip(out["session_date"], range(len(out)), strict=True))
    for i in range(7, len(on)):
        row = out.iloc[by[on[i]]]
        assert row["bx_weekly"] == pytest.approx(hist[i])
        assert row["bx_weekly_prev"] == pytest.approx(hist[i - 1])
        assert row["weekly_bars"] == i + 1


def test_counts_restart_for_a_security_that_starts_later() -> None:
    out = bx_features(LEAK_DAILY)
    b = of(out, "B")  # B's first bar is LEAK_DAYS[40]
    assert b["daily_bars"].iloc[0] == 1
    assert b["weekly_bars"].iloc[0] == 0 and b["monthly_bars"].iloc[0] == 0
    assert math.isnan(b["bx_daily_prev"].iloc[0])


# -- boundaries --------------------------------------------------------------------------------


def test_an_open_week_neither_counts_nor_moves_prev() -> None:
    # 2024-08-12 .. 08-16 (Mon .. Fri): the count and prev stand still until Friday's close
    out = bx_features(LEAK_DAILY)
    week = [date(2024, 8, d) for d in (12, 13, 14, 15)]
    friday = date(2024, 8, 16)
    for column in ("weekly_bars", "bx_weekly_prev", "bx_weekly"):
        values_ = {at(out, "A", d)[column] for d in week}
        assert len(values_) == 1, column
    assert at(out, "A", friday)["weekly_bars"] == at(out, "A", week[-1])["weekly_bars"] + 1
    assert at(out, "A", friday)["bx_weekly_prev"] == pytest.approx(
        at(out, "A", week[-1])["bx_weekly"]
    )


def test_editing_the_running_wednesday_does_not_move_period_prev_or_counts() -> None:
    cut = date(2024, 8, 14)  # Wednesday: the week and August are open
    base = where(LEAK_DAILY, LEAK_DAILY["session_date"] <= cut)
    edited = base.copy()
    on_cut = edited["session_date"] == cut
    edited.loc[on_cut, ["close", "high"]] = edited.loc[on_cut, ["close", "high"]] * 1.5

    before, after = bx_features(base), bx_features(edited)
    for column in ("bx_weekly_prev", "bx_monthly_prev", *COUNT_COLUMNS):
        pd.testing.assert_series_equal(before[column], after[column])
    assert not before["bx_daily"].equals(after["bx_daily"])
    # the daily prev of the cut row reads the day before, which did not change either
    pd.testing.assert_series_equal(before["bx_daily_prev"], after["bx_daily_prev"])


def test_good_friday_week_and_march_count_on_the_next_session() -> None:
    # from mid-2023 so that March 2024's monthly value and the one before it are defined
    out = bx_features(frame(walk("A", sessions("2023-07-03", "2024-04-12"), seed=5)))
    thursday, monday = date(2024, 3, 28), date(2024, 4, 1)
    th, mo = at(out, "A", thursday), at(out, "A", monday)

    assert mo["weekly_bars"] == th["weekly_bars"] + 1
    assert mo["monthly_bars"] == th["monthly_bars"] + 1
    assert mo["bx_weekly_prev"] == pytest.approx(th["bx_weekly"])
    assert mo["bx_monthly_prev"] == pytest.approx(th["bx_monthly"])
    assert not math.isnan(mo["bx_monthly_prev"])


def test_a_restart_known_on_its_ex_date_starts_prev_and_counts_again() -> None:
    # parent and child on one symbol: restart decided on the ex-date itself (no blank stretch)
    spin = events_frame([event("A", E2, "restart", known_on=E2, reason="same_symbol")])
    out = bx_features(SPIN_DAILY, spinoffs=spin)
    restarted = reference(from_day(SPIN_DAILY, "A", E2))
    plain = reference(SPIN_DAILY)

    first = at(out, "A", E2)
    assert all(math.isnan(first[c]) for c in PREV_COLUMNS)
    assert first["daily_bars"] == 1
    assert_columns_close(rows_of(out, "A", E2), of(restarted, "A"), NEW_COLUMNS)
    assert_columns_close(
        rows_of(out, "A", SPIN_DAYS[0], E2),
        rows_of(plain, "A", SPIN_DAYS[0], E2),
        NEW_COLUMNS,
    )


def test_a_restart_decided_later_counts_from_its_ex_date() -> None:
    spin = events_frame([event("A", E2, "restart", known_on=K2, reason="no_child_price")])
    out = bx_features(SPIN_DAILY, spinoffs=spin)
    restarted = reference(from_day(SPIN_DAILY, "A", E2))

    assert at(out, "A", K2)["daily_bars"] == SPIN_DAYS.index(K2) - SPIN_DAYS.index(E2) + 1
    assert_columns_close(rows_of(out, "A", K2), rows_of(restarted, "A", K2), NEW_COLUMNS)


def test_an_adjust_keeps_counting_from_the_first_bar() -> None:
    spin = events_frame([event("A", E1, "adjust", factor=0.8, known_on=K1)])
    out = bx_features(SPIN_DAILY, spinoffs=spin)
    adjusted = reference(scaled_before(SPIN_DAILY, "A", E1, 0.8))

    assert at(out, "A", K1)["daily_bars"] == SPIN_DAYS.index(K1) + 1
    assert_columns_close(rows_of(out, "A", K1), rows_of(adjusted, "A", K1), NEW_COLUMNS)


def test_blank_stretches_blank_every_new_column() -> None:
    spin = events_frame(
        [
            event("A", E1, "adjust", factor=0.8, known_on=K1),
            event("B", E2, "pending"),
        ]
    )
    out = bx_features(SPIN_DAILY, spinoffs=spin)

    window = rows_of(out, "A", E1, K1)
    pending = rows_of(out, "B", E2)
    for part in (window, pending):
        assert len(part) > 0
        assert pd.DataFrame(part[BX_COLUMNS + NEW_COLUMNS]).isna().to_numpy().all()
    # outside the stretches every count is defined
    assert pd.DataFrame(rows_of(out, "A", K1)[COUNT_COLUMNS]).notna().to_numpy().all()
    assert pd.DataFrame(rows_of(out, "B", SPIN_DAYS[0], E2)[COUNT_COLUMNS]).notna().to_numpy().all()


# -- leakage -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cut",
    [
        date(2024, 3, 28),  # Thursday before Good Friday
        date(2024, 4, 1),  # completes the Good Friday week and March
        date(2024, 8, 14),  # Wednesday, week and month open
        date(2024, 5, 31),  # Friday ending a month
    ],
)
def test_later_rows_change_no_new_column_known_at_the_cut(cut: date) -> None:
    full = bx_features(LEAK_DAILY)
    part = bx_features(where(LEAK_DAILY, LEAK_DAILY["session_date"] <= cut))
    assert_columns_close(part, where(full, full["session_date"] <= cut), NEW_COLUMNS, tol=0.0)


def test_later_rows_change_no_new_column_with_a_decided_event() -> None:
    spin = events_frame([event("A", E1, "adjust", factor=0.8, known_on=K1)])
    cut = SPIN_DAYS[190]
    full = bx_features(SPIN_DAILY, spinoffs=spin)
    part = bx_features(where(SPIN_DAILY, SPIN_DAILY["session_date"] <= cut), spinoffs=spin)
    assert_columns_close(part, where(full, full["session_date"] <= cut), NEW_COLUMNS)


def test_a_series_with_no_completed_period_reads_nan_not_an_error() -> None:
    # a listing in its first week: no completed week or month anywhere in the input
    listing = frame(walk("N", sessions("2024-01-02", "2024-01-04"), seed=1))
    out = bx_features(listing)

    assert col(out, "bx_weekly").isna().all() and col(out, "bx_monthly").isna().all()
    assert col(out, "weekly_bars").tolist() == [0, 0, 0]
    assert col(out, "daily_bars").tolist() == [1, 2, 3]


def test_a_restart_read_on_its_ex_date_evening() -> None:
    # day by day, the evening of a restart decided on its ex-date sees one row of the new series
    # and no completed period of it (real shape: CR -> CR 2023-04-04, same_symbol)
    spin = events_frame([event("A", E2, "restart", known_on=E2, reason="same_symbol")])
    for last in (E2, SPIN_DAYS[SPIN_DAYS.index(E2) + 3]):
        out = bx_features(where(SPIN_DAILY, SPIN_DAILY["session_date"] <= last), spinoffs=spin)
        row = at(out, "A", E2)
        assert row["daily_bars"] == 1 and row["weekly_bars"] == 0 and row["monthly_bars"] == 0
        assert math.isnan(row["bx_weekly"]) and math.isnan(row["bx_daily_prev"])
