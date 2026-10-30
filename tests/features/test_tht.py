"""Guards against a THT Fair Value Bands trend filter that does not follow the reverse-engineered
rule (wrong SMA window, a bar touching the line counted as a crossing, a threshold computed over
the wrong crossings, green checked before red, a colour that does not carry over), against one
security's bars leaking into another's, against values that change when later rows or
later-knowable splits are added, and against a spin-off whose factor reaches sessions before the
child's price existed.

Target API (asof.features.tht):

    tht_features(daily, *, spinoffs=None, length=33, min_crossings=20, median_window=None)
        ``daily``: security_id, symbol, session_date, open, high, low, close (+ other columns,
        passed through), already split-adjusted by the caller. Returns it sorted by security
        and session; two rows for one security and session -> ValueError. Adds, per security
        and forward only:
        tht_fv         mean of OHLC4 = (o + h + l + c) / 4 over the last ``length`` rows,
                       the current one included; NaN before ``length`` rows.
        tht_crossings  rows up to and including this one with low < tht_fv < high (strict).
        tht_hu         median of high / fv over those crossings (the last ``median_window`` of
        tht_ld         them when it is an int); low / fv likewise. NaN before the first one.
        tht_color      None while fv is NaN or crossings < min_crossings; else "red" if
                       high < fv * ld, elif "green" if low > fv * hu, else the previous row's
                       colour (None carries as None).
        tht_since      first session of the series the row's values come from.
        ``spinoffs``: the frame bx_features takes (asof.store.spinoffs.spinoff_events), same
        state rules: s < E unchanged; E <= s < known_on (or pending) blank (numbers NaN,
        colour None); s >= known_on "adjust" = the series with prices before E times factor
        (values from E on DO change: an SMA is not scale-invariant), "restart" = the series
        starting at E (tht_since = E).

Source of the rule: TradingView "THT Fair Value Bands" is closed source; the rule is a
reverse-engineering of quantifytools' open "Fair value bands" (OHLC4 SMA, Cross
mode, median thresholds over crossing bars), checked against the chart on MSFT daily (six flips
on the exact day) and MU weekly. The fake numbers below are hand-made (no data source is
imitated); the NVDA split rows reuse test_bx's NVDA_EX (10-for-1, ex 2024-06-10, public record).
Every value in the hand-computed tests is dyadic (exact in binary), so the equality boundaries
(high == fv * ld, fv == low) are exact and not at the mercy of rounding.

Assumptions a real-data check must confirm (tests/features/test_golden_tht.py):
- the rule (SMA 33 of OHLC4, strict crossings, expanding medians, red first, carry-over) is what
  the chart draws: MSFT daily flips 2025-08-21, 10-03, 11-06, 2026-04-15, 06-09, 07-21 and fv
  514.82 / 508.28 / 518.37 must come out on real bars.
- the chart works on split-adjusted prices: NVDA's fv and colour continue across 2024-06-10.
- min_crossings = 20 and median_window = None reproduce the chart. The chart legend's "70" is
  unexplained: all-history and last-70 medians give the same MSFT flips, so the golden case
  cannot tell them apart.
"""

import math
from collections.abc import Iterator, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pytest

from asof.features.bx import adjust_for_splits
from asof.features.tht import tht_features
from asof.store.market import market_root, visible_splits
from tests.features.test_bx import (
    NVDA_EX,
    as_day,
    bar,
    event,
    events_frame,
    frame,
    from_day,
    of,
    scaled_before,
    sessions,
    walk,
    where,
)
from tests.store.test_market import known
from tests.store.test_split_events import split_row, write_split_rows

NAN = float("nan")
NUM_COLUMNS = ["tht_fv", "tht_crossings", "tht_hu", "tht_ld"]
COLOR = "tht_color"
SINCE = "tht_since"
DAYS = sessions("2024-01-02", "2024-03-28")  # enough weekdays for the hand-made series


# =============================================================================================
# builders
# =============================================================================================


def series(bars: Sequence[tuple[float, float, float, float]], sid: str = "A") -> pd.DataFrame:
    """One security's (open, high, low, close) rows on consecutive sessions."""
    return frame([bar(sid, DAYS[i], o, h, lo, c) for i, (o, h, lo, c) in enumerate(bars)])


def flat(p: float) -> tuple[float, float, float, float]:
    return (p, p, p, p)


def colors(df: pd.DataFrame) -> list[str | None]:
    """tht_color as a list; a missing colour (None, or NaN after a frame operation) -> None."""
    return [
        None if v is None or (isinstance(v, float) and math.isnan(v)) else str(v) for v in df[COLOR]
    ]


def counts(df: pd.DataFrame) -> list[int]:
    out = []
    for v in df["tht_crossings"]:
        assert float(v).is_integer(), v
        out.append(int(v))
    return out


def assert_floats(got: Any, expected: list[float]) -> None:
    g = got.tolist()
    assert len(g) == len(expected), (g, expected)
    for a, e in zip(g, expected, strict=True):
        if math.isnan(e):
            assert math.isnan(a), (g, expected)
        else:
            assert a == pytest.approx(e, rel=1e-12, abs=1e-12), (g, expected)


def values(df: pd.DataFrame, security_id: str, lo: date, hi: date | None = None) -> pd.DataFrame:
    """The THT columns of ``security_id`` for lo <= session < hi, indexed by session."""
    part = of(df, security_id)
    mask = part["session_date"] >= lo
    if hi is not None:
        mask &= part["session_date"] < hi
    columns = ["session_date", *NUM_COLUMNS, COLOR]
    return pd.DataFrame(part.loc[mask, columns]).set_index("session_date")


def assert_same_values(got: pd.DataFrame, expected: pd.DataFrame) -> None:
    """Same sessions, same NaN, numbers equal to 1e-9, same colours."""
    assert list(got.index) == list(expected.index)
    assert len(got) > 0
    for column in NUM_COLUMNS:
        g = got[column].to_numpy(dtype=float)
        e = expected[column].to_numpy(dtype=float)
        assert (np.isnan(g) == np.isnan(e)).all(), column
        ok = ~np.isnan(e)
        assert g[ok] == pytest.approx(e[ok], rel=1e-9, abs=1e-9), column
    assert colors(got) == colors(expected)


def blank(df: pd.DataFrame) -> bool:
    return bool(np.isnan(df[NUM_COLUMNS].to_numpy(dtype=float)).all()) and all(
        c is None for c in colors(df)
    )


def since(df: pd.DataFrame, security_id: str, lo: date, hi: date | None = None) -> set[date]:
    part = of(df, security_id)
    mask = part["session_date"] >= lo
    if hi is not None:
        mask &= part["session_date"] < hi
    return {as_day(v) for v in part.loc[mask, SINCE]}


def reference(
    daily: pd.DataFrame, length: int, min_crossings: int, median_window: int | None
) -> pd.DataFrame:
    """A deliberately plain loop over each security's rows, written from the spec alone."""
    out = daily.sort_values(["security_id", "session_date"]).reset_index(drop=True)
    fv_all, n_all, hu_all, ld_all, color_all = [], [], [], [], []
    for _, g in out.groupby("security_id", sort=False):
        o, h, lo, c = (g[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
        ohlc4 = (o + h + lo + c) / 4
        ups: list[float] = []
        downs: list[float] = []
        prev: str | None = None
        for i in range(len(g)):
            fv = float(np.mean(ohlc4[i - length + 1 : i + 1])) if i >= length - 1 else NAN
            if not math.isnan(fv) and lo[i] < fv < h[i]:
                ups.append(h[i] / fv)
                downs.append(lo[i] / fv)
            window = slice(-median_window, None) if median_window else slice(None)
            hu = float(np.median(ups[window])) if ups else NAN
            ld = float(np.median(downs[window])) if downs else NAN
            if math.isnan(fv) or len(ups) < min_crossings:
                color = None
            elif h[i] < fv * ld:
                color = "red"
            elif lo[i] > fv * hu:
                color = "green"
            else:
                color = prev
            prev = color
            fv_all.append(fv)
            n_all.append(float(len(ups)))
            hu_all.append(hu)
            ld_all.append(ld)
            color_all.append(color)
    out["tht_fv"], out["tht_crossings"] = fv_all, n_all
    out["tht_hu"], out["tht_ld"], out[COLOR] = hu_all, ld_all, color_all
    return out


# =============================================================================================
# normal path: hand-computed values
# =============================================================================================


def test_fv_is_the_sma_of_ohlc4_including_the_current_row() -> None:
    # OHLC4: (10+11+9+10)/4 = 10, 12, 14, 16; length 3 -> NaN, NaN, (10+12+14)/3, (12+14+16)/3
    rows = [(10, 11, 9, 10), (12, 13, 11, 12), (14, 15, 13, 14), (16, 17, 15, 16)]
    out = tht_features(series(rows), length=3, min_crossings=1)
    assert_floats(out["tht_fv"], [NAN, NAN, 12.0, 14.0])


def test_ohlc4_uses_all_four_prices() -> None:
    # open 4 high 12 low 2 close 6 -> OHLC4 6 (close alone 6, HL2 7, open-close mean 5)
    out = tht_features(series([(4, 12, 2, 6)]), length=1, min_crossings=1)
    assert out["tht_fv"].tolist() == [6.0]


def test_crossings_and_medians_hand_computed() -> None:
    # length 2; every OHLC4 is 10, so fv = 10 from row 1 on.
    # row  bar               crossing  high/fv  low/fv   hu = median        ld = median
    # 0    flat 10           fv NaN    -        -        NaN                NaN
    # 1    10 12 8 10        yes       1.2      0.8      1.2                0.8
    # 2    10 11 9 10        yes       1.1      0.9      1.15               0.85
    # 3    flat 10           no: low == fv == high       1.15               0.85
    # 4    10 10.5 9.5 10    yes       1.05     0.95     1.1                0.9
    # 5    10 13 7 10        yes       1.3      0.7      (1.1+1.2)/2=1.15   (0.8+0.9)/2=0.85
    rows = [
        flat(10),
        (10, 12, 8, 10),
        (10, 11, 9, 10),
        flat(10),
        (10, 10.5, 9.5, 10),
        (10, 13, 7, 10),
    ]
    out = tht_features(series(rows), length=2, min_crossings=1)

    assert_floats(out["tht_fv"], [NAN, 10, 10, 10, 10, 10])
    assert counts(out) == [0, 1, 2, 2, 3, 4]
    assert_floats(out["tht_hu"], [NAN, 1.2, 1.15, 1.15, 1.1, 1.15])
    assert_floats(out["tht_ld"], [NAN, 0.8, 0.85, 0.85, 0.9, 0.85])


def test_median_window_uses_only_the_last_n_crossings() -> None:
    # same series, last 2 crossings: row 1 {1.2}, row 2 {1.2, 1.1}, row 3 unchanged,
    # row 4 {1.1, 1.05}, row 5 {1.05, 1.3}; low/fv mirrored
    rows = [
        flat(10),
        (10, 12, 8, 10),
        (10, 11, 9, 10),
        flat(10),
        (10, 10.5, 9.5, 10),
        (10, 13, 7, 10),
    ]
    out = tht_features(series(rows), length=2, min_crossings=1, median_window=2)

    assert counts(out) == [0, 1, 2, 2, 3, 4]  # the count is not windowed
    assert_floats(out["tht_hu"], [NAN, 1.2, 1.15, 1.15, 1.075, 1.175])
    assert_floats(out["tht_ld"], [NAN, 0.8, 0.85, 0.85, 0.925, 0.825])


def test_green_flip_carry_and_red_flip() -> None:
    # length 2, min_crossings 2.
    # row  bar             ohlc4  fv     crossing  hu / ld after         colour
    # 0    flat 10         10     NaN    -         NaN                   None (fv NaN)
    # 1    10 11 9 10      10     10     yes       1.1 / 0.9             None (1 crossing)
    # 2    10 11 9 10      10     10     yes       1.1 / 0.9             None (carried None)
    # 3    flat 13         13     11.5   no        1.1 / 0.9             green: 13 > 11.5*1.1
    # 4    13 14 12 13     13     13     yes       1.1 / 0.9 (median)    green carried
    # 5    flat 10         10     11.5   no                              red: 10 < 11.5*0.9
    # 6    10 10.5 9.5 10  10     10     yes       ~1.088 / ~0.912       red carried
    rows = [
        flat(10),
        (10, 11, 9, 10),
        (10, 11, 9, 10),
        flat(13),
        (13, 14, 12, 13),
        flat(10),
        (10, 10.5, 9.5, 10),
    ]
    out = tht_features(series(rows), length=2, min_crossings=2)

    assert_floats(out["tht_fv"], [NAN, 10, 10, 11.5, 13, 11.5, 10])
    assert counts(out) == [0, 1, 2, 2, 3, 3, 4]
    assert colors(out) == [None, None, None, "green", "green", "red", "red"]
    assert out["tht_hu"].iloc[6] == pytest.approx((14 / 13 + 1.1) / 2)
    assert out["tht_ld"].iloc[6] == pytest.approx((0.9 + 12 / 13) / 2)


def test_red_is_checked_before_green() -> None:
    # Both conditions can only hold on a malformed bar (hu > 1 > ld, so low > fv*hu > fv*ld >
    # high needs low > high); constructed only to pin the order of the two checks.
    # rows 0-2 as above (hu 1.1, ld 0.9, fv 10); row 3: open 10 high 8 low 12 close 10 ->
    # ohlc4 10, fv 10; high 8 < 9 (red) and low 12 > 11 (green) -> red
    rows = [flat(10), (10, 11, 9, 10), (10, 11, 9, 10), (10, 8, 12, 10)]
    out = tht_features(series(rows), length=2, min_crossings=2)
    assert colors(out) == [None, None, None, "red"]


def test_length_and_min_crossings_are_honoured() -> None:
    days = sessions("2024-01-02", "2024-12-31")
    daily = frame(walk("A", days, seed=41) + walk("B", days[30:], seed=42, start=20.0))
    for length, min_crossings, window in [(33, 20, None), (10, 5, None), (5, 50, 7), (20, 1, 3)]:
        out = tht_features(daily, length=length, min_crossings=min_crossings, median_window=window)
        ref = reference(daily, length, min_crossings, window)
        for sid in ("A", "B"):
            assert_same_values(values(out, sid, days[0]), values(ref, sid, days[0]))
        a = of(out, "A")
        assert a["tht_fv"].iloc[: length - 1].isna().all()
        assert a["tht_fv"].iloc[length - 1 :].notna().all()
        early = [c for c, n in zip(colors(a), counts(a), strict=True) if n < min_crossings]
        assert all(c is None for c in early)
        assert any(c is not None for c in colors(a))  # the walk does flip at some point


def test_default_parameters_are_33_and_20() -> None:
    days = sessions("2024-01-02", "2024-12-31")
    daily = frame(walk("A", days, seed=43))
    assert_same_values(
        values(tht_features(daily), "A", days[0]),
        values(reference(daily, 33, 20, None), "A", days[0]),
    )


def test_split_adjusted_input_keeps_fv_continuous() -> None:
    # 40 sessions at ~100 then a 10-for-1 split (raw prices before the ex-date x10, made up):
    # the caller's adjust_for_splits makes the series one level, so fv barely moves on the
    # ex-date; on the raw series it would carry 32 rows at 10x into the window
    days = sessions("2024-04-15", "2024-06-28")
    ex = NVDA_EX
    adjusted = walk("S_NVDA", days, seed=44, start=100.0)
    raw = []
    for r in adjusted:
        k = 10.0 if r["session_date"] < ex else 1.0
        raw.append(
            {**r, "symbol": "NVDA", **{p: r[p] * k for p in ("open", "high", "low", "close")}}
        )
    splits = pd.DataFrame.from_records([split_row("NVDA", ex, 1.0, 10.0)])
    out = tht_features(adjust_for_splits(frame(raw), splits))
    fv = dict(zip(out["session_date"], out["tht_fv"], strict=True))
    before = max(d for d in days if d < ex)
    assert abs(fv[ex] / fv[before] - 1) < 0.05
    # the raw series would put fv near 10x the close on the ex-date
    close = dict(zip(out["session_date"], out["close"], strict=True))
    assert 0.5 < fv[ex] / close[ex] < 2


# =============================================================================================
# boundaries
# =============================================================================================


def test_exactly_length_rows_defines_only_the_last_fv() -> None:
    rows = [flat(float(p)) for p in range(1, 34)]  # OHLC4 1..33
    out = tht_features(series(rows))
    assert out["tht_fv"].iloc[:32].isna().all()
    assert out["tht_fv"].iloc[32] == 17.0  # mean of 1..33
    assert bool(tht_features(series(rows[:32]))["tht_fv"].isna().all())
    assert colors(out) == [None] * 33


def test_fv_on_the_low_or_the_high_is_not_a_crossing() -> None:
    # length 2
    # 0 flat 12                 fv NaN
    # 1 8 10 6 8   (ohlc4 8)    fv (12+8)/2 = 10 == high  -> no
    # 2 12 14 10 12 (ohlc4 12)  fv (8+12)/2 = 10 == low   -> no
    # 3 12 14 10 12             fv 12, 10 < 12 < 14       -> yes: hu 14/12, ld 10/12
    rows = [flat(12), (8, 10, 6, 8), (12, 14, 10, 12), (12, 14, 10, 12)]
    out = tht_features(series(rows), length=2, min_crossings=1)
    assert_floats(out["tht_fv"], [NAN, 10, 10, 12])
    assert counts(out) == [0, 0, 0, 1]
    assert_floats(out["tht_hu"], [NAN, NAN, NAN, 14 / 12])
    assert_floats(out["tht_ld"], [NAN, NAN, NAN, 10 / 12])


def test_touching_a_threshold_is_not_a_flip() -> None:
    # length 2, min_crossings 2; hu = 10/8 = 1.25, ld = 6/8 = 0.75 throughout (all exact)
    # row  bar           ohlc4  fv   check                                  colour
    # 0    flat 8        8      NaN                                         None
    # 1    8 10 6 8      8      8    crossing                               None
    # 2    8 10 6 8      8      8    crossing                               None
    # 3    flat 16       16     12   low 16 > 12*1.25 = 15                  green
    # 4    8 9 7 8       8      12   high 9 == 12*0.75: not red             green (carried)
    # 5    flat 4        4      6    high 4 < 6*0.75 = 4.5                  red
    # 6    8 8.5 7.5 8   8      6    low 7.5 == 6*1.25: not green           red (carried)
    rows = [
        flat(8),
        (8, 10, 6, 8),
        (8, 10, 6, 8),
        flat(16),
        (8, 9, 7, 8),
        flat(4),
        (8, 8.5, 7.5, 8),
    ]
    out = tht_features(series(rows), length=2, min_crossings=2)
    assert_floats(out["tht_fv"], [NAN, 8, 8, 12, 12, 6, 6])
    assert counts(out) == [0, 1, 2, 2, 2, 2, 2]
    assert_floats(out["tht_hu"], [NAN, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25])
    assert_floats(out["tht_ld"], [NAN, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75])
    assert colors(out) == [None, None, None, "green", "green", "red", "red"]


def test_colour_starts_exactly_at_min_crossings() -> None:
    # length 2, hu 1.25 / ld 0.75 as above.
    # row 3 (flat 16, fv 12) is green by price but has 2 crossings: None at min 3, green at min 2
    # row 4 (4 13 3 12: ohlc4 8, fv 12, crossing #3) is neither red nor green: carries row 3
    # row 5 (flat 16, fv 12) green at either min
    rows = [flat(8), (8, 10, 6, 8), (8, 10, 6, 8), flat(16), (4, 13, 3, 12), flat(16)]
    at3 = tht_features(series(rows), length=2, min_crossings=3)
    at2 = tht_features(series(rows), length=2, min_crossings=2)
    assert counts(at3) == [0, 1, 2, 2, 3, 3]
    assert colors(at3) == [None, None, None, None, None, "green"]
    assert colors(at2) == [None, None, None, "green", "green", "green"]


def test_two_securities_never_mix() -> None:
    days = sessions("2024-01-02", "2024-09-30")
    a = walk("A", days, seed=45)
    b = walk("B", days[25:], seed=46, start=200.0)
    together = tht_features(frame(a + b))
    alone_a = tht_features(frame(a))
    alone_b = tht_features(frame(b))
    assert_same_values(values(together, "A", days[0]), values(alone_a, "A", days[0]))
    assert_same_values(values(together, "B", days[0]), values(alone_b, "B", days[0]))
    assert of(together, "B")["tht_fv"].iloc[:32].isna().all()  # B's window starts at B's bar
    assert since(together, "B", days[0]) == {days[25]}


def test_two_rows_for_one_security_and_session_are_refused() -> None:
    rows = walk("A", DAYS[:40], seed=47)
    with pytest.raises(ValueError, match="share a security and session"):
        tht_features(frame(rows + [dict(rows[10], symbol="A2")]))


def test_output_is_sorted_and_other_columns_pass_through() -> None:
    days = DAYS[:40]
    rows = walk("B", days, seed=48) + walk("A", days, seed=49)
    for i, r in enumerate(rows):
        r["note"] = f"n{i}"
    shuffled = frame(rows).sample(frac=1.0, random_state=0).reset_index(drop=True)
    out = tht_features(shuffled)

    keys = list(zip(out["security_id"], out["session_date"], strict=True))
    assert keys == sorted(keys)
    assert len(out) == len(rows)
    for column in ("symbol", "volume", "note", "open", "high", "low", "close"):
        assert column in out.columns
    expected = {(r["security_id"], r["session_date"]): r["note"] for r in rows}
    assert dict(zip(keys, out["note"], strict=True)) == expected
    assert out["volume"].tolist() == [1000] * len(rows)


def test_empty_input_returns_the_columns() -> None:
    empty = pd.DataFrame(
        columns=["security_id", "symbol", "session_date", "open", "high", "low", "close"]
    )
    out = tht_features(empty)
    assert out.empty
    for column in [*NUM_COLUMNS, COLOR, SINCE]:
        assert column in out.columns


# =============================================================================================
# leakage: later rows, later-knowable splits
# =============================================================================================

LEAK_DAYS = sessions("2024-01-02", "2024-09-30")
LEAK_DAILY = frame(walk("A", LEAK_DAYS, seed=51) + walk("B", LEAK_DAYS[40:], seed=52, start=8.0))


@pytest.mark.parametrize(
    "cut",
    [LEAK_DAYS[31], LEAK_DAYS[32], LEAK_DAYS[72], LEAK_DAYS[100], LEAK_DAYS[150], LEAK_DAYS[-2]],
    ids=["before_first_fv", "first_fv", "b_first_fv", "mid", "late", "second_last"],
)
@pytest.mark.parametrize("window", [None, 5])
def test_appending_later_rows_changes_nothing_at_or_before_the_cut(
    cut: date, window: int | None
) -> None:
    full = tht_features(LEAK_DAILY, median_window=window)
    part = tht_features(where(LEAK_DAILY, LEAK_DAILY["session_date"] <= cut), median_window=window)
    for sid in ("A", "B"):
        got = values(part, sid, LEAK_DAYS[0])
        if got.empty:
            continue
        assert_same_values(got, values(full, sid, LEAK_DAYS[0], cut + timedelta(days=1)))


def test_editing_a_later_bar_changes_nothing_before_it() -> None:
    # a wild future bar would, under a centred or full-sample statistic, move every median
    day = LEAK_DAYS[120]
    edited = LEAK_DAILY.copy()
    mask = (edited["security_id"] == "A") & (edited["session_date"] == day)
    edited.loc[mask, "high"] = edited.loc[mask, "high"] * 5
    edited.loc[mask, "low"] = edited.loc[mask, "low"] * 0.2
    base, alt = tht_features(LEAK_DAILY), tht_features(edited)
    assert_same_values(values(alt, "A", LEAK_DAYS[0], day), values(base, "A", LEAK_DAYS[0], day))
    assert not of(alt, "A")["tht_hu"].iloc[121:].equals(of(base, "A")["tht_hu"].iloc[121:])


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return market_root(tmp_path)


@pytest.fixture
def con() -> Iterator[duckdb.DuckDBPyConnection]:
    connection = duckdb.connect()
    yield connection
    connection.close()


SPLIT_DAYS = sessions("2024-03-01", "2024-07-31")
CUT = max(d for d in SPLIT_DAYS if d < NVDA_EX)  # 2024-06-07, the last pre-split session


def _raw_split_bars() -> pd.DataFrame:
    """Made-up NVDA-like bars: a walk around 100, raw prices before 2024-06-10 x10."""
    rows = []
    for r in walk("S_NVDA", SPLIT_DAYS, seed=53, start=100.0):
        k = 10.0 if r["session_date"] < NVDA_EX else 1.0
        rows.append(
            {**r, "symbol": "NVDA", **{p: r[p] * k for p in ("open", "high", "low", "close")}}
        )
    return frame(rows)


@pytest.mark.parametrize(
    "available_at",
    [None, known(NVDA_EX) + timedelta(days=2)],
    ids=["on_time", "late_feed"],
)
def test_a_split_not_yet_visible_is_not_applied(
    con: duckdb.DuckDBPyConnection, root: Path, available_at: datetime | None
) -> None:
    # Written to a tmp store and read through the real visible_splits. At 20:30 New York on
    # 06-07 (and, for the late-feed record, until two days after the ex-date) the record is not
    # visible: the THT the night saw is the one on raw prices; the hindsight-adjusted series
    # would show an fv ten times smaller on the same sessions.
    write_split_rows(root, [split_row("NVDA", NVDA_EX, 1.0, 10.0, available_at=available_at)])
    raw = _raw_split_bars()
    as_of = datetime(2024, 6, 8, 0, 30, tzinfo=UTC)
    hindsight_as_of = datetime(2026, 10, 27, tzinfo=UTC)

    at_cut = visible_splits(con, root, as_of).to_pandas()
    assert at_cut.empty
    pit = tht_features(adjust_for_splits(raw, at_cut))
    truncated = tht_features(adjust_for_splits(where(raw, raw["session_date"] <= CUT), at_cut))
    hindsight = tht_features(
        adjust_for_splits(raw, visible_splits(con, root, hindsight_as_of).to_pandas())
    )

    upto = CUT + timedelta(days=1)
    assert_same_values(
        values(pit, "S_NVDA", SPLIT_DAYS[0], upto), values(truncated, "S_NVDA", SPLIT_DAYS[0])
    )
    assert_same_values(
        values(truncated, "S_NVDA", SPLIT_DAYS[0]),
        values(tht_features(where(raw, raw["session_date"] <= CUT)), "S_NVDA", SPLIT_DAYS[0]),
    )
    seen = values(truncated, "S_NVDA", CUT).iloc[0]
    later = values(hindsight, "S_NVDA", CUT, upto).iloc[0]
    assert seen["tht_fv"] == pytest.approx(10 * later["tht_fv"], rel=1e-9)
    # a uniform rescale of the whole visible history leaves ratios and colours alone
    assert seen["tht_hu"] == pytest.approx(later["tht_hu"], rel=1e-9)
    assert colors(values(truncated, "S_NVDA", SPLIT_DAYS[0])) == colors(
        values(hindsight, "S_NVDA", SPLIT_DAYS[0], upto)
    )


# =============================================================================================
# spin-offs: blank window, adjust (values change), restart, pending, composition, tht_since
# =============================================================================================

SPIN_DAYS = sessions("2024-01-02", "2024-12-31")  # 252 sessions
E1, K1 = SPIN_DAYS[150], SPIN_DAYS[153]  # an adjust decided three sessions after its ex-date
E2, K2 = SPIN_DAYS[200], SPIN_DAYS[220]  # a later event decided twenty sessions after it
SPIN_DAILY = frame(walk("A", SPIN_DAYS, seed=61) + walk("B", SPIN_DAYS, seed=62, start=30.0))


def test_no_events_is_the_plain_output_plus_tht_since() -> None:
    plain = tht_features(SPIN_DAILY)
    for spin in (None, events_frame([])):
        out = tht_features(SPIN_DAILY, spinoffs=spin)
        for sid in ("A", "B"):
            assert_same_values(values(out, sid, SPIN_DAYS[0]), values(plain, sid, SPIN_DAYS[0]))
            assert since(out, sid, SPIN_DAYS[0]) == {SPIN_DAYS[0]}


def test_sma_is_not_scale_invariant_so_an_adjust_changes_later_values() -> None:
    # why "adjust" is not a no-op here, unlike BX: scaling the prefix moves the windows that
    # straddle the ex-date
    shifted = tht_features(scaled_before(SPIN_DAILY, "A", E1, 0.8))
    plain = tht_features(SPIN_DAILY)
    a, b = values(shifted, "A", K1, SPIN_DAYS[160]), values(plain, "A", K1, SPIN_DAYS[160])
    assert (np.abs(a["tht_fv"].to_numpy() / b["tht_fv"].to_numpy() - 1) > 0.01).all()


def test_adjust_event_before_window_after() -> None:
    spin = events_frame([event("A", E1, "adjust", factor=0.8, known_on=K1)])
    out = tht_features(SPIN_DAILY, spinoffs=spin)
    plain = tht_features(SPIN_DAILY)
    adjusted = tht_features(scaled_before(SPIN_DAILY, "A", E1, 0.8))

    # before the ex-date: the plain series (NOT the rescaled one, whose fv there is 0.8x)
    assert_same_values(values(out, "A", SPIN_DAYS[0], E1), values(plain, "A", SPIN_DAYS[0], E1))
    window = values(out, "A", E1, K1)
    assert len(window) == 3 and blank(window)
    assert_same_values(values(out, "A", K1), values(adjusted, "A", K1))
    assert values(out, "A", K1)["tht_fv"].iloc[0] != pytest.approx(
        values(plain, "A", K1)["tht_fv"].iloc[0], rel=1e-3
    )
    assert since(out, "A", SPIN_DAYS[0], E1) == {SPIN_DAYS[0]}
    assert since(out, "A", K1) == {SPIN_DAYS[0]}


def test_restart_event_before_window_after() -> None:
    spin = events_frame([event("A", E2, "restart", known_on=K2, reason="no_child_price")])
    out = tht_features(SPIN_DAILY, spinoffs=spin)
    plain = tht_features(SPIN_DAILY)
    restarted = tht_features(from_day(SPIN_DAILY, "A", E2))

    assert_same_values(values(out, "A", SPIN_DAYS[0], E2), values(plain, "A", SPIN_DAYS[0], E2))
    window = values(out, "A", E2, K2)
    assert len(window) == 20 and blank(window)
    assert_same_values(values(out, "A", K2), values(restarted, "A", K2))
    # the restarted SMA needs 33 rows from E2: rows 220..231 have no fv, row 232 does
    first_fv = SPIN_DAYS[232]
    warm = values(out, "A", K2, first_fv)
    assert len(warm) == 12 and bool(np.isnan(warm["tht_fv"].to_numpy(dtype=float)).all())
    assert all(c is None for c in colors(warm))
    assert not math.isnan(values(out, "A", first_fv)["tht_fv"].iloc[0])
    assert since(out, "A", K2) == {E2}
    assert since(out, "A", SPIN_DAYS[0], E2) == {SPIN_DAYS[0]}


def test_pending_event_blanks_from_its_ex_date_for_good() -> None:
    spin = events_frame([event("A", E2, "pending")])
    out = tht_features(SPIN_DAILY, spinoffs=spin)
    plain = tht_features(SPIN_DAILY)

    assert_same_values(values(out, "A", SPIN_DAYS[0], E2), values(plain, "A", SPIN_DAYS[0], E2))
    assert blank(values(out, "A", E2))
    assert_same_values(values(out, "B", SPIN_DAYS[0]), values(plain, "B", SPIN_DAYS[0]))
    assert since(out, "A", SPIN_DAYS[0]) == {SPIN_DAYS[0]}


def test_adjust_known_on_its_ex_date_has_no_blank_window() -> None:
    spin = events_frame([event("A", E1, "adjust", factor=0.8004, known_on=E1)])
    out = tht_features(SPIN_DAILY, spinoffs=spin)
    adjusted = tht_features(scaled_before(SPIN_DAILY, "A", E1, 0.8004))
    assert_same_values(values(out, "A", E1), values(adjusted, "A", E1))
    assert not np.isnan(values(out, "A", E1)[NUM_COLUMNS].to_numpy(dtype=float)).any()


def test_two_adjusts_compose() -> None:
    spin = events_frame(
        [
            event("A", E1, "adjust", factor=0.8, known_on=K1),
            event("A", E2, "adjust", factor=0.5, known_on=K2),
        ]
    )
    out = tht_features(SPIN_DAILY, spinoffs=spin)
    first_only = tht_features(scaled_before(SPIN_DAILY, "A", E1, 0.8))
    both = tht_features(scaled_before(scaled_before(SPIN_DAILY, "A", E1, 0.8), "A", E2, 0.5))

    assert_same_values(values(out, "A", K1, E2), values(first_only, "A", K1, E2))
    assert blank(values(out, "A", E2, K2))
    assert_same_values(values(out, "A", K2), values(both, "A", K2))
    assert since(out, "A", K2) == {SPIN_DAYS[0]}


def test_adjust_then_restart_resets_tht_since() -> None:
    spin = events_frame(
        [
            event("A", E1, "adjust", factor=0.8, known_on=K1),
            event("A", E2, "restart", known_on=K2, reason="no_child_price"),
        ]
    )
    out = tht_features(SPIN_DAILY, spinoffs=spin)
    first_only = tht_features(scaled_before(SPIN_DAILY, "A", E1, 0.8))
    restarted = tht_features(from_day(SPIN_DAILY, "A", E2))

    assert_same_values(values(out, "A", K1, E2), values(first_only, "A", K1, E2))
    assert since(out, "A", K1, E2) == {SPIN_DAYS[0]}
    assert_same_values(values(out, "A", K2), values(restarted, "A", K2))
    assert since(out, "A", K2) == {E2}


def test_events_of_one_security_leave_another_untouched() -> None:
    spin = events_frame(
        [
            event("A", E1, "adjust", factor=0.5, known_on=K1),
            event("A", E2, "restart", known_on=K2, reason="child_too_late"),
        ]
    )
    out = tht_features(SPIN_DAILY, spinoffs=spin)
    plain = tht_features(SPIN_DAILY)
    assert_same_values(values(out, "B", SPIN_DAYS[0]), values(plain, "B", SPIN_DAYS[0]))
    assert since(out, "B", SPIN_DAYS[0]) == {SPIN_DAYS[0]}


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
    # whatever the event turns out to be, every session before known_on reads the same
    base = tht_features(
        SPIN_DAILY, spinoffs=events_frame([event("A", E1, "adjust", factor=0.8, known_on=K1)])
    )
    alt = tht_features(SPIN_DAILY, spinoffs=events_frame([other]))
    assert_same_values(values(alt, "A", SPIN_DAYS[0], K1), values(base, "A", SPIN_DAYS[0], K1))
    assert since(alt, "A", SPIN_DAYS[0], E1) == since(base, "A", SPIN_DAYS[0], E1)
    # and it does matter after known_on (the test would pass vacuously otherwise)
    with pytest.raises(AssertionError):
        assert_same_values(values(alt, "A", K1), values(base, "A", K1))


def test_later_rows_change_nothing_known_at_a_cut_with_events() -> None:
    spin = events_frame([event("A", E1, "adjust", factor=0.8, known_on=K1)])
    cut = SPIN_DAYS[190]
    full = tht_features(SPIN_DAILY, spinoffs=spin)
    part = tht_features(where(SPIN_DAILY, SPIN_DAILY["session_date"] <= cut), spinoffs=spin)
    for sid in ("A", "B"):
        assert_same_values(
            values(part, sid, SPIN_DAYS[0]),
            values(full, sid, SPIN_DAYS[0], cut + timedelta(days=1)),
        )
