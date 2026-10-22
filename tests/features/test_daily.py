"""Guards against daily features that see their own day or the future: an ATR or relative volume
whose window includes the current row (the reaction day's own range would widen its own stop and
its own volume would inflate its average), a rolling window that crosses from one security into
the next, and a feature at day d that changes when later rows are added.

API under test (asof.features.daily):

    daily_features(bars, *, atr_days=14, rvol_days=20, dollar_volume_days=20) -> pd.DataFrame
        one row per input (security_id, session_date) with prev_close, gap, rvol, atr,
        avg_dollar_volume; every rolling window covers the previous rows only.

Bars here are hand-made numbers (no data source is imitated). The rows are already the visible
ones -- the as_of cut happens in visible_bars -- so leakage here means "a row's features depend
on later rows", tested by truncating and by editing the future.
"""

import math

import pandas as pd
import pytest

from asof.features.daily import daily_features

DAYS = [d.date() for d in pd.bdate_range("2026-03-02", "2026-03-06")]
# (open, high, low, close, volume); true ranges from d1: 2, 1, 3, 1
A_ROWS = [
    (10.0, 11.0, 9.0, 10.0, 100),
    (10.0, 12.0, 10.0, 11.0, 200),
    (11.0, 11.5, 10.5, 11.0, 300),
    (13.2, 14.0, 12.0, 13.0, 1000),  # gap +20%, its own true range 3
    (13.0, 13.0, 12.0, 12.5, 400),
]
B_ROWS = [
    (50.0, 51.0, 49.0, 50.0, 10),
    (52.0, 53.0, 51.0, 52.0, 20),
    (52.0, 52.0, 50.0, 51.0, 30),
]
SMALL = {"atr_days": 2, "rvol_days": 2, "dollar_volume_days": 2}


def frame(
    rows: dict[str, list[tuple[float, float, float, float, int]]], start: int = 0
) -> pd.DataFrame:
    records = []
    for security_id, values in rows.items():
        offset = start if security_id == "B" else 0
        for i, (o, h, lo, c, v) in enumerate(values):
            records.append(
                {
                    "security_id": security_id,
                    "session_date": DAYS[i + offset],
                    "open": o,
                    "high": h,
                    "low": lo,
                    "close": c,
                    "volume": v,
                }
            )
    return pd.DataFrame.from_records(records)


def by_key(features: pd.DataFrame) -> pd.DataFrame:
    return features.set_index(["security_id", "session_date"]).sort_index()


def at(features: pd.DataFrame, security_id: str, day_index: int, column: str) -> float:
    return float(by_key(features).loc[(security_id, DAYS[day_index]), column])


# =============================================================================================
# normal path
# =============================================================================================


def test_hand_computed_features_of_one_security() -> None:
    f = daily_features(frame({"A": A_ROWS}), **SMALL)
    assert len(f) == len(A_ROWS)
    assert {"prev_close", "gap", "rvol", "atr", "avg_dollar_volume"} <= set(f.columns)

    assert at(f, "A", 1, "prev_close") == 10.0
    assert at(f, "A", 3, "prev_close") == 11.0
    assert at(f, "A", 3, "gap") == pytest.approx(13.2 / 11.0 - 1)
    assert at(f, "A", 2, "rvol") == pytest.approx(300 / 150)
    assert at(f, "A", 3, "rvol") == pytest.approx(1000 / 250)
    assert at(f, "A", 4, "rvol") == pytest.approx(400 / 650)
    assert at(f, "A", 2, "avg_dollar_volume") == pytest.approx((10 * 100 + 11 * 200) / 2)
    assert at(f, "A", 3, "avg_dollar_volume") == pytest.approx((11 * 200 + 11 * 300) / 2)
    assert at(f, "A", 4, "avg_dollar_volume") == pytest.approx((11 * 300 + 13 * 1000) / 2)


def test_atr_is_a_simple_mean_of_previous_true_ranges() -> None:
    f = daily_features(frame({"A": A_ROWS}), **SMALL)
    # d3: TR of d1 (2) and d2 (1); d4: TR of d2 (1) and d3 (3, which uses |high - prev_close|)
    assert at(f, "A", 3, "atr") == pytest.approx(1.5)
    assert at(f, "A", 4, "atr") == pytest.approx(2.0)


def test_defaults_are_the_scan_windows() -> None:
    # 30 flat sessions: TR 2, volume 1000, close 50 -> atr 2, rvol 1, dollar volume 50,000
    days = [d.date() for d in pd.bdate_range("2026-01-05", periods=30)]
    bars = pd.DataFrame(
        {
            "security_id": "A",
            "session_date": days,
            "open": 50.0,
            "high": 51.0,
            "low": 49.0,
            "close": 50.0,
            "volume": 1000,
        }
    )
    f = by_key(daily_features(bars))
    assert math.isnan(f.loc[("A", days[13]), "atr"])  # 13 previous rows < 14
    assert f.loc[("A", days[15]), "atr"] == pytest.approx(2.0)
    assert math.isnan(f.loc[("A", days[19]), "rvol"])  # 19 previous rows < 20
    assert f.loc[("A", days[20]), "rvol"] == pytest.approx(1.0)
    assert f.loc[("A", days[20]), "avg_dollar_volume"] == pytest.approx(50_000.0)


# =============================================================================================
# boundaries
# =============================================================================================


def test_the_current_row_never_enters_its_own_windows() -> None:
    # d3 has range 3 and volume 1000; if it entered its own windows atr would be 2.0 and rvol
    # 1000 / 650. The reaction day must not widen its own stop.
    f = daily_features(frame({"A": A_ROWS}), **SMALL)
    assert at(f, "A", 3, "atr") != pytest.approx(2.0)
    assert at(f, "A", 3, "rvol") == pytest.approx(4.0)
    assert at(f, "A", 3, "avg_dollar_volume") == pytest.approx(2750.0)


def test_short_history_is_nan() -> None:
    f = daily_features(frame({"A": A_ROWS}), **SMALL)
    assert math.isnan(at(f, "A", 0, "prev_close"))
    assert math.isnan(at(f, "A", 0, "gap"))
    for column in ("rvol", "atr", "avg_dollar_volume"):
        assert math.isnan(at(f, "A", 0, column)), column
        assert math.isnan(at(f, "A", 1, column)), column


def test_windows_never_cross_securities() -> None:
    # B starts on d2, interleaved with A; its first row has no previous close even though A's
    # d1 row sits right before it in the input.
    bars = frame({"A": A_ROWS, "B": B_ROWS}, start=2)
    f = daily_features(bars, **SMALL)
    assert len(f) == len(A_ROWS) + len(B_ROWS)
    assert math.isnan(at(f, "B", 2, "prev_close"))
    assert math.isnan(at(f, "B", 3, "rvol"))
    assert at(f, "B", 4, "rvol") == pytest.approx(30 / 15)
    assert at(f, "B", 3, "gap") == pytest.approx(52.0 / 50.0 - 1)
    # A's values are the same as when computed alone
    alone = by_key(daily_features(frame({"A": A_ROWS}), **SMALL))
    pd.testing.assert_frame_equal(by_key(f).loc[["A"]], alone, check_like=True)


def test_input_order_does_not_matter() -> None:
    bars = frame({"A": A_ROWS, "B": B_ROWS}, start=2)
    shuffled = bars.sample(frac=1.0, random_state=7).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        by_key(daily_features(bars, **SMALL)),
        by_key(daily_features(shuffled, **SMALL)),
        check_like=True,
    )


# =============================================================================================
# leakage: later rows never change earlier features
# =============================================================================================


@pytest.mark.parametrize("cut", range(1, len(A_ROWS)))
def test_features_of_a_day_do_not_depend_on_later_rows(cut: int) -> None:
    full = by_key(daily_features(frame({"A": A_ROWS}), **SMALL))
    truncated = by_key(daily_features(frame({"A": A_ROWS[:cut]}), **SMALL))
    pd.testing.assert_frame_equal(full.iloc[:cut], truncated, check_like=True)


def test_editing_the_future_leaves_the_past_alone() -> None:
    edited = list(A_ROWS)
    edited[4] = (99.0, 500.0, 1.0, 250.0, 10**9)
    base = by_key(daily_features(frame({"A": A_ROWS}), **SMALL))
    other = by_key(daily_features(frame({"A": edited}), **SMALL))
    pd.testing.assert_frame_equal(base.iloc[:4], other.iloc[:4], check_like=True)
