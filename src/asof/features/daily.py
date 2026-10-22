"""Daily-bar features: gap, relative volume, ATR, dollar volume.

Every rolling window covers the rows *before* the current one. On a reaction day that matters
twice: its own range must not widen the stop it sets, and its own volume must not inflate the
average it is compared with. Windows never cross from one security into the next.
"""

import pandas as pd

COLUMNS = ("prev_close", "gap", "rvol", "atr", "avg_dollar_volume")


def daily_features(
    bars: pd.DataFrame, *, atr_days: int = 14, rvol_days: int = 20, dollar_volume_days: int = 20
) -> pd.DataFrame:
    """``bars`` holds security_id, session_date, open, high, low, close, volume. Returns them
    sorted by security and day with ``COLUMNS`` added; NaN where history is too short."""
    out = bars.sort_values(["security_id", "session_date"]).reset_index(drop=True).copy()
    out["prev_close"] = out.groupby("security_id", sort=False)["close"].shift(1)
    out["gap"] = out["open"] / out["prev_close"] - 1
    true_range = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - out["prev_close"]).abs(),
            (out["low"] - out["prev_close"]).abs(),
        ],
        axis=1,
    ).max(axis=1, skipna=False)
    out["_tr"] = true_range
    out["_dollar_volume"] = out["close"] * out["volume"]

    def previous_mean(column: str, days: int) -> pd.Series:
        grouped = out.groupby("security_id", sort=False)[column]
        return grouped.transform(lambda s: s.shift(1).rolling(days, min_periods=days).mean())

    out["atr"] = previous_mean("_tr", atr_days)
    out["rvol"] = out["volume"] / previous_mean("volume", rvol_days)
    out["avg_dollar_volume"] = previous_mean("_dollar_volume", dollar_volume_days)
    return out.drop(columns=["_tr", "_dollar_volume"])
