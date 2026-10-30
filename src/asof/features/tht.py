"""THT Fair Value Bands trend colour on split-adjusted daily bars.

TradingView's "THT Fair Value Bands" is closed source; this is the rule reverse-engineered from
quantifytools' open "Fair value bands" (Cross mode) and checked against the daily chart:

- fair value ``fv`` = simple mean of OHLC4 = (open + high + low + close) / 4 over ``length``
  sessions, the current one included;
- a crossing is a session whose range straddles fv (low < fv < high); the thresholds are the
  medians of high / fv (``hu``) and low / fv (``ld``) over the crossings so far, so every
  security gets bands as wide as its own swings;
- red when high < fv * ld, else green when low > fv * hu, else the previous session's colour.

It stays point-in-time the way ``bx`` does: prices are adjusted only by the splits the caller
passes, and every value runs forward over one security's rows, so a later row changes nothing
before it. Unlike B-Xtrender, a mean of prices is not scale-invariant: a spin-off adjustment
changes the values from its ex-date on, so those are recomputed on the adjusted series.
"""

import numpy as np
import pandas as pd

from asof.features.bx import (
    BLANK,
    _as_dates,
    _require_one_row_per_session,
    spinoff_states,
    state_series,
)

THT_COLUMNS = ["tht_fv", "tht_crossings", "tht_hu", "tht_ld"]
_CODE = "_tht_code"  # 1 green, 0 red, NaN no colour while computing
_SERIES_COLUMNS = [*THT_COLUMNS, _CODE]


def tht_features(
    daily: pd.DataFrame,
    *,
    spinoffs: pd.DataFrame | None = None,
    length: int = 33,
    min_crossings: int = 20,
    median_window: int | None = None,
) -> pd.DataFrame:
    """``daily`` (already split-adjusted) sorted by security and session, with ``THT_COLUMNS``,
    ``tht_color`` ("green", "red", or None while fv is undefined, fewer than ``min_crossings``
    crossings have been seen, or no flip has happened yet) and ``tht_since``, the first session
    of the series a row's values come from. ``median_window`` keeps only the latest crossings
    in the medians; None keeps all of them.

    ``spinoffs`` follows ``bx_features``: sessions before an ex-date keep their values, blank
    until the event is decided, then the adjusted series or a series that restarts at E."""
    if median_window is not None and median_window < 1:
        raise ValueError(f"median_window must be None or at least 1, got {median_window}")
    out = daily.sort_values(["security_id", "session_date"]).reset_index(drop=True)
    _require_one_row_per_session(out)
    params = (length, min_crossings, median_window)
    out[_SERIES_COLUMNS] = _tht_columns(out, *params)
    out["tht_since"] = out.groupby("security_id", sort=False)["session_date"].transform("first")
    if spinoffs is not None and not spinoffs.empty:
        rows = out.groupby("security_id", sort=False).indices
        for sid, events in spinoffs.groupby("security_id", sort=False):
            if sid not in rows:
                continue
            for state, members in spinoff_states(out, rows[sid], events).items():
                if state == BLANK:
                    out.loc[members, _SERIES_COLUMNS] = np.nan
                    continue
                part = state_series(out, rows[sid], state)
                values = pd.DataFrame(
                    _tht_columns(part, *params), index=part.index, columns=_SERIES_COLUMNS
                )
                out.loc[members, _SERIES_COLUMNS] = values.loc[members].to_numpy()
                out.loc[members, "tht_since"] = _as_dates(pd.Series(part["session_date"]))[0]
    code = out.pop(_CODE).to_numpy()
    out["tht_color"] = ["green" if v == 1 else "red" if v == 0 else None for v in code]
    return out


def _tht_columns(
    frame: pd.DataFrame, length: int, min_crossings: int, median_window: int | None
) -> np.ndarray:
    """``_SERIES_COLUMNS`` for ``frame`` (sorted by security and session, one series per
    security), row-aligned."""
    values = np.full((len(frame), len(_SERIES_COLUMNS)), np.nan)
    for idx in frame.groupby("security_id", sort=False).indices.values():
        part = frame.iloc[idx]
        values[idx] = _one_series(
            part["open"].to_numpy(dtype=float),
            part["high"].to_numpy(dtype=float),
            part["low"].to_numpy(dtype=float),
            part["close"].to_numpy(dtype=float),
            length,
            min_crossings,
            median_window,
        )
    return values


def _one_series(
    o: np.ndarray,
    h: np.ndarray,
    lo: np.ndarray,
    c: np.ndarray,
    length: int,
    min_crossings: int,
    median_window: int | None,
) -> np.ndarray:
    fv = np.asarray(pd.Series((o + h + lo + c) / 4).rolling(length).mean(), dtype=float)
    crossing = (lo < fv) & (fv < h)  # False where fv is NaN
    crossings = np.cumsum(crossing).astype(float)
    hu = _running_median(h / fv, crossing, median_window)
    ld = _running_median(lo / fv, crossing, median_window)
    code = np.full(len(fv), np.nan)
    previous = np.nan
    for i in range(len(fv)):
        if np.isnan(fv[i]) or crossings[i] < min_crossings:
            previous = np.nan
        elif h[i] < fv[i] * ld[i]:
            previous = 0.0
        elif lo[i] > fv[i] * hu[i]:
            previous = 1.0
        code[i] = previous
    return np.column_stack([fv, crossings, hu, ld, code])


def _running_median(ratio: np.ndarray, crossing: np.ndarray, window: int | None) -> np.ndarray:
    """At each row, the median of ``ratio`` over the crossings up to and including it (the last
    ``window`` of them); NaN before the first crossing."""
    at = pd.Series(ratio[crossing])
    medians = at.expanding().median() if window is None else at.rolling(window, 1).median()
    out = pd.Series(np.nan, index=range(len(ratio)))
    out[np.flatnonzero(crossing)] = np.asarray(medians)
    return out.ffill().to_numpy()
