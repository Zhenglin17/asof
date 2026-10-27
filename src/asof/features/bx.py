"""B-Xtrender short-term histogram on split-adjusted daily, weekly and monthly bars.

The indicator is TradingView's "B-Xtrender @Puppytherapy": rsi(ema(close, l1) - ema(close, l2),
l3) - 50, with ema / rma / rsi reproduced from the Pine reference so the numbers can be checked
against the chart. Three things keep it point-in-time:

- prices are adjusted only by the splits the caller passes (``visible_splits`` at ``as_of``);
- a weekly or monthly bar exists only once its period is over, and its value is used from the
  close of the session that completes it (``complete_on``), never earlier;
- every recursion runs forward over one security's rows, so a later row changes nothing before
  it.

The store has no exchange calendar. A period is complete at the close of its last weekday when
the security traded that day; otherwise (a holiday such as Good Friday, or a halt) at the close
of the security's next session, the first moment the data shows the period is over.
"""

from datetime import date, timedelta
from typing import Literal

import numpy as np
import pandas as pd

Period = Literal["W", "M"]
PRICES = ("open", "high", "low", "close")
PERIOD_COLUMNS = ["security_id", "period_end", "complete_on", "open", "high", "low", "close"]
_STALE_DAYS = 7  # a long weekend plus a holiday is 4 calendar days


# -- Pine primitives ------------------------------------------------------------------------------


def pine_ema(src: pd.Series, length: int) -> pd.Series:
    """alpha = 2 / (length + 1), seeded with the first value."""
    return pd.Series(src.ewm(alpha=2 / (length + 1), adjust=False).mean(), index=src.index)


def pine_rma(src: pd.Series, length: int) -> pd.Series:
    """alpha = 1 / length, seeded with the plain mean of the first ``length`` defined values
    (leading NaN push the window back); NaN before the seed."""
    out = pd.Series(np.nan, index=src.index, dtype=float)
    defined = np.flatnonzero(src.notna().to_numpy())
    if len(defined) < length:
        return out
    first = int(defined[0])
    seed_at = first + length - 1
    tail = src.iloc[seed_at:].astype(float).copy()
    tail.iloc[0] = float(src.iloc[first : seed_at + 1].mean())
    out.iloc[seed_at:] = tail.ewm(alpha=1 / length, adjust=False).mean().to_numpy()
    return out


def pine_rsi(x: pd.Series, length: int) -> pd.Series:
    """100 when the average loss is 0 (flat included), 0 when the average gain is 0."""
    change = x.diff()
    up = pine_rma(change.clip(lower=0), length)
    down = pine_rma((-change).clip(lower=0), length)
    rs = up / down.where(down != 0)
    rsi = pd.Series(100 - 100 / (1 + rs), index=x.index)
    rsi = rsi.where(up != 0, 0.0).where(down != 0, 100.0)
    return rsi.where(up.notna() & down.notna())


def bx_histogram(close: pd.Series, l1: int = 5, l2: int = 20, l3: int = 5) -> pd.Series:
    """One security's ordered closes -> the short-term histogram, first defined at row l3."""
    return pine_rsi(pine_ema(close, l1) - pine_ema(close, l2), l3) - 50


# -- split adjustment -----------------------------------------------------------------------------


def adjust_for_splits(bars: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Divide each security's prices before a split's ex-date by its factor (several multiply).

    A split (S, d) belongs to the security that carries S at the boundary d: its last bar
    before d or its first bar on/after d has symbol S. That covers a record filed under the old
    or the new symbol on a rename day (NLSP -> NCEL 2025-10-31), a split known before its first
    bar exists, and a label another security took over later. A security with no bar on/after
    d counts only if its last bar is within ``_STALE_DAYS`` of d (the ex-date morning, before
    that day's bar). Splits are used as given; the caller cuts them at ``as_of``."""
    out = bars.sort_values(["security_id", "session_date"]).reset_index(drop=True)
    if out.empty or splits.empty:
        return out
    divisor = np.ones(len(out))
    rows = out.groupby("security_id", sort=False).indices
    holders: dict[str, set[str]] = {}
    for symbol, sid in zip(out["symbol"], out["security_id"], strict=True):
        holders.setdefault(symbol, set()).add(sid)
    days = np.array([d.toordinal() for d in out["session_date"]])
    symbols = out["symbol"].to_numpy()
    for symbol, ex, factor in zip(
        splits["symbol"], _as_dates(pd.Series(splits["ex_date"])), splits["factor"], strict=True
    ):
        for sid in holders.get(symbol, ()):
            idx = rows[sid]
            cut = int(np.searchsorted(days[idx], ex.toordinal()))
            # a security that stopped trading long before d is an earlier holder of the label
            # (gone in 2021, the label reused and split in 2025), not the one that splits
            alive = cut < len(idx) or ex.toordinal() - days[idx[-1]] <= _STALE_DAYS
            before = cut > 0 and alive and symbols[idx[cut - 1]] == symbol
            after = cut < len(idx) and symbols[idx[cut]] == symbol
            if before or after:
                divisor[idx[:cut]] *= float(factor)
    for column in PRICES:
        out[column] = out[column] / divisor
    return out


def _as_dates(values: pd.Series) -> list[date]:
    return [v if type(v) is date else date.fromisoformat(str(v)[:10]) for v in values]


# -- weekly and monthly bars ----------------------------------------------------------------------


def _period_start(day: date, period: Period) -> date:
    return day - timedelta(days=day.weekday()) if period == "W" else day.replace(day=1)


def _last_weekday(start: date, period: Period) -> date:
    if period == "W":
        return start + timedelta(days=4)
    end = (start + timedelta(days=32)).replace(day=1) - timedelta(days=1)
    return end - timedelta(days=max(0, end.weekday() - 4))


def period_bars(daily: pd.DataFrame, period: Period) -> pd.DataFrame:
    """Completed calendar weeks (Monday-Sunday) or months per security: first open, highest
    high, lowest low, last close, ``period_end`` (its last session) and ``complete_on``. A
    period not yet known to be over at the end of the input is left out."""
    df = daily.sort_values(["security_id", "session_date"]).reset_index(drop=True)
    df["period_start"] = [_period_start(d, period) for d in df["session_date"]]
    grouped = df.groupby(["security_id", "period_start"], sort=True)
    out = grouped.agg(
        period_end=("session_date", "last"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    ).reset_index()
    next_first = grouped["session_date"].first().groupby(level=0).shift(-1).to_numpy()
    complete_on = []
    for start, end, later in zip(out["period_start"], out["period_end"], next_first, strict=True):
        if end == _last_weekday(start, period):
            complete_on.append(end)
        else:
            complete_on.append(later if isinstance(later, date) else None)
    out["complete_on"] = complete_on
    out = out[out["complete_on"].notna()]
    return pd.DataFrame(out[PERIOD_COLUMNS]).reset_index(drop=True)


# -- all three per session ------------------------------------------------------------------------


def bx_features(daily: pd.DataFrame, *, l1: int = 5, l2: int = 20, l3: int = 5) -> pd.DataFrame:
    """``daily`` (already split-adjusted) sorted by security and session, with ``bx_daily`` and
    ``bx_weekly`` / ``bx_monthly``: the histogram of the latest completed period whose
    ``complete_on`` is not after the session. Two rows for one security and session (a
    rename day read under both symbols) are an error, not a value."""
    out = daily.sort_values(["security_id", "session_date"]).reset_index(drop=True)
    _require_one_row_per_session(out)
    out["bx_daily"] = out.groupby("security_id", sort=False)["close"].transform(
        lambda c: bx_histogram(c, l1, l2, l3)
    )
    periods: tuple[tuple[Period, str], ...] = (("W", "bx_weekly"), ("M", "bx_monthly"))
    for period, name in periods:
        bars = period_bars(out, period)
        bars[name] = bars.groupby("security_id", sort=False)["close"].transform(
            lambda c: bx_histogram(c, l1, l2, l3)
        )
        out[name] = _as_of_join(out, bars, name)
    return out


def _require_one_row_per_session(daily: pd.DataFrame) -> None:
    """A second row would take one more step of every recursion. Real: SABS and BCYP both
    carry S010438 on 170 days of 2021."""
    repeated = daily[daily.duplicated(["security_id", "session_date"], keep=False)]
    if not repeated.empty:
        pairs = pd.DataFrame(repeated[["security_id", "session_date"]]).drop_duplicates().head(5)
        shown = ", ".join(f"{s} {d}" for s, d in pairs.itertuples(index=False))
        raise ValueError(f"{len(repeated)} rows share a security and session: {shown}")


def _as_of_join(sessions: pd.DataFrame, bars: pd.DataFrame, name: str) -> np.ndarray:
    """For each session row, ``bars[name]`` of the same security's latest ``complete_on`` on
    or before the session."""
    left = pd.DataFrame(
        {
            "security_id": sessions["security_id"].to_numpy(),
            "_t": pd.to_datetime(sessions["session_date"]),
            "_row": np.arange(len(sessions)),
        }
    ).sort_values("_t")
    right = pd.DataFrame(
        {
            "security_id": bars["security_id"].to_numpy(),
            "_t": pd.to_datetime(bars["complete_on"]),
            name: bars[name].to_numpy(dtype=float),
        }
    ).sort_values("_t")
    joined = pd.merge_asof(left, right, on="_t", by="security_id", direction="backward")
    return joined.sort_values("_row")[name].to_numpy(dtype=float)
