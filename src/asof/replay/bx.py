"""What the B-Xtrender replay reads off a signal night to size an entry.

The scan's panel is computed once at the window's end, so its prices carry every split visible
then: a 10-for-1 three months after the signal would make the night's close and ATR a tenth of
what the market showed. Sizing reads the night's own bars instead, adjusted only by the splits
visible that night whose ex-date has come: the ATR is then in the unit of the night's raw close.
"""

from datetime import date

import pandas as pd

from asof.execute.bx import signal_atr
from asof.features.bx import adjust_for_splits


def signal_inputs(
    bars: pd.DataFrame, splits: pd.DataFrame, day: date, atr_days: int
) -> dict[str, tuple[float, float]]:
    """security_id -> (close, ATR) on ``day`` for each security of ``bars`` with a bar on it.

    ``bars``: the bars visible at the night's close (security_id, symbol, session_date, open,
    high, low, close), from enough sessions back for the ATR; ``splits``: the split records
    visible then. A split with its ex-date after ``day`` is left out, even when already on file:
    the night's close is still in the old unit. The ATR is NaN when history is short."""
    if bars.empty:
        return {}
    bars = pd.DataFrame(bars[bars["session_date"] <= day])
    if not splits.empty:
        ex = [d if type(d) is date else date.fromisoformat(str(d)[:10]) for d in splits["ex_date"]]
        splits = pd.DataFrame(splits[[d <= day for d in ex]])
    adjusted = adjust_for_splits(bars, splits)
    out: dict[str, tuple[float, float]] = {}
    for sid, rows in adjusted.groupby("security_id", sort=False):
        if rows["session_date"].iloc[-1] != day:
            continue
        # the last row is the night's own bar: no split with ex-date <= day divides it
        out[str(sid)] = (float(rows["close"].iloc[-1]), signal_atr(rows, atr_days))
    return out
