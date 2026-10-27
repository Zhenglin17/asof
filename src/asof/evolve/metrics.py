"""Statistics of one equity curve, shared by replays and benchmarks so both are read the same way.

A curve is the value at each session's close and ``base``, the value before the first session
(money invested at its open, or a replay's starting equity), so the first day's move counts.
"""

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

TRADING_DAYS = 252
# Daily returns spread less than this are a flat line read through float noise: no Sharpe.
_MIN_SPREAD = 1e-12


@dataclass(frozen=True)
class CurveMetrics:
    sessions: int
    total_return: float
    annualized_return: float
    max_drawdown: float  # largest fall from a running peak, as a positive fraction
    sharpe: float | None  # None with fewer than two returns or no spread


def curve_metrics(
    equity: Sequence[float], *, base: float, risk_free_rate: float = 0.0
) -> CurveMetrics:
    if not equity:
        raise ValueError("an equity curve needs at least one session")
    if base <= 0:
        raise ValueError(f"base must be positive, got {base}")
    if min(equity) <= 0:
        raise ValueError("equity must stay positive")
    values = [base, *equity]
    returns = [b / a - 1 for a, b in zip(values, values[1:], strict=False)]
    total = equity[-1] / base - 1
    peak, worst = base, 0.0
    for value in equity:
        peak = max(peak, value)
        worst = max(worst, (peak - value) / peak)
    sharpe = None
    if len(returns) >= 2:
        daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1
        excess = [r - daily_rf for r in returns]
        spread = statistics.stdev(excess)
        if spread > _MIN_SPREAD:
            sharpe = statistics.fmean(excess) / spread * math.sqrt(TRADING_DAYS)
    return CurveMetrics(
        sessions=len(equity),
        total_return=total,
        annualized_return=(1 + total) ** (TRADING_DAYS / len(equity)) - 1,
        max_drawdown=worst,
        sharpe=sharpe,
    )
