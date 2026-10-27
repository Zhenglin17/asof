"""A replay's daily equity against the protocol's benchmarks over the run's own sessions.

The three criteria are the ones the walk-forward gate will apply (DECISIONS 2026-10-26); here
they are only reported. SPY is the buy-and-hold of the protocol's first symbol, the basket is the
last curve.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb
from sqlmodel import Session

from asof.evolve.benchmark import benchmarks
from asof.evolve.metrics import CurveMetrics, curve_metrics
from asof.evolve.protocol import EvaluationProtocol
from asof.store.models import EquityDaily, Run
from asof.store.queries import visible_equity


@dataclass(frozen=True)
class Versus:
    run_id: int
    start: date
    end: date
    strategy: CurveMetrics
    benchmarks: list[tuple[str, CurveMetrics]]
    mean_exposure: float  # open positions / equity, averaged over sessions
    high_exposure_share: float  # sessions with exposure >= protocol.comparison.high_exposure
    drawdown_not_worse_than_basket: bool
    sharpe_at_least_best: bool  # against the better of SPY and the basket
    beats_spy_and_basket_total: bool


def equity_slice(
    rows: Sequence[EquityDaily], start: date, end: date
) -> tuple[list[EquityDaily], float]:
    """The rows within [start, end] and the equity just before the first of them: the previous
    row's when the run had one, else the run's initial equity."""
    i = next((k for k, r in enumerate(rows) if r.session_date >= start), len(rows))
    kept = [r for r in rows[i:] if r.session_date <= end]
    if i > 0:
        return kept, rows[i - 1].equity
    first = rows[0] if rows else None
    return kept, (first.equity - first.realized - first.unrealized) if first else 0.0


def compare_run(
    session: Session,
    con: duckdb.DuckDBPyConnection,
    root: Path,
    run_id: int,
    protocol: EvaluationProtocol,
    *,
    window: tuple[date, date] | None = None,
) -> Versus:
    """``window`` keeps the run's rows within [first, last]; None is the whole run."""
    run = session.get(Run, run_id)
    if run is None:
        raise LookupError(f"no run {run_id}")
    every = visible_equity(session, run.as_of, run_id=run_id)
    if not every:
        raise LookupError(f"run {run_id} has no daily equity; replay it again to record one")
    rows, base = equity_slice(every, *(window or (date.min, date.max)))
    if not rows:
        raise LookupError(f"run {run_id} has no daily equity in the window {window}")
    start, end = rows[0].session_date, rows[-1].session_date
    rf = protocol.comparison.risk_free_rate
    strategy = curve_metrics([r.equity for r in rows], base=base, risk_free_rate=rf)
    curves = benchmarks(con, root, protocol, start, end)
    days = tuple(r.session_date for r in rows)
    for c in curves:
        if c.sessions != days:
            # The calendar is recomputed from today's store; a bar added since the replay would
            # misalign the two return series without any error.
            raise ValueError(
                f"run {run_id}: {c.label} has {len(c.sessions)} sessions, the run {len(days)}"
            )
    marks = [(c.label, curve_metrics(c.equity, base=1.0, risk_free_rate=rf)) for c in curves]
    spy, basket = marks[0][1], marks[-1][1]
    exposure = [r.gross_exposure / r.equity for r in rows]
    sharpe_ok = (
        strategy.sharpe is not None
        and spy.sharpe is not None
        and basket.sharpe is not None
        and strategy.sharpe >= max(spy.sharpe, basket.sharpe)
    )
    return Versus(
        run_id=run_id,
        start=start,
        end=end,
        strategy=strategy,
        benchmarks=marks,
        mean_exposure=sum(exposure) / len(exposure),
        high_exposure_share=sum(e >= protocol.comparison.high_exposure for e in exposure)
        / len(exposure),
        drawdown_not_worse_than_basket=strategy.max_drawdown <= basket.max_drawdown,
        sharpe_at_least_best=sharpe_ok,
        beats_spy_and_basket_total=strategy.total_return
        > max(spy.total_return, basket.total_return),
    )
