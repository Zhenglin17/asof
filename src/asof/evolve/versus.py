"""A replay's daily equity against the protocol's benchmarks over the run's own sessions.

The three criteria are the ones the walk-forward gate will apply (DECISIONS 2026-10-26); here
they are only reported. SPY is the buy-and-hold of the protocol's first symbol, the basket is the
last curve.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb
from sqlmodel import Session

from asof.evolve.benchmark import benchmarks
from asof.evolve.metrics import CurveMetrics, curve_metrics
from asof.evolve.protocol import EvaluationProtocol
from asof.store.models import Run
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


def compare_run(
    session: Session,
    con: duckdb.DuckDBPyConnection,
    root: Path,
    run_id: int,
    protocol: EvaluationProtocol,
) -> Versus:
    run = session.get(Run, run_id)
    if run is None:
        raise LookupError(f"no run {run_id}")
    rows = visible_equity(session, run.as_of, run_id=run_id)
    if not rows:
        raise LookupError(f"run {run_id} has no daily equity; replay it again to record one")
    start, end = rows[0].session_date, rows[-1].session_date
    rf = protocol.comparison.risk_free_rate
    first = rows[0]
    strategy = curve_metrics(
        [r.equity for r in rows],
        base=first.equity - first.realized - first.unrealized,
        risk_free_rate=rf,
    )
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
