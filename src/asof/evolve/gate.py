"""The gate's verdict on a family of replays, over the protocol's development dates only.

A variant passes when its edge over random entries survives the family's max-statistic
correction, holds up walk-forward (a pick made on earlier months beats random entries on the
months after), survives dropping its best legs and an extra cost, and does not trail the
benchmarks on drawdown, Sharpe ratio or total return. Nothing dated after dev counts: a leg
must enter and exit within it, equity rows are cut at its end, and prices are read as known at
the last dev exit. The minimum detectable edge and the number of versions replayed on dev are
reported, not judged.
"""

import math
import statistics
from calendar import monthrange
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
from sqlmodel import Session

from asof.evolve.metrics import TRADING_DAYS, CurveMetrics, curve_metrics
from asof.evolve.null import (
    Span,
    VariantResult,
    build_panel,
    max_statistic,
    null_returns,
    spans_of_run,
)
from asof.evolve.protocol import EvaluationProtocol, Window
from asof.evolve.versus import Versus, compare_run, equity_slice
from asof.store.models import EquityDaily, Run
from asof.store.queries import runs_of_strategy, visible_equity, visible_trades
from asof.strategy import StrategyConfig

ET = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class Segment:
    train: tuple[date, date]
    test: tuple[date, date]
    picked: int | None  # best train-window Sharpe; None when no run has a row in it: the
    # window cannot be scored at all, which is reported apart from a pick that lost
    legs: int  # the pick's dev legs entering in the test window
    mean: float | None
    null_mean: float | None
    positive: bool


@dataclass(frozen=True)
class GateResult:
    dev: Window
    candidate: int  # best Sharpe over dev
    variants: dict[int, VariantResult]
    segments: list[Segment]
    walk_forward_positive: int
    versus: Versus
    mde: float | None  # one-sided, at the protocol's alpha and mde_power; None below 2 legs
    thin_excess: float | None  # None when dropping the best legs leaves none
    thin_edge_ok: bool
    # The two parts of the thin edge alone (report only): best legs dropped, extra cost charged.
    thin_drop_excess: float | None
    thin_cost_excess: float
    versions_on_dev: int
    left_out: int  # the candidate's legs entering in dev but exiting after it
    seen_before_protocol: bool  # replayed over these dates before the protocol: no verdict
    passed: bool
    reasons: list[str]


def _add_months(day: date, months: int) -> date:
    year, month = divmod(day.month - 1 + months, 12)
    year += day.year
    return date(year, month + 1, min(day.day, monthrange(year, month + 1)[1]))


def walk_forward_windows(
    dev: Window, train_months: int, test_months: int
) -> list[tuple[tuple[date, date], tuple[date, date]]]:
    """(train, test) pairs rolling by ``test_months``; a test window must end within dev."""
    out = []
    k = 0
    while True:
        test_start = _add_months(dev.start, train_months + k * test_months)
        test_end = _add_months(dev.start, train_months + (k + 1) * test_months) - timedelta(days=1)
        if test_end > dev.end:
            return out
        train = (_add_months(dev.start, k * test_months), test_start - timedelta(days=1))
        out.append((train, (test_start, test_end)))
        k += 1


def _close_of(day: date) -> datetime:
    return datetime.combine(day, time(20), tzinfo=ET)


def _sharpe(rows: Sequence[EquityDaily], start: date, end: date, rf: float) -> float | None:
    kept, base = equity_slice(rows, start, end)
    if not kept:
        return None
    return curve_metrics([r.equity for r in kept], base=base, risk_free_rate=rf).sharpe


def _best(sharpes: Mapping[int, float | None]) -> int:
    """Highest Sharpe; no Sharpe ranks below any number; ties go to the lowest run id."""
    return max(sharpes, key=lambda r: (sharpes[r] is not None, sharpes[r] or 0.0, -r))


def _versions_on_dev(session: Session, strategy_id: str | None, dev: Window) -> int:
    """Versions of the strategy with a trade entering, or a mark, in dev: what a reader of the
    verdict should count as tried. A run without a version (older than versions) counts alone."""
    as_of = _close_of(dev.end)
    runs = [r for r in runs_of_strategy(session, strategy_id) if r.id is not None]
    traded = {
        t.run_id
        for t in visible_trades(session, as_of, strategy_id=strategy_id)
        if t.entry_at.astimezone(ET).date() >= dev.start
    }
    tried = {
        r.strategy_version_id if r.strategy_version_id is not None else f"run {r.id}"
        for r in runs
        if r.id in traded
        or any(x.session_date >= dev.start for x in visible_equity(session, as_of, run_id=r.id))  # pyright: ignore[reportArgumentType]
    }
    return len(tried)


def check(
    session: Session,
    con: duckdb.DuckDBPyConnection,
    root: Path,
    run_ids: Sequence[int],
    *,
    cfg: StrategyConfig,
    protocol: EvaluationProtocol,
    permutations: int,
    seed: int,
) -> GateResult:
    """``run_ids`` are the family; ``cfg`` gives the random-entry pool and the slippage."""
    dev = protocol.dev
    rf = protocol.comparison.risk_free_rate
    spans: dict[int, list[Span]] = {}
    marks: dict[int, list[EquityDaily]] = {}
    late: dict[int, int] = {}
    for r in run_ids:
        legs = [s for s in spans_of_run(session, r) if s.entry_session >= dev.start]
        spans[r] = [s for s in legs if s.exit_session <= dev.end]
        late[r] = sum(s.entry_session <= dev.end < s.exit_session for s in legs)
        if not spans[r]:
            raise ValueError(f"run {r} has no leg entering and exiting within dev")
        run = session.get(Run, r)
        assert run is not None  # spans_of_run raised otherwise
        rows = [
            x for x in visible_equity(session, run.as_of, run_id=r) if x.session_date <= dev.end
        ]
        if not any(x.session_date >= dev.start for x in rows):
            raise ValueError(f"run {r} has no daily equity in dev; replay it again to record one")
        marks[r] = rows

    every = [s for legs in spans.values() for s in legs]
    panel = build_panel(
        session, con, root, every, cfg=cfg, as_of=_close_of(max(s.exit_session for s in every))
    )
    real = {r: np.array([s.ret for s in legs]) for r, legs in spans.items()}
    null = {
        r: null_returns(
            legs, panel, permutations=permutations, seed=seed,
            slippage_bps=cfg.execute.fills.slippage_bps,
        )
        for r, legs in spans.items()
    }  # fmt: skip
    family = max_statistic(
        {str(r): v for r, v in real.items()}, {str(r): v for r, v in null.items()}
    )
    variants = {r: family[str(r)] for r in run_ids}

    candidate = _best({r: _sharpe(marks[r], dev.start, dev.end, rf) for r in run_ids})

    wf = protocol.walk_forward
    segments = []
    for train, test in walk_forward_windows(dev, wf.train_months, wf.test_months):
        in_train = [r for r in run_ids if equity_slice(marks[r], *train)[0]]
        if not in_train:
            segments.append(Segment(train, test, None, 0, None, None, False))
            continue
        pick = _best({r: _sharpe(marks[r], *train, rf) for r in in_train})
        idx = [i for i, s in enumerate(spans[pick]) if test[0] <= s.entry_session <= test[1]]
        if not idx:
            segments.append(Segment(train, test, pick, 0, None, None, False))
            continue
        mean = float(real[pick][idx].mean())
        null_mean = float(null[pick][:, idx].mean())
        segments.append(Segment(train, test, pick, len(idx), mean, null_mean, mean > null_mean))
    positive = sum(s.positive for s in segments)

    # Thin edge: the best legs dropped, and an extra cost a side charged to the real legs only
    # (the random legs are what luck earns, not trades anyone pays for).
    returns, draws = real[candidate], null[candidate]
    keep = np.argsort(-returns, kind="stable")[protocol.thin_edge.drop_top :]
    c = protocol.thin_edge.extra_cost_bps / 10_000

    def charged(values: np.ndarray) -> np.ndarray:
        return (1 + values) * (1 - c) / (1 + c) - 1

    thin_excess = thin_drop = None
    if len(keep):
        thin_excess = float(charged(returns[keep]).mean() - draws[:, keep].mean())
        thin_drop = float(returns[keep].mean() - draws[:, keep].mean())
    thin_cost = float(charged(returns).mean() - draws.mean())
    thin_ok = thin_excess is not None and thin_excess > 0

    mde = None
    if len(returns) >= 2:
        z = statistics.NormalDist().inv_cdf
        factor = z(1 - protocol.alpha) + z(protocol.mde_power)
        mde = factor * statistics.stdev(returns.tolist()) / math.sqrt(len(returns))

    versus = compare_run(session, con, root, candidate, protocol, window=(dev.start, dev.end))
    run = session.get(Run, candidate)
    seen = run is not None and run.strategy_id in protocol.seen_before_protocol
    # Windows whose train months hold no row cannot be scored; when too few can, the failure is
    # named for that, not for a lack of edge. The count needed stays the protocol's.
    scorable = sum(s.picked is not None for s in segments)
    walk_forward = "walk_forward" if scorable >= wf.min_positive else "walk_forward_not_evaluable"
    checks = {
        # The protocol counts a strategy replayed over these dates before it only after
        # clean_after: on dev it gets numbers, never a pass.
        "seen_before_protocol": not seen,
        "null": variants[candidate].p_adjusted < protocol.alpha,
        walk_forward: positive >= wf.min_positive,
        "thin_edge": thin_ok,
        "drawdown_not_worse_than_basket": versus.drawdown_not_worse_than_basket,
        "sharpe_at_least_best": versus.sharpe_at_least_best,
        "beats_spy_and_basket_total": versus.beats_spy_and_basket_total,
    }
    reasons = [name for name, ok in checks.items() if not ok]
    return GateResult(
        dev=dev,
        candidate=candidate,
        variants=variants,
        segments=segments,
        walk_forward_positive=positive,
        versus=versus,
        mde=mde,
        thin_excess=thin_excess,
        thin_edge_ok=thin_ok,
        thin_drop_excess=thin_drop,
        thin_cost_excess=thin_cost,
        versions_on_dev=_versions_on_dev(session, run.strategy_id if run else None, dev),
        left_out=late[candidate],
        seen_before_protocol=seen,
        passed=not reasons,
        reasons=reasons,
    )


def sharpe_standard_error(sharpe: float, sessions: int) -> float:
    """Lo (2002) for an annualized Sharpe ratio of daily returns: sqrt((1 + SR^2 / 504) / years)."""
    return math.sqrt((1 + sharpe**2 / (2 * TRADING_DAYS)) / (sessions / TRADING_DAYS))


def _pct(value: float | None) -> str:
    return f"{value:+.2%}" if value is not None else "-"


def _metrics_row(label: str, m: CurveMetrics) -> str:
    sharpe = f"{m.sharpe:.2f}" if m.sharpe is not None else "-"
    return (
        f"| {label} | {m.total_return:+.2%} | {m.annualized_return:+.2%} | "
        f"{m.max_drawdown:.2%} | {sharpe} |"
    )


def render_report(session: Session, result: GateResult, protocol: EvaluationProtocol) -> str:
    dev, cand = result.dev, result.candidate
    verdict = "PASS" if result.passed else "FAIL: " + ", ".join(result.reasons)
    against = result.versus
    out = [
        f"# Gate: runs {', '.join(str(r) for r in result.variants)}",
        "",
        f"Development dates {dev.start} .. {dev.end}; the candidate's equity covers "
        f"{against.start} .. {against.end} of them. Candidate run {cand} (best Sharpe over dev). "
        f"Verdict: **{verdict}**.",
        "",
    ]
    if result.seen_before_protocol:
        out += [
            "This strategy was replayed over these dates before the protocol was registered "
            f"(`seen_before_protocol`): its dev numbers are shown for reference, and the protocol "
            f"judges it only on dates from {protocol.clean_after} on.",
            "",
        ]
    out += [
        "## Random entries (family, max-statistic corrected)",
        "",
        "| run | legs | mean | random | excess | p | p adj |",
        "|---|--:|--:|--:|--:|--:|--:|",
    ]
    for r, v in result.variants.items():
        out.append(
            f"| {r} | {v.trades} | {v.mean:+.2%} | {v.null_mean:+.2%} | {v.excess:+.2%} | "
            f"{v.p_raw:.4f} | {v.p_adjusted:.4f} |"
        )
    wf = protocol.walk_forward
    out += [
        "",
        f"## Walk-forward ({wf.train_months} months to pick, {wf.test_months} to score; "
        f"{result.walk_forward_positive} of {len(result.segments)} positive, "
        f"{wf.min_positive} needed)",
        "",
        "| train | test | pick | legs | mean | random | positive |",
        "|---|---|--:|--:|--:|--:|---|",
    ]
    for s in result.segments:
        status = "not evaluable" if s.picked is None else ("yes" if s.positive else "no")
        out.append(
            f"| {s.train[0]} .. {s.train[1]} | {s.test[0]} .. {s.test[1]} | "
            f"{s.picked if s.picked is not None else '-'} | {s.legs} | {_pct(s.mean)} | "
            f"{_pct(s.null_mean)} | {status} |"
        )
    out.append("")
    out.append(
        "A window is not evaluable when no run has equity in its train months; it counts as not "
        "positive, and the number needed does not shrink."
    )
    if len(result.variants) == 1:
        out.append(
            "With one run in the family every pick is that run: the windows test the stability "
            "of its edge, not a choice."
        )
    te = protocol.thin_edge
    out += [
        "",
        "## Thin edge, MDE, attempts",
        "",
        f"- Thin edge: without the best {te.drop_top} legs (by return) and with "
        f"{te.extra_cost_bps:g} bps more a side on the real legs, excess "
        f"{_pct(result.thin_excess)} a leg ({'ok' if result.thin_edge_ok else 'fails'}). Apart: "
        f"dropping the legs alone {_pct(result.thin_drop_excess)}, the cost alone "
        f"{_pct(result.thin_cost_excess)}.",
        f"- MDE (report only): {_pct(result.mde)} a leg at one-sided alpha {protocol.alpha:g} and "
        f"power {protocol.mde_power:g}, over {result.variants[cand].trades} legs; from the spread "
        "of the real legs, not corrected for the family.",
        f"- Versions of this strategy replayed on dev dates: {result.versions_on_dev}.",
        f"- Legs entering in dev but exiting after it, left out: {result.left_out}.",
        "",
        f"## Benchmarks over {against.start} .. {against.end}",
        "",
        "| | total | annualized | max dd | Sharpe |",
        "|---|--:|--:|--:|--:|",
        _metrics_row(f"run {cand}", result.versus.strategy),
        *(_metrics_row(label, m) for label, m in result.versus.benchmarks),
        "",
        f"Mean exposure {result.versus.mean_exposure:.1%}; criteria: drawdown not worse than the "
        f"basket's {'yes' if result.versus.drawdown_not_worse_than_basket else 'no'}, Sharpe at "
        f"least the better of SPY's and the basket's "
        f"{'yes' if result.versus.sharpe_at_least_best else 'no'}, total above both "
        f"{'yes' if result.versus.beats_spy_and_basket_total else 'no'}.",
        "",
        "## Candidate by period (report only; the verdict reads dev alone)",
        "",
        "| period | sessions | total | max dd | Sharpe | Sharpe s.e. |",
        "|---|--:|--:|--:|--:|--:|",
    ]
    run = session.get(Run, cand)
    rows = visible_equity(session, run.as_of, run_id=cand) if run else []
    holdout = "holdout dates" + (
        " (seen before the protocol)" if result.seen_before_protocol else ""
    )
    periods = [
        ("dev", dev.start, dev.end),
        (holdout, protocol.holdout.start, protocol.holdout.end),
        ("after the model cutoff", protocol.clean_after, date.max),
    ]
    for name, start, end in periods:
        kept, base = equity_slice(rows, start, end)
        if not kept:
            continue
        m = curve_metrics(
            [r.equity for r in kept], base=base, risk_free_rate=protocol.comparison.risk_free_rate
        )
        se = f"{sharpe_standard_error(m.sharpe, m.sessions):.2f}" if m.sharpe is not None else "-"
        sharpe = f"{m.sharpe:.2f}" if m.sharpe is not None else "-"
        out.append(
            f"| {name} {kept[0].session_date} .. {kept[-1].session_date} | {m.sessions} | "
            f"{m.total_return:+.2%} | {m.max_drawdown:.2%} | {sharpe} | {se} |"
        )
    legs = [
        t
        for t in visible_trades(
            session, _close_of(dev.end), strategy_id=run.strategy_id if run else None
        )
        if t.run_id == cand and t.entry_at.astimezone(ET).date() >= dev.start
    ]
    legs.sort(key=lambda t: -t.pnl)
    out += [
        "",
        "## Candidate's top legs by pnl (read them before trusting the verdict)",
        "",
        "| symbol | entry | exit | pnl | return |",
        "|---|---|---|--:|--:|",
        *(
            f"| {t.symbol} | {t.entry_at.astimezone(ET).date()} | "
            f"{t.exit_at.astimezone(ET).date()} | {t.pnl:,.2f} | "
            f"{t.pnl / (t.shares * t.entry_price):+.2%} |"
            for t in legs[:10]
        ),
        "",
        "Legs exiting after dev are left out of every statistic, but a position still open at "
        "dev's end is in the dev equity at its last close.",
        "",
    ]
    return "\n".join(out)
