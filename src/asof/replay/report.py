"""Metrics and a markdown report for one finished replay run.

Returns are never shown alone: the report carries the number of parameter sets tried for this
strategy (each is an attempt for the gate) and, when the caller supplies it, a buy-and-hold
benchmark over the same sessions. The drawdown is taken on realized equity, stepped at each exit
day; positions are not marked to market in between, so it understates the true drawdown.
"""

import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any
from zoneinfo import ZoneInfo

from sqlmodel import Session, col, select

from asof.evolve.protocol import EvaluationProtocol, load_protocol
from asof.store.models import (
    Attribution,
    Decision,
    Direction,
    Document,
    Entity,
    HoldoutOpening,
    Run,
    Trade,
)
from asof.strategy import attempt_count, replay_count

ET = ZoneInfo("America/New_York")
DEFAULT_EQUITY = 100_000.0
TOP_N = 10


@dataclass(frozen=True)
class ReplayMetrics:
    decisions: int
    trades_by_leg: dict[int, int]
    win_rate: float
    mean_r_by_leg: dict[int, float]
    mean_pnl_per_trade: float
    total_return: float
    max_drawdown: float
    top3_share: float | None
    exit_reasons: dict[str, int]
    attempt_count: int
    holding_sessions: list[int] = field(default_factory=list)
    open_decisions: int = 0
    traded_decisions: int = 0
    mean_pnl_by_leg: dict[int, float] = field(default_factory=dict)
    top10_share: float | None = None
    cancelled: dict[str, int] = field(default_factory=dict)
    replay_count: int = 0


def _run(session: Session, run_id: int) -> Run:
    run = session.get(Run, run_id)
    if run is None:
        raise LookupError(f"no run {run_id}")
    return run


def _decisions(session: Session, run_id: int) -> list[Decision]:
    stmt = select(Decision).where(Decision.run_id == run_id).order_by(col(Decision.id))
    return list(session.exec(stmt).all())


def _trades(session: Session, run_id: int) -> list[Trade]:
    stmt = select(Trade).where(Trade.run_id == run_id).order_by(col(Trade.available_at))
    return list(session.exec(stmt).all())


def _initial_equity(decisions: list[Decision]) -> float:
    for decision in decisions:
        equity = decision.payload.get("equity_at_decision")
        if equity is not None:
            return float(equity)
    return DEFAULT_EQUITY


def replay_metrics(
    session: Session, run_id: int, initial_equity: float | None = None
) -> ReplayMetrics:
    run = _run(session, run_id)
    decisions = _decisions(session, run_id)
    trades = _trades(session, run_id)
    equity0 = initial_equity if initial_equity is not None else _initial_equity(decisions)

    by_decision: dict[int, float] = defaultdict(float)
    for t in trades:
        by_decision[t.decision_id] += t.pnl
    traded = list(by_decision.values())
    total = sum(traded)
    win_rate = sum(1 for v in traded if v > 0) / len(traded) if traded else 0.0

    r_by_leg: dict[int, list[float]] = defaultdict(list)
    for t in trades:
        r_by_leg[t.leg].append(t.r_multiple)

    # realized equity, one point per exit day
    by_day: dict[Any, float] = defaultdict(float)
    for t in trades:
        by_day[t.available_at] += t.pnl
    equity, peak, max_dd = equity0, equity0, 0.0
    for day in sorted(by_day):
        equity += by_day[day]
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak)

    best = sorted(traded, reverse=True)
    top3, top10 = sum(best[:3]), sum(best[:10])
    pnl_by_leg: dict[int, list[float]] = defaultdict(list)
    for t in trades:
        pnl_by_leg[t.leg].append(t.pnl)
    cancelled = Counter(
        str(d.payload["cancelled"]) for d in decisions if d.payload.get("cancelled") is not None
    )
    attributed = set(
        session.exec(
            select(Attribution.decision_id).where(
                col(Attribution.decision_id).in_([d.id for d in decisions])
            )
        ).all()
    )
    return ReplayMetrics(
        decisions=len(decisions),
        trades_by_leg=dict(Counter(t.leg for t in trades)),
        win_rate=win_rate,
        mean_r_by_leg={leg: statistics.fmean(rs) for leg, rs in sorted(r_by_leg.items())},
        mean_pnl_per_trade=total / len(trades) if trades else 0.0,
        total_return=total / equity0,
        max_drawdown=max_dd,
        # A share of a loss or of nothing says nothing.
        top3_share=top3 / total if total > 0 else None,
        top10_share=top10 / total if total > 0 else None,
        traded_decisions=len(traded),
        mean_pnl_by_leg={leg: statistics.fmean(v) for leg, v in sorted(pnl_by_leg.items())},
        exit_reasons=dict(Counter(t.exit_reason for t in trades)),
        attempt_count=attempt_count(session, run.strategy_id or ""),
        holding_sessions=[t.holding_sessions for t in trades if t.holding_sessions is not None],
        # A pass or no-call (direction flat) never opens a position.
        open_decisions=sum(
            1
            for d in decisions
            if d.direction == Direction.LONG.value
            and d.id not in attributed
            and "cancelled" not in d.payload
        ),
        cancelled=dict(cancelled),
        replay_count=replay_count(session, run.strategy_id or ""),
    )


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def _top_decisions(session: Session, run_id: int, trades: list[Trade]) -> list[str]:
    legs: dict[int, list[Trade]] = defaultdict(list)
    for t in trades:
        legs[t.decision_id].append(t)
    ranked = sorted(legs.items(), key=lambda item: -abs(sum(t.pnl for t in item[1])))[:TOP_N]
    lines = [
        "| ticker | decided | legs | pnl | R per leg | exits | filing |",
        "|---|---|---|---|---|---|---|",
    ]
    for decision_id, rows in ranked:
        decision = session.get(Decision, decision_id)
        if decision is None:
            continue
        entity = session.get(Entity, decision.entity_id) if decision.entity_id else None
        candidate = decision.payload.get("candidate", {})
        doc_ids = candidate.get("document_ids", [])
        doc = session.get(Document, doc_ids[0]) if doc_ids else None
        filing = (
            f"{doc.title} ({doc.available_at.astimezone(ET):%Y-%m-%d %H:%M} New York)"
            if doc
            else ""
        )
        lines.append(
            f"| {entity.ticker if entity else '?'} | {decision.as_of.astimezone(ET):%Y-%m-%d} "
            f"| {len(rows)} "
            f"| {sum(t.pnl for t in rows):,.2f} "
            f"| {', '.join(f'{t.r_multiple:+.2f}' for t in rows)} "
            f"| {', '.join(t.exit_reason for t in rows)} | {filing} |"
        )
    return lines


def _holdout_lines(session: Session, run: Run, protocol: EvaluationProtocol) -> list[str]:
    sealed = f"{protocol.holdout.start} .. {protocol.holdout.end}"
    if run.strategy_id in protocol.seen_before_protocol:
        return [
            f"Holdout: this strategy was replayed over the sealed holdout {sealed} before the "
            f"evaluation protocol existed, so no date before {protocol.clean_after} is unseen "
            "data for it.",
            "",
        ]
    opened = session.exec(select(HoldoutOpening).where(HoldoutOpening.run_id == run.id)).first()
    if opened is not None:
        return [f"Holdout: this run is the strategy's one opening of {sealed}.", ""]
    return []


def _excursion_lines(trades: list[Trade]) -> list[str]:
    """Medians over legs that recorded excursions; kept = realized % / MFE %, over MFE > 0."""
    rows = [t for t in trades if t.mae_pct is not None and t.mfe_pct is not None]
    if not rows:
        return ["no excursions recorded (trades from before they were recorded)"]
    kept = [
        100 * (t.exit_price / t.entry_price - 1) / t.mfe_pct
        for t in rows
        if t.mfe_pct is not None and t.mfe_pct > 0
    ]
    mae = statistics.median(t.mae_pct or 0.0 for t in rows)
    mfe = statistics.median(t.mfe_pct or 0.0 for t in rows)
    return [
        "Lowest (MAE) and highest (MFE) price while held, in % of the entry fill; the exit "
        "bar counts only its fill. Kept: realized % over MFE %, for legs with MFE > 0.",
        "",
        f"- median MAE: {mae:.2f}% (n={len(rows)})",
        f"- median MFE: {mfe:.2f}% (n={len(rows)})",
        f"- median kept: {statistics.median(kept):.2f} (n={len(kept)})"
        if kept
        else "- median kept: n/a (n=0)",
    ]


def build_report(
    session: Session,
    run_id: int,
    *,
    benchmark: tuple[str, float] | None = None,
    protocol: EvaluationProtocol | None = None,
) -> str:
    """``benchmark``: (label, return over the same sessions), e.g. SPY buy-and-hold."""
    run = _run(session, run_id)
    protocol = protocol or load_protocol()
    m = replay_metrics(session, run_id)
    trades = _trades(session, run_id)
    holding = m.holding_sessions
    legs = ", ".join(f"leg {k}: {v}" for k, v in sorted(m.trades_by_leg.items())) or "0"
    mean_r = ", ".join(f"leg {k}: {v:+.3f}" for k, v in m.mean_r_by_leg.items()) or "n/a"
    pnl_legs = ", ".join(f"leg {k}: {v:,.2f}" for k, v in m.mean_pnl_by_leg.items()) or "n/a"
    bench = f"{benchmark[0]} {_pct(benchmark[1])}" if benchmark else "not computed"
    lines = [
        f"# Replay {run_id}: {run.strategy_id}",
        "",
        f"Run as of {run.as_of.astimezone(ET):%Y-%m-%d %H:%M} New York, "
        f"strategy version {run.strategy_version_id}, "
        f"code {run.git_sha[:12]}.",
        "",
        *_holdout_lines(session, run, protocol),
        "## Summary",
        "",
        "| metric | value |",
        "|---|---|",
        f"| decisions | {m.decisions} (still open at the end: {m.open_decisions}; "
        f"never filled: {sum(m.cancelled.values())}) |",
        f"| trades by leg | {legs} |",
        f"| win rate (per decision with a fill, n={m.traded_decisions}) | {_pct(m.win_rate)} |",
        f"| mean R by leg | {mean_r} |",
        f"| mean pnl per trade | {m.mean_pnl_per_trade:,.2f} (by leg: {pnl_legs}) |",
        f"| total return (closed trades only) | {_pct(m.total_return)} |",
        f"| benchmark | {bench} |",
        f"| max drawdown (realized) | {_pct(m.max_drawdown)} |",
        f"| top 3 / top 10 decisions' share of profit | {_pct(m.top3_share)} / "
        f"{_pct(m.top10_share)} |",
        f"| attempts: parameter sets tried / replay runs | {m.attempt_count} / {m.replay_count} |",
        "| permutation p-value | not computed in v1 |",
        "",
        "## Exit reasons",
        "",
        *[f"- {reason}: {n}" for reason, n in sorted(m.exit_reasons.items())],
        "",
        "## Holding sessions",
        "",
    ]
    if holding:
        q = statistics.quantiles(holding, n=4) if len(holding) > 1 else [holding[0]] * 3
        lines.append(
            f"min {min(holding)}, quartiles {q[0]:.0f} / {q[1]:.0f} / {q[2]:.0f}, "
            f"max {max(holding)}"
        )
    else:
        lines.append("no closed trades")
    lines += ["", "## Excursions (MAE / MFE)", "", *_excursion_lines(trades)]
    lines += ["", f"## Largest decisions by absolute pnl (top {TOP_N})", ""]
    lines += _top_decisions(session, run_id, trades)
    lines += [
        "",
        "## Known limits",
        "",
        "- Universe: companies on today's watchlist, picked with hindsight.",
        "- Daily bars only: within a bar the worst order is assumed; exits decided at a close "
        "fill at the next open.",
        "- Drawdown counts realized pnl only; open positions are not marked to market.",
        "- No borrow, commission or market impact beyond the fixed slippage.",
        "- R is per share, from the entry to the leg's initial hard stop; size comes from the "
        "decision-day close and is capped, so 1R is not exactly the budgeted share of equity.",
        "- When positions fill the gross cap, later candidates of the day are skipped in "
        "security-id order, which has nothing to do with their merit.",
        "- ATR is fixed at the decision; after a large move the stop distance is stale.",
        "- A data-gap exit fills at the last close before the gap, which flatters halts that "
        "reopen lower.",
        "- A rename the vendor records late leaves no bars under the new symbol: the position "
        "exits as data_gap and an order for the next open is dropped.",
        "- Unrecorded splits: an upward jump exits at the last close (price_jump); a downward "
        "one goes through the stops, booking a loss even if it was a split.",
    ]
    return "\n".join(lines) + "\n"
