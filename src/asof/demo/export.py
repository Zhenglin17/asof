"""Replay results as plain, JSON-ready dicts for the static demo page. Read-only over the store.

Every number comes from rows a real replay wrote (Trade, Decision, DecisionCitation) or from
``replay_metrics``, so the page shows what the replay report shows. The shapes are the ones an
HTTP API would return later; the page should not need to change when files become requests.

What is left out on purpose: decision payload fields other than the judge's verdict, stances and
the candidate (a payload may hold local paths or request keys), and a document's ``source_time``
(the period it describes): the page places filings by ``available_at`` only.
"""

import dataclasses
import re
from collections import defaultdict
from collections.abc import Iterable
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

from sqlmodel import Session, col, select

from asof.replay.report import replay_metrics
from asof.store.models import Chunk, Decision, DecisionCitation, Document, Run, Trade

ET = ZoneInfo("America/New_York")
DEFAULT_EQUITY = 100_000.0
_TRADE_FIELDS = (
    "leg",
    "entry_at",
    "exit_at",
    "entry_price",
    "exit_price",
    "shares",
    "pnl",
    "r_multiple",
    "exit_reason",
)


def _iso(value: Any) -> Any:
    return value.isoformat() if isinstance(value, datetime | date) else value


def equity_curve(
    trades: Iterable[Trade], equity0: float, start: date, end: date
) -> list[dict[str, Any]]:
    """Realized equity: a point at ``start``, then one per New York exit date with the P&L of
    every leg that exited that day. Positions are not marked to market in between, like the
    report's drawdown. ``end`` is the replay's last session; exits after it are not expected."""
    by_day: dict[date, float] = defaultdict(float)
    for t in trades:
        by_day[t.exit_at.astimezone(ET).date()] += t.pnl
    curve = [{"date": start.isoformat(), "equity": equity0, "drawdown": 0.0}]
    equity = peak = equity0
    for day in sorted(by_day):
        equity += by_day[day]
        peak = max(peak, equity)
        curve.append(
            {"date": day.isoformat(), "equity": equity, "drawdown": (peak - equity) / peak}
        )
    return curve


def _decisions(session: Session, run_id: int) -> list[Decision]:
    stmt = select(Decision).where(Decision.run_id == run_id)
    return sorted(session.exec(stmt).all(), key=lambda d: (d.as_of, d.id or 0))


def _trades_by_decision(session: Session, run_id: int) -> dict[int, list[Trade]]:
    stmt = select(Trade).where(Trade.run_id == run_id).order_by(col(Trade.leg))
    out: dict[int, list[Trade]] = defaultdict(list)
    for t in session.exec(stmt).all():
        out[t.decision_id].append(t)
    return out


def _judge(decisions: list[Decision]) -> str | None:
    return next((d.payload["judge"] for d in decisions if "judge" in d.payload), None)


def export_run(
    session: Session, run_id: int, params: dict[str, Any], start: date | None = None
) -> dict[str, Any]:
    """One replay: its parameters, the report's metrics and the realized equity curve. The curve
    starts at ``start`` (the replay window's first session, so it lines up with a benchmark),
    else at the first decision, else at the run's as_of."""
    run = session.get(Run, run_id)
    if run is None:
        raise LookupError(f"no run {run_id}")
    metrics = replay_metrics(session, run_id)
    decisions = _decisions(session, run_id)
    equity0 = next(
        (
            float(d.payload["equity_at_decision"])
            for d in decisions
            if "equity_at_decision" in d.payload
        ),
        DEFAULT_EQUITY,
    )
    trades = [t for legs in _trades_by_decision(session, run_id).values() for t in legs]
    if start is None:
        first = [d.as_of.astimezone(ET).date() for d in decisions]
        start = min(first, default=run.as_of.astimezone(ET).date())
    end = run.as_of.astimezone(ET).date()
    return {
        "run_id": run_id,
        "judge": _judge(decisions),
        "params": params,
        "metrics": dataclasses.asdict(metrics),
        "curve": equity_curve(trades, equity0, start, end),
    }


def _candidate_key(d: Decision) -> tuple[Any, datetime]:
    """One candidate: a security on an evening. Two share classes of one company (one entity)
    are two candidates of the same filing, so the entity alone would pair them with each other."""
    security = d.payload.get("candidate", {}).get("security_id")
    return (security if security is not None else d.entity_id, d.as_of)


def _r(legs: list[Trade]) -> float | None:
    return sum(t.r_multiple for t in legs) if legs else None


def export_decisions(
    session: Session, run_id: int, base_run_id: int | None
) -> list[dict[str, Any]]:
    """Every decision of the run with its claims and legs. ``base_r`` is what the decision on
    the same security and the same evening earned in ``base_run_id`` (take-all), so a pass can
    be shown next to what taking it would have done."""
    decisions = _decisions(session, run_id)
    trades = _trades_by_decision(session, run_id)
    base: dict[tuple[Any, datetime], float | None] = {}
    if base_run_id is not None:
        base_trades = _trades_by_decision(session, base_run_id)
        for d in _decisions(session, base_run_id):
            base[_candidate_key(d)] = _r(base_trades.get(d.id or 0, []))

    citations: dict[int, list[DecisionCitation]] = defaultdict(list)
    stmt = (
        select(DecisionCitation)
        .where(col(DecisionCitation.decision_id).in_([d.id for d in decisions]))
        .order_by(col(DecisionCitation.id))
    )
    for c in session.exec(stmt).all():
        citations[c.decision_id].append(c)

    rows = []
    for d in decisions:
        legs = trades.get(d.id or 0, [])
        stances = d.payload.get("stances", [])
        claims = [
            {
                "claim": c.claim,
                "stance": stances[i] if i < len(stances) else None,
                "chunk_id": c.chunk_id,
                "quote": c.quote,
            }
            for i, c in enumerate(citations.get(d.id or 0, []))
        ]
        candidate = d.payload.get("candidate", {})
        rows.append(
            {
                "id": d.id,
                "as_of": d.as_of.isoformat(),
                "entity_id": d.entity_id,
                "symbol": candidate.get("symbol"),
                # take_all has no verdict: every candidate is a take
                "verdict": d.payload.get("verdict", "take"),
                "conviction": d.conviction,
                "candidate": candidate,
                "claims": claims,
                "trades": [{f: _iso(getattr(t, f)) for f in _TRADE_FIELDS} for t in legs],
                "r": _r(legs),
                "base_r": base.get(_candidate_key(d)),
            }
        )
    return rows


def _span(text: str, quote: str) -> tuple[int, int] | None:
    """Where ``quote`` sits in ``text``: exactly, else with any run of whitespace in the text
    (newlines, repeated or no-break spaces) standing for one space of the quote."""
    at = text.find(quote)
    if at >= 0:
        return at, at + len(quote)
    words = quote.split()
    if not words:
        return None
    match = re.search(r"\s+".join(re.escape(w) for w in words), text)
    return match.span() if match else None


def export_chunks(
    session: Session, chunk_ids: Iterable[int], quotes: dict[int, list[str]]
) -> dict[int, dict[str, Any]]:
    """The cited chunks' text with the span of each quote, for highlighting. A quote the
    validator accepted after its own normalisation (table cells joined) may have no span."""
    ids = sorted(set(chunk_ids))
    chunks = session.exec(select(Chunk).where(col(Chunk.id).in_(ids))).all()
    out: dict[int, dict[str, Any]] = {}
    for chunk in sorted(chunks, key=lambda c: c.id or 0):
        spans = sorted(
            {s for q in quotes.get(chunk.id or 0, []) if (s := _span(chunk.text, q)) is not None}
        )
        out[chunk.id or 0] = {
            "document_id": chunk.document_id,
            "section_path": chunk.section_path,
            "text": chunk.text,
            "spans": [list(s) for s in spans],
        }
    return out


def export_timeline(
    session: Session, entity_ids: Iterable[int], start: datetime, end: datetime
) -> list[dict[str, Any]]:
    """The documents of these companies that became public in [start, end], in the order the
    public saw them."""
    ids = sorted(set(entity_ids))
    if not ids:
        return []
    stmt = (
        select(Document)
        .where(col(Document.entity_id).in_(ids))
        .where(Document.available_at >= start)
        .where(Document.available_at <= end)
        .order_by(col(Document.available_at), col(Document.id))
    )
    return [
        {
            "document_id": d.id,
            "entity_id": d.entity_id,
            "doc_type": d.doc_type,
            "title": d.title,
            "available_at": d.available_at.isoformat(),
        }
        for d in session.exec(stmt).all()
    ]
