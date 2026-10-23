"""An llm replay set beside the take-all replay of the same window: what the judge changed.

Decisions are matched by (entity_id, as_of): both runs scan the same calendar and stamp each
decision at that evening's close. A decision's outcome in a run is the sum of its legs' pnl
there, its R the sum of their r_multiple; a decision without trades has none. Each llm verdict
is scored by the take-all outcome of the same candidate, so a pass shows what passing gave up
or saved, and calibration bins score convictions the same way. Every figure is also given before
the model's training cutoff (possibly contaminated: the model may know how the stock moved) and
on or after it. A decision in only one run (a security one run still held) is listed, never
dropped.
"""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from sqlmodel import Session, col, select

from asof.store.models import Decision, Run, Trade

VERDICTS = ("take", "pass", "no_call")
BINS = ((0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.0))


@dataclass(frozen=True)
class RunSummary:
    decisions: int
    traded: int
    total_pnl: float
    # Mean over traded decisions of the sum of their legs' R.
    mean_r: float | None


@dataclass(frozen=True)
class VerdictSummary:
    count: int
    # How many of these candidates the base run traded, and how that went.
    base_outcomes: int
    base_pnl: float
    base_mean_r: float | None


@dataclass(frozen=True)
class CalibrationBin:
    low: float
    high: float
    count: int
    base_outcomes: int
    # Share of base_outcomes with base pnl > 0.
    base_win_rate: float | None


@dataclass(frozen=True)
class Section:
    base: RunSummary
    run: RunSummary
    verdicts: dict[str, VerdictSummary]
    calibration: list[CalibrationBin]


@dataclass(frozen=True)
class Unmatched:
    run_id: int
    entity_id: int | None
    as_of: datetime


@dataclass(frozen=True)
class Comparison:
    base_run_id: int
    run_id: int
    cutoff: datetime
    all: Section
    before_cutoff: Section
    after_cutoff: Section
    unmatched: list[Unmatched]


@dataclass(frozen=True)
class _Row:
    decision: Decision
    # (pnl, R) summed over the decision's legs in its own run; None without trades.
    outcome: tuple[float, float] | None


Key = tuple[int | None, datetime]


def _key(decision: Decision) -> Key:
    return decision.entity_id, decision.as_of


def _rows(session: Session, run_id: int) -> list[_Row]:
    decisions = session.exec(
        select(Decision).where(Decision.run_id == run_id).order_by(col(Decision.id))
    ).all()
    legs: dict[int, list[Trade]] = defaultdict(list)
    for trade in session.exec(select(Trade).where(Trade.run_id == run_id)).all():
        legs[trade.decision_id].append(trade)
    rows = []
    for decision in decisions:
        trades = legs.get(decision.id or -1, [])
        outcome = (
            (sum(t.pnl for t in trades), sum(t.r_multiple for t in trades)) if trades else None
        )
        rows.append(_Row(decision, outcome))
    return rows


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _summary(rows: Sequence[_Row]) -> RunSummary:
    outcomes = [r.outcome for r in rows if r.outcome is not None]
    return RunSummary(
        decisions=len(rows),
        traded=len(outcomes),
        total_pnl=sum(o[0] for o in outcomes),
        mean_r=_mean([o[1] for o in outcomes]),
    )


def _bin_of(conviction: float) -> int:
    for index, (low, high) in enumerate(BINS):
        if low <= conviction < high or (index == len(BINS) - 1 and conviction == high):
            return index
    raise ValueError(f"conviction {conviction} outside [0, 1]")


def _section(base: Sequence[_Row], run: Sequence[_Row], base_by_key: dict[Key, _Row]) -> Section:
    scored: dict[str, list[tuple[float, float]]] = {v: [] for v in VERDICTS}
    counts = dict.fromkeys(VERDICTS, 0)
    bin_counts = [0] * len(BINS)
    bin_outcomes: list[list[float]] = [[] for _ in BINS]
    for row in run:
        verdict = str(row.decision.payload.get("verdict"))
        match = base_by_key.get(_key(row.decision))
        base_outcome = match.outcome if match is not None else None
        if verdict in counts:
            counts[verdict] += 1
            if base_outcome is not None:
                scored[verdict].append(base_outcome)
        if row.decision.conviction is not None:
            index = _bin_of(row.decision.conviction)
            bin_counts[index] += 1
            if base_outcome is not None:
                bin_outcomes[index].append(base_outcome[0])
    verdicts = {
        v: VerdictSummary(
            count=counts[v],
            base_outcomes=len(scored[v]),
            base_pnl=sum(o[0] for o in scored[v]),
            base_mean_r=_mean([o[1] for o in scored[v]]),
        )
        for v in VERDICTS
    }
    calibration = [
        CalibrationBin(
            low=low,
            high=high,
            count=bin_counts[i],
            base_outcomes=len(bin_outcomes[i]),
            base_win_rate=(
                sum(1 for pnl in bin_outcomes[i] if pnl > 0) / len(bin_outcomes[i])
                if bin_outcomes[i]
                else None
            ),
        )
        for i, (low, high) in enumerate(BINS)
    ]
    return Section(_summary(base), _summary(run), verdicts, calibration)


def _run(session: Session, run_id: int) -> Run:
    run = session.get(Run, run_id)
    if run is None:
        raise LookupError(f"no run {run_id}")
    return run


def compare_runs(session: Session, base_run_id: int, run_id: int) -> Comparison:
    base_run, llm_run = _run(session, base_run_id), _run(session, run_id)
    base = _rows(session, base_run_id)
    if base_run.model_id is not None or any(
        r.decision.payload.get("judge") != "take_all" for r in base
    ):
        raise ValueError(f"base run {base_run_id} is not a take_all replay")
    if base_run.strategy_id != llm_run.strategy_id:
        raise ValueError(
            f"runs {base_run_id} and {run_id} replay different strategies "
            f"({base_run.strategy_id!r}, {llm_run.strategy_id!r})"
        )
    if base_run.as_of != llm_run.as_of:
        raise ValueError(
            f"runs {base_run_id} and {run_id} cover different windows "
            f"(end {base_run.as_of}, {llm_run.as_of})"
        )
    cutoff = llm_run.model_training_cutoff
    if cutoff is None:
        raise ValueError(f"run {run_id} has no model_training_cutoff: not an llm replay")
    run = _rows(session, run_id)

    base_by_key = {_key(r.decision): r for r in base}
    run_keys = {_key(r.decision) for r in run}
    unmatched = [
        Unmatched(rid, r.decision.entity_id, r.decision.as_of)
        for rid, rows, other in ((base_run_id, base, run_keys), (run_id, run, set(base_by_key)))
        for r in rows
        if _key(r.decision) not in other
    ]
    unmatched.sort(key=lambda u: (u.as_of, u.run_id))

    def split(rows: Sequence[_Row], before: bool) -> list[_Row]:
        return [r for r in rows if (r.decision.as_of < cutoff) == before]

    return Comparison(
        base_run_id=base_run_id,
        run_id=run_id,
        cutoff=cutoff,
        all=_section(base, run, base_by_key),
        before_cutoff=_section(split(base, True), split(run, True), base_by_key),
        after_cutoff=_section(split(base, False), split(run, False), base_by_key),
        unmatched=unmatched,
    )
