"""The replay loop: for each session d, at as_of = d 20:00 New York --

1. every open decision gets d's bar of its security (or counts a missing one);
2. closed legs become Trade rows, finished decisions whose re-entry window is over get their
   Attribution;
3. the scanner looks for reaction-day candidates, the judge turns each into a Decision, and leg
   1 of each take is ordered for the next open. The take-all judge takes every candidate; the
   llm judge reads its answers from the offline cache (``<root>/llm_cache``), never the network,
   and its takes enter by conviction, highest first.

Bars reach the simulator only through ``BarCache.bars_at(as_of)``; the scanner reads through
``visible_bars`` itself. A replay ending on day E writes exactly what was knowable at E 20:00:
legs still open and decisions whose window is still open are left out.
"""

import subprocess
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import duckdb
from sqlmodel import Session

from asof.attribute import attribute_decision
from asof.evolve.holdout import check_window, record_opening
from asof.evolve.protocol import EvaluationProtocol, load_protocol
from asof.execute import (
    DailyBar,
    DecisionPlan,
    DecisionSim,
    LegResult,
    position_shares,
    sell_price,
)
from asof.judge.cache import CacheMiss, LLMCache
from asof.judge.llm import JudgeOutcome, judge_candidate, one_per_security, order_for_entry
from asof.judge.take_all import judge
from asof.replay.cache import BarCache
from asof.scan.earnings_gap import Candidate, scan
from asof.store.market import visible_master, visible_sessions, visible_splits
from asof.store.models import (
    Decision,
    DecisionCitation,
    DecisionKind,
    Direction,
    EvidenceKind,
    Run,
    RunMode,
    Trade,
)
from asof.store.universe import filer_securities
from asof.strategy import StrategyConfig, apply_overrides, load_strategy, register_version

ET = ZoneInfo("America/New_York")
_OPEN_FILLS = {"break", "observe_timeout", "max_sessions", "hard_gap"}


def close_of(day: date) -> datetime:
    """20:00 New York: when a session's daily bar is final."""
    return datetime.combine(day, time(20), tzinfo=ET)


def _fill_time(day: date, reason: str) -> datetime:
    """When a fill happened, as far as a daily bar tells: exits decided at a close fill at the
    next 09:30 open; an intraday stop sometime in the session (stamped at 16:00); a data-gap
    exit is booked when it is decided, 20:00."""
    if reason in _OPEN_FILLS:
        return datetime.combine(day, time(9, 30), tzinfo=ET)
    if reason == "hard":
        return datetime.combine(day, time(16), tzinfo=ET)
    return close_of(day)


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class _Active:
    """One decision being simulated, with what the loop needs around it."""

    def __init__(
        self, decision: Decision, candidate: Candidate, sim: DecisionSim, window_end: date | None
    ) -> None:
        self.decision = decision
        self.candidate = candidate
        self.sim = sim
        self.window_end = window_end
        self.trades: list[Trade] = []
        self.last_close = candidate.gap_close
        self.last_session = candidate.reaction_session
        self.symbol = candidate.symbol
        self.symbols = {candidate.symbol}
        self.missing = 0


class _Replay:
    def __init__(
        self,
        session: Session,
        con: duckdb.DuckDBPyConnection,
        root: Path,
        cfg: StrategyConfig,
        run: Run,
        calendar: list[date],
    ) -> None:
        self.session = session
        self.con = con
        self.root = root
        self.cfg = cfg
        self.run = run
        self.calendar = calendar
        self.index = {d: i for i, d in enumerate(calendar)}
        self.realized = 0.0
        self.active: list[_Active] = []
        self.cache = LLMCache(root / "llm_cache", offline=True)

    # ---- portfolio -------------------------------------------------------------------------

    def _open_value(self) -> float:
        total = 0.0
        for a in self.active:
            if a.sim.open is not None:
                total += a.sim.open.state.shares * a.last_close
            if a.sim.order is not None:
                total += a.sim.order.shares * a.last_close
        return total

    def _unrealized(self) -> float:
        total = 0.0
        for a in self.active:
            if a.sim.open is not None:
                state = a.sim.open.state
                total += state.shares * (a.last_close - state.entry_price)
        return total

    def equity(self) -> float:
        return self.cfg.execute.account.initial_equity + self.realized + self._unrealized()

    def _sizer(self, budget: float, price: float, hard: float) -> int:
        sizing = self.cfg.execute.sizing
        return position_shares(
            budget=budget,
            price=price,
            hard=hard,
            equity=self.equity(),
            max_position_pct=sizing.max_position_pct,
            max_gross_pct=sizing.max_gross_pct,
            gross_exposure=self._open_value(),
        )

    # ---- one evening -----------------------------------------------------------------------

    def _book(self, a: _Active, leg: LegResult) -> None:
        trade = Trade(
            decision_id=a.decision.id,  # pyright: ignore[reportArgumentType]
            run_id=self.run.id,  # pyright: ignore[reportArgumentType]
            strategy_id=self.cfg.strategy.name,
            security_id=a.candidate.security_id,
            symbol=a.symbol,
            leg=leg.leg,
            direction=Direction.LONG.value,
            entry_at=datetime.combine(leg.entry_session, time(9, 30), tzinfo=ET),
            entry_price=leg.entry_price,
            shares=leg.shares,
            key_level=leg.key_level,
            hard_stop=leg.hard_stop,
            risk_per_share=leg.entry_price - leg.hard_stop,
            exit_at=_fill_time(leg.exit_session, leg.exit_reason),
            exit_price=leg.exit_price,
            exit_reason=leg.exit_reason,
            pnl=leg.pnl,
            r_multiple=leg.r_multiple,
            holding_sessions=leg.holding_sessions,
            available_at=close_of(leg.exit_session),
        )
        self.session.add(trade)
        a.trades.append(trade)
        self.realized += leg.pnl

    def _split_ratio(self, symbols: set[str], after: date, day: date, as_of: datetime) -> float:
        """Product of the splits known at ``as_of`` with an ex-date in (after, day]: a split
        whose ex-date fell on a session without a bar still applies to the next bar."""
        ratio = 1.0
        for split in visible_splits(self.con, self.root, as_of, sorted(symbols)).to_pylist():
            if after < split["ex_date"] <= day:
                ratio *= float(split["new_rate"]) / float(split["old_rate"])
        return ratio

    def step_positions(self, day: date, as_of: datetime, cache: BarCache) -> None:
        if not self.active:
            return
        bars = cache.bars_at(as_of, start=datetime.combine(day, time(0), tzinfo=ET))
        today = [row for row in bars.to_pylist() if row["session_date"] == day]
        by_security = {row["security_id"]: row for row in today}
        data = self.cfg.execute.data
        bps = self.cfg.execute.fills.slippage_bps
        for a in self.active:
            if a.sim.finished:
                continue
            row = by_security.get(a.candidate.security_id)
            if row is None:
                a.missing += 1
                a.sim.skip_session()
                if a.missing >= data.max_missing_sessions and a.sim.open is not None:
                    leg = a.sim.force_exit(day, sell_price(a.last_close, bps), "data_gap")
                    if leg is not None:
                        self._book(a, leg)
                continue
            a.missing = 0
            a.symbols.add(row["symbol"])
            a.symbol = row["symbol"]
            ratio = self._split_ratio(a.symbols, a.last_session, day, as_of)
            if ratio != 1.0:
                a.sim.apply_split(ratio)
                a.last_close /= ratio
            elif row["open"] > data.max_jump_ratio * a.last_close:
                # An unrecorded reverse split would read as a windfall: leave at the last close.
                leg = a.sim.force_exit(day, sell_price(a.last_close, bps), "price_jump")
                if leg is not None:
                    self._book(a, leg)
                a.last_session = day
                continue
            bar = DailyBar(
                session_date=day,
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
            )
            for leg in a.sim.step(bar):
                self._book(a, leg)
            a.last_close = row["close"]
            a.last_session = day

    def attribute_finished(self, day: date) -> None:
        still: list[_Active] = []
        for a in self.active:
            if a.sim.finished and not a.trades:
                # Never filled: say why, so the report does not count it as still open.
                reason = a.sim.cancelled or "no_entry"
                a.decision.payload = {**a.decision.payload, "cancelled": reason}
                continue
            if a.sim.finished and a.window_end is not None and day >= a.window_end:
                if a.trades:
                    row = attribute_decision(
                        a.decision,
                        a.trades,
                        equity_at_decision=a.sim.plan.equity,
                        window_end=a.window_end,
                    )
                    self.session.add(row)
                continue
            still.append(a)
        self.active = still

    def _offline(self, body: dict[str, Any]) -> dict[str, Any]:
        def never(_: dict[str, Any]) -> dict[str, Any]:
            raise CacheMiss("an offline replay may not call the model")

        return self.cache.call(body, never)

    def _record(
        self,
        candidate: Candidate,
        as_of: datetime,
        direction: Direction,
        conviction: float | None,
        extra: dict[str, Any],
    ) -> Decision:
        payload = {**_payload(candidate, self.cfg, self.equity()), **extra}
        decision = Decision(
            run_id=self.run.id,  # pyright: ignore[reportArgumentType]
            kind=DecisionKind.TRADE.value,
            as_of=as_of,
            strategy_id=self.cfg.strategy.name,
            entity_id=candidate.entity_id,
            direction=direction.value,
            conviction=conviction,
            payload=payload,
        )
        self.session.add(decision)
        self.session.flush()
        return decision

    def _judged(self, outcome: JudgeOutcome, as_of: datetime) -> Decision:
        """The decision for one llm outcome, with a citation row per validated claim."""
        long = outcome.verdict == "take"
        extra = {
            "verdict": outcome.verdict,
            "attempts": outcome.attempts,
            "request_keys": outcome.request_keys,
            "failures": outcome.failures,
            "stances": [c.stance for c in outcome.claims],
        }
        direction = Direction.LONG if long else Direction.FLAT
        decision = self._record(outcome.candidate, as_of, direction, outcome.conviction, extra)
        for claim in outcome.claims:
            self.session.add(
                DecisionCitation(
                    decision_id=decision.id,  # pyright: ignore[reportArgumentType]
                    claim=claim.claim,
                    evidence_kind=EvidenceKind.CHUNK.value,
                    chunk_id=claim.chunk_id,
                    quote=claim.quote,
                )
            )
        return decision

    def _enter(self, day: date, candidate: Candidate, decision: Decision) -> None:
        plan = DecisionPlan(
            reaction_session=candidate.reaction_session,
            decision_close=candidate.gap_close,
            gap_low=candidate.gap_low,
            pre_earnings_close=candidate.pre_earnings_close,
            atr=candidate.atr,
            equity=self.equity(),
        )
        sim = DecisionSim(plan, self.cfg.execute, self._sizer)
        window_index = self.index[day] + self.cfg.execute.reentry.window_sessions
        window_end = self.calendar[window_index] if window_index < len(self.calendar) else None
        self.active.append(_Active(decision, candidate, sim, window_end))

    def decide(self, day: date, as_of: datetime) -> None:
        sessions = self.calendar[: self.index[day] + 1]
        candidates = scan(self.session, self.con, self.root, as_of, self.cfg, sessions=sessions)
        held = {a.candidate.security_id for a in self.active}
        if self.cfg.judge.kind == "llm":
            fresh = one_per_security([c for c in candidates if c.security_id not in held])
            outcomes = [
                judge_candidate(self.session, c, as_of, self.cfg.judge, self._offline)
                for c in fresh
            ]
            decisions = {id(o): self._judged(o, as_of) for o in outcomes}
            for outcome in order_for_entry(outcomes):
                self._enter(day, outcome.candidate, decisions[id(outcome)])
            return
        for candidate in judge(candidates, self.cfg):
            if candidate.security_id in held:
                continue  # one live decision per security
            decision = self._record(candidate, as_of, Direction.LONG, None, {})
            self._enter(day, candidate, decision)
            held.add(candidate.security_id)


def _payload(candidate: Candidate, cfg: StrategyConfig, equity: float) -> dict[str, Any]:
    fields = asdict(candidate)
    for key, value in fields.items():
        if isinstance(value, datetime | date):
            fields[key] = value.isoformat()
        elif isinstance(value, tuple):
            fields[key] = list(value)
    return {"judge": cfg.judge.kind, "equity_at_decision": equity, "candidate": fields}


def _universe_symbols(
    session: Session, con: duckdb.DuckDBPyConnection, root: Path, as_of: datetime
) -> list[str]:
    """Every symbol of every security whose company filed an 8-K by ``as_of``: what the
    simulator may need. Only decides what the cache loads, never what a step may see."""
    securities = filer_securities(session, con, root, as_of)
    master = visible_master(con, root, as_of).to_pylist()
    return sorted({r["symbol"] for r in master if r["security_id"] in securities})


def run_replay(
    session: Session,
    con: duckdb.DuckDBPyConnection,
    root: Path,
    strategy_dir: Path,
    start: date,
    end: date,
    overrides: Sequence[str] = (),
    *,
    protocol: EvaluationProtocol | None = None,
    open_holdout: bool = False,
) -> int:
    """Replay ``strategy_dir`` (with ``overrides``) over the sessions in [start, end]; returns
    the Run id. Every parameter set is registered as a StrategyVersion: one more attempt.
    A window reaching the sealed holdout is refused before anything is registered or read."""
    if end < start:
        raise ValueError(f"end {end} is before start {start}")
    cfg = apply_overrides(load_strategy(strategy_dir), overrides)
    opens = check_window(
        session,
        protocol or load_protocol(),
        cfg.strategy.name,
        start,
        end,
        open_holdout=open_holdout,
    )
    version = register_version(session, strategy_dir, overrides)
    end_as_of = close_of(end)
    calendar = visible_sessions(con, root, "1Day", end_as_of)
    steps = [d for d in calendar if start <= d <= end]
    if not steps:
        raise ValueError(f"no sessions between {start} and {end}")

    run = Run(
        strategy_id=cfg.strategy.name,
        strategy_version_id=version.id,
        mode=RunMode.REPLAY.value,
        as_of=end_as_of,
        started_at=datetime.now(UTC),
        git_sha=_git_sha(),
    )
    if cfg.judge.kind == "llm":
        assert cfg.judge.training_cutoff is not None
        run.model_id = cfg.judge.model
        run.model_training_cutoff = datetime.combine(cfg.judge.training_cutoff, time(0), tzinfo=UTC)
    session.add(run)
    session.flush()
    if opens:
        # Committed with the run below: a replay that fails shows nothing and opens nothing.
        record_opening(session, cfg.strategy.name, version.id, start, end, run_id=run.id)  # pyright: ignore[reportArgumentType]

    cache = BarCache(
        con,
        root,
        symbols=_universe_symbols(session, con, root, end_as_of),
        start=datetime.combine(steps[0], time(0), tzinfo=ET),
        end=datetime.combine(steps[-1] + timedelta(days=1), time(0), tzinfo=ET),
    )
    replay = _Replay(session, con, root, cfg, run, calendar)
    for day in steps:
        as_of = close_of(day)
        replay.step_positions(day, as_of, cache)
        replay.attribute_finished(day)
        replay.decide(day, as_of)
    run.finished_at = datetime.now(UTC)
    session.commit()
    return run.id  # pyright: ignore[reportReturnType]
