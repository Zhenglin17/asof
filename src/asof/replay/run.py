"""The replay loop: for each session d, at as_of = d 20:00 New York --

1. every open decision gets d's bar of its security (or counts a missing one);
2. closed legs become Trade rows, finished decisions whose re-entry window is over get their
   Attribution;
3. the scanner looks for reaction-day candidates, the judge turns each into a Decision, and leg
   1 of each take is ordered for the next open. The take-all judge takes every candidate; the
   llm judge reads its answers from the offline cache (``<root>/llm_cache``), never the network,
   and its takes enter by conviction, highest first.

Bars reach the simulator only through ``BarCache.bars_at(as_of)``; the scanner reads through
``visible_bars`` itself. Spin-offs are read once as known at the window's end and used only as
far as they were known each day: a record on file before the ex-date's open freezes the
decision, an event decides it from its ``known_on`` close, and a child is sold at its first
open after that, read with the day's as_of. A replay ending on day E writes exactly what was
knowable at E 20:00: legs still open and decisions whose window is still open are left out.
"""

import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd
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
    spinoff_proceeds,
)
from asof.execute.bx import BxPlan, BxPositionSim, ExitFlags, bx_shares, exit_flags
from asof.judge.cache import CacheMiss, LLMCache
from asof.judge.llm import JudgeOutcome, judge_candidate, one_per_security, order_for_entry
from asof.judge.take_all import judge
from asof.replay.bx import signal_inputs
from asof.replay.cache import BarCache
from asof.scan.bx_trender import BxCandidate, scan_nights
from asof.scan.earnings_gap import Candidate, scan
from asof.store.market import visible_bars, visible_master, visible_sessions, visible_splits
from asof.store.models import (
    Decision,
    DecisionCitation,
    DecisionKind,
    Direction,
    EquityDaily,
    EvidenceKind,
    Run,
    RunMode,
    Trade,
)
from asof.store.spinoffs import spinoff_events, visible_spinoffs
from asof.store.universe import filer_securities
from asof.strategy import (
    BxStrategyConfig,
    StrategyConfig,
    apply_overrides,
    load_any_strategy,
    register_version,
)

ET = ZoneInfo("America/New_York")
_NO_EXIT = ExitFlags(daily_red=False, weekly_down=False)
_OPEN_FILLS = {"break", "observe_timeout", "max_sessions", "hard_gap", "daily_red", "weekly_down"}


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


@dataclass(frozen=True)
class _SpinEvent:
    """One spin-off of a security as known at the window's end; ``first_known`` is when its
    first record reached the store, ``known_on`` the day its status became decidable."""

    ex_date: date
    first_known: datetime
    status: str
    factor: float | None
    known_on: date | None
    children: tuple[tuple[str, float], ...]  # (child symbol, child shares per parent share)


@dataclass
class _Spin:
    """A decision between a spin-off's ex-date and the sale of its children."""

    event: _SpinEvent
    first_session: int  # calendar index of the ex-date's session (wait session 0)
    resolved: bool = False
    sale: dict[str, float] = field(default_factory=dict)  # child symbol -> sale price
    missing: dict[str, int] = field(default_factory=dict)
    selling: bool = False


class _Active:
    """One decision being simulated, with what the loop needs around it. ``window_end`` None:
    attributed once its sim is finished and the window is past (a later session than the
    calendar has means never)."""

    def __init__(
        self,
        decision: Decision,
        *,
        security_id: str,
        symbol: str,
        close: float,
        session: date,
        sim: Any,
        window_end: date | None,
    ) -> None:
        self.decision = decision
        self.security_id = security_id
        self.sim = sim  # DecisionSim or BxPositionSim
        self.window_end = window_end
        self.trades: list[Trade] = []
        self.last_close = close
        self.last_session = session
        self.symbol = symbol
        self.symbols = {symbol}
        self.missing = 0
        self.spin: _Spin | None = None


def _legs(result: LegResult | list[LegResult] | None) -> list[LegResult]:
    """``force_exit`` of either simulator as a list."""
    if result is None:
        return []
    return result if isinstance(result, list) else [result]


class _Replay:
    def __init__(
        self,
        session: Session,
        con: duckdb.DuckDBPyConnection,
        root: Path,
        cfg: StrategyConfig,
        run: Run,
        calendar: list[date],
        spinoffs: dict[str, list[_SpinEvent]] | None = None,
    ) -> None:
        self.spinoffs = spinoffs or {}
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
        return sum(a.sim.exposure(a.last_close) for a in self.active)

    def _unrealized(self) -> float:
        # a leg sold before its children were paid counts at its sale price until booked
        return sum(a.sim.unrealized(a.last_close) for a in self.active)

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

    def mark(self, day: date) -> None:
        """The account at ``day``'s close; a position without a bar today keeps its last close."""
        held = [a.sim.held_shares * a.last_close for a in self.active if a.sim.held_shares > 0]
        self.session.add(
            EquityDaily(
                run_id=self.run.id,  # pyright: ignore[reportArgumentType]
                session_date=day,
                equity=self.equity(),
                realized=self.realized,
                unrealized=self._unrealized(),
                gross_exposure=sum(held),
                open_positions=len(held),
                available_at=close_of(day),
            )
        )

    # ---- one evening -----------------------------------------------------------------------

    def _book(self, a: _Active, leg: LegResult, known: date | None = None) -> None:
        """``known``: the evening the result was complete, if later than its exit (a leg paid
        its spin-off children after its shares were sold)."""
        trade = Trade(
            decision_id=a.decision.id,  # pyright: ignore[reportArgumentType]
            run_id=self.run.id,  # pyright: ignore[reportArgumentType]
            strategy_id=self.cfg.strategy.name,
            security_id=a.security_id,
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
            mae_pct=leg.mae_pct,
            mfe_pct=leg.mfe_pct,
            child_proceeds=leg.child_proceeds,
            spinoff=leg.spinoff,
            available_at=close_of(max(leg.exit_session, known or leg.exit_session)),
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
            row = by_security.get(a.security_id)
            if row is None:
                a.missing += 1
                a.sim.skip_session()
                if a.missing >= data.max_missing_sessions and a.sim.held_shares > 0:
                    price = sell_price(a.last_close, bps)
                    for leg in _legs(a.sim.force_exit(day, price, "data_gap")):
                        self._book(a, leg)
                continue
            a.missing = 0
            a.symbols.add(row["symbol"])
            a.symbol = row["symbol"]
            self._begin_spinoff(a, day)
            ratio = self._split_ratio(a.symbols, a.last_session, day, as_of)
            if ratio != 1.0:
                a.sim.apply_split(ratio)
                a.last_close /= ratio
            elif row["open"] > data.max_jump_ratio * a.last_close:
                # An unrecorded reverse split would read as a windfall: leave at the last close.
                price = sell_price(a.last_close, bps)
                for leg in _legs(a.sim.force_exit(day, price, "price_jump")):
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
            for leg in self._step_sim(a, bar, day):
                self._book(a, leg)
            a.last_close = row["close"]
            a.last_session = day
        for a in self.active:
            if a.spin is not None:
                self._step_spinoff(a, day, as_of)

    def _step_sim(self, a: _Active, bar: DailyBar, day: date) -> list[LegResult]:
        return a.sim.step(bar)

    # ---- spin-offs -------------------------------------------------------------------------

    def _begin_spinoff(self, a: _Active, day: date) -> None:
        """Every spin-off with its ex-date since the last bar, on file before today's open. One
        is settled at a time: another going ex meanwhile is not priced (``overlap_spinoff``)."""
        before_open = datetime.combine(day, time(9, 30), tzinfo=ET)
        for event in self.spinoffs.get(a.security_id, ()):
            if not (a.last_session < event.ex_date <= day and event.first_known <= before_open):
                continue
            if a.spin is not None:
                a.sim.overlap_spinoff()
                continue
            first = next(i for i, d in enumerate(self.calendar) if d >= event.ex_date)
            a.spin = _Spin(event=event, first_session=first)
            a.sim.begin_spinoff()

    def _step_spinoff(self, a: _Active, day: date, as_of: datetime) -> None:
        spin = a.spin
        assert spin is not None
        event = spin.event
        if spin.selling:
            self._sell_children(a, day, as_of)
        elif not spin.resolved:
            known = event.known_on is not None and event.known_on <= day
            if known and event.status == "adjust":
                assert event.factor is not None
                spin.resolved = True
                a.sim.apply_spinoff(event.factor)
                owed = a.sim.owes_children()
                spin.selling = owed
                if not owed:
                    a.spin = None
            elif (known and event.status == "restart") or (
                self.index[day] - spin.first_session >= self.cfg.execute.data.spinoff_max_wait
            ):
                for leg in a.sim.end_spinoff_unpriced():
                    self._book(a, leg, day)
                a.spin = None

    def _sell_children(self, a: _Active, day: date, as_of: datetime) -> None:
        """Each child at its first open after the event's known_on; one without a bar for
        ``max_missing_sessions`` sessions at its last close visible now."""
        spin = a.spin
        assert spin is not None and spin.event.known_on is not None
        if day <= spin.event.known_on:
            return
        left = [c for c, _ in spin.event.children if c not in spin.sale]
        bars = visible_bars(
            self.con,
            self.root,
            "1Day",
            as_of,
            symbols=left,
            start=datetime.combine(spin.event.ex_date, time(0), tzinfo=ET),
        ).to_pylist()
        for child in left:
            rows = sorted(
                (r for r in bars if r["symbol"] == child), key=lambda r: r["session_date"]
            )
            today = [r for r in rows if r["session_date"] == day]
            if today:
                spin.sale[child] = today[0]["open"]
                continue
            spin.missing[child] = spin.missing.get(child, 0) + 1
            if spin.missing[child] >= self.cfg.execute.data.max_missing_sessions:
                if not rows:
                    # no child bar readable at all: nothing to sell it at, the event is unpriced
                    for leg in a.sim.end_spinoff_unpriced():
                        self._book(a, leg, day)
                    a.spin = None
                    return
                spin.sale[child] = rows[-1]["close"]
        if len(spin.sale) < len(spin.event.children):
            return
        proceeds = spinoff_proceeds(
            a.sim.entitled_shares(),
            [(ratio, spin.sale[c]) for c, ratio in spin.event.children],
            self.cfg.execute.fills.slippage_bps,
        )
        for leg in a.sim.credit_children(proceeds):
            self._book(a, leg, day)
        a.spin = None

    def _window_end(self, a: _Active, day: date) -> date | None:
        """The last session another leg of ``a`` could start: the re-entry window."""
        return a.window_end

    def attribute_finished(self, day: date) -> None:
        still: list[_Active] = []
        for a in self.active:
            if a.sim.finished and not a.trades:
                # Never filled: say why, so the report does not count it as still open.
                reason = a.sim.cancelled or "no_entry"
                a.decision.payload = {**a.decision.payload, "cancelled": reason}
                continue
            window_end = self._window_end(a, day)
            if a.sim.finished and window_end is not None and day >= window_end:
                if a.trades:
                    row = attribute_decision(
                        a.decision,
                        a.trades,
                        equity_at_decision=a.sim.plan.equity,
                        window_end=window_end,
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
        self.active.append(
            _Active(
                decision,
                security_id=candidate.security_id,
                symbol=candidate.symbol,
                close=candidate.gap_close,
                session=candidate.reaction_session,
                sim=sim,
                window_end=window_end,
            )
        )

    def decide(self, day: date, as_of: datetime) -> None:
        sessions = self.calendar[: self.index[day] + 1]
        candidates = scan(self.session, self.con, self.root, as_of, self.cfg, sessions=sessions)
        held = {a.security_id for a in self.active}
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


class _BxReplay(_Replay):
    """The B-Xtrender loop: the scan's nights are read once for the window; each night held
    positions sell on that night's exit flags, then the night's candidates are bought in rank
    order until the gross cap leaves no room for a share. One live position per security; a
    position is attributed the night its last share is sold."""

    cfg: BxStrategyConfig  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        session: Session,
        con: duckdb.DuckDBPyConnection,
        root: Path,
        cfg: BxStrategyConfig,
        run: Run,
        calendar: list[date],
        spinoffs: dict[str, list[_SpinEvent]],
        nights: dict[date, tuple[list[BxCandidate], pd.DataFrame]],
        cache: BarCache,
    ) -> None:
        super().__init__(session, con, root, cfg, run, calendar, spinoffs)  # pyright: ignore[reportArgumentType]
        self.nights = nights
        self.bars = cache
        self._flags: dict[str, ExitFlags] = {}
        self._flags_day: date | None = None

    def _step_sim(self, a: _Active, bar: DailyBar, day: date) -> list[LegResult]:
        if self._flags_day != day:
            rows = self.nights[day][1] if day in self.nights else pd.DataFrame()
            self._flags = {str(r["security_id"]): exit_flags(r) for r in rows.to_dict("records")}
            self._flags_day = day
        return a.sim.step(bar, self._flags.get(a.security_id, _NO_EXIT))

    def _window_end(self, a: _Active, day: date) -> date | None:
        return day  # no re-entry window: a later signal is a new decision

    def decide(self, day: date, as_of: datetime) -> None:
        candidates = self.nights.get(day, ([], None))[0]
        held = {a.security_id for a in self.active}
        fresh = [c for c in candidates if c.security_id not in held]
        if not fresh:
            return
        sizing = self.cfg.execute.sizing
        lookback = self.calendar[max(0, self.index[day] - 2 * sizing.atr_days)]
        bars = self.bars.bars_at(as_of, start=datetime.combine(lookback, time(0), tzinfo=ET))
        frame = bars.to_pandas()
        wanted = {c.security_id for c in fresh}
        frame = pd.DataFrame(frame[frame["security_id"].isin(wanted)])
        symbols = sorted(set(frame["symbol"]))
        splits = visible_splits(self.con, self.root, as_of, symbols).to_pandas()
        inputs = signal_inputs(frame, splits, day, sizing.atr_days)
        for c in fresh:
            if c.security_id not in inputs:
                continue
            close, atr = inputs[c.security_id]
            if not atr > 0:
                continue  # too little history to size it: not a full book
            equity = self.equity()
            if bx_shares(equity=equity, close=close, atr=atr, sizing=sizing, gross_exposure=0) == 0:
                # not one share even with an empty book (BRK.A at $554,300 against an $80,000
                # gross cap): this candidate, not the book, is the reason
                continue
            shares = bx_shares(
                equity=equity,
                close=close,
                atr=atr,
                sizing=sizing,
                gross_exposure=self._open_value(),
            )
            if shares == 0:
                return  # the gross cap is reached: the rest of the night waits
            decision = Decision(
                run_id=self.run.id,  # pyright: ignore[reportArgumentType]
                kind=DecisionKind.TRADE.value,
                as_of=as_of,
                strategy_id=self.cfg.strategy.name,
                security_id=c.security_id,
                symbol=c.symbol,
                direction=Direction.LONG.value,
                payload={
                    "equity_at_decision": equity,
                    "close": close,
                    "atr": atr,
                    "shares": shares,
                    "candidate": _jsonable(asdict(c)),
                },
            )
            self.session.add(decision)
            self.session.flush()
            plan = BxPlan(
                signal_session=day, signal_close=close, atr=atr, shares=shares, equity=equity
            )
            self.active.append(
                _Active(
                    decision,
                    security_id=c.security_id,
                    symbol=c.symbol,
                    close=close,
                    session=day,
                    sim=BxPositionSim(plan, self.cfg.execute),
                    window_end=None,
                )
            )


def _jsonable(fields: dict[str, Any]) -> dict[str, Any]:
    for key, value in fields.items():
        if isinstance(value, datetime | date):
            fields[key] = value.isoformat()
        elif isinstance(value, tuple):
            fields[key] = list(value)
    return fields


def _payload(candidate: Candidate, cfg: StrategyConfig, equity: float) -> dict[str, Any]:
    fields = _jsonable(asdict(candidate))
    return {"judge": cfg.judge.kind, "equity_at_decision": equity, "candidate": fields}


def _universe_symbols(
    session: Session, con: duckdb.DuckDBPyConnection, root: Path, as_of: datetime
) -> list[str]:
    """Every symbol of every security whose company filed an 8-K by ``as_of``: what the
    simulator may need. Only decides what the cache loads, never what a step may see."""
    securities = filer_securities(session, con, root, as_of)
    master = visible_master(con, root, as_of).to_pylist()
    return sorted({r["symbol"] for r in master if r["security_id"] in securities})


def _symbols_of(
    con: duckdb.DuckDBPyConnection, root: Path, as_of: datetime, securities: set[str]
) -> list[str]:
    """Every symbol these securities carried, as the master knows at ``as_of``: what the cache
    loads, never what a step may see."""
    master = visible_master(con, root, as_of).to_pylist()
    return sorted({r["symbol"] for r in master if r["security_id"] in securities})


def _spinoffs(
    con: duckdb.DuckDBPyConnection, root: Path, as_of: datetime
) -> dict[str, list[_SpinEvent]]:
    """Every spin-off of a known parent security as known at ``as_of``, by security. An event's
    first record is matched under every symbol its security carried."""
    events = [e for e in spinoff_events(con, root, as_of).to_pylist() if e["security_id"]]
    if not events:
        return {}
    names: dict[str, set[str]] = {}
    for seg in visible_master(con, root, as_of).to_pylist():
        names.setdefault(seg["security_id"], set()).add(seg["symbol"])
    records = visible_spinoffs(con, root, as_of).to_pylist()
    out: dict[str, list[_SpinEvent]] = {}
    for e in events:
        sid = e["security_id"]
        symbols = names.get(sid, set()) | {e["parent_symbol"]}
        known = [
            r["available_at"]
            for r in records
            if r["ex_date"] == e["ex_date"] and r["parent_symbol"] in symbols
        ]
        out.setdefault(sid, []).append(
            _SpinEvent(
                ex_date=e["ex_date"],
                first_known=min(known),
                status=e["status"],
                factor=e["factor"],
                known_on=e["known_on"],
                children=tuple(zip(e["child_symbols"], e["ratio"], strict=True)),
            )
        )
    for rows in out.values():
        rows.sort(key=lambda r: r.ex_date)
    return out


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
    cfg = apply_overrides(load_any_strategy(strategy_dir), overrides)
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
    if isinstance(cfg, StrategyConfig) and cfg.judge.kind == "llm":
        assert cfg.judge.training_cutoff is not None
        run.model_id = cfg.judge.model
        run.model_training_cutoff = datetime.combine(cfg.judge.training_cutoff, time(0), tzinfo=UTC)
    session.add(run)
    session.flush()
    if opens:
        # Committed with the run below: a replay that fails shows nothing and opens nothing.
        record_opening(session, cfg.strategy.name, version.id, start, end, run_id=run.id)  # pyright: ignore[reportArgumentType]

    spinoffs = _spinoffs(con, root, end_as_of)
    replay: _Replay
    if isinstance(cfg, BxStrategyConfig):
        nights = {
            d: (found, rows) for d, found, rows in scan_nights(con, root, start, end, cfg.scan)
        }
        held = {c.security_id for found, _ in nights.values() for c in found}
        lookback = calendar[max(0, calendar.index(steps[0]) - 2 * cfg.execute.sizing.atr_days)]
        cache = BarCache(
            con,
            root,
            symbols=_symbols_of(con, root, end_as_of, held),
            start=datetime.combine(lookback, time(0), tzinfo=ET),
            end=datetime.combine(steps[-1] + timedelta(days=1), time(0), tzinfo=ET),
        )
        replay = _BxReplay(session, con, root, cfg, run, calendar, spinoffs, nights, cache)
    else:
        cache = BarCache(
            con,
            root,
            symbols=_universe_symbols(session, con, root, end_as_of),
            start=datetime.combine(steps[0], time(0), tzinfo=ET),
            end=datetime.combine(steps[-1] + timedelta(days=1), time(0), tzinfo=ET),
        )
        replay = _Replay(session, con, root, cfg, run, calendar, spinoffs)
    for day in steps:
        as_of = close_of(day)
        replay.step_positions(day, as_of, cache)
        replay.mark(day)
        replay.attribute_finished(day)
        replay.decide(day, as_of)
    run.finished_at = datetime.now(UTC)
    session.commit()
    return run.id  # pyright: ignore[reportReturnType]
