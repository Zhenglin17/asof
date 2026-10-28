"""One decision from order to last exit: leg 1, then at most ``max_count`` re-entries.

``DecisionSim`` is a state machine fed one daily bar at a time, so the replay can step every
open decision in calendar order and never hand it a bar from the future. ``simulate_decision``
feeds a whole bar list through the same machine.

Leg 1 buys the open after the reaction day with its key at the gap-day low. Once it has exited,
a bar (the exit day included) that touches the pre-earnings-close zone and closes at or above
its top, within ``window_sessions`` sessions of the reaction day, orders leg 2 for the next open
with its key at the pre-earnings close. A close below that zone at any point since the reaction
day means the gap has filled and ends the setup.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date

from asof.execute.policies import (
    DailyBar,
    LegParams,
    LegState,
    adjust_for_split,
    buy_price,
    hard_stop,
    position_shares,
    r_multiple,
    reentry_budget,
    step_leg,
    zone_bounds,
)
from asof.strategy import ExecuteConfig

# (budget, decision close, hard stop) -> shares
Sizer = Callable[[float, float, float], int]


@dataclass(frozen=True, kw_only=True)
class DecisionPlan:
    reaction_session: date
    decision_close: float
    gap_low: float
    pre_earnings_close: float
    atr: float
    equity: float


@dataclass(frozen=True, kw_only=True)
class LegResult:
    leg: int
    entry_session: date
    entry_price: float
    shares: int
    key_level: float  # the leg's initial key, before any trailing
    hard_stop: float  # the leg's initial hard stop
    exit_session: date
    exit_price: float
    exit_reason: str
    pnl: float
    r_multiple: float
    holding_sessions: int  # bars held, entry and exit day included
    # Lowest / highest price seen while held, in % of entry_price; see DecisionSim.step.
    mae_pct: float
    mfe_pct: float


@dataclass(frozen=True, kw_only=True)
class DecisionOutcome:
    legs: list[LegResult] = field(default_factory=list)
    cancelled: str | None = None


@dataclass
class _Order:
    leg: int
    key: float
    shares: int


@dataclass
class _Open:
    leg: int
    entry_session: date
    initial_key: float
    state: LegState
    # Extremes of the bars held before the exit bar; None until one such bar has been seen.
    low: float | None = None
    high: float | None = None


class DecisionSim:
    def __init__(self, plan: DecisionPlan, cfg: ExecuteConfig, sizer: Sizer | None = None) -> None:
        self.plan = plan
        self.cfg = cfg
        self.atr = plan.atr
        self.pre_close = plan.pre_earnings_close
        self.budget = plan.equity * cfg.sizing.risk_pct / 100
        self.sizer = sizer or self._standalone_sizer
        self.legs: list[LegResult] = []
        self.cancelled: str | None = None
        self.sessions_after = 0
        self.reentries = 0
        self.invalid = False
        self.waiting = False
        self.done = False
        self.order: _Order | None = None
        self.open: _Open | None = None
        # The reaction day itself may already have filled the gap; the replay decides after that
        # bar, so the check is made here rather than by stepping it.
        self.invalid = plan.decision_close < self._zone(self.pre_close)[0]
        hard = hard_stop(plan.gap_low, self.atr, cfg.stop.hard_atr)
        shares = self.sizer(self.budget, plan.decision_close, hard)
        if shares > 0:
            self.order = _Order(leg=1, key=plan.gap_low, shares=shares)
        else:
            self.cancelled = "no_size"
            self.done = True

    def _standalone_sizer(self, budget: float, price: float, hard: float) -> int:
        sizing = self.cfg.sizing
        return position_shares(
            budget=budget,
            price=price,
            hard=hard,
            equity=self.plan.equity,
            max_position_pct=sizing.max_position_pct,
            max_gross_pct=sizing.max_gross_pct,
            gross_exposure=0.0,
        )

    @property
    def params(self) -> LegParams:
        stop = self.cfg.stop
        return LegParams(
            atr=self.atr,
            zone_atr=stop.zone_atr,
            zone_cap_pct=stop.zone_cap_pct,
            hard_atr=stop.hard_atr,
            observe_days=stop.observe_days,
            trail_atr=stop.trail_atr,
            max_sessions=self.cfg.exit.max_sessions,
            slippage_bps=self.cfg.fills.slippage_bps,
        )

    @property
    def finished(self) -> bool:
        return self.done and self.open is None and self.order is None

    def _zone(self, key: float) -> tuple[float, float]:
        stop = self.cfg.stop
        return zone_bounds(key, self.atr, stop.zone_atr, stop.zone_cap_pct)

    def apply_split(self, ratio: float) -> None:
        """A ``ratio``-for-1 split takes effect before the bar of its ex-date is processed."""
        if ratio == 1:
            return
        self.atr /= ratio
        self.pre_close /= ratio
        if self.open is not None:
            self.open.state = adjust_for_split(self.open.state, ratio)
            self.open.initial_key /= ratio
            if self.open.low is not None and self.open.high is not None:
                self.open.low /= ratio
                self.open.high /= ratio
        if self.order is not None:
            self.order.key /= ratio
            self.order.shares = int(self.order.shares * ratio + 1e-9)

    def _close_leg(self, session: date, price: float, reason: str) -> LegResult:
        assert self.open is not None
        state = self.open.state
        # The exit bar adds only the exit price (and its open before an intraday stop): a daily
        # bar does not say whether its high or low came before or after the fill. Should a leg
        # close with nothing seen yet, its entry price stands in.
        held_low, held_high = self.open.low, self.open.high
        if held_low is None or held_high is None:
            held_low = held_high = state.entry_price
        low, high = min(held_low, price), max(held_high, price)
        result = LegResult(
            leg=self.open.leg,
            entry_session=self.open.entry_session,
            entry_price=state.entry_price,
            shares=state.shares,
            key_level=self.open.initial_key,
            hard_stop=state.initial_hard,
            exit_session=session,
            exit_price=price,
            exit_reason=reason,
            pnl=state.shares * (price - state.entry_price),
            r_multiple=r_multiple(state.entry_price, price, state.initial_hard),
            holding_sessions=state.sessions_held + 1,
            mae_pct=100 * (low / state.entry_price - 1),
            mfe_pct=100 * (high / state.entry_price - 1),
        )
        self.legs.append(result)
        self.open = None
        if result.leg == 1 and self.reentries < self.cfg.reentry.max_count and not self.invalid:
            self.waiting = True
        else:
            self.done = True
        return result

    def force_exit(self, session: date, price: float, reason: str) -> LegResult | None:
        """Close whatever is open at ``price`` (e.g. no bars for too long) and stop."""
        result = self._close_leg(session, price, reason) if self.open is not None else None
        if self.order is not None and self.order.leg == 1:
            self.cancelled = reason
        self.order = None
        self.waiting = False
        self.done = True
        return result

    def skip_session(self) -> None:
        """A market session without a bar of this security: an order for its open cannot fill
        and is dropped, and the session still counts toward the re-entry window."""
        if self.finished:
            return
        self.sessions_after += 1
        if self.order is not None:
            if self.order.leg == 1:
                self.cancelled = "no_bar"
            self.order = None
            self.done = True
        if self.waiting and self.sessions_after > self.cfg.reentry.window_sessions:
            self.waiting = False
            self.done = True

    def step(self, bar: DailyBar) -> list[LegResult]:
        """Process one bar of the security; returns the legs that closed on it."""
        closed: list[LegResult] = []
        if self.finished or bar.session_date < self.plan.reaction_session:
            return closed
        reentry_bottom, reentry_top = self._zone(self.pre_close)
        if bar.session_date == self.plan.reaction_session:
            self.invalid = self.invalid or bar.close < reentry_bottom
            return closed
        self.sessions_after += 1

        if self.order is not None:
            order, self.order = self.order, None
            bottom, _ = self._zone(order.key)
            if bar.open <= bottom:
                if order.leg == 1:
                    self.cancelled = "entry_below_zone"
                self.done = True
                return closed
            self.open = _Open(
                leg=order.leg,
                entry_session=bar.session_date,
                initial_key=order.key,
                state=LegState(
                    key=order.key,
                    highest_close=order.key,
                    observe_count=0,
                    pending_exit=None,
                    shares=order.shares,
                    entry_price=buy_price(bar.open, self.cfg.fills.slippage_bps),
                    initial_hard=hard_stop(order.key, self.atr, self.cfg.stop.hard_atr),
                    sessions_held=0,
                ),
            )

        if self.open is not None:
            state, fill = step_leg(self.open.state, bar, self.params)
            self.open.state = state
            if fill is not None:
                if fill.reason == "hard":
                    # An intraday stop fills after the open, so the open was seen while held.
                    o = self.open
                    o.low = bar.open if o.low is None or bar.open < o.low else o.low
                    o.high = bar.open if o.high is None or bar.open > o.high else o.high
                closed.append(self._close_leg(fill.session_date, fill.price, fill.reason))
            else:
                # Entry fills at the open, so the whole range of every bar held counts.
                o = self.open
                o.low = bar.low if o.low is None or bar.low < o.low else o.low
                o.high = bar.high if o.high is None or bar.high > o.high else o.high

        if bar.close < reentry_bottom:
            self.invalid = True
        if self.waiting:
            if self.invalid or self.sessions_after > self.cfg.reentry.window_sessions:
                self.waiting = False
                self.done = True
            elif bar.low <= reentry_top and bar.close >= reentry_top:
                self.waiting = False
                leg1_pnl = sum(leg.pnl for leg in self.legs)
                budget = reentry_budget(self.budget, leg1_pnl, self.cfg.reentry.risk_mode)
                hard = hard_stop(self.pre_close, self.atr, self.cfg.stop.hard_atr)
                shares = self.sizer(budget, bar.close, hard) if budget > 0 else 0
                if shares > 0:
                    self.reentries += 1
                    self.order = _Order(leg=2, key=self.pre_close, shares=shares)
                self.done = True  # nothing further after this leg
        return closed

    def outcome(self) -> DecisionOutcome:
        return DecisionOutcome(legs=list(self.legs), cancelled=self.cancelled)


def simulate_decision(
    plan: DecisionPlan, bars: Sequence[DailyBar], execute: ExecuteConfig
) -> DecisionOutcome:
    sim = DecisionSim(plan, execute)
    for bar in sorted(bars, key=lambda b: b.session_date):
        if sim.finished:
            break
        sim.step(bar)
    return sim.outcome()
