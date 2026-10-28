"""One decision from order to last exit: leg 1, then at most ``max_count`` re-entries.

``DecisionSim`` is a state machine fed one daily bar at a time, so the replay can step every
open decision in calendar order and never hand it a bar from the future. ``simulate_decision``
feeds a whole bar list through the same machine.

Leg 1 buys the open after the reaction day with its key at the gap-day low. Once it has exited,
a bar (the exit day included) that touches the pre-earnings-close zone and closes at or above
its top, within ``window_sessions`` sessions of the reaction day, orders leg 2 for the next open
with its key at the pre-earnings close. A close below that zone at any point since the reaction
day means the gap has filled and ends the setup.

A spin-off is known before its ex-date opens, its price only once the children have traded:
``begin_spinoff`` freezes the decision through that wait (no stop, no re-entry check, an order
for the ex open cancelled), ``apply_spinoff`` rescales its levels by the factor, and
``credit_children`` pays the children's sale into the leg held at the ex-date. An event that
cannot be priced ends with ``end_spinoff_unpriced``: raw levels again, no child.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import date

from asof.execute.holding import Holding
from asof.execute.policies import (
    DailyBar,
    LegParams,
    LegState,
    adjust_for_spinoff,
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
    child_proceeds: float = 0.0  # spin-off children sold, included in pnl and r_multiple
    spinoff: str | None = None  # "adjust" / "restart": held through a spin-off's ex-date


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
    holding: Holding = field(default_factory=Holding)
    child_proceeds: float = 0.0


@dataclass
class _Unpaid:
    """A leg whose shares left before its children were paid: booked once they are."""

    open: _Open
    session: date
    price: float
    reason: str


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
        self.frozen = False  # between a spin-off's ex-date and its resolution
        self.unpaid: _Unpaid | None = None
        self.frozen_closes: list[float] = []  # closes seen frozen, checked once levels are scaled
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
        return self.done and self.open is None and self.order is None and self.unpaid is None

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
            self.open.holding.split(ratio)
        if self.order is not None:
            self.order.key /= ratio
            self.order.shares = int(self.order.shares * ratio + 1e-9)
            if self.order.shares == 0:  # a reverse split left less than one share to buy
                if self.order.leg == 1:
                    self.cancelled = "no_size"
                self.order = None
                self.done = True

    def _result(self, o: _Open, session: date, price: float, reason: str) -> LegResult:
        state = o.state
        # An intraday stop also saw its bar's open (``step``).
        mae, mfe = o.holding.excursions(state.entry_price, price)
        child = o.child_proceeds
        result = LegResult(
            leg=o.leg,
            entry_session=o.entry_session,
            entry_price=state.entry_price,
            shares=state.shares,
            key_level=o.initial_key,
            hard_stop=state.initial_hard,
            exit_session=session,
            exit_price=price,
            exit_reason=reason,
            pnl=state.shares * (price - state.entry_price) + child,
            r_multiple=r_multiple(
                state.entry_price, price, state.initial_hard, child / state.shares
            ),
            holding_sessions=state.sessions_held + 1,
            mae_pct=mae,
            mfe_pct=mfe,
            child_proceeds=child,
            spinoff=o.holding.spinoff,
        )
        self.legs.append(result)
        return result

    @property
    def held_shares(self) -> int:
        return self.open.state.shares if self.open is not None else 0

    def exposure(self, last_close: float) -> float:
        """The open leg and a waiting order, at ``last_close``."""
        shares = (self.open.state.shares if self.open is not None else 0) + (
            self.order.shares if self.order is not None else 0
        )
        return shares * last_close

    def unrealized(self, last_close: float) -> float:
        """Not yet booked: the open leg at ``last_close``, and a leg sold before its children
        were paid at its sale price."""
        total = 0.0
        if self.open is not None:
            state = self.open.state
            total += state.shares * (last_close - state.entry_price)
        if self.unpaid is not None:
            state = self.unpaid.open.state
            total += state.shares * (self.unpaid.price - state.entry_price)
        return total

    def _close_leg(self, session: date, price: float, reason: str) -> LegResult | None:
        """Close the open leg; one still owed its children is held back (None) until paid."""
        assert self.open is not None
        o, self.open = self.open, None
        if o.leg == 1 and self.reentries < self.cfg.reentry.max_count and not self.invalid:
            self.waiting = True
        else:
            self.done = True
        if o.holding.entitled:
            self.unpaid = _Unpaid(open=o, session=session, price=price, reason=reason)
            return None
        return self._result(o, session, price, reason)

    # ---- spin-offs -------------------------------------------------------------------------

    def begin_spinoff(self) -> None:
        """A spin-off's ex-date is the next bar: the leg held now is owed the children."""
        self.frozen = True
        self.frozen_closes = []
        if self.open is not None:
            self.open.holding.begin_spinoff(self.open.state.shares)

    def apply_spinoff(self, factor: float) -> None:
        """The event's factor became known at the last bar's close; the next bar sees the
        rescaled levels. The children are paid later, by ``credit_children``."""
        self.frozen = False
        self.atr *= factor
        self.pre_close *= factor
        held = [x for x in (self.open, self.unpaid.open if self.unpaid else None) if x]
        for o in held:
            if o.holding.entitled:
                o.holding.adjust(factor)
        if self.open is not None and self.open.holding.entitled:
            self.open.state = adjust_for_spinoff(self.open.state, factor)
        # The frozen closes are ex-date prices, comparable with the scaled re-entry zone now.
        bottom, _ = self._zone(self.pre_close)
        if any(close < bottom for close in self.frozen_closes):
            self.invalid = True
            if self.waiting:
                self.waiting = False
                self.done = True
        self.frozen_closes = []

    def entitled_shares(self) -> int:
        """Parent shares the children are owed on (0 when none are owed)."""
        for o in (self.unpaid.open if self.unpaid else None, self.open):
            if o is not None and o.holding.entitled:
                return o.holding.entitled_shares
        return 0

    def owes_children(self) -> bool:
        """Children are still to be paid into a leg (held, or sold and held back)."""
        return (self.open is not None and self.open.holding.entitled) or self.unpaid is not None

    def overlap_spinoff(self) -> None:
        """Another spin-off goes ex while one is still being settled: it is not priced. An
        order for this open is dropped, a waiting setup ends, the held leg is flagged."""
        if self.order is not None:
            if self.order.leg == 1:
                self.cancelled = "spinoff_ex"
            self.order = None
            self.done = True
        if self.waiting:
            self.waiting = False
            self.done = True
        if self.open is not None:
            self.open.holding.spinoff = "restart"

    def credit_children(self, proceeds: float) -> list[LegResult]:
        """The children's sale: paid into the leg held at the ex-date."""
        if self.unpaid is not None:
            u, self.unpaid = self.unpaid, None
            u.open.child_proceeds += proceeds
            u.open.holding.entitled = False
            return [self._result(u.open, u.session, u.price, u.reason)]
        if self.open is not None and self.open.holding.entitled:
            self.open.child_proceeds += proceeds
            self.open.holding.entitled = False
        return []

    def end_spinoff_unpriced(self) -> list[LegResult]:
        """The event cannot be priced (restart, or the wait ran out): raw levels, no child, and
        a waiting re-entry setup ends, its level no longer comparable."""
        self.frozen = False
        if self.waiting:
            self.waiting = False
            self.done = True
        if self.open is not None and self.open.holding.entitled:
            self.open.holding.restart()
        if self.unpaid is not None:
            u, self.unpaid = self.unpaid, None
            u.open.holding.restart()
            return [self._result(u.open, u.session, u.price, u.reason)]
        return []

    def force_exit(self, session: date, price: float, reason: str) -> LegResult | None:
        """Close whatever is open at ``price`` (e.g. no bars for too long) and stop."""
        result = self._close_leg(session, price, reason) if self.open is not None else None
        self.frozen = False
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

        if self.order is not None and self.frozen:
            # Its level and size were planned on prices from before the ex-date.
            if self.order.leg == 1:
                self.cancelled = "spinoff_ex"
            self.order = None
            self.done = True
        if self.frozen:
            return self._step_frozen(bar)

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
                    self.open.holding.see(bar.open, bar.open)
                leg = self._close_leg(fill.session_date, fill.price, fill.reason)
                if leg is not None:
                    closed.append(leg)
            else:
                # Entry fills at the open, so the whole range of every bar held counts.
                self.open.holding.see(bar.low, bar.high)

        if bar.close < reentry_bottom:
            self.invalid = True
        if self.waiting:
            if self.invalid or self.sessions_after > self.cfg.reentry.window_sessions:
                self.waiting = False
                self.done = True
            elif bar.low <= reentry_top and bar.close >= reentry_top:
                self.waiting = False
                leg1_pnl = sum(leg.pnl for leg in self.legs)
                if self.unpaid is not None:  # sold, its children not yet paid: counted as 0
                    u = self.unpaid.open.state
                    leg1_pnl += u.shares * (self.unpaid.price - u.entry_price)
                budget = reentry_budget(self.budget, leg1_pnl, self.cfg.reentry.risk_mode)
                hard = hard_stop(self.pre_close, self.atr, self.cfg.stop.hard_atr)
                shares = self.sizer(budget, bar.close, hard) if budget > 0 else 0
                if shares > 0:
                    self.reentries += 1
                    self.order = _Order(leg=2, key=self.pre_close, shares=shares)
                self.done = True  # nothing further after this leg
        return closed

    def _step_frozen(self, bar: DailyBar) -> list[LegResult]:
        """A bar before the spin-off's factor is known: an exit decided earlier still fills at
        the open; otherwise only the extremes and the session count move."""
        o = self.open
        if o is not None:
            if o.state.pending_exit is not None:
                _, fill = step_leg(o.state, bar, self.params)
                assert fill is not None
                held_back = self._close_leg(fill.session_date, fill.price, fill.reason)
                assert held_back is None  # a leg held at the ex-date is owed its children
            else:
                o.state = replace(o.state, sessions_held=o.state.sessions_held + 1)
                o.holding.see(bar.low, bar.high)
        self.frozen_closes.append(bar.close)
        if self.waiting and self.sessions_after > self.cfg.reentry.window_sessions:
            self.waiting = False
            self.done = True
        return []

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
