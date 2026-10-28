"""One B-Xtrender position, from the order at the signal close to its last share sold.

The caller reads the exit flags at each close from what is visible then and hands them in with
the bar; this machine never reads anything. Per bar while held: the tranches decided at the last
close fill at this open, the bar's range joins the held extremes, then each exit whose flag is up
and that has not fired becomes pending for the next open. Each exit sells one share of the
position (``floor(shares / exits left)``, everything if that is 0) and the last exit what is left;
there is no price stop.

R is cut from ``atr_mult`` x the signal's ATR below the entry: a volatility yardstick (it sized
the position), not a stop, recorded as the trade's ``hard_stop``. Splits rescale it; a spin-off
does not (the raw record, as for earnings_gap).

Spin-offs follow ``DecisionSim``: ``begin_spinoff`` freezes the position (no flag is read; a
tranche decided before still fills at the ex open, held back until its children are paid), and
the children of the shares held at the ex-date are shared pro rata by shares between the
tranches sold while owed and the shares still held.
"""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any

import pandas as pd

from asof.execute.holding import Holding
from asof.execute.policies import DailyBar, buy_price, position_shares, r_multiple, sell_price
from asof.execute.simulate import DecisionOutcome, LegResult
from asof.strategy import BxExecuteConfig, BxSizingConfig

_EPS = 1e-9


@dataclass(frozen=True, kw_only=True)
class ExitFlags:
    daily_red: bool  # tonight's daily histogram < 0
    weekly_down: bool  # last completed weekly bar < the one before it


def _value(row: Mapping[str, Any], key: str) -> float | None:
    value = row.get(key)
    return None if value is None or math.isnan(value) else float(value)


def exit_flags(row: Mapping[str, Any]) -> ExitFlags:
    """The exit flags of one panel row; a flag with a missing value among its inputs is down."""
    daily, weekly = _value(row, "bx_daily"), _value(row, "bx_weekly")
    prev = _value(row, "bx_weekly_prev")
    return ExitFlags(
        daily_red=daily is not None and daily < 0,
        weekly_down=weekly is not None and prev is not None and weekly < prev,
    )


def signal_atr(bars: pd.DataFrame, days: int) -> float:
    """Mean of the last ``days`` true ranges of ``bars`` (one security in date order, already
    split-adjusted), the last row's included: at the signal close its range is known. NaN with
    fewer than ``days + 1`` rows, the first true range needing a previous close."""
    if len(bars) < days + 1:
        return math.nan
    window = bars.iloc[-(days + 1) :]
    high = window["high"].to_numpy()[1:]
    low = window["low"].to_numpy()[1:]
    prev = window["close"].to_numpy()[:-1]
    tr = [max(h - lo, abs(h - c), abs(lo - c)) for h, lo, c in zip(high, low, prev, strict=True)]
    return float(sum(tr) / days)


@dataclass(frozen=True, kw_only=True)
class BxPlan:
    signal_session: date
    signal_close: float
    atr: float  # at the signal close
    shares: int  # sized by the caller
    equity: float


def bx_shares(
    *, equity: float, close: float, atr: float, sizing: BxSizingConfig, gross_exposure: float
) -> int:
    """Shares risking ``risk_pct`` of equity over ``atr_mult`` ATRs, capped by one position's
    share of equity and by the room left under the gross cap. No positive ATR, no shares."""
    if not atr > 0:
        return 0
    return position_shares(
        budget=equity * sizing.risk_pct / 100,
        price=close,
        hard=close - sizing.atr_mult * atr,
        equity=equity,
        max_position_pct=sizing.max_position_pct,
        max_gross_pct=sizing.max_gross_pct,
        gross_exposure=gross_exposure,
    )


@dataclass
class _Sold:
    """A tranche sold while its children were owed: booked once they are paid (or never)."""

    shares: int
    ex_shares: float  # the same shares counted as at the ex-date (later splits undone)
    session: date
    price: float
    reason: str
    entry_price: float
    hard: float
    holding_sessions: int
    prior_child: float  # children of an earlier, settled spin-off these shares carried
    holding: Holding = field(default_factory=Holding)


class BxPositionSim:
    def __init__(self, plan: BxPlan, cfg: BxExecuteConfig) -> None:
        self.plan = plan
        self.cfg = cfg
        self.atr = plan.atr
        self.bps = cfg.fills.slippage_bps
        self.order: int | None = plan.shares if plan.shares > 0 else None
        self.cancelled: str | None = None if plan.shares > 0 else "no_size"
        self.legs: list[LegResult] = []
        self.entered = False
        self.entry_session: date | None = None
        self.entry_price = 0.0
        self.shares = 0
        self.sessions_held = 0
        self.holding = Holding()
        self.fired: set[str] = set()
        self.pending: list[str] = []  # exits read at the last close, sold at the next open
        self.sold_tranches = 0
        self.open_child = 0.0  # children already paid to the shares still held
        self.frozen = False
        self.ex_ratio = 1.0  # splits since the ex-date, to count shares as they were then
        self.held_back: list[_Sold] = []

    # ---- state ------------------------------------------------------------------------------

    @property
    def finished(self) -> bool:
        if self.cancelled is not None:
            return True
        return self.entered and self.shares == 0 and not self.held_back

    def owes_children(self) -> bool:
        """Children are still to be paid (to shares held, or sold and held back)."""
        return (self.shares > 0 and self.holding.entitled) or bool(self.held_back)

    def entitled_shares(self) -> int:
        """Parent shares the children are owed on (0 when none are)."""
        return self.holding.entitled_shares if self.owes_children() else 0

    def outcome(self) -> DecisionOutcome:
        return DecisionOutcome(legs=list(self.legs), cancelled=self.cancelled)

    @property
    def held_shares(self) -> int:
        return self.shares

    def exposure(self, last_close: float) -> float:
        """Shares held or waiting to be bought, at ``last_close``; a held-back tranche is gone."""
        return (self.shares + (self.order or 0)) * last_close

    def unrealized(self, last_close: float) -> float:
        """Not yet booked: the shares held at ``last_close``, the tranches sold but held back for
        their children at their sale price, and the children already paid to the shares held."""
        held = self.shares * (last_close - self.entry_price) if self.shares > 0 else 0.0
        back = sum(s.shares * (s.price - s.entry_price) + s.prior_child for s in self.held_back)
        return held + back + self.open_child

    @property
    def hard(self) -> float:
        return self.entry_price - self.cfg.sizing.atr_mult * self.atr

    # ---- selling ----------------------------------------------------------------------------

    def _sell(self, count: int, session: date, price: float, reason: str) -> list[LegResult]:
        """Sell ``count`` of the shares held: booked now, or held back while children are owed."""
        child = self.open_child * count / self.shares
        self.open_child -= child
        self.shares -= count
        self.sold_tranches += 1
        if self.holding.entitled:
            self.held_back.append(
                _Sold(
                    shares=count,
                    ex_shares=count / self.ex_ratio,
                    session=session,
                    price=price,
                    reason=reason,
                    entry_price=self.entry_price,
                    hard=self.hard,
                    holding_sessions=self.sessions_held,
                    prior_child=child,
                    holding=replace(self.holding),
                )
            )
            return []
        return [
            self._book(
                count,
                session,
                price,
                reason,
                self.entry_price,
                self.hard,
                self.sessions_held,
                self.holding,
                child,
            )
        ]

    def _book(
        self,
        shares: int,
        session: date,
        price: float,
        reason: str,
        entry_price: float,
        hard: float,
        holding_sessions: int,
        holding: Holding,
        child: float,
    ) -> LegResult:
        assert self.entry_session is not None
        mae, mfe = holding.excursions(entry_price, price)
        leg = LegResult(
            leg=1,
            entry_session=self.entry_session,
            entry_price=entry_price,
            shares=shares,
            key_level=entry_price,
            hard_stop=hard,
            exit_session=session,
            exit_price=price,
            exit_reason=reason,
            pnl=shares * (price - entry_price) + child,
            r_multiple=r_multiple(entry_price, price, hard, child / shares),
            holding_sessions=holding_sessions,
            mae_pct=mae,
            mfe_pct=mfe,
            child_proceeds=child,
            spinoff=holding.spinoff,
        )
        self.legs.append(leg)
        return leg

    def _tranche(self) -> int:
        left = len(self.cfg.exits) - self.sold_tranches
        if left <= 1:
            return self.shares
        return math.floor(self.shares / left + _EPS) or self.shares

    # ---- one bar ----------------------------------------------------------------------------

    def step(self, bar: DailyBar, flags: ExitFlags | None) -> list[LegResult]:
        """Process one bar of the security; returns the tranches booked on it."""
        if self.finished or bar.session_date <= self.plan.signal_session:
            return []
        if self.order is not None:
            if self.frozen:
                # sized on prices from before the ex-date
                self.order = None
                self.cancelled = "spinoff_ex"
                return []
            self.entered = True
            self.entry_session = bar.session_date
            self.entry_price = buy_price(bar.open, self.bps)
            self.shares, self.order = self.order, None
        if self.shares == 0:
            return []
        self.sessions_held += 1

        closed: list[LegResult] = []
        price = sell_price(bar.open, self.bps)
        for reason in self.pending:
            if self.shares > 0:
                closed += self._sell(self._tranche(), bar.session_date, price, reason)
        self.pending = []
        if self.shares == 0:
            return closed  # left at the open: the rest of the bar was not held

        self.holding.see(bar.low, bar.high)
        if flags is not None and not self.frozen:
            for name in self.cfg.exits:
                if name not in self.fired and getattr(flags, name):
                    self.fired.add(name)
                    self.pending.append(name)
        return closed

    def skip_session(self) -> None:
        """A session without a bar: an order for its open cannot fill and is dropped."""
        if self.order is not None:
            self.order = None
            self.cancelled = "no_bar"

    def force_exit(self, session: date, price: float, reason: str) -> list[LegResult]:
        """Sell whatever is held at ``price``, or cancel a waiting order, and stop."""
        if self.order is not None:
            self.order = None
            self.cancelled = reason
            return []
        self.pending = []
        if self.shares == 0:
            return []
        self.sold_tranches = len(self.cfg.exits) - 1
        return self._sell(self.shares, session, price, reason)

    # ---- splits and spin-offs ---------------------------------------------------------------

    def apply_split(self, ratio: float) -> None:
        """A ``ratio``-for-1 split takes effect before the bar of its ex-date is processed."""
        if ratio == 1:
            return
        self.atr /= ratio
        if self.order is not None:
            self.order = math.floor(self.order * ratio + _EPS)
            if self.order == 0:  # a reverse split left less than one share to buy
                self.order = None
                self.cancelled = "no_size"
        if self.shares > 0:
            self.shares = math.floor(self.shares * ratio + _EPS)
            self.entry_price /= ratio
            self.holding.split(ratio)
            if self.holding.entitled:
                self.ex_ratio *= ratio

    def begin_spinoff(self) -> None:
        """A spin-off's ex-date is the next bar: the shares held now are owed the children."""
        self.frozen = True
        if self.shares > 0:
            self.holding.begin_spinoff(self.shares)
            self.ex_ratio = 1.0

    def apply_spinoff(self, factor: float) -> None:
        """The event's factor became known at the last bar's close."""
        self.frozen = False
        if self.shares > 0 and self.holding.entitled:
            self.holding.adjust(factor)
        for sold in self.held_back:
            sold.holding.adjust(factor)

    def credit_children(self, proceeds: float) -> list[LegResult]:
        """The children's sale, shared by shares as held at the ex-date."""
        total = self.holding.entitled_shares
        paid = sum(proceeds * sold.ex_shares / total for sold in self.held_back)
        released = self._release(lambda sold: proceeds * sold.ex_shares / total)
        if self.shares > 0 and self.holding.entitled:
            self.open_child += proceeds - paid
        self.holding.entitled = False
        return released

    def end_spinoff_unpriced(self) -> list[LegResult]:
        """The event cannot be priced: raw levels again, no child."""
        self.frozen = False
        if self.shares > 0 and self.holding.entitled:
            self.holding.restart()
        for sold in self.held_back:
            sold.holding.restart()
        self.holding.entitled = False
        return self._release(lambda _: 0.0)

    def _release(self, child_of: Callable[[_Sold], float]) -> list[LegResult]:
        out = []
        for sold in self.held_back:
            sold.holding.entitled = False
            out.append(
                self._book(
                    sold.shares,
                    sold.session,
                    sold.price,
                    sold.reason,
                    sold.entry_price,
                    sold.hard,
                    sold.holding_sessions,
                    sold.holding,
                    sold.prior_child + child_of(sold),
                )
            )
        self.held_back = []
        return out

    def overlap_spinoff(self) -> None:
        """Another spin-off goes ex while one is still being settled: it is not priced."""
        if self.order is not None:
            self.order = None
            self.cancelled = "spinoff_ex"
        if self.shares > 0:
            self.holding.spinoff = "restart"
