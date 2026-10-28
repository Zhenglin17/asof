"""Stop, trailing and sizing arithmetic for one leg, one daily bar at a time.

A stop is a zone around a key level, not a single price. On each bar, in this order (a daily bar
does not say what happened first, so the order is the conservative one):

0. an exit decided at the previous close fills at this open;
1. an open at or below the hard stop fills at the open (``hard_gap``);
2. a low at or below the hard stop fills at the stop (``hard``);
3. otherwise the close decides: below the zone -> exit at the next open (``break``); inside it
   -> watched, and ``observe_days`` closes in a row inside -> exit (``observe_timeout``); above it
   -> the watch resets;
4. the key trails the highest close by ``trail_atr`` ATRs and never falls; the next bar's zone
   and hard stop come from the new key.

A split rescales every level and the share count (``adjust_for_split``); a spin-off rescales only
the levels that drive later stops (``adjust_for_spinoff``), once its factor is known. Until then
the simulator does not step the leg at all (``DecisionSim.begin_spinoff``).
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date

_EPS = 1e-9


@dataclass(frozen=True, kw_only=True)
class DailyBar:
    session_date: date
    open: float
    high: float
    low: float
    close: float


@dataclass(frozen=True, kw_only=True)
class LegParams:
    atr: float
    zone_atr: float
    zone_cap_pct: float
    hard_atr: float
    observe_days: int
    trail_atr: float
    max_sessions: int
    slippage_bps: float


@dataclass(frozen=True, kw_only=True)
class LegState:
    key: float
    highest_close: float
    observe_count: int
    pending_exit: str | None  # the reason of an exit decided at a close, filled at the next open
    shares: int
    entry_price: float
    initial_hard: float
    sessions_held: int


@dataclass(frozen=True, kw_only=True)
class Fill:
    session_date: date
    price: float
    reason: str


def zone_bounds(
    key: float, atr: float, zone_atr: float, zone_cap_pct: float
) -> tuple[float, float]:
    width = min(zone_atr * atr, zone_cap_pct / 100 * key)
    return key - width, key + width


def hard_stop(key: float, atr: float, hard_atr: float) -> float:
    return key - hard_atr * atr


def trail_key(initial_key: float, highest_close: float, atr: float, trail_atr: float) -> float:
    return max(initial_key, highest_close - trail_atr * atr)


def buy_price(price: float, slippage_bps: float) -> float:
    return price * (1 + slippage_bps / 1e4)


def sell_price(price: float, slippage_bps: float) -> float:
    return price * (1 - slippage_bps / 1e4)


def r_multiple(
    entry_price: float, exit_price: float, hard: float, extra_per_share: float = 0.0
) -> float:
    """Gain per share in units of the risk taken at entry (entry minus the initial hard stop);
    ``extra_per_share`` is what else a share paid out while held (spin-off children)."""
    return (exit_price - entry_price + extra_per_share) / (entry_price - hard)


def step_leg(state: LegState, bar: DailyBar, params: LegParams) -> tuple[LegState, Fill | None]:
    day = bar.session_date
    bps = params.slippage_bps
    if state.pending_exit is not None:
        return state, Fill(
            session_date=day, price=sell_price(bar.open, bps), reason=state.pending_exit
        )
    hard = hard_stop(state.key, params.atr, params.hard_atr)
    if bar.open <= hard:
        return state, Fill(session_date=day, price=sell_price(bar.open, bps), reason="hard_gap")
    if bar.low <= hard:
        return state, Fill(session_date=day, price=sell_price(hard, bps), reason="hard")

    bottom, top = zone_bounds(state.key, params.atr, params.zone_atr, params.zone_cap_pct)
    observe = state.observe_count
    pending: str | None = None
    if bar.close < bottom:
        pending = "break"
    elif bar.close <= top:
        observe += 1
        if observe >= params.observe_days:
            pending = "observe_timeout"
    else:
        observe = 0
    highest = max(state.highest_close, bar.close)
    held = state.sessions_held + 1
    if pending is None and held >= params.max_sessions:
        pending = "max_sessions"
    return (
        replace(
            state,
            key=trail_key(state.key, highest, params.atr, params.trail_atr),
            highest_close=highest,
            observe_count=observe,
            pending_exit=pending,
            sessions_held=held,
        ),
        None,
    )


def position_shares(
    *,
    budget: float,
    price: float,
    hard: float,
    equity: float,
    max_position_pct: float,
    max_gross_pct: float,
    gross_exposure: float,
) -> int:
    """Shares whose loss down to ``hard`` is at most ``budget``, capped by one position's share
    of equity and by the room left under the gross exposure cap. ``price`` is the close at the
    decision, never the next open (not known at 20:00 New York)."""
    risk = price - hard
    if budget <= 0 or risk <= 0 or price <= 0:
        return 0
    by_risk = math.floor(budget / risk + _EPS)
    by_position = math.floor(equity * max_position_pct / 100 / price + _EPS)
    room = equity * max_gross_pct / 100 - gross_exposure
    by_gross = math.floor(room / price + _EPS) if room > 0 else 0
    return max(0, min(by_risk, by_position, by_gross))


def reentry_budget(budget: float, leg1_pnl: float, risk_mode: str) -> float:
    """``shared``: what leg 1 left of the decision's budget (a profit adds nothing);
    ``independent``: the full budget again."""
    if risk_mode == "independent":
        return budget
    return budget - max(0.0, -leg1_pnl)


def adjust_for_split(state: LegState, ratio: float) -> LegState:
    """A ``ratio``-for-1 split on a held leg: more shares, every price level divided."""
    if ratio == 1:
        return state
    return replace(
        state,
        shares=math.floor(state.shares * ratio + _EPS),
        key=state.key / ratio,
        highest_close=state.highest_close / ratio,
        entry_price=state.entry_price / ratio,
        initial_hard=state.initial_hard / ratio,
    )


def adjust_for_spinoff(state: LegState, factor: float) -> LegState:
    """A spin-off priced at ``factor`` on a held leg: the levels that drive the next stops are
    rescaled; the shares, the entry and its hard stop stay the raw record the R unit is cut
    from (the child's value is paid in cash, see ``spinoff_proceeds``)."""
    if factor == 1:
        return state
    return replace(state, key=state.key * factor, highest_close=state.highest_close * factor)


def spinoff_proceeds(
    shares: int, children: Sequence[tuple[float, float]], slippage_bps: float
) -> float:
    """Selling the children of ``shares`` parent shares: ``children`` holds (child shares per
    parent share, sale price). A fraction of a child share is paid at the same price."""
    return sum(shares * ratio * sell_price(price, slippage_bps) for ratio, price in children)
