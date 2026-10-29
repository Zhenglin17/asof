"""A synthetic market with a known answer, for tests/evolve/test_planted_edge.py: the B-Xtrender
entry signal is made to work by construction, and the gate must find it (and must find nothing
in a control market built from the same random numbers without it).

Provenance of the fake data (all made up):
- Calendar: tests.scan.bx_world.CAL (weekdays 2023-01-03 .. 2024-06-28 minus the NYSE
  holidays of those years, documented there).
- Securities: STOCKS common stocks P00 .. P29 (security ids S_P00 ..) and SPY (class etf,
  never in the BX pool, the benchmark of the protocol). Volume VOLUME shares a day: at prices
  between ~$15 and ~$200 that is ~$300M .. $4B a day, far inside the $50M tier (the fixture
  guards check every stock stays in it).
- Prices: per stock and session, a gap g ~ N(0, 1%) from the previous close to the open and an
  intraday move r ~ N(0, 2%) from the open to the close, plus high / low wicks of |N(0, 0.5%)|
  (the distribution of tests.features.test_bx.walk). The random numbers are drawn once per
  stock (seed BASE_SEED + index), so the planted and the control market share them exactly.
- Planted edge: whenever ``tests.features.test_bx.reference`` + ``tests.scan.bx_world.ref_signal``
  (the scan's mid signal with warm-ups 1, no daily_min, no market filter) fires for a stock on
  night t, its next H sessions t+1 .. t+H get an extra intraday drift: close = open * (1 + r) *
  (1 + mu). A signal on a night whose next session is already inside a run plants nothing
  (no stacking, no extension): runs are exactly H sessions. Opens, highs and lows follow from
  the drifted closes. Measured on this world: the raw signal fires on ~24% of nights, and the
  runs cover ~half of all stock-sessions, so a random draw often lands on a drifting stock too.
- A daily bar is visible from 20:00 New York of its session (asof.ingest.bars, documented);
  the master segments are known from long before the calendar (tests/store/test_universe.Seg).

Why generating day by day is self-consistent: the BX values of night t (daily histogram, last
completed weekly / monthly bar) use only closes up to t (tests/features/test_bx.py pins that
adding later rows changes no earlier value). Planting at t changes sessions t+1 .. t+H only,
so t's signal, and every earlier one, is the same before and after. The generator therefore
computes the reference over the series as planted so far, takes the first signal night after
the last one handled, plants, and recomputes; ``generate`` asserts after every recompute that no
signal at or before the night just handled moved.

Where the drift sits in a leg: the replay buys the open after the signal night. A leg bought at
the open of session e and sold at the open of x holds the intraday moves of e .. x-1; sold at
the close of x, also x's. ``held_drift`` is the planted factor over those sessions, minus 1.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from tests.features.test_bx import reference
from tests.scan.bx_world import CAL, ref_signal
from tests.store.audit_world import bar as raw_bar
from tests.store.test_spinoffs import write_bars
from tests.store.test_universe import Seg, write_classes, write_master

STOCKS = 30
H = 10
MU = 0.003
BASE_SEED = 7_000
SPY_SEED = 6_999
VOLUME = 20_000_000
START_PRICE = 50.0


@dataclass(frozen=True)
class Stock:
    security_id: str
    symbol: str
    frame: pd.DataFrame  # session_date, open, high, low, close (one row per CAL session)
    drift: np.ndarray  # planted intraday drift per CAL session (0 or mu)
    planted_nights: tuple[date, ...]  # signal nights that planted a run
    signal: np.ndarray  # the reference signal per CAL session, of the final series


@dataclass(frozen=True)
class World:
    mu: float
    stocks: tuple[Stock, ...]
    spy: pd.DataFrame

    def stock(self, security_id: str) -> Stock:
        [s] = [s for s in self.stocks if s.security_id == security_id]
        return s


def _shocks(seed: int, n: int) -> np.ndarray:
    """(n, 4): gap, intraday move, high wick, low wick."""
    rng = np.random.default_rng(seed)
    return np.column_stack(
        [
            rng.normal(0, 0.01, n),
            rng.normal(0, 0.02, n),
            np.abs(rng.normal(0, 0.005, n)),
            np.abs(rng.normal(0, 0.005, n)),
        ]
    )


def _prices(shocks: np.ndarray, drift: np.ndarray, start: float = START_PRICE) -> pd.DataFrame:
    gap, move, up, down = shocks.T
    opens, closes = np.empty(len(CAL)), np.empty(len(CAL))
    close = start
    for i in range(len(CAL)):
        opens[i] = close * (1 + gap[i])
        close = opens[i] * (1 + move[i]) * (1 + drift[i])
        closes[i] = close
    return pd.DataFrame(
        {
            "session_date": CAL,
            "open": opens,
            "high": np.maximum(opens, closes) * (1 + up),
            "low": np.minimum(opens, closes) * (1 - down),
            "close": closes,
        }
    )


def _signal(security_id: str, frame: pd.DataFrame) -> np.ndarray:
    daily = frame.assign(security_id=security_id, symbol=security_id)
    ref = reference(daily)
    return np.array([ref_signal(r, "mid") for r in ref.to_dict("records")], dtype=bool)


def _stock(index: int, mu: float) -> Stock:
    sid, symbol = f"S_P{index:02d}", f"P{index:02d}"
    shocks = _shocks(BASE_SEED + index, len(CAL))
    drift = np.zeros(len(CAL))
    frame = _prices(shocks, drift)
    signal = _signal(sid, frame)
    planted: list[date] = []
    t = -1
    while mu:
        # the first signal night after the last one handled that is not inside a run
        later = [int(i) for i in np.flatnonzero(signal) if i > t and i + 1 < len(CAL)
                 and drift[i + 1] == 0]  # fmt: skip
        if not later:
            break
        t = later[0]
        drift[t + 1 : t + 1 + H] = mu
        planted.append(CAL[t])
        frame = _prices(shocks, drift)
        again = _signal(sid, frame)
        # self-consistency: planting after t moves no signal at or before t
        assert (again[: t + 1] == signal[: t + 1]).all(), (sid, CAL[t])
        signal = again
    return Stock(sid, symbol, frame, drift, tuple(planted), signal)


def generate(mu: float, stocks: int = STOCKS) -> World:
    spy = _prices(_shocks(SPY_SEED, len(CAL)) * 0.5, np.zeros(len(CAL)), start=400.0)
    return World(mu=mu, stocks=tuple(_stock(i, mu) for i in range(stocks)), spy=spy)


def write_store(root: Path, world: World) -> None:
    bars = []
    for symbol, frame in [(s.symbol, s.frame) for s in world.stocks] + [("SPY", world.spy)]:
        for r in frame.to_dict("records"):
            bars.append(
                raw_bar(
                    symbol,
                    r["session_date"],
                    close=float(r["close"]),
                    open=float(r["open"]),
                    high=float(r["high"]),
                    low=float(r["low"]),
                    vwap=float(r["close"]),
                    volume=VOLUME * (5 if symbol == "SPY" else 1),
                )
            )
    write_bars(root, bars)
    segs = [Seg(s.security_id, s.symbol) for s in world.stocks] + [Seg("S_SPY", "SPY", cls="etf")]
    write_master(root, segs)
    write_classes(root, segs)


def held_drift(world: World, security_id: str, entry: date, exit_: date, exit_at: str) -> float:
    """The planted factor a leg held (entry at the open of ``entry``), minus 1."""
    drift = world.stock(security_id).drift
    lo = CAL.index(entry)
    hi = CAL.index(exit_) + (1 if exit_at == "close" else 0)
    return float(np.prod(1 + drift[lo:hi]) - 1)


def drifting_share(world: World, day: date) -> float:
    """Share of the stocks with planted drift on ``day``."""
    i = CAL.index(day)
    return float(np.mean([s.drift[i] > 0 for s in world.stocks]))


def first_planted_night(world: World) -> date:
    nights = [n for s in world.stocks for n in s.planted_nights]
    assert nights, "fixture: nothing planted"
    return min(nights)


def signals_through(world: World, day: date) -> dict[str, list[bool]]:
    i = CAL.index(day)
    return {s.security_id: s.signal[: i + 1].tolist() for s in world.stocks}


def symbols(world: World) -> Sequence[str]:
    return [s.symbol for s in world.stocks]
