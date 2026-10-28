"""Stores and expected numbers for the B-Xtrender replay tests (store: tests/scan/bx_world.py,
provenance there: random-walk prices, 20:00 New York bar availability, split records available
00:00 New York of their ex-date unless a test says otherwise).

Expected values never come from the code under test: signal nights and exit flags come from
``tests.features.test_bx.reference`` over each security's walk prices (period_bars +
bx_histogram, no store), the ATR from ``hand_atr`` over the stored bars, and the share count
from the already-tested ``bx_shares``.

Paths used by the tests (seed 101 = S_AAA of the scan tests, seed 202 = S_BBB; indexes on CAL,
found by a search over the reference and pinned by fixture guards in the tests):
- SIGNAL_I = 211: AAA signals; first daily_red night SIGNAL_I + 3, first weekly_down night
  SIGNAL_I + 5, no further signal until SIGNAL_I + 14.
- AGAIN_I = 239: AAA signals; three more signals while held (first daily_red + 6, weekly_down
  + 10), the next signal after the last sale at + 43.
- BOTH_I = 226: AAA and BBB both signal.
"""

import copy
import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from asof.execute.bx import bx_shares
from asof.strategy import BxSizingConfig, BxStrategyConfig, load_strategy
from tests.scan.bx_world import CAL, THIN_VOLUME, Sec, ref_frame, ref_signal
from tests.store.test_spinoffs import write_bars, write_other_rows
from tests.store.test_split_events import write_split_rows
from tests.store.test_universe import write_classes, write_master
from tests.strategy.test_bx_strategy_config import BX_DEFAULTS, write_bx

AAA = Sec("S_AAA", "AAA", seed=101)
BBB = Sec("S_BBB", "BBB", seed=202)
# Thin all along (outside the $50M tier): keeps every session on the calendar when a test
# removes another security's bars.
KEEP = Sec("S_KEEP", "KEEP", seed=999, volume=THIN_VOLUME)

SIGNAL_I = 211
AGAIN_I = 239
BOTH_I = 226

INITIAL = 100_000.0
BPS = 5.0  # BX_DEFAULTS execute.fills.slippage_bps

# warm-ups of 1 so the 2023-01 world can signal from its first year; no daily_min, no market
# filter: a signal is exactly the reference's state rise + daily rise.
SCAN: dict[str, Any] = {
    "variant": "mid",
    "universe": {"min_adv_usd": 50_000_000, "lookback_sessions": 20, "classes": ["common"]},
    "bx": {"l1": 5, "l2": 20, "l3": 5},
    "warmup": {"daily": 1, "weekly": 1, "monthly": 1},
    "daily_min": None,
    "require_monthly_up": False,
    "market_filter": None,
}


def bx_strategy(parent: Path, *, sizing: Mapping[str, Any] | None = None) -> Path:
    """``parent/bx_trender`` with SCAN and the default execute section (``sizing`` merged)."""
    execute = copy.deepcopy(BX_DEFAULTS["execute"])
    execute["sizing"] = {**execute["sizing"], **(sizing or {})}
    return write_bx(parent, {"scan": SCAN, "execute": execute})


def sizing_of(strategy_dir: Path) -> BxSizingConfig:
    return load_strategy(strategy_dir, BxStrategyConfig).execute.sizing


def build_world(
    root: Path,
    secs: Sequence[Sec],
    *,
    drop: Mapping[str, Iterable[date]] | None = None,
    splits: Sequence[dict[str, Any]] = (),
    spinoffs: Sequence[dict[str, Any]] = (),
    cut: date | None = None,
) -> None:
    """``tests.scan.bx_world.build`` plus ``drop``: security_id -> sessions without a bar."""
    gone = {sid: set(days) for sid, days in (drop or {}).items()}
    bars = []
    for s in secs:
        skip = gone.get(s.security_id, set())
        bars += [b for b in s.bars(cut) if b["session_date"] not in skip]
    write_bars(root, bars)
    segs = [seg for s in secs for seg in s.segs()]
    write_master(root, segs)
    write_classes(root, segs)
    if splits:
        write_split_rows(root, splits)
    if spinoffs:
        write_other_rows(root, spinoffs)


# -- reference ------------------------------------------------------------------------------------


def flagged(secs: Sequence[Sec]) -> pd.DataFrame:
    """The reference frame with the scan's signal and the two exit flags per row."""
    ref = ref_frame(secs).reset_index(drop=True)
    ref["signal"] = [ref_signal(r, "mid") for r in ref.to_dict("records")]
    ref["daily_red"] = ref["bx_daily"] < 0
    ref["weekly_down"] = ref["bx_weekly"] < ref["bx_weekly_prev"]
    return ref


def nights(ref: pd.DataFrame, security_id: str, column: str, lo: date, hi: date) -> list[date]:
    """Sessions in [lo, hi] where ``column`` is True for the security."""
    part = ref[(ref["security_id"] == security_id) & ref[column]]
    return [d for d in part["session_date"] if lo <= d <= hi]


def first_night(ref: pd.DataFrame, security_id: str, column: str, lo: date) -> date:
    found = nights(ref, security_id, column, lo, CAL[-1])
    assert found, f"fixture: no {column} night of {security_id} from {lo}"
    return found[0]


def bar_of(sec: Sec, day: date) -> dict[str, Any]:
    """The stored (raw) bar of ``day``."""
    [row] = [b for b in sec.bars() if b["session_date"] == day]
    return row


def hand_atr(
    rows: Sequence[Mapping[str, Any]],
    day: date,
    days: int,
    splits: Sequence[tuple[date, float]] = (),
) -> float:
    """Mean of the last ``days`` true ranges through ``day`` (its row included), each row before
    a split's ex-date divided by that split's factor (new shares per old share)."""
    ordered = sorted((r for r in rows if r["session_date"] <= day), key=lambda r: r["session_date"])
    adjusted = []
    for r in ordered:
        k = math.prod(f for ex, f in splits if r["session_date"] < ex)
        adjusted.append((r["high"] / k, r["low"] / k, r["close"] / k))
    assert len(adjusted) >= days + 1, "fixture: too few bars for the ATR"
    window = adjusted[-(days + 1) :]
    trs = [
        max(h - lo, abs(h - prev_c), abs(lo - prev_c))
        for (h, lo, _), (_, _, prev_c) in zip(window[1:], window[:-1], strict=True)
    ]
    return sum(trs) / days


def expected_shares(
    sizing: BxSizingConfig, *, equity: float, close: float, atr: float, gross: float = 0.0
) -> int:
    return bx_shares(equity=equity, close=close, atr=atr, sizing=sizing, gross_exposure=gross)


def buy(price: float, bps: float = BPS) -> float:
    return price * (1 + bps / 1e4)


def sell(price: float, bps: float = BPS) -> float:
    return price * (1 - bps / 1e4)
