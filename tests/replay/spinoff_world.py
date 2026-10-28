"""GAPX held through a spin-off, for the replay and null spin-off tests (base store:
tests/scan/world.py; record and bar builders: tests/store/test_spinoffs.py).

GAPX gaps on D = X[0] as in tests/replay/scenario.py (ATR 2, key 54, hard 52, pre-earnings close
50, 125 shares). ``WDC_AFTER`` is the real WDC 2025-02-24 shape in round numbers: leg 1 in at
56.5, a 60.0 close trails the key to 57 (hard 55), then the ex-date X3 opens at 48.0 -- through
the raw hard stop -- because a quarter of a KIDZ share per GAPX share (first close 48.0) left the
company: factor (60 - 0.25 x 48) / 60 = 0.8. Adjusted, the key is 45.6 (hard 44.0); X5 closes
below the trailed zone and the leg leaves at X6's open, 45.0.

Provenance:
- Spin-off record rows: tests/store/test_spinoffs.spin_row (real raw_json shape; available_at
  00:00 New York of the ex-date, the ingest rule).
- Child bars: raw rows, visible 20:00 New York of their session (ingest rule) unless a test
  pushes one later; flat (open = high = low = close) so each test reads one number. Real case:
  SNDK's first bar 2025-02-24 close 48.6, next open 48.30 (bar store probe).
- Child master segments open on the child's first session, known from 00:00 New York that day
  (the rename / listing rule of tests/store/test_market.known).
"""

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlmodel import Session

from tests.replay.scenario import X
from tests.scan.world import Ohlcv, Seg, World, gap_world
from tests.store.test_market import known
from tests.store.test_spinoffs import px, spin_row, write_bars, write_other_rows
from tests.strategy.files import DEFAULTS, write_strategy

SPINOFF_MAX_WAIT = 5  # strategies/earnings_gap/execute.yaml data.spinoff_max_wait


def spin_strategy(parent: Path) -> Path:
    """The approved defaults with ``data.spinoff_max_wait`` spelled out, so the wait tests do not
    depend on what tests/strategy/files.py carries."""
    execute = copy.deepcopy(DEFAULTS["execute"])
    execute["data"] = {**execute["data"], "spinoff_max_wait": SPINOFF_MAX_WAIT}
    return write_strategy(parent, {"execute": execute})


PRE: list[Ohlcv | None] = [
    (56.5, 57.0, 56.0, 56.8, 1_200_000),  # X1: leg 1 in at 56.5
    (57.0, 60.5, 56.8, 60.0, 1_200_000),  # X2: close 60 -> key 57, zone 56.5 .. 57.5, hard 55
]
EX_BAR: Ohlcv = (48.0, 49.0, 47.5, 48.4, 2_000_000)  # the ex-date: open below the raw hard 55
HOLD: Ohlcv = (48.3, 48.8, 47.9, 48.5, 1_500_000)  # above the adjusted zone 45.2 .. 46.0
BREAK: Ohlcv = (46.0, 46.2, 45.0, 45.3, 1_500_000)  # key 46.1: below its zone 45.7 .. 46.5
EXIT: Ohlcv = (45.0, 45.5, 44.8, 45.2, 1_000_000)  # the break fills at this open
FILLER: Ohlcv = EXIT  # never reaches the scaled re-entry zone 39.6 .. 40.4
WDC_AFTER: list[Ohlcv | None] = [*PRE, EX_BAR, HOLD, BREAK, EXIT, *[FILLER] * 20]
EX = 3  # X[EX] is the ex-date
FACTOR = 0.8
CHILD_PROCEEDS = 125 * 0.25 * 48.0  # 1500.0, KIDZ sold at X4's open


@dataclass(frozen=True)
class Child:
    symbol: str  # bucket K: never shares a bars file with GAPX or OTHR
    ratio: float
    first: int  # X index of its first bar
    price: float  # flat from then on
    # X index -> price (None: no bar that session), overriding ``price``
    prices: Mapping[int, float | None] = field(default_factory=dict)
    # X index -> available_at, for a bar that reached the store late
    late: Mapping[int, datetime] = field(default_factory=dict)
    last: int | None = None  # X index of its last bar (default: the end of the world)


KIDZ = Child("KIDZ", 0.25, EX, 48.0)


def child_rows(child: Child, end: int) -> list[dict[str, Any]]:
    rows = []
    for i in range(child.first, (child.last if child.last is not None else end) + 1):
        price = child.prices.get(i, child.price)
        if price is None:
            continue
        rows.append(px(child.symbol, X[i], price, available_at=child.late.get(i)))
    return rows


def spin_world(
    tmp_path: Path,
    session: Session,
    *,
    after: Sequence[Ohlcv | None] = WDC_AFTER,
    children: Sequence[Child] = (KIDZ,),
    ex: int = EX,
    records: Sequence[dict[str, Any]] | None = None,
    record_at: datetime | None = None,
    **world: Any,
) -> World:
    """gap_world plus child bars, child master segments and one spin-off record per child
    (``records`` replaces them; [] writes none). ``world`` goes to gap_world."""
    segs = [
        Seg("S_" + c.symbol, c.symbol, None, valid_from=X[c.first], available_at=known(X[c.first]))
        for c in children
    ]
    w = gap_world(
        tmp_path, session, after=after, extra_segs=[*world.pop("extra_segs", ()), *segs], **world
    )
    end = len(after)
    rows = [row for c in children for row in child_rows(c, end)]
    if rows:
        write_bars(w.root, rows)
    if records is None:
        records = [
            spin_row("GAPX", c.symbol, X[ex], c.ratio, available_at=record_at) for c in children
        ]
    if records:
        write_other_rows(w.root, records)
    return w
