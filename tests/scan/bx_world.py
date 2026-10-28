"""A small market store for the B-Xtrender scan tests, written with the store tests' writers
(raw Parquet bars, security master, instrument classes, splits, spin-off records), plus a
reference computation of the entry signal that uses only ``period_bars`` / ``bx_histogram``.

Provenance of the fake data:
- Calendar: weekdays 2023-01-03 .. 2024-06-28 minus the NYSE holidays of those years
  (documentation: NYSE holiday list; Good Friday 2023-04-07 and 2024-03-29 included). The store
  has no exchange calendar; the sessions are whatever days have bars, as on the real store.
- Prices: ``tests.features.test_bx.walk`` (deterministic random walk, made up). vwap = close so
  dollar volume = close * volume.
- Daily bar availability 20:00 New York of its session, split and spin-off record availability
  00:00 New York of the ex-date: the ingest rules (asof.ingest.bars, corporate_actions), as in
  tests/store. Records with another available_at are constructed (a late or early feed).
- Instrument classes come from the class table keyed by master segment, as in tests/store.
- Volumes are made up: 5M shares a day at ~$50 is ~$250M a day (inside the $50M tier), 1,000
  shares a day is ~$50k (outside).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tests.features.test_bx import reference, walk
from tests.scan.world import close_of
from tests.store.audit_world import bar as raw_bar
from tests.store.test_market import known
from tests.store.test_spinoffs import spin_row, write_bars, write_other_rows
from tests.store.test_split_events import split_row, write_split_rows
from tests.store.test_universe import Seg, write_classes, write_master

__all__ = ["close_of", "known", "spin_row", "split_row"]

NYSE_HOLIDAYS = {
    date(2023, 1, 16),
    date(2023, 2, 20),
    date(2023, 4, 7),
    date(2023, 5, 29),
    date(2023, 6, 19),
    date(2023, 7, 4),
    date(2023, 9, 4),
    date(2023, 11, 23),
    date(2023, 12, 25),
    date(2024, 1, 1),
    date(2024, 1, 15),
    date(2024, 2, 19),
    date(2024, 3, 29),
    date(2024, 5, 27),
    date(2024, 6, 19),
}
CAL: list[date] = [
    d.date() for d in pd.bdate_range("2023-01-03", "2024-06-28") if d.date() not in NYSE_HOLIDAYS
]
LIQUID_VOLUME = 5_000_000
THIN_VOLUME = 1_000


@dataclass(frozen=True)
class Sec:
    """One security trading ``walk(seed, start)`` on CAL[first : last + 1].

    ``old_symbol`` = (symbol, rename day): bars before that day carry the old symbol and the
    master has two segments. ``raw_scale``: (lo, hi, k) multiplies the stored OHLC of sessions
    lo <= d < hi by k (lo / hi None = open-ended): a split's pre-ex prices or a spin-off
    parent's drop. ``volumes`` overrides the volume of given sessions."""

    security_id: str
    symbol: str
    seed: int
    start: float = 50.0
    cls: str = "common"
    volume: int = LIQUID_VOLUME
    first: int = 0
    last: int | None = None
    old_symbol: tuple[str, date] | None = None
    raw_scale: tuple[tuple[date | None, date | None, float], ...] = ()
    volumes: Mapping[date, int] = field(default_factory=dict)

    @property
    def days(self) -> list[date]:
        end = len(CAL) if self.last is None else self.last + 1
        return CAL[self.first : end]

    def symbol_on(self, day: date) -> str:
        if self.old_symbol is not None and day < self.old_symbol[1]:
            return self.old_symbol[0]
        return self.symbol

    def prices(self) -> pd.DataFrame:
        """The walk as daily rows (security_id, symbol, session_date, OHLC): the series a
        correct reader sees after every adjustment the stored records describe."""
        rows = walk(self.security_id, self.days, seed=self.seed, start=self.start)
        out = pd.DataFrame.from_records(rows)
        out["symbol"] = [self.symbol_on(d) for d in out["session_date"]]
        return out

    def bars(self, cut: date | None = None) -> list[dict[str, Any]]:
        out = []
        for r in self.prices().to_dict("records"):
            d = r["session_date"]
            if cut is not None and d > cut:
                continue
            k = 1.0
            for lo, hi, m in self.raw_scale:
                if (lo is None or d >= lo) and (hi is None or d < hi):
                    k *= m
            out.append(
                raw_bar(
                    self.symbol_on(d),
                    d,
                    close=r["close"] * k,
                    open=r["open"] * k,
                    high=r["high"] * k,
                    low=r["low"] * k,
                    vwap=r["close"] * k,
                    volume=self.volumes.get(d, self.volume),
                )
            )
        return out

    def segs(self) -> list[Seg]:
        first = CAL[self.first]
        if self.old_symbol is not None:
            old, day = self.old_symbol
            return [
                Seg(self.security_id, old, valid_to=day, end_available_at=known(day), cls=self.cls),
                Seg(
                    self.security_id,
                    self.symbol,
                    valid_from=day,
                    available_at=known(day),
                    cls=self.cls,
                ),
            ]
        if self.first > 0:  # a listing: its segment is known from its first session
            return [
                Seg(
                    self.security_id,
                    self.symbol,
                    valid_from=first,
                    available_at=known(first),
                    cls=self.cls,
                )
            ]
        return [Seg(self.security_id, self.symbol, cls=self.cls)]


def build(
    root: Path,
    secs: Sequence[Sec],
    *,
    splits: Sequence[dict[str, Any]] = (),
    spinoffs: Sequence[dict[str, Any]] = (),
    cut: date | None = None,
) -> None:
    """Write the store: every bar (only sessions <= ``cut`` when given), the master, the class
    table and the corporate-action files that have rows."""
    write_bars(root, [b for s in secs for b in s.bars(cut)])
    segs = [seg for s in secs for seg in s.segs()]
    write_master(root, segs)
    write_classes(root, segs)
    if splits:
        write_split_rows(root, splits)
    if spinoffs:
        write_other_rows(root, spinoffs)


def after(day: date, n: int) -> date:
    """The session ``n`` sessions after ``day`` on CAL."""
    return CAL[CAL.index(day) + n]


# -- the reference signal -------------------------------------------------------------------------

STATE = {"mid": "weekly", "long": "monthly"}


def ref_frame(secs: Sequence[Sec]) -> pd.DataFrame:
    """``reference`` over the securities' adjusted prices (no events)."""
    return reference(pd.concat([s.prices() for s in secs], ignore_index=True))


def ref_signal(
    row: Mapping[str, Any], variant: str, warm_daily: int = 1, warm_state: int = 1
) -> bool:
    state = STATE[variant]
    values = [row[f"bx_{state}"], row[f"bx_{state}_prev"], row["bx_daily"], row["bx_daily_prev"]]
    if any(np.isnan(float(v)) for v in values):
        return False
    return bool(
        values[0] > values[1]
        and values[2] > values[3]
        and row["daily_bars"] >= warm_daily
        and row[f"{state}_bars"] >= warm_state
    )


def ref_defined(row: Mapping[str, Any], variant: str) -> bool:
    state = STATE[variant]
    values = [row[f"bx_{state}"], row[f"bx_{state}_prev"], row["bx_daily"], row["bx_daily_prev"]]
    return not any(np.isnan(float(v)) for v in values)


def ref_row(ref: pd.DataFrame, security_id: str, day: date) -> dict[str, Any]:
    [i] = np.flatnonzero((ref["security_id"] == security_id) & (ref["session_date"] == day))
    return ref.iloc[int(i)].to_dict()


def signalling(ref: pd.DataFrame, day: date, variant: str) -> set[str]:
    rows = ref[ref["session_date"] == day]
    return {str(r["security_id"]) for _, r in rows.iterrows() if ref_signal(r.to_dict(), variant)}


def find_session(
    ref: pd.DataFrame,
    want: Mapping[tuple[str, str], bool],
    *,
    lo: int = 200,
    weekday: int | None = None,
) -> date:
    """The first session from CAL[lo] on where every (security_id, variant) signals or not as
    ``want`` says (and, with ``weekday``, falls on that weekday). Fixture guard: one exists."""
    for day in CAL[lo:]:
        if weekday is not None and day.weekday() != weekday:
            continue
        if all(
            (sid in signalling(ref, day, variant)) == flag for (sid, variant), flag in want.items()
        ):
            return day
    raise AssertionError(f"fixture: no session satisfies {dict(want)}")


def previous_friday(day: date) -> date:
    return day - timedelta(days=(day.weekday() - 4) % 7 or 7)
