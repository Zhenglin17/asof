"""The entry alone against random entries: every real entry and its draws held a fixed number of
sessions.

``null.py`` keeps each real leg's exit date, and a B-Xtrender exit follows the real security's
own path (its daily BX turning red, its weekly bar shrinking): the random legs inherit an exit
timed to another stock, so the null measures the entry and the exit together. Here each real
entry (the halves of one position are one entry) buys the entry session's open and sells the
open ``hold`` sessions later, and so does each draw from that day's pool: the only difference
left is which security was bought. Prices, splits, spin-offs, slippage and draws are the null's;
an entry draws the same securities for every hold (its key does not name the hold), so holds
compare entry by entry. Every (run, hold, null pool) cell is one member of the max-statistic
family; the "tht" pool keeps the scan pool's THT-green members.

Entries are the run's filled long decisions, not its closed trades: a position still open at
the replay's end has no trade row, and leaving it out would choose entries by how they went on.
B-Xtrender buys the open after its decision, so the entry session follows from the decision;
every visible trade must agree. The interval resamples blocks of ``hold`` sessions: entries of
neighbouring sessions hold through mostly the same days and are not independent.

Only development dates count: an entry must be in dev and so must its fixed exit, and prices are
read as known at dev's last close. A real security without the exit open (a halt, a delisting)
drops the entry from both sides; it is counted, not hidden.
"""

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Literal

import duckdb
import numpy as np
from sqlmodel import Session

from asof.evolve.null import (
    ET,
    Configs,
    Panel,
    Span,
    VariantResult,
    _leg_return,
    build_panel,
    config_of,
    max_statistic,
    null_returns,
)
from asof.evolve.protocol import Window
from asof.store.market import visible_sessions
from asof.store.models import DecisionKind, Direction, Run
from asof.store.queries import visible_decisions, visible_trades


@dataclass(frozen=True)
class Entry:
    security_id: str
    entry_session: date


@dataclass(frozen=True)
class FixedHold:
    spans: list[Span]  # ret: the real security's return over the fixed hold
    beyond_window: int  # fixed exit after ``last`` or past the calendar: dropped
    unpriced: int  # real security without the entry or exit open: dropped from both sides


@dataclass(frozen=True)
class HoldResult:
    run_id: int
    hold: int
    result: VariantResult  # corrected over every (run, hold, null pool) cell
    ci: tuple[float, float]  # per-entry excess over its draws, resampled by entry session
    sd_ratio: float  # spread of the real returns / spread of every draw
    beyond_window: int
    unpriced: int
    null_pool: str = "plain"  # "tht": draws only from the pool's THT-green members


NullPool = Literal["plain", "tht"]


def entries_of(spans: Sequence[Span]) -> list[Entry]:
    """One entry per security and entry session."""
    found = {Entry(s.key.rsplit(":", 2)[0], s.entry_session) for s in spans}
    return sorted(found, key=lambda e: (e.entry_session, e.security_id))


def entries_of_run(
    session: Session, run_id: int, calendar: Sequence[date], *, as_of: datetime
) -> list[Entry]:
    """The run's filled long decisions visible at ``as_of``, each entering the first session
    after its evening; a decision on the calendar's last session has no entry yet."""
    run = session.get(Run, run_id)
    if run is None:
        raise LookupError(f"no run {run_id}")
    found = set()
    decisions = visible_decisions(
        session, as_of, kind=DecisionKind.TRADE, strategy_id=run.strategy_id
    )
    for d in decisions:
        if (
            d.run_id != run_id
            or d.direction != Direction.LONG.value
            or d.security_id is None
            or "cancelled" in d.payload
        ):
            continue
        evening = d.as_of.astimezone(ET).date()
        entry = next((day for day in calendar if day > evening), None)
        if entry is not None:
            found.add(Entry(d.security_id, entry))
    for t in visible_trades(session, as_of, strategy_id=run.strategy_id):
        day = t.entry_at.astimezone(ET).date()
        if t.run_id == run_id and Entry(t.security_id, day) not in found:
            raise ValueError(
                f"{t.security_id} ({t.symbol}) entered on {day} "
                "without a decision the evening before"
            )
    return sorted(found, key=lambda e: (e.entry_session, e.security_id))


def fixed_hold(
    entries: Sequence[Entry],
    panel: Panel,
    calendar: Sequence[date],
    *,
    hold: int,
    last: date,
    slippage_bps: float,
) -> FixedHold:
    """Each entry bought at its session's open and sold at the open ``hold`` sessions later; the
    window is checked before the prices, so an exit after ``last`` is never priced."""
    if hold < 1:
        raise ValueError(f"a hold is at least one session, not {hold}")
    index = {d: i for i, d in enumerate(calendar)}
    bps = slippage_bps / 10_000
    spans, beyond, unpriced = [], 0, 0
    for e in entries:
        pool = panel.pools.get(e.entry_session)
        if pool is None:
            raise ValueError(f"no pool for entry session {e.entry_session}")
        if e.security_id not in pool:
            raise ValueError(
                f"{e.security_id} entered on {e.entry_session} outside that day's pool"
            )
        j = index[e.entry_session] + hold
        if j >= len(calendar) or calendar[j] > last:
            beyond += 1
            continue
        opens = panel.open.get(e.security_id, {})
        if e.entry_session not in opens:
            raise ValueError(
                f"{e.security_id} has no open on {e.entry_session}, the session it was bought"
            )
        if calendar[j] not in opens:
            unpriced += 1
            continue
        span = Span(
            key=f"{e.security_id}:{e.entry_session.isoformat()}:fixed",
            entry_session=e.entry_session,
            exit_session=calendar[j],
            exit_at="open",
            ret=0.0,
        )
        real = _leg_return(panel, e.security_id, span, bps)
        spans.append(Span(span.key, span.entry_session, span.exit_session, "open", real))
    return FixedHold(spans, beyond, unpriced)


def cluster_ci(
    excess: np.ndarray,
    clusters: Sequence[Hashable],
    *,
    draws: int,
    seed: int,
    level: float = 0.90,
) -> tuple[float, float]:
    """A percentile bootstrap of the mean resampling whole clusters: entries that hold through
    the same days move with that market together, so resampling them one by one would narrow
    the interval. The resampling does not depend on ``level``."""
    values = np.asarray(excess, dtype=float)
    if len(values) == 0 or len(values) != len(clusters):
        raise ValueError(f"{len(values)} values for {len(clusters)} cluster labels")
    keys = list(dict.fromkeys(clusters))
    if len(keys) == 1:
        mean = float(values.mean())
        return mean, mean
    position = {k: i for i, k in enumerate(keys)}
    label = np.array([position[c] for c in clusters])
    sums = np.bincount(label, weights=values, minlength=len(keys))
    counts = np.bincount(label, minlength=len(keys))
    picks = np.random.default_rng(seed).integers(0, len(keys), size=(draws, len(keys)))
    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    tail = (1 - level) / 2
    return float(np.quantile(means, tail)), float(np.quantile(means, 1 - tail))


def entry_test(
    session: Session,
    con: duckdb.DuckDBPyConnection,
    root: Path,
    run_ids: Sequence[int],
    *,
    cfg: Configs,
    holds: Sequence[int],
    dev: Window,
    permutations: int,
    seed: int,
    bootstrap: int = 2000,
    null_pools: Sequence[NullPool] = ("plain",),
) -> list[HoldResult]:
    """Each run's entries in dev, held every one of ``holds`` sessions, against draws from each
    of ``null_pools``: "plain" the scan's pool, "tht" its THT-green members (does the BX timing
    add anything among stocks the trend filter already passed). Every (run, hold, pool) is one
    cell of one family; sorted by (run, hold, pool)."""
    if not holds or len(set(holds)) != len(holds) or min(holds) < 1:
        raise ValueError(f"holds must be distinct and at least one session: {list(holds)}")
    if (
        not null_pools
        or len(set(null_pools)) != len(null_pools)
        or not set(null_pools) <= {"plain", "tht"}
    ):
        raise ValueError(f"null pools must be distinct, of plain and tht: {list(null_pools)}")
    for r in run_ids:
        if config_of(cfg, r).strategy.universe != "liquid_common":
            raise ValueError(
                f"run {r}: only B-Xtrender runs, whose entry is the open after the decision; "
                f"universe {config_of(cfg, r).strategy.universe!r}"
            )
        scan = getattr(config_of(cfg, r), "scan", None)
        if "tht" in null_pools and getattr(scan, "tht_filter", None) is None:
            raise ValueError(f"run {r}: the tht null pool needs a scan with a tht_filter")
    as_of = datetime.combine(dev.end, time(20), tzinfo=ET)
    calendar = visible_sessions(con, root, "1Day", as_of)
    index = {d: i for i, d in enumerate(calendar)}
    entries = {}
    for r in run_ids:
        # the real exit does not matter here: a position still open after dev is an entry in it
        entries[r] = [
            e
            for e in entries_of_run(session, r, calendar, as_of=as_of)
            if dev.start <= e.entry_session <= dev.end
        ]
        if not entries[r]:
            raise ValueError(f"run {r} has no entry within dev {dev.start} .. {dev.end}")
    real: dict[str, np.ndarray] = {}
    null: dict[str, np.ndarray] = {}
    parts: dict[tuple[int, int, str], FixedHold] = {}
    for r in run_ids:
        c = config_of(cfg, r)
        # build_panel reads only the entry sessions of its spans
        sessions = [Span(f"{e.security_id}:{e.entry_session}:fixed", e.entry_session,
                         e.entry_session, "open", 0.0) for e in entries[r]]  # fmt: skip
        bps = c.execute.fills.slippage_bps
        for kind in null_pools:
            panel = build_panel(session, con, root, sessions, cfg=c, as_of=as_of,
                                pool_tht=kind == "tht")  # fmt: skip
            for h in holds:
                fh = fixed_hold(entries[r], panel, calendar, hold=h, last=dev.end,
                                slippage_bps=bps)  # fmt: skip
                if not fh.spans:
                    raise ValueError(f"run {r}: no entry in dev has a priced exit {h} sessions on")
                key = f"{r}:{h}:{kind}"
                real[key] = np.array([s.ret for s in fh.spans])
                null[key] = null_returns(
                    fh.spans, panel, permutations=permutations, seed=seed, slippage_bps=bps
                )
                parts[(r, h, kind)] = fh
    family = max_statistic(real, null)
    out = []
    for r, h, kind in sorted(parts):
        key = f"{r}:{h}:{kind}"
        fh, values, drawn = parts[(r, h, kind)], real[key], null[key]
        out.append(
            HoldResult(
                run_id=r,
                hold=h,
                null_pool=kind,
                result=family[key],
                ci=cluster_ci(
                    values - drawn.mean(axis=0),
                    [index[s.entry_session] // h for s in fh.spans],
                    draws=bootstrap,
                    seed=seed,
                ),
                sd_ratio=float(np.std(values) / np.std(drawn)),
                beyond_window=fh.beyond_window,
                unpriced=fh.unpriced,
            )
        )
    return out
