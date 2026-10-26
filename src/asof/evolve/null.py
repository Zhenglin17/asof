"""The random-entry null: would picking securities at random have done as well?

Every real leg keeps its dates and its kind of exit; only the security is replaced by one drawn
uniformly from the strategy's candidate pool of that day: what its scanner could have returned
the evening before entry without the signal (for the earnings gap: one security per 8-K filer,
over the price and dollar-volume thresholds of scan.yaml). A drawn leg buys the entry
session's open and sells the exit session's open or close, with the replay's slippage on both
sides and splits in between applied, so real and random legs are priced the same way. Repeating
the draw K times gives what luck alone earns per leg in the same market and the same pool: a
bull market or a watchlist picked with hindsight lifts both sides equally.

A family of variants (strategy versions) is corrected with the max statistic: each variant's
mean is standardized against its own null, and its adjusted p-value counts the permutations in
which the best of all variants' null z-scores reached it. Draws are keyed by the leg's identity
(security, entry session, leg), not by its position, and run over the whole pool, redrawing past
a member without prices: two variants holding the same leg draw the same security whenever it
is priced for both exits, so similar variants move together under the null and the correction
stays narrow; an exact duplicate costs nothing.

Prices are results, not decisions: they are read as known at the window's last close, the same
reading as the benchmarks. Dividends are left out on both sides (the replay books none).
"""

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
from sqlmodel import Session, col, select

from asof.replay.cache import BarCache
from asof.scan import earnings_gap
from asof.scan.earnings_gap import lookback_sessions
from asof.store.market import visible_bars, visible_master, visible_sessions, visible_splits
from asof.store.models import Run, Trade
from asof.store.universe import filer_securities
from asof.strategy import StrategyConfig

ET = ZoneInfo("America/New_York")
_CANDIDATES = 16  # draws per permutation before falling back to its own stream
POOLS = {"watchlist_with_documents": earnings_gap.pool}


@dataclass(frozen=True)
class Span:
    """One real leg as the null sees it."""

    key: str  # f"{security_id}:{entry_session}:{leg}"
    entry_session: date
    exit_session: date
    exit_at: Literal["open", "close"]
    ret: float  # pnl / (shares * entry_price), net of the replay's slippage


@dataclass(frozen=True)
class Panel:
    sessions: tuple[date, ...]
    open: Mapping[str, Mapping[date, float]]  # security_id -> session -> raw open
    close: Mapping[str, Mapping[date, float]]
    splits: Mapping[str, Sequence[tuple[date, float]]]  # security_id -> (ex_date, new/old)
    pools: Mapping[date, frozenset[str]]  # entry session -> security_ids a draw may pick


@dataclass(frozen=True)
class VariantResult:
    trades: int
    mean: float  # mean real leg return
    null_mean: float  # mean of the per-permutation means
    excess: float  # mean - null_mean
    z: float  # excess in standard deviations of the per-permutation means
    p_raw: float  # share of permutations whose mean reached the real one
    p_adjusted: float  # share whose best variant z reached this variant's z


def _key_seed(key: str) -> int:
    """A seed from the leg's identity that every process agrees on (``hash`` is salted)."""
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "big")


def _leg_return(panel: Panel, sid: str, span: Span, bps: float) -> float:
    exit_prices = panel.open if span.exit_at == "open" else panel.close
    factor = 1.0
    for ex_date, ratio in panel.splits.get(sid, ()):
        if span.entry_session < ex_date <= span.exit_session:
            factor *= ratio
    entry = panel.open[sid][span.entry_session]
    return exit_prices[sid][span.exit_session] * factor * (1 - bps) / (entry * (1 + bps)) - 1


def null_returns(
    spans: Sequence[Span], panel: Panel, *, permutations: int, seed: int, slippage_bps: float
) -> np.ndarray:
    """Shape (permutations, len(spans)): column j holds span j's return with a random security
    in place of the real one, one draw per permutation."""
    if permutations < 1:
        raise ValueError("permutations must be at least 1")
    bps = slippage_bps / 10_000
    out = np.empty((permutations, len(spans)))
    for j, span in enumerate(spans):
        pool = panel.pools.get(span.entry_session)
        if pool is None:
            raise ValueError(f"{span.key}: no pool for entry session {span.entry_session}")
        members = sorted(pool)
        exit_prices = panel.open if span.exit_at == "open" else panel.close
        priced = [
            span.entry_session in panel.open.get(sid, {})
            and span.exit_session in exit_prices.get(sid, {})
            for sid in members
        ]
        if not any(priced):
            raise ValueError(
                f"{span.key}: no security in the pool of {span.entry_session} has prices "
                f"for the entry and the {span.exit_at} of {span.exit_session}"
            )
        returns = [
            _leg_return(panel, sid, span, bps) if ok else math.nan
            for sid, ok in zip(members, priced, strict=True)
        ]
        # Each permutation walks its own fixed row of draws from the whole pool and takes the
        # first priced member: uniform over the priced ones, and two variants holding this leg
        # with different exits still get the same security whenever it is priced for both. A
        # row is never shifted by another row's redraws.
        key_seed = _key_seed(span.key)
        rows = np.random.default_rng([seed, key_seed]).integers(
            0, len(members), size=(permutations, _CANDIDATES)
        )
        for k in range(permutations):
            pick = next((int(i) for i in rows[k] if priced[i]), None)
            if pick is None:
                spare = np.random.default_rng([seed, key_seed, k])
                pick = int(spare.integers(0, len(members)))
                while not priced[pick]:
                    pick = int(spare.integers(0, len(members)))
            out[k, j] = returns[pick]
    return out


def max_statistic(
    real: Mapping[str, np.ndarray], null: Mapping[str, np.ndarray]
) -> dict[str, VariantResult]:
    """``real[v]`` (n_v,) and ``null[v]`` (K, n_v) per variant; every variant needs the same K."""
    if set(real) != set(null):
        raise ValueError("real and null must name the same variants")
    if not real:
        return {}
    counts = {null[v].shape[0] for v in null if null[v].ndim == 2}
    if len(counts) != 1 or any(null[v].ndim != 2 for v in null):
        raise ValueError("every variant needs a 2-d null with the same number of permutations")
    k = counts.pop()
    stats: dict[str, tuple[float, float, float, float, np.ndarray]] = {}
    for v, values in real.items():
        n = len(values)
        if n == 0:
            raise ValueError(f"variant {v} has no trades")
        if null[v].shape[1] != n:
            raise ValueError(f"variant {v}: null has {null[v].shape[1]} columns for {n} trades")
        means = null[v].mean(axis=1)
        centre, sd = float(means.mean()), float(means.std())
        if not sd > 0:
            raise ValueError(f"variant {v}: the null does not vary, there is nothing to test")
        mean = float(np.mean(values))
        p_raw = float((1 + np.sum(means >= mean)) / (1 + k))
        stats[v] = (mean, centre, (mean - centre) / sd, p_raw, (means - centre) / sd)
    best = np.max(np.stack([z_k for *_, z_k in stats.values()]), axis=0)
    return {
        v: VariantResult(
            trades=len(real[v]),
            mean=mean,
            null_mean=centre,
            excess=mean - centre,
            z=z,
            p_raw=p_raw,
            p_adjusted=float((1 + np.sum(best >= z)) / (1 + k)),
        )
        for v, (mean, centre, z, p_raw, _) in stats.items()
    }


def spans_of_run(session: Session, run_id: int) -> list[Span]:
    if session.get(Run, run_id) is None:
        raise LookupError(f"no run {run_id}")
    stmt = select(Trade).where(Trade.run_id == run_id).order_by(col(Trade.entry_at), col(Trade.id))
    spans = []
    for t in session.exec(stmt).all():
        entry = t.entry_at.astimezone(ET)
        exit_ = t.exit_at.astimezone(ET)
        spans.append(
            Span(
                key=f"{t.security_id}:{entry.date().isoformat()}:{t.leg}",
                entry_session=entry.date(),
                exit_session=exit_.date(),
                # 09:30 is a fill at the open; a 16:00 stop and a 20:00 exit price at the close.
                exit_at="open" if exit_.time() == time(9, 30) else "close",
                ret=t.pnl / (t.shares * t.entry_price),
            )
        )
    return spans


def _close_of(day: date) -> datetime:
    return datetime.combine(day, time(20), tzinfo=ET)


def build_panel(
    session: Session,
    con: duckdb.DuckDBPyConnection,
    root: Path,
    spans: Sequence[Span],
    *,
    cfg: StrategyConfig,
    as_of: datetime,
) -> Panel:
    """Pools as the scanner saw them the evening before each entry; prices and splits as known
    at ``as_of``."""
    if cfg.strategy.universe not in POOLS:
        raise ValueError(
            f"unknown universe {cfg.strategy.universe!r}; known: {', '.join(sorted(POOLS))}"
        )
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    calendar = visible_sessions(con, root, "1Day", as_of)
    index = {d: i for i, d in enumerate(calendar)}
    entries = sorted({s.entry_session for s in spans})
    for entry in entries:
        i = index.get(entry)
        if i is None:
            raise ValueError(f"entry session {entry} is not a visible session at {as_of}")
        if i == 0:
            raise ValueError(f"entry session {entry} has no session before it to decide on")
    if not entries:
        return Panel(sessions=(), open={}, close={}, splits={}, pools={})

    # Every evening's pool reads its scan window; one cache serves them all, each read resolved
    # with the master as known that evening.
    window_start = calendar[max(0, index[entries[0]] - 2 - lookback_sessions(cfg))]
    filers = filer_securities(session, con, root, as_of)
    cache = BarCache(
        con,
        root,
        symbols=[r["symbol"] for r in visible_master(con, root, as_of).to_pylist()
                 if r["security_id"] in filers],
        start=datetime.combine(window_start, time(0), tzinfo=ET),
        end=as_of,
    )  # fmt: skip
    pool_of = POOLS[cfg.strategy.universe]
    pools = {
        entry: pool_of(
            session,
            con,
            root,
            _close_of(calendar[index[entry] - 1]),
            cfg,
            sessions=calendar,
            bars_at=cache.bars_at,
        )
        for entry in entries
    }

    first = min(pools)
    pooled = frozenset().union(*pools.values())
    master = [r for r in visible_master(con, root, as_of).to_pylist() if r["security_id"] in pooled]
    symbols = sorted({r["symbol"] for r in master})
    opens: dict[str, dict[date, float]] = {}
    closes: dict[str, dict[date, float]] = {}
    bars = visible_bars(
        con,
        root,
        "1Day",
        as_of,
        symbols=symbols,
        start=datetime.combine(first, time(0), tzinfo=ET),
        end=as_of,
    )
    for r in bars.to_pylist():
        if r["security_id"] in pooled:
            opens.setdefault(r["security_id"], {})[r["session_date"]] = r["open"]
            closes.setdefault(r["security_id"], {})[r["session_date"]] = r["close"]

    # A split row belongs to the security that owned its symbol on the ex-date; a row stored
    # twice is one event.
    splits: dict[str, set[tuple[date, float]]] = {}
    for s in visible_splits(con, root, as_of, symbols).to_pylist():
        for seg in master:
            if (
                seg["symbol"] == s["symbol"]
                and seg["valid_from"] <= s["ex_date"]
                and (seg["valid_to"] is None or s["ex_date"] < seg["valid_to"])
            ):
                ratio = float(s["new_rate"]) / float(s["old_rate"])
                splits.setdefault(seg["security_id"], set()).add((s["ex_date"], ratio))
    return Panel(
        sessions=tuple(d for d in calendar if d >= first),
        open=opens,
        close=closes,
        splits={sid: sorted(rows) for sid, rows in splits.items()},
        pools=pools,
    )


def null_test(
    session: Session,
    con: duckdb.DuckDBPyConnection,
    root: Path,
    run_ids: Sequence[int],
    *,
    cfg: StrategyConfig,
    permutations: int,
    seed: int,
) -> dict[int, VariantResult]:
    """Each run is one variant of the family; the panel is read once, as known at the last exit
    of any of them. ``cfg`` gives the pool's thresholds and the slippage."""
    if cfg.strategy.universe not in POOLS:
        raise ValueError(f"unknown universe {cfg.strategy.universe!r}")
    spans = {r: spans_of_run(session, r) for r in run_ids}
    for r, legs in spans.items():
        if not legs:
            raise ValueError(f"run {r} has no closed trades to test")
    every = [s for legs in spans.values() for s in legs]
    as_of = _close_of(max(s.exit_session for s in every))
    panel = build_panel(session, con, root, every, cfg=cfg, as_of=as_of)
    real = {str(r): np.array([s.ret for s in legs]) for r, legs in spans.items()}
    null = {
        str(r): null_returns(
            legs,
            panel,
            permutations=permutations,
            seed=seed,
            slippage_bps=cfg.execute.fills.slippage_bps,
        )
        for r, legs in spans.items()
    }
    results = max_statistic(real, null)
    return {r: results[str(r)] for r in spans}
