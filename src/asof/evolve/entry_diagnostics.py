"""Diagnostics around a B-Xtrender entry test: numbers to read beside it, not cells of its family.

- ``flip_stats``: how often the THT colour turns, and how many green spells die young (a choppy
  market flips the filter back and forth).
- ``reentry_flags`` / ``closes_of_run``: entries that buy a security back within a few sessions
  of selling it. B-Xtrender re-enters a security only once its position is fully sold, so the
  latest close of the same security before an entry is that position's last leg.
- ``tertile_of``: a night's candidates split by their scan rank, to ask whether the ranking
  (largest weekly rise first) picks better than the bottom of the same list.
- ``group_difference``: one group's mean excess minus the other's, with an interval that
  resamples whole clusters of overlapping holds.
"""

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sqlmodel import Session

from asof.evolve.entry import Entry, cluster_ci, entries_of_run, fixed_hold
from asof.evolve.null import ET, Panel, Span, _draws, _leg_return, build_panel
from asof.evolve.protocol import Window
from asof.scan.bx_trender import BxScanConfig, pool_colors, scan_range
from asof.store.market import visible_sessions
from asof.store.models import Run
from asof.store.queries import visible_trades
from asof.strategy import AnyStrategyConfig, BxStrategyConfig

SESSIONS_PER_YEAR = 252


@dataclass(frozen=True)
class FlipStats:
    flips_per_year: float  # per security-year of sessions with a colour
    short_green_share: float  # completed green spells shorter than ``short``; NaN without any
    by_month: pd.DataFrame  # month, flips, security_sessions


def flip_stats(colors: pd.DataFrame, *, short: int = 5) -> FlipStats:
    """``colors``: security_id, session_date, tht_color. Rows without a colour (None or NaN)
    are skipped: they neither break a spell nor count as a session of it. A flip is a colour
    unlike the security's previous one, counted in the month of the session showing it; a
    green spell is completed when a red follows it."""
    df = pd.DataFrame(colors[colors["tht_color"].isin(["green", "red"])])
    df = df.sort_values(["security_id", "session_date"]).reset_index(drop=True)
    df["month"] = [str(d)[:7] for d in df["session_date"]]
    previous = df.groupby("security_id", sort=False)["tht_color"].shift(1)
    df["flip"] = previous.notna() & (df["tht_color"] != previous)

    spells: list[int] = []  # lengths of completed green spells
    for _, part in df.groupby("security_id", sort=False):
        run = 0
        for c in part["tht_color"]:
            if c == "green":
                run += 1
            else:
                if run:
                    spells.append(run)
                run = 0
    by_month = (
        df.groupby("month", sort=True)
        .agg(flips=("flip", "sum"), security_sessions=("flip", "size"))
        .reset_index()
    )
    years = len(df) / SESSIONS_PER_YEAR
    return FlipStats(
        flips_per_year=float(df["flip"].sum() / years) if years else float("nan"),
        short_green_share=float(np.mean([s < short for s in spells])) if spells else float("nan"),
        by_month=by_month,
    )


def reentry_flags(
    entries: Sequence[Entry],
    closes: Sequence[tuple[str, date]],
    calendar: Sequence[date],
    *,
    within: int = 10,
) -> list[bool]:
    """For each entry: did the same security close a position 1 to ``within`` sessions before
    its entry session? A close on the entry session or later never counts."""
    index = {d: i for i, d in enumerate(calendar)}
    by_security: dict[str, list[int]] = {}
    for sid, day in closes:
        by_security.setdefault(sid, []).append(index[day])
    return [
        any(0 < index[e.entry_session] - x <= within for x in by_security.get(e.security_id, ()))
        for e in entries
    ]


def closes_of_run(session: Session, run_id: int, as_of: datetime) -> list[tuple[str, date]]:
    """Per position (one decision), the New York session of its latest leg exit visible at
    ``as_of``."""
    run = session.get(Run, run_id)
    if run is None:
        raise LookupError(f"no run {run_id}")
    last: dict[int, tuple[str, date]] = {}
    for t in visible_trades(session, as_of, strategy_id=run.strategy_id):
        if t.run_id != run_id:
            continue
        day = t.exit_at.astimezone(ET).date()
        if t.decision_id not in last or last[t.decision_id][1] < day:
            last[t.decision_id] = (t.security_id, day)
    return list(last.values())


def tertile_of(rank: int, n: int) -> int:
    """0 for the top third of a night's ``n`` candidates, 1 the middle, 2 the bottom."""
    if not 1 <= rank <= n:
        raise ValueError(f"rank {rank} outside 1..{n}")
    return 3 * (rank - 1) // n


def group_difference(
    excess: Sequence[float] | np.ndarray,
    flags: Sequence[bool] | np.ndarray,
    clusters: Sequence[Hashable],
    *,
    draws: int,
    seed: int,
    level: float = 0.90,
) -> tuple[float, float, float]:
    """Mean excess of the flagged entries minus the unflagged, and a percentile bootstrap
    interval resampling whole clusters (as ``cluster_ci``). A draw missing one of the groups
    is dropped; when every draw misses one the interval is NaN."""
    values = np.asarray(excess, dtype=float)
    flag = np.asarray(flags, dtype=bool)
    if not len(values) == len(flag) == len(clusters):
        raise ValueError(f"{len(values)} values, {len(flag)} flags, {len(clusters)} clusters")
    if flag.all() or not flag.any():
        raise ValueError("both groups need an entry")
    diff = float(values[flag].mean() - values[~flag].mean())
    keys = list(dict.fromkeys(clusters))
    if len(keys) == 1:
        return diff, diff, diff
    position = {k: i for i, k in enumerate(keys)}
    label = np.array([position[c] for c in clusters])
    n = len(keys)
    sums = np.stack([np.bincount(label, weights=values * g, minlength=n) for g in (flag, ~flag)])
    counts = np.stack(
        [np.bincount(label, weights=g.astype(float), minlength=n) for g in (flag, ~flag)]
    )
    picks = np.random.default_rng(seed).integers(0, n, size=(draws, n))
    s, c = sums[:, picks].sum(axis=2), counts[:, picks].sum(axis=2)
    ok = (c > 0).all(axis=0)
    if not ok.any():
        return diff, float("nan"), float("nan")
    means = s[0, ok] / c[0, ok] - s[1, ok] / c[1, ok]
    tail = (1 - level) / 2
    return diff, float(np.quantile(means, tail)), float(np.quantile(means, 1 - tail))


# -- over a run's development dates ---------------------------------------------------------------


@dataclass(frozen=True)
class GroupExcess:
    entries: int
    excess: float  # mean return minus the pool's mean, per entry
    ci: tuple[float, float]


@dataclass(frozen=True)
class Reentries:
    entries: int
    reentries: int
    difference: tuple[float, float, float]  # re-entries minus the rest: diff, lo, hi


@dataclass(frozen=True)
class RankTertiles:
    nights: int
    unopened: int  # candidates without an open the next session (a halt): never bought
    dropped: int  # fixed exit after dev or without an exit open
    tertiles: tuple[GroupExcess, GroupExcess, GroupExcess]  # top, middle, bottom
    top_minus_bottom: tuple[float, float, float]


def pool_excess(
    entries: Sequence[Entry], panel: Panel, calendar: Sequence[date], *, hold: int, last: date,
    slippage_bps: float,
) -> tuple[list[Entry], np.ndarray]:  # fmt: skip
    """The entries with a priced fixed hold and each one's return minus the mean of every
    priced member of its pool over the same hold: what a random draw earns on average."""
    fh = fixed_hold(entries, panel, calendar, hold=hold, last=last, slippage_bps=slippage_bps)
    bps = slippage_bps / 10_000
    means: dict[date, float] = {}
    kept, out = [], []
    for s in fh.spans:
        if s.entry_session not in means:
            members, priced, _ = _draws(s, panel, permutations=1, seed=0)
            means[s.entry_session] = float(np.mean(
                [_leg_return(panel, m, s, bps) for m, ok in zip(members, priced, strict=True) if ok]
            ))  # fmt: skip
        kept.append(Entry(s.key.rsplit(":", 2)[0], s.entry_session))
        out.append(s.ret - means[s.entry_session])
    return kept, np.array(out)


def _clusters(entries: Sequence[Entry], calendar: Sequence[date], hold: int) -> list[int]:
    index = {d: i for i, d in enumerate(calendar)}
    return [index[e.entry_session] // hold for e in entries]


def _panel(
    session: Session, con: duckdb.DuckDBPyConnection, root: Path, entries: Sequence[Entry],
    cfg: AnyStrategyConfig, as_of: datetime,
) -> Panel:  # fmt: skip
    spans = [Span(f"{e.security_id}:{e.entry_session}:fixed", e.entry_session, e.entry_session,
                  "open", 0.0) for e in entries]  # fmt: skip
    return build_panel(session, con, root, spans, cfg=cfg, as_of=as_of)


def reentries(
    session: Session, con: duckdb.DuckDBPyConnection, root: Path, run_id: int, *,
    cfg: AnyStrategyConfig, dev: Window, hold: int, within: int, bootstrap: int, seed: int,
) -> Reentries:  # fmt: skip
    """The run's entries in dev that buy back a security sold 1 to ``within`` sessions before,
    against the rest, both measured over a fixed ``hold`` against their pool's mean."""
    as_of = datetime.combine(dev.end, time(20), tzinfo=ET)
    calendar = visible_sessions(con, root, "1Day", as_of)
    entries = [e for e in entries_of_run(session, run_id, calendar, as_of=as_of)
               if dev.start <= e.entry_session <= dev.end]  # fmt: skip
    panel = _panel(session, con, root, entries, cfg, as_of)
    kept, excess = pool_excess(entries, panel, calendar, hold=hold, last=dev.end,
                               slippage_bps=cfg.execute.fills.slippage_bps)  # fmt: skip
    flags = reentry_flags(kept, closes_of_run(session, run_id, as_of), calendar, within=within)
    nan = float("nan")
    # nothing to compare when no entry (or every one) is a re-entry: S1 has 3 of 1,817 in dev
    difference = (
        group_difference(excess, flags, _clusters(kept, calendar, hold), draws=bootstrap, seed=seed)
        if 0 < sum(flags) < len(flags)
        else (nan, nan, nan)
    )
    return Reentries(entries=len(kept), reentries=sum(flags), difference=difference)


def rank_tertiles(
    session: Session, con: duckdb.DuckDBPyConnection, root: Path, *,
    cfg: BxStrategyConfig, dev: Window, hold: int, bootstrap: int, seed: int,
) -> RankTertiles:  # fmt: skip
    """Every candidate of every dev night, room or not, entered the next open and held
    ``hold`` sessions, split by its rank into thirds of that night's list."""
    as_of = datetime.combine(dev.end, time(20), tzinfo=ET)
    calendar = visible_sessions(con, root, "1Day", as_of)
    index = {d: i for i, d in enumerate(calendar)}
    found = scan_range(con, root, dev.start, dev.end, cfg.scan)
    entries, third = [], {}
    for night, candidates in found.items():
        if index[night] + 1 >= len(calendar):
            continue
        for c in candidates:
            e = Entry(c.security_id, calendar[index[night] + 1])
            entries.append(e)
            third[e] = tertile_of(c.rank, len(candidates))
    panel = _panel(session, con, root, entries, cfg, as_of)
    opened = [e for e in entries if e.entry_session in panel.open.get(e.security_id, {})]
    kept, excess = pool_excess(opened, panel, calendar, hold=hold, last=dev.end,
                               slippage_bps=cfg.execute.fills.slippage_bps)  # fmt: skip
    group = np.array([third[e] for e in kept])
    clusters = _clusters(kept, calendar, hold)
    parts = []
    for g in (0, 1, 2):
        mask = group == g
        parts.append(GroupExcess(
            entries=int(mask.sum()),
            excess=float(excess[mask].mean()),
            ci=cluster_ci(excess[mask], [c for c, m in zip(clusters, mask, strict=True) if m],
                          draws=bootstrap, seed=seed),
        ))  # fmt: skip
    ends = group != 1
    top_bottom = group_difference(excess[ends], group[ends] == 0,
                                  [c for c, m in zip(clusters, ends, strict=True) if m],
                                  draws=bootstrap, seed=seed)  # fmt: skip
    return RankTertiles(nights=len(found), unopened=len(entries) - len(opened),
                        dropped=len(opened) - len(kept),
                        tertiles=(parts[0], parts[1], parts[2]),
                        top_minus_bottom=top_bottom)  # fmt: skip


def tht_flips(
    con: duckdb.DuckDBPyConnection, root: Path, *, cfg: BxScanConfig, dev: Window, short: int
) -> FlipStats:
    """``flip_stats`` of the THT colours of each dev night's pool members: only the nights a
    security is in the pool count, so a stock that leaves it and comes back joins its spells
    across the gap."""
    if cfg.tht_filter is None:
        raise ValueError("the THT flips need a scan with a tht_filter")
    return flip_stats(pool_colors(con, root, dev.start, dev.end, cfg), short=short)
