"""Earnings-gap scanner: an up-gap on the first session that opened after an earnings release.

Run at 20:00 New York, when the day's bar is final. The filing, not the gap, starts a candidate:
an 8-K whose items include an earnings item (2.02), mapped to its reaction session -- the first
session whose 09:30 open came after EDGAR accepted it. A release at 16:08 reacts the next day; a
release at 07:00 the same day. Everything is read through the visible_* functions at ``as_of``.
"""

import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import duckdb
import pyarrow as pa
from sqlmodel import Session

from asof.features.daily import daily_features
from asof.store.market import visible_bars, visible_master, visible_sessions, visible_splits
from asof.store.models import Document, Entity
from asof.store.queries import visible_documents
from asof.store.universe import filing_cik
from asof.strategy import StrategyConfig

ET = ZoneInfo("America/New_York")
OPEN = time(9, 30)
_ITEMS = re.compile(r"^\d+\.\d+(,\d+\.\d+)*$")


@dataclass(frozen=True)
class Candidate:
    entity_id: int
    security_id: str
    symbol: str
    document_ids: tuple[int, ...]
    available_at: datetime  # earliest release of the event
    reaction_session: date
    pre_earnings_close: float
    gap_open: float
    gap_high: float
    gap_low: float
    gap_close: float
    gap: float
    rvol: float
    atr: float
    avg_dollar_volume: float


def filing_items(title: str) -> frozenset[str]:
    """Items of an 8-K title as the EDGAR ingest writes it: "8-K 2.02,9.01 EX-99.1 ..."."""
    tokens = title.split()
    if len(tokens) < 2 or not tokens[0].startswith("8-K") or not _ITEMS.match(tokens[1]):
        return frozenset()
    return frozenset(tokens[1].split(","))


def session_open(day: date) -> datetime:
    return datetime.combine(day, OPEN, tzinfo=ET)


def reaction_session(available_at: datetime, sessions: Sequence[date]) -> date | None:
    """First session whose 09:30 New York open is strictly after ``available_at``."""
    if available_at.tzinfo is None:
        raise ValueError("available_at must be timezone-aware")
    for day in sessions:
        if session_open(day) > available_at:
            return day
    return None


def lookback_sessions(cfg: StrategyConfig) -> int:
    """Sessions before the reaction day its features read: the longest window plus the close
    before it (a true range needs the previous close)."""
    scan = cfg.scan
    return max(scan.atr_days, scan.rvol_days, scan.dollar_volume_days) + 1


def _earnings_events(
    session: Session, as_of: datetime, today: date, sessions: Sequence[date], cfg: StrategyConfig
) -> dict[int, list[Document]]:
    """Earnings 8-Ks whose reaction session is ``today``, grouped by entity."""
    index = sessions.index(today)
    since = session_open(sessions[index - 1]) if index > 0 else None
    items = set(cfg.scan.earnings_items)
    events: dict[int, list[Document]] = defaultdict(list)
    for doc in visible_documents(session, as_of, doc_type="8-K", since=since):
        if doc.entity_id is None or not items & filing_items(doc.title or ""):
            continue
        if reaction_session(doc.available_at, sessions[index:]) == today and (
            index == 0 or session_open(sessions[index - 1]) <= doc.available_at
        ):
            events[doc.entity_id].append(doc)
    return events


def _security_of(
    rows: Sequence[dict[str, Any]], ciks: set[int], ticker: str, today: date
) -> tuple[str, list[str]] | None:
    """The security one of ``ciks`` points to on ``today`` and every symbol it has used.

    A security is matched when any of its visible segments carries the CIK: after a rename the
    new segment's CIK link may become knowable weeks later (SQ -> XYZ, 2025-01-21; the XYZ link
    is dated 2025-03-19) while the security itself was known all along. Several securities under
    one CIK (share classes) resolve to the one trading under ``ticker``; otherwise none."""
    securities = {r["security_id"] for r in rows if r["cik"] is not None and r["cik"] in ciks}
    live = [
        r
        for r in rows
        if r["security_id"] in securities
        and r["valid_from"] <= today
        and (r["valid_to"] is None or today < r["valid_to"])
    ]
    chosen = {r["security_id"] for r in live}
    if len(chosen) > 1:
        chosen = {r["security_id"] for r in live if r["symbol"] == ticker}
    if len(chosen) != 1:
        return None
    [security_id] = chosen
    symbols = sorted({r["symbol"] for r in rows if r["security_id"] == security_id})
    return security_id, symbols


def scan(
    session: Session,
    con: duckdb.DuckDBPyConnection,
    root: Path,
    as_of: datetime,
    cfg: StrategyConfig,
    *,
    sessions: Sequence[date] | None = None,
) -> list[Candidate]:
    """Candidates whose reaction session is the latest session final at ``as_of``.

    ``sessions`` is the market calendar known at ``as_of`` (``visible_sessions``); the replay
    passes it in to avoid re-reading the whole store each evening.
    """
    if sessions is None:
        sessions = visible_sessions(con, root, "1Day", as_of)
    sessions = [d for d in sessions if datetime.combine(d, time(20), tzinfo=ET) <= as_of]
    if not sessions:
        return []
    today = sessions[-1]
    events = _earnings_events(session, as_of, today, sessions, cfg)
    if not events:
        return []

    master = visible_master(con, root, as_of).to_pylist()
    lookback = lookback_sessions(cfg)
    first = sessions[max(0, len(sessions) - 1 - lookback)]
    start = datetime.combine(first, time(0), tzinfo=ET)
    end = datetime.combine(today + timedelta(days=1), time(0), tzinfo=ET)
    scan_cfg = cfg.scan
    found: list[Candidate] = []
    for entity_id, docs in sorted(events.items()):
        entity = session.get(Entity, entity_id)
        if entity is None:
            continue
        ciks = {cik for d in docs if (cik := filing_cik(d.url)) is not None}
        if not ciks and entity.cik is not None:
            ciks = {entity.cik}
        resolved = _security_of(master, ciks, entity.ticker, today)
        if resolved is None:
            continue
        security_id, symbols = resolved
        splits = visible_splits(con, root, as_of, symbols).to_pylist()
        if any(first <= s["ex_date"] <= today for s in splits):
            continue  # raw prices across a split would fake the gap or the ATR
        bars = visible_bars(con, root, "1Day", as_of, symbols=symbols, start=start, end=end)
        frame = bars.to_pandas()
        frame = frame[frame["security_id"] == security_id]
        if frame.empty:
            continue
        features = daily_features(
            frame,
            atr_days=scan_cfg.atr_days,
            rvol_days=scan_cfg.rvol_days,
            dollar_volume_days=scan_cfg.dollar_volume_days,
        )
        row = features[features["session_date"] == today]
        if row.empty or len(features) < 2:
            continue
        if features["session_date"].iloc[-2] != sessions[-2]:
            continue  # no bar on the previous session: prev_close would be an older close
        r = row.iloc[-1]
        values = [r.prev_close, r.gap, r.rvol, r.atr, r.avg_dollar_volume]
        if any(v != v for v in values):  # NaN: not enough history
            continue
        if not (
            r.gap * 100 >= scan_cfg.min_gap_pct - 1e-9
            and r.rvol >= scan_cfg.min_rvol - 1e-9
            and r.close >= scan_cfg.min_price
            and r.avg_dollar_volume >= scan_cfg.min_avg_dollar_volume - 1e-6
        ):
            continue
        found.append(
            Candidate(
                entity_id=entity_id,
                security_id=security_id,
                symbol=str(r.symbol),
                document_ids=tuple(sorted(d.id for d in docs if d.id is not None)),
                available_at=min(d.available_at for d in docs),
                reaction_session=today,
                pre_earnings_close=float(r.prev_close),
                gap_open=float(r.open),
                gap_high=float(r.high),
                gap_low=float(r.low),
                gap_close=float(r.close),
                gap=float(r.gap),
                rvol=float(r.rvol),
                atr=float(r.atr),
                avg_dollar_volume=float(r.avg_dollar_volume),
            )
        )
    return found


BarsAt = Callable[..., pa.Table]


def pool(
    session: Session,
    con: duckdb.DuckDBPyConnection,
    root: Path,
    as_of: datetime,
    cfg: StrategyConfig,
    *,
    sessions: Sequence[date] | None = None,
    bars_at: BarsAt | None = None,
) -> frozenset[str]:
    """What ``scan`` could have returned at ``as_of`` if every company had just released
    earnings: the random-entry null draws from this. Same rules minus the signal: any visible
    8-K makes a company a filer, one security per company as ``_security_of`` picks it, no split
    in the scan window, a bar on the previous session, today's close and the average dollar
    volume before today over the thresholds. No gap, rvol or earnings-item test.

    ``sessions`` and ``bars_at(as_of, start=...)`` (``visible_bars`` semantics, e.g.
    ``BarCache.bars_at``) only spare the caller re-reading the store each evening."""
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    if sessions is None:
        sessions = visible_sessions(con, root, "1Day", as_of)
    sessions = [d for d in sessions if datetime.combine(d, time(20), tzinfo=ET) <= as_of]
    if not sessions:
        return frozenset()
    today = sessions[-1]
    docs: dict[int, list[Document]] = defaultdict(list)
    for doc in visible_documents(session, as_of, doc_type="8-K"):
        if doc.entity_id is not None:
            docs[doc.entity_id].append(doc)
    if not docs:
        return frozenset()

    master = visible_master(con, root, as_of).to_pylist()
    chosen: dict[str, list[str]] = {}
    for entity_id, filed in sorted(docs.items()):
        entity = session.get(Entity, entity_id)
        if entity is None:
            continue
        ciks = {cik for d in filed if (cik := filing_cik(d.url)) is not None}
        if entity.cik is not None:
            ciks.add(entity.cik)
        resolved = _security_of(master, ciks, entity.ticker, today)
        if resolved is not None:
            chosen[resolved[0]] = resolved[1]
    if not chosen:
        return frozenset()

    first = sessions[max(0, len(sessions) - 1 - lookback_sessions(cfg))]
    symbols = sorted({s for symbols in chosen.values() for s in symbols})
    split = {
        r["symbol"]
        for r in visible_splits(con, root, as_of, symbols).to_pylist()
        if first <= r["ex_date"] <= today
    }
    candidates = {sid for sid, symbols in chosen.items() if not split & set(symbols)}
    start = datetime.combine(first, time(0), tzinfo=ET)
    if bars_at is not None:
        bars = bars_at(as_of, start=start)
    else:
        bars = visible_bars(con, root, "1Day", as_of, symbols=symbols, start=start)
    frame = bars.to_pandas()
    frame = frame[frame["security_id"].isin(candidates)]
    if frame.empty:
        return frozenset()
    scan_cfg = cfg.scan
    features = daily_features(
        frame,
        atr_days=scan_cfg.atr_days,
        rvol_days=scan_cfg.rvol_days,
        dollar_volume_days=scan_cfg.dollar_volume_days,
    )
    # As scan(): without a bar on the previous session, prev_close would be an older close.
    if len(sessions) < 2:
        return frozenset()
    traded = set(features.loc[features["session_date"] == sessions[-2], "security_id"])
    row = features[
        (features["session_date"] == today) & features["security_id"].isin(sorted(traded))
    ]
    keep = row[
        (row["close"] >= scan_cfg.min_price)
        & (row["avg_dollar_volume"] >= scan_cfg.min_avg_dollar_volume - 1e-6)
    ]
    return frozenset(str(sid) for sid in keep["security_id"])
