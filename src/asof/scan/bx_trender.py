"""B-Xtrender entry scan: the liquid common stocks whose higher-period histogram turned up.

Run at 20:00 New York, when the day's bar is final. A security signals on session s when it is
in that night's liquid tier, the last completed state bar (weekly for ``mid``, monthly for
``long``) is above the one before it, the daily histogram is above its previous traded
session's, and its series has run long enough for the seed of the EMAs to have faded
(``warmup``). Exits and sizing belong to the executor.

Every series starts at the security's first visible bar (a later start gives other values).
``scan_range`` computes the series once, at the close of its last session, and reads each
session from it: a split visible by then and effective after s scales every price up to s by
one constant, which leaves the histogram unchanged, and spin-offs are blanked until their
``known_on``; the tests hold it equal to the nightly ``scan``.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd
import pyarrow as pa
import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from asof.features.bx import PRICES, adjust_for_splits, bx_features
from asof.store.market import visible_bars, visible_master, visible_sessions, visible_splits
from asof.store.spinoffs import spinoff_events
from asof.store.universe import liquid_securities

ET = ZoneInfo("America/New_York")
CLOSE = time(20)
STATE = {"mid": "weekly", "long": "monthly"}


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class UniverseConfig(_Section):
    min_adv_usd: float
    lookback_sessions: int
    classes: tuple[str, ...]


class BxParams(_Section):
    l1: int
    l2: int
    l3: int


class WarmupConfig(_Section):
    """Bars of each period a series needs before it may signal: the seed of the EMA20 fades
    slowly. Series started 2020-01 vs 2021-01 agree on the direction of 99% of steps from 40
    daily, 44 weekly and 36 monthly bars on (the table is in the strategy's scan.yaml)."""

    daily: int
    weekly: int
    monthly: int


class BxScanConfig(_Section):
    variant: Literal["mid", "long"]
    universe: UniverseConfig
    bx: BxParams
    warmup: WarmupConfig


def load_scan_config(path: Path, overrides: Sequence[str] = ()) -> BxScanConfig:
    """``path`` is the scan.yaml file. Each override is ``key=value`` or ``a.b=value``, the
    value read as YAML; an unknown key or a value of the wrong type raises ValueError."""
    raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    for override in overrides:
        dotted, sep, value = override.partition("=")
        keys = dotted.split(".")
        if not sep or any(not key for key in keys):
            raise ValueError(f"{override!r}: expected key=value")
        node = raw
        for key in keys[:-1]:
            if not isinstance(node.get(key), dict):
                raise ValueError(f"{override!r}: no section {key!r}")
            node = node[key]
        if keys[-1] not in node or isinstance(node[keys[-1]], dict):
            raise ValueError(f"{override!r}: no setting {keys[-1]!r}")
        node[keys[-1]] = yaml.safe_load(value)
    try:
        return BxScanConfig.model_validate(raw)
    except ValidationError as error:
        raise ValueError(str(error)) from error


@dataclass(frozen=True)
class BxCandidate:
    security_id: str
    symbol: str  # the symbol in use at as_of
    session: date  # the session whose close produced the signal
    variant: str
    state_bx: float
    state_bx_prev: float
    state_complete_on: date  # the session that completed the state bar read
    daily_bx: float
    daily_bx_prev: float
    rank: int  # 1 = largest state_bx - state_bx_prev tonight; ties by security_id
    bx_since: date
    state_bars: int
    adv_usd: float


def _close(day: date) -> datetime:
    return datetime.combine(day, CLOSE, tzinfo=ET)


def bx_panel(
    con: duckdb.DuckDBPyConnection,
    root: Path,
    as_of: datetime,
    security_ids: Sequence[str],
    *,
    l1: int = 5,
    l2: int = 20,
    l3: int = 5,
) -> pd.DataFrame:
    """``bx_features`` of every bar visible at ``as_of`` of these securities, from each one's
    first bar: split-adjusted by the splits visible then, spin-offs as decided then. Bars and
    spin-off records are matched under every symbol a security has carried, so a renamed
    security keeps the events filed under its old name."""
    wanted = set(security_ids)
    symbols = sorted(
        {
            r["symbol"]
            for r in visible_master(con, root, as_of).to_pylist()
            if r["security_id"] in wanted
        }
    )
    if not symbols:
        return bx_features(pd.DataFrame(columns=["security_id", "session_date", *PRICES]))
    bars = visible_bars(con, root, "1Day", as_of, symbols=symbols).to_pandas()
    bars = pd.DataFrame(bars[bars["security_id"].isin(wanted)])
    daily = adjust_for_splits(bars, visible_splits(con, root, as_of, symbols).to_pandas())
    events = spinoff_events(con, root, as_of, symbols).to_pandas()
    events = pd.DataFrame(events[events["security_id"].isin(wanted)])
    return bx_features(daily, spinoffs=events, l1=l1, l2=l2, l3=l3)


def _eligible(rows: pd.DataFrame, tier: pa.Table | pd.DataFrame, cfg: BxScanConfig) -> pd.DataFrame:
    """Tonight's rows of tier securities of the config's classes whose four values are defined
    and whose series are warm, joined with the tier's symbol and adv_usd."""
    state = STATE[cfg.variant]
    tier_df = tier.to_pandas() if isinstance(tier, pa.Table) else tier
    tier_df = pd.DataFrame(tier_df[tier_df["class"].isin(cfg.universe.classes)])
    merged = rows.drop(columns=["symbol"], errors="ignore").merge(
        pd.DataFrame(tier_df[["security_id", "symbol", "adv_usd"]]), on="security_id"
    )
    values = merged[[f"bx_{state}", f"bx_{state}_prev", "bx_daily", "bx_daily_prev"]]
    keep = (
        values.notna().all(axis=1)
        & (merged["daily_bars"] >= cfg.warmup.daily)
        & (merged[f"{state}_bars"] >= getattr(cfg.warmup, state))
    )
    return pd.DataFrame(merged[keep.to_numpy()])


def signals(
    rows: pd.DataFrame, tier: pa.Table | pd.DataFrame, cfg: BxScanConfig
) -> list[BxCandidate]:
    """``rows``: the panel rows of one session. The candidates among them, ranked."""
    state = STATE[cfg.variant]
    ok = _eligible(rows, tier, cfg)
    rises = (ok[f"bx_{state}"] > ok[f"bx_{state}_prev"]) & (ok["bx_daily"] > ok["bx_daily_prev"])
    ok = pd.DataFrame(ok[rises.to_numpy()])
    # the bar that turned up most ranks first: a value pinned near +50 that rose 0.01 does not
    # outrank one that just turned (2025-03-14: JTAI weekly 49.99 after 49.99 led by level)
    ok["_rise"] = ok[f"bx_{state}"] - ok[f"bx_{state}_prev"]
    ok = ok.sort_values(["_rise", "security_id"], ascending=[False, True])
    return [
        BxCandidate(
            security_id=str(r["security_id"]),
            symbol=str(r["symbol"]),
            session=_day(r["session_date"]),
            variant=cfg.variant,
            state_bx=float(r[f"bx_{state}"]),
            state_bx_prev=float(r[f"bx_{state}_prev"]),
            state_complete_on=_day(r[f"{state}_complete_on"]),
            daily_bx=float(r["bx_daily"]),
            daily_bx_prev=float(r["bx_daily_prev"]),
            rank=rank,
            bx_since=_day(r["bx_since"]),
            state_bars=int(r[f"{state}_bars"]),
            adv_usd=float(r["adv_usd"]),
        )
        for rank, r in enumerate(ok.to_dict("records"), start=1)
    ]


def _day(value: Any) -> date:
    return value if type(value) is date else date.fromisoformat(str(value)[:10])


def _tier(
    con: duckdb.DuckDBPyConnection, root: Path, as_of: datetime, cfg: BxScanConfig
) -> pa.Table:
    u = cfg.universe
    return liquid_securities(
        con,
        root,
        as_of,
        min_adv_usd=u.min_adv_usd,
        lookback_sessions=u.lookback_sessions,
        classes=u.classes,
    )


def _sessions(con: duckdb.DuckDBPyConnection, root: Path, as_of: datetime) -> list[date]:
    """Sessions final at ``as_of``: visible, and their 20:00 close not after it."""
    return [d for d in visible_sessions(con, root, "1Day", as_of) if _close(d) <= as_of]


def _tonight(
    con: duckdb.DuckDBPyConnection, root: Path, as_of: datetime, cfg: BxScanConfig
) -> tuple[date, pd.DataFrame, pa.Table] | None:
    sessions = _sessions(con, root, as_of)
    if not sessions:
        return None
    tier = _tier(con, root, as_of, cfg)
    ids = tier.column("security_id").to_pylist()
    panel = bx_panel(con, root, as_of, ids, **cfg.bx.model_dump())
    rows = pd.DataFrame(panel[panel["session_date"] == sessions[-1]])
    return sessions[-1], rows, tier


def scan_and_pool(
    con: duckdb.DuckDBPyConnection, root: Path, as_of: datetime, cfg: BxScanConfig
) -> tuple[date | None, list[BxCandidate], frozenset[str]]:
    """The session read (None before any is final), ``scan`` and ``pool`` from one panel."""
    found = _tonight(con, root, as_of, cfg)
    if found is None:
        return None, [], frozenset()
    day, rows, tier = found
    eligible = frozenset(_eligible(rows, tier, cfg)["security_id"].astype(str))
    return day, signals(rows, tier, cfg), eligible


def scan(
    con: duckdb.DuckDBPyConnection, root: Path, as_of: datetime, cfg: BxScanConfig
) -> list[BxCandidate]:
    """Candidates of the latest session final at ``as_of``."""
    return scan_and_pool(con, root, as_of, cfg)[1]


def pool(
    con: duckdb.DuckDBPyConnection, root: Path, as_of: datetime, cfg: BxScanConfig
) -> frozenset[str]:
    """What ``scan`` could have returned at ``as_of`` without the two rises: tonight's tier
    securities with all four values defined and a warm series. The random-entry null draws
    from this."""
    return scan_and_pool(con, root, as_of, cfg)[2]


def _nights(
    con: duckdb.DuckDBPyConnection, root: Path, start: date, end: date, cfg: BxScanConfig
) -> list[tuple[date, pd.DataFrame, pa.Table]]:
    """Each session from ``start`` through ``end`` with its panel rows and its tier.

    The panel is computed once at the close of ``end``. That reads the master as known then:
    on a night inside a segment's publication lag (NKLA's segment starts 2020-06-04 and is
    published 06-08; LAC's rename on 10-04 is recorded 11-02) the nightly reader attributed
    the bars differently, and a child listed late left its spin-off pending. A tier security
    carrying such a symbol, or whose spin-off child does, is recomputed at that night's close."""
    as_of = _close(end)
    days = [d for d in _sessions(con, root, as_of) if start <= d]
    tiers = {d: _tier(con, root, _close(d), cfg) for d in days}
    ids = sorted({sid for t in tiers.values() for sid in t.column("security_id").to_pylist()})
    panel = bx_panel(con, root, as_of, ids, **cfg.bx.model_dump())
    by_day = {d: g for d, g in panel.groupby("session_date", sort=False)}
    empty = panel.iloc[0:0]

    master = visible_master(con, root, as_of).to_pylist()
    lags: dict[str, list[tuple[date, datetime]]] = {}  # symbol -> (first night, published)
    for seg in master:
        if seg["valid_from"] is not None and seg["available_at"] > _close(seg["valid_from"]):
            lags.setdefault(seg["symbol"], []).append((seg["valid_from"], seg["available_at"]))
        end_at = seg["end_available_at"]
        if seg["valid_to"] is not None and end_at is not None and end_at > _close(seg["valid_to"]):
            lags.setdefault(seg["symbol"], []).append((seg["valid_to"], end_at))
    symbols: dict[str, set[str]] = {}
    for seg in master:
        symbols.setdefault(seg["security_id"], set()).add(seg["symbol"])
    for event in spinoff_events(con, root, as_of).to_pylist():
        if event["security_id"]:
            symbols.setdefault(event["security_id"], set()).update(event["child_symbols"])

    def lagged(sid: str, day: date) -> bool:
        return any(
            first <= day and _close(day) < published
            for symbol in symbols.get(sid, ())
            for first, published in lags.get(symbol, ())
        )

    out = []
    for d in days:
        rows = pd.DataFrame(by_day.get(d, empty))
        stale = [sid for sid in tiers[d].column("security_id").to_pylist() if lagged(sid, d)]
        if stale:
            fresh = bx_panel(con, root, _close(d), stale, **cfg.bx.model_dump())
            rows = pd.concat(
                [rows[~rows["security_id"].isin(stale)], fresh[fresh["session_date"] == d]],
                ignore_index=True,
            )
        out.append((d, rows, tiers[d]))
    return out


def scan_range(
    con: duckdb.DuckDBPyConnection, root: Path, start: date, end: date, cfg: BxScanConfig
) -> dict[date, list[BxCandidate]]:
    """``scan`` of every session from ``start`` through ``end``. Sessions without a candidate
    are left out."""
    out: dict[date, list[BxCandidate]] = {}
    for d, rows, tier in _nights(con, root, start, end, cfg):
        found = signals(rows, tier, cfg)
        if found:
            out[d] = found
    return out


def pool_range(
    con: duckdb.DuckDBPyConnection, root: Path, start: date, end: date, cfg: BxScanConfig
) -> dict[date, frozenset[str]]:
    """``pool`` of every session from ``start`` through ``end``."""
    return {
        d: frozenset(_eligible(rows, tier, cfg)["security_id"].astype(str))
        for d, rows, tier in _nights(con, root, start, end, cfg)
    }
