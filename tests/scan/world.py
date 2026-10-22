"""A small store for scan and replay tests: Parquet daily bars, a security master with CIKs, and
SEC 8-K documents in SQLite, all written the way the real ingest writes them.

Provenance of the fake data:
- Daily bar availability 20:00 New York of its session: documented in asof.ingest.bars and
  computed by ``write_partition`` itself (real: RDDT's 2026-05-01 bar).
- Master rows carry ``cik`` / ``cik_available_at`` like the real master (real: RDDT S015043,
  cik 1713445, cik_available_at 2024-04-01 04:00 UTC).
- 8-K documents: one row per exhibit, ``external_id`` = "<accession>/<file>", title
  "<form> <items> <exhibit type> <description>" (real: RDDT 0001713445-26-000067 has two rows,
  earningspressreleaseq126.htm "8-K 2.02,9.01 EX-99.1 EX-99.1" and exhibit992q126.htm
  "8-K 2.02,9.01 EX-99.2 EX-99.2", both available 2026-04-30 20:08:35 UTC); ``available_at`` is
  the EDGAR acceptance time.
- The calendar is whatever sessions have bars: 2026-01-19 (MLK day), 2026-02-16 (Presidents'
  Day) and 2026-04-03 (Good Friday) are left out, as on the real exchange calendar.
- Gap-day numbers (gap 10%, rvol 3, ATR 2, $50M a day) are made up and chosen to sit on round
  values so boundaries can be hit exactly.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sqlmodel import Session

from asof.ingest.alpaca import Bar
from asof.ingest.security_master import MASTER_FILE, MASTER_SCHEMA
from asof.store.market import market_root
from asof.store.models import Document, Entity, Source
from tests.ingest.fakes import daily
from tests.store.factories import add, pk
from tests.store.test_market import store_daily

ET = ZoneInfo("America/New_York")
HOLIDAYS = {date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3)}
SESSIONS: list[date] = [
    d.date() for d in pd.bdate_range("2026-01-05", "2026-06-30") if d.date() not in HOLIDAYS
]
HISTORY = 29  # quiet sessions before the reaction day
D = SESSIONS[HISTORY]  # 2026-02-17, the Tuesday after Presidents' Day
PREV = SESSIONS[HISTORY - 1]  # 2026-02-13, a Friday
KNOWN_EARLY = datetime(2016, 1, 4, tzinfo=ET).astimezone(UTC)

GAPX_CIK = 1111
OTHR_CIK = 2222


def et(day: date, hh: int, mm: int = 0, ss: int = 0) -> datetime:
    return datetime.combine(day, time(hh, mm, ss), tzinfo=ET)


def close_of(day: date) -> datetime:
    """20:00 New York: when the daily bar of ``day`` becomes visible; the replay's as_of."""
    return et(day, 20)


# Friday after the close; the next session is Tuesday because Monday is a holiday.
RELEASE = et(PREV, 16, 8)

Ohlcv = tuple[float, float, float, float, int]
QUIET: Ohlcv = (50.0, 51.0, 49.0, 50.0, 1_000_000)  # true range 2, $50M a day
GAP: Ohlcv = (55.0, 57.0, 54.0, 56.0, 3_000_000)  # gap +10%, rvol 3


def bar(symbol: str, day: date, values: Ohlcv) -> Bar:
    o, h, lo, c, v = values
    return daily(symbol, day, open=o, high=h, low=lo, close=c, volume=v, vwap=c)


@dataclass(frozen=True)
class Seg:
    security_id: str
    symbol: str
    cik: int | None
    cik_available_at: datetime | None = KNOWN_EARLY
    valid_from: date = date(2020, 1, 2)
    available_at: datetime = KNOWN_EARLY
    valid_to: date | None = None
    end_available_at: datetime | None = None


def write_master(root: Path, segs: Sequence[Seg]) -> None:
    table = pa.table(
        {
            "security_id": pa.array([s.security_id for s in segs], pa.string()),
            "symbol": pa.array([s.symbol for s in segs], pa.string()),
            "valid_from": pa.array([s.valid_from for s in segs], pa.date32()),
            "valid_to": pa.array([s.valid_to for s in segs], pa.date32()),
            "cik": pa.array([s.cik for s in segs], pa.int64()),
            "cik_available_at": pa.array(
                [s.cik_available_at if s.cik is not None else None for s in segs],
                pa.timestamp("us", tz="UTC"),
            ),
            "cusip": pa.array([None for _ in segs], pa.string()),
            "evidence": pa.array(["sec" for _ in segs], pa.string()),
            "available_at": pa.array([s.available_at for s in segs], pa.timestamp("us", tz="UTC")),
            "end_available_at": pa.array(
                [s.end_available_at for s in segs], pa.timestamp("us", tz="UTC")
            ),
        },
        schema=MASTER_SCHEMA,
    )
    path = root / MASTER_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def edgar(session: Session) -> int:
    return pk(add(session, Source(name="sec_edgar", base_url="https://www.sec.gov")))


def company(session: Session, ticker: str, cik: int) -> int:
    return pk(add(session, Entity(kind="company", ticker=ticker, name=f"{ticker} Inc.", cik=cik)))


def filing(
    session: Session,
    *,
    source_id: int,
    entity_id: int,
    accession: str,
    available_at: datetime,
    exhibits: Iterable[tuple[str, str]],
    doc_type: str = "8-K",
    filer_cik: int | None = None,
) -> list[int]:
    """One Document per (file, title), as the EDGAR ingest stores an 8-K's exhibits. With
    ``filer_cik`` the URL has the real archive shape ".../edgar/data/<cik>/<accession>/<file>"
    (real: XOM's 2023 8-Ks sit under data/34088/, its 2026-10 ones under data/2115436/)."""
    ids = []
    for file, title in exhibits:
        doc = Document(
            source_id=source_id,
            entity_id=entity_id,
            doc_type=doc_type,
            external_id=f"{accession}/{file}",
            title=title,
            url=(
                f"https://www.sec.gov/Archives/edgar/data/{filer_cik}/"
                f"{accession.replace('-', '')}/{file}"
                if filer_cik is not None
                else f"https://www.sec.gov/Archives/edgar/data/{accession}/{file}"
            ),
            content_hash=f"hash-{accession}-{file}",
            source_time=available_at - timedelta(days=1),
            available_at=available_at.astimezone(UTC),
            received_at=available_at.astimezone(UTC) + timedelta(days=1),
        )
        ids.append(pk(add(session, doc)))
    return ids


EARNINGS = (
    ("ex991.htm", "8-K 2.02,9.01 EX-99.1 PRESS RELEASE"),
    ("ex992.htm", "8-K 2.02,9.01 EX-99.2 CFO COMMENTARY"),
)


@dataclass
class World:
    root: Path
    source_id: int
    gapx: int  # entity id
    othr: int  # entity id, quiet all along, never files
    earnings_docs: list[int]
    bars: list[Bar] = field(default_factory=list)


def gap_world(
    tmp_path: Path,
    session: Session,
    *,
    reaction: Ohlcv = GAP,
    after: Sequence[Ohlcv | None] = (),
    othr_bars: Mapping[date, Ohlcv] | None = None,
    release_at: datetime = RELEASE,
    exhibits: Iterable[tuple[str, str]] = EARNINGS,
    gapx_cik_available_at: datetime = KNOWN_EARLY,
    extra_bars: Sequence[Bar] = (),
    extra_segs: Sequence[Seg] = (),
    missing_history: Sequence[date] = (),
    gapx_segs: Sequence[Seg] | None = None,
    old_symbol: tuple[str, date] | None = None,
    entity_cik: int = GAPX_CIK,
    filer_cik: int | None = None,
) -> World:
    """GAPX: HISTORY quiet sessions, the reaction bar on D, then ``after`` (one per session;
    None = no GAPX bar that session). OTHR: a bar on every one of those sessions, quiet unless
    ``othr_bars`` says otherwise, so the market calendar never has holes. Market root =
    tmp_path/market, matching the store fixture's tmp_path/meta.db.

    ``missing_history``: sessions with no GAPX bar. ``old_symbol`` = (symbol, rename day): GAPX
    bars before that day are filed under the old symbol, as a renamed security's are.
    ``gapx_segs`` replaces GAPX's master segment(s). ``entity_cik`` is the company's CIK today,
    ``filer_cik`` the one its filing was made under."""
    root = market_root(tmp_path)
    days = SESSIONS[: HISTORY + 1 + len(after)]

    def label(d: date) -> str:
        return old_symbol[0] if old_symbol is not None and d < old_symbol[1] else "GAPX"

    gapx = [bar(label(d), d, QUIET) for d in days[:HISTORY] if d not in missing_history]
    gapx.append(bar("GAPX", D, reaction))
    gapx += [
        bar("GAPX", d, v) for d, v in zip(days[HISTORY + 1 :], after, strict=True) if v is not None
    ]
    othr = [bar("OTHR", d, (othr_bars or {}).get(d, QUIET)) for d in days]
    bars = [*gapx, *othr, *extra_bars]
    store_daily(root, bars)
    write_master(
        root,
        [
            *(gapx_segs or [Seg("S_GAPX", "GAPX", GAPX_CIK, gapx_cik_available_at)]),
            Seg("S_OTHR", "OTHR", OTHR_CIK),
            *extra_segs,
        ],
    )
    source_id = edgar(session)
    gapx_id = company(session, "GAPX", entity_cik)
    othr_id = company(session, "OTHR", OTHR_CIK)
    docs = filing(
        session,
        source_id=source_id,
        entity_id=gapx_id,
        accession="0000001111-26-000003",
        available_at=release_at,
        exhibits=exhibits,
        filer_cik=filer_cik,
    )
    return World(root, source_id, gapx_id, othr_id, docs, bars)
