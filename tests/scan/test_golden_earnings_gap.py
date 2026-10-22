"""Guards against the scanner disagreeing with the real store on a known event: RDDT's Q1 2026
results (8-K 0001713445-26-000067, items 2.02 and 9.01, accepted 2026-04-30 16:08:35 New York)
must be a candidate at 2026-05-01 20:00 New York with the numbers below, and must not be one a
second earlier, when the 05-01 daily bar is not final.

Runs on the real store ($ASOF_DATA_DIR/meta.db and market/), opened read-only, and skips when
the store is absent, so CI does not run it. Numbers hand-checked on 2026-10-22 from
visible_bars(symbols=['RDDT']) at 2026-05-27 20:00 New York:
- pre-earnings close (2026-04-30) 147.23; gap open 165.51, high 173.00, low 156.49, close 166.48
- gap = 165.51 / 147.23 - 1 = 0.124159
- rvol = 14,256,573 / mean(volume 2026-04-02 .. 04-30, 20 sessions) = 3.552760
- atr = mean true range 2026-04-13 .. 04-30 (14 sessions) = 8.277921
- security S015043, cik 1713445 (cik_available_at 2024-04-01 04:00 UTC)
- documents: earningspressreleaseq126.htm and exhibit992q126.htm, one row each
"""

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import pytest
from sqlalchemy import Engine
from sqlmodel import Session, col, create_engine, select

from asof.scan.earnings_gap import scan
from asof.store.db import default_data_dir, default_db_path
from asof.store.market import market_root
from asof.store.models import Document
from asof.strategy import load_strategy

ET = ZoneInfo("America/New_York")
AS_OF = datetime(2026, 5, 1, 20, 0, tzinfo=ET)
RELEASE = datetime(2026, 4, 30, 20, 8, 35, tzinfo=UTC)
ACCESSION = "0001713445-26-000067"
REPO_STRATEGY = Path(__file__).resolve().parents[2] / "strategies" / "earnings_gap"


def read_only_engine() -> Engine:
    return create_engine(f"sqlite:///file:{default_db_path()}?mode=ro&uri=true")


pytestmark = [
    pytest.mark.realdata,
    pytest.mark.skipif(not default_db_path().exists(), reason="no real store"),
]


@pytest.fixture(scope="module")
def session() -> Iterator[Session]:
    engine = read_only_engine()
    with Session(engine) as s:
        yield s
    engine.dispose()


@pytest.fixture(scope="module")
def con() -> Iterator[duckdb.DuckDBPyConnection]:
    connection = duckdb.connect()
    yield connection
    connection.close()


def rddt(candidates: list) -> list:
    return [c for c in candidates if c.symbol == "RDDT"]


def test_rddt_is_a_candidate_once_the_gap_day_bar_is_final(
    session: Session, con: duckdb.DuckDBPyConnection
) -> None:
    cfg = load_strategy(REPO_STRATEGY)
    [c] = rddt(scan(session, con, market_root(default_data_dir()), AS_OF, cfg))
    assert c.security_id == "S015043"
    assert c.reaction_session.isoformat() == "2026-05-01"
    assert c.available_at == RELEASE
    assert c.pre_earnings_close == pytest.approx(147.23)
    assert (c.gap_open, c.gap_high, c.gap_low, c.gap_close) == pytest.approx(
        (165.51, 173.00, 156.49, 166.48)
    )
    assert c.gap == pytest.approx(0.124159, abs=1e-6)
    assert c.rvol == pytest.approx(3.552760, abs=1e-6)
    assert c.atr == pytest.approx(8.277921, abs=1e-6)
    external = set(
        session.exec(
            select(Document.external_id).where(col(Document.id).in_(list(c.document_ids)))
        ).all()
    )
    assert external == {
        f"{ACCESSION}/earningspressreleaseq126.htm",
        f"{ACCESSION}/exhibit992q126.htm",
    }


def test_rddt_is_not_a_candidate_one_minute_before(
    session: Session, con: duckdb.DuckDBPyConnection
) -> None:
    cfg = load_strategy(REPO_STRATEGY)
    before = AS_OF - timedelta(minutes=1)
    assert rddt(scan(session, con, market_root(default_data_dir()), before, cfg)) == []
