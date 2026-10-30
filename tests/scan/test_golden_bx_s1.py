# Guards against the new B-Xtrender THT entry filter changing S1 when it is off: the strategy's
# scan.yaml, with tht_filter null, must give exactly the candidates and pool it gave before the
# filter existed, on one real evening.
"""Runs on the real store ($ASOF_DATA_DIR/market/), and skips when the store is absent, so CI
does not run it.

Recorded 2026-10-29 from the code before the filter existed (commit c737581), with
load_scan_config(strategies/bx_trender/scan.yaml) and scan_and_pool at 2024-01-10 20:00 New
York: session 2024-01-10, 101 candidates in the rank order below (first: KNSL, S004439, weekly
-6.387 after -36.714), pool of 1,066 securities. Symbols, not security ids, are compared: a
master rebuild may renumber the ids.
"""

from collections.abc import Iterator
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import pytest

from asof.scan.bx_trender import load_scan_config, scan_and_pool
from asof.store.db import default_data_dir, default_db_path
from asof.store.market import market_root

ET = ZoneInfo("America/New_York")
AS_OF = datetime(2024, 1, 10, 20, 0, tzinfo=ET)
SCAN_YAML = Path(__file__).resolve().parents[2] / "strategies" / "bx_trender" / "scan.yaml"
S1_CANDIDATES = [
    "KNSL", "LLY", "MCK", "CELH", "FOXA", "EPD", "HQY", "SWN", "BRK.B", "CNC", "BRK.A", "FE",
    "STZ", "CRBG", "CIEN", "ED", "ARCC", "PAA", "NU", "CINF", "COR", "GSK", "NVO", "CHH", "RGA",
    "PG", "EG", "SNY", "BCE", "ARWR", "EXC", "PDCO", "ACGL", "TEVA", "PGR", "SWAV", "TM", "KMB",
    "SRPT", "ETR", "NYCB", "ABBV", "APO", "ARMK", "WMT", "AZN", "OGN", "MTCH", "VTRS", "MRK",
    "CTLT", "NI", "TW", "IP", "AXSM", "MNST", "KO", "BSX", "AMN", "LNC", "CHD", "HIG", "MDT",
    "TOST", "NXST", "FTNT", "HCA", "FIS", "LSXMK", "STE", "TAP", "SMCI", "XRAY", "WTRG", "JNPR",
    "POL", "UAL", "UPS", "DVA", "PEG", "MRVL", "MMSI", "NVST", "MODG", "DOX", "ABT", "LUV", "CCCC",
    "ASND", "GPN", "VEEV", "GKOS", "APLS", "AWK", "THC", "MMC", "AYI", "WPC", "DAL", "ARDX", "OMC",
]  # fmt: skip
S1_POOL_SIZE = 1066

pytestmark = [
    pytest.mark.realdata,
    pytest.mark.skipif(not default_db_path().exists(), reason="no real store"),
]


@pytest.fixture(scope="module")
def con() -> Iterator[duckdb.DuckDBPyConnection]:
    connection = duckdb.connect()
    yield connection
    connection.close()


def test_s1_with_the_tht_filter_null_gives_the_candidates_of_before(
    con: duckdb.DuckDBPyConnection,
) -> None:
    cfg = load_scan_config(SCAN_YAML)
    assert cfg.tht_filter is None

    day, found, pool = scan_and_pool(con, market_root(default_data_dir()), AS_OF, cfg)

    assert day == date(2024, 1, 10)
    assert [c.symbol for c in found] == S1_CANDIDATES
    assert [c.rank for c in found] == list(range(1, len(S1_CANDIDATES) + 1))
    assert len(pool) == S1_POOL_SIZE
