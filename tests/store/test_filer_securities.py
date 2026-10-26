"""Guards against the "companies that file" universe seeing filings or identities from the future:
an 8-K published after ``as_of``, a security-master segment (a new listing) not yet knowable, a
CIK link not yet knowable. Also guards the join itself: a security is counted once whatever
symbols it traded under, and only documents of the asked ``doc_type`` count.

API under test (asof.store.universe, new helper; the logic of replay.run._universe_symbols):

    filer_securities(session, con, root, as_of, doc_type="8-K") -> frozenset[str]
        security_ids of visible_master(as_of) whose cik belongs to an Entity with a document of
        doc_type visible at as_of (asof.store.queries.visible_documents).

Fake data provenance:
- Document visibility is ``available_at <= as_of`` (documented in asof.store.queries).
- Master segments carry ``cik`` and ``cik_available_at``; ``visible_master`` shows a cik only
  once its ``cik_available_at`` has passed and a segment only once its ``available_at`` has
  (documented in asof.ingest.master_truncation.mask_master). CIK values here are made up.
- Entity.cik is the SEC CIK of the filer (documented in asof.store.models). ASSUMPTION: the
  Entity table's cik and the master's cik use the same integer form (no zero padding, no
  string). Check on the real store: the number of master CIKs with at least one 8-K Entity
  match, versus the number of distinct 8-K filer CIKs (replay's universe size on a known date,
  e.g. 2024-06-03, must not drop when the replay switches to this helper).
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sqlmodel import Session

from asof.ingest.security_master import MASTER_FILE, MASTER_SCHEMA
from asof.store.market import market_root
from asof.store.universe import filer_securities
from tests.scan.world import filing
from tests.store.factories import new_document, new_entity, new_source, pk

ET = ZoneInfo("America/New_York")
TS_UTC = pa.timestamp("us", tz="UTC")
KNOWN_EARLY = datetime(2016, 1, 4, tzinfo=ET).astimezone(UTC)
AS_OF = datetime(2025, 2, 4, 20, tzinfo=ET)  # 20:00 New York, a session's close


@dataclass(frozen=True)
class CikSeg:
    """One master segment with its CIK link. ``cik_available_at`` defaults to ``available_at``."""

    security_id: str
    symbol: str
    cik: int | None
    valid_from: date = date(2016, 1, 4)
    valid_to: date | None = None
    available_at: datetime = KNOWN_EARLY
    end_available_at: datetime | None = None
    cik_available_at: datetime | None = None


def write_cik_master(root: Path, segs: Sequence[CikSeg]) -> Path:
    """Like tests.store.test_universe.write_master, with the cik columns filled in."""
    table = pa.table(
        {
            "security_id": pa.array([s.security_id for s in segs], pa.string()),
            "symbol": pa.array([s.symbol for s in segs], pa.string()),
            "valid_from": pa.array([s.valid_from for s in segs], pa.date32()),
            "valid_to": pa.array([s.valid_to for s in segs], pa.date32()),
            "cik": pa.array([s.cik for s in segs], pa.int64()),
            "cik_available_at": pa.array(
                [None if s.cik is None else (s.cik_available_at or s.available_at) for s in segs],
                TS_UTC,
            ),
            "cusip": pa.array([None for _ in segs], pa.string()),
            "evidence": pa.array(["bars" for _ in segs], pa.string()),
            "available_at": pa.array([s.available_at for s in segs], TS_UTC),
            "end_available_at": pa.array([s.end_available_at for s in segs], TS_UTC),
        },
        schema=MASTER_SCHEMA,
    )
    path = root / MASTER_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


def filer(
    session: Session, source_id: int, cik: int, ticker: str, filed: Sequence[tuple[str, datetime]]
) -> int:
    """An Entity with ``cik`` and one document per ``(doc_type, available_at)``."""
    entity = new_entity(session, cik=cik, ticker=ticker, name=f"{ticker} Corp")
    for doc_type, available in filed:
        new_document(
            session,
            source_id=source_id,
            entity_id=pk(entity),
            available_at=available,
            doc_type=doc_type,
        )
    return pk(entity)


EARLY = datetime(2025, 1, 15, 12, tzinfo=UTC)
RENAMED = date(2024, 6, 3)
RENAMED_KNOWN = datetime(2024, 6, 3, tzinfo=ET).astimezone(UTC)  # 00:00 New York of the rename


@pytest.fixture
def con() -> Iterator[duckdb.DuckDBPyConnection]:
    connection = duckdb.connect()
    yield connection
    connection.close()


@pytest.fixture
def source_id(session: Session) -> int:
    return pk(new_source(session))


# =============================================================================================
# normal path
# =============================================================================================


def test_only_securities_of_8k_filers_are_returned_once_each(
    session: Session, con: duckdb.DuckDBPyConnection, tmp_path: Path, source_id: int
) -> None:
    root = market_root(tmp_path)
    # S_A traded as A then A2 (one security, two segments); S_B's company files only 10-Qs.
    write_cik_master(
        root,
        [
            CikSeg("S_A", "A", 101, valid_to=RENAMED, end_available_at=RENAMED_KNOWN),
            CikSeg("S_A", "A2", 101, valid_from=RENAMED, available_at=RENAMED_KNOWN),
            CikSeg("S_B", "B", 102),
            CikSeg("S_N", "N", None),  # no CIK in the master: never a filer's security
        ],
    )
    filer(session, source_id, 101, "A", [("8-K", EARLY)])
    filer(session, source_id, 102, "B", [("10-Q", EARLY)])

    got = filer_securities(session, con, root, AS_OF)

    assert got == frozenset({"S_A"})
    assert isinstance(got, frozenset)


def test_doc_type_selects_which_filings_count(
    session: Session, con: duckdb.DuckDBPyConnection, tmp_path: Path, source_id: int
) -> None:
    root = market_root(tmp_path)
    write_cik_master(root, [CikSeg("S_A", "A", 101), CikSeg("S_B", "B", 102)])
    filer(session, source_id, 101, "A", [("8-K", EARLY)])
    filer(session, source_id, 102, "B", [("10-Q", EARLY), ("8-K", EARLY)])

    assert filer_securities(session, con, root, AS_OF) == frozenset({"S_A", "S_B"})
    assert filer_securities(session, con, root, AS_OF, doc_type="10-Q") == frozenset({"S_B"})


# =============================================================================================
# boundaries
# =============================================================================================


def test_nobody_filed_gives_an_empty_set(
    session: Session, con: duckdb.DuckDBPyConnection, tmp_path: Path, source_id: int
) -> None:
    root = market_root(tmp_path)
    write_cik_master(root, [CikSeg("S_A", "A", 101)])
    # A document with no entity, and an entity without a CIK: neither links to a security.
    new_document(session, source_id=source_id, available_at=EARLY, doc_type="8-K")
    filer_id = pk(new_entity(session, cik=None, ticker="Q", kind="crypto", name="No CIK"))
    new_document(
        session, source_id=source_id, entity_id=filer_id, available_at=EARLY, doc_type="8-K"
    )

    assert filer_securities(session, con, root, AS_OF) == frozenset()


def test_a_filing_published_exactly_at_as_of_counts(
    session: Session, con: duckdb.DuckDBPyConnection, tmp_path: Path, source_id: int
) -> None:
    root = market_root(tmp_path)
    write_cik_master(root, [CikSeg("S_A", "A", 101)])
    filer(session, source_id, 101, "A", [("8-K", AS_OF)])

    assert filer_securities(session, con, root, AS_OF) == frozenset({"S_A"})


def test_naive_as_of_is_refused(
    session: Session, con: duckdb.DuckDBPyConnection, tmp_path: Path, source_id: int
) -> None:
    root = market_root(tmp_path)
    write_cik_master(root, [CikSeg("S_A", "A", 101)])
    filer(session, source_id, 101, "A", [("8-K", EARLY)])

    with pytest.raises(ValueError):
        filer_securities(session, con, root, datetime(2025, 2, 4, 20))


# =============================================================================================
# leakage
# =============================================================================================


def test_a_company_whose_first_8k_comes_after_as_of_is_not_a_filer_yet(
    session: Session, con: duckdb.DuckDBPyConnection, tmp_path: Path, source_id: int
) -> None:
    root = market_root(tmp_path)
    write_cik_master(root, [CikSeg("S_A", "A", 101), CikSeg("S_L", "L", 105)])
    filer(session, source_id, 101, "A", [("8-K", EARLY)])
    # L's first 8-K is on file one microsecond after as_of (it is in the database already).
    filer(session, source_id, 105, "L", [("8-K", AS_OF + timedelta(microseconds=1))])

    assert filer_securities(session, con, root, AS_OF) == frozenset({"S_A"})
    later = AS_OF + timedelta(days=1)
    assert filer_securities(session, con, root, later) == frozenset({"S_A", "S_L"})


def test_a_master_segment_not_yet_known_contributes_nothing(
    session: Session, con: duckdb.DuckDBPyConnection, tmp_path: Path, source_id: int
) -> None:
    root = market_root(tmp_path)
    listed = datetime(2025, 2, 10, tzinfo=ET).astimezone(UTC)  # the new listing's first bar
    write_cik_master(
        root,
        [
            CikSeg("S_A", "A", 101),
            CikSeg("S_NEW", "NEW", 101, valid_from=date(2025, 2, 10), available_at=listed),
        ],
    )
    filer(session, source_id, 101, "A", [("8-K", EARLY)])

    assert filer_securities(session, con, root, AS_OF) == frozenset({"S_A"})
    assert filer_securities(session, con, root, listed) == frozenset({"S_A", "S_NEW"})


def test_a_cik_link_not_yet_known_contributes_nothing(
    session: Session, con: duckdb.DuckDBPyConnection, tmp_path: Path, source_id: int
) -> None:
    root = market_root(tmp_path)
    linked = AS_OF + timedelta(hours=1)
    write_cik_master(root, [CikSeg("S_A", "A", 101, cik_available_at=linked)])
    filer(session, source_id, 101, "A", [("8-K", EARLY)])

    assert filer_securities(session, con, root, AS_OF) == frozenset()
    assert filer_securities(session, con, root, linked) == frozenset({"S_A"})


def test_a_company_is_matched_by_the_cik_its_filing_was_made_under(
    session: Session, con: duckdb.DuckDBPyConnection, tmp_path: Path, source_id: int
) -> None:
    # Real shape (XOM): the master's segment before 2026-07-16 carries 34088, the entity today
    # 2115436; the filing's archive URL names 34088. Its own CIK alone would match nothing.
    root = market_root(tmp_path)
    write_cik_master(root, [CikSeg("S_X", "X", 34088)])
    entity = new_entity(session, cik=2115436, ticker="X", name="X Corp")
    filing(
        session,
        source_id=source_id,
        entity_id=pk(entity),
        accession="0000034088-25-000001",
        available_at=EARLY,
        exhibits=[("ex991.htm", "8-K 2.02,9.01 EX-99.1 RESULTS")],
        filer_cik=34088,
    )

    assert filer_securities(session, con, root, AS_OF) == frozenset({"S_X"})
