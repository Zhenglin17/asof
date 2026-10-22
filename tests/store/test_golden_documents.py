"""Guards against the real document store drifting from hand-checked facts: when a filing became
public, which company it is attached to, and that its chunks still read the filing's words.

Runs on the real store ($ASOF_DATA_DIR/meta.db and the text files under docs/sec) and skips when
no EDGAR documents are there, so CI does not run it. Acceptance times were checked by hand on
2026-10-21 against each filing's index page and EDGAR header (ACCEPTANCE-DATETIME).
"""

from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta

import pytest
from sqlmodel import Session, col, select

from asof.ingest.edgar import periodic_gaps, stale_reports
from asof.store.db import default_data_dir, default_db_path, make_engine
from asof.store.documents import text_path
from asof.store.models import Document, Entity, Source
from asof.store.queries import visible_chunks, visible_documents

SECOND = timedelta(seconds=1)
TEXT_ROOT = default_data_dir() / "docs" / "sec"


def _has_edgar_documents() -> bool:
    """Skip only when the store is absent; any other failure (a locked file) must fail loudly."""
    if not default_db_path().exists():
        return False
    engine = make_engine(default_db_path())
    try:
        with Session(engine) as session:
            return (
                session.exec(select(Source).where(Source.name == "sec_edgar")).first() is not None
            )
    finally:
        engine.dispose()


pytestmark = [
    pytest.mark.realdata,
    pytest.mark.skipif(not _has_edgar_documents(), reason="no EDGAR documents in the real store"),
]


@pytest.fixture(scope="module")
def session() -> Iterator[Session]:
    engine = make_engine(default_db_path())
    with Session(engine) as s:
        yield s
    engine.dispose()


def entity(session: Session, ticker: str) -> Entity:
    return session.exec(select(Entity).where(Entity.ticker == ticker)).one()


def document(session: Session, external_id: str) -> Document:
    return session.exec(select(Document).where(Document.external_id == external_id)).one()


def test_nvda_q2_fy27_release_is_public_at_16_21_new_york(session: Session) -> None:
    # Hand-checked: index page "Accepted 2026-08-26 16:21:19"; the submissions JSON says
    # 2026-08-27T00:21:19Z, four hours late, and must not be what the store holds.
    release = document(session, "0001045810-26-000073/q2fy27pr.htm")
    public = datetime(2026, 8, 26, 20, 21, 19, tzinfo=UTC)
    assert release.available_at == public
    assert release.entity_id == entity(session, "NVDA").id

    nvda = entity(session, "NVDA").id
    before = {d.id for d in visible_documents(session, public - SECOND, entity_id=nvda)}
    assert release.id not in before
    chunks = visible_chunks(session, public, document_id=release.id)
    sentence = "of $96.2 billion, up 18% from the previous quarter and up 106% from a year ago."
    assert any(sentence in c.text for c in chunks)
    text = text_path(TEXT_ROOT, release.external_id).read_text(encoding="utf-8")
    assert all(text[c.char_start : c.char_end] == c.text for c in chunks)


def test_late_nvda_10q_waits_for_its_next_business_day(session: Session) -> None:
    # Hand-checked: accepted Friday 2023-08-25 19:36:34 New York, Filing Date 2023-08-28.
    ten_q = document(session, "0001045810-23-000175/nvda-20230730.htm")
    opens = datetime(2023, 8, 28, 10, 0, tzinfo=UTC)  # 06:00 EDT
    assert ten_q.available_at == opens
    nvda = entity(session, "NVDA").id
    assert ten_q.id not in {
        d.id for d in visible_documents(session, opens - SECOND, entity_id=nvda)
    }
    assert ten_q.id in {d.id for d in visible_documents(session, opens, entity_id=nvda)}


def test_xom_reports_under_its_old_cik_belong_to_xom(session: Session) -> None:
    # Hand-checked: Exxon Mobil Corp (CIK 34088) filed the Q2 2024 results 8-K, accepted
    # 2024-08-02 06:31:19 New York; XOM moved to CIK 2115436 only on 2026-07-16.
    xom = entity(session, "XOM")
    assert xom.cik == 2115436
    release = document(session, "0000034088-24-000048/f8k2q24991.htm")
    assert release.entity_id == xom.id
    assert release.available_at == datetime(2024, 8, 2, 10, 31, 19, tzinfo=UTC)
    seen = visible_documents(session, datetime(2024, 8, 3, tzinfo=UTC), entity_id=xom.id)
    assert release.id in {d.id for d in seen}


def test_ccj_pdf_only_annual_report_keeps_only_the_6k_cover(session: Session) -> None:
    # Hand-checked 2026-10-22: 6-K 0001193125-25-072172 has the 6-K cover and one exhibit,
    # EX-99.1 d859263dex991.pdf (Cameco 2024 annual report); read as HTML it was 7.5 MB of PDF.
    stored = session.exec(
        select(Document).where(col(Document.external_id).startswith("0001193125-25-072172/"))
    ).all()
    assert [d.external_id for d in stored] == ["0001193125-25-072172/d859263d6k.htm"]
    assert stored[0].entity_id == entity(session, "CCJ").id
    chunks = visible_chunks(session, stored[0].available_at, document_id=stored[0].id)
    assert any("Cameco Corporation" in c.text for c in chunks)


def test_rcat_transition_report_is_stored_as_a_periodic_report(session: Session) -> None:
    # Hand-checked: 10-KT for the eight months to 2024-12-31, accepted 2025-03-31 17:28:47
    # New York, two minutes before the cutoff, so public the same evening.
    report = document(session, "0001641172-25-001892/form10-kt.htm")
    assert report.doc_type == "10-KT"
    assert report.entity_id == entity(session, "RCAT").id
    public = datetime(2025, 3, 31, 21, 28, 47, tzinfo=UTC)
    assert report.available_at == public
    chunks = visible_chunks(session, public, document_id=report.id)
    assert any(
        "For the transition period from May 1, 2024 to December 31, 2024" in c.text for c in chunks
    )
    assert "Part II > Item 7" in {c.section_path for c in chunks}


def test_companies_that_moved_from_20f_to_10k_show_no_gap(session: Session) -> None:
    # Hand-checked: Celestica 20-F FY2022, FY2023, then 10-K FY2024 and 10-Qs; Shopify the same
    # from 40-F; IREN 20-F to June 2024, then a 10-K for the year to June 2025.
    movers = {"CLS", "SHOP", "IREN"}
    assert not [gap for gap in periodic_gaps(session) if gap[0] in movers]
    assert not [s for s in stale_reports(session, date(2026, 10, 22)) if s[0] in movers]


def test_uec_results_furnished_under_item_701_are_stored(session: Session) -> None:
    # Hand-checked 2026-10-22: UEC furnishes its results only under Item 7.01; the Q3 FY2025
    # release 0001437749-25-019173 was accepted 2025-06-02 16:30:47 New York (EDT).
    release = document(session, "0001437749-25-019173/ex_826004.htm")
    assert release.entity_id == entity(session, "UEC").id
    assert release.title is not None and release.title.startswith("8-K 7.01,9.01 EX-99.1")
    public = datetime(2025, 6, 2, 20, 30, 47, tzinfo=UTC)
    assert release.available_at == public
    chunks = visible_chunks(session, public, document_id=release.id)
    assert any("Files Quarterly Report" in c.text for c in chunks)


def test_late_winter_filing_waits_for_06_00_eastern_standard_time(session: Session) -> None:
    # Hand-checked: UEC's Anfield 8-K 0001437749-25-001228 was accepted 2025-01-15 18:33:12 EST,
    # after the 17:30 cutoff; its Filing Date is 2025-01-16, so 06:00 EST = 11:00 UTC.
    notice = document(session, "0001437749-25-001228/ex_766405.htm")
    assert notice.available_at == datetime(2025, 1, 16, 11, 0, tzinfo=UTC)
