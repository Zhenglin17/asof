"""Guards against look-ahead and wrong-company filings: a filing readable before the SEC
disseminated it, acceptance times read in the wrong zone, an old CIK's filings attached to a
company after it stopped being that company, and re-downloads that rewrite stored evidence.

Fixtures under fixtures/edgar/ are real SEC responses saved 2026-10-21 (measured), trimmed to a
few rows. Fakes built on top of them say what they imitate next to their definition.

Measured 2026-10-21: the submissions JSON field acceptanceDateTime is wrong for most filers (the
New York clock shifted to UTC twice: NVDA 0001045810-26-000073 JSON 2026-08-27T00:21:19Z, index
page and EDGAR header 2026-08-26 16:21:19 New York) and right for some (Exxon 34088). Acceptance
is therefore read from the index page "Accepted" line, which matched the EDGAR header
ACCEPTANCE-DATETIME on every filing checked.

Assumptions that only real data can confirm, with the check that does it:
- Assumed: filingDate already encodes SEC's business-day calendar, so 06:00 New York on filingDate
  is never before dissemination. Measured on 2 late cases (NVDA 2023 10-Q accepted 19:36, XOM 2022
  8-K accepted 18:11, both dated the next business day); check by asserting available_at >= the
  index-page acceptance across every stored filing.
- Assumed: a CIK with no filings since `since` serves the same JSON shape with empty columns
  (the new XOM holding company CIK 2115436 below). Check: run the sync for XOM on real data.
- Assumed: SEC leaves reportDate empty ("") for forms without a period. All fixture rows have
  one. Check: audit that every stored document has a source_time.
"""

import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import pyarrow as pa
import pytest
from sqlalchemy import Engine
from sqlmodel import Session, col, select

from asof.ingest.edgar import (
    CikWindow,
    FilingRef,
    IndexEntry,
    SyncResult,
    Throttled,
    audit_filings,
    available_at,
    belongs,
    cik_windows,
    coverage_gaps,
    list_filings,
    load_coverage_exceptions,
    missing_filings,
    parse_accepted,
    parse_filing_index,
    parse_submission_rows,
    periodic_gaps,
    rate_limited,
    retrying,
    security_segments,
    select_files,
    stale_reports,
    sync_company,
    trading_segments,
    wanted,
)
from asof.ingest.filing_text import chunk_text, html_to_text
from asof.ingest.http import FetchResult
from asof.store.db import init_db, make_engine
from asof.store.documents import audit_documents
from asof.store.models import Chunk, Document, Entity, Source
from asof.store.queries import visible_chunks, visible_documents
from tests.ingest.fakes import FakeTime, ScriptedFetch, raw_reply, reply

EDGAR = Path(__file__).parent / "fixtures" / "edgar"
NY = ZoneInfo("America/New_York")
SECOND = timedelta(seconds=1)
NOW = datetime(2026, 10, 21, 12, 0, tzinfo=UTC)

NVDA_CIK = 1045810
XOM_OLD_CIK = 34088
XOM_NEW_CIK = 2115436
# Measured from the security master: XOM moved to a new holding-company CIK on 2026-07-16.
XOM_SEGMENTS: list[tuple[int | None, date, date | None]] = [
    (XOM_OLD_CIK, date(2020, 1, 2), date(2026, 7, 16)),
    (XOM_NEW_CIK, date(2026, 7, 16), None),
]
# Measured from the security master: FIG traded under Fortress's CIK until 2025-08-19.
FIG_SEGMENTS: list[tuple[int | None, date, date | None]] = [
    (1380393, date(2020, 1, 2), date(2025, 8, 19)),
    (1579878, date(2025, 8, 19), None),
]


def load_json(name: str) -> dict[str, Any]:
    return json.loads((EDGAR / name).read_text())


def fixture_bytes(name: str) -> bytes:
    return (EDGAR / name).read_bytes()


def nvda_refs() -> dict[str, FilingRef]:
    rows = load_json("submissions-CIK0001045810.json")["filings"]["recent"]
    return {r.accession: r for r in parse_submission_rows(NVDA_CIK, rows)}


def xom_refs() -> dict[str, FilingRef]:
    rows = load_json("submissions-CIK0000034088.json")["filings"]["recent"]
    return {r.accession: r for r in parse_submission_rows(XOM_OLD_CIK, rows)}


def index(accession: str) -> list[IndexEntry]:
    return parse_filing_index((EDGAR / f"{accession}-index.htm").read_text())


def archive_url(cik: int, accession: str, filename: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/{filename}"


def index_url(cik: int, accession: str) -> str:
    return archive_url(cik, accession, f"{accession}-index.htm")


def submissions_url(cik: int) -> str:
    return f"https://data.sec.gov/submissions/CIK{cik:010d}.json"


class RoutedFetch:
    """Answers by URL instead of by order, so tests do not fix the order filings are processed.

    Imitates SEC's static endpoints: the same URL always returns the same bytes (measured: the
    submissions JSON, index pages and archive files are plain GETs with no paging parameters).
    Unknown URLs fail the test, so an unexpected download is caught.
    """

    def __init__(self, routes: Mapping[str, FetchResult]) -> None:
        self.routes = dict(routes)
        self.calls: list[tuple[str, dict[str, str]]] = []

    def __call__(self, url: str, params: Mapping[str, str]) -> FetchResult:
        self.calls.append((url, dict(params)))
        if url not in self.routes:
            raise AssertionError(f"unexpected request: {url}")
        return self.routes[url]

    @property
    def urls(self) -> list[str]:
        return [url for url, _ in self.calls]


# --- parse_submission_rows ------------------------------------------------------------------


def test_parse_real_nvda_rows() -> None:
    refs = nvda_refs()
    assert len(refs) == 5
    assert refs["0001045810-26-000073"] == FilingRef(
        cik=NVDA_CIK,
        accession="0001045810-26-000073",
        form="8-K",
        items=("2.02", "9.01"),
        filing_date=date(2026, 8, 26),
        report_date=date(2026, 8, 26),
        primary_document="nvda-20260826.htm",
    )
    ten_q = refs["0001045810-26-000075"]
    assert (ten_q.form, ten_q.report_date, ten_q.primary_document) == (
        "10-Q",
        date(2026, 7, 26),
        "nvda-20260726.htm",
    )


def test_acceptance_is_read_from_the_index_page_as_new_york_time() -> None:
    # Measured: the index page of 0001045810-26-000073 says "Accepted 2026-08-26 16:21:19" (New
    # York), as does the EDGAR header. Reading that clock as UTC would date the filing four hours
    # early -- a look-ahead leak.
    accepted = parse_accepted((EDGAR / f"{NVDA_8K}-index.htm").read_text())
    assert accepted == datetime(2026, 8, 26, 20, 21, 19, tzinfo=UTC)
    assert accepted.astimezone(NY).replace(tzinfo=None) == datetime(2026, 8, 26, 16, 21, 19)


def test_the_listing_acceptance_field_is_not_used() -> None:
    # Measured: the JSON says 2026-08-27T00:21:19Z for the filing accepted 16:21:19 New York
    # (20:21:19 UTC). Nothing on a FilingRef may carry that wrong clock.
    ref = nvda_refs()["0001045810-26-000073"]
    assert not any(isinstance(v, datetime) for v in vars(ref).values())


def test_index_page_without_an_accepted_line_is_an_error() -> None:
    with pytest.raises(ValueError):
        parse_accepted("<html><body>no such line</body></html>")


def test_blank_items_and_report_date() -> None:
    rows = load_json("submissions-CIK0001045810.json")["filings"]["recent"]
    # Measured: periodic forms carry items "" (the NVDA 10-Q row). Assumed: reportDate "" for
    # forms without a period; made here by blanking the 10-Q row (see module docstring).
    i = rows["accessionNumber"].index("0001045810-26-000075")
    one_row = {key: [values[i]] for key, values in rows.items()}
    one_row["reportDate"] = [""]
    (ref,) = parse_submission_rows(NVDA_CIK, one_row)
    assert ref.items == ()
    assert ref.report_date is None


def test_parse_history_page_rows() -> None:
    # Measured: history pages are the bare columnar dict, no "filings" wrapper.
    refs = parse_submission_rows(XOM_OLD_CIK, load_json("CIK0000034088-submissions-001.json"))
    assert [r.filing_date for r in refs] == [date(2019, 11, 1), date(2019, 8, 2), date(2019, 4, 26)]
    assert all(r.items == ("2.02", "7.01") for r in refs)


# --- list_filings ---------------------------------------------------------------------------


def test_list_filings_reads_recent_and_skips_old_history_pages() -> None:
    fetch = ScriptedFetch([raw_reply(200, fixture_bytes("submissions-CIK0001045810.json"))])
    refs = list_filings(fetch, NVDA_CIK, date(2026, 1, 1))
    assert fetch.urls == [submissions_url(NVDA_CIK)]  # history page ends 2020-09-15
    assert sorted(r.accession for r in refs) == [
        "0001045810-26-000021",
        "0001045810-26-000073",
        "0001045810-26-000075",
        "0001045810-26-000078",
    ]


def test_since_equal_to_filing_date_is_kept_one_day_later_dropped() -> None:
    body = fixture_bytes("submissions-CIK0001045810.json")
    on_day = list_filings(ScriptedFetch([raw_reply(200, body)]), NVDA_CIK, date(2026, 8, 26))
    day_after = list_filings(ScriptedFetch([raw_reply(200, body)]), NVDA_CIK, date(2026, 8, 27))
    assert sorted(r.accession for r in on_day) == [
        "0001045810-26-000073",
        "0001045810-26-000075",
        "0001045810-26-000078",
    ]
    assert [r.accession for r in day_after] == ["0001045810-26-000078"]


def test_history_page_fetched_only_when_it_reaches_since() -> None:
    routes = {
        submissions_url(XOM_OLD_CIK): raw_reply(
            200, fixture_bytes("submissions-CIK0000034088.json")
        ),
        "https://data.sec.gov/submissions/CIK0000034088-submissions-001.json": raw_reply(
            200, fixture_bytes("CIK0000034088-submissions-001.json")
        ),
        # -002 (filingTo 2005-10-03) is deliberately absent: requesting it fails the test.
    }
    fetch = RoutedFetch(routes)
    refs = list_filings(fetch, XOM_OLD_CIK, date(2019, 1, 1))
    assert sorted(fetch.urls) == sorted(routes)
    assert len(refs) == 7
    assert min(r.filing_date for r in refs) == date(2019, 4, 26)


def test_history_page_ending_on_since_is_fetched_and_one_day_before_is_not() -> None:
    main = raw_reply(200, fixture_bytes("submissions-CIK0000034088.json"))
    page = raw_reply(200, fixture_bytes("CIK0000034088-submissions-001.json"))
    page_url = "https://data.sec.gov/submissions/CIK0000034088-submissions-001.json"

    fetch = RoutedFetch({submissions_url(XOM_OLD_CIK): main, page_url: page})
    refs = list_filings(fetch, XOM_OLD_CIK, date(2020, 1, 2))  # page filingTo == 2020-01-02
    assert page_url in fetch.urls
    assert len(refs) == 4  # the page's 2019 rows are older than since

    fetch = RoutedFetch({submissions_url(XOM_OLD_CIK): main})
    assert len(list_filings(fetch, XOM_OLD_CIK, date(2020, 1, 3))) == 4
    assert fetch.urls == [submissions_url(XOM_OLD_CIK)]


@pytest.mark.parametrize("status", [403, 404, 503])
def test_non_200_raises(status: int) -> None:
    # Assumed: SEC answers 403 to clients without a User-Agent and 503 under load (documented in
    # SEC's "Accessing EDGAR Data" fair-access page). Error type is not part of the contract;
    # the status code must be in the message.
    fetch = ScriptedFetch([reply(status)])
    with pytest.raises((ValueError, RuntimeError), match=str(status)):
        list_filings(fetch, NVDA_CIK, date(2026, 1, 1))


# --- available_at (leakage) -----------------------------------------------------------------

# Measured acceptance (index page "Accepted" line == EDGAR header), New York time.
ACCEPTED_NY = {
    "0001045810-26-000073": datetime(2026, 8, 26, 16, 21, 19),
    "0001045810-26-000075": datetime(2026, 8, 26, 16, 36, 0),
    "0001045810-26-000021": datetime(2026, 2, 25, 16, 42, 19),
    "0001045810-23-000175": datetime(2023, 8, 25, 19, 36, 34),
    "0000034088-26-000065": datetime(2026, 5, 1, 6, 31, 52),
    "0000034088-22-000071": datetime(2022, 12, 2, 18, 11, 9),
}


def accepted(accession: str) -> datetime:
    return ACCEPTED_NY[accession].replace(tzinfo=NY).astimezone(UTC)


def test_measured_acceptance_matches_the_saved_index_pages() -> None:
    for accession in ("0001045810-26-000073", "0001045810-26-000075", "0000034088-26-000065"):
        html = (EDGAR / f"{accession}-index.htm").read_text()
        assert parse_accepted(html) == accepted(accession)


def test_available_at_is_the_acceptance_time_when_filed_in_hours() -> None:
    ref = nvda_refs()["0001045810-26-000073"]
    assert available_at(ref, accepted(ref.accession)) == datetime(
        2026, 8, 26, 20, 21, 19, tzinfo=UTC
    )


def test_late_filing_is_not_available_until_its_filing_date() -> None:
    # Measured: accepted 2023-08-25 19:36:34 New York (a Friday evening), filingDate 2023-08-28.
    ref = nvda_refs()["0001045810-23-000175"]
    assert ref.filing_date == date(2023, 8, 28)
    assert available_at(ref, accepted(ref.accession)) == datetime(
        2023, 8, 28, 10, 0, tzinfo=UTC
    )  # 06:00 EDT


def test_late_filing_in_winter_uses_eastern_standard_time() -> None:
    # Measured: XOM 8-K 5.02 accepted 2022-12-02 18:11:09 New York, filingDate 2022-12-05.
    ref = xom_refs()["0000034088-22-000071"]
    assert available_at(ref, accepted(ref.accession)) == datetime(
        2022, 12, 5, 11, 0, tzinfo=UTC
    )  # 06:00 EST


def test_after_close_filing_that_kept_its_date_is_available_at_acceptance() -> None:
    # Measured: NVDA 10-K accepted 2026-02-25 16:42:19 New York kept filingDate 2026-02-25.
    ref = nvda_refs()["0001045810-26-000021"]
    assert available_at(ref, accepted(ref.accession)) == datetime(
        2026, 2, 25, 21, 42, 19, tzinfo=UTC
    )  # EST


def test_early_morning_filing_is_available_at_acceptance() -> None:
    # Measured: XOM 8-K 2.02 accepted 2026-05-01 06:31:52 New York, after the 06:00 floor.
    ref = xom_refs()["0000034088-26-000065"]
    assert available_at(ref, accepted(ref.accession)) == datetime(
        2026, 5, 1, 10, 31, 52, tzinfo=UTC
    )


def test_available_at_never_precedes_acceptance_or_the_filing_date() -> None:
    for ref in [*nvda_refs().values(), *xom_refs().values()]:
        if ref.accession not in ACCEPTED_NY:
            continue
        at = available_at(ref, accepted(ref.accession))
        assert at.utcoffset() == timedelta(0)
        assert at >= accepted(ref.accession)
        assert at.astimezone(NY).date() >= ref.filing_date


# --- wanted -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("form", "items", "expected"),
    [
        ("10-K", (), True),
        ("10-K/A", (), True),
        ("10-Q", (), True),
        ("10-Q/A", (), True),
        ("20-F", (), True),
        ("20-F/A", (), True),
        ("40-F", (), True),
        ("40-F/A", (), True),
        ("6-K", (), True),
        ("6-K/A", (), True),
        ("8-K", ("2.02", "9.01"), True),
        ("8-K", ("2.02", "7.01"), True),
        ("8-K/A", ("2.02",), True),
        ("8-K", ("7.01", "9.01"), True),
        ("8-K/A", ("7.01",), True),
        ("8-K", ("8.01",), False),
        ("8-K", ("5.02",), False),
        ("8-K", ("1.01", "2.01", "3.01", "3.03", "5.02", "5.03", "9.01"), False),
        ("8-K", (), False),
        ("8-K/A", ("9.01",), False),
        ("4", (), False),
        ("S-8", (), False),
        ("424B2", (), False),
        ("DEF 14A", (), False),
        ("SC 13G", (), False),
    ],
)
def test_wanted(form: str, items: tuple[str, ...], expected: bool) -> None:
    base = nvda_refs()["0001045810-26-000073"]
    assert wanted(replace(base, form=form, items=items)) is expected


def test_wanted_on_the_real_rows() -> None:
    refs = {**nvda_refs(), **xom_refs()}
    assert sorted(a for a, r in refs.items() if wanted(r)) == [
        "0000034088-26-000065",  # 8-K 2.02,7.01
        "0000034088-26-000093",  # 10-Q
        "0001045810-23-000175",  # 10-Q
        "0001045810-26-000021",  # 10-K
        "0001045810-26-000073",  # 8-K 2.02,9.01
        "0001045810-26-000075",  # 10-Q
    ]


# --- parse_filing_index -----------------------------------------------------------------------


def test_parse_real_8k_index_reads_the_document_table_only() -> None:
    entries = index("0001045810-26-000073")
    assert entries[:4] == [
        IndexEntry(seq=1, description="8-K", filename="nvda-20260826.htm", doc_type="8-K"),
        IndexEntry(seq=2, description="EX-99.1", filename="q2fy27pr.htm", doc_type="EX-99.1"),
        IndexEntry(
            seq=3, description="EX-99.2", filename="q2fy27cfocommentary.htm", doc_type="EX-99.2"
        ),
        IndexEntry(seq=7, description="NVDA LOGO", filename="nvdalogoa19.jpg", doc_type="GRAPHIC"),
    ]
    # Chosen: the "Complete submission text file" row is kept, with no seq and no type.
    assert entries[4:] == [
        IndexEntry(
            seq=None,
            description="Complete submission text file",
            filename="0001045810-26-000073.txt",
            doc_type="",
        )
    ]
    # The second table ("Data Files": EX-101.*, XML) is not read.
    assert not any(e.doc_type.startswith("EX-101") for e in entries)


def test_parse_index_with_blank_descriptions() -> None:
    # Measured: TSM's 6-K lists its slide images with an empty Description cell.
    entries = index("0001046179-26-000726")
    graphics = [e for e in entries if e.doc_type == "GRAPHIC"]
    assert len(graphics) == 13
    assert all(e.description == "" for e in graphics)
    assert entries[1] == IndexEntry(
        seq=2, description="EX-99.1", filename="a3q26e_withguidancexfinal.htm", doc_type="EX-99.1"
    )


# --- select_files -----------------------------------------------------------------------------


def _filenames(entries: list[IndexEntry]) -> list[str]:
    return [e.filename for e in entries]


def test_select_earnings_8k_exhibits() -> None:
    ref = xom_refs()["0000034088-26-000065"]
    assert _filenames(select_files(ref, index(ref.accession))) == [
        "livef8k1q26991.htm",
        "livef8k1q26992.htm",
    ]
    nvda = nvda_refs()["0001045810-26-000073"]
    assert _filenames(select_files(nvda, index(nvda.accession))) == [
        "q2fy27pr.htm",
        "q2fy27cfocommentary.htm",
    ]


def test_select_10q_primary_document_only() -> None:
    ref = nvda_refs()["0001045810-26-000075"]
    # Not EX-10.1, EX-31.*, EX-32.*, nor the logo.
    assert _filenames(select_files(ref, index(ref.accession))) == ["nvda-20260726.htm"]


def test_select_6k_exhibits() -> None:
    # Built from the measured index page: TSM CIK 1046179, filed 2026-10-15, primary
    # tsm-20261015x6k.htm.
    ref = FilingRef(
        cik=1046179,
        accession="0001046179-26-000726",
        form="6-K",
        items=(),
        filing_date=date(2026, 10, 15),
        report_date=date(2026, 9, 30),
        primary_document="tsm-20261015x6k.htm",
    )
    assert _filenames(select_files(ref, index(ref.accession))) == [
        "a3q26e_withguidancexfinal.htm",
        "a3q26presentatione.htm",
    ]


def test_select_40f_exhibits_in_index_order() -> None:
    # Built from the measured index page: Cameco CIK 1009001, filed 2026-03-19, primary
    # d34605d40f.htm.
    ref = FilingRef(
        cik=1009001,
        accession="0001193125-26-116229",
        form="40-F",
        items=(),
        filing_date=date(2026, 3, 19),
        report_date=date(2025, 12, 31),
        primary_document="d34605d40f.htm",
    )
    selected = select_files(ref, index(ref.accession))
    assert [e.doc_type for e in selected] == [f"EX-99.{i}" for i in range(1, 17)]
    assert selected[1].filename == "d34605dex992.htm"  # "iXBRL" marker not part of the name


def test_8k_without_ex99_falls_back_to_the_primary_document() -> None:
    ref = nvda_refs()["0001045810-26-000073"]
    entries = [e for e in index(ref.accession) if not e.doc_type.startswith("EX-99")]
    assert _filenames(select_files(ref, entries)) == ["nvda-20260826.htm"]


def test_amended_periodic_form_selects_its_primary_document() -> None:
    ref = replace(nvda_refs()["0001045810-26-000075"], form="10-Q/A")
    assert _filenames(select_files(ref, index(ref.accession))) == ["nvda-20260726.htm"]


# --- cik_windows / belongs --------------------------------------------------------------------


def test_cik_windows_for_xom() -> None:
    assert cik_windows(XOM_NEW_CIK, XOM_SEGMENTS) == [
        CikWindow(XOM_OLD_CIK, date(2020, 1, 2), date(2026, 7, 16)),
        CikWindow(XOM_NEW_CIK, None, None),
    ]


def test_cik_windows_for_fig() -> None:
    assert cik_windows(1579878, FIG_SEGMENTS) == [
        CikWindow(1380393, date(2020, 1, 2), date(2025, 8, 19)),
        CikWindow(1579878, None, None),
    ]


def test_cik_windows_entity_cik_without_segments_and_cikless_segments() -> None:
    segments: list[tuple[int | None, date, date | None]] = [(None, date(2020, 1, 2), None)]
    assert cik_windows(NVDA_CIK, segments) == [CikWindow(NVDA_CIK, None, None)]
    assert cik_windows(NVDA_CIK, []) == [CikWindow(NVDA_CIK, None, None)]


def test_cik_windows_one_window_per_segment_of_another_cik() -> None:
    segments: list[tuple[int | None, date, date | None]] = [
        (777, date(2021, 1, 1), date(2021, 6, 1)),
        (NVDA_CIK, date(2021, 6, 1), date(2022, 1, 1)),
        (777, date(2022, 1, 1), date(2023, 1, 1)),
        (NVDA_CIK, date(2023, 1, 1), None),
    ]
    assert cik_windows(NVDA_CIK, segments) == [
        CikWindow(777, date(2021, 1, 1), date(2021, 6, 1)),
        CikWindow(777, date(2022, 1, 1), date(2023, 1, 1)),
        CikWindow(NVDA_CIK, None, None),
    ]


def test_old_cik_filing_before_the_switch_belongs() -> None:
    windows = cik_windows(XOM_NEW_CIK, XOM_SEGMENTS)
    assert belongs(xom_refs()["0000034088-26-000065"], windows)  # 8-K 2.02 of 2026-05-01


def test_old_cik_10q_after_the_switch_does_not_belong() -> None:
    # Measured: the old XOM CIK still lists a 10-Q of 2026-08-03. It is a joint filing that the
    # new holding company's listing carries too, so it arrives through the new CIK; through the
    # old one it must not, or any later filing of the old registrant would be attached as well.
    windows = cik_windows(XOM_NEW_CIK, XOM_SEGMENTS)
    assert not belongs(xom_refs()["0000034088-26-000093"], windows)


def test_filings_outside_every_window_do_not_belong() -> None:
    windows = cik_windows(XOM_NEW_CIK, XOM_SEGMENTS)
    page = load_json("CIK0000034088-submissions-001.json")
    for ref in parse_submission_rows(XOM_OLD_CIK, page):  # 2019, before the 2020-01-02 start
        assert not belongs(ref, windows)
    assert not belongs(nvda_refs()["0001045810-26-000073"], windows)  # unrelated CIK


def test_entity_cik_belongs_at_any_date() -> None:
    windows = cik_windows(XOM_NEW_CIK, XOM_SEGMENTS)
    base = xom_refs()["0000034088-26-000065"]
    for day in [date(1999, 1, 4), date(2030, 1, 4)]:
        assert belongs(replace(base, cik=XOM_NEW_CIK, filing_date=day), windows)


def test_window_bounds_use_the_filing_date() -> None:
    windows = cik_windows(XOM_NEW_CIK, XOM_SEGMENTS)
    base = xom_refs()["0000034088-26-000065"]

    def on(day: date) -> FilingRef:
        return replace(base, filing_date=day)

    assert belongs(on(date(2026, 7, 15)), windows)  # last day inside [start, end)
    assert not belongs(on(date(2026, 7, 16)), windows)  # end is exclusive
    assert not belongs(on(date(2020, 1, 1)), windows)  # before the start
    assert belongs(on(date(2020, 1, 2)), windows)  # start is inclusive


# --- rate_limited -----------------------------------------------------------------------------


def test_rate_limited_spaces_calls_and_passes_results_through() -> None:
    clock = FakeTime(start=1_000.0)
    seen: list[float] = []
    inner = ScriptedFetch([reply(200, {"n": i}) for i in range(4)])

    def timed(url: str, params: Mapping[str, str]) -> FetchResult:
        seen.append(clock.monotonic())
        return inner(url, params)

    fetch = rate_limited(timed, 10.0, clock=clock.monotonic, sleep=clock.sleep)

    first = fetch("https://data.sec.gov/a", {"q": "1"})
    clock.advance(0.04)
    fetch("https://data.sec.gov/b", {})
    fetch("https://data.sec.gov/c", {})
    clock.advance(5.0)
    fetch("https://data.sec.gov/d", {})

    assert json.loads(first.body) == {"n": 0}
    assert inner.calls[0] == ("https://data.sec.gov/a", {"q": "1"})
    assert seen == pytest.approx([1_000.0, 1_000.1, 1_000.2, 1_005.2])
    assert sum(clock.sleeps) == pytest.approx(0.16)  # waits only as long as needed


# --- sync_company -----------------------------------------------------------------------------


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    eng = make_engine(tmp_path / "meta.db")
    init_db(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    with Session(engine) as s:
        yield s


def add_entity(session: Session, ticker: str, cik: int, name: str) -> Entity:
    entity = Entity(kind="company", ticker=ticker, name=name, cik=cik)
    session.add(entity)
    session.commit()
    session.refresh(entity)
    return entity


PRESS_RELEASE = fixture_bytes("nvda-q2fy27pr.htm")
NVDA_8K = "0001045810-26-000073"
NVDA_10Q = "0001045810-26-000075"
NVDA_FILES = [
    archive_url(NVDA_CIK, NVDA_10Q, "nvda-20260726.htm"),
    archive_url(NVDA_CIK, NVDA_8K, "q2fy27pr.htm"),
    archive_url(NVDA_CIK, NVDA_8K, "q2fy27cfocommentary.htm"),
]


def nvda_routes() -> dict[str, FetchResult]:
    # Every file body is the real EX-99.1: the sync must not care what a body says, only store it.
    routes = {
        submissions_url(NVDA_CIK): raw_reply(200, fixture_bytes("submissions-CIK0001045810.json")),
        index_url(NVDA_CIK, NVDA_8K): raw_reply(200, fixture_bytes(f"{NVDA_8K}-index.htm")),
        index_url(NVDA_CIK, NVDA_10Q): raw_reply(200, fixture_bytes(f"{NVDA_10Q}-index.htm")),
    }
    routes.update({url: raw_reply(200, PRESS_RELEASE) for url in NVDA_FILES})
    return routes


def sync_nvda(session: Session, fetch: RoutedFetch, tmp_path: Path, entity: Entity) -> SyncResult:
    # since 2026-08-01 keeps 8-K 8.01 (not wanted), the 10-Q and the 8-K 2.02.
    result = sync_company(
        session,
        fetch,
        entity=entity,
        windows=cik_windows(NVDA_CIK, [(NVDA_CIK, date(2020, 1, 2), None)]),
        since=date(2026, 8, 1),
        raw_dir=tmp_path / "raw",
        text_dir=tmp_path / "text",
        now=NOW,
    )
    session.commit()
    return result


def documents(session: Session) -> dict[str, Document]:
    return {d.external_id: d for d in session.exec(select(Document)).all()}


def test_sync_stores_one_document_per_selected_file(session: Session, tmp_path: Path) -> None:
    nvda = add_entity(session, "NVDA", NVDA_CIK, "NVIDIA CORP")
    fetch = RoutedFetch(nvda_routes())
    result = sync_nvda(session, fetch, tmp_path, nvda)

    assert result == SyncResult(added=3, skipped=0, files=3)
    docs = documents(session)
    assert sorted(docs) == [
        "0001045810-26-000073/q2fy27cfocommentary.htm",
        "0001045810-26-000073/q2fy27pr.htm",
        "0001045810-26-000075/nvda-20260726.htm",
    ]
    assert sorted(set(fetch.urls) & set(NVDA_FILES)) == sorted(NVDA_FILES)

    source = session.exec(select(Source).where(Source.name == "sec_edgar")).one()
    pr = docs["0001045810-26-000073/q2fy27pr.htm"]
    assert pr.source_id == source.id
    assert pr.entity_id == nvda.id
    assert pr.doc_type == "8-K"
    assert pr.title == "8-K 2.02,9.01 EX-99.1 EX-99.1"
    assert pr.url == archive_url(NVDA_CIK, NVDA_8K, "q2fy27pr.htm")
    assert pr.source_time == datetime(2026, 8, 26, tzinfo=UTC)
    assert pr.available_at == datetime(2026, 8, 26, 20, 21, 19, tzinfo=UTC)
    assert pr.received_at == NOW
    assert pr.supersedes_id is None

    ten_q = docs["0001045810-26-000075/nvda-20260726.htm"]
    assert ten_q.doc_type == "10-Q"
    assert ten_q.title == "10-Q 10-Q 10-Q"
    assert ten_q.source_time == datetime(2026, 7, 26, tzinfo=UTC)  # report date, not filing date
    assert ten_q.available_at == datetime(2026, 8, 26, 20, 36, tzinfo=UTC)  # 16:36 New York


def test_sync_saves_raw_and_text_files_that_match_the_store(
    session: Session, tmp_path: Path
) -> None:
    nvda = add_entity(session, "NVDA", NVDA_CIK, "NVIDIA CORP")
    sync_nvda(session, RoutedFetch(nvda_routes()), tmp_path, nvda)

    raw = tmp_path / "raw" / str(NVDA_CIK) / NVDA_8K / "q2fy27pr.htm"
    assert raw.read_bytes() == PRESS_RELEASE
    text_file = tmp_path / "text" / NVDA_8K / "q2fy27pr.htm.txt"
    text = text_file.read_text(encoding="utf-8")
    assert text == html_to_text(PRESS_RELEASE.decode("utf-8"))

    pr = documents(session)[f"{NVDA_8K}/q2fy27pr.htm"]
    assert pr.content_hash == hashlib.sha256(text.encode("utf-8")).hexdigest()
    chunks = session.exec(
        select(Chunk).where(Chunk.document_id == pr.id).order_by(col(Chunk.ordinal))
    ).all()
    expected = chunk_text(text, form="8-K", exhibit="EX-99.1")
    assert [(c.char_start, c.char_end, c.section_path) for c in chunks] == [
        (e.char_start, e.char_end, e.section_path) for e in expected
    ]
    assert all(c.text == text[c.char_start : c.char_end] for c in chunks)
    assert audit_documents(session, tmp_path / "text") == []


def test_synced_filing_is_invisible_one_second_before_available_at(
    session: Session, tmp_path: Path
) -> None:
    nvda = add_entity(session, "NVDA", NVDA_CIK, "NVIDIA CORP")
    sync_nvda(session, RoutedFetch(nvda_routes()), tmp_path, nvda)
    released = datetime(2026, 8, 26, 20, 21, 19, tzinfo=UTC)  # 16:21:19 New York

    before = visible_documents(session, released - SECOND, entity_id=nvda.id)
    assert before == []
    assert visible_chunks(session, released - SECOND) == []

    at = visible_documents(session, released, entity_id=nvda.id)
    assert sorted(d.external_id for d in at) == [
        f"{NVDA_8K}/q2fy27cfocommentary.htm",
        f"{NVDA_8K}/q2fy27pr.htm",
    ]  # the 10-Q, accepted 15 minutes later, is still in the future
    assert {c.document_id for c in visible_chunks(session, released)} == {d.id for d in at}


def test_resync_adds_nothing_and_downloads_no_documents(session: Session, tmp_path: Path) -> None:
    nvda = add_entity(session, "NVDA", NVDA_CIK, "NVIDIA CORP")
    sync_nvda(session, RoutedFetch(nvda_routes()), tmp_path, nvda)
    doc_count = len(session.exec(select(Document)).all())
    chunk_count = len(session.exec(select(Chunk)).all())

    # Only the listing and index pages are served now; asking for any file fails the test.
    routes = {u: r for u, r in nvda_routes().items() if u not in NVDA_FILES}
    again = sync_nvda(session, RoutedFetch(routes), tmp_path, nvda)

    assert again == SyncResult(added=0, skipped=3, files=0)
    assert len(session.exec(select(Document)).all()) == doc_count
    assert len(session.exec(select(Chunk)).all()) == chunk_count


def test_raw_file_already_on_disk_is_reused_not_downloaded_or_overwritten(
    session: Session, tmp_path: Path
) -> None:
    nvda = add_entity(session, "NVDA", NVDA_CIK, "NVIDIA CORP")
    cached = b"<html><body><p>Cached copy of the press release.</p></body></html>"
    raw = tmp_path / "raw" / str(NVDA_CIK) / NVDA_8K / "q2fy27pr.htm"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(cached)

    routes = nvda_routes()
    del routes[archive_url(NVDA_CIK, NVDA_8K, "q2fy27pr.htm")]
    result = sync_nvda(session, RoutedFetch(routes), tmp_path, nvda)

    assert result == SyncResult(added=3, skipped=0, files=2)
    assert raw.read_bytes() == cached
    text = (tmp_path / "text" / NVDA_8K / "q2fy27pr.htm.txt").read_text(encoding="utf-8")
    assert text == "Cached copy of the press release."


def test_sync_drops_old_cik_filings_outside_the_window(session: Session, tmp_path: Path) -> None:
    xom = add_entity(session, "XOM", XOM_NEW_CIK, "Exxon Mobil Corp")
    accession = "0000034088-26-000065"
    # Assumed shape (see module docstring): the new CIK has no filings since `since`.
    empty_columns = {
        key: [] for key in load_json("submissions-CIK0000034088.json")["filings"]["recent"]
    }
    new_cik_listing = {
        "cik": f"{XOM_NEW_CIK:010d}",
        "filings": {"recent": empty_columns, "files": []},
    }
    routes = {
        submissions_url(XOM_OLD_CIK): raw_reply(
            200, fixture_bytes("submissions-CIK0000034088.json")
        ),
        submissions_url(XOM_NEW_CIK): reply(200, new_cik_listing),
        index_url(XOM_OLD_CIK, accession): raw_reply(200, fixture_bytes(f"{accession}-index.htm")),
        archive_url(XOM_OLD_CIK, accession, "livef8k1q26991.htm"): raw_reply(200, PRESS_RELEASE),
        archive_url(XOM_OLD_CIK, accession, "livef8k1q26992.htm"): raw_reply(200, PRESS_RELEASE),
        # Nothing for the post-switch 10-Q 0000034088-26-000093: requesting it fails the test.
    }
    fetch = RoutedFetch(routes)
    result = sync_company(
        session,
        fetch,
        entity=xom,
        windows=cik_windows(XOM_NEW_CIK, XOM_SEGMENTS),
        since=date(2026, 4, 1),
        raw_dir=tmp_path / "raw",
        text_dir=tmp_path / "text",
        now=NOW,
    )
    session.commit()

    assert result == SyncResult(added=2, skipped=0, files=2)
    assert not any("000003408826000093" in url for url in fetch.urls)
    docs = documents(session)
    assert sorted(docs) == [f"{accession}/livef8k1q26991.htm", f"{accession}/livef8k1q26992.htm"]
    assert all(d.entity_id == xom.id for d in docs.values())
    assert docs[f"{accession}/livef8k1q26991.htm"].title == "8-K 2.02,7.01 EX-99.1 NEWS RELEASE"
    assert (tmp_path / "raw" / str(XOM_OLD_CIK) / accession / "livef8k1q26991.htm").exists()


# --- security_segments ------------------------------------------------------------------------


def master_table(rows: list[tuple[str, str, date, date | None, int | None]]) -> pa.Table:
    """A security master with only the columns security_segments reads filled in."""
    return pa.Table.from_pylist(
        [
            {"security_id": s, "symbol": sym, "valid_from": f, "valid_to": t, "cik": c}
            for s, sym, f, t, c in rows
        ],
        schema=pa.schema(
            [
                ("security_id", pa.string()),
                ("symbol", pa.string()),
                ("valid_from", pa.date32()),
                ("valid_to", pa.date32()),
                ("cik", pa.int64()),
            ]
        ),
    )


def test_security_segments_follow_the_open_segment_to_its_security() -> None:
    # Measured rows of the real master for XOM, plus an unrelated security that once used "XOM".
    master = master_table(
        [
            ("S1", "XOM", date(2020, 1, 2), date(2026, 7, 16), XOM_OLD_CIK),
            ("S1", "XOM", date(2026, 7, 16), None, XOM_NEW_CIK),
            ("S9", "XOMA", date(2020, 1, 2), None, 791908),
        ]
    )
    assert security_segments(master, "XOM") == [("XOM", *segment) for segment in XOM_SEGMENTS]
    assert security_segments(master, "NVDA") == []


def test_security_segments_include_the_securitys_other_symbols() -> None:
    master = master_table(
        [
            ("S2", "FB", date(2020, 1, 2), date(2022, 6, 9), 1326801),
            ("S2", "META", date(2022, 6, 9), None, 1326801),
        ]
    )
    assert security_segments(master, "META") == [
        ("FB", 1326801, date(2020, 1, 2), date(2022, 6, 9)),
        ("META", 1326801, date(2022, 6, 9), None),
    ]


# --- retrying -----------------------------------------------------------------------------------


def test_retrying_retries_throttling_and_server_errors_then_passes_through() -> None:
    clock = FakeTime()
    inner = ScriptedFetch([reply(429), reply(503), reply(200, {"ok": True})])
    fetch = retrying(inner, attempts=4, sleep=clock.sleep)
    result = fetch("https://data.sec.gov/x", {})
    assert result.status == 200
    assert len(inner.calls) == 3
    assert len(clock.sleeps) == 2 and clock.sleeps[1] > clock.sleeps[0]  # backs off


def test_retrying_does_not_retry_a_missing_file() -> None:
    clock = FakeTime()
    inner = ScriptedFetch([reply(404)])
    assert retrying(inner, attempts=4, sleep=clock.sleep)("https://x.sec.gov/y", {}).status == 404
    assert clock.sleeps == []


def test_retrying_gives_up_with_the_last_answer() -> None:
    clock = FakeTime()
    inner = ScriptedFetch([reply(503)] * 3)
    assert retrying(inner, attempts=3, sleep=clock.sleep)("https://x.sec.gov/y", {}).status == 503
    assert len(inner.calls) == 3


def test_retrying_retries_network_errors_and_reraises_the_last() -> None:
    clock = FakeTime()
    calls: list[str] = []

    def flaky(url: str, params: Mapping[str, str]) -> FetchResult:
        calls.append(url)
        if len(calls) < 2:
            raise httpx.ConnectError("reset")
        return reply(200, {})

    assert retrying(flaky, attempts=3, sleep=clock.sleep)("https://x.sec.gov/y", {}).status == 200

    def down(url: str, params: Mapping[str, str]) -> FetchResult:
        raise httpx.ReadTimeout("slow")

    with pytest.raises(httpx.ReadTimeout):
        retrying(down, attempts=2, sleep=clock.sleep)("https://x.sec.gov/y", {})


# --- audit_filings / coverage_gaps --------------------------------------------------------------


def test_audit_filings_is_clean_after_a_sync(session: Session, tmp_path: Path) -> None:
    nvda = add_entity(session, "NVDA", NVDA_CIK, "NVIDIA CORP")
    sync_nvda(session, RoutedFetch(nvda_routes()), tmp_path, nvda)
    assert audit_filings(session, tmp_path / "raw") == []


def test_audit_filings_flags_a_document_public_before_its_acceptance(
    session: Session, tmp_path: Path
) -> None:
    nvda = add_entity(session, "NVDA", NVDA_CIK, "NVIDIA CORP")
    sync_nvda(session, RoutedFetch(nvda_routes()), tmp_path, nvda)
    doc = documents(session)[f"{NVDA_8K}/q2fy27pr.htm"]
    doc.available_at = doc.available_at - timedelta(hours=4)  # the JSON-field mistake, reversed
    session.add(doc)
    session.commit()
    problems = audit_filings(session, tmp_path / "raw")
    assert len(problems) == 1
    assert doc.external_id in problems[0]


def test_audit_filings_flags_a_missing_index_page(session: Session, tmp_path: Path) -> None:
    nvda = add_entity(session, "NVDA", NVDA_CIK, "NVIDIA CORP")
    sync_nvda(session, RoutedFetch(nvda_routes()), tmp_path, nvda)
    (tmp_path / "raw" / str(NVDA_CIK) / NVDA_10Q / f"{NVDA_10Q}-index.htm").unlink()
    problems = audit_filings(session, tmp_path / "raw")
    assert len(problems) == 1
    assert NVDA_10Q in problems[0]


def test_coverage_gaps_name_companies_missing_earnings_or_periodic_reports(
    session: Session, tmp_path: Path
) -> None:
    nvda = add_entity(session, "NVDA", NVDA_CIK, "NVIDIA CORP")
    add_entity(session, "UUUU", 1385849, "Energy Fuels Inc")
    session.add(Entity(kind="etf", ticker="SPY", name="SPDR S&P 500", cik=884394))
    session.commit()
    assert coverage_gaps(session) == [
        ("NVDA", "earnings"),
        ("NVDA", "periodic"),
        ("UUUU", "earnings"),
        ("UUUU", "periodic"),
    ]
    sync_nvda(session, RoutedFetch(nvda_routes()), tmp_path, nvda)
    # NVDA now has an earnings 8-K and a 10-Q; funds are never expected to file either.
    assert coverage_gaps(session) == [("UUUU", "earnings"), ("UUUU", "periodic")]


def test_coverage_exceptions_need_every_field(tmp_path: Path) -> None:
    path = tmp_path / "docs_exceptions.yaml"
    path.write_text(
        "- key: 'UUUU:earnings'\n  reason: 'press releases not furnished on 8-K'\n"
        "  added: 2026-10-21\n  by: greyson\n"
    )
    assert load_coverage_exceptions(path) == {
        "UUUU:earnings": "press releases not furnished on 8-K"
    }
    path.write_text("- key: 'UUUU:earnings'\n  reason: ''\n  added: 2026-10-21\n  by: greyson\n")
    with pytest.raises(ValueError):
        load_coverage_exceptions(path)
    path.write_text("- key: 'UUUU:quarterly'\n  reason: 'x'\n  added: 2026-10-21\n  by: greyson\n")
    with pytest.raises(ValueError):
        load_coverage_exceptions(path)
    assert load_coverage_exceptions(tmp_path / "absent.yaml") == {}


# --- review follow-ups ---------------------------------------------------------------------------


def test_parse_accepted_on_a_winter_index_page() -> None:
    # Measured: NVDA 10-K 0001045810-26-000021, "Accepted 2026-02-25 16:42:19" (EST, UTC-5).
    html = (EDGAR / "0001045810-26-000021-index.htm").read_text()
    assert parse_accepted(html) == datetime(2026, 2, 25, 21, 42, 19, tzinfo=UTC)


def test_10k_also_selects_ex13_but_20f_does_not() -> None:
    # Documented: in a 10-K, EX-13 is the annual report to shareholders and can hold the body;
    # measured: in TSM's 20-F, EX-13.1/13.2 are 4-5 KB section 906 certifications.
    ref = replace(nvda_refs()["0001045810-26-000021"], primary_document="nvda-10k.htm")
    entries = [
        IndexEntry(1, "10-K", "nvda-10k.htm", "10-K"),
        IndexEntry(2, "ANNUAL REPORT", "ex13.htm", "EX-13"),
        IndexEntry(3, "", "ex31.htm", "EX-31.1"),
    ]
    assert _filenames(select_files(ref, entries)) == ["nvda-10k.htm", "ex13.htm"]
    twenty_f = replace(ref, form="20-F")
    assert _filenames(select_files(twenty_f, entries)) == ["nvda-10k.htm"]


def test_trading_segments_keep_another_ciks_segment_only_where_it_traded() -> None:
    # Measured: FIG's old segment carries Fortress's CIK over zero-volume placeholder bars;
    # XOM's old segment traded every day until the switch.
    segments: list[tuple[str, int | None, date, date | None]] = [
        ("FIG", 1380393, date(2020, 1, 2), date(2025, 8, 19)),
        ("FIG", 1579878, date(2025, 8, 19), None),
    ]
    asked: list[tuple[str, date, date | None]] = []

    def no_trades(symbol: str, start: date, end: date | None) -> tuple[date, date] | None:
        asked.append((symbol, start, end))
        return None

    assert trading_segments(1579878, segments, no_trades) == [(1579878, date(2025, 8, 19), None)]
    assert asked == [("FIG", date(2020, 1, 2), date(2025, 8, 19))]  # own CIK is never clipped

    xom: list[tuple[str, int | None, date, date | None]] = [
        ("XOM", XOM_OLD_CIK, date(2020, 1, 2), date(2026, 7, 16)),
        ("XOM", XOM_NEW_CIK, date(2026, 7, 16), None),
    ]
    traded = {("XOM", date(2020, 1, 2)): (date(2020, 1, 2), date(2026, 7, 15))}
    assert trading_segments(XOM_NEW_CIK, xom, lambda s, f, t: traded.get((s, f))) == [
        (XOM_OLD_CIK, date(2020, 1, 2), date(2026, 7, 16)),  # last traded day + 1, exclusive
        (XOM_NEW_CIK, date(2026, 7, 16), None),
    ]


def test_wanted_filing_with_no_selectable_file_is_reported(
    session: Session, tmp_path: Path
) -> None:
    nvda = add_entity(session, "NVDA", NVDA_CIK, "NVIDIA CORP")
    routes = nvda_routes()
    # The listing names nvda-20260726.htm as the 10-Q's primary document; an index without it
    # (assumed: a renamed or missing file) must not pass silently.
    page = fixture_bytes(f"{NVDA_10Q}-index.htm").replace(b"nvda-20260726.htm", b"renamed.htm")
    routes[index_url(NVDA_CIK, NVDA_10Q)] = raw_reply(200, page)
    result = sync_nvda(session, RoutedFetch(routes), tmp_path, nvda)
    assert result.added == 2
    assert len(result.errors) == 1 and NVDA_10Q in result.errors[0]


def test_one_failing_file_does_not_stop_the_others(session: Session, tmp_path: Path) -> None:
    nvda = add_entity(session, "NVDA", NVDA_CIK, "NVIDIA CORP")
    routes = nvda_routes()
    routes[archive_url(NVDA_CIK, NVDA_8K, "q2fy27cfocommentary.htm")] = reply(404)
    result = sync_nvda(session, RoutedFetch(routes), tmp_path, nvda)
    assert result.added == 2
    assert len(result.errors) == 1 and "q2fy27cfocommentary.htm" in result.errors[0]


def test_throttling_stops_the_run(session: Session, tmp_path: Path) -> None:
    # Documented: SEC answers 403 "Request Rate Threshold Exceeded" to a client over the limit;
    # continuing would only extend the block.
    nvda = add_entity(session, "NVDA", NVDA_CIK, "NVIDIA CORP")
    routes = nvda_routes()
    routes[archive_url(NVDA_CIK, NVDA_8K, "q2fy27pr.htm")] = reply(403)
    with pytest.raises(Throttled):
        sync_nvda(session, RoutedFetch(routes), tmp_path, nvda)


def test_a_throttle_page_served_with_200_is_not_cached(session: Session, tmp_path: Path) -> None:
    # Documented: SEC's rate-limit page reads "Request Rate Threshold Exceeded"; assumed it can
    # arrive with status 200 from a cache in front of EDGAR.
    nvda = add_entity(session, "NVDA", NVDA_CIK, "NVIDIA CORP")
    routes = nvda_routes()
    throttle = b"<html><body><h1>Request Rate Threshold Exceeded</h1></body></html>"
    routes[archive_url(NVDA_CIK, NVDA_8K, "q2fy27pr.htm")] = raw_reply(200, throttle)
    routes[index_url(NVDA_CIK, NVDA_10Q)] = raw_reply(200, throttle)
    result = sync_nvda(session, RoutedFetch(routes), tmp_path, nvda)
    assert result.added == 1  # only the CFO commentary
    assert len(result.errors) == 2
    assert not (tmp_path / "raw" / str(NVDA_CIK) / NVDA_8K / "q2fy27pr.htm").exists()
    assert not (tmp_path / "raw" / str(NVDA_CIK) / NVDA_10Q / f"{NVDA_10Q}-index.htm").exists()


def test_a_file_with_no_text_is_counted_not_stored(session: Session, tmp_path: Path) -> None:
    # Measured shape: TSM's 6-K presentation exhibit is a page of slide images.
    nvda = add_entity(session, "NVDA", NVDA_CIK, "NVIDIA CORP")
    routes = nvda_routes()
    slides = b"<html><body><img src='slide1.jpg'><img src='slide2.jpg'></body></html>"
    routes[archive_url(NVDA_CIK, NVDA_8K, "q2fy27cfocommentary.htm")] = raw_reply(200, slides)
    result = sync_nvda(session, RoutedFetch(routes), tmp_path, nvda)
    assert (result.added, result.empty, result.errors) == (2, 1, ())
    assert f"{NVDA_8K}/q2fy27cfocommentary.htm" not in documents(session)


def test_a_document_held_by_another_company_is_a_conflict(session: Session, tmp_path: Path) -> None:
    nvda = add_entity(session, "NVDA", NVDA_CIK, "NVIDIA CORP")
    sync_nvda(session, RoutedFetch(nvda_routes()), tmp_path, nvda)
    # Assumed: a joint filing listed under two watchlist companies' CIKs.
    other = add_entity(session, "NVDX", 9999999, "Joint filer")
    routes = {u: r for u, r in nvda_routes().items() if u not in NVDA_FILES}
    result = sync_nvda(session, RoutedFetch(routes), tmp_path, other)
    assert result.skipped == 0
    assert len(result.errors) == 3 and all("NVDA" in e for e in result.errors)


def test_plain_text_exhibit_keeps_its_lines(session: Session, tmp_path: Path) -> None:
    nvda = add_entity(session, "NVDA", NVDA_CIK, "NVIDIA CORP")
    routes = nvda_routes()
    page = fixture_bytes(f"{NVDA_8K}-index.htm").replace(b"q2fy27pr.htm", b"q2fy27pr.txt")
    routes[index_url(NVDA_CIK, NVDA_8K)] = raw_reply(200, page)
    routes[archive_url(NVDA_CIK, NVDA_8K, "q2fy27pr.txt")] = raw_reply(
        200, b"NVIDIA Announces Results\n\nRevenue was $96.2 billion.\nUp 18%."
    )
    sync_nvda(session, RoutedFetch(routes), tmp_path, nvda)
    text = (tmp_path / "text" / NVDA_8K / "q2fy27pr.txt.txt").read_text(encoding="utf-8")
    assert text == "NVIDIA Announces Results\n\nRevenue was $96.2 billion.\nUp 18%."


def test_periodic_gaps_name_a_missing_quarter(session: Session, tmp_path: Path) -> None:
    nvda = add_entity(session, "NVDA", NVDA_CIK, "NVIDIA CORP")
    source = session.exec(select(Source)).first() or Source(name="sec_edgar")
    session.add(source)
    session.flush()
    for i, period in enumerate([date(2025, 4, 27), date(2025, 7, 27), date(2026, 1, 25)]):
        session.add(
            Document(
                source_id=source.id,  # pyright: ignore[reportArgumentType]
                entity_id=nvda.id,
                doc_type="10-Q" if i < 2 else "10-K",
                external_id=f"acc-{i}/f.htm",
                content_hash="x",
                source_time=datetime.combine(period, datetime.min.time(), tzinfo=UTC),
                available_at=NOW,
                received_at=NOW,
            )
        )
    session.commit()
    # Q3 FY26 (period 2025-10-26) is missing: 182 days between July and January.
    assert periodic_gaps(session) == [("NVDA", date(2025, 7, 27), date(2026, 1, 25))]


def test_select_skips_exhibits_with_no_text_format() -> None:
    # Measured: Cameco 6-K 0001193125-25-072172 files its 2024 annual report only as a PDF
    # (EX-99.1 d859263dex991.pdf); read as HTML it became 7.5 MB of "%PDF-1.7 ..." bytes.
    ref = replace(
        nvda_refs()["0001045810-26-000073"],
        accession="0001193125-25-072172",
        form="6-K",
        primary_document="d859263d6k.htm",
    )
    entries = [
        IndexEntry(1, "6-K", "d859263d6k.htm", "6-K"),
        IndexEntry(2, "CAMECO 2024 ANNUAL REPORT", "d859263dex991.pdf", "EX-99.1"),
        IndexEntry(None, "Complete submission text file", "0001193125-25-072172.txt", ""),
    ]
    assert _filenames(select_files(ref, entries)) == ["d859263d6k.htm"]
    with_html = [*entries, IndexEntry(3, "", "d859263dex992.htm", "EX-99.2")]
    assert _filenames(select_files(ref, with_html)) == ["d859263dex992.htm"]
    annual = replace(ref, form="10-K", primary_document="d859263d6k.htm")
    pdf_ex13 = [*entries, IndexEntry(4, "", "ex13.pdf", "EX-13")]
    assert _filenames(select_files(annual, pdf_ex13)) == ["d859263d6k.htm"]


def _add_reports(session: Session, ticker: str, cik: int, reports: list[tuple[str, date]]) -> None:
    company = add_entity(session, ticker, cik, ticker)
    source = session.exec(select(Source)).first() or Source(name="sec_edgar")
    session.add(source)
    session.flush()
    for i, (form, period) in enumerate(reports):
        session.add(
            Document(
                source_id=source.id,  # pyright: ignore[reportArgumentType]
                entity_id=company.id,
                doc_type=form,
                external_id=f"{ticker}-{cik}-{i}/f.htm",
                content_hash="x",
                source_time=datetime.combine(period, datetime.min.time(), tzinfo=UTC),
                available_at=NOW,
                received_at=NOW,
            )
        )
    session.commit()


def test_periodic_gaps_allow_a_16_week_fourth_quarter(session: Session) -> None:
    # Measured: Costco's 52/53-week year ends with a 16-week quarter, 10-Q 2023-05-07 then
    # 10-K 2023-09-03 (119 days); a missing quarter is 180 days or more.
    _add_reports(
        session,
        "COST",
        909832,
        [("10-Q", date(2023, 2, 12)), ("10-Q", date(2023, 5, 7)), ("10-K", date(2023, 9, 3))],
    )
    assert periodic_gaps(session) == []


def test_periodic_gaps_judge_an_annual_filer_year_by_the_annual_limit(session: Session) -> None:
    # Measured: Celestica filed 20-Fs, then from FY2024 a 10-K and 10-Qs.
    _add_reports(
        session,
        "CLS",
        1030894,
        [
            ("20-F", date(2022, 12, 31)),
            ("20-F", date(2023, 12, 31)),
            ("10-K", date(2024, 12, 31)),
            ("10-Q", date(2025, 3, 31)),
            ("10-Q", date(2025, 9, 30)),
        ],
    )
    # The 20-F years are fine; the skipped June 2025 quarter is not.
    assert periodic_gaps(session) == [("CLS", date(2025, 3, 31), date(2025, 9, 30))]


def test_transition_report_is_a_periodic_report() -> None:
    # Measured: Red Cat moved its year end from April 30 to December 31 and filed 10-KT
    # 0001641172-25-001892 for the eight months to 2024-12-31.
    ref = replace(
        nvda_refs()["0001045810-26-000021"],
        cik=748268,
        accession="0001641172-25-001892",
        form="10-KT",
        items=(),
        primary_document="rcat-10kt.htm",
    )
    assert wanted(ref)
    assert wanted(replace(ref, form="10-QT/A"))
    entries = [
        IndexEntry(1, "10-KT", "rcat-10kt.htm", "10-KT"),
        IndexEntry(2, "ANNUAL REPORT", "ex13.htm", "EX-13"),
        IndexEntry(3, "", "ex31.htm", "EX-31.1"),
    ]
    assert _filenames(select_files(ref, entries)) == ["rcat-10kt.htm", "ex13.htm"]


def test_periodic_gaps_count_a_transition_period(session: Session) -> None:
    _add_reports(
        session,
        "RCAT",
        748268,
        [
            ("10-Q", date(2024, 10, 31)),
            ("10-KT", date(2024, 12, 31)),
            ("10-Q", date(2025, 3, 31)),
        ],
    )
    assert periodic_gaps(session) == []


def test_periodic_gaps_ignore_amendments_of_earlier_periods(session: Session) -> None:
    # Measured: Albemarle filed 10-K/A 0000915913-23-000025 for FY2021 on 2023-01-26, inside the
    # window; its period is not a report the company skipped after.
    _add_reports(
        session,
        "ALB",
        915913,
        [
            ("10-K/A", date(2021, 12, 31)),
            ("10-K", date(2022, 12, 31)),
            ("10-Q", date(2023, 3, 31)),
        ],
    )
    assert periodic_gaps(session) == []


def test_periodic_gaps_treat_a_10k_filer_without_10qs_as_quarterly(session: Session) -> None:
    # None in the store today: a 10-K filer whose 10-Qs all went missing would otherwise be
    # judged by the annual limit and pass.
    _add_reports(
        session, "NVDA", NVDA_CIK, [("10-K", date(2024, 1, 28)), ("10-K", date(2025, 1, 26))]
    )
    assert periodic_gaps(session) == [("NVDA", date(2024, 1, 28), date(2025, 1, 26))]


def test_periodic_gaps_after_a_20f_use_the_annual_limit_only_up_to_an_annual_report(
    session: Session,
) -> None:
    # IREN's real sequence goes 20-F 2024-06-30 -> 10-K 2025-06-30; had the first 10-Q
    # (2024-09-30) been missing, 20-F -> 10-Q 2024-12-31 is a skipped quarter.
    _add_reports(
        session, "IREN", 1878848, [("20-F", date(2024, 6, 30)), ("10-Q", date(2024, 12, 31))]
    )
    assert periodic_gaps(session) == [("IREN", date(2024, 6, 30), date(2024, 12, 31))]


def test_stale_reports_name_companies_whose_latest_report_is_overdue(session: Session) -> None:
    # A CIK change the master does not know yet (XOM moved to 2115436 on 2026-07-16) drops the
    # latest reports while every earlier pair still looks continuous.
    _add_reports(
        session, "XOM", 2115436, [("10-Q", date(2025, 12, 31)), ("10-Q", date(2026, 3, 31))]
    )
    _add_reports(session, "TSM", 1046179, [("20-F", date(2025, 12, 31))])
    # The longest wait for a report on time: a Q3 10-Q, then a 10-K 92 days later plus a small
    # filer's 90-day deadline and a 15-day NT 10-K extension, 197 days; 200 is the limit.
    assert stale_reports(session, date(2026, 10, 17)) == []
    assert stale_reports(session, date(2026, 10, 22)) == [("XOM", date(2026, 3, 31))]
    # A 20-F is due four months after the year end, plus a 15-day NT 20-F extension.
    assert stale_reports(session, date(2027, 5, 15)) == [("XOM", date(2026, 3, 31))]
    assert ("TSM", date(2025, 12, 31)) in stale_reports(session, date(2027, 5, 16))


def test_stale_report_keys_are_valid_exception_keys(tmp_path: Path) -> None:
    path = tmp_path / "docs_exceptions.yaml"
    path.write_text(
        "- key: 'XYZ:periodic-stale:2026-03-31'\n  reason: 'delisted'\n"
        "  added: 2026-10-22\n  by: greyson\n"
    )
    assert load_coverage_exceptions(path) == {"XYZ:periodic-stale:2026-03-31": "delisted"}


def test_sync_saves_the_listing_it_read(session: Session, tmp_path: Path) -> None:
    nvda = add_entity(session, "NVDA", NVDA_CIK, "NVIDIA CORP")
    sync_nvda(session, RoutedFetch(nvda_routes()), tmp_path, nvda)
    saved = tmp_path / "raw" / str(NVDA_CIK) / "submissions" / "CIK0001045810.json"
    assert saved.read_bytes() == fixture_bytes("submissions-CIK0001045810.json")


def test_8k_title_names_its_items(session: Session, tmp_path: Path) -> None:
    # Retrieval must tell an earnings release (2.02) from a Reg FD item (7.01).
    nvda = add_entity(session, "NVDA", NVDA_CIK, "NVIDIA CORP")
    sync_nvda(session, RoutedFetch(nvda_routes()), tmp_path, nvda)
    titles = {
        d.title for d in session.exec(select(Document)).all() if d.external_id.startswith(NVDA_8K)
    }
    assert titles and all(t is not None and t.startswith("8-K 2.02,9.01 EX-99") for t in titles)
    ten_q = session.exec(
        select(Document).where(col(Document.external_id).startswith(NVDA_10Q))
    ).one()
    assert ten_q.title is not None and ten_q.title.startswith("10-Q 10-Q")


def test_missing_filings_compare_the_saved_listing_with_the_store(
    session: Session, tmp_path: Path
) -> None:
    nvda = add_entity(session, "NVDA", NVDA_CIK, "NVIDIA CORP")
    raw = tmp_path / "raw"
    assert missing_filings(session, raw, date(2026, 8, 1)) == [
        f"NVDA: no saved listing for CIK {NVDA_CIK}"
    ]
    routes = nvda_routes()
    routes[index_url(NVDA_CIK, NVDA_10Q)] = raw_reply(404, b"")
    result = sync_nvda(session, RoutedFetch(routes), tmp_path, nvda)
    assert len(result.errors) == 1
    # The 8-K 8.01 of 2026-09-03 is listed too but not wanted; the 10-Q is wanted and absent.
    assert missing_filings(session, raw, date(2026, 8, 1)) == [
        f"NVDA {NVDA_10Q} 10-Q 2026-08-26: listed but not stored"
    ]
    sync_nvda(session, RoutedFetch(nvda_routes()), tmp_path, nvda)
    assert missing_filings(session, raw, date(2026, 8, 1)) == []


def test_missing_filings_name_a_selected_file_left_out(session: Session, tmp_path: Path) -> None:
    # The 8-K's second exhibit fails to download: the accession is stored through its first.
    nvda = add_entity(session, "NVDA", NVDA_CIK, "NVIDIA CORP")
    routes = nvda_routes()
    routes[archive_url(NVDA_CIK, NVDA_8K, "q2fy27cfocommentary.htm")] = raw_reply(404, b"")
    sync_nvda(session, RoutedFetch(routes), tmp_path, nvda)
    assert missing_filings(session, tmp_path / "raw", date(2026, 8, 1)) == [
        f"NVDA {NVDA_8K}/q2fy27cfocommentary.htm: selected but not stored"
    ]


def test_missing_filings_accept_a_file_whose_text_is_empty(
    session: Session, tmp_path: Path
) -> None:
    # An image-only exhibit has no text and is never stored; that is not a missing filing.
    nvda = add_entity(session, "NVDA", NVDA_CIK, "NVIDIA CORP")
    routes = nvda_routes()
    blank = b'<html><body><img src="slide1.jpg"></body></html>'
    routes[archive_url(NVDA_CIK, NVDA_8K, "q2fy27cfocommentary.htm")] = raw_reply(200, blank)
    result = sync_nvda(session, RoutedFetch(routes), tmp_path, nvda)
    assert result.empty == 1
    assert missing_filings(session, tmp_path / "raw", date(2026, 8, 1)) == []


def test_periodic_gaps_keep_two_companies_with_one_ticker_apart(session: Session) -> None:
    # Company tickers are not unique (only the CIK is): a reused ticker or a CIK move without a
    # rebuild leaves two rows; one company's reports must not fill the other's gap.
    _add_reports(session, "XOM", 34088, [("10-Q", date(2025, 3, 31)), ("10-Q", date(2025, 9, 30))])
    _add_reports(session, "XOM", 2115436, [("10-Q", date(2025, 6, 30))])
    assert periodic_gaps(session) == [("XOM", date(2025, 3, 31), date(2025, 9, 30))]


def test_a_refused_listing_does_not_replace_the_saved_one(tmp_path: Path) -> None:
    good = fixture_bytes("submissions-CIK0001045810.json")
    cache = tmp_path / "submissions"
    list_filings(ScriptedFetch([raw_reply(200, good)]), NVDA_CIK, date(2026, 1, 1), cache)
    refusal = b"<html><body>Request Rate Threshold Exceeded</body></html>"
    with pytest.raises(RuntimeError):
        list_filings(ScriptedFetch([raw_reply(200, refusal)]), NVDA_CIK, date(2026, 1, 1), cache)
    assert (cache / "CIK0001045810.json").read_bytes() == good
