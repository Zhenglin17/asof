"""SEC EDGAR filings: which ones a company has, when each became public, and storing their text.

Three facts decide what a replay may read:
- When a filing became public. The acceptance time is read from the filing's index page
  ("Accepted", New York time, identical to ACCEPTANCE-DATETIME in the EDGAR header). The
  ``acceptanceDateTime`` of the submissions JSON is not used: measured 2026-10-21, it is the New
  York clock shifted to UTC twice for most filers (NVDA 0001045810-26-000073: JSON
  2026-08-27T00:21:19Z, accepted 2026-08-26 16:21:19 New York, i.e. 4 hours late; 5 in winter)
  but correct for others (Exxon 34088). A late filing is dated the next business day and,
  per SEC's filer manual (assumed to still hold), not disseminated before then: NVDA's 10-Q
  accepted Friday 2023-08-25 19:36 has filingDate 2023-08-28. So ``available_at`` is also never
  before 06:00 New York on ``filingDate``; SEC's filingDate carries its own business-day
  calendar and cutoff, so neither is hard-coded here.
- Which company a filing belongs to. A company can move to a new CIK (XOM to its holding company
  2115436 on 2026-07-16); filings of an earlier CIK count only while the security traded under it.
- Which files of a filing carry the content (``select_files``).
"""

import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from datetime import time as clock_time
from html.parser import HTMLParser
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pyarrow as pa
import yaml
from sqlmodel import Session, col, select

from asof.ingest.filing_text import chunk_text, html_to_text, plain_to_text
from asof.ingest.http import Fetch, FetchResult
from asof.store.documents import (
    DocumentSpec,
    add_document,
    find_document,
    get_or_create_source,
    text_path,
)
from asof.store.models import Document, Entity, EntityKind, Source

NEW_YORK = ZoneInfo("America/New_York")
SOURCE_NAME = "sec_edgar"
SEC_BASE_URL = "https://www.sec.gov"
SUBMISSIONS_URL = "https://data.sec.gov/submissions"
_LISTING_DIR = "submissions"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"
# SEC's fair-access limit is 10 requests a second per client; stay under it.
REQUESTS_PER_SECOND = 8.0
# The earliest a filing dated D is disseminated: EDGAR opens at 06:00 New York.
DISSEMINATION_OPENS = clock_time(6, 0)

_PERIODIC = frozenset({"10-K", "10-Q", "20-F"})
_EXHIBIT_FORMS = frozenset({"8-K", "6-K", "40-F"})


@dataclass(frozen=True)
class FilingRef:
    cik: int
    accession: str
    form: str
    items: tuple[str, ...]
    filing_date: date
    report_date: date | None
    primary_document: str


# A transition report covers the stub period of a changed fiscal year (Red Cat's 10-KT for the
# eight months to 2024-12-31) and is read like the form it stands in for.
_TRANSITION = {"10-KT": "10-K", "10-QT": "10-Q"}


def _base_form(form: str) -> str:
    form = form.removesuffix("/A")
    return _TRANSITION.get(form, form)


def parse_submission_rows(cik: int, rows: Mapping[str, list]) -> list[FilingRef]:
    """The columnar rows of a submissions JSON (``filings.recent``, or a history page)."""
    refs: list[FilingRef] = []
    for i, accession in enumerate(rows["accessionNumber"]):
        items = rows["items"][i] or ""
        report = rows["reportDate"][i] or ""
        refs.append(
            FilingRef(
                cik=cik,
                accession=accession,
                form=rows["form"][i],
                items=tuple(item.strip() for item in items.split(",") if item.strip()),
                filing_date=date.fromisoformat(rows["filingDate"][i]),
                report_date=date.fromisoformat(report) if report else None,
                primary_document=rows["primaryDocument"][i],
            )
        )
    return refs


class Throttled(RuntimeError):
    """SEC refused the client (403): over the rate limit or without a contact User-Agent."""


# Phrases of SEC's refusal page; a body containing one is never stored as a filing.
_REFUSAL_MARKERS = (b"Request Rate Threshold Exceeded", b"Undeclared Automated Tool")


def _get(fetch: Fetch, url: str) -> bytes:
    result = fetch(url, {})
    if result.status == 403:
        raise Throttled(f"GET {url} answered HTTP 403; wait ten minutes before running again")
    if result.status != 200:
        raise RuntimeError(f"GET {url} answered HTTP {result.status}")
    return result.body


def _listing_name(cik: int) -> str:
    return f"CIK{cik:010d}.json"


def list_filings(
    fetch: Fetch, cik: int, since: date, cache_dir: Path | None = None
) -> list[FilingRef]:
    """Every filing of the CIK dated ``since`` or later, following history pages as needed.

    With ``cache_dir`` every page read is saved there, replacing the previous copy, so an audit
    can later compare what EDGAR listed with what was stored.
    """

    def read(name: str) -> dict:
        body = _get(fetch, f"{SUBMISSIONS_URL}/{name}")
        # Checked before saving: a refusal page answered with 200 must not replace a good copy.
        if any(marker in body for marker in _REFUSAL_MARKERS):
            raise RuntimeError(f"GET {name} answered a refusal page")
        try:
            page = json.loads(body)
        except ValueError as error:
            raise RuntimeError(f"GET {name} is not JSON: {error}") from error
        if not isinstance(page, dict) or ("filings" not in page and "accessionNumber" not in page):
            raise RuntimeError(f"GET {name} is not a submissions listing")
        if cache_dir is not None:
            _write_once(cache_dir / name, body)
        return page

    listing = read(_listing_name(cik))
    pages = [listing["filings"]["recent"]]
    for page in listing["filings"].get("files", []):
        if date.fromisoformat(page["filingTo"]) >= since:
            pages.append(read(page["name"]))
    return [
        ref
        for rows in pages
        for ref in parse_submission_rows(cik, rows)
        if ref.filing_date >= since
    ]


_ACCEPTED = re.compile(
    r'Accepted</div>\s*<div class="info">\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*<'
)


def parse_accepted(index_html: str) -> datetime:
    """The "Accepted" line of a filing index page (New York time), as aware UTC."""
    found = _ACCEPTED.search(index_html)
    if found is None:
        raise ValueError("index page has no Accepted time")
    local = datetime.strptime(found.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=NEW_YORK)
    return local.astimezone(UTC)


def available_at(ref: FilingRef, accepted_at: datetime) -> datetime:
    """When the filing was public: acceptance, but never before its filing date opens."""
    opens = datetime.combine(ref.filing_date, DISSEMINATION_OPENS, tzinfo=NEW_YORK)
    return max(accepted_at, opens.astimezone(UTC))


def wanted(ref: FilingRef) -> bool:
    base = _base_form(ref.form)
    if base == "8-K":
        # 2.02 is the results item; 7.01 (Reg FD) carries guidance, investor-day decks and, at
        # UEC and Energy Fuels, the results themselves (UEC: 7.01 every quarter since 2023).
        return "2.02" in ref.items or "7.01" in ref.items
    return base in _PERIODIC or base in _EXHIBIT_FORMS


@dataclass(frozen=True)
class IndexEntry:
    seq: int | None
    description: str
    filename: str
    doc_type: str


class _IndexParser(HTMLParser):
    """Rows of the first ``Document Format Files`` table of a filing index page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._state = "before"  # before -> inside -> done
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._link: list[str] | None = None
        self._link_text: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._state == "before" and tag == "table":
            if dict(attrs).get("summary") == "Document Format Files":
                self._state = "inside"
        elif self._state != "inside":
            return
        elif tag == "tr":
            self._row = []
        elif tag == "td" and self._row is not None:
            self._cell, self._link_text = [], None
        elif tag == "a" and self._cell is not None:
            self._link = []

    def handle_endtag(self, tag: str) -> None:
        if self._state != "inside":
            return
        if tag == "a" and self._link is not None:
            self._link_text = "".join(self._link).strip()
            self._link = None
        elif tag == "td" and self._cell is not None and self._row is not None:
            # The Document cell's link text is the file name; the iXBRL marker after it is not.
            text = self._link_text if self._link_text is not None else "".join(self._cell)
            self._row.append(" ".join(text.split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None
        elif tag == "table":
            self._state = "done"

    def handle_data(self, data: str) -> None:
        if self._link is not None:
            self._link.append(data)
        if self._cell is not None:
            self._cell.append(data)


def parse_filing_index(html: str) -> list[IndexEntry]:
    parser = _IndexParser()
    parser.feed(html)
    entries: list[IndexEntry] = []
    for row in parser.rows:
        if len(row) < 4:
            continue
        seq, description, filename, doc_type = row[:4]
        entries.append(
            IndexEntry(
                seq=int(seq) if seq.isdigit() else None,
                description=description,
                filename=filename,
                doc_type=doc_type,
            )
        )
    return entries


_TEXT_SUFFIXES = (".htm", ".html", ".txt")


def select_files(ref: FilingRef, entries: Sequence[IndexEntry]) -> list[IndexEntry]:
    """The files whose text is the filing's content.

    A periodic report is its primary document (its exhibits are certifications and contracts).
    An earnings 8-K, a 6-K or a 40-F carries its content in EX-99 exhibits (the 8-K body only
    says "see Exhibit 99.1"; a 40-F wraps the annual information form and statements), falling
    back to the primary document when there is none.

    Only files that read as text count: BABA, BIDU, CCJ and NOK file some 6-K exhibits (annual
    reports, AGM materials) only as PDF, whose content their 20-F or 40-F already carries.
    """
    entries = [e for e in entries if e.filename.lower().endswith(_TEXT_SUFFIXES)]
    primary = [e for e in entries if e.filename == ref.primary_document]
    if _base_form(ref.form) in _PERIODIC:
        # A 10-K's EX-13 is the annual report to shareholders and may carry the body the
        # primary document only incorporates by reference; a 20-F's EX-13 is a certification.
        if _base_form(ref.form) == "10-K":
            return primary[:1] + [e for e in entries if e.doc_type.startswith("EX-13")]
        return primary[:1]
    exhibits = [e for e in entries if e.doc_type.startswith("EX-99")]
    return exhibits or primary[:1]


@dataclass(frozen=True)
class CikWindow:
    cik: int
    start: date | None
    end: date | None


def cik_windows(
    entity_cik: int, segments: Sequence[tuple[int | None, date, date | None]]
) -> list[CikWindow]:
    """Where each CIK's filings count for the company.

    The company's own CIK counts at any date. Another CIK seen on the security's master segments
    (a predecessor such as Exxon Mobil Corp 34088 before XOM moved to 2115436) counts only on the
    days a segment carried it; outside them it is someone else, e.g. Fortress on FIG's old segment.
    """
    windows = {CikWindow(entity_cik, None, None)}
    for cik, valid_from, valid_to in segments:
        if cik is not None and cik != entity_cik:
            windows.add(CikWindow(cik, valid_from, valid_to))
    return sorted(windows, key=lambda w: (w.cik, w.start or date.min))


Segment = tuple[str, int | None, date, date | None]
"""(symbol, cik, valid_from, valid_to) of one security-master segment."""


def security_segments(master: pa.Table, ticker: str) -> list[Segment]:
    """Every segment of the security now trading as ``ticker``, oldest first."""
    rows = master.select(["security_id", "symbol", "valid_from", "valid_to", "cik"]).to_pylist()
    current = {r["security_id"] for r in rows if r["symbol"] == ticker and r["valid_to"] is None}
    return sorted(
        (
            (r["symbol"], r["cik"], r["valid_from"], r["valid_to"])
            for r in rows
            if r["security_id"] in current
        ),
        key=lambda s: s[2],
    )


TradedSpan = Callable[[str, date, date | None], tuple[date, date] | None]
"""(symbol, start, end exclusive) -> first and last session with volume > 0, or None."""


def trading_segments(
    entity_cik: int, segments: Sequence[Segment], traded: TradedSpan
) -> list[tuple[int | None, date, date | None]]:
    """Segments for ``cik_windows``, another CIK's clipped to the days the security traded.

    A CIK on a segment of zero-volume placeholder bars is a stale SEC row, not a predecessor
    (Fortress 1380393 on FIG until Figma listed), so its filings must not reach the company.
    """
    out: list[tuple[int | None, date, date | None]] = []
    for symbol, cik, start, end in segments:
        if cik is None or cik == entity_cik:
            out.append((cik, start, end))
            continue
        span = traded(symbol, start, end)
        if span is not None:
            out.append((cik, span[0], span[1] + timedelta(days=1)))
    return out


def belongs(ref: FilingRef, windows: Sequence[CikWindow]) -> bool:
    """A window of the filer's CIK covers the filing date (SEC's own date, set at acceptance)."""
    day = ref.filing_date
    return any(
        w.cik == ref.cik and (w.start is None or w.start <= day) and (w.end is None or day < w.end)
        for w in windows
    )


def rate_limited(
    fetch: Fetch,
    per_second: float,
    *,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Fetch:
    """Starts consecutive requests at least 1/per_second apart."""
    gap = 1.0 / per_second
    last: list[float] = []

    def limited(url: str, params: Mapping[str, str]) -> FetchResult:
        if last:
            wait = last[0] + gap - clock()
            if wait > 0:
                sleep(wait)
        last[:] = [clock()]
        return fetch(url, params)

    return limited


_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


def retrying(
    fetch: Fetch,
    *,
    attempts: int = 4,
    backoff: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Fetch:
    """Retries throttling, server errors and dropped connections with growing waits.

    The last answer (or network error) is returned as is; a 404 is an answer, not a failure.
    """

    def patient(url: str, params: Mapping[str, str]) -> FetchResult:
        for attempt in range(attempts):
            last_try = attempt == attempts - 1
            try:
                result = fetch(url, params)
            except httpx.TransportError:
                if last_try:
                    raise
            else:
                if result.status not in _RETRY_STATUSES or last_try:
                    return result
            sleep(backoff * 2**attempt)
        raise AssertionError("unreachable")

    return patient


@dataclass(frozen=True)
class SyncResult:
    added: int
    skipped: int
    files: int
    empty: int = 0  # files with no text (slides of images), counted, not stored
    errors: tuple[str, ...] = ()


def _archive_url(ref: FilingRef, filename: str) -> str:
    return f"{ARCHIVES_URL}/{ref.cik}/{ref.accession.replace('-', '')}/{filename}"


def _decode(body: bytes) -> str:
    # Older filings are often Windows-1252 rather than UTF-8; neither is declared reliably.
    try:
        return body.decode("utf-8")
    except UnicodeDecodeError:
        return body.decode("cp1252", errors="replace")


def _write_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    partial.write_bytes(data)
    partial.replace(path)


def _cached(
    fetch: Fetch, url: str, path: Path, check: Callable[[bytes], object] | None = None
) -> tuple[bytes, bool]:
    """The file at path, downloading it first if absent; raw files are written once, never again.

    A download is stored only after ``check`` accepts it, and never when it is SEC's refusal page:
    a cached bad answer would be served as the filing on every later run.
    """
    if path.exists():
        return path.read_bytes(), False
    body = _get(fetch, url)
    if any(marker in body for marker in _REFUSAL_MARKERS):
        raise RuntimeError(f"GET {url} returned SEC's refusal page")
    if check is not None:
        check(body)
    _write_once(path, body)
    return body, True


def _to_text(filename: str, body: bytes) -> str:
    raw = _decode(body)
    return plain_to_text(raw) if filename.lower().endswith(".txt") else html_to_text(raw)


def sync_company(
    session: Session,
    fetch: Fetch,
    *,
    entity: Entity,
    windows: Sequence[CikWindow],
    since: date,
    raw_dir: Path,
    text_dir: Path,
    now: datetime,
) -> SyncResult:
    """Store every wanted filing of the company since ``since``. Flushes; the caller commits."""
    assert entity.id is not None
    source = get_or_create_source(session, SOURCE_NAME, SEC_BASE_URL)
    added = skipped = files = empty = 0
    errors: list[str] = []
    for cik in sorted({w.cik for w in windows}):
        for ref in list_filings(fetch, cik, since, raw_dir / str(cik) / _LISTING_DIR):
            if not (wanted(ref) and belongs(ref, windows)):
                continue
            folder = raw_dir / str(ref.cik) / ref.accession
            index_name = f"{ref.accession}-index.htm"
            try:
                index_bytes, _ = _cached(
                    fetch,
                    _archive_url(ref, index_name),
                    folder / index_name,
                    check=lambda body: parse_accepted(_decode(body)),
                )
                index_html = _decode(index_bytes)
                public_at = available_at(ref, parse_accepted(index_html))
            except Throttled:
                raise
            except (RuntimeError, ValueError) as error:
                errors.append(f"{ref.accession} {ref.form}: {error}")
                continue
            selected = select_files(ref, parse_filing_index(index_html))
            if not selected:
                errors.append(
                    f"{ref.accession} {ref.form}: no file selected (primary document "
                    f"{ref.primary_document} is not in the index)"
                )
            for entry in selected:
                external_id = f"{ref.accession}/{entry.filename}"
                existing = find_document(session, source, external_id)
                if existing is not None:
                    if existing.entity_id == entity.id:
                        skipped += 1
                    else:
                        holder = session.get(Entity, existing.entity_id)
                        errors.append(
                            f"{external_id}: already stored for "
                            f"{holder.ticker if holder else existing.entity_id}"
                        )
                    continue
                url = _archive_url(ref, entry.filename)
                try:
                    body, downloaded = _cached(fetch, url, folder / entry.filename)
                    files += downloaded
                    text = _to_text(entry.filename, body)
                except Throttled:
                    raise
                except (RuntimeError, ValueError) as error:
                    errors.append(f"{external_id}: {error}")
                    continue
                if not text:
                    empty += 1
                    continue
                _write_once(text_path(text_dir, external_id), text.encode("utf-8"))
                period = ref.report_date or ref.filing_date
                spec = DocumentSpec(
                    entity_id=entity.id,
                    doc_type=ref.form,
                    external_id=external_id,
                    title=_title(ref, entry),
                    url=url,
                    text=text,
                    source_time=datetime.combine(period, clock_time(0), tzinfo=UTC),
                    available_at=public_at,
                    received_at=now,
                )
                chunks = chunk_text(text, form=ref.form, exhibit=entry.doc_type or None)
                if add_document(session, source, spec, chunks) is not None:
                    added += 1
    return SyncResult(added=added, skipped=skipped, files=files, empty=empty, errors=tuple(errors))


def _title(ref: FilingRef, entry: IndexEntry) -> str:
    # An 8-K's items tell an earnings release (2.02) from a Reg FD item (7.01).
    items = ",".join(ref.items) if _base_form(ref.form) == "8-K" else ""
    return " ".join(part for part in (ref.form, items, entry.doc_type, entry.description) if part)


def _edgar_documents(session: Session) -> list[Document]:
    """Every stored EDGAR document regardless of as_of: for audits only, never a read path."""
    stmt = (
        select(Document)
        .join(Source, col(Source.id) == col(Document.source_id))
        .where(Source.name == SOURCE_NAME)
        .order_by(col(Document.id))
    )
    return list(session.exec(stmt).all())


def audit_filings(session: Session, raw_dir: Path) -> list[str]:
    """Every EDGAR document is public no earlier than its acceptance on the saved index page."""
    problems: list[str] = []
    accepted: dict[Path, datetime] = {}
    for document in _edgar_documents(session):
        accession = document.external_id.split("/", 1)[0]
        # The archive URL names the filer: .../data/<cik>/<accession without dashes>/<file>.
        cik = (document.url or "").split("/data/", 1)[-1].split("/", 1)[0]
        index = raw_dir / cik / accession / f"{accession}-index.htm"
        if index not in accepted:
            if not index.exists():
                problems.append(f"{document.external_id}: index page {index} is missing")
                continue
            accepted[index] = parse_accepted(_decode(index.read_bytes()))
        if document.available_at < accepted[index]:
            problems.append(
                f"{document.external_id}: available_at {document.available_at.isoformat()} is "
                f"before its acceptance {accepted[index].isoformat()}"
            )
    return problems


_EARNINGS_FORMS = frozenset({"8-K", "6-K"})
_PERIODIC_FORMS = frozenset({"10-K", "10-Q", "20-F", "40-F"})


def coverage_gaps(session: Session) -> list[tuple[str, str]]:
    """(ticker, "earnings" | "periodic") for each company lacking that kind of filing."""
    companies = session.exec(
        select(Entity).where(Entity.kind == EntityKind.COMPANY).order_by(col(Entity.ticker))
    ).all()
    forms: dict[int, set[str]] = {}
    for document in _edgar_documents(session):
        if document.entity_id is not None:
            forms.setdefault(document.entity_id, set()).add(_base_form(document.doc_type))
    gaps: list[tuple[str, str]] = []
    for company in companies:
        held = forms.get(company.id or -1, set())
        if not held & _EARNINGS_FORMS:
            gaps.append((company.ticker, "earnings"))
        if not held & _PERIODIC_FORMS:
            gaps.append((company.ticker, "periodic"))
    return gaps


def missing_filings(session: Session, raw_dir: Path, since: date) -> list[str]:
    """Wanted filings on a company's own saved EDGAR listing that the store does not hold.

    The listing is the one the last sync read (``<raw_dir>/<cik>/submissions/``). Each listed
    filing's selected files are recomputed from its saved index page and compared one by one,
    so an 8-K stored through one of its two exhibits is caught; a file is excused only when its
    saved copy reads as empty text (an image-only slide deck). A predecessor CIK's listing is
    not compared: which of its filings belong is decided by the security master's trading days,
    not by the listing alone.
    """
    companies = session.exec(
        select(Entity).where(Entity.kind == EntityKind.COMPANY).order_by(col(Entity.ticker))
    ).all()
    held: dict[int, set[str]] = {}
    for document in _edgar_documents(session):
        if document.entity_id is not None:
            held.setdefault(document.entity_id, set()).add(document.external_id)
    problems: list[str] = []
    for company in companies:
        if company.cik is None or company.id is None:
            continue
        folder = raw_dir / str(company.cik) / _LISTING_DIR
        main = folder / _listing_name(company.cik)
        if not main.exists():
            problems.append(f"{company.ticker}: no saved listing for CIK {company.cik}")
            continue
        listing = json.loads(main.read_bytes())
        pages = [listing["filings"]["recent"]]
        for page in listing["filings"].get("files", []):
            if date.fromisoformat(page["filingTo"]) < since:
                continue
            path = folder / page["name"]
            if not path.exists():
                problems.append(f"{company.ticker}: listing page {page['name']} is not saved")
                continue
            pages.append(json.loads(path.read_bytes()))
        stored = held.get(company.id, set())
        for rows in pages:
            for ref in parse_submission_rows(company.cik, rows):
                if ref.filing_date >= since and wanted(ref):
                    problems += _missing_files(company.ticker, ref, raw_dir, stored)
    return problems


def _missing_files(ticker: str, ref: FilingRef, raw_dir: Path, stored: set[str]) -> list[str]:
    folder = raw_dir / str(ref.cik) / ref.accession
    index = folder / f"{ref.accession}-index.htm"
    if not index.exists():
        return [f"{ticker} {ref.accession} {ref.form} {ref.filing_date}: listed but not stored"]
    problems: list[str] = []
    for entry in select_files(ref, parse_filing_index(_decode(index.read_bytes()))):
        external_id = f"{ref.accession}/{entry.filename}"
        if external_id in stored:
            continue
        saved = folder / entry.filename
        if saved.exists() and not _to_text(entry.filename, saved.read_bytes()):
            continue
        problems.append(f"{ticker} {external_id}: selected but not stored")
    return problems


_EXCEPTION_FIELDS = ("key", "reason", "added", "by")
_EXCEPTION_KEY = re.compile(
    r"[A-Z0-9.\-]+:(earnings|periodic|periodic-(gap|stale):\d{4}-\d{2}-\d{2})"
)


def load_coverage_exceptions(path: Path) -> dict[str, str]:
    """Reviewed coverage gaps, ``TICKER:earnings`` or ``TICKER:periodic`` to the reason."""
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text()) or []
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a list of exceptions")
    loaded: dict[str, str] = {}
    for item in raw:
        if not isinstance(item, dict) or set(item) != set(_EXCEPTION_FIELDS):
            raise ValueError(f"{path}: entry {item!r} must have exactly {_EXCEPTION_FIELDS}")
        if any(item[name] is None or str(item[name]).strip() == "" for name in _EXCEPTION_FIELDS):
            raise ValueError(f"{path}: entry {item!r} has an empty field")
        if not isinstance(item["added"], date):
            raise ValueError(f"{path}: entry {item!r}: added must be an ISO date")
        if not _EXCEPTION_KEY.fullmatch(str(item["key"])):
            raise ValueError(f"{path}: entry {item!r}: key must look like TICKER:earnings")
        loaded[str(item["key"])] = str(item["reason"])
    return loaded


# Longest normal distance between consecutive report periods: a quarter is 91 days and a 52/53-week
# fiscal calendar moves a period end by up to a week; an annual-only filer reports once a year.
# Costco's 52/53-week year ends with a 16- or 17-week quarter (119 days); a skipped quarter is
# 180 days or more.
_QUARTERLY_GAP_DAYS = 120
_ANNUAL_GAP_DAYS = 380
_ANNUAL_ONLY = frozenset({"20-F", "40-F"})
_ANNUAL_REPORTS = frozenset({"10-K", "20-F", "40-F"})
# The longest wait for a report filed on time: a Q3 10-Q, then a 10-K 92 days later plus a small
# filer's 90-day deadline and a 15-day NT 10-K extension (197 days); a 20-F is due four months
# after the year end, plus a 15-day NT 20-F extension.
_QUARTERLY_STALE_DAYS = 200
_ANNUAL_STALE_DAYS = 500


def _periodic_reports(session: Session) -> dict[tuple[str, int], dict[date, set[str]]]:
    """(ticker, entity id) -> period -> base forms of its original (not amended) reports.

    Keyed by the entity, not the ticker: company tickers are not unique (only the CIK is), and
    one company's reports must not fill another's gap.
    """
    reports: dict[tuple[str, int], dict[date, set[str]]] = {}
    for document in _edgar_documents(session):
        form = _base_form(document.doc_type)
        amended = document.doc_type.endswith("/A")
        if document.entity_id is None or form not in _PERIODIC_FORMS or amended:
            continue
        company = session.get(Entity, document.entity_id)
        if company is None:
            continue
        periods = reports.setdefault((company.ticker, document.entity_id), {})
        periods.setdefault(document.source_time.date(), set()).add(form)
    return reports


def _quarterly(periods: dict[date, set[str]]) -> bool:
    # A 10-K filer owes 10-Qs too: judging one whose 10-Qs all went missing by the annual limit
    # would pass it.
    return any(forms & {"10-K", "10-Q"} for forms in periods.values())


def periodic_gaps(session: Session) -> list[tuple[str, date, date]]:
    """(ticker, period, next period) wherever a company's periodic reports skip a period.

    Quarterly filers (any 10-K or 10-Q) are expected every quarter, annual-only filers (20-F /
    40-F) every year. From a 20-F or 40-F period to the next annual report a year's wait is
    fine even at a company that later files 10-Qs (Celestica, Shopify, IREN moved over in
    2024-2025); to a 10-Q it is not. Amendments repeat a period and are ignored.
    """
    gaps: list[tuple[str, date, date]] = []
    for (ticker, _), periods in _periodic_reports(session).items():
        quarterly = _quarterly(periods)
        ordered = sorted(periods)
        for earlier, later in zip(ordered, ordered[1:], strict=False):
            annual = not quarterly or bool(
                periods[earlier] & _ANNUAL_ONLY and periods[later] & _ANNUAL_REPORTS
            )
            limit = _ANNUAL_GAP_DAYS if annual else _QUARTERLY_GAP_DAYS
            if (later - earlier).days > limit:
                gaps.append((ticker, earlier, later))
    return sorted(gaps)


def stale_reports(session: Session, today: date) -> list[tuple[str, date]]:
    """(ticker, latest period) wherever the next periodic report is overdue by ``today``.

    Gaps between stored reports cannot see reports missing at the end, as when a company's new
    CIK is not known yet (XOM moved to 2115436 on 2026-07-16).
    """
    stale: list[tuple[str, date]] = []
    for (ticker, _), periods in _periodic_reports(session).items():
        latest = max(periods)
        annual = not _quarterly(periods) or bool(periods[latest] & _ANNUAL_ONLY)
        limit = _ANNUAL_STALE_DAYS if annual else _QUARTERLY_STALE_DAYS
        if (today - latest).days > limit:
            stale.append((ticker, latest))
    return sorted(stale)
