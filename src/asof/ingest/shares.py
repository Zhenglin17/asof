"""Shares outstanding from SEC XBRL, for a rough market cap: what each company reported and when
that report became public.

One companyconcept reply per company and concept holds every fact the company tagged, with the
filing that carried it. Three concepts, in the order a reader prefers them:
- ``dei:EntityCommonStockSharesOutstanding``, the cover count of a 10-K / 10-Q / 20-F. A company
  reporting it per share class (GOOGL) or not tagging it (SHOP) answers 404.
- ``us-gaap:WeightedAverageNumberOfSharesOutstandingBasic``, all classes together, averaged over
  the period: close enough for a cap threshold.
- ``ifrs-full:WeightedAverageShares``, the same for IFRS filers (NVO, AZN, ERIC, TECK).
- ``us-gaap:CommonStockSharesOutstanding``, the balance-sheet count: GOOGL tags it while its
  average starts later; for HSY it leaves out class B (about a quarter), fine for a threshold.
Measured 2026-10-29 over the 2,984 CIKs of the 2020-01..2025-06 liquid tier: 2,621 have the cover
count, 271 more the us-gaap average, and 92 neither (before the IFRS concept was added).

The API gives the filing date only. A filing accepted after 17:30 New York already carries the
next business day as its date (see ``asof.ingest.edgar``), so a fact counts as public from 00:00
New York of the day after ``filed``: never early, at most a day late.
"""

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pyarrow as pa
import pyarrow.parquet as pq

from asof.ingest.http import Fetch

NEW_YORK = ZoneInfo("America/New_York")
CONCEPT_URL = "https://data.sec.gov/api/xbrl/companyconcept"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts"
CONCEPTS = (
    ("dei", "EntityCommonStockSharesOutstanding"),
    ("us-gaap", "WeightedAverageNumberOfSharesOutstandingBasic"),
    ("ifrs-full", "WeightedAverageShares"),
    ("us-gaap", "CommonStockSharesOutstanding"),
)
SHARES_FILE = "fundamentals/shares_outstanding.parquet"
SHARES_SCHEMA = pa.schema(
    [
        ("cik", pa.int64()),
        ("concept", pa.string()),
        ("accn", pa.string()),
        ("form", pa.string()),
        ("end", pa.date32()),
        ("filed", pa.date32()),
        ("shares", pa.float64()),
        ("available_at", pa.timestamp("us", tz="UTC")),
        ("start", pa.date32()),
    ]
)
_REQUIRED = ("end", "filed", "val", "accn")
_MISSING = "null"  # cached body of a 404: the concept is not reported


@dataclass(frozen=True)
class ShareFact:
    cik: int
    concept: str  # "taxonomy:tag"
    accn: str
    form: str  # as filed, amendments included ("10-K/A")
    end: date  # the date the count is as of (a duration concept: its period's end)
    filed: date
    shares: float
    available_at: datetime
    # a duration concept's period start; one filing tags a quarter's and a half-year's average
    # with the same end (108,337 such pairs, 931 apart by more than 1.5x, measured 2026-10-29)
    start: date | None = None


def _public_from(filed: date) -> datetime:
    return datetime.combine(filed + timedelta(days=1), time(0), tzinfo=NEW_YORK).astimezone(UTC)


def parse_concept(cik: int, taxonomy: str, tag: str, payload: Mapping[str, Any]) -> list[ShareFact]:
    """The ``shares`` facts of one reply; a fact without a date, value or filing, or with a
    count of zero or less, is dropped, and a fact listed twice is kept once."""
    concept = f"{taxonomy}:{tag}"
    out: list[ShareFact] = []
    seen: set[tuple[str, str, str | None, float]] = set()
    for raw in (payload.get("units") or {}).get("shares") or []:
        if any(raw.get(k) is None for k in _REQUIRED) or raw["val"] <= 0:
            continue
        key = (raw["accn"], raw["end"], raw.get("start"), float(raw["val"]))
        if key in seen:
            continue
        seen.add(key)
        filed = date.fromisoformat(raw["filed"])
        out.append(
            ShareFact(
                cik=cik,
                concept=concept,
                accn=raw["accn"],
                form=raw.get("form") or "",
                end=date.fromisoformat(raw["end"]),
                filed=filed,
                shares=float(raw["val"]),
                available_at=_public_from(filed),
                start=date.fromisoformat(raw["start"]) if raw.get("start") else None,
            )
        )
    return out


def _concept_url(cik: int, taxonomy: str, tag: str) -> str:
    return f"{CONCEPT_URL}/CIK{cik:010d}/{taxonomy}/{tag}.json"


def fetch_share_facts(ciks: Iterable[int], fetch: Fetch, cache_dir: Path) -> list[ShareFact]:
    """Every concept of every CIK, each read from ``cache_dir`` when a reply (a 404 included) is
    cached there, else asked once; an empty reply is read from the company's companyfacts. A
    failure other than 404 raises and is not cached: retries and pacing belong to ``fetch``
    (``retrying(rate_limited(...))``, as for EDGAR)."""
    out: list[ShareFact] = []
    for cik in ciks:
        for taxonomy, tag in CONCEPTS:
            path = cache_dir / f"{cik:010d}" / f"{taxonomy}-{tag}.json"
            payload = _cached_get(_concept_url(cik, taxonomy, tag), path, fetch)
            if payload is not None and not isinstance(
                (payload.get("units") or {}).get("shares", []), list
            ):
                payload = _from_company_facts(cik, taxonomy, tag, fetch, cache_dir)
            if payload is not None:
                out.extend(parse_concept(cik, taxonomy, tag, payload))
    return out


def _cached_get(url: str, path: Path, fetch: Fetch) -> Any:
    """The JSON at ``url`` (None for a 404), cached at ``path`` once it parses: a cut-off
    reply is never cached."""
    if path.exists():
        return json.loads(path.read_text())
    result = fetch(url, {})
    if result.status == 404:
        text = _MISSING
    elif result.status == 200:
        text = result.body.decode()
    else:
        raise RuntimeError(f"{url}: HTTP {result.status}")
    payload = json.loads(text)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text)
    tmp.replace(path)
    return payload


def _from_company_facts(
    cik: int, taxonomy: str, tag: str, fetch: Fetch, cache_dir: Path
) -> dict[str, Any] | None:
    """One concept out of the company's companyfacts reply, for a companyconcept reply whose
    ``shares`` came back as an empty object (ABT dei and us-gaap, measured 2026-10-29, while
    companyfacts held 68 and 221 facts)."""
    url = f"{FACTS_URL}/CIK{cik:010d}.json"
    full = _cached_get(url, cache_dir / f"{cik:010d}" / "companyfacts.json", fetch)
    if full is None:
        return None
    return (full.get("facts") or {}).get(taxonomy, {}).get(tag)


def write_share_facts(facts: Sequence[ShareFact], root: Path) -> Path:
    """Replace the shares file under the market ``root`` with ``facts``."""
    path = root / SHARES_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([asdict(f) for f in facts], schema=SHARES_SCHEMA)
    tmp = path.with_suffix(".tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)
    return path
