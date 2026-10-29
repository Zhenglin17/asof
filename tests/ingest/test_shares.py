"""Guards against share counts read before they were public and against a fetch that turns a
missing concept into an error or a server error into "no facts": a fact stamped available on its
filing date (or earlier), facts of other units, broken or zero facts stored as counts, the same
fact stored once per copy, and re-downloads of what the cache already holds.

API (asof.ingest.shares):

    CONCEPTS = (("dei", "EntityCommonStockSharesOutstanding"),
                ("us-gaap", "WeightedAverageNumberOfSharesOutstandingBasic"),
                ("ifrs-full", "WeightedAverageShares"),
                ("us-gaap", "CommonStockSharesOutstanding"))
    ShareFact(cik, concept "taxonomy:tag", accn, form, end, filed, shares, available_at, start)
    parse_concept(cik, taxonomy, tag, payload) -> list[ShareFact]
    fetch_share_facts(ciks, fetch, cache_dir) -> list[ShareFact]
    SHARES_FILE = "fundamentals/shares_outstanding.parquet"
    write_share_facts(facts, root) -> Path

Fake data provenance:
- AAPL dei reply (``AAPL_DEI``): the real companyconcept reply shape, fetched 2026-10-29
  (measured; the first fact is real, the others are constructed in the same shape).
- AZN ifrs-full duration fact with ``start`` (``AZN_IFRS``): real fact, fetched 2026-10-29
  (measured).
- GOOGL dei and SHOP dei answer HTTP 404 (measured 2026-10-29): a concept a company does not
  report is a 404, not an empty reply.
- A fact repeated in a later filing (same end and value, another accn) is real (measured
  2026-10-29: comparatives); a fact listed twice with the same accn is assumed.
- ``filed`` is a date only (measured). Assumed, from SEC's filer manual as cited in
  asof.ingest.edgar: a filing accepted after 17:30 New York already carries the next business
  day as its filing date, so 00:00 New York of filed + 1 day is never before dissemination.

Assumptions that only real data can confirm, with the check that does it:
- filed + 1 day 00:00 New York is never before the filing was public. Check: for every stored
  fact whose accn is also a stored EDGAR document, available_at >= that document's
  available_at (the index-page acceptance). Run once 2026-10-29: 2,981 filings matched, none
  public before its document; not yet an audit check.
- "Shares" is the only unit carrying a share count. Check: the real-data standard cases in
  tests/store/test_golden_shares.py (AAPL, NVDA, TSM, TECK each have a row).
- Exact duplicates within one reply exist or are harmless. Check: audit F2 (a duplicate kept
  twice would not change a ratio, so this one is benign either way).
"""

import json
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, fields
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pyarrow.parquet as pq
import pytest

from asof.ingest.http import FetchResult
from asof.ingest.shares import (
    CONCEPTS,
    SHARES_FILE,
    ShareFact,
    fetch_share_facts,
    parse_concept,
    write_share_facts,
)
from asof.store.market import market_root
from tests.ingest.fakes import reply

NY = ZoneInfo("America/New_York")
AAPL_CIK = 320193
AZN_CIK = 901832
DEI = ("dei", "EntityCommonStockSharesOutstanding")
GAAP = ("us-gaap", "WeightedAverageNumberOfSharesOutstandingBasic")
IFRS = ("ifrs-full", "WeightedAverageShares")
BAL = ("us-gaap", "CommonStockSharesOutstanding")

# The real reply for AAPL dei (first fact real, 2026-10-29); a "USD" unit is added to prove that
# only "shares" is read.
AAPL_DEI: dict[str, Any] = {
    "cik": AAPL_CIK,
    "taxonomy": "dei",
    "tag": "EntityCommonStockSharesOutstanding",
    "entityName": "Apple Inc.",
    "units": {
        "shares": [
            {
                "end": "2026-07-17",
                "val": 14594180000,
                "accn": "0000320193-26-000020",
                "fy": 2026,
                "fp": "Q3",
                "form": "10-Q",
                "filed": "2026-07-31",
                "frame": "CY2026Q2I",
            },
            {
                "end": "2026-01-16",
                "val": 14700000000,
                "accn": "0000320193-26-000006",
                "fy": 2026,
                "fp": "Q1",
                "form": "10-Q",
                "filed": "2026-01-30",
            },
        ],
        "USD": [
            {
                "end": "2026-07-17",
                "val": 999,
                "accn": "0000320193-26-000020",
                "form": "10-Q",
                "filed": "2026-07-31",
            }
        ],
    },
}

# The real AZN ifrs-full fact (2026-10-29): a duration concept carries "start" as well.
AZN_IFRS: dict[str, Any] = {
    "cik": AZN_CIK,
    "taxonomy": "ifrs-full",
    "tag": "WeightedAverageShares",
    "entityName": "AstraZeneca PLC",
    "units": {
        "shares": [
            {
                "start": "2025-01-01",
                "end": "2025-12-31",
                "val": 1550000000,
                "accn": "0001104659-26-019130",
                "fy": 2025,
                "fp": "FY",
                "form": "20-F",
                "filed": "2026-02-24",
                "frame": "CY2025",
            }
        ]
    },
}


def url(cik: int, concept: tuple[str, str]) -> str:
    taxonomy, tag = concept
    return f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/{taxonomy}/{tag}.json"


def payload(cik: int, concept: tuple[str, str], facts: list[dict[str, Any]]) -> dict[str, Any]:
    taxonomy, tag = concept
    return {
        "cik": cik,
        "taxonomy": taxonomy,
        "tag": tag,
        "entityName": f"Company {cik}",
        "units": {"shares": facts},
    }


def raw_fact(
    end: str = "2024-04-19",
    val: float = 1_000_000_000,
    accn: str = "0000001001-24-000001",
    filed: str = "2024-05-03",
    form: str = "10-Q",
    **extra: Any,
) -> dict[str, Any]:
    return {"end": end, "val": val, "accn": accn, "form": form, "filed": filed, **extra}


class RoutedFetch:
    """Answers by URL (the order of the requests is the module's business) and records every
    URL asked for. A URL without a route is a test failure, never a silent 404."""

    def __init__(self, routes: Mapping[str, FetchResult]) -> None:
        self.routes = dict(routes)
        self.urls: list[str] = []

    def __call__(self, url: str, params: Mapping[str, str]) -> FetchResult:
        self.urls.append(url)
        if url not in self.routes:
            raise AssertionError(f"unexpected request: {url}")
        return self.routes[url]


def no_network(url: str, params: Mapping[str, str]) -> FetchResult:
    raise AssertionError(f"a cached concept was fetched again: {url}")


def missing(cik: int, *concepts: tuple[str, str]) -> dict[str, FetchResult]:
    return {url(cik, c): reply(404, {"message": "Not Found"}) for c in concepts}


# =============================================================================================
# constants and the record
# =============================================================================================


def test_concepts_are_the_four_in_priority_order() -> None:
    assert CONCEPTS == (DEI, GAAP, IFRS, BAL)


def test_shares_file_lives_under_fundamentals() -> None:
    assert SHARES_FILE == "fundamentals/shares_outstanding.parquet"


def test_share_fact_is_frozen() -> None:
    [fact] = parse_concept(AZN_CIK, *IFRS, AZN_IFRS)
    with pytest.raises(FrozenInstanceError):
        fact.shares = 1.0  # type: ignore[misc]


# =============================================================================================
# parse_concept: normal
# =============================================================================================


def test_parse_reads_the_real_aapl_reply_shape() -> None:
    facts = parse_concept(AAPL_CIK, *DEI, AAPL_DEI)

    by_accn = {f.accn: f for f in facts}
    assert set(by_accn) == {"0000320193-26-000020", "0000320193-26-000006"}  # USD unit ignored
    fact = by_accn["0000320193-26-000020"]
    assert fact == ShareFact(
        cik=AAPL_CIK,
        concept="dei:EntityCommonStockSharesOutstanding",
        accn="0000320193-26-000020",
        form="10-Q",
        end=date(2026, 7, 17),
        filed=date(2026, 7, 31),
        shares=14594180000.0,
        available_at=datetime(2026, 8, 1, tzinfo=NY),
    )
    assert isinstance(fact.shares, float)


def test_parse_reads_a_duration_fact_by_its_end() -> None:
    [fact] = parse_concept(AZN_CIK, *IFRS, AZN_IFRS)

    assert fact.concept == "ifrs-full:WeightedAverageShares"
    assert fact.end == date(2025, 12, 31)
    assert fact.filed == date(2026, 2, 24)
    assert fact.form == "20-F"
    assert fact.shares == 1_550_000_000.0


def test_available_at_is_midnight_new_york_after_the_filing_date_in_both_seasons() -> None:
    summer = raw_fact(filed="2024-07-31", accn="a-1")
    winter = raw_fact(filed="2024-01-31", accn="a-2")
    facts = {f.accn: f for f in parse_concept(1001, *DEI, payload(1001, DEI, [summer, winter]))}

    # EDT: 04:00 UTC; EST: 05:00 UTC
    assert facts["a-1"].available_at == datetime(2024, 8, 1, 4, 0, tzinfo=UTC)
    assert facts["a-2"].available_at == datetime(2024, 2, 1, 5, 0, tzinfo=UTC)
    assert facts["a-1"].available_at.tzinfo is not None


def test_available_at_crosses_a_month_and_a_year_end() -> None:
    fact = raw_fact(filed="2024-12-31")
    [parsed] = parse_concept(1001, *DEI, payload(1001, DEI, [fact]))
    assert parsed.available_at == datetime(2025, 1, 1, tzinfo=NY)


def test_the_form_is_kept_as_given() -> None:
    facts = parse_concept(
        1001,
        *DEI,
        payload(
            1001, DEI, [raw_fact(form="10-K/A", accn="a-1"), raw_fact(form="40-F", accn="a-2")]
        ),
    )
    assert sorted(f.form for f in facts) == ["10-K/A", "40-F"]


def test_the_same_fact_in_a_later_filing_is_a_separate_fact() -> None:
    # comparatives: one count repeated by the next filing under its own accn (measured)
    first = raw_fact(accn="a-1", filed="2024-05-03")
    again = raw_fact(accn="a-2", filed="2024-08-02")
    facts = parse_concept(1001, *DEI, payload(1001, DEI, [first, again]))
    assert sorted((f.accn, f.filed) for f in facts) == [
        ("a-1", date(2024, 5, 3)),
        ("a-2", date(2024, 8, 2)),
    ]


# =============================================================================================
# parse_concept: boundaries
# =============================================================================================


@pytest.mark.parametrize("val", [0, -5, -1_000_000_000])
def test_a_count_of_zero_or_less_is_dropped(val: int) -> None:
    facts = parse_concept(1001, *DEI, payload(1001, DEI, [raw_fact(val=val), raw_fact(accn="ok")]))
    assert [f.accn for f in facts] == ["ok"]


@pytest.mark.parametrize("field", ["end", "filed", "val", "accn"])
def test_a_fact_missing_a_required_field_is_dropped(field: str) -> None:
    broken = raw_fact(accn="broken")
    del broken[field]
    facts = parse_concept(1001, *DEI, payload(1001, DEI, [broken, raw_fact(accn="ok")]))
    assert [f.accn for f in facts] == ["ok"]


def test_an_exact_duplicate_is_kept_once() -> None:
    fact = raw_fact(accn="a-1")
    # same concept, accn, end and value; frame / fp differ the way the API repeats facts
    copy = {**fact, "frame": "CY2024Q1I", "fp": "Q2"}
    facts = parse_concept(1001, *DEI, payload(1001, DEI, [fact, copy]))
    assert len(facts) == 1


def test_same_accn_and_end_with_another_value_is_not_a_duplicate() -> None:
    facts = parse_concept(
        1001,
        *DEI,
        payload(1001, DEI, [raw_fact(accn="a-1", val=100), raw_fact(accn="a-1", val=200)]),
    )
    assert sorted(f.shares for f in facts) == [100.0, 200.0]


def test_a_reply_without_a_shares_unit_has_no_facts() -> None:
    usd_only = {**payload(1001, DEI, []), "units": {"USD": [raw_fact()]}}
    assert parse_concept(1001, *DEI, usd_only) == []
    assert parse_concept(1001, *DEI, {**payload(1001, DEI, []), "units": {}}) == []


def test_the_cik_and_concept_come_from_the_arguments() -> None:
    [fact] = parse_concept(2002, *GAAP, payload(2002, GAAP, [raw_fact()]))
    assert fact.cik == 2002
    assert fact.concept == "us-gaap:WeightedAverageNumberOfSharesOutstandingBasic"


# =============================================================================================
# fetch_share_facts
# =============================================================================================


def test_fetch_asks_every_concept_of_every_cik(tmp_path: Path) -> None:
    routes = {
        url(AAPL_CIK, DEI): reply(200, AAPL_DEI),
        **missing(AAPL_CIK, GAAP, IFRS, BAL),
        url(AZN_CIK, IFRS): reply(200, AZN_IFRS),
        **missing(AZN_CIK, DEI, GAAP, BAL),
    }
    fetch = RoutedFetch(routes)

    facts = fetch_share_facts([AAPL_CIK, AZN_CIK], fetch, tmp_path / "cache")

    assert sorted(fetch.urls) == sorted(routes)
    assert len(fetch.urls) == 8  # each exactly once
    assert {(f.cik, f.concept) for f in facts} == {
        (AAPL_CIK, "dei:EntityCommonStockSharesOutstanding"),
        (AZN_CIK, "ifrs-full:WeightedAverageShares"),
    }
    assert len(facts) == 3


def test_fetch_reads_every_concept_even_when_the_first_has_facts(tmp_path: Path) -> None:
    gaap = payload(1001, GAAP, [raw_fact(accn="g-1")])
    ifrs = payload(1001, IFRS, [raw_fact(accn="i-1")])
    routes = {
        url(1001, DEI): reply(200, payload(1001, DEI, [raw_fact(accn="d-1")])),
        url(1001, GAAP): reply(200, gaap),
        url(1001, IFRS): reply(200, ifrs),
        **missing(1001, BAL),
    }

    facts = fetch_share_facts([1001], RoutedFetch(routes), tmp_path / "cache")

    assert {f.accn for f in facts} == {"d-1", "g-1", "i-1"}


def test_a_404_means_no_facts_not_an_error(tmp_path: Path) -> None:
    # GOOGL dei and SHOP dei answer 404 (measured 2026-10-29)
    fetch = RoutedFetch(missing(1001, DEI, GAAP, IFRS, BAL))
    assert fetch_share_facts([1001], fetch, tmp_path / "cache") == []
    assert len(fetch.urls) == 4


@pytest.mark.parametrize("status", [500, 403, 429])
def test_any_other_failure_raises_and_is_asked_once(tmp_path: Path, status: int) -> None:
    # retries belong to the fetch the caller passes in (retrying(rate_limited(...)), as for EDGAR)
    routes = {
        url(1001, DEI): reply(status, {"message": "error"}),
        url(1001, GAAP): reply(200, payload(1001, GAAP, [raw_fact()])),
        url(1001, IFRS): reply(200, payload(1001, IFRS, [])),
        **missing(1001, BAL),
    }
    fetch = RoutedFetch(routes)

    with pytest.raises(RuntimeError):
        fetch_share_facts([1001], fetch, tmp_path / "cache")
    assert fetch.urls.count(url(1001, DEI)) == 1


def test_a_cached_concept_is_not_fetched_again(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    routes = {url(AAPL_CIK, DEI): reply(200, AAPL_DEI), **missing(AAPL_CIK, GAAP, IFRS, BAL)}
    first = fetch_share_facts([AAPL_CIK], RoutedFetch(routes), cache)

    second = fetch_share_facts([AAPL_CIK], no_network, cache)

    assert sorted(second, key=lambda f: f.accn) == sorted(first, key=lambda f: f.accn)
    assert len(second) == 2


def test_a_cached_404_is_not_fetched_again(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    fetch_share_facts([1001], RoutedFetch(missing(1001, DEI, GAAP, IFRS, BAL)), cache)
    assert fetch_share_facts([1001], no_network, cache) == []


def test_the_cache_holds_one_file_per_cik_and_concept(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    routes = {
        url(AAPL_CIK, DEI): reply(200, AAPL_DEI),
        **missing(AAPL_CIK, GAAP, IFRS, BAL),
        **missing(AZN_CIK, DEI, GAAP, BAL),
        url(AZN_CIK, IFRS): reply(200, AZN_IFRS),
    }
    fetch_share_facts([AAPL_CIK, AZN_CIK], RoutedFetch(routes), cache)
    assert len([p for p in cache.rglob("*") if p.is_file()]) == 8


def test_only_a_new_cik_is_fetched_on_a_second_call(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    fetch_share_facts([1001], RoutedFetch(missing(1001, DEI, GAAP, IFRS, BAL)), cache)
    later = RoutedFetch(missing(2002, DEI, GAAP, IFRS, BAL))

    fetch_share_facts([1001, 2002], later, cache)

    assert sorted(later.urls) == sorted(url(2002, c) for c in (DEI, GAAP, IFRS, BAL))


def test_a_failed_request_is_not_cached(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    failing = {
        url(1001, DEI): reply(500, {"message": "error"}),
        **missing(1001, GAAP, IFRS, BAL),
    }
    with pytest.raises(RuntimeError):
        fetch_share_facts([1001], RoutedFetch(failing), cache)

    healed = RoutedFetch(
        {
            url(1001, DEI): reply(200, payload(1001, DEI, [raw_fact()])),
            **missing(1001, GAAP, IFRS, BAL),
        }
    )
    facts = fetch_share_facts([1001], healed, cache)

    assert url(1001, DEI) in healed.urls
    assert len(facts) == 1


def test_no_ciks_makes_no_request(tmp_path: Path) -> None:
    assert fetch_share_facts([], no_network, tmp_path / "cache") == []


# =============================================================================================
# write_share_facts
# =============================================================================================


def test_write_puts_the_file_under_the_market_root(tmp_path: Path) -> None:
    root = market_root(tmp_path)
    facts = parse_concept(AAPL_CIK, *DEI, AAPL_DEI)

    path = write_share_facts(facts, root)

    assert path == root / SHARES_FILE
    table = pq.read_table(path)
    assert table.num_rows == 2
    assert {f.name for f in fields(ShareFact)} <= set(table.schema.names)


def test_write_replaces_the_previous_file(tmp_path: Path) -> None:
    root = market_root(tmp_path)
    write_share_facts(parse_concept(AAPL_CIK, *DEI, AAPL_DEI), root)

    path = write_share_facts(parse_concept(AZN_CIK, *IFRS, AZN_IFRS), root)

    assert pq.read_table(path).column("cik").to_pylist() == [AZN_CIK]


def test_write_of_no_facts_leaves_an_empty_file(tmp_path: Path) -> None:
    root = market_root(tmp_path)
    write_share_facts(parse_concept(AAPL_CIK, *DEI, AAPL_DEI), root)

    path = write_share_facts([], root)

    assert pq.read_table(path).num_rows == 0


def test_payload_fixtures_are_json_serialisable() -> None:
    # fixture sanity: the scripted replies are what the real API would send
    for body in (AAPL_DEI, AZN_IFRS):
        assert json.loads(json.dumps(body)) == body


# =============================================================================================
# fetch_share_facts: an empty companyconcept reply falls back to companyfacts
# =============================================================================================

# Measured 2026-10-29: for 43 (CIK, concept) pairs companyconcept answers 200 with
# "units": {"shares": {}} (ABT dei and us-gaap), while companyfacts for the same CIK holds the
# facts (ABT dei: 68). A sample of 40 CIKs x 3 concepts otherwise agreed fact for fact.


def facts_url(cik: int) -> str:
    return f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"


def empty_reply(cik: int, concept: tuple[str, str]) -> dict[str, Any]:
    return {**payload(cik, concept, []), "units": {"shares": {}}}


def companyfacts(cik: int, concepts: Mapping[tuple[str, str], list[dict[str, Any]]]) -> dict:
    facts: dict[str, dict[str, Any]] = {}
    for (taxonomy, tag), rows in concepts.items():
        facts.setdefault(taxonomy, {})[tag] = {"units": {"shares": rows}}
    return {"cik": cik, "entityName": f"Company {cik}", "facts": facts}


def test_an_empty_concept_reply_is_read_from_companyfacts(tmp_path: Path) -> None:
    full = companyfacts(
        1001, {DEI: [raw_fact(accn="d-1")], GAAP: [raw_fact(accn="g-1"), raw_fact(accn="g-2")]}
    )
    routes = {
        url(1001, DEI): reply(200, empty_reply(1001, DEI)),
        url(1001, GAAP): reply(200, empty_reply(1001, GAAP)),
        **missing(1001, IFRS, BAL),
        facts_url(1001): reply(200, full),
    }
    fetch = RoutedFetch(routes)

    facts = fetch_share_facts([1001], fetch, tmp_path / "cache")

    assert sorted((f.concept, f.accn) for f in facts) == [
        ("dei:EntityCommonStockSharesOutstanding", "d-1"),
        ("us-gaap:WeightedAverageNumberOfSharesOutstandingBasic", "g-1"),
        ("us-gaap:WeightedAverageNumberOfSharesOutstandingBasic", "g-2"),
    ]
    assert fetch.urls.count(facts_url(1001)) == 1  # one companyfacts reply serves both


def test_companyfacts_is_cached_too(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    routes = {
        url(1001, DEI): reply(200, empty_reply(1001, DEI)),
        **missing(1001, GAAP, IFRS, BAL),
        facts_url(1001): reply(200, companyfacts(1001, {DEI: [raw_fact(accn="d-1")]})),
    }
    first = fetch_share_facts([1001], RoutedFetch(routes), cache)
    assert fetch_share_facts([1001], no_network, cache) == first


def test_a_concept_absent_from_companyfacts_has_no_facts(tmp_path: Path) -> None:
    routes = {
        url(1001, DEI): reply(200, empty_reply(1001, DEI)),
        **missing(1001, GAAP, IFRS, BAL),
        facts_url(1001): reply(200, companyfacts(1001, {GAAP: [raw_fact()]})),
    }
    assert fetch_share_facts([1001], RoutedFetch(routes), tmp_path / "cache") == []


def test_a_full_concept_reply_never_asks_companyfacts(tmp_path: Path) -> None:
    routes = {url(AAPL_CIK, DEI): reply(200, AAPL_DEI), **missing(AAPL_CIK, GAAP, IFRS, BAL)}
    fetch = RoutedFetch(routes)  # a companyfacts request would be an unexpected URL
    assert len(fetch_share_facts([AAPL_CIK], fetch, tmp_path / "cache")) == 2


def test_a_duration_facts_start_is_kept() -> None:
    [fact] = parse_concept(AZN_CIK, *IFRS, AZN_IFRS)
    assert fact.start == date(2025, 1, 1)
    [instant] = parse_concept(1001, *DEI, payload(1001, DEI, [raw_fact()]))
    assert instant.start is None


def test_a_quarter_and_a_half_year_of_one_filing_are_both_kept() -> None:
    quarter = raw_fact(accn="a-1", end="2024-06-30", val=9e8, start="2024-04-01")
    half = raw_fact(accn="a-1", end="2024-06-30", val=9e8, start="2024-01-01")
    facts = parse_concept(1001, *GAAP, payload(1001, GAAP, [quarter, half]))
    assert sorted(f.start for f in facts if f.start) == [date(2024, 1, 1), date(2024, 4, 1)]


def test_a_reply_that_does_not_parse_is_not_cached(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cut_off = FetchResult(200, {}, b'{"cik": 1001, "units": {"sha')
    cut = {url(1001, DEI): cut_off, **missing(1001, GAAP, IFRS, BAL)}
    with pytest.raises(json.JSONDecodeError):
        fetch_share_facts([1001], RoutedFetch(cut), cache)
    whole = reply(200, payload(1001, DEI, [raw_fact()]))
    healed = RoutedFetch({url(1001, DEI): whole, **missing(1001, GAAP, IFRS, BAL)})
    assert len(fetch_share_facts([1001], healed, cache)) == 1
