"""Guards against a silently wrong universe: bad watchlist entries fail loudly, never vanish."""

from collections.abc import Callable
from datetime import date
from pathlib import Path

import pytest

from asof.ingest.sec_tickers import SecListing, SecListings, load_sec_listings
from asof.ingest.watchlist import (
    ResolveResult,
    WatchlistItem,
    load_watchlist,
    resolve_watchlist,
)
from asof.store.entities import EntitySpec
from asof.store.models import EntityKind

REPO_ROOT = Path(__file__).resolve().parents[2]
ADDED = date(2026, 10, 5)
SECTOR_FUNDS = {"XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"}

WATCHLIST = """\
groups:
  - id: sectors
    label: The eleven GICS sectors
    kind: etf
    added: 2026-10-05
    items:
      - {ticker: XLK, note: "Technology"}
      - {ticker: xle}
      - {ticker: BTC, kind: crypto, added: 2026-11-01, name: Bitcoin}
  - id: mega_cap_tech
    label: Mega-cap technology
    kind: company
    added: 2026-10-06
    items:
      - {ticker: AAPL}
"""


def item(
    ticker: str, kind: str, *, name: str | None = None, note: str | None = None
) -> WatchlistItem:
    return WatchlistItem(ticker=ticker, kind=kind, group="g", added=ADDED, name=name, note=note)


@pytest.fixture
def listings(sec_dir: Path) -> SecListings:
    return load_sec_listings(sec_dir)


def test_items_inherit_group_defaults_and_may_override_them(
    write_watchlist: Callable[[str], Path],
) -> None:
    items = load_watchlist(write_watchlist(WATCHLIST))

    assert items == [
        WatchlistItem(ticker="XLK", kind="etf", group="sectors", added=ADDED, note="Technology"),
        WatchlistItem(ticker="XLE", kind="etf", group="sectors", added=ADDED),
        WatchlistItem(
            ticker="BTC", kind="crypto", group="sectors", added=date(2026, 11, 1), name="Bitcoin"
        ),
        WatchlistItem(
            ticker="AAPL", kind="company", group="mega_cap_tech", added=date(2026, 10, 6)
        ),
    ]
    assert items[1].name is None
    assert items[1].note is None


def test_item_may_supply_kind_and_added_when_the_group_has_none(
    write_watchlist: Callable[[str], Path],
) -> None:
    path = write_watchlist(
        """\
groups:
  - id: coins
    label: Coins
    items:
      - {ticker: eth, kind: crypto, added: 2026-11-01}
"""
    )

    assert load_watchlist(path) == [
        WatchlistItem(ticker="ETH", kind="crypto", group="coins", added=date(2026, 11, 1))
    ]


@pytest.mark.parametrize(
    "second",
    ["{ticker: XLK}", "{ticker: xlk}", "{ticker: XLK, kind: company}"],
    ids=["same", "different_case", "different_kind"],
)
def test_duplicate_ticker_is_rejected(write_watchlist: Callable[[str], Path], second: str) -> None:
    path = write_watchlist(
        f"""\
groups:
  - id: sectors
    label: Sectors
    kind: etf
    added: 2026-10-05
    items:
      - {{ticker: XLK}}
      - {{ticker: XLE}}
  - id: again
    label: Again
    kind: etf
    added: 2026-10-05
    items:
      - {second}
"""
    )

    with pytest.raises(ValueError, match="XLK"):
        load_watchlist(path)


@pytest.mark.parametrize(
    ("group_fields", "item_fields"),
    [
        ("kind: forex\n    added: 2026-10-05", ""),
        ("kind: etf\n    added: 2026-10-05", ", kind: forex"),
        ("added: 2026-10-05", ""),
        ("kind: etf", ""),
    ],
    ids=["unknown_group_kind", "unknown_item_kind", "no_kind", "no_added"],
)
def test_invalid_or_incomplete_item_is_rejected(
    write_watchlist: Callable[[str], Path], group_fields: str, item_fields: str
) -> None:
    path = write_watchlist(
        f"""\
groups:
  - id: sectors
    label: Sectors
    {group_fields}
    items:
      - {{ticker: XLK{item_fields}}}
"""
    )

    with pytest.raises(ValueError):
        load_watchlist(path)


def test_unquoted_ticker_that_yaml_reads_as_a_boolean_is_rejected(
    write_watchlist: Callable[[str], Path],
) -> None:
    path = write_watchlist(
        """\
groups:
  - id: semis
    label: Semis
    kind: company
    added: 2026-10-05
    items:
      - {ticker: ON}
"""
    )

    with pytest.raises(ValueError, match="quoted"):
        load_watchlist(path)


@pytest.mark.parametrize(
    "text", ["groups: []\n", "label: no groups here\n"], ids=["empty", "missing"]
)
def test_file_without_groups_is_rejected(write_watchlist: Callable[[str], Path], text: str) -> None:
    with pytest.raises(ValueError):
        load_watchlist(write_watchlist(text))


def test_repository_watchlist_is_valid() -> None:
    items = load_watchlist(REPO_ROOT / "configs" / "watchlist.yaml")
    tickers = [i.ticker for i in items]
    kinds = {kind.value for kind in EntityKind}

    assert len(items) >= 250
    assert len(set(tickers)) == len(tickers)
    assert all(t == t.upper() for t in tickers)
    assert all(i.kind in kinds for i in items)
    assert all(isinstance(i.added, date) for i in items)
    assert {i.ticker for i in items if i.kind == "etf"} >= SECTOR_FUNDS


def test_listed_company_takes_cik_and_name_from_the_sec(listings: SecListings) -> None:
    result = resolve_watchlist([item("AAPL", "company", name="ignored")], listings)

    assert result == ResolveResult(
        entities=[EntitySpec(kind="company", ticker="AAPL", name="Apple Inc.", cik=320193)],
        missing_companies=[],
        unmatched_others=[],
    )


def test_unlisted_company_is_reported_and_produces_no_entity(
    listings: SecListings,
) -> None:
    result = resolve_watchlist([item("ZZZZ", "company"), item("MU", "company")], listings)

    assert result.missing_companies == ["ZZZZ"]
    assert result.unmatched_others == []
    assert [e.ticker for e in result.entities] == ["MU"]


@pytest.mark.parametrize(
    ("fields", "expected_name"),
    [
        (
            {"name": "Technology Select Sector SPDR", "note": "Technology"},
            "Technology Select Sector SPDR",
        ),
        ({"note": "Technology"}, "Technology"),
        ({}, "XLK"),
    ],
    ids=["item_name", "item_note", "ticker"],
)
def test_fund_name_falls_back_when_the_sec_has_none(
    listings: SecListings, fields: dict[str, str], expected_name: str
) -> None:
    result = resolve_watchlist([item("XLK", "etf", **fields)], listings)

    assert result.entities == [
        EntitySpec(
            kind="etf", ticker="XLK", name=expected_name, cik=1064641, series_id="S000006415"
        )
    ]
    assert result.unmatched_others == []


def test_fund_prefers_the_sec_name(listings: SecListings) -> None:
    items = [item("SPY", "etf", name="mine", note="S&P 500"), item("QQQ", "etf", note="Nasdaq-100")]

    result = resolve_watchlist(items, listings)

    assert result.entities == [
        EntitySpec(kind="etf", ticker="SPY", name="SPDR S&P 500 ETF TRUST", cik=884394),
        EntitySpec(
            kind="etf",
            ticker="QQQ",
            name="INVESCO QQQ TRUST, SERIES 1",
            cik=1067839,
            series_id="S000101292",
        ),
    ]


def test_unlisted_fund_is_reported_but_still_produces_an_entity(
    listings: SecListings,
) -> None:
    result = resolve_watchlist([item("DRAM", "etf", note="Memory chips")], listings)

    assert result == ResolveResult(
        entities=[EntitySpec(kind="etf", ticker="DRAM", name="Memory chips")],
        missing_companies=[],
        unmatched_others=["DRAM"],
    )


def test_company_keeps_its_own_cik_when_a_fund_shares_its_ticker() -> None:
    listings = SecListings(
        companies={"SPCX": SecListing("SPCX", 1181412, "SPACE EXPLORATION TECHNOLOGIES CORP")},
        funds={"SPCX": SecListing("SPCX", 1719812, series_id="S000070261")},
    )

    result = resolve_watchlist([item("SPCX", "company")], listings)

    assert result.entities == [
        EntitySpec(
            kind="company",
            ticker="SPCX",
            name="SPACE EXPLORATION TECHNOLOGIES CORP",
            cik=1181412,
        )
    ]


def test_fund_listed_for_two_registrants_is_a_conflict_until_a_cik_is_given() -> None:
    listings = SecListings(
        companies={"GLD": SecListing("GLD", 1222333, "SPDR GOLD TRUST")},
        funds={"GLD": SecListing("GLD", 999, series_id="S000000009")},
    )

    undecided = resolve_watchlist([item("GLD", "etf", note="Gold")], listings)
    decided = resolve_watchlist(
        [WatchlistItem(ticker="GLD", kind="etf", group="g", added=ADDED, cik=1222333)], listings
    )

    assert (undecided.entities, undecided.conflicts) == ([], ["GLD"])
    assert decided.conflicts == []
    assert decided.entities == [
        EntitySpec(kind="etf", ticker="GLD", name="SPDR GOLD TRUST", cik=1222333)
    ]


def test_company_under_another_cik_than_the_file_says_is_a_conflict(
    listings: SecListings,
) -> None:
    wrong = WatchlistItem(ticker="AAPL", kind="company", group="g", added=ADDED, cik=1)

    result = resolve_watchlist([wrong], listings)

    assert (result.entities, result.conflicts, result.missing_companies) == ([], ["AAPL"], [])


@pytest.mark.parametrize(
    "entry", ["{ticker: ~}", "{ticker: 123}", "{ticker: 2026-10-05}", "AAPL", "{note: no ticker}"]
)
def test_item_that_is_not_a_quoted_ticker_mapping_is_rejected(
    write_watchlist: Callable[[str], Path], entry: str
) -> None:
    path = write_watchlist(
        f"""\
groups:
  - id: g
    label: G
    kind: company
    added: 2026-10-05
    items:
      - {entry}
"""
    )

    with pytest.raises(ValueError):
        load_watchlist(path)


def test_explicit_cik_is_read_from_the_file(write_watchlist: Callable[[str], Path]) -> None:
    path = write_watchlist(
        """\
groups:
  - id: g
    label: G
    kind: etf
    added: 2026-10-05
    items:
      - {ticker: GLD, cik: 1222333}
"""
    )

    assert load_watchlist(path)[0].cik == 1222333


def test_crypto_is_never_looked_up(listings: SecListings) -> None:
    # A fund trades under the same symbol; the coin must not inherit its identity.
    fund = SecListing("BTC", 2015034, "Grayscale Bitcoin Mini Trust ETF")
    listings = SecListings(companies={**listings.companies, "BTC": fund}, funds=listings.funds)
    items = [item("BTC", "crypto", name="Bitcoin"), item("ETH", "crypto", note="not a name")]

    result = resolve_watchlist(items, listings)

    assert result == ResolveResult(
        entities=[
            EntitySpec(kind="crypto", ticker="BTC", name="Bitcoin"),
            EntitySpec(kind="crypto", ticker="ETH", name="ETH"),
        ],
        missing_companies=[],
        unmatched_others=[],
    )


def test_entities_follow_item_order_and_reports_collect_every_miss(
    listings: SecListings,
) -> None:
    items = [
        item("XLE", "etf"),
        item("ZZZZ", "company"),
        item("BTC", "crypto"),
        item("UFO", "etf"),
        item("AAPL", "company"),
        item("YYYY", "company"),
        item("DRAM", "etf"),
        item("XLK", "etf"),
    ]

    result = resolve_watchlist(items, listings)

    assert [e.ticker for e in result.entities] == ["XLE", "BTC", "UFO", "AAPL", "DRAM", "XLK"]
    assert result.missing_companies == ["ZZZZ", "YYYY"]
    assert result.unmatched_others == ["UFO", "DRAM"]


def test_no_items_resolve_to_nothing(listings: SecListings) -> None:
    assert resolve_watchlist([], listings) == ResolveResult([], [], [])
