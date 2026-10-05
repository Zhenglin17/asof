"""Guards against wrong identities: a ticker must map to the CIK and series id the SEC lists."""

import json
import shutil
from pathlib import Path

import pytest

from asof.ingest import sec_tickers
from asof.ingest.sec_tickers import (
    COMPANY_FILE,
    FUND_FILE,
    AmbiguousTicker,
    SecListing,
    SecListings,
    download_sec_listings,
    load_sec_listings,
    parse_company_tickers,
    parse_fund_tickers,
)

SECTOR_SPDR_CIK = 1064641
XLK = SecListing(ticker="XLK", cik=SECTOR_SPDR_CIK, series_id="S000006415")
XLE = SecListing(ticker="XLE", cik=SECTOR_SPDR_CIK, series_id="S000006410")
FUND_QQQ = SecListing(ticker="QQQ", cik=1067839, series_id="S000101292")


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_file_names_match_the_sec_downloads() -> None:
    assert COMPANY_FILE == "company_tickers.json"
    assert FUND_FILE == "company_tickers_mf.json"


def test_company_table_maps_ticker_to_cik_and_title(sec_dir: Path) -> None:
    listings = parse_company_tickers(read(sec_dir / COMPANY_FILE))

    assert set(listings) == {"AAPL", "MU", "META", "SPY", "QQQ"}
    assert listings["AAPL"] == SecListing(ticker="AAPL", cik=320193, name="Apple Inc.")
    assert listings["SPY"] == SecListing(ticker="SPY", cik=884394, name="SPDR S&P 500 ETF TRUST")
    assert all(listing.series_id is None for listing in listings.values())
    assert all(type(listing.cik) is int for listing in listings.values())


def test_company_tickers_are_upper_cased() -> None:
    raw = {"0": {"cik_str": 1067983, "ticker": "brk-b", "title": "BERKSHIRE HATHAWAY INC"}}

    listings = parse_company_tickers(raw)

    assert list(listings) == ["BRK-B"]
    assert listings["BRK-B"].ticker == "BRK-B"


def test_empty_company_table_gives_no_listings() -> None:
    assert parse_company_tickers({}) == {}


def test_fund_table_keeps_series_id_and_has_no_names(sec_dir: Path) -> None:
    listings = parse_fund_tickers(read(sec_dir / FUND_FILE))

    assert listings == {"XLK": XLK, "XLE": XLE, "QQQ": FUND_QQQ}
    assert all(listing.name is None for listing in listings.values())


def test_funds_sharing_a_cik_stay_distinct_by_series(sec_dir: Path) -> None:
    listings = parse_fund_tickers(read(sec_dir / FUND_FILE))

    assert listings["XLK"].cik == listings["XLE"].cik == SECTOR_SPDR_CIK
    assert listings["XLK"].series_id != listings["XLE"].series_id


def test_fund_columns_are_located_by_field_name(sec_dir: Path, shuffled_sec_dir: Path) -> None:
    shuffled = read(shuffled_sec_dir / FUND_FILE)
    assert shuffled["fields"] != read(sec_dir / FUND_FILE)["fields"]

    assert parse_fund_tickers(shuffled) == {"XLK": XLK, "XLE": XLE, "QQQ": FUND_QQQ}


def test_fund_rows_with_empty_symbol_are_skipped() -> None:
    raw = {
        "fields": ["cik", "seriesId", "classId", "symbol"],
        "data": [
            [SECTOR_SPDR_CIK, "S000006411", "C000017597", ""],
            [SECTOR_SPDR_CIK, "S000006415", "C000017601", "xlk"],
        ],
    }

    assert parse_fund_tickers(raw) == {"XLK": XLK}


def test_fund_table_without_rows_gives_no_listings() -> None:
    raw = {"fields": ["cik", "seriesId", "classId", "symbol"], "data": []}
    assert parse_fund_tickers(raw) == {}


def test_load_keeps_the_two_tables_apart(sec_dir: Path) -> None:
    listings = load_sec_listings(sec_dir)

    assert set(listings.companies) == {"AAPL", "MU", "META", "SPY", "QQQ"}
    assert set(listings.funds) == {"XLK", "XLE", "QQQ"}
    assert listings.company("MU") == SecListing(
        ticker="MU", cik=723125, name="MICRON TECHNOLOGY INC"
    )
    assert listings.company("XLK") is None
    assert listings.company("ZZZZ") is None
    assert listings.fund("ZZZZ") is None


def test_fund_lookup_prefers_the_fund_table_and_falls_back_to_the_company_table(
    sec_dir: Path,
) -> None:
    listings = load_sec_listings(sec_dir)

    assert listings.fund("XLK") == XLK
    # A stand-alone trust is listed only as a company.
    assert listings.fund("SPY") == SecListing(
        ticker="SPY", cik=884394, name="SPDR S&P 500 ETF TRUST"
    )
    # Same registrant in both tables: series from the fund row, name from the company row.
    assert listings.fund("QQQ") == SecListing(
        ticker="QQQ", cik=1067839, name="INVESCO QQQ TRUST, SERIES 1", series_id="S000101292"
    )


def test_company_is_not_replaced_by_an_unrelated_fund_with_its_ticker() -> None:
    # Seen in the live tables: a stale fund row shares a newly listed company's ticker.
    company = SecListing(ticker="SPCX", cik=1181412, name="SPACE EXPLORATION TECHNOLOGIES CORP")
    stale_fund = SecListing(ticker="SPCX", cik=1719812, series_id="S000070261")
    listings = SecListings(companies={"SPCX": company}, funds={"SPCX": stale_fund})

    assert listings.company("SPCX") == company
    # As a fund the ticker is ambiguous until a CIK says which registrant is meant.
    with pytest.raises(AmbiguousTicker, match="SPCX"):
        listings.fund("SPCX")
    assert listings.fund("SPCX", cik=1719812) == stale_fund
    assert listings.fund("SPCX", cik=1181412) == company
    assert listings.fund("SPCX", cik=1) is None


@pytest.mark.parametrize("missing", [COMPANY_FILE, FUND_FILE])
def test_missing_table_file_raises(sec_dir: Path, tmp_path: Path, missing: str) -> None:
    for name in {COMPANY_FILE, FUND_FILE} - {missing}:
        shutil.copy(sec_dir / name, tmp_path / name)

    with pytest.raises(FileNotFoundError):
        load_sec_listings(tmp_path)


def serve(sec_dir: Path, broken: str | None = None):
    def fetch(url: str, user_agent: str) -> bytes:
        assert user_agent == "tests contact@example.com"
        name = url.rsplit("/", 1)[1]
        return b'{"error": "rate limited"}' if name == broken else (sec_dir / name).read_bytes()

    return fetch


def test_download_saves_both_tables(
    sec_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sec_tickers, "MIN_ROWS", 1)
    target = tmp_path / "raw" / "sec"

    download_sec_listings(target, "tests contact@example.com", fetch=serve(sec_dir))

    assert load_sec_listings(target) == load_sec_listings(sec_dir)
    assert sorted(p.name for p in target.iterdir()) == [COMPANY_FILE, FUND_FILE]


@pytest.mark.parametrize("broken", [COMPANY_FILE, FUND_FILE])
def test_failed_download_leaves_the_saved_pair_untouched(
    sec_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, broken: str
) -> None:
    monkeypatch.setattr(sec_tickers, "MIN_ROWS", 1)
    for name in (COMPANY_FILE, FUND_FILE):
        (tmp_path / name).write_text("saved earlier")

    with pytest.raises(ValueError, match=broken):
        download_sec_listings(tmp_path, "tests contact@example.com", fetch=serve(sec_dir, broken))

    assert {p.name: p.read_text() for p in tmp_path.iterdir()} == {
        COMPANY_FILE: "saved earlier",
        FUND_FILE: "saved earlier",
    }


def test_implausibly_small_download_is_refused(sec_dir: Path, tmp_path: Path) -> None:
    # The fixture tables parse fine but are far smaller than the real ones.
    with pytest.raises(ValueError, match="rows"):
        download_sec_listings(tmp_path, "tests contact@example.com", fetch=serve(sec_dir))

    assert list(tmp_path.iterdir()) == []
