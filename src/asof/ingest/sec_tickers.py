"""SEC ticker tables: which CIK, and for funds which series, a ticker belongs to today.

Two files are published. `company_tickers.json` lists operating companies and stand-alone
trusts with a name. `company_tickers_mf.json` lists funds with a series id and no name; many
funds share one CIK there, so the series id is what tells them apart.

Both are snapshots of the current mapping. They say nothing about who held a ticker in the past.
The two tables are kept apart: the same ticker can appear in both for unrelated registrants
(a stale fund row can outlive the fund and collide with a newly listed company).
"""

import json
import os
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

COMPANY_FILE = "company_tickers.json"
FUND_FILE = "company_tickers_mf.json"
SEC_FILES_URL = "https://www.sec.gov/files"
# Each real table has well over ten thousand rows; an error body has none.
MIN_ROWS = 1000


@dataclass(frozen=True)
class SecListing:
    ticker: str
    cik: int
    name: str | None = None
    series_id: str | None = None


def parse_company_tickers(raw: dict[str, Any]) -> dict[str, SecListing]:
    listings: dict[str, SecListing] = {}
    for row in raw.values():
        ticker = str(row["ticker"]).upper()
        listings[ticker] = SecListing(ticker=ticker, cik=int(row["cik_str"]), name=row["title"])
    return listings


def parse_fund_tickers(raw: dict[str, Any]) -> dict[str, SecListing]:
    rows = raw.get("data") or []
    if not rows:
        return {}
    # Column order is whatever `fields` says, not a fixed position.
    fields = raw["fields"]
    cik_at, series_at, symbol_at = (fields.index(f) for f in ("cik", "seriesId", "symbol"))

    listings: dict[str, SecListing] = {}
    for row in rows:
        symbol = row[symbol_at]
        if not symbol:
            continue
        ticker = str(symbol).upper()
        listings[ticker] = SecListing(ticker=ticker, cik=int(row[cik_at]), series_id=row[series_at])
    return listings


class AmbiguousTicker(ValueError):
    """Both tables list the ticker, for different registrants, and nothing says which is meant."""


@dataclass(frozen=True)
class SecListings:
    companies: dict[str, SecListing]
    funds: dict[str, SecListing]

    def company(self, ticker: str) -> SecListing | None:
        """A company is only ever looked up in the company table."""
        return self.companies.get(ticker)

    def fund(self, ticker: str, cik: int | None = None) -> SecListing | None:
        """Fund table first; stand-alone trusts such as SPY live only in the company table.

        When both tables list the ticker under different CIKs, one of the rows is someone else
        (often a stale fund row). Pass `cik` to say which registrant is meant.
        """
        rows = [row for row in (self.funds.get(ticker), self.companies.get(ticker)) if row]
        if cik is not None:
            rows = [row for row in rows if row.cik == cik]
        if not rows:
            return None
        if len(rows) == 1:
            return rows[0]
        fund, company = rows
        if fund.cik != company.cik:
            raise AmbiguousTicker(
                f"{ticker} is listed under CIK {fund.cik} (fund) and CIK {company.cik} (company)"
            )
        # Same registrant: the fund row has the series, the company row has the name.
        return replace(fund, name=company.name)


def load_sec_listings(directory: Path) -> SecListings:
    """Read both saved tables."""
    return SecListings(
        companies=parse_company_tickers(json.loads((directory / COMPANY_FILE).read_text())),
        funds=parse_fund_tickers(json.loads((directory / FUND_FILE).read_text())),
    )


def _fetch(url: str, user_agent: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def download_sec_listings(
    directory: Path, user_agent: str, fetch: Callable[[str, str], bytes] = _fetch
) -> None:
    """Fetch both tables, then replace the saved pair only if both look like real tables.

    The SEC rejects requests whose User-Agent has no contact address, and an error body can be
    valid JSON, so each download must parse as its table and be plausibly large.
    """
    parsers = {COMPANY_FILE: parse_company_tickers, FUND_FILE: parse_fund_tickers}
    bodies: dict[str, bytes] = {}
    for name, parse in parsers.items():
        body = fetch(f"{SEC_FILES_URL}/{name}", user_agent)
        try:
            rows = len(parse(json.loads(body)))
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            raise ValueError(f"{name}: the download is not a ticker table") from error
        if rows < MIN_ROWS:
            raise ValueError(f"{name}: only {rows} rows, expected at least {MIN_ROWS}")
        bodies[name] = body

    directory.mkdir(parents=True, exist_ok=True)
    for name, body in bodies.items():
        partial = directory / f"{name}.part"
        partial.write_bytes(body)
        os.replace(partial, directory / name)
