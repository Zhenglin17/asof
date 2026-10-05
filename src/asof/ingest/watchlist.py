"""The watchlist file and how its tickers become entities."""

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, field_validator

from asof.ingest.sec_tickers import AmbiguousTicker, SecListings
from asof.store.entities import EntitySpec
from asof.store.models import EntityKind


class WatchlistItem(BaseModel):
    ticker: str
    kind: str
    group: str
    added: date
    name: str | None = None
    note: str | None = None
    # Only needed when the SEC lists the ticker for more than one registrant.
    cik: int | None = None

    @field_validator("kind")
    @classmethod
    def _known_kind(cls, value: str) -> str:
        if value not in set(EntityKind):
            raise ValueError(f"unknown kind {value!r}")
        return value


def load_watchlist(path: Path) -> list[WatchlistItem]:
    """Read the file in order. `kind` and `added` fall back from the item to its group."""
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as error:
        raise ValueError(f"{path}: not valid YAML: {error}") from error
    groups = raw.get("groups") if isinstance(raw, dict) else None
    if not groups:
        raise ValueError(f"{path}: no groups")

    items: list[WatchlistItem] = []
    seen: set[str] = set()
    for group in groups:
        if not isinstance(group, dict) or not isinstance(group.get("id"), str):
            raise ValueError(f"{path}: every group needs an id")
        for entry in group.get("items") or []:
            if not isinstance(entry, dict):
                raise ValueError(f"{path}: item {entry!r} in {group['id']} must be a mapping")
            symbol = entry.get("ticker")
            if not isinstance(symbol, str):
                # YAML reads unquoted ON, NO, 123 or ~ as a boolean, a number or null.
                raise ValueError(f"{path}: ticker {symbol!r} in {group['id']} must be quoted")
            ticker = symbol.upper()
            if ticker in seen:
                raise ValueError(f"{path}: duplicate ticker {ticker}")
            seen.add(ticker)

            kind = entry.get("kind", group.get("kind"))
            added = entry.get("added", group.get("added"))
            if kind is None or added is None:
                raise ValueError(f"{path}: {ticker} needs both kind and added")
            items.append(
                WatchlistItem(
                    ticker=ticker,
                    kind=kind,
                    group=group["id"],
                    added=added,
                    name=entry.get("name"),
                    note=entry.get("note"),
                    cik=entry.get("cik"),
                )
            )
    return items


@dataclass(frozen=True)
class ResolveResult:
    entities: list[EntitySpec] = field(default_factory=list)
    # Companies the SEC does not list. No entity is produced: a company needs its CIK.
    missing_companies: list[str] = field(default_factory=list)
    # Funds the SEC does not list. Still produced, without identifiers.
    unmatched_others: list[str] = field(default_factory=list)
    # Tickers the SEC lists for more than one registrant, or under another CIK than the file
    # says. No entity is produced: guessing here is how one instrument gets another's identity.
    conflicts: list[str] = field(default_factory=list)


def resolve_watchlist(items: list[WatchlistItem], listings: SecListings) -> ResolveResult:
    result = ResolveResult()
    for item in items:
        if item.kind == EntityKind.CRYPTO:
            # A fund may trade under a coin's symbol; the coin must not take its identity.
            result.entities.append(
                EntitySpec(kind=item.kind, ticker=item.ticker, name=item.name or item.ticker)
            )
            continue

        if item.kind == EntityKind.COMPANY:
            listing = listings.company(item.ticker)
            if listing is None:
                result.missing_companies.append(item.ticker)
                continue
            if item.cik is not None and item.cik != listing.cik:
                result.conflicts.append(item.ticker)
                continue
            name = listing.name or item.name or item.ticker
            result.entities.append(
                EntitySpec(kind=item.kind, ticker=item.ticker, name=name, cik=listing.cik)
            )
            continue

        try:
            listing = listings.fund(item.ticker, item.cik)
        except AmbiguousTicker:
            result.conflicts.append(item.ticker)
            continue
        fallback = item.name or item.note or item.ticker
        if listing is None:
            result.unmatched_others.append(item.ticker)
            result.entities.append(EntitySpec(kind=item.kind, ticker=item.ticker, name=fallback))
            continue
        result.entities.append(
            EntitySpec(
                kind=item.kind,
                ticker=item.ticker,
                name=listing.name or fallback,
                cik=listing.cik,
                series_id=listing.series_id,
            )
        )
    return result
