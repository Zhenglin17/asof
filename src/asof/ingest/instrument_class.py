"""Instrument class of every security-master segment: common stock, ETF, warrant, unit...

The scanner only looks at common stocks and ETFs; warrants and SPAC units move 50% in a day and
would drown it. Nobody publishes an instrument-type field (Alpaca's asset table has none, the SEC
tables list the issuer), so the class is read off the *name* a segment carried, through an ordered
rule file (``configs/instrument_class_rules.yaml``). The rules were calibrated on the 2026-10-05
Alpaca asset table and the SEC ticker history; the rule file's header lists the traps they avoid.

Names come from two places, chosen per segment so a reused ticker is named after the right era:
Alpaca's asset table (one active and possibly one inactive row per symbol; the active row names
the open segment, the inactive row the closed ones) and the SEC monthly snapshots (the issuer's
name as of each snapshot inside the segment). 6,281 of 21,815 segments only have an SEC name,
which is the issuer's name and says nothing about the instrument ("ADIAL PHARMACEUTICALS, INC."
for the warrant ADILW); for those, the Nasdaq fifth-letter convention (W warrant, R right, U unit,
P preferred) is trusted only when the issuer also lists a sibling ticker: a shorter one that is a
prefix of it (ADIL for ADILW, EXE for EXEEW) or a 5-letter share class on the same 4-letter stem
(LBRDA for the preferred LBRDP). An SEC-only name under a CIK that lists ``SHELF_MIN_TICKERS``
or more tickers is an issuer shelf, not a company: Barclays Bank PLC, UBS AG and Credit Suisse AG
carry 28-78 ETN tickers each (TVIX, DGAZ, UGAZ), ProShares Trust II 20. Those segments are left
``unknown`` for the audit rather than guessed ``common`` from "AG" or "PLC".

There is no ``as_of`` here: the class is a property of the name and the symbol, not of the time
of observation. The read path filters segments through the security master's ``available_at``;
this table only decorates them.
"""

from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from asof.ingest.sec_tickers import load_sec_listings
from asof.ingest.security_master import MASTER_FILE, SEC_HISTORY_START
from asof.ingest.universe import load_assets, load_history

CLASSES = (
    "common",
    "etf",
    "leveraged_etf",
    "warrant",
    "unit",
    "preferred",
    "right",
    "note",
    "crypto",
    "unknown",
)
LOOKUPS = ("sec_funds", "nasdaq_suffix", "sec_shelf", "sec_companies")
NASDAQ_SUFFIX = {"W": "warrant", "R": "right", "U": "unit", "P": "preferred"}
NASDAQ_SUFFIX_SHAPE = re.compile(r"[A-Z]{4}[WRUP]")
MIN_PREFIX_LEN = 3  # EXE -> EXEEW counts; a 2-letter prefix matches too much by accident
SHELF_MIN_TICKERS = 20  # tickers under one CIK; SPACs and preferred-heavy companies stay < 20
SEC_NAME_SOURCES = ("sec", "none")  # name sources that carry the issuer's name, not the product's
CLASS_FILE = "symbols/instrument_class.parquet"
CLASS_SCHEMA = pa.schema(
    [
        ("security_id", pa.string()),
        ("symbol", pa.string()),
        ("valid_from", pa.date32()),
        ("valid_to", pa.date32()),
        ("name", pa.string()),
        ("name_source", pa.string()),
        ("class", pa.string()),
        ("rule", pa.string()),
    ]
)
NO_MATCH = ("unknown", "none")
OVERRIDE_RULE = "override"
OVERRIDE_KEYS = frozenset({"symbol", "class", "reason", "valid_from"})


@dataclass(frozen=True)
class ClassOverride:
    """A human decision for segments the name rules cannot settle (an issuer shelf: BAM is the
    common stock under a CIK that also lists 20+ notes, TVIX an ETN under "CREDIT SUISSE AG").
    ``valid_from`` pins one segment of the symbol; ``None`` covers every segment of it."""

    symbol: str
    cls: str
    reason: str
    valid_from: date | None = None


def load_class_overrides(path: Path) -> list[ClassOverride]:
    """``configs/instrument_class_overrides.yaml``: a list of ``symbol`` / ``class`` / ``reason``
    mappings, optionally dated. Every entry must name a real class (never ``unknown``) and a
    reason; duplicates and unknown keys are errors."""
    if not path.exists():
        raise FileNotFoundError(f"{path}: overrides file not found")
    raw = yaml.safe_load(path.read_text()) or []
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a list of overrides")
    loaded: list[ClassOverride] = []
    seen: set[tuple[str, date | None]] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError(f"{path}: every override must be a mapping: {item!r}")
        extra = set(item) - OVERRIDE_KEYS
        if extra:
            raise ValueError(f"{path}: unknown keys {sorted(extra)} in {item!r}")
        symbol, cls, reason = item.get("symbol"), item.get("class"), item.get("reason")
        if not isinstance(symbol, str) or not symbol:
            raise ValueError(f"{path}: override needs a symbol: {item!r}")
        if cls not in CLASSES or cls == "unknown":
            raise ValueError(f"{path}: {symbol}: class must be one of {CLASSES[:-1]}: {cls!r}")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"{path}: {symbol}: override needs a reason")
        valid_from = item.get("valid_from")
        if valid_from is not None and (
            isinstance(valid_from, datetime) or not isinstance(valid_from, date)
        ):
            raise ValueError(f"{path}: {symbol}: valid_from must be a date: {valid_from!r}")
        key = (symbol, valid_from)
        if key in seen:
            raise ValueError(f"{path}: duplicate override for {symbol} ({valid_from})")
        seen.add(key)
        loaded.append(ClassOverride(symbol, cls, reason, valid_from))
    return loaded


def apply_overrides(table: pa.Table, overrides: Sequence[ClassOverride]) -> pa.Table:
    """Set ``class`` and ``rule="override"`` on every row an override matches, in file order
    (a later override wins). An override matching no row is an error: a stale human decision
    must be noticed, not skipped. An undated override whose rows belong to more than one
    security is an error too: the ticker was reused and each era needs its own dated line."""
    if not overrides:
        return table
    symbols = table.column("symbol").to_pylist()
    security_ids = table.column("security_id").to_pylist()
    valid_froms = table.column("valid_from").to_pylist()
    classes = table.column("class").to_pylist()
    rules = table.column("rule").to_pylist()
    for override in overrides:
        hits = [
            i
            for i, (symbol, valid_from) in enumerate(zip(symbols, valid_froms, strict=True))
            if symbol == override.symbol
            and (override.valid_from is None or valid_from == override.valid_from)
        ]
        if not hits:
            raise ValueError(
                f"override for {override.symbol} ({override.valid_from or 'all segments'}) "
                "matches no security-master segment"
            )
        securities = {security_ids[i] for i in hits}
        if override.valid_from is None and len(securities) > 1:
            # A reused ticker: DGAZ the ETN and whatever company takes the symbol later are
            # different securities, and one undated decision must not silently cover both.
            raise ValueError(
                f"override for {override.symbol} without valid_from spans {len(securities)} "
                f"securities ({', '.join(sorted(securities))}); add valid_from to pin one segment"
            )
        for i in hits:
            classes[i] = override.cls
            rules[i] = OVERRIDE_RULE
    table = table.set_column(
        table.schema.get_field_index("class"), "class", pa.array(classes, pa.string())
    )
    return table.set_column(
        table.schema.get_field_index("rule"), "rule", pa.array(rules, pa.string())
    )


@dataclass(frozen=True)
class Rule:
    """One line of the rule file: a class and exactly one way of matching it."""

    id: str
    cls: str
    symbol: str | None = None
    name: str | None = None
    lookup: str | None = None


@dataclass(frozen=True)
class Lookups:
    """What the SEC tables know: fund tickers, and which tickers share a CIK."""

    fund_symbols: frozenset[str]
    ciks_by_ticker: dict[str, frozenset[int]]
    tickers_by_cik: dict[int, frozenset[str]]


def load_rules(path: Path) -> list[Rule]:
    """Read the ordered rule list; refuse anything a later rule or audit could misread."""
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path}: expected a non-empty list of rules")
    rules: list[Rule] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict) or "id" not in item or "class" not in item:
            raise ValueError(f"{path}: every rule needs an id and a class: {item!r}")
        rule_id, cls = str(item["id"]), str(item["class"])
        if rule_id in seen:
            raise ValueError(f"{path}: duplicate rule id {rule_id}")
        seen.add(rule_id)
        if cls not in CLASSES:
            raise ValueError(f"{path}: rule {rule_id}: unknown class {cls!r}")
        matchers = {k: item[k] for k in ("symbol", "name", "lookup") if item.get(k) is not None}
        if len(matchers) != 1:
            raise ValueError(f"{path}: rule {rule_id}: exactly one of symbol/name/lookup")
        if "lookup" in matchers and matchers["lookup"] not in LOOKUPS:
            raise ValueError(f"{path}: rule {rule_id}: unknown lookup {matchers['lookup']!r}")
        for key in ("symbol", "name"):
            if key in matchers:
                try:
                    re.compile(str(matchers[key]))
                except re.error as error:
                    raise ValueError(f"{path}: rule {rule_id}: bad {key} regex: {error}") from error
        rules.append(
            Rule(
                id=rule_id,
                cls=cls,
                symbol=_opt_str(matchers.get("symbol")),
                name=_opt_str(matchers.get("name")),
                lookup=_opt_str(matchers.get("lookup")),
            )
        )
    return rules


def _opt_str(value: Any) -> str | None:
    return None if value is None else str(value)


@lru_cache(maxsize=256)
def _compiled(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern)


def _nasdaq_suffix_class(symbol: str, lookups: Lookups) -> str | None:
    """Fifth-letter class, only when the issuer also lists a prefix ticker (ADIL for ADILW)."""
    if not NASDAQ_SUFFIX_SHAPE.fullmatch(symbol):
        return None
    for cik in lookups.ciks_by_ticker.get(symbol, ()):
        for sibling in lookups.tickers_by_cik.get(cik, ()):
            if sibling == symbol:
                continue
            is_prefix = len(sibling) >= MIN_PREFIX_LEN and symbol.startswith(sibling)
            same_stem = len(sibling) == 5 and sibling[:4] == symbol[:4]
            if is_prefix or same_stem:
                return NASDAQ_SUFFIX[symbol[-1]]
    return None


def _on_issuer_shelf(symbol: str, lookups: Lookups) -> bool:
    return any(
        len(lookups.tickers_by_cik.get(cik, ())) >= SHELF_MIN_TICKERS
        for cik in lookups.ciks_by_ticker.get(symbol, ())
    )


def classify(
    symbol: str,
    name: str,
    cik: int | None,
    lookups: Lookups,
    rules: Sequence[Rule],
    name_source: str = "alpaca_active",
) -> tuple[str, str]:
    """First rule that matches wins; returns ``(class, rule id)`` or ``("unknown", "none")``.

    ``name_source`` only matters to the ``sec_shelf`` lookup, which applies to issuer names.
    """
    lowered = name.lower()
    for rule in rules:
        if rule.symbol is not None:
            if _compiled(rule.symbol).fullmatch(symbol):
                return rule.cls, rule.id
        elif rule.name is not None:
            if lowered and _compiled(rule.name).search(lowered):
                return rule.cls, rule.id
        elif rule.lookup == "sec_funds":
            if symbol in lookups.fund_symbols:
                return rule.cls, rule.id
        elif rule.lookup == "nasdaq_suffix":
            cls = _nasdaq_suffix_class(symbol, lookups)
            if cls is not None:
                return cls, rule.id
        elif rule.lookup == "sec_shelf":
            if name_source in SEC_NAME_SOURCES and _on_issuer_shelf(symbol, lookups):
                return rule.cls, rule.id
        elif rule.lookup == "sec_companies":
            if cik is not None or symbol in lookups.ciks_by_ticker:
                return rule.cls, rule.id
    return NO_MATCH


def build_lookups(sec_history: pa.Table, fund_symbols: Iterable[str]) -> Lookups:
    by_ticker: dict[str, set[int]] = defaultdict(set)
    by_cik: dict[int, set[str]] = defaultdict(set)
    for ticker, cik in zip(
        sec_history.column("ticker").to_pylist(),
        sec_history.column("cik").to_pylist(),
        strict=True,
    ):
        by_ticker[ticker].add(cik)
        by_cik[cik].add(ticker)
    return Lookups(
        fund_symbols=frozenset(fund_symbols),
        ciks_by_ticker={t: frozenset(c) for t, c in by_ticker.items()},
        tickers_by_cik={c: frozenset(t) for c, t in by_cik.items()},
    )


def _mode(names: list[str]) -> str | None:
    return Counter(names).most_common(1)[0][0] if names else None


def segment_names(
    master: pa.Table, assets: pa.Table, sec_history: pa.Table
) -> list[tuple[str, str]]:
    """``(name, source)`` per master row, naming each segment after its own era.

    Alpaca's active row names the open segment. Its inactive row belongs to one segment only:
    the latest closed one when an active row exists and spells a different name, or the symbol's
    latest segment when there is no active row (a delisted symbol). An inactive row repeating the
    active name is a duplicate of today's company, not the old era (229 of 249 two-row symbols on
    2026-10-05: P, ECHO), and names nothing on its own.

    Open segment: active row > SEC snapshots since ``valid_from`` > inactive row > any SEC
    snapshot. Closed segment: its own inactive row > SEC snapshots inside the segment > a
    differing inactive row > active row > any SEC snapshot. SEC names take the most frequent
    spelling. Empty Alpaca names count as absent; the first row per (symbol, status) wins.
    """
    alpaca: dict[str, dict[str, str]] = defaultdict(dict)
    for symbol, name, status in zip(
        assets.column("symbol").to_pylist(),
        assets.column("name").to_pylist(),
        assets.column("status").to_pylist(),
        strict=True,
    ):
        if name and status not in alpaca[symbol]:
            alpaca[symbol][status] = name
    sec: dict[str, list[tuple[date, str]]] = defaultdict(list)
    for snapshot, ticker, name in zip(
        sec_history.column("snapshot_date").to_pylist(),
        sec_history.column("ticker").to_pylist(),
        sec_history.column("name").to_pylist(),
        strict=True,
    ):
        sec[ticker].append((snapshot, name))

    symbols = master.column("symbol").to_pylist()
    froms = master.column("valid_from").to_pylist()
    tos = master.column("valid_to").to_pylist()
    has_open: dict[str, bool] = defaultdict(bool)
    latest_closed: dict[str, date] = {}
    for symbol, valid_from, valid_to in zip(symbols, froms, tos, strict=True):
        if valid_to is None:
            has_open[symbol] = True
        elif symbol not in latest_closed or valid_from > latest_closed[symbol]:
            latest_closed[symbol] = valid_from

    out: list[tuple[str, str]] = []
    for symbol, valid_from, valid_to in zip(symbols, froms, tos, strict=True):
        rows = alpaca.get(symbol, {})
        active, inactive = rows.get("active"), rows.get("inactive")
        history = sec.get(symbol, [])
        inside = _mode(
            [n for d, n in history if d >= valid_from and (valid_to is None or d < valid_to)]
        )
        anywhere = _mode([n for _, n in history])
        if valid_to is None:
            order = [
                (active, "alpaca_active"),
                (inside, "sec"),
                (inactive, "alpaca_inactive"),
                (anywhere, "sec"),
            ]
        else:
            owns_inactive = (
                inactive is not None
                and inactive != active
                and latest_closed[symbol] == valid_from
                and (active is not None or not has_open[symbol])
            )
            order = [
                (inactive if owns_inactive else None, "alpaca_inactive"),
                (inside, "sec"),
                (inactive if inactive != active else None, "alpaca_inactive"),
                (active, "alpaca_active"),
                (anywhere, "sec"),
            ]
        for name, source in order:
            if name is not None:
                out.append((name, source))
                break
        else:
            out.append(("", "none"))
    return out


def build_instrument_class(
    master: pa.Table,
    assets: pa.Table,
    sec_history: pa.Table,
    fund_symbols: Iterable[str],
    rules: Sequence[Rule],
    overrides: Sequence[ClassOverride] = (),
) -> pa.Table:
    """One row per master segment, in master order; ``overrides`` are applied last."""
    lookups = build_lookups(sec_history, fund_symbols)
    names = segment_names(master, assets, sec_history)
    classes: list[str] = []
    hits: list[str] = []
    for (name, source), symbol, cik in zip(
        names, master.column("symbol").to_pylist(), master.column("cik").to_pylist(), strict=True
    ):
        cls, rule_id = classify(symbol, name, cik, lookups, rules, name_source=source)
        classes.append(cls)
        hits.append(rule_id)
    table = pa.table(
        {
            "security_id": master.column("security_id"),
            "symbol": master.column("symbol"),
            "valid_from": master.column("valid_from"),
            "valid_to": master.column("valid_to"),
            "name": pa.array([n for n, _ in names], pa.string()),
            "name_source": pa.array([s for _, s in names], pa.string()),
            "class": pa.array(classes, pa.string()),
            "rule": pa.array(hits, pa.string()),
        },
        schema=CLASS_SCHEMA,
    )
    return apply_overrides(table, overrides)


def build_from_store(
    market_root: Path,
    sec_dir: Path,
    rules: Sequence[Rule],
    overrides: Sequence[ClassOverride] = (),
) -> pa.Table:
    """Classify the security master on disk using the saved Alpaca and SEC tables.

    SEC snapshots before ``SEC_HISTORY_START`` are a frozen stale table (see
    ``security_master``) and are ignored here as well.
    """
    master = pq.read_table(market_root / MASTER_FILE)
    history = load_history(market_root)
    fresh = [d >= SEC_HISTORY_START for d in history.column("snapshot_date").to_pylist()]
    history = history.filter(pa.array(fresh, pa.bool_()))
    loaded = load_assets(market_root)
    assets = pa.table(
        {
            "symbol": pa.array([a.symbol for a in loaded], pa.string()),
            "name": pa.array([a.name for a in loaded], pa.string()),
            "status": pa.array([a.status for a in loaded], pa.string()),
        }
    )
    funds = load_sec_listings(sec_dir).funds
    return build_instrument_class(master, assets, history, funds, rules, overrides)


def write_instrument_class(table: pa.Table, market_root: Path) -> Path:
    path = market_root / CLASS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        pq.write_table(table, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path
