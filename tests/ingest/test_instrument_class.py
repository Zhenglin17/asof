"""Guards against a segment of the security master wearing the wrong instrument class: an ADR
tagged preferred, a Treasury-bond ETF tagged note, a buffer ETF tagged leveraged, a SPAC unit
tagged common, a warrant known only by its issuer's SEC name tagged common, and a segment
named after the wrong era of a reused ticker.

Target API (asof.ingest.instrument_class, new module):

    CLASSES, LOOKUPS, NASDAQ_SUFFIX, CLASS_FILE, CLASS_SCHEMA
    Rule(id, cls, symbol=None, name=None, lookup=None)        frozen dataclass
    load_rules(path) -> list[Rule]
    Lookups(fund_symbols, ciks_by_ticker, tickers_by_cik)     frozen dataclass
    classify(symbol, name, cik, lookups, rules) -> (class, rule_id)
    segment_names(master, assets, sec_history) -> [(name, name_source), ...]  (master row order)
    build_instrument_class(master, assets, sec_history, fund_symbols, rules) -> pa.Table
    write_instrument_class(table, market_root) -> Path

No ``as_of`` here: the class of a segment is a property of its name and symbol, not of when it
was observed, so there is no leakage case in this file. The read path filters segments on the
security master's ``available_at``; this table only decorates them.

Fake data provenance -- every name below was read off the 2026-10-05 Alpaca asset table or the
SEC ticker history on 2026-10-08 ("real") unless marked ASSUMPTION. Assumptions must be checked
on real data by audit A6 (count of ``unknown`` segments, expected 3: RALS, RFUN, ASBH; and
``unknown`` = 0 inside the liquid tier):

- Rule outcomes for AAPL, SKHY, BAC.PRB, IEI, MGR, TQQQ, UJUL, RFIX, PSQ, SOXL, SPY, GLD, DHC,
  GGN, USDP, FTIVU, GME.WS, RMG.U, AMH, ADILW, EXEEW: real names, real classes.
- ZGYHR "Yunhong International Right": the trailing "Right" is enough for ``name_right``; ZGYH
  and ZGYHR share CIK 1773086 in the SEC history (checked 2026-10-08).
- A 5-letter W symbol whose issuer lists neither a >= 3-letter prefix nor a 5-letter sibling on
  the same stem: real example NXNVW (NextNav, common ticker NN); falls to ``name_common`` through
  "INC" and is accepted as a known miss.
- Issuer shelves: Credit Suisse AG / UBS AG / Barclays Bank PLC list 28-78 ETN tickers each in the
  SEC history; the SEC-only names of TVIX, DGAZ, UGAZ are the bank's name (real).
- BTC two segments: closed 2021-06-03..2024-04-01 named by the SEC ("Grayscale Bitcoin Trust
  (BTC)"), open segment named by Alpaca's active row ("Grayscale Bitcoin Mini Trust ETF"): real.
- P and ECHO: one active and one inactive Alpaca row with the same name: real (249 symbols have
  two rows; 229 active+inactive, 20 both inactive).
- One ticker first a SPAC unit, later a common stock after reuse: ASSUMPTION; audit A6 reports
  symbols whose segments carry different classes.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from asof.ingest import instrument_class as ic
from asof.ingest.instrument_class import (
    CLASS_FILE,
    CLASS_SCHEMA,
    CLASSES,
    LOOKUPS,
    NASDAQ_SUFFIX,
    Lookups,
    Rule,
    build_instrument_class,
    classify,
    load_rules,
    segment_names,
    write_instrument_class,
)
from asof.ingest.security_master import MASTER_SCHEMA

RULES_PATH = Path(__file__).resolve().parents[2] / "configs" / "instrument_class_rules.yaml"
RULE_IDS_IN_FILE = [
    "sym_warrant",
    "sym_unit",
    "sym_right",
    "sym_preferred",
    "name_warrant",
    "name_unit",
    "name_right",
    "name_preferred",
    "name_note",
    "name_leveraged",
    "name_company_not_fund",
    "name_etf",
    "lookup_sec_funds",
    "lookup_nasdaq_suffix",
    "lookup_sec_shelf",
    "name_common",
    "lookup_sec_companies",
]
CLASS_COLUMNS = [
    "security_id",
    "symbol",
    "valid_from",
    "valid_to",
    "name",
    "name_source",
    "class",
    "rule",
]
TS_UTC = pa.timestamp("us", tz="UTC")
AVAILABLE = datetime(2020, 1, 1, tzinfo=UTC)  # any instant; this module never reads it


# =============================================================================================
# builders
# =============================================================================================


Seg = tuple[str, str, date, date | None, int | None]  # security_id, symbol, from, to, cik


def master_table(rows: Sequence[Seg]) -> pa.Table:
    return pa.table(
        {
            "security_id": pa.array([r[0] for r in rows], pa.string()),
            "symbol": pa.array([r[1] for r in rows], pa.string()),
            "valid_from": pa.array([r[2] for r in rows], pa.date32()),
            "valid_to": pa.array([r[3] for r in rows], pa.date32()),
            "cik": pa.array([r[4] for r in rows], pa.int64()),
            "cusip": pa.array([None for _ in rows], pa.string()),
            "evidence": pa.array(["bars" for _ in rows], pa.string()),
            "available_at": pa.array([AVAILABLE for _ in rows], TS_UTC),
        },
        schema=MASTER_SCHEMA,
    )


Asset = tuple[str, str, str]  # symbol, name, status


def assets_table(rows: Sequence[Asset]) -> pa.Table:
    return pa.table(
        {
            "symbol": pa.array([r[0] for r in rows], pa.string()),
            "name": pa.array([r[1] for r in rows], pa.string()),
            "status": pa.array([r[2] for r in rows], pa.string()),
        }
    )


Snap = tuple[date, str, int, str]  # snapshot_date, ticker, cik, name


def sec_table(rows: Sequence[Snap]) -> pa.Table:
    rows = sorted(rows)
    return pa.table(
        {
            "snapshot_date": pa.array([r[0] for r in rows], pa.date32()),
            "ticker": pa.array([r[1] for r in rows], pa.string()),
            "cik": pa.array([r[2] for r in rows], pa.int64()),
            "name": pa.array([r[3] for r in rows], pa.string()),
        }
    )


def lookups_from(pairs: Sequence[tuple[str, int]], funds: Sequence[str] = ()) -> Lookups:
    """(ticker, cik) pairs as they would come out of the SEC history."""
    by_ticker: dict[str, set[int]] = {}
    by_cik: dict[int, set[str]] = {}
    for ticker, cik in pairs:
        by_ticker.setdefault(ticker, set()).add(cik)
        by_cik.setdefault(cik, set()).add(ticker)
    return Lookups(
        fund_symbols=frozenset(funds),
        ciks_by_ticker={t: frozenset(c) for t, c in by_ticker.items()},
        tickers_by_cik={c: frozenset(t) for c, t in by_cik.items()},
    )


EMPTY_LOOKUPS = lookups_from([])


def write_rules(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "rules.yaml"
    path.write_text(text)
    return path


# =============================================================================================
# fixtures
# =============================================================================================


@pytest.fixture(scope="module")
def rules() -> list[Rule]:
    return load_rules(RULES_PATH)


@pytest.fixture
def real_lookups() -> Lookups:
    # SEC history pairs (real CIKs for ADIL/ADILW 1513525, SPY 884394; others as noted).
    return lookups_from(
        [
            ("AAPL", 320193),
            ("SPY", 884394),  # real: SPY sits in the SEC company table as an ETF trust
            ("AMH", 1562401),
            ("ADIL", 1513525),
            ("ADILW", 1513525),
            ("EXE", 895126),
            ("EXEEW", 895126),
            ("ZGYH", 1773086),  # real: Yunhong's common shares share the CIK (SEC 2020-11..2021-06)
            ("ZGYHR", 1773086),
            ("LBRDA", 1611983),  # real: Liberty Broadband share classes A/K and preferred P
            ("LBRDK", 1611983),
            ("LBRDP", 1611983),
            ("QQ", 4242),  # ASSUMPTION: a 2-letter prefix must not count
            ("QQQQW", 4242),
        ],
        funds=["TQQQ", "GGN", "IEI", "SPCX"],  # real: TQQQ, and a stale SPCX fund row (4a)
    )


# =============================================================================================
# constants and the rule file
# =============================================================================================


def test_constants_cover_every_class_and_lookup_named_in_the_rule_file() -> None:
    assert CLASSES == (
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
    assert LOOKUPS == ("sec_funds", "nasdaq_suffix", "sec_shelf", "sec_companies")
    assert NASDAQ_SUFFIX == {"W": "warrant", "R": "right", "U": "unit", "P": "preferred"}
    assert set(NASDAQ_SUFFIX.values()) <= set(CLASSES)
    assert CLASS_FILE == "symbols/instrument_class.parquet"
    assert CLASS_SCHEMA.names == CLASS_COLUMNS
    assert CLASS_SCHEMA.field("valid_from").type == pa.date32()
    assert CLASS_SCHEMA.field("valid_to").type == pa.date32()
    assert all(
        CLASS_SCHEMA.field(c).type == pa.string()
        for c in ("security_id", "symbol", "name", "name_source", "class", "rule")
    )


def test_load_real_rules_keeps_file_order(rules: list[Rule]) -> None:
    assert [r.id for r in rules] == RULE_IDS_IN_FILE
    for r in rules:
        assert r.cls in CLASSES, r.id
        assert sum(x is not None for x in (r.symbol, r.name, r.lookup)) == 1, r.id
    by_id = {r.id: r for r in rules}
    assert by_id["sym_preferred"].symbol == r".*\.PR[A-Z]?$"
    assert by_id["name_warrant"].name == r"\bwarrants?\b"
    assert by_id["lookup_sec_funds"].lookup == "sec_funds"
    assert by_id["lookup_sec_funds"].cls == "etf"
    assert by_id["lookup_nasdaq_suffix"].lookup == "nasdaq_suffix"
    assert by_id["lookup_sec_companies"].lookup == "sec_companies"
    assert by_id["lookup_sec_companies"].cls == "common"
    # the order the comments in the file promise: leveraged before etf, both before the fund
    # lookup, the suffix lookup before common, common before the company lookup
    pos = {r.id: i for i, r in enumerate(rules)}
    assert pos["name_leveraged"] < pos["name_company_not_fund"] < pos["name_etf"]
    assert pos["name_etf"] < pos["lookup_sec_funds"]
    assert pos["lookup_nasdaq_suffix"] < pos["lookup_sec_shelf"] < pos["name_common"]
    assert pos["name_common"] < pos["lookup_sec_companies"]


def test_rule_is_frozen_and_takes_class_as_cls() -> None:
    rule = Rule(id="x", cls="etf", name="etf")
    assert (rule.symbol, rule.lookup) == (None, None)
    with pytest.raises((AttributeError, TypeError)):
        rule.cls = "common"  # type: ignore[misc]


def test_load_rules_small_file(tmp_path: Path) -> None:
    path = write_rules(
        tmp_path,
        "- id: a\n  class: warrant\n  symbol: '.*\\.WS$'\n"
        "- id: b\n  class: etf\n  lookup: sec_funds\n"
        "- id: c\n  class: common\n  name: 'inc'\n",
    )
    loaded = load_rules(path)
    assert loaded == [
        Rule(id="a", cls="warrant", symbol=r".*\.WS$"),
        Rule(id="b", cls="etf", lookup="sec_funds"),
        Rule(id="c", cls="common", name="inc"),
    ]


def test_load_rules_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_rules(tmp_path / "absent.yaml")


@pytest.mark.parametrize(
    ("label", "text"),
    [
        ("empty list", "[]\n"),
        ("empty file", ""),
        ("not a list", "id: a\nclass: common\nname: inc\n"),
        ("missing id", "- class: common\n  name: inc\n"),
        ("missing class", "- id: a\n  name: inc\n"),
        ("no matcher", "- id: a\n  class: common\n"),
        ("two matchers", "- id: a\n  class: common\n  name: inc\n  symbol: 'A.*'\n"),
        (
            "three matchers",
            "- id: a\n  class: common\n  name: inc\n  symbol: 'A'\n  lookup: sec_funds\n",
        ),
        ("unknown class", "- id: a\n  class: bond\n  name: inc\n"),
        ("unknown lookup", "- id: a\n  class: common\n  lookup: sec_people\n"),
        (
            "duplicate id",
            "- id: a\n  class: common\n  name: inc\n- id: a\n  class: etf\n  name: etf\n",
        ),
        ("bad symbol regex", "- id: a\n  class: common\n  symbol: '(unclosed'\n"),
        ("bad name regex", "- id: a\n  class: common\n  name: '[unclosed'\n"),
    ],
)
def test_load_rules_rejects_bad_files(tmp_path: Path, label: str, text: str) -> None:
    path = write_rules(tmp_path, text)
    with pytest.raises(ValueError):
        load_rules(path)


# =============================================================================================
# classify: real names through the real rule file
# =============================================================================================


REAL_CASES: list[tuple[str, str, int | None, str, str]] = [
    # symbol, Alpaca/SEC name, cik on the master row, expected class, expected rule
    ("AAPL", "Apple Inc. Common Stock", 320193, "common", "name_company_not_fund"),
    ("SKHY", "SK hynix Inc. American Depositary Shares", None, "common", "name_common"),
    (
        "BAC.PRB",
        "Bank of America Corporation Depositary Shares, each representing a 1/1,000th interest "
        "in a share of 6.000% Non-Cumulative Preferred Stock, Series GG",
        None,
        "preferred",
        "sym_preferred",
    ),
    ("IEI", "iShares 3-7 Year Treasury Bond ETF", None, "etf", "name_etf"),
    (
        "MGR",
        "Affiliated Managers Group, Inc. 5.875% Junior Subordinated Notes due 2059",
        None,
        "note",
        "name_note",
    ),
    ("TQQQ", "ProShares UltraPro QQQ", None, "leveraged_etf", "name_leveraged"),
    ("UJUL", "Innovator U.S. Equity Ultra Buffer ETF - July", None, "etf", "name_etf"),
    ("RFIX", "Simplify Bond Bull ETF", None, "etf", "name_etf"),
    ("PSQ", "ProShares Short QQQ", None, "leveraged_etf", "name_leveraged"),
    ("SOXL", "Direxion Daily Semiconductor Bull 3X ETF", None, "leveraged_etf", "name_leveraged"),
    ("SPY", "State Street SPDR S&P 500 ETF Trust", 884394, "etf", "name_etf"),
    ("GLD", "SPDR Gold Trust, SPDR Gold Shares", None, "etf", "name_etf"),
    (
        "DHC",
        "Diversified Healthcare Trust Common Shares of Beneficial Interest",
        None,
        "common",
        "name_common",
    ),
    # closed-end fund: the issuer list inside name_etf ("gamco ... trust"), not sec_funds
    ("GGN", "GAMCO Global Gold, Natural Resources & Income Trust", None, "etf", "name_etf"),
    ("USDP", "USD PARTNERS LP COM UNIT REPSTG LTD PARTNER INTS", None, "common", "name_common"),
    ("FTIVU", "FinTech Acquisition Corp. IV Unit", None, "unit", "name_unit"),
    ("GME.WS", "GameStop Corp. Warrants to Purchase Common Stock", None, "warrant", "sym_warrant"),
    ("RMG.U", "RMG Acquisition Corp.", None, "unit", "sym_unit"),
    ("AMH", "AMERICAN HOMES 4 RENT", 1562401, "common", "lookup_sec_companies"),
    # SEC-only names: the issuer's name, class from the Nasdaq suffix
    ("ADILW", "ADIAL PHARMACEUTICALS, INC.", 1513525, "warrant", "lookup_nasdaq_suffix"),
    ("EXEEW", "EXPAND ENERGY Corp", 895126, "warrant", "lookup_nasdaq_suffix"),
    # real (SEC history 2020-11..2021-06): ZGYH and ZGYHR share CIK 1773086; the name alone
    # ends in "Right" and the name rule fires first
    ("ZGYHR", "Yunhong International Right", 1773086, "right", "name_right"),
    # reviewer counter-examples (2026-10-08), all real names
    ("PFBC", "Preferred Bank Common Stock", None, "common", "name_company_not_fund"),
    ("APTS", "PREFERRED APARTMENT COMMUNITIES INC", 1481832, "common", "name_common"),
    ("WRNT", "WARRANTEE INC.", 1868941, "common", "name_common"),
    ("FBRT", "Franklin BSP Realty Trust, Inc.", None, "common", "name_company_not_fund"),
    ("BXSL", "Blackstone Secured Lending Fund", None, "common", "name_company_not_fund"),
    (
        "SMCIP",
        "Super Micro Computer, Inc. Depositary Shares representing Preferred Stock",
        None,
        "preferred",
        "name_preferred",
    ),
    (
        "NFEGP",
        "New Fortress Energy Inc. Series A Mandatorily Convertible Preferred Stock",
        None,
        "preferred",
        "name_preferred",
    ),
    ("GRZZP", "Grizzly Energy LLC Preferred Series A", None, "preferred", "name_preferred"),
    ("BANXR", "ArrowMark Financial Corp. Right", None, "right", "name_right"),
    ("LBRDP", "Liberty Broadband Corp", 1611983, "preferred", "lookup_nasdaq_suffix"),
    # the three real unknowns (2026-10-08 build)
    ("RALS", "ProShares RAFI Long/Short", None, "unknown", "none"),
    ("RFUN", "RiverFront Dynamic Unconstrained Income", None, "unknown", "none"),
    ("ASBH", "American Savings Bank, N.A.", None, "unknown", "none"),
]


@pytest.mark.parametrize(
    ("symbol", "name", "cik", "expected_class", "expected_rule"),
    REAL_CASES,
    ids=[c[0] for c in REAL_CASES],
)
def test_classify_real_examples(
    rules: list[Rule],
    real_lookups: Lookups,
    symbol: str,
    name: str,
    cik: int | None,
    expected_class: str,
    expected_rule: str,
) -> None:
    assert classify(symbol, name, cik, real_lookups, rules) == (expected_class, expected_rule)


def test_classify_first_match_wins_leveraged_before_fund_lookup(
    rules: list[Rule], real_lookups: Lookups
) -> None:
    assert "TQQQ" in real_lookups.fund_symbols
    assert classify("TQQQ", "ProShares UltraPro QQQ", None, real_lookups, rules) == (
        "leveraged_etf",
        "name_leveraged",
    )
    # a fund-table symbol with no usable name falls through to the lookup
    assert classify("TQQQ", "", None, real_lookups, rules) == ("etf", "lookup_sec_funds")


def test_classify_common_stock_in_name_beats_fund_table(
    rules: list[Rule], real_lookups: Lookups
) -> None:
    # real: SPCX sits on a stale SEC fund-table row (found in 4a), but Alpaca names it
    # "Space Exploration Technologies Corp. Class A Common Stock"
    assert "SPCX" in real_lookups.fund_symbols
    name = "Space Exploration Technologies Corp. Class A Common Stock"
    assert classify("SPCX", name, None, real_lookups, rules) == (
        "common",
        "name_company_not_fund",
    )
    assert classify("SPCX", "", None, real_lookups, rules) == ("etf", "lookup_sec_funds")


def test_classify_sec_company_table_does_not_mean_common(
    rules: list[Rule], real_lookups: Lookups
) -> None:
    assert "SPY" in real_lookups.ciks_by_ticker
    assert classify("SPY", "SPDR S&P 500 ETF TRUST", 884394, real_lookups, rules) == (
        "etf",
        "name_etf",
    )


def test_classify_empty_name_skips_name_rules_only(
    rules: list[Rule], real_lookups: Lookups
) -> None:
    # symbol rule still fires
    assert classify("GME.WS", "", None, real_lookups, rules) == ("warrant", "sym_warrant")
    assert classify("BAC.PRB", "", None, real_lookups, rules) == ("preferred", "sym_preferred")
    # lookups still fire
    assert classify("ADILW", "", 1513525, real_lookups, rules) == (
        "warrant",
        "lookup_nasdaq_suffix",
    )
    assert classify("AAPL", "", 320193, real_lookups, rules) == (
        "common",
        "lookup_sec_companies",
    )
    # nothing else to go on -> unknown, never a name rule
    assert classify("AAPL", "", None, EMPTY_LOOKUPS, rules) == ("unknown", "none")


def test_classify_no_match_is_unknown_none(rules: list[Rule]) -> None:
    assert classify("XXXX", "Something Without Keywords", None, EMPTY_LOOKUPS, rules) == (
        "unknown",
        "none",
    )
    assert classify("XXXX", "", None, EMPTY_LOOKUPS, rules) == ("unknown", "none")


def test_classify_name_is_lower_cased_before_search(rules: list[Rule]) -> None:
    assert classify("ABC", "ACME WARRANTS", None, EMPTY_LOOKUPS, rules) == (
        "warrant",
        "name_warrant",
    )
    assert classify("ABC", "ADIAL PHARMACEUTICALS, INC.", None, EMPTY_LOOKUPS, rules) == (
        "common",
        "name_common",
    )


def test_classify_symbol_rule_is_fullmatch_and_name_rule_is_search() -> None:
    custom = [Rule(id="s", cls="unit", symbol="A"), Rule(id="n", cls="common", name="inc")]
    assert classify("AB", "", None, EMPTY_LOOKUPS, custom) == ("unknown", "none")
    assert classify("A", "", None, EMPTY_LOOKUPS, custom) == ("unit", "s")
    assert classify("ZZ", "Zinc Ltd", None, EMPTY_LOOKUPS, custom) == ("common", "n")


def test_classify_returns_in_rule_order_not_class_order() -> None:
    custom = [Rule(id="late", cls="etf", name="trust"), Rule(id="early", cls="common", name="inc")]
    assert classify("X", "Inc Trust", None, EMPTY_LOOKUPS, custom) == ("etf", "late")
    assert classify("X", "Inc Trust", None, EMPTY_LOOKUPS, list(reversed(custom))) == (
        "common",
        "early",
    )


# -- nasdaq_suffix --------------------------------------------------------------------------------


SUFFIX_ONLY = [Rule(id="suf", cls="unknown", lookup="nasdaq_suffix")]


@pytest.mark.parametrize(
    ("symbol", "expected"),
    [("ADILW", "warrant"), ("ADILR", "right"), ("ADILU", "unit"), ("ADILP", "preferred")],
)
def test_nasdaq_suffix_class_comes_from_last_letter_not_from_rule(
    symbol: str, expected: str
) -> None:
    lookups = lookups_from([("ADIL", 1513525), (symbol, 1513525)])
    assert classify(symbol, "", None, lookups, SUFFIX_ONLY) == (expected, "suf")
    # rule class is ignored even when it is a legal class
    other = [Rule(id="suf", cls="common", lookup="nasdaq_suffix")]
    assert classify(symbol, "", None, lookups, other) == (expected, "suf")


def test_nasdaq_suffix_prefix_of_three_letters_is_enough() -> None:
    lookups = lookups_from([("EXE", 895126), ("EXEEW", 895126)])
    assert classify("EXEEW", "", None, lookups, SUFFIX_ONLY) == ("warrant", "suf")


def test_nasdaq_suffix_prefix_shorter_than_three_does_not_count(rules: list[Rule]) -> None:
    lookups = lookups_from([("QQ", 4242), ("QQQQW", 4242)])
    assert classify("QQQQW", "", None, lookups, SUFFIX_ONLY) == ("unknown", "none")
    # with the real rules the SEC issuer name decides instead: ASSUMPTION (no real example)
    assert classify("QQQQW", "QQ HOLDINGS, INC.", 4242, lookups, rules) == (
        "common",
        "name_common",
    )


def test_nasdaq_suffix_needs_a_sibling_under_the_same_cik() -> None:
    # the symbol itself is the only ticker on its CIK
    assert classify("ADILW", "", None, lookups_from([("ADILW", 1)]), SUFFIX_ONLY) == (
        "unknown",
        "none",
    )
    # a prefix ticker exists but under another CIK
    lookups = lookups_from([("ADIL", 1), ("ADILW", 2)])
    assert classify("ADILW", "", None, lookups, SUFFIX_ONLY) == ("unknown", "none")
    # the symbol is not in the SEC history at all, even if the master row has a CIK
    lookups = lookups_from([("ADIL", 1)])
    assert classify("ADILW", "", 1, lookups, SUFFIX_ONLY) == ("unknown", "none")


def test_nasdaq_suffix_sibling_must_be_a_prefix_or_share_the_four_letter_stem() -> None:
    lookups = lookups_from([("ADIAL", 1), ("ADILW", 1)])
    assert classify("ADILW", "", None, lookups, SUFFIX_ONLY) == ("unknown", "none")
    # real: LBRDA / LBRDK are share classes on the stem LBRD; LBRDP is the preferred
    lookups = lookups_from([("LBRDA", 2), ("LBRDP", 2)])
    assert classify("LBRDP", "", None, lookups, SUFFIX_ONLY) == ("preferred", "suf")
    # a 5-letter sibling on another stem is not a share class of this one
    lookups = lookups_from([("LBRAA", 3), ("LBRDP", 3)])
    assert classify("LBRDP", "", None, lookups, SUFFIX_ONLY) == ("unknown", "none")


# -- sec_shelf ------------------------------------------------------------------------------------


SHELF_ONLY = [Rule(id="shelf", cls="unknown", lookup="sec_shelf")]


def shelf_lookups(n_tickers: int, cik: int = 1053092) -> Lookups:
    # real: Credit Suisse AG (CIK 1053092) lists 28 ETN tickers in the SEC history; TVIX, DGAZ
    # and UGAZ have no Alpaca row, so their segment name is "CREDIT SUISSE AG"
    return lookups_from([(f"T{i:04d}", cik) for i in range(n_tickers - 1)] + [("TVIX", cik)])


def test_sec_shelf_leaves_issuer_named_etns_unknown(rules: list[Rule]) -> None:
    lookups = shelf_lookups(ic.SHELF_MIN_TICKERS)
    assert classify("TVIX", "CREDIT SUISSE AG", 1053092, lookups, rules, name_source="sec") == (
        "unknown",
        "lookup_sec_shelf",
    )
    assert classify("TVIX", "", 1053092, lookups, rules, name_source="none") == (
        "unknown",
        "lookup_sec_shelf",
    )


def test_sec_shelf_ignores_alpaca_named_segments(rules: list[Rule]) -> None:
    lookups = shelf_lookups(ic.SHELF_MIN_TICKERS)
    # real: BNKU on the Bank of Montreal shelf has an Alpaca name that says what it is
    name = "MicroSectors U.S. Big Banks Index 3X Leveraged ETNs due March 25, 2039"
    assert classify("TVIX", name, 1053092, lookups, rules, name_source="alpaca_active") == (
        "leveraged_etf",
        "name_leveraged",
    )
    assert classify("TVIX", "CREDIT SUISSE AG", 1053092, lookups, rules) == (
        "common",
        "name_common",
    )


def test_sec_shelf_threshold_is_inclusive_and_counts_tickers_per_cik() -> None:
    assert classify("TVIX", "", None, shelf_lookups(ic.SHELF_MIN_TICKERS), SHELF_ONLY, "sec") == (
        "unknown",
        "shelf",
    )
    assert classify(
        "TVIX", "", None, shelf_lookups(ic.SHELF_MIN_TICKERS - 1), SHELF_ONLY, "sec"
    ) == ("unknown", "none")
    # the symbol must itself be on that CIK
    assert classify("ELSE", "", None, shelf_lookups(ic.SHELF_MIN_TICKERS), SHELF_ONLY, "sec") == (
        "unknown",
        "none",
    )


@pytest.mark.parametrize("symbol", ["ADIL", "ADILWW", "ADI.W", "ADILX", "adilw", "ADL1W"])
def test_nasdaq_suffix_shape_is_exactly_four_letters_plus_wrup(symbol: str) -> None:
    lookups = lookups_from([("ADI", 1), ("ADIL", 1), ("ADL", 1), (symbol, 1)])
    assert classify(symbol, "", None, lookups, SUFFIX_ONLY) == ("unknown", "none")


# -- sec_funds / sec_companies --------------------------------------------------------------------


def test_sec_funds_lookup_is_exact_symbol_membership() -> None:
    only = [Rule(id="f", cls="etf", lookup="sec_funds")]
    lookups = lookups_from([], funds=["IEI"])
    assert classify("IEI", "", None, lookups, only) == ("etf", "f")
    assert classify("IEIX", "", None, lookups, only) == ("unknown", "none")
    assert classify("iei", "", None, lookups, only) == ("unknown", "none")


def test_sec_companies_lookup_fires_on_cik_or_on_history_membership() -> None:
    only = [Rule(id="c", cls="common", lookup="sec_companies")]
    lookups = lookups_from([("AMH", 1562401)])
    assert classify("AMH", "", None, lookups, only) == ("common", "c")  # in history, no cik
    assert classify("NEWC", "", 77, lookups, only) == ("common", "c")  # cik, not in history
    assert classify("NEWC", "", None, lookups, only) == ("unknown", "none")


def test_sec_companies_rule_class_is_honoured() -> None:
    only = [Rule(id="c", cls="etf", lookup="sec_companies")]
    assert classify("AMH", "", 1562401, EMPTY_LOOKUPS, only) == ("etf", "c")


# =============================================================================================
# segment_names
# =============================================================================================


D = date
BTC_SEC = "Grayscale Bitcoin Trust (BTC)"
BTC_ALPACA = "Grayscale Bitcoin Mini Trust ETF"


def test_segment_names_btc_closed_segment_takes_sec_open_segment_takes_alpaca_active() -> None:
    master = master_table(
        [
            ("btc-1", "BTC", D(2021, 6, 3), D(2024, 4, 1), 1588489),
            ("btc-2", "BTC", D(2024, 4, 1), None, None),
        ]
    )
    assets = assets_table([("BTC", BTC_ALPACA, "active")])  # real: no inactive row
    sec = sec_table(
        [
            (D(2021, 7, 1), "BTC", 1588489, BTC_SEC),
            (D(2022, 7, 1), "BTC", 1588489, BTC_SEC),
            (D(2023, 7, 1), "BTC", 1588489, BTC_SEC),
            (D(2024, 2, 1), "BTC", 1588489, BTC_SEC),
        ]
    )
    assert segment_names(master, assets, sec) == [
        (BTC_SEC, "sec"),
        (BTC_ALPACA, "alpaca_active"),
    ]


def test_segment_names_open_segment_priority() -> None:
    """active > in-range SEC mode > inactive > any SEC mode > none."""
    master = master_table([("x", "P", D(2024, 1, 1), None, 5)])
    in_range = [(D(2024, 6, 1), "P", 5, "P IN RANGE")]
    before = [(D(2020, 6, 1), "P", 5, "P OLD")]

    both = assets_table([("P", "Everpure, Inc.", "active"), ("P", "Everpure, Inc.", "inactive")])
    assert segment_names(master, both, sec_table(in_range + before)) == [
        ("Everpure, Inc.", "alpaca_active")
    ]

    inactive_only = assets_table([("P", "Everpure, Inc.", "inactive")])
    assert segment_names(master, inactive_only, sec_table(in_range + before)) == [
        ("P IN RANGE", "sec")
    ]
    assert segment_names(master, inactive_only, sec_table(before)) == [
        ("Everpure, Inc.", "alpaca_inactive")
    ]

    none = assets_table([])
    assert segment_names(master, none, sec_table(before)) == [("P OLD", "sec")]
    assert segment_names(master, none, sec_table([])) == [("", "none")]


def test_segment_names_closed_segment_priority() -> None:
    """inactive > in-segment SEC mode > active > any SEC mode > none."""
    master = master_table([("x", "ECHO", D(2020, 1, 1), D(2022, 1, 1), 7)])
    inside = [(D(2021, 1, 1), "ECHO", 7, "ECHO INSIDE")]
    after = [(D(2023, 1, 1), "ECHO", 7, "ECHO AFTER")]

    both = assets_table([("ECHO", "Echo Old Co", "inactive"), ("ECHO", "Echo New Co", "active")])
    assert segment_names(master, both, sec_table(inside + after)) == [
        ("Echo Old Co", "alpaca_inactive")
    ]

    active_only = assets_table([("ECHO", "Echo New Co", "active")])
    assert segment_names(master, active_only, sec_table(inside + after)) == [("ECHO INSIDE", "sec")]
    assert segment_names(master, active_only, sec_table(after)) == [
        ("Echo New Co", "alpaca_active")
    ]

    none = assets_table([])
    assert segment_names(master, none, sec_table(after)) == [("ECHO AFTER", "sec")]
    assert segment_names(master, none, sec_table([])) == [("", "none")]


def test_segment_names_closed_segment_distrusts_inactive_row_repeating_active_name() -> None:
    """real: 229 of 249 two-row symbols have active and inactive rows with the same name (P,
    ECHO); the old era's name is then only in the SEC snapshots (BEAT 2020-2021 was BioTelemetry,
    not Heartbeam)."""
    master = master_table([("x", "BEAT", D(2020, 1, 2), D(2021, 10, 1), 1574774)])
    new = "Heartbeam, Inc. Common Stock"
    assets = assets_table([("BEAT", new, "active"), ("BEAT", new, "inactive")])
    inside = sec_table([(D(2020, 6, 1), "BEAT", 1574774, "BIOTELEMETRY, INC.")])
    assert segment_names(master, assets, inside) == [("BIOTELEMETRY, INC.", "sec")]
    # no SEC name inside the segment: the duplicated Alpaca name is still better than nothing
    assert segment_names(master, assets, sec_table([])) == [(new, "alpaca_active")]
    # an inactive row with a different name is the old era and still wins
    old = assets_table([("BEAT", new, "active"), ("BEAT", "BioTelemetry, Inc.", "inactive")])
    assert segment_names(master, old, inside) == [("BioTelemetry, Inc.", "alpaca_inactive")]


def test_segment_names_sec_range_bounds() -> None:
    """valid_from is inclusive, valid_to exclusive; an open segment has no upper bound."""
    closed = master_table([("x", "Z", D(2021, 1, 1), D(2022, 1, 1), 9)])
    on_from = sec_table([(D(2021, 1, 1), "Z", 9, "ON FROM")])
    on_to = sec_table([(D(2022, 1, 1), "Z", 9, "ON TO")])
    assert segment_names(closed, assets_table([]), on_from) == [("ON FROM", "sec")]
    # the snapshot on valid_to is outside the segment, but still the only name anywhere
    assert segment_names(closed, assets_table([("Z", "Z Active", "active")]), on_to) == [
        ("Z Active", "alpaca_active")
    ]
    assert segment_names(closed, assets_table([]), on_to) == [("ON TO", "sec")]

    opened = master_table([("x", "Z", D(2021, 1, 1), None, 9)])
    far = sec_table([(D(2030, 1, 1), "Z", 9, "FAR FUTURE")])
    assert segment_names(opened, assets_table([("Z", "Z Inactive", "inactive")]), far) == [
        ("FAR FUTURE", "sec")
    ]


def test_segment_names_sec_mode_takes_the_name_seen_most_often() -> None:
    master = master_table([("x", "M", D(2020, 1, 1), None, 3)])
    sec = sec_table(
        [
            (D(2020, 2, 1), "M", 3, "MESA INC"),
            (D(2020, 3, 1), "M", 3, "MESA INC"),
            (D(2020, 4, 1), "M", 3, "MESA, INC."),
        ]
    )
    assert segment_names(master, assets_table([]), sec) == [("MESA INC", "sec")]
    # the SEC name is read for the segment's ticker, not for other tickers of the same CIK
    sec_other = sec_table(
        [
            (D(2020, 2, 1), "M", 3, "MESA INC"),
            (D(2020, 3, 1), "MM", 3, "MESA MM"),
            (D(2020, 4, 1), "MM", 3, "MESA MM"),
        ]
    )
    assert segment_names(master, assets_table([]), sec_other) == [("MESA INC", "sec")]


def test_segment_names_empty_alpaca_name_is_absent() -> None:
    master = master_table([("x", "E", D(2020, 1, 1), None, None)])
    assets = assets_table([("E", "", "active"), ("E", "E Old Name", "inactive")])
    assert segment_names(master, assets, sec_table([])) == [("E Old Name", "alpaca_inactive")]
    assert segment_names(master, assets_table([("E", "", "active")]), sec_table([])) == [
        ("", "none")
    ]


def test_segment_names_duplicate_alpaca_rows_take_the_first() -> None:
    master = master_table([("x", "DUP", D(2020, 1, 1), None, None)])
    assets = assets_table([("DUP", "First Active", "active"), ("DUP", "Second Active", "active")])
    assert segment_names(master, assets, sec_table([])) == [("First Active", "alpaca_active")]
    # both rows inactive (real: 20 such symbols) -> first inactive row names a closed segment
    closed = master_table([("x", "DUP", D(2020, 1, 1), D(2021, 1, 1), None)])
    inactive = assets_table([("DUP", "First Inactive", "inactive"), ("DUP", "Second", "inactive")])
    assert segment_names(closed, inactive, sec_table([])) == [("First Inactive", "alpaca_inactive")]


def test_segment_names_follow_master_row_order_and_ignore_unrelated_symbols() -> None:
    master = master_table(
        [
            ("b", "BBB", D(2020, 1, 1), None, None),
            ("a1", "AAA", D(2019, 1, 1), D(2020, 1, 1), None),
            ("c", "CCC", D(2020, 1, 1), None, None),
            ("a2", "AAA", D(2020, 1, 1), None, None),
        ]
    )
    assets = assets_table(
        [
            ("ZZZ", "Unrelated", "active"),
            ("AAA", "AAA New", "active"),
            ("AAA", "AAA Old", "inactive"),
            ("BBB", "BBB Co", "active"),
        ]
    )
    assert segment_names(master, assets, sec_table([])) == [
        ("BBB Co", "alpaca_active"),
        ("AAA Old", "alpaca_inactive"),
        ("", "none"),
        ("AAA New", "alpaca_active"),
    ]


def test_segment_names_empty_master() -> None:
    assert (
        segment_names(master_table([]), assets_table([("A", "A", "active")]), sec_table([])) == []
    )


# =============================================================================================
# build_instrument_class
# =============================================================================================


@pytest.fixture
def world() -> dict[str, Any]:
    """A master with every naming path and every rule family, in a deliberately mixed order."""
    master = master_table(
        [
            ("aapl", "AAPL", D(2016, 1, 4), None, 320193),
            ("btc-1", "BTC", D(2021, 6, 3), D(2024, 4, 1), 1588489),
            ("adilw", "ADILW", D(2019, 10, 2), None, 1513525),  # SEC-only name
            ("pace-1", "PACE", D(2019, 1, 2), D(2021, 6, 30), 11),  # ASSUMPTION: SPAC unit era
            ("tqqq", "TQQQ", D(2016, 1, 4), None, None),
            ("btc-2", "BTC", D(2024, 4, 1), None, None),
            ("pace-2", "PACE", D(2023, 3, 1), None, 12),  # ASSUMPTION: reused as a common stock
            ("amh", "AMH", D(2016, 1, 4), None, 1562401),  # no name anywhere -> sec_companies
            ("gld-noname", "NONAME", D(2016, 1, 4), None, None),  # nothing -> unknown
            ("fundonly", "FUNDONLY", D(2016, 1, 4), None, None),  # only in the fund table
        ]
    )
    assets = assets_table(
        [
            ("AAPL", "Apple Inc. Common Stock", "active"),
            ("BTC", BTC_ALPACA, "active"),
            ("PACE", "Pace Acquisition Corp. Unit", "inactive"),
            ("PACE", "Pace Industries Inc. Common Stock", "active"),
            ("TQQQ", "ProShares UltraPro QQQ", "active"),
        ]
    )
    sec = sec_table(
        [
            (D(2021, 7, 1), "BTC", 1588489, BTC_SEC),
            (D(2023, 7, 1), "BTC", 1588489, BTC_SEC),
            (D(2020, 1, 1), "ADIL", 1513525, "ADIAL PHARMACEUTICALS, INC."),
            (D(2020, 1, 1), "ADILW", 1513525, "ADIAL PHARMACEUTICALS, INC."),
            (D(2020, 1, 1), "AAPL", 320193, "Apple Inc."),
        ]
    )
    return {"master": master, "assets": assets, "sec": sec, "funds": ["TQQQ", "FUNDONLY"]}


def build(world: dict[str, Any], rules: list[Rule]) -> pa.Table:
    return build_instrument_class(
        world["master"], world["assets"], world["sec"], world["funds"], rules
    )


def test_build_schema_rows_and_order(world: dict[str, Any], rules: list[Rule]) -> None:
    out = build(world, rules)
    assert out.schema.equals(CLASS_SCHEMA)
    assert out.num_rows == world["master"].num_rows
    for col in ("security_id", "symbol", "valid_from", "valid_to"):
        assert out.column(col).to_pylist() == world["master"].column(col).to_pylist(), col
    assert all(c in CLASSES for c in out.column("class").to_pylist())


def test_build_classes_and_rules(world: dict[str, Any], rules: list[Rule]) -> None:
    out = build(world, rules)
    rows = {r["security_id"]: r for r in out.to_pylist()}
    assert (rows["aapl"]["class"], rows["aapl"]["rule"]) == ("common", "name_company_not_fund")
    assert (rows["btc-1"]["class"], rows["btc-2"]["class"]) == ("etf", "etf")
    assert (rows["adilw"]["class"], rows["adilw"]["rule"]) == ("warrant", "lookup_nasdaq_suffix")
    assert (rows["tqqq"]["class"], rows["tqqq"]["rule"]) == ("leveraged_etf", "name_leveraged")
    assert (rows["amh"]["class"], rows["amh"]["rule"]) == ("common", "lookup_sec_companies")
    assert (rows["fundonly"]["class"], rows["fundonly"]["rule"]) == ("etf", "lookup_sec_funds")
    assert (rows["gld-noname"]["class"], rows["gld-noname"]["rule"]) == ("unknown", "none")


def test_build_names_and_sources(world: dict[str, Any], rules: list[Rule]) -> None:
    out = build(world, rules)
    rows = {r["security_id"]: r for r in out.to_pylist()}
    assert (rows["btc-1"]["name"], rows["btc-1"]["name_source"]) == (BTC_SEC, "sec")
    assert (rows["btc-2"]["name"], rows["btc-2"]["name_source"]) == (BTC_ALPACA, "alpaca_active")
    assert rows["adilw"]["name_source"] == "sec"
    assert (rows["amh"]["name"], rows["amh"]["name_source"]) == ("", "none")
    assert rows["gld-noname"]["name_source"] == "none"


def test_build_same_symbol_two_segments_can_differ_in_class(
    world: dict[str, Any], rules: list[Rule]
) -> None:
    out = build(world, rules)
    rows = {r["security_id"]: r for r in out.to_pylist()}
    assert rows["pace-1"]["name_source"] == "alpaca_inactive"
    assert (rows["pace-1"]["class"], rows["pace-1"]["rule"]) == ("unit", "name_unit")
    assert rows["pace-2"]["name_source"] == "alpaca_active"
    assert (rows["pace-2"]["class"], rows["pace-2"]["rule"]) == ("common", "name_company_not_fund")


def test_build_uses_master_cik_for_sec_companies(rules: list[Rule]) -> None:
    master = master_table(
        [("with", "NEWC", D(2020, 1, 1), None, 77), ("without", "NEWD", D(2020, 1, 1), None, None)]
    )
    out = build_instrument_class(master, assets_table([]), sec_table([]), [], rules)
    assert out.column("class").to_pylist() == ["common", "unknown"]
    assert out.column("rule").to_pylist() == ["lookup_sec_companies", "none"]


def test_build_lookups_come_from_sec_history(rules: list[Rule]) -> None:
    master = master_table([("w", "ADILW", D(2020, 1, 1), None, None)])
    with_sibling = sec_table(
        [
            (D(2020, 1, 1), "ADIL", 1513525, "ADIAL PHARMACEUTICALS, INC."),
            (D(2020, 1, 1), "ADILW", 1513525, "ADIAL PHARMACEUTICALS, INC."),
        ]
    )
    without = sec_table([(D(2020, 1, 1), "ADILW", 1513525, "ADIAL PHARMACEUTICALS, INC.")])
    assert build_instrument_class(master, assets_table([]), with_sibling, [], rules).column(
        "class"
    ).to_pylist() == ["warrant"]
    # no sibling: the issuer name says "INC" -> common (the failure mode the suffix rule exists for)
    assert build_instrument_class(master, assets_table([]), without, [], rules).column(
        "class"
    ).to_pylist() == ["common"]


def test_build_empty_master(rules: list[Rule]) -> None:
    out = build_instrument_class(master_table([]), assets_table([]), sec_table([]), [], rules)
    assert out.schema.equals(CLASS_SCHEMA)
    assert out.num_rows == 0


def test_build_accepts_any_iterable_of_fund_symbols(rules: list[Rule]) -> None:
    master = master_table([("f", "FUNDONLY", D(2020, 1, 1), None, None)])
    out = build_instrument_class(
        master, assets_table([]), sec_table([]), (s for s in ["FUNDONLY"]), rules
    )
    assert out.column("class").to_pylist() == ["etf"]


# =============================================================================================
# write_instrument_class
# =============================================================================================


def test_write_round_trip_and_path(
    world: dict[str, Any], rules: list[Rule], tmp_path: Path
) -> None:
    out = build(world, rules)
    root = tmp_path / "market"  # does not exist yet
    path = write_instrument_class(out, root)
    assert path == root / CLASS_FILE
    assert path.is_file()
    back = pq.read_table(path)
    assert back.schema.equals(CLASS_SCHEMA)
    assert back.to_pylist() == out.to_pylist()


def test_write_leaves_no_temp_file_and_overwrites(
    world: dict[str, Any], rules: list[Rule], tmp_path: Path
) -> None:
    out = build(world, rules)
    path = write_instrument_class(out, tmp_path)
    leftovers = [p for p in path.parent.iterdir() if p != path]
    assert leftovers == []
    smaller = out.slice(0, 2)
    assert write_instrument_class(smaller, tmp_path) == path
    assert pq.read_table(path).num_rows == 2


# =============================================================================================
# module hygiene
# =============================================================================================


def test_module_does_not_touch_the_network_or_the_store() -> None:
    src = Path(ic.__file__).read_text()
    for banned in ("requests", "httpx", "urllib", "alpaca"):
        assert not re.search(rf"^\s*(import|from)\s+{banned}\b", src, re.M), banned
