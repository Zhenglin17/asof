"""Guards against three ways the class table mis-tags a segment after step 6-b: a when-issued
symbol (HONAV, PRG.WI) tagged common from its issuer's SEC name or left unknown; an unknown
segment (SPLG, name empty) that a later segment of the same security (SPYM, an ETF) could have
named; and six shelf ETNs that qualified for the liquid tier while unknown (DWT, UWT, FBGX, FIHD,
FLGE, OIL) having no human decision on file.

Target changes (asof.ingest.instrument_class and the two config files):

    A. CLASSES gains "when_issued" (keep "unknown" last); the rule file gets a first rule
       `sym_when_issued` (class when_issued, symbol '.*\\.(WI|RTWI)$|[A-Z]{4}V') before
       `sym_warrant`; the same regex is `asof.ingest.security_master.WHEN_ISSUED`;
       `asof.store.universe.LIQUID_CLASSES` does not list it.
    I. INHERIT_RULE = "inherit_later": a post-pass in build_instrument_class, after classify
       and before apply_overrides. An `unknown` row takes the class of the nearest later
       segment (larger valid_from) of the same security_id whose class is not unknown; when the
       later known segments disagree it stays unknown/none; an earlier known class never
       propagates forward; no later segment -> unchanged.
    J. configs/instrument_class_overrides.yaml gains DWT, UWT, FBGX, FIHD, FLGE (leveraged_etf)
       and OIL (etf, valid_from 2020-01-02).

No ``as_of`` here: the class of a segment is a property of its name, its symbol and its
neighbours in the security master, not of when it was observed, so there is no leakage case in
this file. The read path filters segments on the security master's ``available_at``.

Fake data provenance -- "real" = read off the store on 2026-10-16 (step6b-counts.md sections
A, I, J); ASSUMPTION = a shape this file invents. Assumptions must be checked on real data by
audit M5 (unknown segments in the liquid tier) and the golden cases:

- [real] Nasdaq fifth-letter code V = when-issued / when-distributed; 73 symbols of the shape
  [A-Z]{4}V have bars, the longest-lived (BBIGV) traded 25 days, and no real company uses the
  form. CHECK on real data: after the rule lands, every [A-Z]{4}V and .WI/.RTWI segment is
  `when_issued`, and M5 drops the 46 when-issued rows it listed as unknown.
- [real] The SEC lists 22 of the V symbols for a snapshot or two (TYDEV "Cryptyde, Inc.",
  CMCSV "COMCAST CORP", CIK 1166691), so a symbol rule must run before any name rule; the tests
  feed those names and expect when_issued. CHECK: none of the 22 is `common` after the build.
- [ASSUMPTION] no liquid-tier symbol of the shape [A-Z]{4}V is a real company (a 5-letter
  common whose fifth letter happens to be V). CHECK: `asof market liquid` lists no
  when_issued symbol, and the audit's class histogram for the tier has 0 `when_issued`.
- [real] SPLG: unknown, name empty, segment 2020-01-02..2025-10-31, same security as SPYM
  (etf) -> inherits etf. CHECK (golden): SPLG's segment reads etf / inherit_later; the 7
  qualified ETF names of section I (SPLG, IRBO, CLTL, RYT, VPN, BTHM, HSPX) leave M5.
- [real] 12 unknown rows have only an EARLIER known class and must stay unknown; 1 chain has
  two different later classes and must stay unknown. CHECK: M5 still lists exactly those 13
  (none liquid) under rule `none`.
- [ASSUMPTION] a `.WI` segment chained into the same security as the regular name (AAN.WI ->
  AAN): section A says the master will stop chaining them, so this shape may vanish; the test
  only pins that rule A wins over the inheritance pass if it is still there.
- [ASSUMPTION] the inheritance pass reads the class the rule file produced, not the class an
  override later sets on the neighbour (the spec puts it before apply_overrides). CHECK: no
  real row depends on it today (the six J overrides have no unknown earlier neighbour).
- [real] the six J symbols: DWT / UWT (Citigroup 3x crude ETNs), FBGX / FIHD (UBS 2x ETNs),
  FLGE (Credit Suisse 2x) -> leveraged_etf; OIL (Barclays iPath crude ETN) -> etf, and the OIL
  label has a later segment of another security after 2020-07-29, so its override must carry
  valid_from 2020-01-02. CHECK: the build does not raise on the real master (an undated OIL
  entry spanning two securities is a loader error) and M5's `lookup_sec_shelf` liquid count
  goes from 6 to 0.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pytest

from asof.ingest import instrument_class as ic
from asof.ingest.instrument_class import (
    CLASSES,
    ClassOverride,
    Rule,
    build_instrument_class,
    classify,
    load_class_overrides,
    load_rules,
)
from tests.ingest.test_instrument_class import (
    EMPTY_LOOKUPS,
    RULES_PATH,
    assets_table,
    lookups_from,
    master_table,
    sec_table,
    write_overrides,
    write_rules,
)
from tests.ingest.test_instrument_class import rules as rules  # noqa: F401  (fixture)
from tests.ingest.test_instrument_class import world as world  # noqa: F401  (fixture)

OVERRIDES_PATH = RULES_PATH.with_name("instrument_class_overrides.yaml")
WHEN_ISSUED_REGEX = r".*\.(WI|RTWI)$|[A-Z]{4}V"
OLD_CLASSES = (
    "common",
    "etf",
    "leveraged_etf",
    "warrant",
    "unit",
    "preferred",
    "right",
    "note",
    "crypto",
)
D = date


# =============================================================================================
# A. when_issued: the class, the rule, and the shared regex
# =============================================================================================


def test_classes_gain_when_issued_and_keep_unknown_last() -> None:
    assert "when_issued" in CLASSES
    assert CLASSES[-1] == "unknown"
    assert set(CLASSES) == set(OLD_CLASSES) | {"when_issued", "unknown"}
    assert len(CLASSES) == len(set(CLASSES))


def test_real_rule_file_starts_with_sym_when_issued(rules: list[Rule]) -> None:
    assert rules[0].id == "sym_when_issued"
    assert rules[0].cls == "when_issued"
    assert rules[0].symbol == WHEN_ISSUED_REGEX
    assert (rules[0].name, rules[0].lookup) == (None, None)
    assert rules[1].id == "sym_warrant"
    assert [r.id for r in rules].count("sym_when_issued") == 1


def test_when_issued_regex_is_shared_with_the_security_master() -> None:
    # the master skips when-issued old names when chaining renames (section A); one spelling
    from asof.ingest import security_master

    pattern = getattr(security_master, "WHEN_ISSUED", None)
    assert pattern is not None, "security_master.WHEN_ISSUED is missing"
    by_id = {r.id: r for r in load_rules(RULES_PATH)}
    assert by_id["sym_when_issued"].symbol == pattern.pattern


def test_when_issued_is_not_a_liquid_class() -> None:
    from asof.store.universe import LIQUID_CLASSES

    assert "when_issued" not in LIQUID_CLASSES
    assert set(LIQUID_CLASSES) <= set(CLASSES)


@pytest.mark.parametrize(
    ("symbol", "name", "cik"),
    [
        ("HONAV", "", None),  # real: Honeywell Aerospace when-issued, no name anywhere
        ("PRG.WI", "PRG CO", None),  # real: PROG Holdings when-issued
        ("RIV.RTWI", "", None),  # real: rights when-issued
        ("CMCSV", "COMCAST CORP", 1166691),  # real: SEC-listed for a snapshot, issuer name
        ("TYDEV", "Cryptyde, Inc.", None),  # real: SEC-listed, issuer name says INC
        ("BBIGV", "", None),  # real: the longest-lived V symbol, 25 traded days
        ("ABCDV", "Abcd Corp. Warrants to Purchase Common Stock", None),  # ASSUMPTION
    ],
    ids=["HONAV", "PRG.WI", "RIV.RTWI", "CMCSV", "TYDEV", "BBIGV", "ABCDV-named-warrant"],
)
def test_classify_when_issued_wins_before_every_name_rule(
    rules: list[Rule], symbol: str, name: str, cik: int | None
) -> None:
    assert classify(symbol, name, cik, EMPTY_LOOKUPS, rules) == ("when_issued", "sym_when_issued")


def test_classify_when_issued_ignores_the_sec_name_and_the_cik_lookups(rules: list[Rule]) -> None:
    # real: CMCSV sits in the SEC history under Comcast's CIK next to CMCSA; the suffix lookup
    # and the company lookup both know the symbol, and neither may decide
    lookups = lookups_from([("CMCSA", 1166691), ("CMCSV", 1166691)])
    assert classify("CMCSV", "COMCAST CORP", 1166691, lookups, rules, name_source="sec") == (
        "when_issued",
        "sym_when_issued",
    )
    assert classify("CMCSV", "", 1166691, lookups, rules, name_source="none") == (
        "when_issued",
        "sym_when_issued",
    )


@pytest.mark.parametrize(
    ("symbol", "name", "expected"),
    [
        ("HONA", "Honeywell Aerospace Common Stock", ("common", "name_company_not_fund")),
        ("GOOGL", "Alphabet Inc. Class A Common Stock", ("common", "name_company_not_fund")),
        ("BRK.B", "Berkshire Hathaway Inc. Class B", ("common", "name_common")),
        ("GME.WS", "GameStop Corp. Warrants to Purchase Common Stock", ("warrant", "sym_warrant")),
        ("RMG.U", "RMG Acquisition Corp.", ("unit", "sym_unit")),
        ("ZGYH.RT", "", ("right", "sym_right")),
        ("BAC.PRB", "", ("preferred", "sym_preferred")),
    ],
    ids=["HONA", "GOOGL", "BRK.B", "GME.WS", "RMG.U", "ZGYH.RT", "BAC.PRB"],
)
def test_classify_other_symbols_are_unaffected(
    rules: list[Rule], symbol: str, name: str, expected: tuple[str, str]
) -> None:
    assert classify(symbol, name, None, EMPTY_LOOKUPS, rules) == expected


@pytest.mark.parametrize(
    "symbol",
    [
        "ABCV",  # four letters: the V is the fourth, not the Nasdaq fifth-letter code
        "ABCDEV",  # six letters
        "honav",  # lower case is not an Alpaca spelling
        "ABCDVW",  # V in fifth place but a sixth letter follows
        "ABC.WIX",  # suffix must end the symbol
        "ABCWI",  # no dot: 5 letters ending in I
        "ABC.V",  # a one-letter dotted suffix is a share class
        "AB1DV",  # digit inside
    ],
)
def test_when_issued_rule_is_a_fullmatch_on_the_exact_shape(rules: list[Rule], symbol: str) -> None:
    cls, rule_id = classify(symbol, "", None, EMPTY_LOOKUPS, rules)
    assert rule_id != "sym_when_issued", (symbol, cls, rule_id)
    assert cls != "when_issued"


def test_load_rules_accepts_when_issued_as_a_class(tmp_path: Path) -> None:
    path = write_rules(tmp_path, "- id: w\n  class: when_issued\n  symbol: '[A-Z]{4}V'\n")
    assert load_rules(path) == [Rule(id="w", cls="when_issued", symbol="[A-Z]{4}V")]


def test_load_class_overrides_accepts_when_issued_as_a_class(tmp_path: Path) -> None:
    text = "- symbol: HONAV\n  class: when_issued\n  reason: ASSUMPTION\n"
    assert load_class_overrides(write_overrides(tmp_path, text)) == [
        ClassOverride("HONAV", "when_issued", "ASSUMPTION")
    ]


def test_build_tags_when_issued_segments_from_the_symbol_alone(rules: list[Rule]) -> None:
    # real shapes: HONAV has no row anywhere; CMCSV is SEC-named for one snapshot (issuer name)
    master = master_table(
        [
            ("honav", "HONAV", D(2025, 7, 25), D(2025, 7, 31), None),
            ("cmcsv", "CMCSV", D(2025, 1, 2), D(2025, 1, 10), 1166691),
            ("prg-wi", "PRG.WI", D(2020, 11, 20), D(2020, 12, 1), None),
        ]
    )
    sec = sec_table(
        [
            (D(2025, 1, 3), "CMCSV", 1166691, "COMCAST CORP"),
            (D(2025, 1, 3), "CMCSA", 1166691, "COMCAST CORP"),
        ]
    )
    out = build_instrument_class(master, assets_table([]), sec, [], rules)
    assert out.column("class").to_pylist() == ["when_issued"] * 3
    assert out.column("rule").to_pylist() == ["sym_when_issued"] * 3
    assert out.column("name").to_pylist() == ["", "COMCAST CORP", ""]


# =============================================================================================
# I. inherit_later: an unknown segment named by a later segment of the same security
# =============================================================================================


SPLG_SPYM = [
    # real: SPLG unknown (no name anywhere) 2020-01-02..2025-10-31, then SPYM the ETF; one
    # security. The later row is listed first so the pass cannot rely on master row order.
    ("spdr-500", "SPYM", D(2025, 10, 31), None, None),
    ("spdr-500", "SPLG", D(2020, 1, 2), D(2025, 10, 31), None),
]
SPYM_ALPACA = assets_table([("SPYM", "SPDR Portfolio S&P 500 ETF", "active")])


def build_plain(
    master_rows: list[Any], rules: list[Rule], **kwargs: Any
) -> dict[tuple[str, date], dict[str, Any]]:
    out = build_instrument_class(
        master_table(master_rows), SPYM_ALPACA, sec_table([]), [], rules, **kwargs
    )
    return {(r["symbol"], r["valid_from"]): r for r in out.to_pylist()}


def test_inherit_rule_constant() -> None:
    assert ic.INHERIT_RULE == "inherit_later"
    assert ic.INHERIT_RULE != ic.OVERRIDE_RULE
    assert ic.INHERIT_RULE not in {r.id for r in load_rules(RULES_PATH)}


def test_inherit_later_splg_takes_etf_from_spym(rules: list[Rule]) -> None:
    rows = build_plain(SPLG_SPYM, rules)
    splg, spym = rows[("SPLG", D(2020, 1, 2))], rows[("SPYM", D(2025, 10, 31))]
    assert (spym["class"], spym["rule"]) == ("etf", "name_etf")
    assert (splg["class"], splg["rule"]) == ("etf", ic.INHERIT_RULE)
    # the pass changes class and rule only; the name columns still say nobody named SPLG
    assert (splg["name"], splg["name_source"]) == ("", "none")
    assert (splg["security_id"], splg["valid_to"]) == ("spdr-500", D(2025, 10, 31))


def test_inherit_later_keeps_master_row_order_and_schema(rules: list[Rule]) -> None:
    out = build_instrument_class(master_table(SPLG_SPYM), SPYM_ALPACA, sec_table([]), [], rules)
    assert out.schema.equals(ic.CLASS_SCHEMA)
    assert out.column("symbol").to_pylist() == ["SPYM", "SPLG"]
    assert out.column("class").to_pylist() == ["etf", "etf"]
    assert out.column("rule").to_pylist() == ["name_etf", ic.INHERIT_RULE]


def test_inherit_later_earlier_known_class_never_propagates_forward(rules: list[Rule]) -> None:
    # real: 12 unknown rows have only an earlier known class and must stay unknown
    rows = build_plain(
        [
            ("x", "SPYM", D(2016, 1, 4), D(2020, 1, 2), None),  # etf, earlier
            ("x", "SPLG", D(2020, 1, 2), None, None),  # unknown, later
        ],
        rules,
    )
    assert (rows[("SPYM", D(2016, 1, 4))]["class"]) == "etf"
    assert (rows[("SPLG", D(2020, 1, 2))]["class"], rows[("SPLG", D(2020, 1, 2))]["rule"]) == (
        "unknown",
        "none",
    )


def test_inherit_later_disagreeing_later_classes_stay_unknown(rules: list[Rule]) -> None:
    # real: one chain has two different later classes. ASSUMPTION on the shape: the nearest
    # later is an ETF, the one after it a common stock; disagreement wins over nearness.
    rows = build_plain(
        [
            ("x", "SPLG", D(2020, 1, 2), D(2021, 1, 4), None),  # unknown
            ("x", "SPYM", D(2021, 1, 4), D(2022, 1, 3), None),  # etf
            ("x", "AAPL", D(2022, 1, 3), None, 320193),  # common via sec_companies
        ],
        rules,
    )
    assert rows[("SPYM", D(2021, 1, 4))]["class"] == "etf"
    assert rows[("AAPL", D(2022, 1, 3))]["class"] == "common"
    assert (rows[("SPLG", D(2020, 1, 2))]["class"], rows[("SPLG", D(2020, 1, 2))]["rule"]) == (
        "unknown",
        "none",
    )


def test_inherit_later_agreeing_later_classes_inherit(rules: list[Rule]) -> None:
    # two later segments, both ETFs through different names: no disagreement, so inherit
    rows = {
        (r["symbol"], r["valid_from"]): r
        for r in build_instrument_class(
            master_table(
                [
                    ("x", "SPLG", D(2020, 1, 2), D(2021, 1, 4), None),
                    ("x", "SPYM", D(2021, 1, 4), D(2022, 1, 3), None),
                    ("x", "IEI", D(2022, 1, 3), None, None),
                ]
            ),
            assets_table(
                [
                    ("SPYM", "SPDR Portfolio S&P 500 ETF", "active"),
                    ("IEI", "iShares 3-7 Year Treasury Bond ETF", "active"),
                ]
            ),
            sec_table([]),
            [],
            rules,
        ).to_pylist()
    }
    assert (rows[("SPLG", D(2020, 1, 2))]["class"], rows[("SPLG", D(2020, 1, 2))]["rule"]) == (
        "etf",
        ic.INHERIT_RULE,
    )


def test_inherit_later_skips_unknown_neighbours_and_fills_every_unknown_before(
    rules: list[Rule],
) -> None:
    # two unknown segments in a row, then the ETF: both inherit, the nearer one through the
    # farther one's position (the nearest KNOWN later segment is what counts)
    rows = build_plain(
        [
            ("x", "SPLA", D(2018, 1, 2), D(2019, 1, 2), None),
            ("x", "SPLG", D(2019, 1, 2), D(2025, 10, 31), None),
            ("x", "SPYM", D(2025, 10, 31), None, None),
        ],
        rules,
    )
    for key in (("SPLA", D(2018, 1, 2)), ("SPLG", D(2019, 1, 2))):
        assert (rows[key]["class"], rows[key]["rule"]) == ("etf", ic.INHERIT_RULE), key


def test_inherit_later_no_later_segment_is_unchanged(rules: list[Rule]) -> None:
    rows = build_plain([("x", "SPLG", D(2020, 1, 2), None, None)], rules)
    assert (rows[("SPLG", D(2020, 1, 2))]["class"], rows[("SPLG", D(2020, 1, 2))]["rule"]) == (
        "unknown",
        "none",
    )
    # the only later row is unknown as well
    rows = build_plain(
        [
            ("x", "SPLG", D(2020, 1, 2), D(2021, 1, 4), None),
            ("x", "SPLH", D(2021, 1, 4), None, None),
        ],
        rules,
    )
    assert [rows[k]["rule"] for k in sorted(rows)] == ["none", "none"]


def test_inherit_later_needs_the_same_security_not_the_same_symbol(rules: list[Rule]) -> None:
    # a reused ticker: the later holder of the label is another security and names nothing.
    # The new holder is classified through its CIK (no name anywhere, so the old closed
    # segment cannot borrow an Alpaca or SEC name either); the old era stays unknown.
    out = build_instrument_class(
        master_table(
            [
                ("old", "NEWC", D(2016, 1, 4), D(2018, 1, 2), None),  # unknown: nothing known
                ("new", "NEWC", D(2025, 10, 31), None, 77),  # common via sec_companies
            ]
        ),
        assets_table([]),
        sec_table([]),
        [],
        rules,
    )
    assert out.column("class").to_pylist() == ["unknown", "common"]
    assert out.column("rule").to_pylist() == ["none", "lookup_sec_companies"]


def test_inherit_later_does_not_touch_known_rows(world: dict[str, Any], rules: list[Rule]) -> None:
    out = build_instrument_class(
        world["master"], world["assets"], world["sec"], world["funds"], rules
    )
    rules_hit = out.column("rule").to_pylist()
    assert ic.INHERIT_RULE not in rules_hit  # no unknown row in the world has a later sibling
    rows = {r["security_id"]: r for r in out.to_pylist()}
    assert (rows["gld-noname"]["class"], rows["gld-noname"]["rule"]) == ("unknown", "none")
    assert (rows["btc-1"]["class"], rows["btc-1"]["rule"]) == ("etf", "name_etf")


def test_when_issued_segment_is_classified_by_rule_a_not_inherited(rules: list[Rule]) -> None:
    # ASSUMPTION: AAN.WI chained into AAN's security (the master may stop doing this); if it
    # is, rule A has already named it and the pass must leave it alone
    rows = {
        (r["symbol"], r["valid_from"]): r
        for r in build_instrument_class(
            master_table(
                [
                    ("aan", "AAN.WI", D(2020, 11, 20), D(2020, 12, 1), None),
                    ("aan", "AAN", D(2020, 12, 1), None, None),
                ]
            ),
            assets_table([("AAN", "The Aaron's Company, Inc. Common Stock", "active")]),
            sec_table([]),
            [],
            rules,
        ).to_pylist()
    }
    assert (rows[("AAN", D(2020, 12, 1))]["class"]) == "common"
    assert (
        rows[("AAN.WI", D(2020, 11, 20))]["class"],
        rows[("AAN.WI", D(2020, 11, 20))]["rule"],
    ) == (
        "when_issued",
        "sym_when_issued",
    )


def test_override_on_an_inherited_row_still_wins(rules: list[Rule]) -> None:
    rows = build_plain(
        SPLG_SPYM,
        rules,
        overrides=[
            ClassOverride("SPLG", "leveraged_etf", "ASSUMPTION: a human disagrees", D(2020, 1, 2))
        ],
    )
    assert (rows[("SPLG", D(2020, 1, 2))]["class"], rows[("SPLG", D(2020, 1, 2))]["rule"]) == (
        "leveraged_etf",
        ic.OVERRIDE_RULE,
    )
    assert (rows[("SPYM", D(2025, 10, 31))]["class"], rows[("SPYM", D(2025, 10, 31))]["rule"]) == (
        "etf",
        "name_etf",
    )


def test_inherit_later_reads_the_rule_class_not_the_override_on_the_neighbour(
    rules: list[Rule],
) -> None:
    # the pass runs before overrides: SPLG inherits what the rules said about SPYM (etf),
    # even when a human then relabels SPYM itself
    rows = build_plain(
        SPLG_SPYM,
        rules,
        overrides=[ClassOverride("SPYM", "leveraged_etf", "ASSUMPTION", D(2025, 10, 31))],
    )
    assert (rows[("SPYM", D(2025, 10, 31))]["class"], rows[("SPYM", D(2025, 10, 31))]["rule"]) == (
        "leveraged_etf",
        ic.OVERRIDE_RULE,
    )
    assert (rows[("SPLG", D(2020, 1, 2))]["class"], rows[("SPLG", D(2020, 1, 2))]["rule"]) == (
        "etf",
        ic.INHERIT_RULE,
    )


def test_inherit_later_on_an_empty_master(rules: list[Rule]) -> None:
    out = build_instrument_class(master_table([]), assets_table([]), sec_table([]), [], rules)
    assert out.num_rows == 0


# =============================================================================================
# J. six dated overrides for the shelf ETNs that qualified for the liquid tier
# =============================================================================================


SHELF_ETNS = {
    "DWT": "leveraged_etf",  # real: Citigroup VelocityShares 3x Inverse Crude ETN
    "UWT": "leveraged_etf",  # real: Citigroup VelocityShares 3x Long Crude ETN
    "FBGX": "leveraged_etf",  # real: UBS ETRACS 2x Leveraged Long FANG+ ETN
    "FIHD": "leveraged_etf",  # real: UBS ETRACS 2x Leveraged High Dividend ETN
    "FLGE": "leveraged_etf",  # real: Credit Suisse FI Large Cap Growth Enhanced ETN (2x)
    "OIL": "etf",  # real: Barclays iPath Series B S&P GSCI Crude Oil ETN
}


@pytest.fixture(scope="module")
def real_overrides() -> list[ClassOverride]:
    return load_class_overrides(OVERRIDES_PATH)


def test_real_overrides_file_lists_the_six_shelf_etns(real_overrides: list[ClassOverride]) -> None:
    by_symbol: dict[str, list[ClassOverride]] = {}
    for o in real_overrides:
        by_symbol.setdefault(o.symbol, []).append(o)
    for symbol, cls in SHELF_ETNS.items():
        assert symbol in by_symbol, symbol
        assert {o.cls for o in by_symbol[symbol]} == {cls}, symbol


def test_real_overrides_oil_is_dated_to_the_etn_segment(
    real_overrides: list[ClassOverride],
) -> None:
    # real: the OIL label has a later segment of another security after 2020-07-29; an
    # undated entry would span two securities and the loader refuses it at build time
    oil = [o for o in real_overrides if o.symbol == "OIL"]
    assert len(oil) == 1
    assert oil[0].valid_from == D(2020, 1, 2)
    assert oil[0].cls == "etf"


def test_real_overrides_file_passes_the_loader_checks(real_overrides: list[ClassOverride]) -> None:
    for o in real_overrides:
        assert o.cls in CLASSES and o.cls != "unknown", o.symbol
        assert o.reason.strip(), o.symbol
        assert o.valid_from is None or isinstance(o.valid_from, date), o.symbol
    keys = [(o.symbol, o.valid_from) for o in real_overrides]
    assert len(keys) == len(set(keys))
    # an undated entry and a dated one for the same symbol would make file order decide
    undated = {o.symbol for o in real_overrides if o.valid_from is None}
    dated = {o.symbol for o in real_overrides if o.valid_from is not None}
    assert not (undated & dated)


def test_real_overrides_file_keeps_the_earlier_decisions(
    real_overrides: list[ClassOverride],
) -> None:
    symbols = {o.symbol for o in real_overrides}
    assert {"BAM", "NYMT", "TVIX", "UGAZ", "DGAZ", "FNGA", "FNGB", "XXXX"} <= symbols
    assert len([o for o in real_overrides if o.symbol == "BAM"]) == 3


def test_build_applies_a_dated_oil_override_to_the_etn_segment_only(rules: list[Rule]) -> None:
    # real shape: OIL the Barclays ETN (SEC name "BARCLAYS BANK PLC", shelf CIK 312070) until
    # 2020-07-29, then another security under the same label. ASSUMPTION: the later holder is
    # named by an Alpaca active row (the name is OILK's, borrowed); the closed ETN segment keeps
    # the SEC name because an active row only names the open segment when SEC rows exist inside
    cik = 312070
    master = master_table(
        [
            ("oil-etn", "OIL", D(2020, 1, 2), D(2020, 7, 29), cik),
            ("oil-2", "OIL", D(2020, 7, 30), None, None),
        ]
    )
    assets = assets_table([("OIL", "ProShares K-1 Free Crude Oil ETF", "active")])
    shelf = sec_table(
        [
            (D(2020, 2, 1), f"B{i:04d}", cik, "BARCLAYS BANK PLC")
            for i in range(ic.SHELF_MIN_TICKERS - 1)
        ]
        + [(D(2020, 2, 1), "OIL", cik, "BARCLAYS BANK PLC")]
    )
    plain = build_instrument_class(master, assets, shelf, [], rules)
    assert plain.column("name_source").to_pylist() == ["sec", "alpaca_active"]
    assert plain.column("class").to_pylist() == ["unknown", "etf"]
    assert plain.column("rule").to_pylist() == ["lookup_sec_shelf", "name_etf"]
    with pytest.raises(ValueError, match=r"OIL.*valid_from"):
        build_instrument_class(
            master, assets, shelf, [], rules, overrides=[ClassOverride("OIL", "etf", "iPath")]
        )
    out = build_instrument_class(
        master,
        assets,
        shelf,
        [],
        rules,
        overrides=[ClassOverride("OIL", "etf", "iPath crude ETN", D(2020, 1, 2))],
    )
    assert out.column("class").to_pylist() == ["etf", "etf"]
    assert out.column("rule").to_pylist() == [ic.OVERRIDE_RULE, "name_etf"]
