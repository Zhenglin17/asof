"""Guards against the store audit missing a company filer whose segment the class table calls a
fund (M9): an ``etf`` / ``leveraged_etf`` segment whose CIK files periodic company reports with
XBRL share counts. Such a segment enters the liquid tier as a fund although it is an operating
company, and a fund-only strategy would trade it. The check must also leave alone what a human
already decided (``rule == "override"``), the bank and issuer shelves that list exchange-traded
notes under one CIK, real funds without company filings and common stocks; and it must not use
a filing or a master segment that was not public at ``as_of``.

Target API (asof.store.audit; implementation pending):

    CHECKS["M9"] = CheckSpec("M9", "fund-class segment of a company filer", "liquid",
                             f"{SYM}:{DATE}")
        Registered right after M8 (the M checks stay together; F1 / F2 remain the last two).
        Rows: class-table segments visible in the master at ``as_of`` (matched to the master on
        security_id, symbol, valid_from), class in {etf, leveraged_etf}, master ``cik`` not
        null, and the CIK has a share fact (``asof.store.shares.share_facts``) with form in
        {10-K, 10-K/A, 10-Q, 10-Q/A, 20-F, 20-F/A, 40-F, 40-F/A} and ``available_at <= as_of``.
        Exempt: ``rule == "override"``; a CIK listing >= ``SHELF_MIN_TICKERS`` distinct tickers
        in the SEC ticker history (snapshots before ``SEC_HISTORY_START`` ignored, as in
        ``asof.ingest.instrument_class.build_from_store``). No shares file: no rows.
        Columns: key, in_liquid, excepted, security_id, symbol, valid_from, name, rule, cik.
        Key: ``SYMBOL:valid_from``. In scope when the security_id is in the liquid scope.

Fake data provenance (real cases read off the store on 2026-10-30, reported by the caller):
- FB "Facebook Inc", 2020-01-02 .. 2022-06-09, CIK 1326801, tagged etf by the SEC fund lookup
  because today's fund table gives the ticker FB to a ProShares fund (CIK 1174610). Facebook
  files 10-K / 10-Q with dei:EntityCommonStockSharesOutstanding.
- PRTO "RCN Pareto Strategic Allocation ETF" 2020-01-02 .. 2020-01-10 (name_etf), CIK 1359931
  (Protara / Proteon Therapeutics, a 10-K filer).
- BMO MicroSectors notes (CIK 927971) list many tickers under one CIK: an issuer shelf.
- TVIX (CIK 1159510, Credit Suisse, few tickers) is classed by a human override.
- A real fund (SPY-like) has no company share facts; at most an N-CSR-type form.
- Share facts become public at 00:00 New York of filed + 1 day (ingest.shares._public_from).

ASSUMPTIONS to check on real data (``asof market audit`` on the real store, then the cases):
- After the fund-lookup fix, M9 on the real store lists only the residues the caller named
  (PRTO 2020-01-02 among them) and every liquid row is fixed or excepted with a reason.
- FB 2020-01-02 is no longer classed etf after the fix; if it is, M9 lists FB:2020-01-02.
- BMO MicroSectors' CIK 927971 really lists >= SHELF_MIN_TICKERS distinct tickers in the
  fresh SEC history (otherwise its ETNs would appear as M9 rows).
- No real fund (an ETF trust filing N-CSR / 485BPOS) carries XBRL share facts under the forms
  above; a fund that does would show up as a false M9 row.
"""

from collections.abc import Iterator, Sequence
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import duckdb
import pytest

from asof.ingest.instrument_class import SHELF_MIN_TICKERS
from asof.ingest.security_master import SEC_HISTORY_START
from asof.ingest.shares import ShareFact
from asof.store.audit import CHECKS, CheckResult, run_audit
from asof.store.market import market_root
from tests.store.audit_world import (
    DAYS,
    LATE,
    SID_X1,
    SID_X2,
    Seg,
    World,
    class_row,
    default_sec,
    exc,
    iso,
    keys,
    rows,
)
from tests.store.test_market import KNOWN_EARLY, avail, known

NY = ZoneInfo("America/New_York")
DEI = "dei:EntityCommonStockSharesOutstanding"
HEAD = ["key", "in_liquid", "excepted"]
COMPANY_FORMS = ("10-K", "10-K/A", "10-Q", "10-Q/A", "20-F", "20-F/A", "40-F", "40-F/A")

FB_CIK = 1326801  # Facebook / Meta
PRTO_CIK = 1359931  # Protara (formerly Proteon) Therapeutics
SPY_CIK = 884394  # SPDR S&P 500 ETF Trust
BMO_CIK = 927971  # Bank of Montreal (MicroSectors ETNs)
CS_CIK = 1159510  # Credit Suisse (VelocityShares ETNs)
SID_FB, SID_PRTO, SID_SPY = SID_X1, SID_X2, "S000010"
SID_NOTE, SID_NOTE2, SID_COMMON = "S000011", "S000012", "S000013"

_n = iter(range(1, 10_000))


def fact(cik: int, filed: date, *, form: str = "10-K", shares: float = 2.85e9) -> ShareFact:
    """One share fact as the ingest stores it: public from 00:00 New York of filed + 1 day."""
    return ShareFact(
        cik=cik,
        concept=DEI,
        accn=f"{cik:010d}-20-{next(_n):06d}",
        form=form,
        end=filed - timedelta(days=7),
        filed=filed,
        shares=float(shares),
        available_at=datetime.combine(filed + timedelta(days=1), time(0), tzinfo=NY).astimezone(
            UTC
        ),
    )


FB_10K = fact(FB_CIK, date(2020, 1, 30))  # Facebook's FY2019 10-K, filed 2020-01-30


def fb_seg(**kw: Any) -> Seg:
    fields: dict[str, Any] = {
        "valid_from": DAYS[0],
        "valid_to": date(2022, 6, 9),
        "available_at": KNOWN_EARLY,
        "end_available_at": known(date(2022, 6, 9)),
        "cik": FB_CIK,
        "cls": "etf",
        "rule": "lookup_sec_funds",
        "name_source": "alpaca_active",
    }
    fields.update(kw)
    return Seg(SID_FB, "FB", **fields)


NAMES = {
    "FB": "Facebook Inc",
    "PRTO": "RCN Pareto Strategic Allocation ETF",
    "SPY": "SPDR S&P 500 ETF Trust",
    "FNGU": "MicroSectors FANG+ Index 3X Leveraged ETN",
    "TVIX": "VelocityShares Daily 2x VIX Short Term ETN",
    "ZIV": "VelocityShares Daily Inverse VIX Medium Term ETN",
}


def with_names(segs: Sequence[Seg]) -> list[dict[str, Any]]:
    """Class rows for ``segs`` with the real names (the world's default is "<symbol> Inc")."""
    out = []
    for s in segs:
        row = class_row(s)
        if s.symbol in NAMES:
            row["name"] = NAMES[s.symbol]
        out.append(row)
    return out


def shelf_tickers(own: str, n: int) -> list[str]:
    """``n`` distinct tickers under one CIK, ``own`` among them (BMO's FNGU, FNGD, ...)."""
    return [own] + [f"BM{chr(65 + i // 26)}{chr(65 + i % 26)}" for i in range(n - 1)]


@pytest.fixture
def con() -> Iterator[duckdb.DuckDBPyConnection]:
    connection = duckdb.connect()
    yield connection
    connection.close()


@pytest.fixture
def world(tmp_path: Path) -> World:
    return World(market_root(tmp_path))


def m9(
    con: duckdb.DuckDBPyConnection,
    world: World,
    *,
    as_of: datetime = LATE,
    exceptions: Sequence[Any] = (),
    qualified: set[str] | None = None,
) -> CheckResult:
    world.class_rows = with_names(world.segs)
    world.write()
    [result] = run_audit(
        con,
        world.root,
        as_of=as_of,
        liquid_ids=world.liquid_ids,
        qualified_ids=qualified,
        exceptions=exceptions,
        checks=["M9"],
    )
    return result


def add(world: World, *segs: Seg, liquid: bool = True) -> None:
    world.segs = [*world.segs, *segs]
    if liquid:
        world.liquid_ids |= {s.security_id for s in segs}


# =============================================================================================
# registration
# =============================================================================================


def test_m9_is_registered_after_m8_with_the_liquid_gate() -> None:
    spec = CHECKS["M9"]
    assert spec.id == "M9"
    assert spec.gate == "liquid"
    assert spec.title == "fund-class segment of a company filer"
    order = list(CHECKS)
    assert order.index("M9") == order.index("M8") + 1
    assert CHECKS["M9"].key_pattern == CHECKS["M5"].key_pattern


def test_shelf_threshold_is_the_instrument_class_one() -> None:
    assert SHELF_MIN_TICKERS == 20


# =============================================================================================
# normal path
# =============================================================================================


def test_m9_lists_facebook_tagged_etf_and_nothing_else(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    # FB carries an etf class from today's fund table; Facebook files 10-Ks. Next to it: a real
    # fund whose CIK only has an N-CSR-type fact, a fund with no facts at all, and a common
    # stock that files 10-Ks. Only FB is a row.
    add(
        world,
        fb_seg(),
        Seg(SID_SPY, "SPY", cik=SPY_CIK, cls="etf", rule="lookup_sec_funds"),
        Seg(SID_COMMON, "AAPL", cik=320193, cls="common", rule="lookup_sec_companies"),
    )
    world.shares = [
        FB_10K,
        fact(SPY_CIK, date(2019, 11, 26), form="N-CSR"),
        fact(320193, date(2019, 10, 30)),
        fact(1001, date(2019, 11, 1)),  # AAA (common, base world)
    ]
    result = m9(con, world)

    assert result.rows.schema.names == [
        *HEAD,
        "security_id",
        "symbol",
        "valid_from",
        "name",
        "rule",
        "cik",
    ]
    assert rows(result) == [
        {
            "key": "FB:2020-01-02",
            "in_liquid": True,
            "excepted": False,
            "security_id": SID_FB,
            "symbol": "FB",
            "valid_from": date(2020, 1, 2),
            "name": "Facebook Inc",
            "rule": "lookup_sec_funds",
            "cik": FB_CIK,
        }
    ]
    assert result.fails("liquid") is True


def test_m9_lists_a_name_etf_segment_of_a_10k_filer(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    # PRTO: the name says ETF, the CIK is a biotech that files 10-Ks (real residue after the fix)
    add(
        world,
        Seg(
            SID_PRTO,
            "PRTO",
            valid_from=DAYS[0],
            valid_to=DAYS[6],
            available_at=KNOWN_EARLY,
            end_available_at=known(DAYS[6]),
            cik=PRTO_CIK,
            cls="etf",
            rule="name_etf",
        ),
    )
    world.shares = [fact(PRTO_CIK, date(2019, 11, 12), form="10-Q")]
    result = m9(con, world)
    assert [(r["key"], r["name"], r["rule"], r["cik"]) for r in rows(result)] == [
        (f"PRTO:{iso(DAYS[0])}", "RCN Pareto Strategic Allocation ETF", "name_etf", PRTO_CIK)
    ]


def test_m9_leveraged_etf_class_counts_too(con: duckdb.DuckDBPyConnection, world: World) -> None:
    add(world, fb_seg(cls="leveraged_etf", rule="name_leveraged"))
    world.shares = [FB_10K]
    assert keys(m9(con, world)) == ["FB:2020-01-02"]


@pytest.mark.parametrize("form", COMPANY_FORMS)
def test_m9_every_company_report_form_counts(
    con: duckdb.DuckDBPyConnection, world: World, form: str
) -> None:
    add(world, fb_seg())
    world.shares = [fact(FB_CIK, date(2020, 1, 30), form=form)]
    assert keys(m9(con, world)) == ["FB:2020-01-02"]


# =============================================================================================
# boundaries: what is not a row
# =============================================================================================


@pytest.mark.parametrize("form", ["N-CSR", "485BPOS", "8-K", "6-K", "S-1"])
def test_m9_facts_under_other_forms_do_not_count(
    con: duckdb.DuckDBPyConnection, world: World, form: str
) -> None:
    add(world, fb_seg())
    world.shares = [fact(FB_CIK, date(2020, 1, 30), form=form)]
    assert m9(con, world).rows.num_rows == 0


@pytest.mark.parametrize("cls", ["common", "unknown", "preferred", "warrant"])
def test_m9_other_classes_are_not_rows(
    con: duckdb.DuckDBPyConnection, world: World, cls: str
) -> None:
    add(world, fb_seg(cls=cls, rule="lookup_sec_companies"))
    world.shares = [FB_10K]
    assert m9(con, world).rows.num_rows == 0


def test_m9_a_fund_segment_without_a_cik_is_not_a_row(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    # the master carries no CIK, so nothing ties the segment to Facebook's filings
    add(world, fb_seg(cik=None))
    world.shares = [FB_10K]
    assert m9(con, world).rows.num_rows == 0


def test_m9_without_a_shares_file_reports_nothing(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    add(world, fb_seg())
    world.shares = []  # the world writes no shares file
    result = m9(con, world)
    assert result.rows.num_rows == 0 and result.fails("liquid") is False


def test_m9_override_is_exempt_but_its_unreviewed_sibling_is_not(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    # TVIX was classed by a human (rule override); ZIV under the same CIK was not, and Credit
    # Suisse files 20-Fs with share facts. Only ZIV is a row: the exemption is the row's rule.
    add(
        world,
        Seg(SID_NOTE, "TVIX", cik=CS_CIK, cls="leveraged_etf", rule="override"),
        Seg(SID_NOTE2, "ZIV", cik=CS_CIK, cls="etf", rule="name_etf"),
    )
    world.shares = [fact(CS_CIK, date(2019, 3, 22), form="20-F")]
    assert keys(m9(con, world)) == [f"ZIV:{iso(Seg(SID_NOTE2, 'ZIV').valid_from)}"]


def shelf_world(world: World, n_tickers: int, *, stale: Sequence[str] = ()) -> None:
    """BMO's FNGU (leveraged_etf) under a CIK listing ``n_tickers`` distinct tickers in the
    fresh SEC history, plus ``stale`` tickers that appear only in a snapshot before
    SEC_HISTORY_START. Bank of Montreal files 40-Fs."""
    note = Seg(SID_NOTE, "FNGU", cik=BMO_CIK, cls="leveraged_etf", rule="name_leveraged")
    add(world, note)
    sec = default_sec(world.segs)
    sec += [(SEC_HISTORY_START, t, BMO_CIK) for t in shelf_tickers("FNGU", n_tickers)]
    sec += [(SEC_HISTORY_START - timedelta(days=200), t, BMO_CIK) for t in stale]
    world.sec = sorted(set(sec))
    world.shares = [fact(BMO_CIK, date(2019, 12, 3), form="40-F")]


def test_m9_issuer_shelf_at_the_threshold_is_exempt(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    shelf_world(world, SHELF_MIN_TICKERS)
    assert m9(con, world).rows.num_rows == 0


def test_m9_one_ticker_below_the_shelf_threshold_is_a_row(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    shelf_world(world, SHELF_MIN_TICKERS - 1)
    assert [(r["key"], r["cik"]) for r in rows(m9(con, world))] == [
        (f"FNGU:{iso(Seg(SID_NOTE, 'FNGU').valid_from)}", BMO_CIK)
    ]


def test_m9_shelf_count_ignores_snapshots_before_the_sec_history_start(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    # 19 fresh tickers plus one that only the stale pre-2019-10-02 table lists: not a shelf
    shelf_world(world, SHELF_MIN_TICKERS - 1, stale=["BMSTALE"])
    assert keys(m9(con, world)) == [f"FNGU:{iso(Seg(SID_NOTE, 'FNGU').valid_from)}"]


# =============================================================================================
# gate and exceptions
# =============================================================================================


def test_m9_outside_the_liquid_scope_fails_only_the_all_tier(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    add(world, fb_seg(), liquid=False)
    world.shares = [FB_10K]
    result = m9(con, world)
    assert [(r["key"], r["in_liquid"]) for r in rows(result)] == [("FB:2020-01-02", False)]
    assert result.fails("liquid") is False and result.fails("all") is True


def test_m9_inside_the_liquid_scope_fails_the_liquid_tier(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    add(world, fb_seg())
    world.shares = [FB_10K]
    result = m9(con, world)
    assert result.liquid == 1 and result.fails("liquid") is True


def test_m9_scope_is_the_union_over_every_class(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    # a leveraged_etf is never in the common/etf tier union; the CLI passes the union over every
    # class (qualified ids), as for M5, so such a segment is judged inside the liquid scope
    add(world, fb_seg(cls="leveraged_etf"), liquid=False)
    world.shares = [FB_10K]
    assert m9(con, world).liquid == 0
    result = m9(con, world, qualified={SID_FB})
    assert result.liquid == 1 and result.fails("liquid") is True


def test_m9_an_exception_on_the_key_clears_the_gate(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    add(world, fb_seg())
    world.shares = [FB_10K]
    result = m9(con, world, exceptions=[exc("M9", "FB:2020-01-02")])
    assert [(r["key"], r["excepted"]) for r in rows(result)] == [("FB:2020-01-02", True)]
    assert result.liquid == 0 and result.fails("liquid") is False


# =============================================================================================
# leakage: facts and segments not public at as_of
# =============================================================================================


def test_m9_a_10k_filed_after_as_of_does_not_make_a_row(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    # the filer's first 10-K is filed on 2020-07-30 (public 2020-07-31 00:00 New York); the
    # file already holds it, but a reader on 2020-06-01 could not have seen it
    add(world, fb_seg())
    late_10k = fact(FB_CIK, date(2020, 7, 30))
    world.shares = [late_10k]
    assert m9(con, world, as_of=known(date(2020, 6, 1))).rows.num_rows == 0
    # the instant before it is public, and the instant it is
    assert (
        m9(con, world, as_of=late_10k.available_at - timedelta(microseconds=1)).rows.num_rows == 0
    )
    assert keys(m9(con, world, as_of=late_10k.available_at)) == ["FB:2020-01-02"]


def test_m9_a_segment_not_yet_in_the_master_at_as_of_is_not_a_row(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    # the class table already holds the segment, the 10-K is long public, but the segment
    # itself only becomes knowable at 00:00 New York of DAYS[10]
    start = DAYS[10]
    add(world, fb_seg(valid_from=start, available_at=known(start)))
    world.shares = [fact(FB_CIK, date(2019, 1, 31))]
    assert m9(con, world, as_of=avail(DAYS[5])).rows.num_rows == 0
    assert keys(m9(con, world)) == [f"FB:{iso(start)}"]
