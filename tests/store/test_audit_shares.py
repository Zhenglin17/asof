"""Guards against the share-count audit checks missing the two ways a market-cap filter goes
wrong: a liquid common stock that has no usable count on a tier night (F1, it silently drops out
of every cap filter), and a count whose cap no count could give (F2, a count in the wrong unit:
the filter reads it as no count) -- and against either check judging with a count, a bar or a
tier row not yet public at the night it judges.

Target API (asof.store.audit; implementation pending):

    CHECKS["F1"] = CheckSpec("F1", "liquid security without a visible share count", None, SYM)
        Per tier row of ``liquid_tier`` (class "common" only, as_of not after the audit's), the
        security must have a row in ``visible_shares(con, root, row.as_of, ...,
        max_age_days=SHARE_MAX_AGE_DAYS)``. One row per symbol (the symbol the security carries
        on the night), however many nights it is missing. Without ``liquid_tier`` F1 reports
        nothing (the nights come from the dated tier).
    CHECKS["F2"] = CheckSpec("F2", "liquid security with an implausible market cap", None,
                             SYM:DATE)
        Per tier night, ``market_caps`` at 20:00 New York of the last session before it, judged
        by ``implausible`` with the tier row's adv_usd, SHARE_MAX_TURNOVER and SHARE_MAX_CAP.
        One row per symbol and count end, with the nights it was implausible. Listed, not gated.
    SHARE_MAX_AGE_DAYS: the age limit F1 reads counts with (value not pinned here).

The checks are appended after C2 in report order.

Fake data provenance: the audit world of tests/store/audit_world.py (AAA cik 1001, BBB cik
1002 liquid; CCC cik 1003 not). Share facts are constructed with the ingest's availability rule
(00:00 New York of filed + 1 day). Tier rows at 00:00 New York of a month start, as
asof.store.universe.tier_union writes them.

Real-data check: ``asof market audit`` on the real store with the shares file; every F1 / F2
row in the liquid tier is either fixed or excepted with a reason (the standard cases of
tests/store/test_golden_shares.py cover AAPL, NVDA, TSM, TECK).
"""

import re
from collections.abc import Iterator, Sequence
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import pyarrow as pa
import pytest

from asof.ingest.shares import ShareFact
from asof.store import audit as audit_mod
from asof.store.audit import CHECKS, CheckResult, run_audit
from asof.store.market import market_root
from asof.store.universe import LIQUID_SCHEMA
from tests.store.audit_world import (
    LATE,
    SID_AAA,
    SID_BBB,
    World,
    exc,
    keys,
    rows,
)
from tests.store.test_market import known

NY = ZoneInfo("America/New_York")
DEI = "dei:EntityCommonStockSharesOutstanding"
GAAP = "us-gaap:WeightedAverageNumberOfSharesOutstandingBasic"
FEB, MAR = date(2020, 2, 1), date(2020, 3, 1)  # tier nights (month starts)
CIK = {"AAA": 1001, "BBB": 1002, "CCC": 1003}

_n = iter(range(1, 10_000))


def fact(
    symbol_or_cik: str | int,
    end: date,
    shares: float,
    *,
    filed: date | None = None,
    concept: str = DEI,
    accn: str | None = None,
) -> ShareFact:
    cik = CIK[symbol_or_cik] if isinstance(symbol_or_cik, str) else symbol_or_cik
    filed = filed or end + timedelta(days=14)
    return ShareFact(
        cik=cik,
        concept=concept,
        accn=accn or f"{cik:010d}-20-{next(_n):06d}",
        form="10-Q",
        end=end,
        filed=filed,
        shares=float(shares),
        available_at=datetime.combine(filed + timedelta(days=1), time(0), tzinfo=NY).astimezone(
            UTC
        ),
    )


def tier(entries: Sequence[tuple[date, str, str, str]]) -> pa.Table:
    """Liquid-tier rows from (month start, security_id, symbol, class); as_of 00:00 New York."""
    return pa.Table.from_pylist(
        [
            {
                "as_of": known(day),
                "security_id": sid,
                "symbol": symbol,
                "class": cls,
                "adv_usd": 1e8,
                "sessions_traded": 20,
            }
            for day, sid, symbol, cls in entries
        ],
        schema=LIQUID_SCHEMA,
    )


TWO_NIGHTS = tier(
    [
        (FEB, SID_AAA, "AAA", "common"),
        (FEB, SID_BBB, "BBB", "common"),
        (MAR, SID_AAA, "AAA", "common"),
        (MAR, SID_BBB, "BBB", "common"),
    ]
)


@pytest.fixture
def con() -> Iterator[duckdb.DuckDBPyConnection]:
    connection = duckdb.connect()
    yield connection
    connection.close()


@pytest.fixture
def world(tmp_path: Path) -> World:
    return World(market_root(tmp_path))


def run(
    con: duckdb.DuckDBPyConnection,
    world: World,
    check: str,
    *,
    liquid_tier: pa.Table | None = None,
    exceptions: Sequence[audit_mod.AuditException] = (),
    as_of: datetime = LATE,
) -> CheckResult:
    world.write()
    [result] = run_audit(
        con,
        world.root,
        as_of=as_of,
        liquid_ids=world.liquid_ids,
        exceptions=exceptions,
        checks=[check],
        liquid_tier=liquid_tier,
    )
    return result


# =============================================================================================
# registry
# =============================================================================================


def test_specs() -> None:
    f1, f2 = CHECKS["F1"], CHECKS["F2"]
    assert (f1.gate, f2.gate) == (None, None)
    assert f1.title == "liquid security without a visible share count"
    assert f2.title == "liquid security with an implausible market cap"
    assert re.fullmatch(f1.key_pattern, "BRK.B")
    assert not re.fullmatch(f1.key_pattern, "BBB:2020-02-01")
    assert re.fullmatch(f2.key_pattern, "AAA:2019-12-31")
    assert not re.fullmatch(f2.key_pattern, "AAA")
    assert list(CHECKS)[-2:] == ["F1", "F2"]


def test_share_max_age_is_a_positive_day_count() -> None:
    assert isinstance(audit_mod.SHARE_MAX_AGE_DAYS, int) and audit_mod.SHARE_MAX_AGE_DAYS > 0


# =============================================================================================
# F1: normal
# =============================================================================================


def test_f1_lists_a_tier_common_stock_without_a_count_once(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    world.shares = [fact("AAA", date(2019, 12, 27), 1e9)]

    result = run(con, world, "F1", liquid_tier=TWO_NIGHTS)

    assert keys(result) == ["BBB"]  # missing on both nights, one row
    assert result.total == 1


def test_f1_ignores_a_security_outside_the_tier(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    # CCC has no count either, but it is never in the tier
    world.shares = [fact("AAA", date(2019, 12, 27), 1e9), fact("BBB", date(2019, 12, 27), 1e9)]
    assert keys(run(con, world, "F1", liquid_tier=TWO_NIGHTS)) == []


def test_f1_ignores_other_classes(con: duckdb.DuckDBPyConnection, world: World) -> None:
    world.shares = [fact("AAA", date(2019, 12, 27), 1e9)]
    etf_bbb = tier([(FEB, SID_AAA, "AAA", "common"), (FEB, SID_BBB, "BBB", "etf")])
    assert keys(run(con, world, "F1", liquid_tier=etf_bbb)) == []


# =============================================================================================
# F1: boundaries
# =============================================================================================


def test_f1_a_count_older_than_the_limit_is_missing(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    limit = audit_mod.SHARE_MAX_AGE_DAYS
    world.shares = [
        fact("AAA", FEB - timedelta(days=limit), 1e9, filed=FEB - timedelta(days=limit - 1)),
        fact("BBB", FEB - timedelta(days=limit + 1), 1e9, filed=FEB - timedelta(days=limit)),
    ]
    feb_only = tier([(FEB, SID_AAA, "AAA", "common"), (FEB, SID_BBB, "BBB", "common")])
    assert keys(run(con, world, "F1", liquid_tier=feb_only)) == ["BBB"]


def test_f1_is_listed_not_gated(con: duckdb.DuckDBPyConnection, world: World) -> None:
    result = run(con, world, "F1", liquid_tier=TWO_NIGHTS)
    assert keys(result) == ["AAA", "BBB"]
    assert result.fails("liquid") is False and result.fails("all") is False


def test_f1_an_exception_silences_the_row(con: duckdb.DuckDBPyConnection, world: World) -> None:
    world.shares = [fact("AAA", date(2019, 12, 27), 1e9)]
    result = run(con, world, "F1", liquid_tier=TWO_NIGHTS, exceptions=[exc("F1", "BBB")])
    assert [(r["key"], r["excepted"]) for r in rows(result)] == [("BBB", True)]
    assert result.total == 0


def test_f1_without_a_dated_tier_reports_nothing(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    assert run(con, world, "F1", liquid_tier=None).total == 0


# =============================================================================================
# F1: leakage
# =============================================================================================


def test_f1_judges_each_night_with_the_counts_public_by_then(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    # BBB's count is filed on 02-10: it exists and is visible at the audit's as_of, but not on
    # the February night
    world.shares = [
        fact("AAA", date(2019, 12, 27), 1e9),
        fact("BBB", date(2019, 12, 27), 1e9, filed=date(2020, 2, 10)),
    ]
    feb = tier([(FEB, SID_AAA, "AAA", "common"), (FEB, SID_BBB, "BBB", "common")])
    mar = tier([(MAR, SID_AAA, "AAA", "common"), (MAR, SID_BBB, "BBB", "common")])

    assert keys(run(con, world, "F1", liquid_tier=feb)) == ["BBB"]
    assert keys(run(con, world, "F1", liquid_tier=mar)) == []


def test_f1_count_filed_the_day_before_the_night_covers_it(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    # filed 01-31: public from 00:00 New York 02-01, exactly the tier night
    world.shares = [
        fact("AAA", date(2019, 12, 27), 1e9),
        fact("BBB", date(2019, 12, 27), 1e9, filed=date(2020, 1, 31)),
    ]
    assert keys(run(con, world, "F1", liquid_tier=TWO_NIGHTS)) == []


def test_f1_tier_rows_after_the_audit_as_of_do_not_count(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    later = tier([(date(2021, 2, 1), SID_BBB, "BBB", "common")])  # after LATE (2021-01-04)
    assert run(con, world, "F1", liquid_tier=later).total == 0


# =============================================================================================
# F2: normal (tier rows trade 1e8 a day; AAA closes at 100, BBB at 60; the session before FEB is
# 2020-01-31, before MAR the last stored one, 2020-02-06)
# =============================================================================================


def test_f2_lists_a_cap_below_a_days_dollar_volume_once_per_count(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    # VALE-like: 1e5 shares at 100 is a 1e7 cap trading 1e8 a day
    world.shares = [fact("AAA", date(2019, 12, 27), 1e5), fact("BBB", date(2019, 12, 27), 1e9)]

    result = run(con, world, "F2", liquid_tier=TWO_NIGHTS)

    assert [(r["key"], r["nights"], r["first_night"], r["last_night"]) for r in rows(result)] == [
        ("AAA:2019-12-27", 2, date(2020, 1, 31), date(2020, 2, 6))
    ]
    assert rows(result)[0]["raw_cap"] == 1e7


def test_f2_lists_a_cap_above_the_ceiling(con: duckdb.DuckDBPyConnection, world: World) -> None:
    world.shares = [fact("AAA", date(2019, 12, 27), 1e9), fact("BBB", date(2019, 12, 27), 1e11)]
    assert keys(run(con, world, "F2", liquid_tier=TWO_NIGHTS)) == ["BBB:2019-12-27"]


def test_f2_is_listed_not_gated(con: duckdb.DuckDBPyConnection, world: World) -> None:
    world.shares = [fact("AAA", date(2019, 12, 27), 1e5)]
    result = run(con, world, "F2", liquid_tier=TWO_NIGHTS)
    assert result.total == 1
    assert result.fails("liquid") is False
    assert result.fails("all") is False


def test_f2_an_exception_silences_the_row(con: duckdb.DuckDBPyConnection, world: World) -> None:
    world.shares = [fact("AAA", date(2019, 12, 27), 1e5)]
    result = run(con, world, "F2", liquid_tier=TWO_NIGHTS, exceptions=[exc("F2", "AAA:2019-12-27")])
    assert [(r["key"], r["excepted"]) for r in rows(result)] == [("AAA:2019-12-27", True)]


# =============================================================================================
# F2: boundaries
# =============================================================================================


def test_f2_a_plausible_or_missing_count_is_not_listed(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    world.shares = [fact("AAA", date(2019, 12, 27), 1e9)]  # BBB has none: that is F1's row
    assert run(con, world, "F2", liquid_tier=TWO_NIGHTS).total == 0


def test_f2_without_a_dated_tier_reports_nothing(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    world.shares = [fact("AAA", date(2019, 12, 27), 1e5)]
    assert run(con, world, "F2").total == 0


def test_f2_ignores_other_classes(con: duckdb.DuckDBPyConnection, world: World) -> None:
    world.shares = [fact("AAA", date(2019, 12, 27), 1e5)]
    etf = tier([(FEB, SID_AAA, "AAA", "etf")])
    assert run(con, world, "F2", liquid_tier=etf).total == 0


def test_f2_without_a_shares_file_reports_nothing(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    assert run(con, world, "F2", liquid_tier=TWO_NIGHTS).total == 0


# =============================================================================================
# F2: leakage
# =============================================================================================


def test_f2_judges_each_night_with_the_count_public_by_then(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    # the wrong count is filed 2020-02-03: after the session before FEB, before the one before MAR
    world.shares = [
        fact("AAA", date(2019, 12, 27), 1e9),
        fact("AAA", date(2020, 1, 31), 1e5, filed=date(2020, 2, 3)),
    ]
    result = run(con, world, "F2", liquid_tier=TWO_NIGHTS)
    assert [(r["key"], r["nights"], r["first_night"]) for r in rows(result)] == [
        ("AAA:2020-01-31", 1, date(2020, 2, 6))
    ]


def test_f2_tier_rows_after_the_audit_as_of_do_not_count(
    con: duckdb.DuckDBPyConnection, world: World
) -> None:
    world.shares = [fact("AAA", date(2019, 12, 27), 1e5)]
    result = run(con, world, "F2", liquid_tier=TWO_NIGHTS, as_of=datetime(2020, 2, 15, tzinfo=UTC))
    assert [(r["key"], r["nights"]) for r in rows(result)] == [("AAA:2019-12-27", 1)]
