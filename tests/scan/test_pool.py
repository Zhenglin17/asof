"""Guards against the null's candidate pool being wider than what the earnings-gap scanner could
have returned (a penny stock, a thin name, a company's warrant, a non-filer) or built from data a
reader at ``as_of`` did not have: a later huge-volume day lifting the average, a later close
lifting the price, an 8-K accepted after ``as_of``, a CIK link not yet knowable.

API under test (asof.scan.earnings_gap, new function):

    pool(session, con, root, as_of, cfg, *, bars_at=None) -> frozenset[str]
        The securities scan() would consider at ``as_of`` minus the signal: entities with an
        8-K (any item) visible at as_of; CIKs = the filings' archive-URL CIKs plus the entity's
        own; one security per entity via _security_of on ``today`` (the latest session final at
        as_of); skipped when a visible split has its ex-date in the scan window; kept when
        today's close >= scan.min_price and the 20-session average dollar volume before today
        >= scan.min_avg_dollar_volume (NaN = too little history = out). No gap, rvol or
        earnings-item test.
        ``bars_at(as_of, start)``: optional reader with visible_bars semantics.

The store is built with tests/scan/world.py's writers (provenance notes there). Additional fake
data, all made up: every name trades flat (open = close) at a fixed price and volume chosen so
dollar volume sits on round numbers ($50M, exactly $20M, one share short of $20M).
The config is tests/strategy/files.py's mirror of the approved earnings_gap thresholds
($5 close, $20M average dollar volume over 20 sessions).

Interface points this file pins down where the spec left a choice:
- ``bars_at`` is called with the reader's ``as_of`` first and a start bound; the fake readers
  accept ``start`` positionally or by keyword.
- An ``as_of`` before any final session gives an empty set; a naive ``as_of`` raises ValueError.
- Two securities under one CIK with neither trading under the entity's ticker: neither is in
  (as _security_of decides for scan()).

Assumptions these fixtures cannot settle; check them on the real store:
- The pool is a superset of scan()'s output on every replay evening. Check: replay any window
  and assert every Candidate.security_id is in pool() at the same as_of (zero exceptions).
- The pool's median size on real data is about what scan() could ever reach (one security per
  filer that clears $5 / $20M), well under the 335 of "every security of every 8-K filer".
  Check: median pool size over the run 1 window, printed beside the old number.
"""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import duckdb
import pyarrow as pa
import pytest
from sqlmodel import Session

from asof.ingest.alpaca import Bar
from asof.scan.earnings_gap import pool, scan
from asof.store.market import market_root, visible_bars
from asof.strategy import StrategyConfig, apply_overrides, load_strategy
from tests.scan.world import (
    HISTORY,
    KNOWN_EARLY,
    PREV,
    SESSIONS,
    D,
    Seg,
    bar,
    close_of,
    company,
    edgar,
    et,
    filing,
    gap_world,
    write_master,
)
from tests.store.test_market import store_daily, write_splits
from tests.strategy.files import write_strategy

SECOND = timedelta(seconds=1)
AS_OF = close_of(D)  # 2026-02-17 20:00 New York
AFTER = 3  # sessions of bars on disk after D
DAYS = SESSIONS[: HISTORY + 1 + AFTER]
EARLY = et(SESSIONS[0], 8)  # 2026-01-05 08:00 New York: every early 8-K is long visible
NEXT = SESSIONS[HISTORY + 1]


def config(tmp_path: Path, *overrides: str) -> StrategyConfig:
    return apply_overrides(load_strategy(write_strategy(tmp_path / "strategies")), list(overrides))


@dataclass(frozen=True)
class Name:
    """One security trading flat at ``close`` x ``volume`` on DAYS[first:]."""

    security_id: str
    symbol: str
    cik: int
    close: float
    volume: int
    first: int = 0


@dataclass(frozen=True)
class Filer:
    ticker: str
    cik: int  # the entity's CIK today
    filings: tuple[tuple[str, datetime, int | None], ...]  # (doc_type, available_at, url CIK)


NAMES: tuple[Name, ...] = (
    Name("S_LIQ", "LIQ", 1001, 50.0, 1_000_000),  # $50M a day
    Name("S_PENNY", "PENNY", 1002, 4.99, 10_000_000),  # $49.9M a day under $5
    Name("S_THIN", "THIN", 1003, 50.0, 399_999),  # $50 short of $20M a day
    Name("S_EDGE", "EDGE", 1010, 50.0, 400_000),  # exactly $20M a day
    Name("S_FIVE", "FIVE", 1011, 5.0, 10_000_000),  # exactly $5
    Name("S_DUAL", "DUAL", 1004, 30.0, 2_000_000),  # common ...
    Name("S_DUALW", "DUAL.WS", 1004, 30.0, 2_000_000),  # ... and its warrant, same CIK
    Name("S_BRKA", "BRK.A", 1012, 30.0, 2_000_000),  # two classes, entity ticker "BRK"
    Name("S_BRKB", "BRK.B", 1012, 30.0, 2_000_000),
    Name("S_NOFILE", "NOFILE", 1005, 50.0, 1_000_000),
    Name("S_TENQ", "TENQ", 1006, 50.0, 1_000_000),
    Name("S_LATE", "LATE", 1007, 50.0, 1_000_000),
    Name("S_SHORT20", "SHORT20", 1008, 50.0, 1_000_000, first=HISTORY - 20),  # 20 sessions
    Name("S_SHORT19", "SHORT19", 1013, 50.0, 1_000_000, first=HISTORY - 19),  # 19 sessions
    Name("S_OLDCIK", "OLDCIK", 1015, 50.0, 1_000_000),  # master still carries the old CIK
    Name("S_UNION", "UNION", 1016, 50.0, 1_000_000),
    Name("S_RISER", "RISER", 1017, 4.0, 10_000_000),  # $40M a day, closes at 6 after D
    Name("S_HALT", "HALT", 1014, 50.0, 1_000_000),  # no bar on D
)
EIGHT_K_ANY = (("8-K", EARLY, None),)
FILERS: tuple[Filer, ...] = (
    Filer("LIQ", 1001, EIGHT_K_ANY),
    Filer("PENNY", 1002, EIGHT_K_ANY),
    Filer("THIN", 1003, EIGHT_K_ANY),
    Filer("EDGE", 1010, EIGHT_K_ANY),
    Filer("FIVE", 1011, EIGHT_K_ANY),
    Filer("DUAL", 1004, EIGHT_K_ANY),
    Filer("BRK", 1012, EIGHT_K_ANY),
    Filer("TENQ", 1006, (("10-Q", EARLY, None),)),
    Filer("LATE", 1007, (("8-K", AS_OF + SECOND, None),)),  # accepted one second after as_of
    Filer("SHORT20", 1008, EIGHT_K_ANY),
    Filer("SHORT19", 1013, EIGHT_K_ANY),
    # XOM shape (real: filings under 34088, the entity carries 2115436 today).
    Filer("OLDCIK", 7777, (("8-K", EARLY, 1015),)),
    # The filing's URL CIK matches nothing in the master; the entity's own CIK does.
    Filer("UNION", 1016, (("8-K", EARLY, 9999),)),
    Filer("RISER", 1017, EIGHT_K_ANY),
    Filer("HALT", 1014, EIGHT_K_ANY),
)
# What the pool holds at AS_OF with the approved thresholds.
EXPECTED = frozenset({"S_LIQ", "S_EDGE", "S_FIVE", "S_DUAL", "S_SHORT20", "S_OLDCIK", "S_UNION"})

Ohlcv = tuple[float, float, float, float, int]


def flat(n: Name) -> Ohlcv:
    return (n.close, n.close, n.close, n.close, n.volume)


def pool_world(
    tmp_path: Path,
    session: Session,
    *,
    override: Mapping[tuple[str, date], Ohlcv | None] | None = None,
    segs: Sequence[Seg] | None = None,
) -> Path:
    """NAMES on DAYS (the reaction-day calendar of tests/scan/world.py plus AFTER sessions),
    FILERS in SQLite. ``override`` replaces a bar; None removes it."""
    root = market_root(tmp_path)
    changes = dict(override or {})
    changes.setdefault(("HALT", D), None)  # HALT does not trade on D
    changes.setdefault(("RISER", NEXT), (6.0, 6.0, 6.0, 6.0, 10_000_000))
    bars: list[Bar] = []
    for n in NAMES:
        for day in DAYS[n.first :]:
            values = changes[(n.symbol, day)] if (n.symbol, day) in changes else flat(n)
            if values is not None:
                bars.append(bar(n.symbol, day, values))
    store_daily(root, bars)
    write_master(root, segs or [Seg(n.security_id, n.symbol, n.cik) for n in NAMES])
    source_id = edgar(session)
    for i, f in enumerate(FILERS):
        entity_id = company(session, f.ticker, f.cik)
        for j, (doc_type, available, url_cik) in enumerate(f.filings):
            items = "7.01,9.01" if doc_type == "8-K" else ""  # not an earnings item
            filing(
                session,
                source_id=source_id,
                entity_id=entity_id,
                accession=f"0000{f.cik:06d}-26-{i:03d}{j:03d}",
                available_at=available,
                exhibits=[("ex991.htm", f"{doc_type} {items} EX-99.1 INVESTOR UPDATE".strip())],
                doc_type=doc_type,
                filer_cik=url_cik,
            )
    return root


@pytest.fixture
def con() -> Iterator[duckdb.DuckDBPyConnection]:
    connection = duckdb.connect()
    yield connection
    connection.close()


# =============================================================================================
# normal path
# =============================================================================================


def test_the_pool_is_what_the_scanner_could_return_minus_the_signal(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection
) -> None:
    root = pool_world(tmp_path, session)

    got = pool(session, con, root, AS_OF, config(tmp_path))

    assert isinstance(got, frozenset)
    # In: liquid filers whose 8-K is not an earnings release (any item counts), the exact
    # thresholds, the old-CIK filing, the entity-CIK fallback, 20 sessions of history.
    # Out: under $5, under $20M, the warrant, the two-class company without its ticker, the
    # non-filer, the 10-Q filer, the late filer, 19 sessions of history, no bar today.
    assert got == EXPECTED


def test_a_company_contributes_only_the_security_the_scanner_would_choose(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection
) -> None:
    root = pool_world(tmp_path, session)

    got = pool(session, con, root, AS_OF, config(tmp_path))

    assert "S_DUAL" in got
    assert "S_DUALW" not in got
    assert not {"S_BRKA", "S_BRKB"} & got


def test_an_earnings_gap_candidate_is_in_the_pool(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection
) -> None:
    # Consistency with scan(): the pool is the scanner without the gap and rvol filters.
    world = gap_world(tmp_path, session)
    cfg = config(tmp_path)

    [candidate] = scan(session, con, world.root, close_of(D), cfg)
    got = pool(session, con, world.root, close_of(D), cfg)

    assert candidate.security_id in got
    assert got == frozenset({"S_GAPX"})  # OTHR is liquid but never files


def test_a_custom_bar_reader_is_used(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection
) -> None:
    root = pool_world(tmp_path, session)
    cfg = config(tmp_path)
    calls: list[datetime] = []

    def reader(as_of: datetime, start: datetime | None = None) -> pa.Table:
        calls.append(as_of)
        return visible_bars(con, root, "1Day", as_of, start=start)

    def nothing(as_of: datetime, start: datetime | None = None) -> pa.Table:
        return visible_bars(con, root, "1Day", as_of, start=start).slice(0, 0)

    assert pool(session, con, root, AS_OF, cfg, bars_at=reader) == EXPECTED
    assert calls and all(c == AS_OF for c in calls)
    assert pool(session, con, root, AS_OF, cfg, bars_at=nothing) == frozenset()


# =============================================================================================
# boundaries
# =============================================================================================


@pytest.mark.parametrize(
    ("override", "liq_in"),
    [
        ("scan.min_price=50.0", True),  # LIQ closes at exactly 50
        ("scan.min_price=50.01", False),
        ("scan.min_avg_dollar_volume=50000000", True),  # LIQ averages exactly $50M
        ("scan.min_avg_dollar_volume=50000001", False),
    ],
)
def test_thresholds_are_inclusive_and_read_from_the_config(
    tmp_path: Path,
    session: Session,
    con: duckdb.DuckDBPyConnection,
    override: str,
    liq_in: bool,
) -> None:
    root = pool_world(tmp_path, session)

    got = pool(session, con, root, AS_OF, config(tmp_path, override))

    assert ("S_LIQ" in got) == liq_in


def test_gap_and_rvol_thresholds_do_not_filter_the_pool(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection
) -> None:
    root = pool_world(tmp_path, session)
    strict = config(tmp_path, "scan.min_gap_pct=50.0", "scan.min_rvol=10.0")

    assert pool(session, con, root, AS_OF, strict) == EXPECTED


@pytest.mark.parametrize(
    ("ex_date", "liq_in"),
    [
        (SESSIONS[HISTORY - 21], False),  # the first session of the 21-session window
        (SESSIONS[HISTORY - 22], True),  # the session before it
        (SESSIONS[HISTORY - 5], False),
        (D, False),  # today
    ],
)
def test_a_split_inside_the_scan_window_takes_the_security_out(
    tmp_path: Path,
    session: Session,
    con: duckdb.DuckDBPyConnection,
    ex_date: date,
    liq_in: bool,
) -> None:
    root = pool_world(tmp_path, session)
    write_splits(root, [("LIQ", ex_date, 1.0, 2.0)])

    got = pool(session, con, root, AS_OF, config(tmp_path))

    assert ("S_LIQ" in got) == liq_in
    assert got - {"S_LIQ"} == EXPECTED - {"S_LIQ"}


def test_twenty_sessions_of_history_are_enough_and_nineteen_are_not(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection
) -> None:
    root = pool_world(tmp_path, session)

    got = pool(session, con, root, AS_OF, config(tmp_path))

    assert "S_SHORT20" in got
    assert "S_SHORT19" not in got


def test_no_session_final_yet_gives_an_empty_pool_and_naive_as_of_raises(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection
) -> None:
    root = pool_world(tmp_path, session)
    cfg = config(tmp_path)

    assert pool(session, con, root, close_of(SESSIONS[0]) - SECOND, cfg) == frozenset()
    with pytest.raises(ValueError):
        pool(session, con, root, datetime(2026, 2, 17, 20), cfg)


def test_no_bar_on_the_previous_session_takes_the_security_out_as_in_scan(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection
) -> None:
    # scan() refuses a reaction day whose previous session has no bar (prev_close would be an
    # older close); the pool mirrors it.
    root = pool_world(tmp_path, session, override={("LIQ", PREV): None})

    got = pool(session, con, root, AS_OF, config(tmp_path))

    assert got == EXPECTED - {"S_LIQ"}


# =============================================================================================
# leakage (the future is on disk but not visible)
# =============================================================================================


def test_a_huge_volume_day_after_as_of_does_not_lift_the_average(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection
) -> None:
    # THIN trades $1B on the session after D; the bar is on disk while the pool is read at D.
    root = pool_world(
        tmp_path, session, override={("THIN", NEXT): (50.0, 50.0, 50.0, 50.0, 20_000_000)}
    )
    cfg = config(tmp_path)

    assert "S_THIN" not in pool(session, con, root, AS_OF, cfg)
    # On NEXT itself the day is today's, outside its own average; one session later it counts.
    assert "S_THIN" not in pool(session, con, root, close_of(NEXT), cfg)
    assert "S_THIN" in pool(session, con, root, close_of(SESSIONS[HISTORY + 2]), cfg)


def test_the_close_that_counts_is_todays_not_a_later_one(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection
) -> None:
    # RISER closes at 4 through D and at 6 on NEXT.
    root = pool_world(tmp_path, session)
    cfg = config(tmp_path)

    assert "S_RISER" not in pool(session, con, root, AS_OF, cfg)
    assert "S_RISER" in pool(session, con, root, close_of(NEXT), cfg)


def test_an_8k_accepted_after_as_of_is_not_seen(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection
) -> None:
    root = pool_world(tmp_path, session)
    cfg = config(tmp_path)

    assert "S_LATE" not in pool(session, con, root, AS_OF, cfg)
    assert "S_LATE" in pool(session, con, root, AS_OF + SECOND, cfg)


def test_one_second_before_the_close_today_is_still_the_previous_session(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection
) -> None:
    # At 19:59:59 on D the D bars are on disk but not final: today is PREV, on which HALT traded.
    root = pool_world(tmp_path, session)
    cfg = config(tmp_path)

    assert "S_HALT" in pool(session, con, root, AS_OF - SECOND, cfg)
    assert "S_HALT" not in pool(session, con, root, AS_OF, cfg)
    assert pool(session, con, root, AS_OF - SECOND, cfg) == pool(
        session, con, root, close_of(PREV), cfg
    )


def test_a_cik_link_not_yet_knowable_contributes_nothing(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection
) -> None:
    later = AS_OF + timedelta(hours=1)
    segs = [
        Seg(n.security_id, n.symbol, n.cik, later if n.symbol == "LIQ" else KNOWN_EARLY)
        for n in NAMES
    ]
    root = pool_world(tmp_path, session, segs=segs)
    cfg = config(tmp_path)

    assert "S_LIQ" not in pool(session, con, root, AS_OF, cfg)
    assert "S_LIQ" in pool(session, con, root, later, cfg)


def test_a_split_on_file_only_after_as_of_does_not_take_the_security_out(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection
) -> None:
    root = pool_world(tmp_path, session)
    write_splits(
        root,
        [("LIQ", SESSIONS[HISTORY - 5], 1.0, 2.0)],
        available_at=(AS_OF + timedelta(days=3)).astimezone(UTC),
    )

    assert "S_LIQ" in pool(session, con, root, AS_OF, config(tmp_path))
