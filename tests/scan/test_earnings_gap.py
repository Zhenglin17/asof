"""Guards against the earnings-gap scanner trading on something it could not have known at 20:00
New York: a reaction-day bar that is not final yet, a filing mapped to the session it was
released on instead of the first session that opened after it, a CIK-to-security link not yet
knowable, or a split record not yet published; and against it missing or duplicating real
signals: thresholds off by one at their edges, non-earnings 8-Ks, gap-downs, one event split into
several candidates by its exhibits, and gaps distorted by a split.

API under test (asof.scan.earnings_gap):

    filing_items(title) -> frozenset[str]
    reaction_session(available_at, sessions) -> date | None
    scan(session, con, root, as_of, cfg: StrategyConfig) -> list[Candidate]
        Candidate: entity_id, security_id, symbol, document_ids, available_at, reaction_session,
        pre_earnings_close, gap_open, gap_high, gap_low, gap_close, gap, rvol, atr,
        avg_dollar_volume

The store is tests/scan/world.py (provenance notes there). Release-time cases (16:08 after the
close, 07:00 pre-market, exactly 09:30) mirror real EDGAR acceptance times: RDDT accepted
2026-04-30 16:08:35 New York (real); the pre-market and 09:30 cases are constructed.
"""

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import duckdb
import pytest
from sqlmodel import Session

from asof.scan.earnings_gap import filing_items, reaction_session, scan
from asof.strategy import apply_overrides, load_strategy
from tests.scan.world import (
    GAPX_CIK,
    HISTORY,
    KNOWN_EARLY,
    PREV,
    QUIET,
    SESSIONS,
    D,
    Seg,
    bar,
    close_of,
    et,
    filing,
    gap_world,
)
from tests.store.test_market import write_splits
from tests.strategy.files import write_strategy

SECOND = timedelta(seconds=1)


def config(tmp_path: Path, *overrides: str):  # returns a StrategyConfig
    return apply_overrides(load_strategy(write_strategy(tmp_path / "strategies")), list(overrides))


# =============================================================================================
# filing_items
# =============================================================================================


@pytest.mark.parametrize(
    ("title", "items"),
    [
        ("8-K 2.02,9.01 EX-99.1 PRESS RELEASE", {"2.02", "9.01"}),  # real title shape
        ("8-K 2.02,9.01 EX-99.2 EX-99.2", {"2.02", "9.01"}),  # real: RDDT exhibit992q126.htm
        ("8-K 7.01,9.01 EX-99.1 EXHIBIT 99.1", {"7.01", "9.01"}),
        ("8-K 2.02 EX-99.1 PRESS RELEASE", {"2.02"}),
    ],
)
def test_items_are_read_from_the_title(title: str, items: set[str]) -> None:
    assert filing_items(title) == frozenset(items)


@pytest.mark.parametrize(
    "title",
    [
        "10-Q 10-Q 10-Q",  # real title of RDDT's 10-Q
        "8-K EX-99.1 PRESS RELEASE",  # an 8-K without items
        "",
        "S-1 EX-99.1 ITEM 2.02 MENTIONED IN THE DESCRIPTION",
    ],
)
def test_no_items_when_not_an_8k_or_none_listed(title: str) -> None:
    assert filing_items(title) == frozenset()


def test_item_like_numbers_in_the_description_are_not_items() -> None:
    assert filing_items("8-K 7.01 EX-99.1 SEE ITEM 2.02 OF THE PRIOR REPORT") == frozenset({"7.01"})


# =============================================================================================
# reaction_session
# =============================================================================================

SPRING = [date(2026, 4, 29), date(2026, 4, 30), date(2026, 5, 1), date(2026, 5, 4)]
WINTER = [date(2026, 1, 15), date(2026, 1, 16), date(2026, 1, 20)]  # 01-19 is a holiday


def test_after_close_release_reacts_next_session() -> None:
    # real: RDDT accepted 2026-04-30 20:08:35 UTC = 16:08:35 New York
    assert reaction_session(datetime(2026, 4, 30, 20, 8, 35, tzinfo=UTC), SPRING) == date(
        2026, 5, 1
    )


def test_premarket_release_reacts_same_day() -> None:
    assert reaction_session(et(date(2026, 5, 1), 7), SPRING) == date(2026, 5, 1)


def test_release_exactly_at_the_open_reacts_next_session() -> None:
    assert reaction_session(et(date(2026, 5, 1), 9, 30), SPRING) == date(2026, 5, 4)


def test_release_one_second_before_the_open_reacts_same_day() -> None:
    assert reaction_session(et(date(2026, 5, 1), 9, 29, 59), SPRING) == date(2026, 5, 1)


def test_friday_and_weekend_releases_react_monday() -> None:
    assert reaction_session(et(date(2026, 5, 1), 16, 30), SPRING) == date(2026, 5, 4)
    assert reaction_session(et(date(2026, 5, 2), 11), SPRING) == date(2026, 5, 4)


def test_open_is_09_30_new_york_in_winter_too() -> None:
    # EST: 09:30 New York = 14:30 UTC
    assert reaction_session(datetime(2026, 1, 15, 14, 29, 59, tzinfo=UTC), WINTER) == date(
        2026, 1, 15
    )
    assert reaction_session(datetime(2026, 1, 15, 14, 30, tzinfo=UTC), WINTER) == date(2026, 1, 16)


def test_holidays_are_skipped_by_the_calendar() -> None:
    assert reaction_session(et(date(2026, 1, 16), 16, 5), WINTER) == date(2026, 1, 20)


def test_no_later_session_means_none() -> None:
    assert reaction_session(et(date(2026, 5, 4), 10), SPRING) is None
    assert reaction_session(et(date(2026, 5, 4), 10), []) is None


def test_naive_release_time_is_refused() -> None:
    with pytest.raises(ValueError):
        reaction_session(datetime(2026, 5, 1, 7), SPRING)


# =============================================================================================
# scan: normal path
# =============================================================================================


def test_gap_after_an_earnings_8k_is_a_candidate(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection
) -> None:
    world = gap_world(tmp_path, session)
    [c] = scan(session, con, world.root, close_of(D), config(tmp_path))
    assert c.entity_id == world.gapx
    assert c.security_id == "S_GAPX"
    assert c.symbol == "GAPX"
    assert sorted(c.document_ids) == sorted(world.earnings_docs)  # both exhibits, one candidate
    assert c.available_at == et(SESSIONS[HISTORY - 1], 16, 8)
    assert c.reaction_session == D
    assert c.pre_earnings_close == 50.0
    assert (c.gap_open, c.gap_high, c.gap_low, c.gap_close) == (55.0, 57.0, 54.0, 56.0)
    assert c.gap == pytest.approx(0.10)
    assert c.rvol == pytest.approx(3.0)
    assert c.atr == pytest.approx(2.0)
    assert c.avg_dollar_volume == pytest.approx(50_000_000.0)


def test_two_earnings_filings_for_one_reaction_make_one_candidate(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection
) -> None:
    world = gap_world(tmp_path, session)
    later = filing(
        session,
        source_id=world.source_id,
        entity_id=world.gapx,
        accession="0000001111-26-000004",
        available_at=et(SESSIONS[HISTORY - 1], 18),
        exhibits=[("ex991.htm", "8-K 2.02,9.01 EX-99.1 SUPPLEMENTAL INFORMATION")],
    )
    [c] = scan(session, con, world.root, close_of(D), config(tmp_path))
    assert sorted(c.document_ids) == sorted(world.earnings_docs + later)
    assert c.available_at == et(SESSIONS[HISTORY - 1], 16, 8)  # the earliest


def test_premarket_release_on_the_gap_day_is_a_candidate(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection
) -> None:
    world = gap_world(tmp_path, session, release_at=et(D, 7))
    [c] = scan(session, con, world.root, close_of(D), config(tmp_path))
    assert c.reaction_session == D


# =============================================================================================
# scan: boundaries
# =============================================================================================


@pytest.mark.parametrize(
    ("override", "found"),
    [
        ("scan.min_gap_pct=10.0", True),  # gap is exactly 10%
        ("scan.min_gap_pct=10.01", False),
        ("scan.min_rvol=3.0", True),  # rvol is exactly 3
        ("scan.min_rvol=3.01", False),
        ("scan.min_price=56.0", True),  # gap-day close is exactly 56
        ("scan.min_price=56.01", False),
        ("scan.min_avg_dollar_volume=50000000", True),  # exactly $50M
        ("scan.min_avg_dollar_volume=50000001", False),
    ],
)
def test_thresholds_are_inclusive(
    tmp_path: Path,
    session: Session,
    con: duckdb.DuckDBPyConnection,
    override: str,
    found: bool,
) -> None:
    world = gap_world(tmp_path, session)
    candidates = scan(session, con, world.root, close_of(D), config(tmp_path, override))
    assert len(candidates) == (1 if found else 0)


def test_gap_down_is_not_a_long_candidate(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection
) -> None:
    world = gap_world(tmp_path, session, reaction=(45.0, 46.0, 44.0, 45.0, 3_000_000))
    assert scan(session, con, world.root, close_of(D), config(tmp_path)) == []


def test_non_earnings_8k_is_ignored(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection
) -> None:
    world = gap_world(
        tmp_path, session, exhibits=[("ex991.htm", "8-K 7.01,9.01 EX-99.1 INVESTOR DAY")]
    )
    assert scan(session, con, world.root, close_of(D), config(tmp_path)) == []


def test_other_document_types_are_ignored(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection
) -> None:
    world = gap_world(tmp_path, session, exhibits=[])
    filing(
        session,
        source_id=world.source_id,
        entity_id=world.gapx,
        accession="0000001111-26-000005",
        available_at=et(SESSIONS[HISTORY - 1], 16, 8),
        exhibits=[("q.htm", "10-Q 10-Q 10-Q")],
        doc_type="10-Q",
    )
    assert scan(session, con, world.root, close_of(D), config(tmp_path)) == []


def test_release_exactly_at_the_open_belongs_to_the_next_session(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection
) -> None:
    world = gap_world(tmp_path, session, release_at=et(D, 9, 30), after=[QUIET])
    assert scan(session, con, world.root, close_of(D), config(tmp_path)) == []


def test_only_the_reaction_day_scan_reports_the_event(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection
) -> None:
    # The replay calls scan every session; the event must not come back the next evening.
    world = gap_world(tmp_path, session, after=[(56.0, 57.0, 55.0, 56.5, 1_200_000)])
    cfg = config(tmp_path)
    assert len(scan(session, con, world.root, close_of(D), cfg)) == 1
    assert scan(session, con, world.root, close_of(SESSIONS[HISTORY + 1]), cfg) == []


@pytest.mark.parametrize(
    ("ex_date", "found"),
    [
        (SESSIONS[HISTORY - 5], False),  # inside the 20-session lookback
        (D, False),  # on the reaction day: the gap itself is a split artefact
        (SESSIONS[HISTORY - 25], True),  # well before the lookback
    ],
)
def test_a_split_near_the_gap_skips_the_candidate(
    tmp_path: Path,
    session: Session,
    con: duckdb.DuckDBPyConnection,
    ex_date: date,
    found: bool,
) -> None:
    world = gap_world(tmp_path, session)
    write_splits(world.root, [("GAPX", ex_date, 1.0, 2.0)])
    candidates = scan(session, con, world.root, close_of(D), config(tmp_path))
    assert len(candidates) == (1 if found else 0)


# =============================================================================================
# scan: leakage (the future is on disk but not visible)
# =============================================================================================


def test_one_second_before_the_bar_is_final_there_is_nothing(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection
) -> None:
    # 19:59:59 New York: the 8-K and the gap-day bar are both on disk; the bar is not final
    world = gap_world(tmp_path, session)
    assert scan(session, con, world.root, close_of(D) - SECOND, config(tmp_path)) == []


def test_a_cik_link_not_yet_knowable_finds_no_security(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection
) -> None:
    world = gap_world(tmp_path, session, gapx_cik_available_at=close_of(D) + SECOND)
    assert scan(session, con, world.root, close_of(D), config(tmp_path)) == []


def test_a_filing_accepted_after_as_of_is_invisible_even_on_a_gap_day(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection
) -> None:
    # The gap is there, the 8-K is in the store, but EDGAR accepted it 30 minutes after as_of.
    world = gap_world(tmp_path, session, release_at=et(D, 20, 30))
    assert scan(session, con, world.root, close_of(D), config(tmp_path)) == []


def test_a_split_record_published_later_does_not_skip_the_candidate(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection
) -> None:
    # A split inside the lookback whose record only becomes available after as_of: a reader at
    # as_of did not know it, so the candidate stands.
    world = gap_world(tmp_path, session)
    write_splits(
        world.root,
        [("GAPX", SESSIONS[HISTORY - 5], 1.0, 2.0)],
        available_at=(close_of(D) + timedelta(days=3)).astimezone(UTC),
    )
    assert len(scan(session, con, world.root, close_of(D), config(tmp_path))) == 1


def test_bars_after_as_of_do_not_change_the_candidate(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection
) -> None:
    wild = [(500.0, 900.0, 1.0, 800.0, 90_000_000)] * 3
    world = gap_world(tmp_path, session, after=wild)
    [c] = scan(session, con, world.root, close_of(D), config(tmp_path))
    assert (c.gap_open, c.gap_high, c.gap_low, c.gap_close) == (55.0, 57.0, 54.0, 56.0)
    assert c.rvol == pytest.approx(3.0)
    assert c.atr == pytest.approx(2.0)
    assert c.avg_dollar_volume == pytest.approx(50_000_000.0)


def test_another_symbol_gapping_without_a_filing_is_not_a_candidate(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection
) -> None:
    # NOFL gaps exactly like GAPX on D, is in the master with a CIK, but filed nothing: the
    # filing, not the gap, starts a candidate.
    world = gap_world(
        tmp_path,
        session,
        extra_bars=[bar("NOFL", d, QUIET) for d in SESSIONS[:HISTORY]]
        + [bar("NOFL", D, (55.0, 57.0, 54.0, 56.0, 3_000_000))],
        extra_segs=[Seg("S_NOFL", "NOFL", 3333)],
    )
    [c] = scan(session, con, world.root, close_of(D), config(tmp_path))
    assert c.symbol == "GAPX"


# =============================================================================================
# identity: the filing's own CIK, matched per security (reviewer H3), and the previous close
# =============================================================================================


def test_a_filing_made_under_an_old_cik_still_finds_its_security(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection
) -> None:
    # XOM shape (real): filings until 2026-07 under CIK 34088, the entity today carries 2115436.
    # The master's segment for the gap day still says 34088.
    world = gap_world(tmp_path, session, entity_cik=7777, filer_cik=GAPX_CIK)
    [c] = scan(session, con, world.root, close_of(D), config(tmp_path))
    assert c.security_id == "S_GAPX"


def test_today_cik_alone_does_not_match_an_old_filing(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection
) -> None:
    # The same world without the archive URL falls back to the entity's CIK of today, which the
    # master did not carry on the gap day: no candidate rather than a guess.
    world = gap_world(tmp_path, session, entity_cik=7777)
    assert scan(session, con, world.root, close_of(D), config(tmp_path)) == []


def test_a_renamed_security_is_found_through_its_earlier_segment(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection
) -> None:
    # SQ -> XYZ shape (real): XYZ from 2025-01-21, its CIK link dated 2025-03-19, SQ's known since
    # 2020. The gap day falls between: the new segment's CIK is not yet knowable, the security is.
    rename = SESSIONS[HISTORY - 10]
    world = gap_world(
        tmp_path,
        session,
        gapx_segs=[
            Seg(
                "S_GAPX",
                "OLDX",
                GAPX_CIK,
                KNOWN_EARLY,
                valid_to=rename,
                end_available_at=et(rename, 0),
            ),
            Seg(
                "S_GAPX",
                "GAPX",
                GAPX_CIK,
                close_of(D) + timedelta(days=30),
                valid_from=rename,
                available_at=et(rename, 0),
            ),
        ],
        old_symbol=("OLDX", rename),
        filer_cik=GAPX_CIK,
    )
    [c] = scan(session, con, world.root, close_of(D), config(tmp_path))
    assert (c.security_id, c.symbol) == ("S_GAPX", "GAPX")
    assert c.pre_earnings_close == 50.0


def test_no_bar_on_the_previous_session_means_no_candidate(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection
) -> None:
    # A halt the day before: the previous row is two sessions old, so its close is not the
    # pre-earnings close and the gap would be measured against the wrong price.
    world = gap_world(tmp_path, session, missing_history=[PREV])
    assert scan(session, con, world.root, close_of(D), config(tmp_path)) == []
