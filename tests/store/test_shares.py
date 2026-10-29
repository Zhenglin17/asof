"""Guards against share counts and market caps that use what a reader on the night could not
know: a fact before its filing was public, a split recorded later, a CIK not yet tied to the
security or one it carries only later, and a higher-priority concept that becomes public later
displacing the one already visible. Also guards the choice itself (priority, latest end,
deterministic ties), the split adjustment window ``end < ex_date <= night``, the age cut-off
and the cap thresholds.

API (asof.store.shares):

    visible_shares(con, root, as_of, security_ids, *, max_age_days) -> pa.Table
        columns SHARE_COLUMNS below; one row per security with an eligible fact
    market_caps(con, root, as_of, security_ids, *, max_age_days) -> pa.Table
        SHARE_COLUMNS + close, raw_cap
    cap_passes(caps, *, min_cap, min_raw_cap_ratio_unknown) -> frozenset[str]

The night is the New York calendar date of ``as_of``. Facts are written with
asof.ingest.shares.write_share_facts, so these tests also prove the file round-trips.

Fake data provenance:
- Share facts are constructed. Their ``available_at`` follows the ingest rule (00:00 New York
  of filed + 1 day; see tests/ingest/test_shares.py for its basis).
- The AAPL-like split case uses the real shape: AAPL's 10-Q filed 2020-07-31 reported
  4,275,634,000 shares as of 2020-07-17 (the cover count, real per SEC companyconcept; not
  re-fetched here), and the 4-for-1 split went ex on 2020-08-31 (real). Bars are constructed.
- The CIK change follows XOM moving to its holding company on 2026-07-16 (real mechanism, see
  asof.ingest.edgar); CIKs, dates and counts here are constructed.
- Split records and master segments use the builders of tests/store/audit_world.py.

What only real data can confirm (tests/store/test_golden_shares.py; ``asof market audit`` F1 /
F2): that the cover count is pre-split when the split comes after the cover date (AAPL 2020,
NVDA 2024), that 20-F filers report ordinary shares, not ADSs (TSM), and that 40-F filers
report in the listed share (TECK).
"""

import itertools
import shutil
from collections import defaultdict
from collections.abc import Iterator, Sequence
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from asof.ingest.bars import bucket_of
from asof.ingest.corporate_actions import SPLIT_SCHEMA
from asof.ingest.shares import ShareFact, write_share_facts
from asof.store.market import market_root
from asof.store.shares import cap_passes, implausible, market_caps, visible_shares
from tests.store.audit_world import START, Seg, bar, split, write_master
from tests.store.test_market import KNOWN_EARLY, known
from tests.store.test_universe import store_raw

NY = ZoneInfo("America/New_York")
DEI = "dei:EntityCommonStockSharesOutstanding"
GAAP = "us-gaap:WeightedAverageNumberOfSharesOutstandingBasic"
IFRS = "ifrs-full:WeightedAverageShares"
SHARE_COLUMNS = [
    "security_id",
    "symbol",
    "cik",
    "concept",
    "form",
    "end",
    "filed",
    "shares",
    "split_factor",
    "age_days",
    "ratio_unknown",
]
CAP_COLUMNS = [*SHARE_COLUMNS, "close", "raw_cap"]
NIGHT = date(2024, 6, 28)  # a Friday session
AGE = 400  # max_age_days wherever the age is not under test

_accn = itertools.count(1)


# =============================================================================================
# builders
# =============================================================================================


def evening(day: date, hour: int = 21, minute: int = 0) -> datetime:
    """``hour:minute`` New York on ``day``: after 20:00 the day's bar is visible."""
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=NY)


def midnight(day: date) -> datetime:
    return datetime.combine(day, time(0), tzinfo=NY)


def fact(
    cik: int,
    end: date,
    filed: date,
    shares: float,
    *,
    concept: str = DEI,
    form: str = "10-Q",
    accn: str | None = None,
) -> ShareFact:
    """A fact as the ingest stores it: available from 00:00 New York of filed + 1 day."""
    return ShareFact(
        cik=cik,
        concept=concept,
        accn=accn or f"{cik:010d}-24-{next(_accn):06d}",
        form=form,
        end=end,
        filed=filed,
        shares=float(shares),
        available_at=midnight(filed + timedelta(days=1)).astimezone(UTC),
    )


def make_store(
    root: Path,
    segs: Sequence[Seg],
    facts: Sequence[ShareFact] | None,
    *,
    splits: Sequence[dict[str, Any]] = (),
    bars: Sequence[dict[str, Any]] = (),
) -> Path:
    """Master, share facts (None: no file at all), split records and daily bars."""
    shutil.rmtree(root, ignore_errors=True)
    write_master(root, segs)
    if facts is not None:
        write_share_facts(list(facts), root)
    if splits:
        path = root / "corporate_actions" / "splits.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(list(splits), schema=SPLIT_SCHEMA), path)
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for b in bars:
        grouped[(b["session_date"].year, bucket_of(b["symbol"]))].append(b)
    for (year, bucket), group in grouped.items():
        store_raw(root, group, year, bucket)
    return root


def one(table: pa.Table) -> dict[str, Any]:
    rows = table.to_pylist()
    assert len(rows) == 1, rows
    return rows[0]


def by_sid(table: pa.Table) -> dict[str, dict[str, Any]]:
    return {r["security_id"]: r for r in table.to_pylist()}


def aaa(**kw: Any) -> Seg:
    return Seg("S1", "AAA", cik=1001, **kw)


@pytest.fixture
def con() -> Iterator[duckdb.DuckDBPyConnection]:
    connection = duckdb.connect()
    yield connection
    connection.close()


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return market_root(tmp_path)


def shares_at(
    con: duckdb.DuckDBPyConnection,
    root: Path,
    as_of: datetime,
    ids: Sequence[str] = ("S1",),
    max_age_days: int = AGE,
) -> pa.Table:
    return visible_shares(con, root, as_of, list(ids), max_age_days=max_age_days)


# =============================================================================================
# visible_shares: normal
# =============================================================================================


def test_the_latest_end_of_the_top_concept_is_returned_with_every_column(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    make_store(
        root,
        [aaa()],
        [
            fact(1001, date(2024, 1, 19), date(2024, 2, 2), 0.9e9),
            fact(1001, date(2024, 4, 19), date(2024, 5, 3), 1.0e9, accn="0000001001-24-000100"),
        ],
    )

    table = shares_at(con, root, evening(NIGHT))

    assert table.schema.names == SHARE_COLUMNS
    assert one(table) == {
        "security_id": "S1",
        "symbol": "AAA",
        "cik": 1001,
        "concept": DEI,
        "form": "10-Q",
        "end": date(2024, 4, 19),
        "filed": date(2024, 5, 3),
        "shares": 1.0e9,
        "split_factor": 1.0,
        "age_days": 70,  # 2024-06-28 - 2024-04-19
        "ratio_unknown": False,
    }


def test_latest_end_wins_over_a_later_filed_comparative(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # the 10-Q of August repeats January's count as a comparative; April's is still the newest
    make_store(
        root,
        [aaa()],
        [
            fact(1001, date(2024, 4, 19), date(2024, 5, 3), 1.0e9),
            fact(1001, date(2024, 1, 19), date(2024, 8, 2), 0.9e9),
        ],
    )
    row = one(shares_at(con, root, evening(date(2024, 8, 30))))
    assert (row["end"], row["shares"]) == (date(2024, 4, 19), 1.0e9)


def test_dei_wins_over_a_newer_us_gaap_fact(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    make_store(
        root,
        [aaa()],
        [
            fact(1001, date(2024, 1, 19), date(2024, 2, 2), 0.9e9, concept=DEI),
            fact(1001, date(2024, 3, 31), date(2024, 5, 3), 1.0e9, concept=GAAP),
        ],
    )
    row = one(shares_at(con, root, evening(NIGHT)))
    assert (row["concept"], row["shares"]) == (DEI, 0.9e9)


def test_without_dei_us_gaap_is_used(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    make_store(
        root,
        [aaa()],
        [
            fact(1001, date(2024, 3, 31), date(2024, 5, 3), 1.0e9, concept=GAAP),
            fact(1001, date(2023, 12, 31), date(2024, 3, 1), 1.1e9, concept=IFRS, form="20-F"),
        ],
    )
    row = one(shares_at(con, root, evening(NIGHT)))
    assert (row["concept"], row["shares"]) == (GAAP, 1.0e9)


def test_without_dei_and_us_gaap_ifrs_is_used(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    make_store(
        root,
        [aaa()],
        [fact(1001, date(2023, 12, 31), date(2024, 3, 1), 1.55e9, concept=IFRS, form="20-F")],
    )
    row = one(shares_at(con, root, evening(NIGHT)))
    assert (row["concept"], row["shares"], row["ratio_unknown"]) == (IFRS, 1.55e9, True)


def test_a_too_old_dei_fact_falls_back_to_us_gaap(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    make_store(
        root,
        [aaa()],
        [
            fact(1001, date(2023, 1, 20), date(2023, 2, 3), 0.8e9, concept=DEI),  # 525 days old
            fact(1001, date(2024, 3, 31), date(2024, 5, 3), 1.0e9, concept=GAAP),
        ],
    )
    row = one(shares_at(con, root, evening(NIGHT), max_age_days=AGE))
    assert row["concept"] == GAAP


def test_a_split_after_the_cover_date_multiplies_the_count(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # AAPL-like: 4,275,634,000 shares as of 2020-07-17 (10-Q filed 07-31), 4:1 ex 2020-08-31
    make_store(
        root,
        [aaa()],
        [fact(1001, date(2020, 7, 17), date(2020, 7, 31), 4_275_634_000)],
        splits=[split("AAA", date(2020, 8, 31), 1.0, 4.0)],
    )

    before = one(shares_at(con, root, evening(date(2020, 8, 28))))
    after = one(shares_at(con, root, evening(date(2020, 9, 1))))

    assert (before["shares"], before["split_factor"]) == (4_275_634_000.0, 1.0)
    assert (after["shares"], after["split_factor"]) == (17_102_536_000.0, 4.0)
    assert after["end"] == date(2020, 7, 17)  # the fact itself is unchanged


def test_several_splits_multiply_and_a_reverse_split_divides(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    make_store(
        root,
        [aaa(), Seg("S2", "BBB", cik=2002)],
        [
            fact(1001, date(2024, 1, 19), date(2024, 2, 2), 1.0e9),
            fact(2002, date(2024, 1, 19), date(2024, 2, 2), 1.0e9),
        ],
        splits=[
            split("AAA", date(2024, 3, 1), 1.0, 2.0),
            split("AAA", date(2024, 5, 1), 1.0, 3.0),
            split("BBB", date(2024, 3, 1), 10.0, 1.0, kind="reverse_split"),
        ],
    )
    rows = by_sid(shares_at(con, root, evening(NIGHT), ["S1", "S2"]))
    assert rows["S1"]["split_factor"] == pytest.approx(6.0)
    assert rows["S1"]["shares"] == pytest.approx(6.0e9)
    assert rows["S2"]["split_factor"] == pytest.approx(0.1)
    assert rows["S2"]["shares"] == pytest.approx(1.0e8)


def test_a_split_under_the_securitys_earlier_name_applies(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    cut = date(2024, 3, 1)
    make_store(
        root,
        [
            Seg("S1", "OLD", START, cut, KNOWN_EARLY, known(cut), cik=1001),
            Seg("S1", "NEW", cut, None, known(cut), cik=1001),
        ],
        [fact(1001, date(2024, 1, 19), date(2024, 2, 2), 1.0e9)],
        splits=[split("OLD", date(2024, 2, 15), 1.0, 2.0)],
    )
    row = one(shares_at(con, root, evening(NIGHT)))
    assert (row["symbol"], row["split_factor"], row["shares"]) == ("NEW", 2.0, 2.0e9)


def test_a_split_of_the_labels_earlier_holder_does_not_apply(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # XYZ belonged to SB until 2024-01-02, then SC took the label; the 2023-12-01 split is SB's
    cut = date(2024, 1, 2)
    make_store(
        root,
        [
            Seg("SB", "XYZ", START, cut, KNOWN_EARLY, known(cut), cik=3003),
            Seg("SC", "XYZ", cut, None, known(cut), cik=4004),
        ],
        [fact(4004, date(2023, 11, 1), date(2023, 11, 10), 5.0e8)],
        splits=[split("XYZ", date(2023, 12, 1), 1.0, 2.0)],
    )
    row = one(shares_at(con, root, evening(NIGHT), ["SC"]))
    assert (row["split_factor"], row["shares"]) == (1.0, 5.0e8)


@pytest.mark.parametrize(
    ("form", "unknown"),
    [
        ("10-K", False),
        ("10-Q", False),
        ("10-K/A", False),
        ("10-Q/A", False),
        ("40-F", False),
        ("40-F/A", False),
        ("20-F", True),
        ("20-F/A", True),
    ],
)
def test_ratio_unknown_marks_the_20f_family_only(
    con: duckdb.DuckDBPyConnection, root: Path, form: str, unknown: bool
) -> None:
    make_store(root, [aaa()], [fact(1001, date(2023, 12, 31), date(2024, 3, 1), 1e9, form=form)])
    row = one(shares_at(con, root, evening(NIGHT)))
    assert (row["form"], row["ratio_unknown"]) == (form, unknown)


def test_only_the_requested_securities_are_returned(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    make_store(
        root,
        [aaa(), Seg("S2", "BBB", cik=2002), Seg("S3", "CCC", cik=3003)],
        [
            fact(1001, date(2024, 4, 19), date(2024, 5, 3), 1e9),
            fact(2002, date(2024, 4, 19), date(2024, 5, 3), 2e9),
            fact(3003, date(2024, 4, 19), date(2024, 5, 3), 3e9),
        ],
    )
    table = shares_at(con, root, evening(NIGHT), ["S1", "S3", "S404"])
    assert sorted(table.column("security_id").to_pylist()) == ["S1", "S3"]


def test_another_ciks_facts_are_not_used(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    make_store(root, [aaa()], [fact(9999, date(2024, 4, 19), date(2024, 5, 3), 1e9)])
    assert shares_at(con, root, evening(NIGHT)).num_rows == 0


# =============================================================================================
# visible_shares: boundaries
# =============================================================================================


def test_age_of_exactly_max_age_days_is_kept_one_more_is_dropped(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    end = NIGHT - timedelta(days=90)  # 2024-03-30
    make_store(root, [aaa()], [fact(1001, end, date(2024, 4, 15), 1e9)])

    kept = one(shares_at(con, root, evening(NIGHT), max_age_days=90))
    assert kept["age_days"] == 90
    assert shares_at(con, root, evening(NIGHT), max_age_days=89).num_rows == 0


def test_the_age_is_counted_from_the_new_york_night_not_the_utc_date(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # 23:30 New York on 06-28 is already 06-29 in UTC; the night is still 06-28
    end = NIGHT - timedelta(days=90)
    make_store(root, [aaa()], [fact(1001, end, date(2024, 4, 15), 1e9)])
    row = one(shares_at(con, root, evening(NIGHT, 23, 30), max_age_days=90))
    assert row["age_days"] == 90


def test_a_split_on_the_fact_end_is_not_applied(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    # the count as of the end date already reflects a split that went ex that day
    end = date(2024, 4, 19)
    make_store(
        root,
        [aaa()],
        [fact(1001, end, date(2024, 5, 3), 1e9)],
        splits=[split("AAA", end, 1.0, 2.0)],
    )
    assert one(shares_at(con, root, evening(NIGHT)))["split_factor"] == 1.0


def test_a_split_on_the_night_is_applied(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    make_store(
        root,
        [aaa()],
        [fact(1001, date(2024, 4, 19), date(2024, 5, 3), 1e9)],
        splits=[split("AAA", NIGHT, 1.0, 2.0)],
    )
    row = one(shares_at(con, root, evening(NIGHT)))
    assert (row["split_factor"], row["shares"]) == (2.0, 2e9)


def test_a_later_filing_breaks_a_tie_on_end(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    end = date(2024, 4, 19)
    make_store(
        root,
        [aaa()],
        [
            # the larger accn is filed earlier: the filing date decides first
            fact(1001, end, date(2024, 5, 3), 1.0e9, accn="0000001001-24-000900"),
            fact(1001, end, date(2024, 8, 2), 1.1e9, accn="0000001001-24-000100"),
        ],
    )
    row = one(shares_at(con, root, evening(date(2024, 8, 30))))
    assert (row["filed"], row["shares"]) == (date(2024, 8, 2), 1.1e9)


def test_the_largest_accn_breaks_a_tie_on_end_and_filed(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    end, filed = date(2024, 4, 19), date(2024, 5, 3)
    make_store(
        root,
        [aaa()],
        [
            fact(1001, end, filed, 1.3e9, accn="0000001001-24-000010"),
            fact(1001, end, filed, 1.2e9, accn="0000001001-24-000002"),
        ],
    )
    assert one(shares_at(con, root, evening(NIGHT)))["shares"] == 1.3e9


def test_the_tie_break_does_not_depend_on_file_order(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    end, filed = date(2024, 4, 19), date(2024, 5, 3)
    a = fact(1001, end, filed, 1.3e9, accn="0000001001-24-000010")
    b = fact(1001, end, filed, 1.2e9, accn="0000001001-24-000002")
    make_store(root, [aaa()], [b, a])
    first = one(shares_at(con, root, evening(NIGHT)))["shares"]
    make_store(root, [aaa()], [a, b])
    assert one(shares_at(con, root, evening(NIGHT)))["shares"] == first == 1.3e9


def test_a_security_without_an_eligible_fact_is_absent(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    make_store(
        root,
        [aaa(), Seg("S2", "BBB", cik=2002)],
        [fact(1001, date(2024, 4, 19), date(2024, 5, 3), 1e9)],
    )
    table = shares_at(con, root, evening(NIGHT), ["S1", "S2"])
    assert table.column("security_id").to_pylist() == ["S1"]


def test_a_segment_without_a_cik_has_no_count(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    make_store(
        root, [Seg("S1", "AAA", cik=None)], [fact(1001, date(2024, 4, 19), date(2024, 5, 3), 1e9)]
    )
    assert shares_at(con, root, evening(NIGHT)).num_rows == 0


def test_a_security_whose_segment_ended_before_the_night_is_absent(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    gone = date(2024, 6, 1)
    make_store(
        root,
        [Seg("S1", "AAA", START, gone, KNOWN_EARLY, known(gone), cik=1001)],
        [fact(1001, date(2024, 4, 19), date(2024, 5, 3), 1e9)],
    )
    assert shares_at(con, root, evening(NIGHT)).num_rows == 0


def test_no_shares_file_gives_an_empty_table_with_the_columns(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    make_store(root, [aaa()], None)
    table = shares_at(con, root, evening(NIGHT))
    assert table.num_rows == 0
    assert table.schema.names == SHARE_COLUMNS


def test_a_naive_as_of_is_rejected(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    make_store(root, [aaa()], [fact(1001, date(2024, 4, 19), date(2024, 5, 3), 1e9)])
    with pytest.raises(ValueError):
        shares_at(con, root, datetime(2024, 6, 28, 21, 0))


# =============================================================================================
# visible_shares: leakage (the data exists on disk but is not knowable yet)
# =============================================================================================


def test_a_fact_filed_on_d_is_hidden_that_evening_and_visible_at_midnight(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    filed = date(2024, 5, 3)
    make_store(
        root,
        [aaa()],
        [
            fact(1001, date(2024, 1, 19), date(2024, 2, 2), 0.9e9),
            fact(1001, date(2024, 4, 19), filed, 1.0e9),
        ],
    )

    evening_of_d = one(shares_at(con, root, evening(filed, 20)))
    last_minute = one(shares_at(con, root, evening(filed, 23, 59)))
    next_midnight = one(shares_at(con, root, midnight(filed + timedelta(days=1))))

    assert evening_of_d["shares"] == last_minute["shares"] == 0.9e9
    assert (next_midnight["shares"], next_midnight["end"]) == (1.0e9, date(2024, 4, 19))


def test_a_lone_fact_filed_on_d_does_not_exist_on_d(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    filed = date(2024, 5, 3)
    make_store(root, [aaa()], [fact(1001, date(2024, 4, 19), filed, 1.0e9)])
    assert shares_at(con, root, evening(filed, 20)).num_rows == 0
    assert shares_at(con, root, midnight(filed + timedelta(days=1))).num_rows == 1


def test_a_split_recorded_after_as_of_is_not_applied(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # the record of a split that went ex on the night arrives only the next morning
    late = {**split("AAA", NIGHT, 1.0, 2.0), "available_at": midnight(NIGHT + timedelta(days=1))}
    make_store(
        root,
        [aaa()],
        [fact(1001, date(2024, 4, 19), date(2024, 5, 3), 1e9)],
        splits=[late],
    )
    assert one(shares_at(con, root, evening(NIGHT)))["split_factor"] == 1.0
    assert one(shares_at(con, root, evening(NIGHT + timedelta(days=1))))["split_factor"] == 2.0


def test_a_future_split_is_not_applied(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    make_store(
        root,
        [aaa()],
        [fact(1001, date(2024, 4, 19), date(2024, 5, 3), 1e9)],
        splits=[split("AAA", NIGHT + timedelta(days=3), 1.0, 10.0)],
    )
    assert one(shares_at(con, root, evening(NIGHT)))["shares"] == 1e9


def test_a_cik_is_not_used_before_it_is_knowable(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    tied = date(2024, 7, 1)
    make_store(
        root,
        [aaa(cik_available_at=midnight(tied).astimezone(UTC))],
        [fact(1001, date(2024, 4, 19), date(2024, 5, 3), 1e9)],
    )
    assert shares_at(con, root, evening(NIGHT)).num_rows == 0
    assert one(shares_at(con, root, evening(tied)))["cik"] == 1001


def test_the_companys_later_cik_is_not_used_before_the_security_carries_it(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # XOM-like: the security moves to a holding company's CIK on 07-16; the holding company
    # filed a count before that
    move = date(2024, 7, 16)
    make_store(
        root,
        [
            Seg("S1", "AAA", START, move, KNOWN_EARLY, known(move), cik=1001),
            Seg("S1", "AAA", move, None, known(move), cik=2002),
        ],
        [
            fact(1001, date(2024, 4, 19), date(2024, 5, 3), 4.0e9),
            fact(1001, date(2024, 7, 1), date(2024, 7, 18), 4.1e9),  # old CIK, after the move
            fact(2002, date(2024, 6, 30), date(2024, 7, 10), 4.4e9),  # new CIK, before the move
        ],
    )

    before = one(shares_at(con, root, evening(date(2024, 7, 12))))
    after = one(shares_at(con, root, evening(date(2024, 7, 22))))

    assert (before["cik"], before["shares"]) == (1001, 4.0e9)
    assert (after["cik"], after["shares"]) == (2002, 4.4e9)


def test_a_dei_fact_public_later_does_not_displace_a_visible_us_gaap_fact(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    make_store(
        root,
        [aaa()],
        [
            fact(1001, date(2024, 3, 31), date(2024, 4, 30), 1.0e9, concept=GAAP),
            fact(1001, date(2024, 4, 19), date(2024, 7, 1), 1.1e9, concept=DEI),
        ],
    )
    assert one(shares_at(con, root, evening(NIGHT)))["concept"] == GAAP
    assert one(shares_at(con, root, evening(date(2024, 7, 3))))["concept"] == DEI


# =============================================================================================
# market_caps
# =============================================================================================


def caps_at(
    con: duckdb.DuckDBPyConnection,
    root: Path,
    as_of: datetime,
    ids: Sequence[str] = ("S1",),
) -> pa.Table:
    return market_caps(con, root, as_of, list(ids), max_age_days=AGE)


def test_market_cap_is_shares_times_the_nights_close(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    make_store(
        root,
        [aaa()],
        [fact(1001, date(2024, 4, 19), date(2024, 5, 3), 1e9)],
        bars=[
            bar("AAA", date(2024, 6, 27), close=48.0, volume=10_000),
            bar("AAA", NIGHT, close=50.0, volume=10_000),
        ],
    )

    table = caps_at(con, root, evening(NIGHT))

    assert table.schema.names == CAP_COLUMNS
    row = one(table)
    assert (row["close"], row["raw_cap"]) == (50.0, 5e10)
    assert row["shares"] == 1e9


def test_on_a_split_day_the_raw_close_meets_the_adjusted_count(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # AAPL-like 2020-08-31: raw close 129.04 after 4:1, cover count from before the split
    ex = date(2020, 8, 31)
    make_store(
        root,
        [aaa()],
        [fact(1001, date(2020, 7, 17), date(2020, 7, 31), 4_275_634_000)],
        splits=[split("AAA", ex, 1.0, 4.0)],
        bars=[bar("AAA", ex, close=129.04, volume=10_000)],
    )
    row = one(caps_at(con, root, evening(ex)))
    assert row["close"] == 129.04
    assert row["raw_cap"] == pytest.approx(17_102_536_000 * 129.04)


def test_a_security_without_a_bar_that_night_is_absent(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    make_store(
        root,
        [aaa(), Seg("S2", "BBB", cik=2002)],
        [
            fact(1001, date(2024, 4, 19), date(2024, 5, 3), 1e9),
            fact(2002, date(2024, 4, 19), date(2024, 5, 3), 1e9),
        ],
        bars=[
            bar("AAA", NIGHT, close=50.0, volume=10_000),
            bar("BBB", date(2024, 6, 27), close=20.0, volume=10_000),  # the day before only
        ],
    )
    table = caps_at(con, root, evening(NIGHT), ["S1", "S2"])
    assert table.column("security_id").to_pylist() == ["S1"]


def test_an_untraded_placeholder_bar_is_not_a_close(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    make_store(
        root,
        [aaa()],
        [fact(1001, date(2024, 4, 19), date(2024, 5, 3), 1e9)],
        bars=[bar("AAA", NIGHT, close=50.0, volume=0)],
    )
    assert caps_at(con, root, evening(NIGHT)).num_rows == 0


def test_a_security_without_a_count_is_absent_even_with_a_bar(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    make_store(
        root,
        [aaa()],
        [fact(9999, date(2024, 4, 19), date(2024, 5, 3), 1e9)],
        bars=[bar("AAA", NIGHT, close=50.0, volume=10_000)],
    )
    assert caps_at(con, root, evening(NIGHT)).num_rows == 0


def test_the_nights_bar_is_not_used_before_it_is_visible(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # the daily bar becomes visible at 20:00 New York; at 19:00 the session has no close yet
    make_store(
        root,
        [aaa()],
        [fact(1001, date(2024, 4, 19), date(2024, 5, 3), 1e9)],
        bars=[
            bar("AAA", date(2024, 6, 27), close=48.0, volume=10_000),
            bar("AAA", NIGHT, close=50.0, volume=10_000),
        ],
    )
    with pytest.raises(ValueError, match="not a session"):
        caps_at(con, root, evening(NIGHT, 19))  # an empty answer would read as "no count"
    assert one(caps_at(con, root, evening(NIGHT, 20)))["close"] == 50.0


# =============================================================================================
# cap_passes
# =============================================================================================


def caps_table(rows: Sequence[tuple[str, str, float, bool]]) -> pa.Table:
    """(security_id, form, raw_cap, ratio_unknown) in the shape market_caps returns."""
    full = [
        {
            "security_id": sid,
            "symbol": sid,
            "cik": 1000 + i,
            "concept": DEI,
            "form": form,
            "end": date(2024, 3, 31),
            "filed": date(2024, 5, 3),
            "shares": raw_cap / 10.0,
            "split_factor": 1.0,
            "age_days": 89,
            "ratio_unknown": unknown,
            "close": 10.0,
            "raw_cap": raw_cap,
        }
        for i, (sid, form, raw_cap, unknown) in enumerate(rows)
    ]
    return pa.Table.from_pylist(full)


def bars_of(caps: pa.Table, adv: float = 1e6) -> dict[str, Any]:
    """The thresholds of the cap filter, every security trading ``adv`` a day: far below its
    cap, so no cap here is implausible unless a test says so."""
    return {
        "adv_usd": {sid: adv for sid in caps.column("security_id").to_pylist()},
        "min_cap": 5e8,
        "min_raw_cap_ratio_unknown": 5e9,
        "max_turnover": 1.0,
        "max_cap": 5e12,
    }


def test_cap_passes_judges_domestic_40f_and_20f_filers() -> None:
    caps = caps_table(
        [
            ("DOM_BIG", "10-K", 6e8, False),
            ("DOM_SMALL", "10-Q", 4e8, False),
            ("CAN_40F", "40-F", 7e8, False),  # a 40-F count is in the listed share
            ("ADR_BIG", "20-F", 6e9, True),
            ("ADR_MID", "20-F", 3e9, True),  # above min_cap, below the 20-F bar
        ]
    )
    passed = cap_passes(caps, **bars_of(caps))
    assert passed == frozenset({"DOM_BIG", "CAN_40F", "ADR_BIG"})
    assert isinstance(passed, frozenset)


def test_a_cap_exactly_at_the_threshold_passes() -> None:
    caps = caps_table(
        [
            ("AT", "10-K", 5e8, False),
            ("BELOW", "10-K", 5e8 - 1, False),
            ("AT_20F", "20-F", 5e9, True),
            ("BELOW_20F", "20-F", 5e9 - 1, True),
        ]
    )
    assert cap_passes(caps, **bars_of(caps)) == frozenset({"AT", "AT_20F"})


def test_an_empty_table_passes_nothing() -> None:
    caps = caps_table([("X", "10-K", 1e12, False)]).slice(0, 0)
    assert cap_passes(caps, **bars_of(caps)) == frozenset()


def test_the_thresholds_are_required_keywords() -> None:
    caps = caps_table([("X", "10-K", 1e12, False)])
    with pytest.raises(TypeError):
        cap_passes(caps)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        cap_passes(caps, 5e8, 5e9)  # type: ignore[misc]
    with pytest.raises(TypeError):
        cap_passes(caps, min_cap=5e8)  # type: ignore[call-arg]


def test_cap_passes_reads_market_caps_output(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    make_store(
        root,
        [aaa(), Seg("S2", "BBB", cik=2002)],
        [
            fact(1001, date(2024, 4, 19), date(2024, 5, 3), 1e8),
            fact(2002, date(2023, 12, 31), date(2024, 4, 1), 1e9, form="20-F"),
        ],
        bars=[
            bar("AAA", NIGHT, close=10.0, volume=10_000),  # 1e9 domestic: passes 5e8
            bar("BBB", NIGHT, close=3.0, volume=10_000),  # 3e9 under 20-F: fails 5e9
        ],
    )
    caps = market_caps(con, root, evening(NIGHT), ["S1", "S2"], max_age_days=AGE)
    assert cap_passes(caps, **bars_of(caps)) == frozenset({"S1"})


def test_an_average_filed_after_a_split_is_not_multiplied_again(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # SHOP (measured 2026-10-29): the 2021 average was 0.12bn in the 10-K of 2022-02 and 1.25bn,
    # restated for the 10-for-1 of 2022-06-29, in the 10-K of 2023-02. Dates follow; counts round.
    make_store(
        root,
        [aaa()],
        [fact(1001, date(2021, 12, 31), date(2023, 2, 15), 1.25e9, concept=GAAP)],
        splits=[split("AAA", date(2022, 6, 29), 1.0, 10.0)],
    )
    row = one(shares_at(con, root, evening(date(2023, 6, 30)), max_age_days=730))
    assert (row["shares"], row["split_factor"]) == (1.25e9, 1.0)


def test_an_average_filed_before_a_split_is_multiplied(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    make_store(
        root,
        [aaa()],
        [fact(1001, date(2021, 12, 31), date(2022, 2, 16), 1.25e8, concept=GAAP)],
        splits=[split("AAA", date(2022, 6, 29), 1.0, 10.0)],
    )
    row = one(shares_at(con, root, evening(date(2022, 7, 29))))
    assert (row["shares"], row["split_factor"]) == (1.25e9, 10.0)


def test_a_cover_count_before_a_split_filed_after_it_is_multiplied(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # a cover count is as of its date, whenever it is filed: the split after that date applies
    make_store(
        root,
        [aaa()],
        [fact(1001, date(2022, 6, 24), date(2022, 7, 29), 1.25e8)],
        splits=[split("AAA", date(2022, 6, 29), 1.0, 10.0)],
    )
    row = one(shares_at(con, root, evening(date(2022, 8, 31))))
    assert (row["shares"], row["split_factor"]) == (1.25e9, 10.0)


def test_a_cap_below_a_days_dollar_volume_is_implausible() -> None:
    # VALE (measured 2026-10-29): an IFRS count of 5.18m, in thousands, gave a $61m cap to a
    # stock trading $305m a day
    caps = caps_table([("VALE", "20-F", 6.1e7, True), ("OK", "10-K", 6e8, False)])
    adv = {"VALE": 3.05e8, "OK": 3.05e8}
    assert implausible(caps, adv_usd=adv, max_turnover=1.0, max_cap=5e12) == frozenset({"VALE"})


def test_a_cap_above_the_ceiling_is_implausible() -> None:
    caps = caps_table([("BIG", "10-K", 5e12 + 1, False), ("AT", "10-K", 5e12, False)])
    adv = {"BIG": 1e8, "AT": 1e8}
    assert implausible(caps, adv_usd=adv, max_turnover=1.0, max_cap=5e12) == frozenset({"BIG"})


def test_turnover_exactly_at_the_limit_is_plausible() -> None:
    caps = caps_table([("AT", "10-K", 1e8, False), ("OVER", "10-K", 1e8 - 1, False)])
    adv = {"AT": 1e8, "OVER": 1e8}
    assert implausible(caps, adv_usd=adv, max_turnover=1.0, max_cap=5e12) == frozenset({"OVER"})


def test_a_cap_without_a_dollar_volume_is_an_error() -> None:
    caps = caps_table([("X", "10-K", 1e9, False)])
    with pytest.raises(ValueError, match="X"):
        implausible(caps, adv_usd={}, max_turnover=1.0, max_cap=5e12)


def test_an_implausible_cap_is_judged_like_a_missing_count() -> None:
    # HUGE: no usable count and 1e8 a day, under 5e8 -> out; SMALL_UNIT: a 6e8 cap trading 7e8 a
    # day is no count, but 7e8 a day >= 1.0 x 5e8 -> in by its volume
    caps = caps_table([("HUGE", "10-K", 9e12, False), ("SMALL_UNIT", "10-K", 6e8, False)])
    kw = bars_of(caps)
    kw["adv_usd"] = {"HUGE": 1e8, "SMALL_UNIT": 7e8}
    assert cap_passes(caps, **kw) == frozenset({"SMALL_UNIT"})


def test_a_candidate_without_a_count_passes_by_volume_alone() -> None:
    # V (measured 2026-10-29): no all-class count in any concept, ~$2bn traded a day
    caps = caps_table([("DOM", "10-K", 6e8, False)])
    kw = bars_of(caps)
    kw["adv_usd"] = {"DOM": 1e6, "V": 2e9, "AT": 5e8, "BELOW": 5e8 - 1}
    assert cap_passes(caps, **kw) == frozenset({"DOM", "V", "AT"})


def test_only_candidates_are_judged() -> None:
    # a cap without a candidate entry has no volume to judge it by: an error, never a pass
    caps = caps_table([("X", "10-K", 6e8, False)])
    kw = bars_of(caps)
    kw["adv_usd"] = {}
    with pytest.raises(ValueError, match="X"):
        cap_passes(caps, **kw)


def test_a_cap_on_a_day_without_a_session_is_an_error(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    make_store(
        root,
        [aaa()],
        [fact(1001, date(2024, 4, 19), date(2024, 5, 3), 1e9)],
        bars=[bar("AAA", NIGHT, close=50.0, volume=10_000)],
    )
    with pytest.raises(ValueError, match="not a session"):
        caps_at(con, root, evening(date(2024, 6, 29)))  # a Saturday


def test_a_quarter_wins_over_the_half_year_of_the_same_filing_in_either_order(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # a 10-Q tags the quarter's and the half-year's average with one end; after a May listing
    # the half-year average counts the thin months before it
    def avg(start: date, shares: float) -> ShareFact:
        f = fact(1001, date(2024, 6, 30), date(2024, 8, 2), shares, concept=GAAP, accn="a-q2")
        return ShareFact(**{**f.__dict__, "start": start})

    quarter, half = avg(date(2024, 4, 1), 9e8), avg(date(2024, 1, 1), 4e8)
    for order in ([quarter, half], [half, quarter]):
        make_store(root, [aaa()], order)
        row = one(shares_at(con, root, evening(date(2024, 8, 30))))
        assert row["shares"] == 9e8


def test_a_count_from_a_6k_of_a_20f_filer_is_ratio_unknown(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    make_store(
        root,
        [aaa()],
        [
            fact(1001, date(2023, 12, 31), date(2024, 4, 1), 1e9, form="20-F"),
            fact(1001, date(2024, 3, 31), date(2024, 5, 20), 1.1e9, form="6-K"),
        ],
    )
    row = one(shares_at(con, root, evening(NIGHT)))
    assert (row["form"], row["ratio_unknown"]) == ("6-K", True)


def test_a_6k_count_before_any_annual_report_is_ratio_unknown(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    make_store(root, [aaa()], [fact(1001, date(2024, 3, 31), date(2024, 5, 20), 1e9, form="6-K")])
    assert one(shares_at(con, root, evening(NIGHT)))["ratio_unknown"] is True


def test_a_company_moving_to_the_10k_is_domestic_once_its_10k_is_public(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    make_store(
        root,
        [aaa()],
        [
            fact(1001, date(2022, 12, 31), date(2023, 4, 1), 1e9, form="20-F"),
            fact(1001, date(2023, 12, 31), date(2024, 2, 20), 1e9, form="10-K"),
        ],
    )
    before = one(shares_at(con, root, evening(date(2024, 2, 20)), max_age_days=730))
    after = one(shares_at(con, root, evening(date(2024, 2, 21)), max_age_days=730))
    assert (before["ratio_unknown"], after["ratio_unknown"]) == (True, False)


BAL = "us-gaap:CommonStockSharesOutstanding"


def test_the_balance_sheet_count_is_the_last_resort_and_split_from_its_filing(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # GOOGL-like: only the balance-sheet count; restated like an average, so a split between
    # its end and its filing is already in it
    make_store(
        root,
        [aaa()],
        [fact(1001, date(2022, 6, 30), date(2022, 7, 27), 1.3e10, concept=BAL)],
        splits=[split("AAA", date(2022, 7, 18), 1.0, 20.0)],
    )
    row = one(shares_at(con, root, evening(date(2022, 8, 31))))
    assert (row["concept"], row["shares"], row["split_factor"]) == (BAL, 1.3e10, 1.0)


def test_any_other_concept_wins_over_the_balance_sheet_count(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    make_store(
        root,
        [aaa()],
        [
            fact(1001, date(2024, 3, 31), date(2024, 5, 3), 1.0e9, concept=BAL),
            fact(1001, date(2023, 12, 31), date(2024, 3, 1), 1.1e9, concept=IFRS, form="20-F"),
        ],
    )
    assert one(shares_at(con, root, evening(NIGHT)))["concept"] == IFRS
