# Guards the batched market filter of the B-Xtrender range scan against disagreeing with the
# nightly ``market_ok`` -- in particular against look-ahead from reading the index once at the
# window's end (a split record, a bar or a label holder known only later deciding an earlier
# night) -- and against re-reading the index's whole history once per night.
"""API under test (asof.scan.bx_trender):

    market_ok_nights(con, root, nights: Sequence[date], flt: MarketFilter) -> dict[date, bool]
        reads ``flt.symbol``'s bars and split records once (as known at the close of the last
        night) and returns, for every night d, exactly ``market_ok(con, root, _close(d), flt)``:
        closes final at 20:00 New York of d, adjusted only by split records with available_at <=
        that close, the security carrying the symbol that night, too short a history -> False.
        Empty ``nights`` -> {}.
    scan_nights / pool_range call it once per window when cfg.market_filter is set; without a
        filter every night is open and the index is not read.

Fake data: the index world of tests/scan/test_bx_filters.py (``world``, ``spy_bars``: a made-up
"SPY", class etf, hand-checkable closes; stocks from tests/scan/bx_world.py, provenance there).
Bar availability 20:00 New York of the session and split availability 00:00 New York of the
ex-date are the ingest rules (asof.ingest.bars, corporate_actions); records with another
available_at are constructed (a late feed). The label-reuse world is modelled on a measured case:
PARA was Paramount's until 2025-08-07 and Banzai's afterwards (tests/store/test_golden_cases.py,
test_ffr_and_para_were_reused_by_unrelated_securities); here the old holder's segment ends and
the new one's starts on the same session, both known from 00:00 New York of it. No network.

Assumptions a real-data check must confirm:
- On the real store, for the dev window (2020-01 .. 2025-06), market_ok_nights over every
  session equals market_ok per night for SPY (a one-off audit run; expected 0 differing nights).
  The constructed late split and label reuse here do not occur for SPY in reality (SPY has no
  split record and one holder), so only that equality on real data shows nothing else differs.
- The real SPY history has no session missing inside the window (counted by the audit named in
  tests/scan/test_bx_filters.py); a gap would be read identically by both paths, but would make
  the filter's nights differ from the hand SMA.
"""

from collections.abc import Callable, Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pytest

import asof.scan.bx_trender as bx
from asof.scan.bx_trender import (
    MarketFilter,
    market_ok,
    pool_range,
    scan_nights,
)
from asof.store.market import market_root
from tests.scan.bx_world import CAL, after, close_of, known, split_row
from tests.scan.test_bx_filters import (
    FILTER,
    SPY,
    WINDOW_END,
    WINDOW_START,
    cfg_of,
    flt,
    peak,
    ref_ok,
    rising,
    world,
)
from tests.scan.test_bx_trender import AAA, BASE, assert_same_candidates
from tests.store.audit_world import bar as raw_bar
from tests.store.test_spinoffs import write_bars
from tests.store.test_universe import Seg, write_classes, write_master

WINDOW = [d for d in CAL if WINDOW_START <= d <= WINDOW_END]
ALL = list(CAL)


def market_ok_nights(
    con: duckdb.DuckDBPyConnection, root: Path, nights: Sequence[date], f: MarketFilter
) -> dict[date, bool]:
    """The function under test, looked up at call time so a missing one fails each test with
    an AttributeError instead of the whole module failing to import."""
    return bx.market_ok_nights(con, root, nights, f)  # type: ignore[attr-defined]


def per_night(
    con: duckdb.DuckDBPyConnection, root: Path, nights: Sequence[date], f: MarketFilter
) -> dict[date, bool]:
    """The reference: the nightly filter, one full read per night."""
    return {d: market_ok(con, root, close_of(d), f) for d in nights}


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return market_root(tmp_path)


# =============================================================================================
# normal path
# =============================================================================================


def test_every_session_equals_the_nightly_filter(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    world(root, [AAA], peak)
    expected = per_night(con, root, ALL, flt())
    # fixture: the filter is off before 200 closes, on while rising, off after the turn
    assert expected[CAL[100]] is False and expected[CAL[250]] is True
    assert expected[CAL[-1]] is False

    assert market_ok_nights(con, root, ALL, flt()) == expected


def test_a_short_window_inside_history_equals_the_nightly_filter(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # only the window's nights are asked for, yet each one reads all history before it
    world(root, [AAA], peak)
    got = market_ok_nights(con, root, WINDOW, flt())
    assert got == per_night(con, root, WINDOW, flt())
    assert got == {d: ref_ok(peak, CAL.index(d)) for d in WINDOW}
    assert any(got.values()) and not all(got.values())  # fixture: it turns inside the window


def test_scan_nights_and_pool_range_are_unchanged(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    world(root, BASE, peak)
    filtered, plain = cfg_of(market_filter=FILTER), cfg_of()
    open_ = per_night(con, root, WINDOW, flt())

    got = scan_nights(con, root, WINDOW_START, WINDOW_END, filtered)
    ref = scan_nights(con, root, WINDOW_START, WINDOW_END, plain)
    assert [d for d, _, _ in got] == WINDOW
    # fixture: a closed night with a candidate the filter must remove
    assert any(c and not open_[d] for d, c, _ in ref)
    for (d, found, rows), (_, ref_found, ref_rows) in zip(got, ref, strict=True):
        assert_same_candidates(found, ref_found if open_[d] else [])
        assert rows.equals(ref_rows), d  # the panel rows do not depend on the filter

    pools = pool_range(con, root, WINDOW_START, WINDOW_END, filtered)
    plain_pools = pool_range(con, root, WINDOW_START, WINDOW_END, plain)
    assert pools == {d: plain_pools[d] if open_[d] else frozenset() for d in WINDOW}


# =============================================================================================
# boundaries
# =============================================================================================


def test_nights_before_the_window_fills_are_false(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # CAL[198] has 199 closes (False), CAL[199] the 200th (True, rising)
    world(root, [AAA], rising)
    nights = CAL[0:3] + CAL[195:205]
    got = market_ok_nights(con, root, nights, flt())
    assert got == per_night(con, root, nights, flt())
    assert [got[d] for d in nights if d <= CAL[198]] == [False] * 7
    assert got[CAL[199]] is True


def test_a_night_before_any_index_bar_is_false(
    con: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    # 2022-12-30 precedes the first stored session (2023-01-03): no SPY bar yet. A store
    # without the symbol at all is False on every night.
    with_spy, without = market_root(tmp_path / "with"), market_root(tmp_path / "without")
    world(with_spy, [AAA], rising)
    world(without, [AAA], None)
    nights = [date(2022, 12, 30), CAL[0], CAL[250]]

    got = market_ok_nights(con, with_spy, nights, flt(n=1))
    assert got == per_night(con, with_spy, nights, flt(n=1))
    assert got[date(2022, 12, 30)] is False
    assert market_ok_nights(con, without, nights, flt()) == dict.fromkeys(nights, False)
    assert market_ok_nights(con, without, [CAL[250]], flt(symbol="QQQ")) == {CAL[250]: False}


def test_no_nights_is_an_empty_answer(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    world(root, [AAA], rising)
    assert market_ok_nights(con, root, [], flt()) == {}


def test_a_day_that_is_not_a_session_reads_as_market_ok_at_its_close(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # A Saturday and a holiday (Good Friday 2024-03-29) inside the history, and a weekday past
    # the store's last session: each reads the last final close before it, as market_ok does.
    world(root, [AAA], peak)
    saturday = next(d for d in (CAL[300] + timedelta(days=k) for k in range(7)) if d.weekday() == 5)
    holiday = date(2024, 3, 29)
    past_end = CAL[-1] + timedelta(days=3)
    nights = sorted([CAL[250], saturday, holiday, CAL[330], past_end])
    assert saturday not in CAL and holiday not in CAL and past_end not in CAL  # fixture

    assert market_ok_nights(con, root, nights, flt()) == per_night(con, root, nights, flt())


# =============================================================================================
# leakage: data that exists in the store but is not yet visible on an earlier night
# =============================================================================================


SPLIT_EX = CAL[300]  # inside WINDOW
SPLIT_SEEN = CAL[310]  # the record is published an hour after this night's close


def late_split_world(root: Path, secs: Sequence[Any] = (AAA,)) -> None:
    """RISING with every raw close before SPLIT_EX doubled (a 2-for-1 in the raw prices), and
    the split record published only after SPLIT_SEEN's close (a constructed late feed). Before
    the record is visible, the post-ex closes (~130) sit under a mean pulled up to ~250: False;
    once visible, the adjusted series is RISING again: True."""
    late = close_of(SPLIT_SEEN) + timedelta(hours=1)
    world(
        root,
        secs,
        rising,
        scale_before=(SPLIT_EX, 2.0),
        splits=[split_row(SPY, SPLIT_EX, 1.0, 2.0, available_at=late)],
    )


def test_a_split_recorded_later_does_not_adjust_earlier_nights(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    late_split_world(root)
    expected = per_night(con, root, WINDOW, flt())
    # fixture: the nights between the ex-date and the record are closed; after it, open again.
    # Adjusting every night by the splits known at the window's end would open them.
    assert [expected[d] for d in WINDOW if SPLIT_EX <= d <= SPLIT_SEEN] == [False] * 11
    assert all(expected[d] for d in WINDOW if d > SPLIT_SEEN)

    assert market_ok_nights(con, root, WINDOW, flt()) == expected


def test_scan_nights_does_not_apply_a_later_split_record_early(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    late_split_world(root, BASE)
    filtered, plain = cfg_of(market_filter=FILTER), cfg_of()
    open_ = per_night(con, root, WINDOW, flt())

    got = scan_nights(con, root, WINDOW_START, WINDOW_END, filtered)
    ref = scan_nights(con, root, WINDOW_START, WINDOW_END, plain)
    for (d, found, _), (_, ref_found, _) in zip(got, ref, strict=True):
        assert_same_candidates(found, ref_found if open_[d] else [])
    pools = pool_range(con, root, WINDOW_START, WINDOW_END, filtered)
    plain_pools = pool_range(con, root, WINDOW_START, WINDOW_END, plain)
    assert pools == {d: plain_pools[d] if open_[d] else frozenset() for d in WINDOW}


def test_the_next_sessions_bar_does_not_decide_tonight(
    con: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    # The index crashes to 1.0 on the session after D (inside the window), and D's own bar is
    # published an hour late (D reads the session before it). A batch read at the window's end
    # holds both bars; neither may reach D.
    day = CAL[295]
    edited, cut = market_root(tmp_path / "edited"), market_root(tmp_path / "cut")
    world(
        edited,
        [AAA],
        peak,
        edits={after(day, 1): 1.0},
        available={day: close_of(day) + timedelta(hours=1)},
    )
    world(cut, [AAA], peak, cut=day)

    got = market_ok_nights(con, edited, WINDOW, flt())
    assert got == per_night(con, edited, WINDOW, flt())
    assert got[day] is True and got[after(day, 1)] is False  # fixture: the crash is seen next
    for d in WINDOW:
        if d <= day:
            assert got[d] == market_ok(con, cut, close_of(d), flt()), d


OLD_HOLDER_UNTIL = CAL[300]  # the old holder's last session is CAL[299]


def old_level(i: int) -> float:
    return rising(i)


def new_level(i: int) -> float:
    return round(500 + 1.0 * i, 4)


def reused_label_world(root: Path) -> None:
    """SPY is S_OLD's (RISING, ~130) through CAL[299] and S_NEW's (500 + i, rising) from
    CAL[300]; the new segment is known from 00:00 New York of CAL[300]. Reading at the window's
    end, both segments are visible."""
    rows = [
        raw_bar(
            SPY, d, close=old_level(i) if d < OLD_HOLDER_UNTIL else new_level(i), volume=5_000_000
        )
        for i, d in enumerate(CAL)
    ]
    segs = [
        Seg(
            "S_OLD",
            SPY,
            valid_to=OLD_HOLDER_UNTIL,
            end_available_at=known(OLD_HOLDER_UNTIL),
            cls="etf",
        ),
        Seg(
            "S_NEW",
            SPY,
            valid_from=OLD_HOLDER_UNTIL,
            available_at=known(OLD_HOLDER_UNTIL),
            cls="etf",
        ),
    ]
    rows += AAA.bars()
    segs += AAA.segs()
    write_bars(root, rows)
    write_master(root, segs)
    write_classes(root, segs)


def test_each_night_reads_the_label_holder_of_that_night(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # n = 5. Before CAL[300] the holder is S_OLD (rising, long history): True. On CAL[300..303]
    # S_NEW has 1..4 closes: too short, False -- mixing S_OLD's closes in would read True
    # (500 > ~130), and reading tonight's holder on earlier nights would read False. From
    # CAL[304] S_NEW has 5 rising closes: True.
    reused_label_world(root)
    f = flt(n=5)
    expected = per_night(con, root, WINDOW, f)
    i0 = CAL.index(OLD_HOLDER_UNTIL)
    assert all(expected[d] for d in WINDOW if d < OLD_HOLDER_UNTIL)  # fixture
    assert [expected[CAL[i]] for i in range(i0, i0 + 4)] == [False] * 4
    assert all(expected[d] for d in WINDOW if d >= CAL[i0 + 4])

    assert market_ok_nights(con, root, WINDOW, f) == expected


# =============================================================================================
# reads: the index is read once per window, not once per night
# =============================================================================================


def count_index_reads(monkeypatch: pytest.MonkeyPatch, symbol: str = SPY) -> dict[str, int]:
    """Wrap the readers the scan module imported; count calls that can return the index (its
    symbol asked for, or no symbol filter at all)."""
    counts = {"bars": 0, "splits": 0}

    def reads_index(symbols: Sequence[str] | None) -> bool:
        return symbols is None or symbol in symbols

    def wrap(name: str, key: str, symbols_of: Callable[..., Any]) -> None:
        original = getattr(bx, name)

        def counted(*args: Any, **kwargs: Any) -> Any:
            if reads_index(symbols_of(*args, **kwargs)):
                counts[key] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(bx, name, counted)

    wrap("visible_bars", "bars", lambda *a, **k: k.get("symbols"))
    wrap("visible_splits", "splits", lambda *a, **k: k.get("symbols", a[3] if len(a) > 3 else None))
    return counts


THIRTY = CAL[290:320]


def test_scan_nights_reads_the_index_at_most_twice_over_thirty_nights(
    con: duckdb.DuckDBPyConnection, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world(root, BASE, peak)
    assert len(THIRTY) == 30
    counts = count_index_reads(monkeypatch)

    nights = scan_nights(con, root, THIRTY[0], THIRTY[-1], cfg_of(market_filter=FILTER))

    assert len(nights) == 30
    assert counts["bars"] >= 1  # the counter sees the filter's reads at all
    assert counts["bars"] <= 2, counts
    assert counts["splits"] <= 2, counts


def test_pool_range_reads_the_index_at_most_twice_over_thirty_nights(
    con: duckdb.DuckDBPyConnection, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world(root, BASE, peak)
    counts = count_index_reads(monkeypatch)

    pools = pool_range(con, root, THIRTY[0], THIRTY[-1], cfg_of(market_filter=FILTER))

    assert len(pools) == 30
    assert 1 <= counts["bars"] <= 2, counts
    assert counts["splits"] <= 2, counts


def test_market_ok_nights_reads_the_index_once(
    con: duckdb.DuckDBPyConnection, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world(root, [AAA], peak)
    counts = count_index_reads(monkeypatch)

    got = market_ok_nights(con, root, THIRTY, flt())

    assert len(got) == 30
    assert counts["bars"] == 1, counts
    assert counts["splits"] <= 1, counts


def test_without_a_filter_the_index_is_not_read(
    con: duckdb.DuckDBPyConnection, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world(root, BASE, peak)
    counts = count_index_reads(monkeypatch)

    nights = scan_nights(con, root, THIRTY[0], THIRTY[-1], cfg_of())

    assert len(nights) == 30
    assert counts == {"bars": 0, "splits": 0}
