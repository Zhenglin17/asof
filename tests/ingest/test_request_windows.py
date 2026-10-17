"""Guards against the backfill never asking for a symbol that is still listed because its SEC
window closed: the SEC company table has no ETFs, so an ETF that an SEC snapshot once carried by
accident gets a window that ends when the snapshot series stops carrying it, and from then on
the backfill never requests it again (measured 2026-10-16: NOBL was in SEC snapshots only
2019-10 .. 2020-06; Alpaca lists it as active; the store holds only its 2020 bars; 197 active
symbols share this shape). ``windows_for_requests`` drops the window of every currently listed
symbol so that it is requested in every period; closed windows of delisted names still skip the
requests that cannot return anything.

Target API (asof.ingest.bars; implementation pending):

    windows_for_requests(windows: Mapping[str, Window], active: Collection[str])
        -> dict[str, Window]

This module has no ``as_of`` of its own: it shapes requests, and every bar it brings in obeys
the usual availability rule (asserted once below through ``visible_bars``). What the fakes
assume about the real data must be checked on the real store:

- After a backfill with the new windows, ``asof market status`` / a DuckDB count of NOBL rows
  per year must show 2021 .. 2026, not 2020 alone (basis: measured 2026-10-16, the 197 active
  symbols with a closed SEC window; recount after the backfill, expect close to 0 of them with
  no bar after their window's end).
- ``asof market audit`` C2 (listed asset with stale bars) is the standing check: a listed symbol
  whose last bar is more than 10 sessions old must be excepted with a reason or fixed.
- The request count must not grow for delisted names: a symbol inactive in Alpaca with a closed
  window (ELNK) must still produce no request in a year after its window (assumption: Alpaca's
  ``active`` list is the right notion of "still listed"; spot-check a few inactive names with
  bars in the store's last year).
"""

from datetime import UTC, date, datetime
from pathlib import Path

import duckdb
import pytest

from asof.ingest.bars import ET, Window, backfill, windows_for_requests
from asof.store.market import visible_bars
from tests.ingest.fakes import FakeBarClient, daily

NOW_2027 = datetime(2027, 1, 15, tzinfo=UTC)
# NOBL: in SEC snapshots 2019-10-02 .. 2020-06-30 only (measured 2026-10-16), active in Alpaca.
# ELNK: delisted 2019 (measured 2026-10-06 shape, used by test_bars as well), inactive.
# AAPL: open window, active.
WINDOWS: dict[str, Window] = {
    "NOBL": (date(2019, 10, 2), date(2020, 7, 1)),
    "ELNK": (date(2018, 1, 22), date(2019, 6, 1)),
    "AAPL": (date(2018, 1, 22), None),
}
ACTIVE = {"NOBL", "AAPL"}
NOBL_2021 = [
    daily("NOBL", date(2021, 1, 4), close=86.0),
    daily("NOBL", date(2021, 6, 1), close=92.0),
]


def requested(client: FakeBarClient) -> set[str]:
    return {s for c in client.calls for s in c.symbols}


def run_2021(client: FakeBarClient, root: Path, windows: dict[str, Window]) -> None:
    backfill(
        client,
        root,
        "1Day",
        date(2021, 1, 1),
        date(2021, 12, 31),
        ["AAPL", "ELNK", "NOBL"],
        now=lambda: NOW_2027,
        windows=windows,
        active=ACTIVE,
    )


# --- normal path ------------------------------------------------------------------------------


def test_active_symbols_lose_their_window_and_the_rest_are_unchanged() -> None:
    out = windows_for_requests(WINDOWS, ACTIVE)

    assert out == {"ELNK": (date(2018, 1, 22), date(2019, 6, 1))}


def test_returns_a_new_dict_and_leaves_the_input_alone() -> None:
    windows = dict(WINDOWS)

    out = windows_for_requests(windows, ACTIVE)

    assert out is not windows
    assert windows == WINDOWS
    out["ZZZ"] = (date(2020, 1, 1), None)
    assert "ZZZ" not in windows


def test_listed_symbol_with_a_closed_window_is_requested_after_the_window(tmp_path: Path) -> None:
    # Measured 2026-10-16: with the raw windows NOBL was never requested after 2020.
    client = FakeBarClient(NOBL_2021)

    run_2021(client, tmp_path, windows_for_requests(WINDOWS, ACTIVE))

    assert "NOBL" in requested(client)
    assert "ELNK" not in requested(client)  # delisted and its window is closed: still skipped
    stored = duckdb.connect()
    try:
        table = visible_bars(stored, tmp_path, "1Day", NOW_2027, symbols=["NOBL"], resolve=False)
    finally:
        stored.close()
    assert table.column("session_date").to_pylist() == [date(2021, 1, 4), date(2021, 6, 1)]


def test_raw_windows_still_skip_the_listed_symbol_documenting_the_old_behaviour(
    tmp_path: Path,
) -> None:
    # The old behaviour, kept as a contrast: NOBL's SEC window ends 2020-07-01, so a 2021
    # partition never asks for it (measured 2026-10-16: 197 active symbols like NOBL).
    client = FakeBarClient(NOBL_2021)

    run_2021(client, tmp_path, WINDOWS)

    assert requested(client) == {"AAPL"}


# --- boundaries -------------------------------------------------------------------------------


def test_active_symbols_absent_from_the_windows_are_a_no_op() -> None:
    assert windows_for_requests(WINDOWS, {"SPY", "NOBL"}) == {
        "ELNK": WINDOWS["ELNK"],
        "AAPL": WINDOWS["AAPL"],
    }


def test_empty_inputs() -> None:
    assert windows_for_requests({}, set()) == {}
    assert windows_for_requests({}, {"NOBL"}) == {}
    assert windows_for_requests(WINDOWS, set()) == WINDOWS
    assert windows_for_requests(WINDOWS, frozenset(WINDOWS)) == {}


def test_accepts_any_collection_for_active() -> None:
    assert windows_for_requests(WINDOWS, ["NOBL", "AAPL"]) == {"ELNK": WINDOWS["ELNK"]}
    assert windows_for_requests(WINDOWS, ("NOBL",)) == {
        "ELNK": WINDOWS["ELNK"],
        "AAPL": WINDOWS["AAPL"],
    }


def test_request_count_does_not_grow_for_inactive_closed_windows(tmp_path: Path) -> None:
    # Only the listed symbols are asked for; ELNK adds no request of its own, not even the
    # windows-driven second pass (it has a window, but it never overlaps 2021).
    client = FakeBarClient(NOBL_2021)

    run_2021(client, tmp_path, windows_for_requests(WINDOWS, ACTIVE))

    assert all("ELNK" not in c.symbols for c in client.calls)
    assert len(client.calls) == 2  # one active batch per bucket (A, N); no second pass


# --- leakage ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("as_of", "visible"),
    [
        (datetime(2021, 1, 4, 19, 59, tzinfo=ET), []),
        (datetime(2021, 1, 4, 20, 0, tzinfo=ET), [date(2021, 1, 4)]),
    ],
    ids=["before_post_market_close", "at_post_market_close"],
)
def test_bar_requested_thanks_to_the_listing_obeys_the_availability_rule(
    tmp_path: Path, as_of: datetime, visible: list[date]
) -> None:
    # The 2021-01-04 NOBL bar is on disk only because the window was lifted; it is still
    # knowable from 20:00 New York of its session, like every other daily bar (docs: ingest.bars).
    client = FakeBarClient(NOBL_2021)
    run_2021(client, tmp_path, windows_for_requests(WINDOWS, ACTIVE))

    con = duckdb.connect()
    try:
        table = visible_bars(con, tmp_path, "1Day", as_of, symbols=["NOBL"], resolve=False)
    finally:
        con.close()

    assert table.column("session_date").to_pylist() == visible
