"""Guards against the backfill silently discarding bars Alpaca returned: SEC validity windows
may decide what to *request*, never what to *keep*. Only two filters remain at write time: a
bar not yet final at fetch time, and a duplicate (symbol, t), of which the last one wins.

Target API (asof.ingest.bars after the 4b-2b change):

    write_partition(path, bars, timeframe, fetched_at) -> int        # no ``windows`` parameter
    _to_table(bars, timeframe, fetched_at) -> pa.Table                # no ``windows`` parameter
    _fetch_partition(client, partition, symbols, batch_size, fetched_at, windows, active)
    backfill(client, root, timeframe, start, end, symbols, *, force, batch_size, now,
             on_partition, windows, active) -> BackfillReport         # ``windows`` = request hint

Fake data provenance:
- BRK.B: the SEC ticker table spells it ``BRK-B``; normalize_ticker folds that to ``BRK.B``,
  but older SEC snapshots in our history only pick it up from 2020-06, so its SEC window
  starts 2020-06-01 while Alpaca has bars from 2020-01-02 (observed: BRK.B had no 2020-01..05
  rows after the 4a backfill; design 4b-2b §3 and §9 case 6).
- MMC -> MRSH 2026-01-14 (Alpaca ``name_change``, measured 2026-10-06): Alpaca serves the
  whole history under the newest name when asked for it alone.
"""

from collections.abc import Iterator, Sequence
from datetime import UTC, date, datetime
from pathlib import Path

import duckdb
import pyarrow.parquet as pq
import pytest

from asof.ingest.alpaca import Bar
from asof.ingest.bars import ET, backfill, write_partition
from asof.store.market import visible_bars
from tests.ingest.fakes import BarsCall, FakeBarClient, daily

NOW_2027 = datetime(2027, 1, 15, tzinfo=UTC)
BRKB_WINDOW = {"BRK.B": (date(2020, 6, 1), None)}
BRKB_2020 = [
    daily("BRK.B", date(2020, 1, 2), close=226.0),
    daily("BRK.B", date(2020, 3, 23), close=162.0),
    daily("BRK.B", date(2020, 5, 29), close=185.0),
    daily("BRK.B", date(2020, 6, 1), close=187.0),
    daily("BRK.B", date(2020, 12, 31), close=232.0),
]


def sessions(path: Path) -> list[tuple[str, date]]:
    rows = pq.read_table(path).to_pylist()
    return [(r["symbol"], r["session_date"]) for r in rows]


# --- normal path: bars outside the SEC window are written ------------------------------------


def test_bars_before_the_sec_window_are_written(tmp_path: Path) -> None:
    # BRK.B's SEC window starts 2020-06-01; Alpaca answers with bars from 2020-01-02. All five
    # must land in the 2020 file (before this change the first three were dropped).
    client = FakeBarClient(BRKB_2020)

    report = backfill(
        client,
        tmp_path,
        "1Day",
        date(2020, 1, 1),
        date(2020, 12, 31),
        ["BRK.B"],
        now=lambda: NOW_2027,
        windows=BRKB_WINDOW,
        active={"BRK.B"},
    )

    assert report.rows == 5
    assert sessions(tmp_path / "bars_1d" / "year=2020" / "bucket=B.parquet") == [
        ("BRK.B", date(2020, 1, 2)),
        ("BRK.B", date(2020, 3, 23)),
        ("BRK.B", date(2020, 5, 29)),
        ("BRK.B", date(2020, 6, 1)),
        ("BRK.B", date(2020, 12, 31)),
    ]


def test_write_partition_no_longer_accepts_windows(tmp_path: Path) -> None:
    path = tmp_path / "bucket=B.parquet"

    with pytest.raises(TypeError):
        write_partition(path, BRKB_2020, "1Day", NOW_2027, windows=BRKB_WINDOW)  # type: ignore[call-arg]

    assert write_partition(path, BRKB_2020, "1Day", NOW_2027) == 5


def test_history_served_under_a_newer_name_is_written_as_returned(tmp_path: Path) -> None:
    # Alpaca, asked for MRSH alone, hands back MMC's 2026-01 history labelled MRSH. The raw
    # store keeps it under MRSH; attributing it to a security is the security master's job.
    class ChainClient(FakeBarClient):
        def iter_bars(
            self, symbols: Sequence[str], timeframe: str, start: datetime, end: datetime
        ) -> Iterator[Bar]:
            self.calls.append(BarsCall(tuple(symbols), timeframe, start, end))
            wanted = set(symbols)
            out: list[Bar] = []
            for bar in self.bars:
                if not start <= bar.t <= end:
                    continue
                if bar.symbol in wanted:
                    out.append(bar)
                elif bar.symbol == "MMC" and "MRSH" in wanted:
                    out.append(bar.model_copy(update={"symbol": "MRSH"}))
            if {"MMC", "MRSH"} <= wanted:
                out = [b for b in out if b.symbol != "MMC"]
            return iter(out)

    client = ChainClient(
        [
            daily("MMC", date(2026, 1, 5)),
            daily("MMC", date(2026, 1, 13)),
            daily("MRSH", date(2026, 1, 14)),
        ]
    )
    # SEC first lists MRSH in its 2026-02-02 snapshot, so its window opens at the snapshot
    # before, 2026-01-10 here: the relabelled 2026-01-05 bar lies outside it and used to be lost.
    windows = {"MMC": (date(2018, 1, 22), date(2026, 2, 2)), "MRSH": (date(2026, 1, 10), None)}

    report = backfill(
        client,
        tmp_path,
        "1Day",
        date(2026, 1, 1),
        date(2026, 12, 31),
        ["MMC", "MRSH"],
        now=lambda: NOW_2027,
        windows=windows,
        active={"MRSH"},
    )

    assert report.rows == 5
    assert sessions(tmp_path / "bars_1d" / "year=2026" / "bucket=M.parquet") == [
        ("MMC", date(2026, 1, 5)),
        ("MMC", date(2026, 1, 13)),
        ("MRSH", date(2026, 1, 5)),
        ("MRSH", date(2026, 1, 13)),
        ("MRSH", date(2026, 1, 14)),
    ]


# --- boundaries: the two remaining write-time filters -----------------------------------------


def test_bars_not_final_at_fetch_time_are_still_dropped(tmp_path: Path) -> None:
    # Fetched 2020-06-01 10:00 New York, during the session: that day's bar is not final.
    fetched_at = datetime(2020, 6, 1, 10, 0, tzinfo=ET)
    client = FakeBarClient(BRKB_2020)

    report = backfill(
        client,
        tmp_path,
        "1Day",
        date(2020, 1, 1),
        date(2020, 12, 31),
        ["BRK.B"],
        now=lambda: fetched_at,
        windows=BRKB_WINDOW,
    )

    assert report.rows == 3
    assert [d for _, d in sessions(tmp_path / "bars_1d" / "year=2020" / "bucket=B.parquet")] == [
        date(2020, 1, 2),
        date(2020, 3, 23),
        date(2020, 5, 29),
    ]


def test_duplicate_symbol_and_time_keeps_the_last_one_even_outside_the_window(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bucket=B.parquet"
    bars = [
        daily("BRK.B", date(2020, 1, 2), close=1.0),
        daily("BRK.B", date(2020, 1, 2), close=226.0),
    ]

    count = write_partition(path, bars, "1Day", NOW_2027)

    assert count == 1
    assert pq.read_table(path).column("close").to_pylist() == [226.0]


def test_window_outside_the_period_still_skips_the_request_but_keeps_what_comes_back(
    tmp_path: Path,
) -> None:
    # ELNK's window ended in 2019: it is not asked for in 2020. BRK.B is asked for and every
    # bar it returns is stored, including the five months before its SEC window opens.
    client = FakeBarClient(BRKB_2020 + [daily("ELNK", date(2020, 1, 2))])
    windows = {**BRKB_WINDOW, "ELNK": (date(2018, 1, 22), date(2019, 6, 1))}

    report = backfill(
        client,
        tmp_path,
        "1Day",
        date(2020, 1, 1),
        date(2020, 12, 31),
        ["BRK.B", "ELNK"],
        now=lambda: NOW_2027,
        windows=windows,
        active={"BRK.B"},
    )

    requested = {s for c in client.calls for s in c.symbols}
    assert requested == {"BRK.B"}
    assert report.rows == 5
    assert sessions(tmp_path / "bars_1d" / "year=2020" / "bucket=B.parquet")[0] == (
        "BRK.B",
        date(2020, 1, 2),
    )
    assert not (tmp_path / "bars_1d" / "year=2020" / "bucket=E.parquet").exists() or (
        pq.read_table(tmp_path / "bars_1d" / "year=2020" / "bucket=E.parquet").num_rows == 0
    )


def test_bars_outside_the_partition_period_are_still_not_filed_there(tmp_path: Path) -> None:
    # Dropping the window filter must not loosen the partition filter: a 2019 session that a
    # sloppy server returns with the 2020 request does not go into the 2020 file.
    client = FakeBarClient(BRKB_2020 + [daily("BRK.B", date(2019, 12, 31))], respect_range=False)

    report = backfill(
        client,
        tmp_path,
        "1Day",
        date(2020, 1, 1),
        date(2020, 12, 31),
        ["BRK.B"],
        now=lambda: NOW_2027,
        windows=BRKB_WINDOW,
    )

    assert report.rows == 5
    assert date(2019, 12, 31) not in [
        d for _, d in sessions(tmp_path / "bars_1d" / "year=2020" / "bucket=B.parquet")
    ]


# --- leakage: a kept out-of-window bar obeys the same availability rule as any other ----------


def test_out_of_window_bar_is_invisible_before_its_post_market_close(tmp_path: Path) -> None:
    # The 2020-01-02 BRK.B bar is now on disk although its SEC window had not opened. It is
    # still only knowable from 20:00 New York on 2020-01-02, like every other daily bar.
    root = tmp_path / "market"
    client = FakeBarClient(BRKB_2020)
    backfill(
        client,
        root,
        "1Day",
        date(2020, 1, 1),
        date(2020, 12, 31),
        ["BRK.B"],
        now=lambda: NOW_2027,
        windows=BRKB_WINDOW,
    )
    con = duckdb.connect()
    try:
        before = visible_bars(con, root, "1Day", as_of=datetime(2020, 1, 2, 19, 59, tzinfo=ET))
        after = visible_bars(con, root, "1Day", as_of=datetime(2020, 1, 2, 20, 0, tzinfo=ET))
    finally:
        con.close()

    assert before.num_rows == 0
    assert after.column("session_date").to_pylist() == [date(2020, 1, 2)]
    assert after.column("available_at").to_pylist() == [datetime(2020, 1, 3, 1, 0, tzinfo=UTC)]
