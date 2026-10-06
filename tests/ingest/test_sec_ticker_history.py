"""Guards against a universe that only knows today's tickers: delisted names must come from
dated SEC snapshots, and each ticker must stay attached to the snapshot date that saw it."""

import itertools
import json
import string
from datetime import date
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from asof.ingest.sec_ticker_history import (
    CDX_URL,
    EXCHANGE_TARGET,
    SNAPSHOT_TARGET,
    SecTickerRow,
    build_history,
    download_snapshot,
    historical_symbols,
    list_snapshots,
    otc_only_symbols,
    parse_snapshot,
    snapshot_url,
    symbol_windows,
    write_history,
)
from tests.ingest.fakes import ScriptedFetch, gzipped, raw_reply, reply

TS = "20200113041413"
CDX_PARAMS = {
    "url": SNAPSHOT_TARGET,
    "output": "json",
    "from": "2018",
    "filter": "statuscode:200",
    "collapse": "timestamp:6",
    "fl": "timestamp",
}


def entry(cik: int, ticker: str, title: str) -> dict[str, Any]:
    return {"cik_str": cik, "ticker": ticker, "title": title}


def big_snapshot(n: int = 1000) -> dict[str, dict[str, Any]]:
    """A snapshot with ``n`` distinct, valid, alphabetic tickers (the real file has ~10k)."""
    names = ("".join(letters) for letters in itertools.product(string.ascii_uppercase, repeat=3))
    return {
        str(i): entry(1_000_000 + i, ticker, f"Company {ticker}")
        for i, ticker in zip(range(n), names, strict=False)
    }


SMALL = {
    "0": entry(320193, "AAPL", "Apple Inc."),
    "1": entry(1067983, "BRK-B", "BERKSHIRE HATHAWAY INC"),
    "2": entry(719739, "SIVB", "SVB FINANCIAL GROUP"),
}


def leftovers(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.rglob("*") if p.is_file())


# --- normal path -------------------------------------------------------------------------------


def test_list_snapshots_queries_the_cdx_index_and_drops_the_header_row() -> None:
    fetch = ScriptedFetch([reply(200, [["timestamp"], [TS], ["20200201093000"]])])

    assert list_snapshots(fetch) == [TS, "20200201093000"]
    assert fetch.calls == [(CDX_URL, CDX_PARAMS)]


def test_list_snapshots_respects_since_year() -> None:
    fetch = ScriptedFetch([reply(200, [["timestamp"], ["20220101000000"]])])

    list_snapshots(fetch, since_year=2022)

    assert fetch.params[0]["from"] == "2022"


def test_snapshot_url_points_at_the_raw_archived_file() -> None:
    assert snapshot_url(TS) == f"http://web.archive.org/web/{TS}id_/https://{SNAPSHOT_TARGET}"


def test_download_writes_the_snapshot_under_its_date(tmp_path: Path) -> None:
    payload = big_snapshot()
    fetch = ScriptedFetch([reply(200, payload)])

    path = download_snapshot(TS, tmp_path, fetch)

    assert path == tmp_path / "20200113.json"
    assert json.loads(path.read_bytes()) == payload
    assert fetch.calls == [(snapshot_url(TS), {})]
    assert leftovers(tmp_path) == ["20200113.json"]


def test_download_gunzips_a_compressed_body(tmp_path: Path) -> None:
    payload = big_snapshot()
    fetch = ScriptedFetch([raw_reply(200, gzipped(payload))])

    path = download_snapshot(TS, tmp_path, fetch)

    assert json.loads(path.read_bytes()) == payload


def test_parse_snapshot_normalizes_tickers_and_keeps_cik_as_int() -> None:
    rows = parse_snapshot(json.dumps(SMALL).encode(), date(2020, 1, 13))

    assert rows == [
        SecTickerRow(date(2020, 1, 13), "AAPL", 320193, "Apple Inc."),
        SecTickerRow(date(2020, 1, 13), "BRK.B", 1067983, "BERKSHIRE HATHAWAY INC"),
        SecTickerRow(date(2020, 1, 13), "SIVB", 719739, "SVB FINANCIAL GROUP"),
    ]
    assert all(type(r.cik) is int for r in rows)


def test_build_history_reads_every_dated_file_sorted_by_date_then_ticker(tmp_path: Path) -> None:
    (tmp_path / "20200113.json").write_text(json.dumps(SMALL))
    (tmp_path / "20230601.json").write_text(
        json.dumps(
            {
                "0": entry(320193, "AAPL", "Apple Inc."),
                "1": entry(1418091, "TWTR", "Twitter, Inc."),
                "2": entry(1067983, "BRK-B", "BERKSHIRE HATHAWAY INC"),
            }
        )
    )

    table = build_history(tmp_path)

    assert table.schema.names == ["snapshot_date", "ticker", "cik", "name"]
    assert table.schema.field("snapshot_date").type == pa.date32()
    assert table.schema.field("ticker").type == pa.string()
    assert table.schema.field("cik").type == pa.int64()
    assert table.schema.field("name").type == pa.string()
    assert list(
        zip(
            table.column("snapshot_date").to_pylist(),
            table.column("ticker").to_pylist(),
            table.column("cik").to_pylist(),
            strict=True,
        )
    ) == [
        (date(2020, 1, 13), "AAPL", 320193),
        (date(2020, 1, 13), "BRK.B", 1067983),
        (date(2020, 1, 13), "SIVB", 719739),
        (date(2023, 6, 1), "AAPL", 320193),
        (date(2023, 6, 1), "BRK.B", 1067983),
        (date(2023, 6, 1), "TWTR", 1418091),
    ]


def test_write_history_round_trips_through_parquet(tmp_path: Path) -> None:
    (tmp_path / "raw" / "sec").mkdir(parents=True)
    (tmp_path / "raw" / "sec" / "20200113.json").write_text(json.dumps(SMALL))
    table = build_history(tmp_path / "raw" / "sec")
    target = tmp_path / "market" / "symbols" / "sec_history.parquet"

    write_history(table, target)

    assert pq.read_table(target).equals(table)
    assert leftovers(target.parent) == ["sec_history.parquet"]


def test_historical_symbols_is_the_union_over_all_snapshots(tmp_path: Path) -> None:
    (tmp_path / "20200113.json").write_text(json.dumps(SMALL))
    (tmp_path / "20230601.json").write_text(
        json.dumps({"0": entry(320193, "AAPL", "Apple Inc."), "1": entry(1, "TWTR", "Twitter")})
    )

    assert historical_symbols(build_history(tmp_path)) == {"AAPL", "BRK.B", "SIVB", "TWTR"}


# --- boundaries --------------------------------------------------------------------------------


def test_empty_cdx_answer_gives_no_snapshots() -> None:
    assert list_snapshots(ScriptedFetch([reply(200, [])])) == []


def test_existing_snapshot_is_not_downloaded_again(tmp_path: Path) -> None:
    existing = tmp_path / "20200113.json"
    existing.write_text("already here")
    fetch = ScriptedFetch()

    assert download_snapshot(TS, tmp_path, fetch) == existing
    assert existing.read_text() == "already here"
    assert fetch.calls == []


def test_two_timestamps_on_the_same_day_share_one_file(tmp_path: Path) -> None:
    payload = big_snapshot()
    fetch = ScriptedFetch([reply(200, payload)])

    first = download_snapshot("20200113041413", tmp_path, fetch)
    second = download_snapshot("20200113235959", tmp_path, fetch)

    assert first == second
    assert len(fetch.calls) == 1


def test_implausibly_small_snapshot_is_refused_and_nothing_is_written(tmp_path: Path) -> None:
    fetch = ScriptedFetch([reply(200, big_snapshot(999))])

    with pytest.raises(ValueError):
        download_snapshot(TS, tmp_path, fetch)

    assert leftovers(tmp_path) == []


@pytest.mark.parametrize(
    "body",
    [b"<html>Wayback Machine error</html>", b"", b"[]", json.dumps(["not", "a", "dict"]).encode()],
    ids=["html", "empty", "empty_list", "list"],
)
def test_non_dict_body_is_refused_and_nothing_is_written(tmp_path: Path, body: bytes) -> None:
    fetch = ScriptedFetch([raw_reply(200, body)])

    with pytest.raises(ValueError):
        download_snapshot(TS, tmp_path, fetch)

    assert leftovers(tmp_path) == []


def test_parse_drops_rows_whose_ticker_is_not_a_valid_symbol() -> None:
    raw = {
        "0": entry(320193, "AAPL", "Apple Inc."),
        "1": entry(1, "CIC_DELISTED", "noise"),
        "2": entry(2, "384CNT069", "noise"),
        "3": entry(3, "", "blank"),
    }

    rows = parse_snapshot(json.dumps(raw).encode(), date(2020, 1, 13))

    assert [r.ticker for r in rows] == ["AAPL"]


def test_parse_of_an_empty_snapshot_gives_no_rows() -> None:
    assert parse_snapshot(b"{}", date(2020, 1, 13)) == []


def test_build_history_of_an_empty_directory_has_the_schema_and_no_rows(tmp_path: Path) -> None:
    table = build_history(tmp_path)

    assert table.num_rows == 0
    assert table.schema.names == ["snapshot_date", "ticker", "cik", "name"]


def test_build_history_ignores_files_that_are_not_dated_snapshots(tmp_path: Path) -> None:
    (tmp_path / "20200113.json").write_text(json.dumps(SMALL))
    (tmp_path / "README.txt").write_text("not a snapshot")
    (tmp_path / "20200113.json.tmp").write_text("half written")

    assert build_history(tmp_path).num_rows == len(SMALL)


def test_build_history_drops_invalid_tickers_like_parse_snapshot(tmp_path: Path) -> None:
    (tmp_path / "20200113.json").write_text(
        json.dumps({"0": entry(320193, "AAPL", "Apple Inc."), "1": entry(1, "384CNT069", "x")})
    )

    assert build_history(tmp_path).column("ticker").to_pylist() == ["AAPL"]


def test_write_history_overwrites_atomically(tmp_path: Path) -> None:
    (tmp_path / "20200113.json").write_text(json.dumps(SMALL))
    table = build_history(tmp_path)
    target = tmp_path / "out" / "sec_history.parquet"

    write_history(table, target)
    write_history(table, target)

    assert pq.read_table(target).equals(table)
    assert leftovers(target.parent) == ["sec_history.parquet"]


def test_historical_symbols_of_an_empty_history_is_empty(tmp_path: Path) -> None:
    assert historical_symbols(build_history(tmp_path)) == set()


# --- leakage -----------------------------------------------------------------------------------


def test_a_ticker_first_listed_later_is_not_attributed_to_an_earlier_snapshot(
    tmp_path: Path,
) -> None:
    # Both snapshots exist on disk; a reader filtering on snapshot_date <= 2020-01-13 must not
    # see RIVN, which only entered the SEC table in the 2023 snapshot.
    (tmp_path / "20200113.json").write_text(json.dumps(SMALL))
    (tmp_path / "20230601.json").write_text(
        json.dumps(
            {"0": entry(320193, "AAPL", "Apple Inc."), "1": entry(1874178, "RIVN", "Rivian")}
        )
    )
    table = build_history(tmp_path)

    as_of = date(2020, 1, 13)
    visible = {
        ticker
        for snapshot_date, ticker in zip(
            table.column("snapshot_date").to_pylist(),
            table.column("ticker").to_pylist(),
            strict=True,
        )
        if snapshot_date <= as_of
    }

    assert "RIVN" not in visible
    assert visible == {"AAPL", "BRK.B", "SIVB"}
    assert "RIVN" in historical_symbols(table)


def test_a_reused_ticker_keeps_both_ciks_with_their_dates(tmp_path: Path) -> None:
    # BBBY was Bed Bath & Beyond until 2023 and a different registrant later. The history
    # must keep both rows apart by date instead of collapsing the ticker to one company.
    (tmp_path / "20200113.json").write_text(
        json.dumps({"0": entry(886158, "BBBY", "BED BATH & BEYOND INC")})
    )
    (tmp_path / "20260105.json").write_text(json.dumps({"0": entry(1, "BBBY", "New Registrant")}))

    table = build_history(tmp_path)

    assert list(
        zip(
            table.column("snapshot_date").to_pylist(),
            table.column("ticker").to_pylist(),
            table.column("cik").to_pylist(),
            strict=True,
        )
    ) == [(date(2020, 1, 13), "BBBY", 886158), (date(2026, 1, 5), "BBBY", 1)]


# --- validity windows and exchange snapshots ---------------------------------------------------


def test_symbol_windows_span_from_the_snapshot_before_first_sighting_to_the_one_after_last(
    tmp_path: Path,
) -> None:
    # Snapshots: Jan, Feb, Apr, Jul 2020. TWTR appears Feb..Apr, so it existed at some point
    # after the Jan snapshot and vanished before the Jul one; those are its bounds.
    (tmp_path / "20200110.json").write_text(json.dumps({"0": entry(1, "AAPL", "Apple")}))
    (tmp_path / "20200210.json").write_text(
        json.dumps({"0": entry(1, "AAPL", "Apple"), "1": entry(2, "TWTR", "Twitter")})
    )
    (tmp_path / "20200410.json").write_text(
        json.dumps({"0": entry(1, "AAPL", "Apple"), "1": entry(2, "TWTR", "Twitter")})
    )
    (tmp_path / "20200710.json").write_text(json.dumps({"0": entry(1, "AAPL", "Apple")}))

    windows = symbol_windows(build_history(tmp_path))

    assert windows["TWTR"] == (date(2020, 1, 10), date(2020, 7, 10))
    # Present in the first and the latest snapshot: open at both ends we can know about.
    assert windows["AAPL"] == (date(2020, 1, 10), None)


def test_symbol_windows_of_an_empty_history_is_empty(tmp_path: Path) -> None:
    assert symbol_windows(build_history(tmp_path)) == {}


def test_exchange_snapshot_download_and_otc_only_detection(tmp_path: Path) -> None:
    payload = {
        "fields": ["cik", "name", "ticker", "exchange"],
        "data": [[1, "Fannie Mae", "FNMA", "OTC"], [2, "Apple", "AAPL", "Nasdaq"]]
        + [[10 + i, f"Co {i}", f"ZZ{i:03d}".replace("0", "A"), "NYSE"] for i in range(1000)],
    }
    payload2 = {
        "fields": ["cik", "name", "ticker", "exchange"],
        "data": [
            [1, "Fannie Mae", "FNMA", "OTC"],
            [3, "Sorrento", "SRNE", "Nasdaq"],
            [4, "X", "NOEX", ""],
        ]
        + [[10 + i, f"Co {i}", f"ZZ{i:03d}".replace("0", "A"), "NYSE"] for i in range(1000)],
    }
    payload3 = {
        "fields": ["cik", "name", "ticker", "exchange"],
        "data": [[3, "Sorrento", "SRNE", "OTC"], [4, "X", "NOEX", "OTC"]]
        + [[10 + i, f"Co {i}", f"ZZ{i:03d}".replace("0", "A"), "NYSE"] for i in range(1000)],
    }
    fetch = ScriptedFetch([reply(200, payload), reply(200, payload2), reply(200, payload3)])
    directory = tmp_path / "exchange"

    for ts in ["20210701000000", "20220101000000", "20230101000000"]:
        download_snapshot(ts, directory, fetch, target=EXCHANGE_TARGET)

    assert fetch.urls[0] == snapshot_url("20210701000000", target=EXCHANGE_TARGET)
    assert "company_tickers_exchange.json" in fetch.urls[0]
    otc = otc_only_symbols(directory)
    assert "FNMA" in otc  # OTC in every snapshot
    assert "SRNE" not in otc  # was on Nasdaq once, later OTC: keep
    assert "NOEX" in otc  # blank then OTC: never seen on an exchange
    assert "AAPL" not in otc


def test_exchange_snapshot_with_wrong_shape_is_refused(tmp_path: Path) -> None:
    fetch = ScriptedFetch([reply(200, big_snapshot())])  # the company_tickers shape, not this one

    with pytest.raises(ValueError):
        download_snapshot(TS, tmp_path, fetch, target=EXCHANGE_TARGET)

    assert leftovers(tmp_path) == []
