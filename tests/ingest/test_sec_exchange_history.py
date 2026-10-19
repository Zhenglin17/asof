"""Guards against the SEC exchange snapshots being read into the wrong shape: a ticker spelled
differently from the rest of the store (BRK-B vs BRK.B), an exchange spelled in mixed case
("Nasdaq") that a later filter on "NASDAQ" would miss, a column order assumed instead of read
from ``fields``, or a stray file in the directory taken for a snapshot.

Target API (asof.ingest.sec_ticker_history; implementation pending, step 6-c):

    EXCHANGE_HISTORY_SCHEMA: snapshot_date date32, ticker string (normalize_ticker),
                             cik int64, exchange string (upper-case)
    EXCHANGE_HISTORY_PATH = "symbols/sec_exchange_history.parquet"   (under the market root)
    build_exchange_history(directory: Path) -> pa.Table
        every ``YYYYMMDD.json`` ({"fields": [...], "data": [[...], ...]}) as rows; a missing
        directory gives an empty table of that schema.

No ``as_of`` here: the function is a parser of dated files, and the date it attaches is the
file name's. Point-in-time reading happens in the audit (I10, tests/store/test_audit_i10.py).

Fake data provenance: the payload shape and the spellings below are the real ones (27 files
2021-07-23 .. 2026-08-01 under /data/asof/raw/sec/company_tickers_exchange, read 2026-10-19):
fields ["cik", "name", "ticker", "exchange"]; exchange values "Nasdaq" (120,878 rows), "NYSE"
(96,379), "OTC" (62,915), "" (5,848), null (5,078), "CBOE" (528), "Cboe US" (22), "NAS" (2);
Barrick 756894 GOLD NYSE until 2025-02-16 and B NYSE from 2025-05-31; Barnes 9984 B NYSE until
2024-12-07; Inhibrx 1739614 INBX Nasdaq until 2024-02-04, 2007919 from 2024-09-27.

Assumptions that need a check on the real files (they decide what I10 counts as "listed"):
 1. "Cboe US" (22 rows) and "NAS" (2 rows) are exchange listings that upper-casing alone turns
    into "CBOE US" / "NAS", which I10's {NASDAQ, NYSE, CBOE} would not count. Whether they
    should be mapped is not decided by the spec -> list the tickers carrying them on the real
    files before the first I10 run is read.
 2. Null and "" exchanges (10,926 rows, mostly funds and delisted names) are kept as "" or
    dropped; neither is a listing. Not asserted here beyond "not one of the three exchanges".
 3. The real files carry no other field order than the one above; the reordered-fields test is
    constructed (the code must still read ``fields``, as ``otc_only_symbols`` does).
 4. The file date is the snapshot date (the Wayback capture day); the audit treats a snapshot as
    knowable from 00:00 New York of that date (same assumption as C2's assets snapshots).
"""

import json
from datetime import date
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from asof.ingest import sec_ticker_history as sth

FIELDS = ["cik", "name", "ticker", "exchange"]


def snapshot(directory: Path, day: str, data: list[list[Any]], fields: list[str] = FIELDS) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{day}.json").write_text(json.dumps({"fields": fields, "data": data}))


def as_rows(table: pa.Table) -> list[tuple[date, str, int, str]]:
    return sorted(
        (r["snapshot_date"], r["ticker"], r["cik"], r["exchange"]) for r in table.to_pylist()
    )


def test_schema_and_path() -> None:
    assert sth.EXCHANGE_HISTORY_PATH == "symbols/sec_exchange_history.parquet"
    assert sth.EXCHANGE_HISTORY_SCHEMA.names == ["snapshot_date", "ticker", "cik", "exchange"]
    assert sth.EXCHANGE_HISTORY_SCHEMA.field("snapshot_date").type == pa.date32()
    assert sth.EXCHANGE_HISTORY_SCHEMA.field("ticker").type == pa.string()
    assert sth.EXCHANGE_HISTORY_SCHEMA.field("cik").type == pa.int64()
    assert sth.EXCHANGE_HISTORY_SCHEMA.field("exchange").type == pa.string()


def test_dated_snapshots_become_rows_with_normalised_tickers_and_upper_case_exchanges(
    tmp_path: Path,
) -> None:
    snapshot(
        tmp_path,
        "20240204",
        [
            [756894, "BARRICK GOLD CORP", "GOLD", "NYSE"],
            [9984, "BARNES GROUP INC", "B", "NYSE"],
            [1739614, "Inhibrx, Inc.", "INBX", "Nasdaq"],
            [1067983, "BERKSHIRE HATHAWAY INC", "BRK-B", "NYSE"],
            [1100663, "ISHARES TRUST", "IVV", "CBOE"],
        ],
    )
    snapshot(
        tmp_path,
        "20250531",
        [
            [756894, "BARRICK MINING CORP", "B", "NYSE"],
            [2007919, "Inhibrx Biosciences, Inc.", "INBX", "Nasdaq"],
        ],
    )

    table = sth.build_exchange_history(tmp_path)

    assert table.schema.equals(sth.EXCHANGE_HISTORY_SCHEMA)
    assert as_rows(table) == [
        (date(2024, 2, 4), "B", 9984, "NYSE"),
        (date(2024, 2, 4), "BRK.B", 1067983, "NYSE"),
        (date(2024, 2, 4), "GOLD", 756894, "NYSE"),
        (date(2024, 2, 4), "INBX", 1739614, "NASDAQ"),
        (date(2024, 2, 4), "IVV", 1100663, "CBOE"),
        (date(2025, 5, 31), "B", 756894, "NYSE"),
        (date(2025, 5, 31), "INBX", 2007919, "NASDAQ"),
    ]
    # the table round-trips through Parquet unchanged (the audit reads it from disk)
    path = tmp_path / "out.parquet"
    pq.write_table(table, path)
    assert as_rows(pq.read_table(path)) == as_rows(table)


def test_column_order_comes_from_fields(tmp_path: Path) -> None:
    snapshot(
        tmp_path,
        "20230526",
        [["NYSE", "GOLD", 756894, "BARRICK GOLD CORP"]],
        fields=["exchange", "ticker", "cik", "name"],
    )

    assert as_rows(sth.build_exchange_history(tmp_path)) == [
        (date(2023, 5, 26), "GOLD", 756894, "NYSE")
    ]


def test_rows_without_a_usable_ticker_are_skipped(tmp_path: Path) -> None:
    snapshot(
        tmp_path,
        "20230526",
        [
            [1, "No Ticker Inc", None, "NYSE"],
            [2, "Blank Ticker Inc", "", "Nasdaq"],
            [756894, "BARRICK GOLD CORP", "GOLD", "NYSE"],
        ],
    )

    assert as_rows(sth.build_exchange_history(tmp_path)) == [
        (date(2023, 5, 26), "GOLD", 756894, "NYSE")
    ]


def test_null_or_blank_exchange_is_never_one_of_the_listing_exchanges(tmp_path: Path) -> None:
    snapshot(
        tmp_path,
        "20230526",
        [[3, "Fund A", "FNDAX", None], [4, "Fund B", "FNDBX", ""], [5, "Otc Co", "OTCY", "OTC"]],
    )

    rows = sth.build_exchange_history(tmp_path).to_pylist()

    assert {r["ticker"] for r in rows} <= {"FNDAX", "FNDBX", "OTCY"}
    assert all(r["exchange"] not in ("NASDAQ", "NYSE", "CBOE") for r in rows)
    assert [r["exchange"] for r in rows if r["ticker"] == "OTCY"] == ["OTC"]


def test_only_dated_json_files_are_snapshots(tmp_path: Path) -> None:
    snapshot(tmp_path, "20230526", [[756894, "BARRICK GOLD CORP", "GOLD", "NYSE"]])
    (tmp_path / "latest.json").write_text(
        json.dumps({"fields": FIELDS, "data": [[1, "X", "XXX", "NYSE"]]})
    )
    (tmp_path / "20230601.json.tmp").write_text("{not json")
    (tmp_path / "notes.txt").write_text("hello")

    assert as_rows(sth.build_exchange_history(tmp_path)) == [
        (date(2023, 5, 26), "GOLD", 756894, "NYSE")
    ]


def test_missing_directory_gives_an_empty_table(tmp_path: Path) -> None:
    table = sth.build_exchange_history(tmp_path / "nowhere")

    assert table.num_rows == 0
    assert table.schema.equals(sth.EXCHANGE_HISTORY_SCHEMA)
