"""Guards against corporate actions that are stored wrong or known too early: a split, rename
or dividend must keep Alpaca's exact dates and rates, bad rows must be quarantined rather than
dropped or let through, and nothing is visible before 00:00 New York of its effective day.

Target API (asof.ingest.corporate_actions, new module):

    fetch_corporate_actions(client, types: Sequence[str], start: date, end: date) -> list[dict]
        GET {client.DATA_URL}/v1beta1/corporate-actions, one chain of pages per calendar year,
        limit=1000, follows next_page_token; every record gets ``_kind`` = the key it sat under
        in ``corporate_actions`` (``forward_splits``, ``name_changes``, ...).
    write_raw(rows, root: Path, type_: str, year: int) -> None
        <root>/raw/alpaca/corporate_actions/<type_>/<year>.json, written atomically.
    parse_splits(rows) -> (pa.Table, pa.Table)          forward / reverse / unit splits in one
    parse_name_changes(rows) -> (pa.Table, pa.Table)
    parse_mergers(rows) -> (pa.Table, pa.Table)
    parse_dividends(rows) -> (pa.Table, pa.Table)
    parse_others(rows) -> (pa.Table, pa.Table)
        Second element is the rejects table: kind, raw_json, reason.
    asof.store.market.visible_splits(con, root, as_of, symbols=None) -> pa.Table
        reads <market root>/corporate_actions/splits.parquet, filters available_at <= as_of.

Fake data provenance (all shapes copied from a 2026-10-06 probe of the real endpoint and the
raw download it kept):
- paging: ``{"corporate_actions": {"<kind>": [...]}, "next_page_token": "..."|null}`` (probe
  script, 7 requests per type per year chunk, 269 for dividends)
- NVDA forward_split 2024-06-10 1->10, cusip 67066G104 (real record)
- HCTI reverse_split 2025-08-01 249->1, old_cusip 42227W207 -> new_cusip 42227W306 (real)
- AAN reverse_split 2020-12-01 with old_cusip = new_cusip = "" (real; 292 such rows)
- ADOCU unit_split 2020-12-10, columns old/new/alternate_symbol, no ex_date (real)
- AIMAU unit_split 2022-06-16 with new_rate 0 and new_symbol "" (real; one of the 2 rows)
- MMC->MRSH name_change 2026-01-14, cusip unchanged 571748102 (real)
- AMAM->AMAM name_change 2023-10-12, only the CUSIP changed (real; 220 such rows)
- TWTR cash_merger 2022-10-28 rate 54.2 with no acquirer fields (real)
- APE->AMC stock_merger 2023-08-25 acquirer_rate 0.1 (real)
- AGN->ABBV stock_and_cash_merger 2020-05-08 acquirer_rate 0.866 cash_rate 120.3 (real)
- cash_dividend "A" 2020-03-30 without sub_type; AEBI 2026-06-05 with sub_type
  return_of_capital (real: the field appears from record 225,367 on)
- YOKEY cash_dividend with ex_date 3026-03-31 (real bad row)
- VNO.PRN cash_dividend ex_date 2026-12-15, already announced (real; ex_date in the future)
- AMC -> APE spin_off ex_date 2022-08-22 but process_date 2024-07-10 (real)
- worthless_removal / redemption rows whose symbol is a CUSIP-like string (real)
- Hyphenated symbol spelling (``XYZ-PRA``) in an Alpaca record: ASSUMPTION, not observed; the
  probe saw only dot spellings. Covered on real data by audit A7 / golden case 14 (BAC.PRB).
"""

import json
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from asof.ingest.corporate_actions import (
    download_all,
    fetch_corporate_actions,
    parse_dividends,
    parse_mergers,
    parse_name_changes,
    parse_others,
    parse_splits,
    write_raw,
)
from asof.store.market import market_root, visible_splits

ET = "America/New_York"


def et_midnight(day: date) -> datetime:
    from zoneinfo import ZoneInfo

    return datetime(day.year, day.month, day.day, tzinfo=ZoneInfo(ET)).astimezone(UTC)


# =============================================================================================
# real-shaped records
# =============================================================================================

NVDA_SPLIT = {
    "cusip": "67066G104",
    "due_bill_redemption_date": "2024-06-10",
    "ex_date": "2024-06-10",
    "id": "50199fac-0af8-43ef-9846-eaf64c6d322d",
    "new_rate": 10,
    "old_rate": 1,
    "payable_date": "2024-06-10",
    "process_date": "2024-06-10",
    "record_date": "2024-06-07",
    "symbol": "NVDA",
    "_kind": "forward_splits",
}
HCTI_REVERSE = {
    "ex_date": "2025-08-01",
    "id": "fd8bdfbe-bee8-4221-ad03-30ce3f51eb6f",
    "new_cusip": "42227W306",
    "new_rate": 1,
    "old_cusip": "42227W207",
    "old_rate": 249,
    "payable_date": "2025-08-01",
    "process_date": "2025-08-01",
    "record_date": "2025-08-01",
    "symbol": "HCTI",
    "_kind": "reverse_splits",
}
AAN_REVERSE_EMPTY_CUSIP = {
    "ex_date": "2020-12-01",
    "id": "e026eee7-235b-4613-b05e-19106d219063",
    "new_cusip": "",
    "new_rate": 0.5,
    "old_cusip": "",
    "old_rate": 2,
    "process_date": "2020-12-01",
    "symbol": "AAN",
    "_kind": "reverse_splits",
}
ADOCU_UNIT = {
    "alternate_cusip": "G4000A110",
    "alternate_rate": 1,
    "alternate_symbol": "ADOCW",
    "effective_date": "2020-12-10",
    "id": "3683ce21-ac3d-4483-b522-9af292b50f4b",
    "new_cusip": "G4000A102",
    "new_rate": 1,
    "new_symbol": "ADOC",
    "old_cusip": "G4000A128",
    "old_rate": 1,
    "old_symbol": "ADOCU",
    "process_date": "2020-12-10",
    "_kind": "unit_splits",
}
AIMAU_UNIT_ZERO_RATE = {
    "alternate_cusip": "G0135E126",
    "alternate_rate": 1,
    "alternate_symbol": "AIMAW",
    "effective_date": "2022-06-16",
    "id": "ed896f6c-d783-404e-9513-f6deb5659793",
    "new_cusip": "",
    "new_rate": 0,
    "new_symbol": "",
    "old_cusip": "G0135E100",
    "old_rate": 1,
    "old_symbol": "AIMAU",
    "process_date": "2022-06-16",
    "_kind": "unit_splits",
}
MMC_RENAME = {
    "id": "9a57f85c-2359-41c2-8ec8-35634f5b3f1e",
    "new_cusip": "571748102",
    "new_symbol": "MRSH",
    "old_cusip": "571748102",
    "old_symbol": "MMC",
    "process_date": "2026-01-14",
    "_kind": "name_changes",
}
AMAM_CUSIP_ONLY = {
    "id": "9e6a7c76-c89d-46df-888e-79ed8b0aebbb",
    "new_cusip": "641871108",
    "new_symbol": "AMAM",
    "old_cusip": "02290A102",
    "old_symbol": "AMAM",
    "process_date": "2023-10-12",
    "_kind": "name_changes",
}
TWTR_CASH_MERGER = {
    "acquiree_cusip": "90184L102",
    "acquiree_symbol": "TWTR",
    "effective_date": "2022-10-28",
    "id": "0cfca8cd-6187-47d5-8ee0-45fde44d03e5",
    "process_date": "2022-10-28",
    "rate": 54.2,
    "_kind": "cash_mergers",
}
APE_STOCK_MERGER = {
    "acquiree_cusip": "00165C203",
    "acquiree_rate": 1,
    "acquiree_symbol": "APE",
    "acquirer_cusip": "00165C302",
    "acquirer_rate": 0.1,
    "acquirer_symbol": "AMC",
    "effective_date": "2023-08-25",
    "id": "5e6f60bc-98dc-4e52-8add-3c9122f81166",
    "payable_date": "2023-08-25",
    "process_date": "2023-08-25",
    "_kind": "stock_mergers",
}
AGN_STOCK_AND_CASH = {
    "acquiree_cusip": "G0177J108",
    "acquiree_rate": 1,
    "acquiree_symbol": "AGN",
    "acquirer_cusip": "00287Y109",
    "acquirer_rate": 0.866,
    "acquirer_symbol": "ABBV",
    "cash_rate": 120.3,
    "effective_date": "2020-05-08",
    "id": "dfd703bf-adad-46e4-bd18-c4fcbc6ed81d",
    "process_date": "2020-05-08",
    "_kind": "stock_and_cash_mergers",
}
A_DIVIDEND = {
    "cusip": "00846U101",
    "ex_date": "2020-03-30",
    "foreign": False,
    "id": "dfc470cc-8582-4281-912b-e2cca6fc6c7d",
    "payable_date": "2020-04-22",
    "process_date": "2020-04-22",
    "rate": 0.18,
    "record_date": "2020-03-31",
    "special": False,
    "symbol": "A",
    "_kind": "cash_dividends",
}
AEBI_DIVIDEND_SUBTYPE = {
    "cusip": "H00501108",
    "ex_date": "2026-06-05",
    "foreign": True,
    "id": "3d23d883-f413-4de9-ba5a-2656bcca723a",
    "payable_date": "2026-06-25",
    "process_date": "2026-06-25",
    "rate": 0.025,
    "record_date": "2026-06-05",
    "special": False,
    "sub_type": "return_of_capital",
    "symbol": "AEBI",
    "_kind": "cash_dividends",
}
YOKEY_BAD_YEAR = {
    "cusip": "986008100",
    "ex_date": "3026-03-31",
    "foreign": True,
    "id": "290da0f2-8a85-4128-85e5-41d37a79843f",
    "payable_date": "2026-07-14",
    "process_date": "2026-07-14",
    "rate": 0.381675,
    "record_date": "2026-03-31",
    "special": False,
    "symbol": "YOKEY",
    "_kind": "cash_dividends",
}
VNO_FUTURE_DIVIDEND = {
    "cusip": "929042810",
    "ex_date": "2026-12-15",
    "foreign": False,
    "id": "f217e345-0097-4a32-86db-9b24a6dada11",
    "payable_date": "2026-01-02",
    "process_date": "2026-01-02",
    "rate": 0.328125,
    "record_date": "2026-12-15",
    "special": False,
    "symbol": "VNO.PRN",
    "_kind": "cash_dividends",
}
APE_SPIN_OFF = {
    "due_bill_redemption_date": "2022-08-23",
    "ex_date": "2022-08-22",
    "id": "efe9202c-31f0-491f-aaed-0f7cf63c52b6",
    "new_cusip": "00165C203",
    "new_rate": 1,
    "new_symbol": "APE",
    "payable_date": "2022-08-19",
    "process_date": "2024-07-10",
    "record_date": "2022-08-15",
    "source_cusip": "00165C104",
    "source_rate": 1,
    "source_symbol": "AMC",
    "_kind": "spin_offs",
}
ACSAY_STOCK_DIVIDEND = {
    "cusip": "00089H106",
    "ex_date": "2020-07-02",
    "id": "4093e23f-1c42-46df-aaf3-6e6523c33c59",
    "payable_date": "2020-07-29",
    "process_date": "2020-07-02",
    "rate": 1.0625,
    "record_date": "2020-07-06",
    "symbol": "ACSAY",
    "_kind": "stock_dividends",
}
LILA_RIGHTS = {
    "ex_date": "2020-09-11",
    "expiration_date": "2020-09-25",
    "id": "66812f1e-a900-4f6c-9931-9a6738839b23",
    "new_cusip": "G9001E136",
    "new_symbol": "LILRV",
    "payable_date": "2020-09-09",
    "process_date": "2020-09-09",
    "rate": 0.269,
    "record_date": "2020-09-08",
    "source_cusip": "G9001E102",
    "source_symbol": "LILA",
    "_kind": "rights_distributions",
}
WORTHLESS = {
    "cusip": "521RGT019",
    "id": "19297549-9f96-44a3-8005-276a5bb2fffa",
    "process_date": "2023-11-16",
    "symbol": "521RGT019",
    "_kind": "worthless_removals",
}

REJECT_COLUMNS = ["kind", "raw_json", "reason"]


def page(kind: str, items: list[dict[str, Any]], token: str | None = None) -> dict[str, Any]:
    """One response page as the real endpoint shapes it."""
    return {"corporate_actions": {kind: items}, "next_page_token": token}


class ScriptedActionsClient:
    """Stands in for AlpacaClient: records every ``_request`` and answers from a script."""

    DATA_URL = "https://data.alpaca.markets"

    def __init__(self, pages: list[Any]) -> None:
        self.pages = list(pages)
        self.calls: list[tuple[str, dict[str, str]]] = []

    def _request(self, url: str, params: dict[str, str]) -> Any:
        self.calls.append((url, dict(params)))
        if not self.pages:
            raise AssertionError(f"unexpected request #{len(self.calls)}: {url} {params}")
        return self.pages.pop(0)

    @property
    def params(self) -> list[dict[str, str]]:
        return [p for _, p in self.calls]


def stripped(record: dict[str, Any]) -> dict[str, Any]:
    """The record as the endpoint sends it, before ``_kind`` is attached."""
    return {k: v for k, v in record.items() if k != "_kind"}


# =============================================================================================
# fetch_corporate_actions
# =============================================================================================


def test_fetch_requests_the_endpoint_per_year_with_limit_1000_and_tags_each_record() -> None:
    client = ScriptedActionsClient(
        [
            page("forward_splits", [stripped(NVDA_SPLIT)]),
            page("forward_splits", []),
        ]
    )

    rows = fetch_corporate_actions(client, ["forward_split"], date(2024, 1, 1), date(2025, 6, 30))

    assert [url for url, _ in client.calls] == [
        "https://data.alpaca.markets/v1beta1/corporate-actions",
        "https://data.alpaca.markets/v1beta1/corporate-actions",
    ]
    assert [(p["start"], p["end"]) for p in client.params] == [
        ("2024-01-01", "2024-12-31"),
        ("2025-01-01", "2025-06-30"),
    ]
    assert all(p["limit"] == "1000" for p in client.params)
    assert all(p["types"] == "forward_split" for p in client.params)
    assert rows == [NVDA_SPLIT]  # the real record plus ``_kind``


def test_fetch_follows_next_page_token_within_a_year() -> None:
    client = ScriptedActionsClient(
        [
            page("reverse_splits", [stripped(HCTI_REVERSE)], token="tok-1"),
            page("reverse_splits", [stripped(AAN_REVERSE_EMPTY_CUSIP)]),
        ]
    )

    rows = fetch_corporate_actions(client, ["reverse_split"], date(2020, 1, 1), date(2020, 12, 31))

    assert "page_token" not in client.params[0]
    assert client.params[1]["page_token"] == "tok-1"
    assert (client.params[1]["start"], client.params[1]["end"]) == ("2020-01-01", "2020-12-31")
    assert [r["symbol"] for r in rows] == ["HCTI", "AAN"]
    assert {r["_kind"] for r in rows} == {"reverse_splits"}


def test_fetch_passes_several_types_in_one_request_and_keeps_each_kind() -> None:
    client = ScriptedActionsClient(
        [
            {
                "corporate_actions": {
                    "forward_splits": [stripped(NVDA_SPLIT)],
                    "name_changes": [stripped(MMC_RENAME)],
                },
                "next_page_token": None,
            }
        ]
    )

    rows = fetch_corporate_actions(
        client, ["forward_split", "name_change"], date(2026, 1, 1), date(2026, 12, 31)
    )

    assert set(client.params[0]["types"].split(",")) == {"forward_split", "name_change"}
    assert sorted(r["_kind"] for r in rows) == ["forward_splits", "name_changes"]


def test_fetch_single_partial_year_is_one_chunk_from_start_to_end() -> None:
    client = ScriptedActionsClient([page("forward_splits", [])])

    assert (
        fetch_corporate_actions(client, ["forward_split"], date(2026, 3, 1), date(2026, 10, 6))
        == []
    )
    assert [(p["start"], p["end"]) for p in client.params] == [("2026-03-01", "2026-10-06")]


def test_fetch_tolerates_an_empty_or_null_page() -> None:
    client = ScriptedActionsClient([None, {"corporate_actions": {}, "next_page_token": None}])

    rows = fetch_corporate_actions(client, ["redemption"], date(2020, 1, 1), date(2021, 12, 31))

    assert rows == []
    assert len(client.calls) == 2


def test_fetch_rejects_a_start_after_the_end() -> None:
    client = ScriptedActionsClient([])

    with pytest.raises(ValueError):
        fetch_corporate_actions(client, ["forward_split"], date(2025, 1, 1), date(2024, 1, 1))

    assert client.calls == []


# =============================================================================================
# write_raw
# =============================================================================================


def leftovers(directory: Path) -> list[str]:
    if not directory.exists():
        return []
    return sorted(str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file())


def test_write_raw_stores_the_rows_verbatim_under_type_and_year(tmp_path: Path) -> None:
    rows = [NVDA_SPLIT, HCTI_REVERSE]

    write_raw(rows, tmp_path, "forward_split", 2024)

    path = tmp_path / "raw" / "alpaca" / "corporate_actions" / "forward_split" / "2024.json"
    assert json.loads(path.read_bytes()) == rows
    assert leftovers(tmp_path) == ["raw/alpaca/corporate_actions/forward_split/2024.json"]


def test_write_raw_overwrites_without_leaving_a_temp_file(tmp_path: Path) -> None:
    write_raw([NVDA_SPLIT], tmp_path, "forward_split", 2024)
    write_raw([], tmp_path, "forward_split", 2024)

    path = tmp_path / "raw" / "alpaca" / "corporate_actions" / "forward_split" / "2024.json"
    assert json.loads(path.read_bytes()) == []
    assert leftovers(tmp_path) == ["raw/alpaca/corporate_actions/forward_split/2024.json"]


# =============================================================================================
# parse_splits
# =============================================================================================

SPLIT_COLUMNS = [
    "symbol",
    "ex_date",
    "old_rate",
    "new_rate",
    "factor",
    "kind",
    "old_cusip",
    "new_cusip",
    "available_at",
]


def test_parse_splits_merges_the_three_kinds_into_one_table() -> None:
    table, rejects = parse_splits([NVDA_SPLIT, HCTI_REVERSE, ADOCU_UNIT])

    assert table.schema.names == SPLIT_COLUMNS
    assert rejects.num_rows == 0
    rows = {r["symbol"]: r for r in table.to_pylist()}
    assert set(rows) == {"NVDA", "HCTI", "ADOCU"}

    nvda = rows["NVDA"]
    assert nvda["ex_date"] == date(2024, 6, 10)
    assert (nvda["old_rate"], nvda["new_rate"]) == (1.0, 10.0)
    assert nvda["factor"] == pytest.approx(10.0)
    assert nvda["kind"] == "forward_split"
    assert nvda["old_cusip"] == "67066G104"
    assert nvda["available_at"] == datetime(2024, 6, 10, 4, 0, tzinfo=UTC)  # 00:00 EDT

    hcti = rows["HCTI"]
    assert hcti["factor"] == pytest.approx(1 / 249)
    assert hcti["kind"] == "reverse_split"
    assert (hcti["old_cusip"], hcti["new_cusip"]) == ("42227W207", "42227W306")
    assert hcti["available_at"] == datetime(2025, 8, 1, 4, 0, tzinfo=UTC)

    unit = rows["ADOCU"]  # symbol = old_symbol, date = effective_date
    assert unit["kind"] == "unit_split"
    assert unit["ex_date"] == date(2020, 12, 10)
    assert unit["factor"] == pytest.approx(1.0)
    assert (unit["old_cusip"], unit["new_cusip"]) == ("G4000A128", "G4000A102")
    assert unit["available_at"] == datetime(2020, 12, 10, 5, 0, tzinfo=UTC)  # 00:00 EST


def test_parse_splits_column_types() -> None:
    table, _ = parse_splits([NVDA_SPLIT])

    assert table.schema.field("ex_date").type == pa.date32()
    assert table.schema.field("factor").type == pa.float64()
    assert table.schema.field("available_at").type == pa.timestamp("us", tz="UTC")
    assert table.schema.field("old_cusip").type == pa.string()


def test_parse_splits_turns_empty_cusip_strings_into_nulls() -> None:
    table, rejects = parse_splits([AAN_REVERSE_EMPTY_CUSIP])

    [row] = table.to_pylist()
    assert rejects.num_rows == 0
    assert (row["old_cusip"], row["new_cusip"]) == (None, None)
    assert row["factor"] == pytest.approx(0.25)


def test_parse_splits_normalizes_the_symbol_spelling() -> None:
    # ASSUMPTION: Alpaca spelled a preferred with a hyphen; only dot spellings were observed.
    row = {**HCTI_REVERSE, "symbol": "xyz-pra"}

    table, _ = parse_splits([row])

    assert table.column("symbol").to_pylist() == ["XYZ.PRA"]


@pytest.mark.parametrize(
    ("bad", "reason_word"),
    [
        (AIMAU_UNIT_ZERO_RATE, "rate"),
        ({**HCTI_REVERSE, "old_rate": None}, "rate"),
        ({k: v for k, v in HCTI_REVERSE.items() if k != "new_rate"}, "rate"),
        ({**HCTI_REVERSE, "old_rate": -5}, "rate"),
        ({**NVDA_SPLIT, "ex_date": "3024-06-10"}, "date"),
        ({**NVDA_SPLIT, "ex_date": "not-a-date"}, "date"),
        ({**NVDA_SPLIT, "symbol": ""}, "symbol"),
        ({k: v for k, v in NVDA_SPLIT.items() if k != "symbol"}, "symbol"),
        ({**ADOCU_UNIT, "old_symbol": ""}, "symbol"),
    ],
    ids=[
        "unit_zero_rate",
        "null_rate",
        "missing_rate",
        "negative_rate",
        "year_after_2100",
        "unparseable_date",
        "empty_symbol",
        "missing_symbol",
        "unit_empty_old_symbol",
    ],
)
def test_parse_splits_quarantines_bad_rows_with_the_original_json(
    bad: dict[str, Any], reason_word: str
) -> None:
    table, rejects = parse_splits([NVDA_SPLIT, bad])

    assert table.column("symbol").to_pylist() == ["NVDA"]
    assert rejects.schema.names == REJECT_COLUMNS
    [reject] = rejects.to_pylist()
    assert reject["kind"] == bad["_kind"].rstrip("s")
    assert json.loads(reject["raw_json"]) == bad
    assert reason_word in reject["reason"].lower()


def test_parse_splits_of_nothing_gives_empty_tables_with_the_schema() -> None:
    table, rejects = parse_splits([])

    assert table.num_rows == 0
    assert table.schema.names == SPLIT_COLUMNS
    assert rejects.num_rows == 0
    assert rejects.schema.names == REJECT_COLUMNS


# =============================================================================================
# parse_name_changes / parse_mergers / parse_dividends / parse_others
# =============================================================================================


def test_parse_name_changes_keeps_both_symbols_and_cusips() -> None:
    table, rejects = parse_name_changes([MMC_RENAME, AMAM_CUSIP_ONLY])

    assert table.schema.names == [
        "old_symbol",
        "new_symbol",
        "process_date",
        "old_cusip",
        "new_cusip",
        "available_at",
    ]
    assert rejects.num_rows == 0
    rows = table.to_pylist()
    mmc = next(r for r in rows if r["old_symbol"] == "MMC")
    assert mmc == {
        "old_symbol": "MMC",
        "new_symbol": "MRSH",
        "process_date": date(2026, 1, 14),
        "old_cusip": "571748102",
        "new_cusip": "571748102",
        "available_at": datetime(2026, 1, 14, 5, 0, tzinfo=UTC),  # 00:00 EST
    }
    # A CUSIP-only change keeps old_symbol == new_symbol; it is data, not an error.
    amam = next(r for r in rows if r["old_symbol"] == "AMAM")
    assert amam["new_symbol"] == "AMAM"
    assert (amam["old_cusip"], amam["new_cusip"]) == ("02290A102", "641871108")


def test_parse_name_changes_rejects_a_missing_new_symbol_or_bad_date() -> None:
    bad_symbol = {**MMC_RENAME, "new_symbol": ""}
    bad_date = {**MMC_RENAME, "process_date": "2026-13-45"}

    table, rejects = parse_name_changes([MMC_RENAME, bad_symbol, bad_date])

    assert table.num_rows == 1
    assert [r["kind"] for r in rejects.to_pylist()] == ["name_change", "name_change"]
    assert [json.loads(r["raw_json"]) for r in rejects.to_pylist()] == [bad_symbol, bad_date]


def test_parse_mergers_covers_cash_stock_and_mixed_deals() -> None:
    table, rejects = parse_mergers([TWTR_CASH_MERGER, APE_STOCK_MERGER, AGN_STOCK_AND_CASH])

    assert table.schema.names == [
        "acquiree_symbol",
        "acquirer_symbol",
        "kind",
        "effective_date",
        "rate",
        "available_at",
    ]
    assert rejects.num_rows == 0
    rows = {r["acquiree_symbol"]: r for r in table.to_pylist()}
    assert rows["TWTR"]["acquirer_symbol"] is None
    assert rows["TWTR"]["kind"] == "cash_merger"
    assert rows["TWTR"]["rate"] == pytest.approx(54.2)
    assert rows["TWTR"]["effective_date"] == date(2022, 10, 28)
    assert rows["TWTR"]["available_at"] == datetime(2022, 10, 28, 4, 0, tzinfo=UTC)
    assert rows["APE"]["acquirer_symbol"] == "AMC"
    assert rows["APE"]["kind"] == "stock_merger"
    assert rows["APE"]["rate"] == pytest.approx(0.1)  # shares of acquirer per acquiree share
    assert rows["AGN"]["kind"] == "stock_and_cash_merger"
    assert rows["AGN"]["acquirer_symbol"] == "ABBV"


def test_parse_mergers_rejects_a_missing_acquiree() -> None:
    bad = {k: v for k, v in TWTR_CASH_MERGER.items() if k != "acquiree_symbol"}

    table, rejects = parse_mergers([bad])

    assert table.num_rows == 0
    [reject] = rejects.to_pylist()
    assert reject["kind"] == "cash_merger"
    assert "symbol" in reject["reason"].lower()


def test_parse_dividends_handles_rows_with_and_without_sub_type() -> None:
    table, rejects = parse_dividends([A_DIVIDEND, AEBI_DIVIDEND_SUBTYPE])

    assert table.schema.names == [
        "symbol",
        "ex_date",
        "rate",
        "special",
        "sub_type",
        "available_at",
    ]
    assert rejects.num_rows == 0
    rows = {r["symbol"]: r for r in table.to_pylist()}
    assert rows["A"]["sub_type"] is None
    assert rows["A"]["rate"] == pytest.approx(0.18)
    assert rows["A"]["special"] is False
    assert rows["A"]["ex_date"] == date(2020, 3, 30)
    assert rows["A"]["available_at"] == datetime(2020, 3, 30, 4, 0, tzinfo=UTC)
    assert rows["AEBI"]["sub_type"] == "return_of_capital"


def test_parse_dividends_quarantines_the_year_3026_row() -> None:
    table, rejects = parse_dividends([A_DIVIDEND, YOKEY_BAD_YEAR])

    assert table.column("symbol").to_pylist() == ["A"]
    [reject] = rejects.to_pylist()
    assert reject["kind"] == "cash_dividend"
    assert json.loads(reject["raw_json"]) == YOKEY_BAD_YEAR
    assert "date" in reject["reason"].lower()


def test_parse_dividends_keeps_an_announced_future_ex_date() -> None:
    # Announced 2026-01-02 for ex_date 2026-12-15: stored as is; available_at lies in the
    # future so visible_* will hide it until then.
    table, rejects = parse_dividends([VNO_FUTURE_DIVIDEND])

    [row] = table.to_pylist()
    assert rejects.num_rows == 0
    assert row["symbol"] == "VNO.PRN"
    assert row["ex_date"] == date(2026, 12, 15)
    assert row["available_at"] == datetime(2026, 12, 15, 5, 0, tzinfo=UTC)


def test_parse_others_keeps_original_fields_and_tags_the_kind() -> None:
    table, rejects = parse_others([APE_SPIN_OFF, ACSAY_STOCK_DIVIDEND, LILA_RIGHTS, WORTHLESS])

    assert rejects.num_rows == 0
    assert {"kind", "available_at"} <= set(table.schema.names)
    assert {"source_symbol", "new_symbol", "symbol", "rate"} <= set(table.schema.names)
    rows = table.to_pylist()
    by_kind = {r["kind"]: r for r in rows}
    assert set(by_kind) == {
        "spin_off",
        "stock_dividend",
        "rights_distribution",
        "worthless_removal",
    }
    spin = by_kind["spin_off"]
    assert (spin["source_symbol"], spin["new_symbol"]) == ("AMC", "APE")
    # ex_date, not the much later process_date, decides when a spin-off is known.
    assert spin["available_at"] == datetime(2022, 8, 22, 4, 0, tzinfo=UTC)
    assert by_kind["stock_dividend"]["symbol"] == "ACSAY"
    assert by_kind["stock_dividend"]["available_at"] == datetime(2020, 7, 2, 4, 0, tzinfo=UTC)
    assert by_kind["rights_distribution"]["available_at"] == datetime(2020, 9, 11, 4, 0, tzinfo=UTC)
    # No ex_date on a worthless removal: process_date is the only effective date it has.
    assert by_kind["worthless_removal"]["available_at"] == datetime(2023, 11, 16, 5, 0, tzinfo=UTC)


def test_every_parser_puts_available_at_at_new_york_midnight_of_the_effective_day() -> None:
    cases = [
        (
            parse_splits,
            [NVDA_SPLIT, HCTI_REVERSE, ADOCU_UNIT],
            ["2024-06-10", "2025-08-01", "2020-12-10"],
        ),
        (parse_name_changes, [MMC_RENAME], ["2026-01-14"]),
        (parse_mergers, [TWTR_CASH_MERGER, APE_STOCK_MERGER], ["2022-10-28", "2023-08-25"]),
        (parse_dividends, [A_DIVIDEND, VNO_FUTURE_DIVIDEND], ["2020-03-30", "2026-12-15"]),
    ]
    for parse, rows, days in cases:
        table, _ = parse(rows)
        got = sorted(table.column("available_at").to_pylist())
        want = sorted(et_midnight(date.fromisoformat(d)) for d in days)
        assert got == want, parse.__name__
        assert all(v.utcoffset().total_seconds() == 0 for v in got), parse.__name__


# =============================================================================================
# leakage: visible_splits
# =============================================================================================


@pytest.fixture
def con() -> Iterator[duckdb.DuckDBPyConnection]:
    connection = duckdb.connect()
    yield connection
    connection.close()


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return market_root(tmp_path)


def store_splits(root: Path, rows: list[dict[str, Any]]) -> None:
    table, _ = parse_splits(rows)
    path = root / "corporate_actions" / "splits.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def test_split_is_invisible_before_midnight_new_york_of_its_ex_date_and_visible_from_then(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    # NVDA 1->10 on 2024-06-10. 00:00 New York that day is 04:00 UTC (EDT): one microsecond
    # earlier the split does not exist yet; at 04:00 UTC the 09:35 scanner must find it.
    store_splits(root, [NVDA_SPLIT, HCTI_REVERSE])

    before = visible_splits(con, root, as_of=datetime(2024, 6, 10, 3, 59, tzinfo=UTC))
    at = visible_splits(con, root, as_of=datetime(2024, 6, 10, 4, 0, tzinfo=UTC))

    assert before.num_rows == 0
    assert at.column("symbol").to_pylist() == ["NVDA"]
    assert at.column("factor").to_pylist() == pytest.approx([10.0])
    assert set(at.schema.names) == set(SPLIT_COLUMNS)


def test_visible_splits_hides_a_later_split_of_the_same_symbol(
    con: duckdb.DuckDBPyConnection, root: Path
) -> None:
    nvda_2021 = {**NVDA_SPLIT, "ex_date": "2021-07-20", "process_date": "2021-07-20", "new_rate": 4}
    store_splits(root, [nvda_2021, NVDA_SPLIT])

    table = visible_splits(con, root, as_of=datetime(2023, 1, 1, tzinfo=UTC), symbols=["NVDA"])

    assert table.column("ex_date").to_pylist() == [date(2021, 7, 20)]
    assert table.column("factor").to_pylist() == pytest.approx([4.0])


def test_visible_splits_symbols_filter(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    store_splits(root, [NVDA_SPLIT, HCTI_REVERSE])
    as_of = datetime(2026, 1, 1, tzinfo=UTC)

    assert visible_splits(con, root, as_of, symbols=["HCTI"]).column("symbol").to_pylist() == [
        "HCTI"
    ]
    assert sorted(visible_splits(con, root, as_of).column("symbol").to_pylist()) == ["HCTI", "NVDA"]
    assert visible_splits(con, root, as_of, symbols=["ZZZZ"]).num_rows == 0


def test_visible_splits_rejects_a_naive_as_of(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    store_splits(root, [NVDA_SPLIT])

    with pytest.raises(ValueError):
        visible_splits(con, root, as_of=datetime(2026, 1, 1))


def test_visible_splits_without_a_file_is_empty(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    assert visible_splits(con, root, as_of=datetime(2026, 1, 1, tzinfo=UTC)).num_rows == 0


# =============================================================================================
# download_all: an empty response must not erase a good raw year file
# =============================================================================================
# Reviewer fix (2026-10-07). Old code wrote ``[]`` over the existing file whenever the fetch
# returned nothing -- a transient empty page from the endpoint silently deleted a whole year of
# raw records. Empty responses do happen (assumption from the probe: types with no events in a
# year, e.g. unit_split in some years, return an empty page), so the rule is: refuse only when
# the file on disk has rows; an empty or absent file is written as usual.


def raw_year(root: Path, type_: str, year: int) -> Path:
    return root / "raw" / "alpaca" / "corporate_actions" / type_ / f"{year}.json"


def test_download_all_refuses_to_overwrite_a_non_empty_year_with_an_empty_response(
    tmp_path: Path,
) -> None:
    write_raw([NVDA_SPLIT], tmp_path, "forward_split", 2024)
    path = raw_year(tmp_path, "forward_split", 2024)
    before = path.read_bytes()
    client = ScriptedActionsClient([page("forward_splits", [])])

    with pytest.raises(RuntimeError) as excinfo:
        download_all(
            client, tmp_path, date(2024, 1, 1), date(2024, 12, 31), types=["forward_split"]
        )

    assert "forward_split" in str(excinfo.value)
    assert "2024" in str(excinfo.value)
    assert path.read_bytes() == before  # untouched, byte for byte
    assert json.loads(path.read_bytes()) == [NVDA_SPLIT]
    assert leftovers(tmp_path) == ["raw/alpaca/corporate_actions/forward_split/2024.json"]
    assert len(client.calls) == 1


@pytest.mark.parametrize("existing", ["empty", "absent"])
def test_download_all_writes_an_empty_response_over_an_empty_or_absent_year(
    tmp_path: Path, existing: str
) -> None:
    if existing == "empty":
        write_raw([], tmp_path, "forward_split", 2024)
    client = ScriptedActionsClient([page("forward_splits", [])])

    counts = download_all(
        client, tmp_path, date(2024, 1, 1), date(2024, 12, 31), types=["forward_split"]
    )

    assert counts == {"forward_split": 0}
    assert json.loads(raw_year(tmp_path, "forward_split", 2024).read_bytes()) == []
    assert leftovers(tmp_path) == ["raw/alpaca/corporate_actions/forward_split/2024.json"]


def test_download_all_a_non_empty_response_replaces_the_year_as_before(tmp_path: Path) -> None:
    # Guard: the refusal is only about EMPTY responses; a normal refresh still overwrites.
    write_raw([HCTI_REVERSE], tmp_path, "forward_split", 2024)  # stale content of any shape
    client = ScriptedActionsClient([page("forward_splits", [stripped(NVDA_SPLIT)])])

    counts = download_all(
        client, tmp_path, date(2024, 1, 1), date(2024, 12, 31), types=["forward_split"]
    )

    assert counts == {"forward_split": 1}
    assert json.loads(raw_year(tmp_path, "forward_split", 2024).read_bytes()) == [NVDA_SPLIT]
