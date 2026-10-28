"""Guards against a spin-off stored without its distribution ratio (every reader would then have
nothing to adjust the parent's history by) or with a ratio read from the wrong field, and against
the change leaking into the other kinds stored in ``others.parquet``.

Target API (asof.ingest.corporate_actions):

    parse_others(rows) -> (pa.Table, pa.Table)       schema OTHER_SCHEMA, unchanged
        kind ``spin_off``: ``rate`` = new_rate / source_rate of the raw record, i.e. child shares
        per parent share. A missing or zero ``source_rate`` (or a missing ``new_rate``) gives
        ``rate`` NULL; the row is kept, never quarantined. Every other kind keeps ``rate`` as
        today (the record's own ``rate`` field, or NULL).

Fake data provenance:
- GE -> GEHC spin_off ex 2023-01-04, new_rate 0.33333, source_rate 1 (real raw record, read off
  others.parquet on 2026-10-27; id / cusips / payable / record dates constructed in the AMC ->
  APE record's shape).
- GE -> GEV spin_off ex 2024-04-02, new_rate 0.25, source_rate 1 (real, same probe).
- AMC -> APE spin_off, new_rate 1, source_rate 1 (real; tests/ingest/test_corporate_actions).
- new_rate / source_rate arrive as JSON int or float (real: both, store probe 2026-10-27).
- A zero or missing source_rate: ASSUMPTION, not observed (0 of 237 real spin-off rows on
  2026-10-27). Covered on real data by the golden case below (no spin-off row has rate NULL).
- ACSAY stock_dividend rate 1.0625 and LILA rights rate 0.269 (real; same fixtures).

Real-data check: after `asof market actions` rebuilds the tables, every spin_off row of
others.parquet has a non-null rate, and GE 2023-01-04 reads 0.33333
(tests/store/test_golden_spinoffs.py).
"""

from datetime import UTC, datetime
from typing import Any

import pytest

from asof.ingest.corporate_actions import OTHER_SCHEMA, parse_others
from tests.ingest.test_corporate_actions import (
    ACSAY_STOCK_DIVIDEND,
    APE_SPIN_OFF,
    LILA_RIGHTS,
    WORTHLESS,
)

GEHC_SPIN_OFF: dict[str, Any] = {
    "ex_date": "2023-01-04",
    "id": "00000000-0000-0000-0000-00000000gehc",
    "new_cusip": "36266G107",
    "new_rate": 0.33333,
    "new_symbol": "GEHC",
    "payable_date": "2023-01-03",
    "process_date": "2023-01-04",
    "record_date": "2022-12-16",
    "source_cusip": "369604301",
    "source_rate": 1,
    "source_symbol": "GE",
    "_kind": "spin_offs",
}
GEV_SPIN_OFF: dict[str, Any] = {
    **GEHC_SPIN_OFF,
    "ex_date": "2024-04-02",
    "id": "00000000-0000-0000-0000-000000000gev",
    "new_rate": 0.25,
    "new_symbol": "GEV",
    "process_date": "2024-04-02",
}


def only(rows: list[dict[str, Any]]) -> dict[str, Any]:
    table, rejects = parse_others(rows)
    assert rejects.num_rows == 0, rejects.to_pylist()
    [row] = table.to_pylist()
    return row


# =============================================================================================
# normal path
# =============================================================================================


def test_spin_off_rate_is_child_shares_per_parent_share() -> None:
    row = only([GEHC_SPIN_OFF])

    assert row["kind"] == "spin_off"
    assert (row["source_symbol"], row["new_symbol"]) == ("GE", "GEHC")
    assert row["rate"] == pytest.approx(0.33333)
    assert row["available_at"] == datetime(2023, 1, 4, 5, 0, tzinfo=UTC)


def test_every_spin_off_in_a_batch_gets_its_own_rate() -> None:
    table, rejects = parse_others([GEHC_SPIN_OFF, GEV_SPIN_OFF, APE_SPIN_OFF])

    assert rejects.num_rows == 0 and table.schema == OTHER_SCHEMA
    rates = {r["new_symbol"]: r["rate"] for r in table.to_pylist()}
    assert rates == {"GEHC": pytest.approx(0.33333), "GEV": 0.25, "APE": 1.0}


# =============================================================================================
# boundaries
# =============================================================================================


def test_rate_divides_by_a_source_rate_other_than_one() -> None:
    # 1 child share for every 3 parent shares, quoted as new 1 / source 3
    row = only([{**GEHC_SPIN_OFF, "new_rate": 1, "source_rate": 3}])
    assert row["rate"] == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    "change",
    [
        {"source_rate": 0},
        {"source_rate": None},
        {"source_rate": ""},
        {"new_rate": None},
    ],
    ids=["zero_source_rate", "null_source_rate", "empty_source_rate", "null_new_rate"],
)
def test_unusable_rates_give_null_and_keep_the_row(change: dict[str, Any]) -> None:
    row = only([{**GEHC_SPIN_OFF, **change}])
    assert row["rate"] is None
    assert row["new_symbol"] == "GEHC"


def test_missing_source_rate_key_gives_null() -> None:
    record = {k: v for k, v in GEHC_SPIN_OFF.items() if k != "source_rate"}
    assert only([record])["rate"] is None


def test_other_kinds_keep_their_own_rate() -> None:
    table, rejects = parse_others([ACSAY_STOCK_DIVIDEND, LILA_RIGHTS, WORTHLESS])

    assert rejects.num_rows == 0
    rates = {r["kind"]: r["rate"] for r in table.to_pylist()}
    assert rates == {
        "stock_dividend": 1.0625,
        "rights_distribution": 0.269,
        "worthless_removal": None,
    }


def test_a_rights_record_with_new_and_source_rates_is_not_treated_as_a_spin_off() -> None:
    # only spin_off rows are re-read; a rights row carrying the same two fields keeps its rate
    record = {**LILA_RIGHTS, "new_rate": 5, "source_rate": 1}
    assert only([record])["rate"] == 0.269
