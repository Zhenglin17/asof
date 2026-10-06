"""Guards against survivorship bias in the symbol universe: delisted names stay in, OTC noise
stays out, and every asset listing is kept as a dated snapshot rather than overwritten."""

from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from asof.ingest.universe import (
    HISTORY_PATH,
    build_universe,
    load_universe,
    save_assets_snapshot,
)
from tests.ingest.fakes import make_asset

AAPL = make_asset("AAPL", exchange="NASDAQ")
MSFT = make_asset("MSFT", exchange="NASDAQ")
SIVB = make_asset("SIVB", exchange="NYSE", status="inactive", tradable=False)
OTC = make_asset("AAPCF", exchange="OTC")
DELISTED_JUNK = make_asset("CIC_DELISTED", exchange="NYSE", status="inactive", tradable=False)


def leftovers(directory: Path) -> list[str]:
    return sorted(str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file())


# --- normal path -------------------------------------------------------------------------------


def test_universe_is_the_sorted_union_of_listed_assets_and_sec_history() -> None:
    universe = build_universe([MSFT, AAPL, SIVB], historical=["TWTR", "BRK-B"])

    assert universe == ["AAPL", "BRK.B", "MSFT", "SIVB", "TWTR"]


def test_otc_assets_are_dropped_by_default() -> None:
    assert build_universe([AAPL, OTC], historical=[]) == ["AAPL"]


def test_invalid_symbols_are_dropped_from_both_sources() -> None:
    universe = build_universe([AAPL, DELISTED_JUNK], historical=["384CNT069", "TWTR"])

    assert universe == ["AAPL", "TWTR"]


def test_symbols_present_in_both_sources_appear_once() -> None:
    universe = build_universe(
        [AAPL, make_asset("BRK.B", exchange="NYSE")], historical=["BRK-B", "AAPL"]
    )

    assert universe == ["AAPL", "BRK.B"]


def test_save_assets_snapshot_writes_a_dated_parquet_with_the_asset_columns(tmp_path: Path) -> None:
    path = save_assets_snapshot([SIVB, AAPL], tmp_path, date(2026, 10, 5))

    assert path == tmp_path / "assets" / "snapshot_date=2026-10-05" / "assets.parquet"
    table = pq.read_table(path)
    assert table.schema.names == ["symbol", "name", "exchange", "status", "tradable", "asset_class"]
    assert sorted(table.to_pylist(), key=lambda row: row["symbol"]) == [
        {
            "symbol": "AAPL",
            "name": "AAPL Inc.",
            "exchange": "NASDAQ",
            "status": "active",
            "tradable": True,
            "asset_class": "us_equity",
        },
        {
            "symbol": "SIVB",
            "name": "SIVB Inc.",
            "exchange": "NYSE",
            "status": "inactive",
            "tradable": False,
            "asset_class": "us_equity",
        },
    ]


# --- boundaries --------------------------------------------------------------------------------


def test_empty_inputs_give_an_empty_universe() -> None:
    assert build_universe([], historical=[]) == []


def test_exclusions_are_configurable() -> None:
    assert build_universe([AAPL, OTC], historical=[], exclude_exchanges=frozenset()) == [
        "AAPCF",
        "AAPL",
    ]
    assert build_universe([AAPL, SIVB], historical=[], exclude_exchanges=frozenset({"NYSE"})) == [
        "AAPL"
    ]


def test_otc_symbol_is_kept_when_sec_history_lists_it() -> None:
    assert build_universe([OTC], historical=["AAPCF"]) == ["AAPCF"]


def test_historical_tickers_are_normalized_before_filtering() -> None:
    assert build_universe([], historical=[" brk-b ", "aaic-pb"]) == ["AAIC.PRB", "BRK.B"]


def test_saving_the_same_snapshot_date_twice_replaces_it_cleanly(tmp_path: Path) -> None:
    save_assets_snapshot([AAPL], tmp_path, date(2026, 10, 5))
    path = save_assets_snapshot([AAPL, MSFT], tmp_path, date(2026, 10, 5))

    assert sorted(pq.read_table(path).column("symbol").to_pylist()) == ["AAPL", "MSFT"]
    assert leftovers(tmp_path) == ["assets/snapshot_date=2026-10-05/assets.parquet"]


# --- leakage -----------------------------------------------------------------------------------


def test_inactive_assets_stay_in_the_universe_so_replays_see_delisted_names() -> None:
    # Dropping inactive assets would make every backtest trade only the survivors.
    universe = build_universe([AAPL, SIVB], historical=[])

    assert "SIVB" in universe


def test_a_later_snapshot_does_not_touch_an_earlier_one(tmp_path: Path) -> None:
    earlier = save_assets_snapshot([AAPL, SIVB], tmp_path, date(2026, 1, 5))
    before = earlier.read_bytes()

    later = save_assets_snapshot([AAPL], tmp_path, date(2026, 10, 5))

    assert later != earlier
    assert earlier.read_bytes() == before
    assert sorted(pq.read_table(earlier).column("symbol").to_pylist()) == ["AAPL", "SIVB"]
    assert pq.read_table(later).column("symbol").to_pylist() == ["AAPL"]
    assert leftovers(tmp_path) == [
        "assets/snapshot_date=2026-01-05/assets.parquet",
        "assets/snapshot_date=2026-10-05/assets.parquet",
    ]


# --- load_universe: what backfill sees on disk --------------------------------------------------


def test_load_universe_combines_the_newest_assets_snapshot_with_sec_history(
    tmp_path: Path,
) -> None:
    save_assets_snapshot([AAPL, OTC], tmp_path, date(2026, 1, 5))
    save_assets_snapshot([AAPL, MSFT, DELISTED_JUNK], tmp_path, date(2026, 10, 5))
    history = tmp_path / HISTORY_PATH
    history.parent.mkdir(parents=True)
    pq.write_table(
        pa.table({"ticker": ["BRK.B", "SIVB"]}, schema=pa.schema([("ticker", pa.string())])),
        history,
    )

    assert load_universe(tmp_path) == ["AAPL", "BRK.B", "MSFT", "SIVB"]


def test_load_universe_of_an_empty_root_is_empty(tmp_path: Path) -> None:
    assert load_universe(tmp_path) == []


def test_excluded_symbols_are_dropped_unless_alpaca_lists_them_on_an_exchange() -> None:
    # FNMA only ever traded OTC per SEC: out. SRNE is excluded by SEC data but Alpaca lists it
    # on Nasdaq: the exchange listing wins.
    universe = build_universe(
        [AAPL, make_asset("SRNE", exchange="NASDAQ")],
        historical=["FNMA", "SRNE", "TWTR"],
        exclude_symbols={"FNMA", "SRNE"},
    )

    assert universe == ["AAPL", "SRNE", "TWTR"]
