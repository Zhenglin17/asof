"""Guards against survivorship bias in the symbol universe: delisted names stay in, OTC noise
stays out, both names of every rename are requested, and every asset listing is kept as a dated
snapshot rather than overwritten.

The renamed-symbol tests have no ``as_of`` to leak through. What their fixtures assume about the
real data has to be checked on the real files instead:

- Alpaca's ``name_changes`` symbols are already in Alpaca's dot spelling, so ``normalize_ticker``
  is a pass-through for them (assumption): count name-change rows whose old_symbol or new_symbol
  contains ``-`` or lowercase; expect zero.
- An old name that Alpaca now lists only on OTC is not worth requesting (assumption): list the
  name-change symbols that sit on OTC in the newest assets snapshot and spot-check a few.
- SPLG -> SPYM is a real name change and SPLG is absent from every assets snapshot (reported on
  real data when this was added; recheck after the next ``asof market universe``).
- After the change the universe should grow by roughly the 630 old + 716 new names reported
  missing; a much smaller delta means the exclusions above are eating real symbols.
"""

from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from asof.ingest.corporate_actions import NAME_CHANGE_SCHEMA, actions_dir
from asof.ingest.universe import (
    HISTORY_PATH,
    build_universe,
    load_renamed,
    load_universe,
    rename_chains,
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


# --- renamed symbols: both names of every Alpaca name change --------------------------------
# Alpaca prunes old inactive names from its asset list and SEC's company table has no ETFs, so
# a renamed ETF (SPLG -> SPYM, a real name change) was in neither source. The old and the new
# name of every name change are a third source.


def write_name_changes(root: Path, renames: Sequence[tuple[str, str]]) -> Path:
    """``<root>/corporate_actions/name_changes.parquet`` with one row per (old, new) pair, in
    the schema ``corporate_actions.build_tables`` writes. Dates and CUSIPs are filler: only the
    two symbol columns matter to the universe."""
    path = actions_dir(root) / "name_changes.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    count = len(renames)
    pq.write_table(
        pa.table(
            {
                "old_symbol": [old for old, _ in renames],
                "new_symbol": [new for _, new in renames],
                "process_date": [date(2026, 1, 14)] * count,
                "old_cusip": ["571748102"] * count,
                "new_cusip": ["571748102"] * count,
                "available_at": [datetime(2026, 1, 14, 5, 0, tzinfo=UTC)] * count,
            },
            schema=NAME_CHANGE_SCHEMA,
        ),
        path,
    )
    return path


def test_renamed_symbols_join_the_universe_old_and_new() -> None:
    universe = build_universe([AAPL], historical=[], renamed=["SPLG", "SPYM"])

    assert universe == ["AAPL", "SPLG", "SPYM"]


def test_renamed_is_positional_after_historical() -> None:
    assert build_universe([AAPL], ["TWTR"], ["SPLG"]) == ["AAPL", "SPLG", "TWTR"]


def test_renamed_symbols_present_in_other_sources_appear_once() -> None:
    universe = build_universe(
        [AAPL, MSFT], historical=["SPLG", "BRK-B"], renamed=["AAPL", "SPLG", "SPYM", "BRK.B"]
    )

    assert universe == ["AAPL", "BRK.B", "MSFT", "SPLG", "SPYM"]


def test_load_renamed_returns_the_sorted_union_of_old_and_new_symbols(tmp_path: Path) -> None:
    # AMAM -> AMAM is a real row (only the CUSIP changed); the name must not appear twice.
    write_name_changes(tmp_path, [("SPLG", "SPYM"), ("MMC", "MRSH"), ("AMAM", "AMAM")])

    assert load_renamed(tmp_path) == ["AMAM", "MMC", "MRSH", "SPLG", "SPYM"]


def test_load_universe_combines_assets_history_and_name_changes(tmp_path: Path) -> None:
    save_assets_snapshot([AAPL, OTC], tmp_path, date(2026, 10, 5))
    history = tmp_path / HISTORY_PATH
    history.parent.mkdir(parents=True)
    pq.write_table(
        pa.table({"ticker": ["SIVB"]}, schema=pa.schema([("ticker", pa.string())])), history
    )
    write_name_changes(tmp_path, [("SPLG", "SPYM")])

    assert load_universe(tmp_path) == ["AAPL", "SIVB", "SPLG", "SPYM"]


# --- renamed symbols: boundaries -------------------------------------------------------------


def test_load_renamed_of_a_root_without_name_changes_is_empty(tmp_path: Path) -> None:
    assert load_renamed(tmp_path) == []

    actions_dir(tmp_path).mkdir(parents=True)  # the directory alone is not a table either
    assert load_renamed(tmp_path) == []


def test_load_universe_without_name_changes_is_unchanged(tmp_path: Path) -> None:
    save_assets_snapshot([AAPL, MSFT], tmp_path, date(2026, 10, 5))
    history = tmp_path / HISTORY_PATH
    history.parent.mkdir(parents=True)
    pq.write_table(
        pa.table({"ticker": ["SIVB"]}, schema=pa.schema([("ticker", pa.string())])), history
    )

    assert load_universe(tmp_path) == ["AAPL", "MSFT", "SIVB"]


def test_renamed_with_nothing_else_is_just_the_renamed_names() -> None:
    assert build_universe([], historical=[], renamed=["SPLG", "SPYM"]) == ["SPLG", "SPYM"]
    assert build_universe([], historical=[], renamed=[]) == []


def test_invalid_renamed_symbols_are_dropped() -> None:
    universe = build_universe(
        [AAPL], historical=[], renamed=["384CNT069", "CIC_DELISTED", "SPLG", "SPYM"]
    )

    assert universe == ["AAPL", "SPLG", "SPYM"]


def test_renamed_symbol_that_alpaca_lists_on_otc_is_dropped() -> None:
    # Unlike SEC history, a name change does not vouch for an OTC listing.
    assert build_universe([OTC, AAPL], historical=[], renamed=["AAPCF", "SPLG"]) == [
        "AAPL",
        "SPLG",
    ]
    # The exchange filter is the configured one, not OTC by name.
    assert build_universe(
        [SIVB, AAPL], historical=[], renamed=["SIVB"], exclude_exchanges=frozenset({"NYSE"})
    ) == ["AAPL"]


def test_renamed_symbol_still_kept_when_sec_history_vouches_for_it() -> None:
    # The historical path is unchanged: SEC history listing an Alpaca-OTC name keeps it, and
    # the same name also being a rename does not take that away.
    assert build_universe([OTC], historical=["AAPCF"], renamed=["AAPCF"]) == ["AAPCF"]


def test_renamed_symbol_in_exclude_symbols_is_dropped() -> None:
    universe = build_universe(
        [AAPL], historical=[], renamed=["FNMA", "SPLG"], exclude_symbols={"FNMA"}
    )

    assert universe == ["AAPL", "SPLG"]


def test_renamed_tickers_are_normalized_before_filtering() -> None:
    # Alpaca's name changes should already be in dot spelling (see the module docstring for the
    # real-data check); hyphen and case variants must still collapse onto the same symbol.
    universe = build_universe([], historical=[], renamed=[" splg ", "brk-b", "aaic-pb", "BRK.B"])

    assert universe == ["AAIC.PRB", "BRK.B", "SPLG"]


def test_renamed_tickers_that_cannot_be_normalized_are_skipped() -> None:
    assert build_universe([], historical=[], renamed=["", "   ", "A-B-C", "SPLG"]) == ["SPLG"]


# --- rename chains: the names one security has traded under --------------------------------------


def test_rename_chains_connect_transitive_renames_and_skip_placeholders(tmp_path: Path) -> None:
    # VII -> BNZI -> PARA is one chain even though no record links VII to PARA directly; a
    # CUSIP placeholder on one side of a record (real feed: "1ORIC" -> ORIC) links nothing.
    write_name_changes(
        tmp_path,
        [("VII", "BNZI"), ("BNZI", "PARA"), ("SPLG", "SPYM"), ("1ORIC", "ORIC"), ("AMAM", "AMAM")],
    )
    assert rename_chains(tmp_path) == [
        frozenset({"BNZI", "PARA", "VII"}),
        frozenset({"SPLG", "SPYM"}),
    ]


def test_rename_chains_merge_the_two_holders_of_a_reused_symbol(tmp_path: Path) -> None:
    # Paramount's VIAC -> PARA and Banzai's BNZI -> PARA share the symbol, so they share a set;
    # chains decide only when a request is repeated, never who owns a bar.
    write_name_changes(tmp_path, [("VIAC", "PARA"), ("BNZI", "PARA")])
    assert rename_chains(tmp_path) == [frozenset({"BNZI", "PARA", "VIAC"})]


def test_rename_chains_of_a_root_without_name_changes_is_empty(tmp_path: Path) -> None:
    assert rename_chains(tmp_path) == []
