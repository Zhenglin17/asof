"""Guards against the spin-off adjustment drifting away from facts checked by hand on the real
store: a factor that no longer matches the prices, a lagging child no longer waited for, a false
record (HON 2026-06-29) back in play, and GE's B-Xtrender no longer matching TradingView once
its two spin-offs are applied.

Runs on the real store under ``$ASOF_DATA_DIR/market`` (default /data/asof) and skips when it
is absent, like tests/store/test_golden_cases.py. Which facts are hand-checked and which are
store reads pinned as a regression guard:

- hand-checked against TradingView (session note 2026-10-27, B-Xtrender 5/20/5 histogram): GE
  with splits and both spin-offs applied reads daily 21.05 and weekly 21.50 on 2025-06-27,
  monthly 28.93 / 22.20 / -19.61 for 2025-06 / 2025-12 / 2026-09 (TradingView 28.97 / 22.20 /
  -19.62; the monthly gap is warm-up, the store starts in 2020). Without the spin-offs the
  same reads were 21.62 / 34.15 / 28.01 / -18.06.
- store reads (probe 2026-10-27): GE 2024-04-01 close 175.36, GEV 2024-04-02 close 140.00;
  GE 2023-01-03 84.98, GEHC 2023-01-04 60.49 at 0.33333; OPEN 2025-11-17 7.84, its three
  children first trade 2025-11-24; LEN -> MRP first bar 2025-02-07; MPTI.WS first bar
  2025-04-30, 36 sessions after 2025-03-10, the 20th MPTI session being 2025-04-07; no SDZNY
  bar at all, the 20th NVS session after 2023-10-04 being 2023-11-01; CR -> CR 2023-04-04.
- HON 2026-06-29 (closes 232.21 -> 227.80, HONA 220.19 at 0.5): disproved by the prices,
  listed in configs/spinoff_rejections.yaml.
"""

from datetime import UTC, date, datetime
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from asof.ingest.instrument_class import CLASS_FILE
from asof.ingest.security_master import MASTER_FILE
from asof.store.audit import load_exceptions, run_audit
from asof.store.db import default_data_dir
from asof.store.market import market_root, visible_bars, visible_splits
from asof.store.universe import LIQUID_CLASSES, tier_union

ROOT = market_root(default_data_dir())
NOW = datetime(2026, 10, 9, tzinfo=UTC)  # as in test_golden_cases: after the verified store
OTHERS = ROOT / "corporate_actions" / "others.parquet"
REPO_CONFIGS = Path(__file__).resolve().parents[2] / "configs"
SPINOFF_CHECK = "S6"  # see tests/store/test_audit_spinoffs.py

pytestmark = [
    pytest.mark.realdata,
    pytest.mark.skipif(
        not (ROOT / MASTER_FILE).exists()
        or not (ROOT / CLASS_FILE).exists()
        or not OTHERS.exists(),
        reason=f"no real store with corporate actions under {ROOT}",
    ),
]


@pytest.fixture(scope="module")
def con() -> duckdb.DuckDBPyConnection:
    return duckdb.connect()


def events_of(con: duckdb.DuckDBPyConnection, parent: str) -> dict[date, dict]:
    from asof.store.spinoffs import spinoff_events

    rows = spinoff_events(con, ROOT, NOW, parents=[parent]).to_pylist()
    return {r["ex_date"]: r for r in rows}


# -- ingest ----------------------------------------------------------------------------------------


def test_every_stored_spin_off_has_a_rate(con: duckdb.DuckDBPyConnection) -> None:
    row = con.execute(
        f"SELECT count(*) FILTER (WHERE rate IS NULL), "
        f"max(rate) FILTER (WHERE source_symbol = 'GE' AND new_symbol = 'GEHC') "
        f"FROM '{OTHERS}' WHERE kind = 'spin_off'"
    ).fetchone()
    assert row is not None
    missing, ge = row
    assert missing == 0
    assert ge == pytest.approx(0.33333)


# -- events ----------------------------------------------------------------------------------------


def test_ge_gehc_and_gev_adjust_on_their_ex_dates(con: duckdb.DuckDBPyConnection) -> None:
    ge = events_of(con, "GE")

    gehc, gev = ge[date(2023, 1, 4)], ge[date(2024, 4, 2)]
    assert (gehc["status"], gehc["known_on"]) == ("adjust", date(2023, 1, 4))
    assert gehc["parent_prev_close"] == 84.98 and list(gehc["child_closes"]) == [60.49]
    assert gehc["factor"] == pytest.approx(0.7627, abs=1e-3)
    assert (gev["status"], gev["known_on"]) == ("adjust", date(2024, 4, 2))
    assert gev["parent_prev_close"] == 175.36 and list(gev["child_closes"]) == [140.0]
    assert gev["factor"] == pytest.approx(0.8004, abs=1e-3)


def test_open_three_children_are_one_event(con: duckdb.DuckDBPyConnection) -> None:
    from asof.store.spinoffs import visible_spinoffs

    day = date(2025, 11, 18)
    listed = visible_spinoffs(con, ROOT, NOW, ["OPEN"]).to_pylist()
    assert sorted(r["child_symbol"] for r in listed if r["ex_date"] == day) == [
        "OPENL",
        "OPENW",
        "OPENZ",
    ]  # OPEN -> OPENW is stored twice
    row = events_of(con, "OPEN")[day]
    assert list(row["child_symbols"]) == ["OPENL", "OPENW", "OPENZ"]
    assert row["status"] == "adjust"
    assert row["factor"] == pytest.approx(0.9849, abs=1e-3)
    assert row["child_session"] == row["known_on"] == date(2025, 11, 24)


def test_len_waits_thirteen_sessions_for_mrp(con: duckdb.DuckDBPyConnection) -> None:
    row = events_of(con, "LEN")[date(2025, 1, 21)]
    assert (row["status"], row["known_on"]) == ("adjust", date(2025, 2, 7))
    assert row["factor"] == pytest.approx(0.9055, abs=1e-3)


def test_mpti_warrant_came_too_late(con: duckdb.DuckDBPyConnection) -> None:
    row = events_of(con, "MPTI")[date(2025, 3, 10)]
    assert (row["status"], row["reason"], row["known_on"]) == (
        "restart",
        "child_too_late",
        date(2025, 4, 7),
    )


def test_nvs_sandoz_never_priced(con: duckdb.DuckDBPyConnection) -> None:
    row = events_of(con, "NVS")[date(2023, 10, 4)]
    assert (row["status"], row["reason"], row["known_on"]) == (
        "restart",
        "no_child_price",
        date(2023, 11, 1),
    )


def test_cr_same_symbol(con: duckdb.DuckDBPyConnection) -> None:
    row = events_of(con, "CR")[date(2023, 4, 4)]
    assert (row["status"], row["reason"], row["known_on"]) == (
        "restart",
        "same_symbol",
        date(2023, 4, 4),
    )


def test_hon_2026_06_29_is_rejected_and_solstice_stays(con: duckdb.DuckDBPyConnection) -> None:
    from asof.store.spinoffs import visible_spinoffs

    hon = events_of(con, "HON")
    assert date(2026, 6, 29) not in hon
    assert date(2025, 10, 30) in hon
    stored = visible_spinoffs(con, ROOT, NOW, ["HON"], rejected=()).to_pylist()
    assert date(2026, 6, 29) in {r["ex_date"] for r in stored}


# -- the feature -----------------------------------------------------------------------------------


def bx_of(con: duckdb.DuckDBPyConnection, symbol: str, on: date) -> pd.DataFrame:
    """BX of the security carrying ``symbol`` on ``on``, from its first bar, with the splits
    and spin-offs known at NOW."""
    from asof.features.bx import adjust_for_splits, bx_features
    from asof.store.spinoffs import spinoff_events

    bars = visible_bars(con, ROOT, "1Day", NOW, symbols=[symbol]).to_pandas()
    [sid] = set(bars.loc[bars["session_date"] == on, "security_id"])
    bars = bars[bars["security_id"] == sid]
    daily = adjust_for_splits(bars, visible_splits(con, ROOT, NOW, [symbol]).to_pandas())
    spin = spinoff_events(con, ROOT, NOW, parents=[symbol]).to_pandas()
    out = bx_features(daily, spinoffs=spin)
    return pd.DataFrame(out[out["security_id"] == sid]).set_index("session_date")


def test_ge_bx_matches_tradingview_with_both_spin_offs(con: duckdb.DuckDBPyConnection) -> None:
    ge = bx_of(con, "GE", date(2024, 4, 1))

    assert ge.loc[date(2025, 6, 27), "bx_daily"] == pytest.approx(21.05, abs=0.01)
    assert ge.loc[date(2025, 6, 27), "bx_weekly"] == pytest.approx(21.50, abs=0.01)
    assert ge.loc[date(2025, 6, 30), "bx_monthly"] == pytest.approx(28.93, abs=0.01)
    assert ge.loc[date(2025, 12, 31), "bx_monthly"] == pytest.approx(22.20, abs=0.01)
    assert ge.loc[date(2026, 9, 30), "bx_monthly"] == pytest.approx(-19.61, abs=0.01)
    since = {date.fromisoformat(str(v)[:10]) for v in ge["bx_since"]}
    assert since == {date.fromisoformat(str(ge.index.min())[:10])}  # two adjusts, no restart


def test_nvs_bx_restarts_on_the_ex_date_twenty_sessions_later(
    con: duckdb.DuckDBPyConnection,
) -> None:
    nvs = bx_of(con, "NVS", date(2023, 10, 3))
    ex, known_on = date(2023, 10, 4), date(2023, 11, 1)

    window = nvs.loc[(nvs.index >= ex) & (nvs.index < known_on)]
    assert len(window) == 20
    assert window[["bx_daily", "bx_weekly", "bx_monthly"]].isna().all().all()
    assert not pd.isna(nvs.loc[known_on, "bx_daily"])
    since = {date.fromisoformat(str(v)[:10]) for v in nvs.loc[nvs.index >= known_on, "bx_since"]}
    assert since == {ex}


# -- the audit -------------------------------------------------------------------------------------


def test_bbig_tyde_is_a_disproved_restart(con: duckdb.DuckDBPyConnection) -> None:
    # TYDE fell 3.10 -> 2.03 on its first day: priced at that close BBIG reads -19.6%
    row = events_of(con, "BBIG")[date(2022, 6, 30)]
    assert (row["status"], row["reason"], row["known_on"]) == (
        "restart",
        "factor_disproved",
        date(2022, 6, 30),
    )
    assert row["total_return"] == pytest.approx(-0.196, abs=1e-3)
    assert row["factor"] is None


def test_audit_lists_the_disproved_spin_offs(con: duckdb.DuckDBPyConnection) -> None:
    # the CLI's scope: tier from 2020-01 through NOW, exceptions from the shipped file
    qualified = tier_union(con, ROOT, date(2020, 1, 1), NOW.date(), classes=None)
    liquid_ids = {r["security_id"] for r in qualified.to_pylist() if r["class"] in LIQUID_CLASSES}
    [result] = run_audit(
        con,
        ROOT,
        as_of=NOW,
        liquid_ids=liquid_ids,
        exceptions=load_exceptions(REPO_CONFIGS / "audit_exceptions.yaml"),
        checks=[SPINOFF_CHECK],
        liquid_tier=qualified,
    )

    liquid = [r["key"] for r in result.rows.to_pylist() if r["in_liquid"]]
    assert liquid == ["BBIG:2022-06-30"]
    assert "HON:2026-06-29" not in result.rows.column("key").to_pylist()
    assert result.fails("liquid") is False
