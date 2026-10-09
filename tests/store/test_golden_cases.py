"""Guards against the real store drifting away from facts a human has checked: each case is a
security whose history was verified by hand (SEC filings, the exchange's notices, TradingView),
asserted through the same read path strategies use. A rebuild of the master, a new rule or a
refetch that silently breaks one of them fails here, where 700 unit tests on fake data would
stay green.

Runs on the real store under ``$ASOF_DATA_DIR/market`` (default /data/asof) and skips when it is
absent, so CI does not run it. Values were read off the store on 2026-10-08 and checked by hand.
"""

from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from asof.ingest.instrument_class import CLASS_FILE
from asof.ingest.security_master import MASTER_FILE
from asof.store.audit import run_audit
from asof.store.db import default_data_dir
from asof.store.market import market_root, visible_bars, visible_master, visible_splits
from asof.store.universe import liquid_securities

ET = ZoneInfo("America/New_York")
ROOT = market_root(default_data_dir())
NOW = datetime(2026, 10, 9, tzinfo=UTC)  # after the last bar of the verified store

pytestmark = [
    pytest.mark.realdata,
    pytest.mark.skipif(
        not (ROOT / MASTER_FILE).exists() or not (ROOT / CLASS_FILE).exists(),
        reason=f"no real store under {ROOT}",
    ),
]


def ny(year: int, month: int, day: int, hour: int = 0) -> datetime:
    return datetime(year, month, day, hour, tzinfo=ET)


@pytest.fixture(scope="module")
def con() -> duckdb.DuckDBPyConnection:
    return duckdb.connect()


def segments(
    con: duckdb.DuckDBPyConnection, as_of: datetime, *symbols: str
) -> list[tuple[str, str, date, date | None]]:
    rows = visible_master(con, ROOT, as_of).to_pylist()
    return [
        (r["security_id"], r["symbol"], r["valid_from"], r["valid_to"])
        for r in rows
        if r["symbol"] in symbols
    ]


def ids(con: duckdb.DuckDBPyConnection, *symbols: str) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {s: set() for s in symbols}
    for sid, symbol, _, _ in segments(con, NOW, *symbols):
        found[symbol].add(sid)
    return found


def bars(con: duckdb.DuckDBPyConnection, as_of: datetime, symbols: list[str], **kw) -> pa.Table:
    return visible_bars(con, ROOT, "1Day", as_of, symbols=symbols, **kw)


def chain(con: duckdb.DuckDBPyConnection, security_id: str) -> list[tuple[str, date, date | None]]:
    rows = visible_master(con, ROOT, NOW).to_pylist()
    return [
        (r["symbol"], r["valid_from"], r["valid_to"])
        for r in rows
        if r["security_id"] == security_id
    ]


# -- identity ------------------------------------------------------------------------------------


def test_sivb_is_one_security_that_stopped_trading_on_2023_03_09(
    con: duckdb.DuckDBPyConnection,
) -> None:
    table = bars(con, NOW, ["SIVB", "SIVBQ"], include_untraded=True)
    assert set(table.column("security_id").to_pylist()) == {"S005506"}
    assert set(table.column("symbol").to_pylist()) == {"SIVB"}  # SIVBQ is OTC, not stored
    assert table.num_rows == 802
    assert max(table.column("session_date").to_pylist()) == date(2023, 3, 9)


def test_mmc_became_mrsh_and_the_rename_is_unknown_before_it_happened(
    con: duckdb.DuckDBPyConnection,
) -> None:
    assert segments(con, ny(2025, 12, 31, 12), "MMC", "MRSH") == [
        ("S003965", "MMC", date(2020, 1, 2), None)
    ]
    assert segments(con, ny(2026, 2, 1), "MMC", "MRSH") == [
        ("S003965", "MMC", date(2020, 1, 2), date(2026, 1, 14)),
        ("S003965", "MRSH", date(2026, 1, 14), None),
    ]
    # one timeline: never two rows on one session
    days = bars(con, NOW, ["MMC", "MRSH"], start=ny(2025, 12, 1)).column("session_date")
    assert len(days) == len(set(days.to_pylist()))


def test_bk_in_2022_is_bk_and_bny_carries_nothing(con: duckdb.DuckDBPyConnection) -> None:
    as_of = ny(2022, 3, 1)
    assert segments(con, as_of, "BK", "BNY") == [("S000730", "BK", date(2020, 1, 2), None)]
    table = bars(con, as_of, ["BK", "BNY"], start=ny(2022, 2, 1))
    assert set(table.column("symbol").to_pylist()) == {"BK"}  # BNY's copy of BK's past is gone
    assert set(table.column("security_id").to_pylist()) == {"S000730"}
    assert chain(con, "S000730") == [
        ("BK", date(2020, 1, 2), date(2026, 5, 21)),
        ("BNY", date(2026, 5, 21), None),
    ]


def test_bbby_is_two_companies(con: duckdb.DuckDBPyConnection) -> None:
    # Bed Bath & Beyond (CIK 886158) delisted 2023; Beyond Inc (ex-Overstock, CIK 1130713) took
    # the ticker on 2025-08-29 and renamed to NXH on 2026-08-17
    assert ids(con, "BBBY")["BBBY"] == {"S000610", "S004509"}
    assert chain(con, "S000610") == [("BBBY", date(2020, 1, 2), date(2023, 6, 1))]
    assert chain(con, "S004509") == [
        ("OSTK", date(2020, 1, 2), date(2023, 11, 6)),
        ("BYON", date(2023, 11, 6), date(2025, 8, 29)),
        ("BBBY", date(2025, 8, 29), date(2026, 8, 17)),
        ("NXH", date(2026, 8, 17), None),
    ]
    master = visible_master(con, ROOT, NOW).to_pylist()
    ciks = {r["security_id"]: r["cik"] for r in master if r["symbol"] == "BBBY"}
    assert ciks == {"S000610": 886158, "S004509": 1130713}


def test_share_classes_with_one_cik_stay_separate(con: duckdb.DuckDBPyConnection) -> None:
    found = ids(con, "GOOG", "GOOGL", "GME", "GME.WS", "BRK.A", "BRK.B")
    assert found["GOOG"] == {"S002639"} and found["GOOGL"] == {"S002640"}
    assert found["GME"] == {"S002602"} and found["GME.WS"] == {"S016926"}
    assert found["BRK.A"] == {"S007209"} and found["BRK.B"] == {"S007210"}
    master = visible_master(con, ROOT, NOW).to_pylist()
    cik = {r["symbol"]: r["cik"] for r in master if r["symbol"] in found}
    assert cik["GOOG"] == cik["GOOGL"] == 1652044
    assert cik["GME"] == cik["GME.WS"] == 1326380


@pytest.mark.xfail(
    strict=True,
    reason="BRK.B's first five months of 2020 were dropped at write time in 4b-2; the full "
    "refetch (4b-2b step 5) brings them back -- then this passes and the marker must go",
)
def test_brk_b_starts_on_the_first_session_of_2020(con: duckdb.DuckDBPyConnection) -> None:
    days = bars(con, NOW, ["BRK.B"]).column("session_date").to_pylist()
    assert min(days) == date(2020, 1, 2)


def test_renamed_watchlist_names_kept_their_identity(con: duckdb.DuckDBPyConnection) -> None:
    # Pure Storage -> P (2026-04-17) and EchoStar SATS -> ECHO (2026-06-24): renames, not exits
    assert chain(con, "S004882") == [
        ("PSTG", date(2020, 1, 2), date(2026, 4, 17)),
        ("P", date(2026, 4, 17), None),
    ]
    assert chain(con, "S005312") == [
        ("SATS", date(2020, 1, 2), date(2026, 6, 24)),
        ("ECHO", date(2026, 6, 24), None),
    ]


def test_ffr_and_para_were_reused_by_unrelated_securities(con: duckdb.DuckDBPyConnection) -> None:
    assert ids(con, "FFR")["FFR"] == {"S002191", "S005227"}
    assert chain(con, "S002191") == [
        ("FFR", date(2020, 1, 2), date(2022, 10, 3)),
        ("DTRE", date(2022, 10, 3), None),
    ]
    assert [s for s, _, _ in chain(con, "S005227")] == ["RTTR", "QLGN", "AIXC", "FFR"]
    # Paramount (VIAC -> PARA, merged away 2025-08-07) and Banzai (VII -> BNZI -> PARA)
    assert ids(con, "PARA")["PARA"] == {"S006965", "S006441"}
    assert chain(con, "S006965") == [
        ("VIAC", date(2020, 2, 13), date(2022, 2, 17)),
        ("PARA", date(2022, 2, 17), date(2026, 8, 7)),
    ]
    assert [s for s, _, _ in chain(con, "S006441")] == ["VII", "BNZI", "PARA"]
    assert ids(con, "VII")["VII"] == {"S006441", "S018729"}  # VII reused again in 2026


def test_nkla_appears_only_once_the_rename_was_filed(con: duckdb.DuckDBPyConnection) -> None:
    # VTIQ -> NKLA took effect 2020-06-04 but Alpaca filed it 2020-06-08; until then the reader
    # sees VTIQ's frozen placeholders, from then on one timeline under NKLA
    early = bars(con, ny(2020, 6, 6), ["VTIQ", "NKLA"], start=ny(2020, 6, 1), include_untraded=True)
    assert [(r["symbol"], r["session_date"], r["traded"]) for r in early.to_pylist()] == [
        ("VTIQ", date(2020, 6, 1), True),
        ("VTIQ", date(2020, 6, 2), True),
        ("VTIQ", date(2020, 6, 3), True),
        ("VTIQ", date(2020, 6, 4), False),
        ("VTIQ", date(2020, 6, 5), False),
    ]
    late = bars(
        con,
        datetime(2020, 6, 8, 4, tzinfo=UTC),
        ["VTIQ", "NKLA"],
        start=ny(2020, 6, 1),
        include_untraded=True,
    )
    assert [(r["symbol"], r["session_date"]) for r in late.to_pylist()][-2:] == [
        ("NKLA", date(2020, 6, 4)),
        ("NKLA", date(2020, 6, 5)),
    ]
    assert set(late.column("security_id").to_pylist()) == {"S006538"}


def test_bbuc_bars_before_the_filing_belong_to_the_old_holder(
    con: duckdb.DuckDBPyConnection,
) -> None:
    # Brookfield Business: S013086 ended under BBUC on 2026-03-31, filed 2026-04-21
    window = {"start": ny(2026, 4, 1), "end": ny(2026, 4, 10)}
    then = bars(con, ny(2026, 4, 10), ["BBUC"], **window)
    now = bars(con, NOW, ["BBUC"], **window)
    assert set(then.column("security_id").to_pylist()) == {"S013086"}
    assert set(now.column("security_id").to_pylist()) == {"S000629"}
    assert then.num_rows == now.num_rows == 6


def test_arnc_two_events_on_one_day_leave_no_overlap(con: duckdb.DuckDBPyConnection) -> None:
    # 2020-04-01: Arconic Inc renamed itself Howmet (HWM) and the spun-off Arconic Corp took ARNC
    assert segments(con, ny(2020, 3, 31, 12), "ARNC", "HWM") == [
        ("S000437", "ARNC", date(2020, 1, 2), None)
    ]
    assert segments(con, datetime(2020, 4, 1, 5, tzinfo=UTC), "ARNC", "HWM") == [
        ("S000437", "ARNC", date(2020, 1, 2), date(2020, 4, 1)),
        ("S000437", "HWM", date(2020, 4, 1), None),
        ("S007015", "ARNC", date(2020, 4, 1), None),
    ]
    # known imprecision: the SEC kept ARNC under Arconic Inc's CIK until 2020-07-10, and so does
    # the master; no check can see it because both sides agree
    master = visible_master(con, ROOT, NOW).to_pylist()
    arnc = [(r["valid_from"], r["cik"]) for r in master if r["security_id"] == "S007015"]
    assert arnc == [(date(2020, 4, 1), 4281), (date(2020, 7, 10), 1790982)]


# -- splits --------------------------------------------------------------------------------------


def test_split_days_are_flagged_and_the_raw_prices_are_unadjusted(
    con: duckdb.DuckDBPyConnection,
) -> None:
    splits = visible_splits(con, ROOT, NOW, ["NVDA", "AAPL", "CMND"]).to_pylist()
    factors = {(r["symbol"], r["ex_date"]): r["factor"] for r in splits}
    assert factors[("NVDA", date(2021, 7, 20))] == 4.0
    assert factors[("NVDA", date(2024, 6, 10))] == 10.0
    assert factors[("AAPL", date(2020, 8, 31))] == 4.0
    assert factors[("CMND", date(2026, 10, 5))] == pytest.approx(1 / 8)

    flagged = {
        (r["symbol"], r["session_date"])
        for r in bars(con, NOW, ["NVDA", "AAPL", "CMND"]).to_pylist()
        if r["split_day"]
    }
    assert {
        ("NVDA", date(2021, 7, 20)),
        ("NVDA", date(2024, 6, 10)),
        ("AAPL", date(2020, 8, 31)),
        ("CMND", date(2026, 10, 5)),
    } <= flagged

    nvda = bars(con, NOW, ["NVDA"], start=ny(2024, 6, 7), end=ny(2024, 6, 11)).to_pylist()
    assert [r["close"] for r in nvda] == [1208.88, 121.79]


def test_hcti_reverse_splits_compound_to_the_tradingview_price(
    con: duckdb.DuckDBPyConnection,
) -> None:
    # three reverse splits (1:10, 1:249, 1:60); TradingView shows 2025-06-12's high adjusted to
    # the 2026 share count as about 647
    splits = visible_splits(con, ROOT, NOW, ["HCTI"]).to_pylist()
    assert [(r["ex_date"], round(1 / r["factor"])) for r in splits] == [
        (date(2023, 5, 26), 10),
        (date(2025, 8, 1), 249),
        (date(2026, 2, 10), 60),
    ]
    [bar] = bars(con, NOW, ["HCTI"], start=ny(2025, 6, 12), end=ny(2025, 6, 13)).to_pylist()
    later = [r["factor"] for r in splits if r["ex_date"] > date(2025, 6, 12)]
    adjusted = bar["high"] / (later[0] * later[1])
    assert 640 <= adjusted <= 660


# -- instrument classes and the liquid tier ------------------------------------------------------


def test_instrument_classes_of_reference_symbols() -> None:
    rows = pq.read_table(ROOT / CLASS_FILE).to_pylist()
    first = {}
    for r in sorted(rows, key=lambda r: r["valid_from"]):
        first.setdefault(r["symbol"], r)
    expected = {
        "TQQQ": "leveraged_etf",
        "TVIX": "leveraged_etf",  # by override: the issuer's name says nothing
        "SPY": "etf",
        "AAPL": "common",
        "PFBC": "common",  # "Preferred Bank": a name, not a share class
        "FBRT": "common",
        "BAC.PRB": "preferred",
        "ACIC.U": "unit",
        "GME.WS": "warrant",
        "EXEEW": "warrant",
    }
    assert {s: first[s]["class"] for s in expected} == expected
    assert first["TVIX"]["rule"] == "override"
    # BEAT: BioTelemetry until its 2021 takeover, HeartBeam afterwards; two securities
    beat = sorted((r for r in rows if r["symbol"] == "BEAT"), key=lambda r: r["valid_from"])
    assert [r["name"] for r in beat] == ["BIOTELEMETRY, INC.", "Heartbeam, Inc. Common Stock"]
    assert beat[0]["security_id"] != beat[1]["security_id"]


def test_liquid_tier_membership_at_reference_dates(con: duckdb.DuckDBPyConnection) -> None:
    def tier(as_of: datetime) -> dict[str, str]:
        return {
            r["symbol"]: r["security_id"] for r in liquid_securities(con, ROOT, as_of).to_pylist()
        }

    march_2021 = tier(ny(2021, 3, 1))
    assert {s: march_2021.get(s) for s in ("SIVB", "GME", "AMC", "BBBY")} == {
        "SIVB": "S005506",
        "GME": "S002602",
        "AMC": "S000291",
        "BBBY": "S000610",
    }
    assert "SIVB" not in tier(ny(2026, 10, 1))
    june_2022 = tier(ny(2022, 6, 1))
    assert june_2022.get("BK") == "S000730" and "BNY" not in june_2022
    assert "EXEEW" not in tier(ny(2026, 2, 4))


# -- the audit on the real store -----------------------------------------------------------------


def test_audit_numbers_that_must_hold(con: duckdb.DuckDBPyConnection) -> None:
    results = {
        r.spec.id: r
        for r in run_audit(
            con,
            ROOT,
            as_of=NOW,
            liquid_ids=set(),
            checks=["P1", "P2", "P3", "P4", "P5", "M1", "M4", "I4", "I7", "S3"],
        )
    }
    for check in ("P1", "P2", "P3", "P4", "P5", "M1", "M4"):
        assert results[check].total == 0, check
    assert results["I7"].notes["min_traded_symbols"] > 1_000
    i4 = {r["key"]: r["bars"] for r in results["I4"].rows.to_pylist()}
    assert i4["BBUC"] == 14
    s3 = {r["key"]: r for r in results["S3"].rows.to_pylist()}
    exeew = s3["EXEEW:2026-02-03"]
    assert (exeew["prev_close"], exeew["close"], exeew["next_close"]) == (105.0, 0.0101, 125.0)
