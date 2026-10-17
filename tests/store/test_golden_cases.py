"""Guards against the real store drifting away from facts a human has checked: each case is a
security whose history was verified by hand (SEC filings, the exchange's notices, TradingView),
asserted through the same read path strategies use. A rebuild of the master, a new rule or a
refetch that silently breaks one of them fails here, where 700 unit tests on fake data would
stay green.

Runs on the real store under ``$ASOF_DATA_DIR/market`` (default /data/asof) and skips when it is
absent, so CI does not run it. Values were read off the store on 2026-10-08 and checked by hand.
"""

from datetime import UTC, date, datetime, timedelta
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


def sid(con: duckdb.DuckDBPyConnection, symbol: str, day: date) -> str:
    """The security that held ``symbol`` on ``day``, as the master is known at NOW. Ids are
    assigned by content and shift whenever the store is refetched, so cases name a security by
    a symbol and a date rather than by id."""
    found = {
        security_id
        for security_id, _, start, end in segments(con, NOW, symbol)
        if start <= day and (end is None or day < end)
    }
    assert len(found) == 1, (symbol, day, found)
    return found.pop()


# -- identity ------------------------------------------------------------------------------------


def test_sivb_is_one_security_that_stopped_trading_on_2023_03_09(
    con: duckdb.DuckDBPyConnection,
) -> None:
    table = bars(con, NOW, ["SIVB", "SIVBQ"], include_untraded=True)
    assert set(table.column("security_id").to_pylist()) == {sid(con, "SIVB", date(2022, 1, 3))}
    assert set(table.column("symbol").to_pylist()) == {"SIVB"}  # SIVBQ is OTC, not stored
    assert table.num_rows == 802
    assert max(table.column("session_date").to_pylist()) == date(2023, 3, 9)


def test_mmc_became_mrsh_and_the_rename_is_unknown_before_it_happened(
    con: duckdb.DuckDBPyConnection,
) -> None:
    mmc = sid(con, "MMC", date(2020, 1, 2))
    assert segments(con, ny(2025, 12, 31, 12), "MMC", "MRSH") == [
        (mmc, "MMC", date(2020, 1, 2), None)
    ]
    assert segments(con, ny(2026, 2, 1), "MMC", "MRSH") == [
        (mmc, "MMC", date(2020, 1, 2), date(2026, 1, 14)),
        (mmc, "MRSH", date(2026, 1, 14), None),
    ]
    # one timeline: never two rows on one session
    days = bars(con, NOW, ["MMC", "MRSH"], start=ny(2025, 12, 1)).column("session_date")
    assert len(days) == len(set(days.to_pylist()))


def test_bk_in_2022_is_bk_and_bny_carries_nothing(con: duckdb.DuckDBPyConnection) -> None:
    as_of = ny(2022, 3, 1)
    bk = sid(con, "BK", date(2022, 3, 1))
    assert segments(con, as_of, "BK", "BNY") == [(bk, "BK", date(2020, 1, 2), None)]
    table = bars(con, as_of, ["BK", "BNY"], start=ny(2022, 2, 1))
    assert set(table.column("symbol").to_pylist()) == {"BK"}  # BNY's copy of BK's past is gone
    assert set(table.column("security_id").to_pylist()) == {bk}
    assert chain(con, bk) == [
        ("BK", date(2020, 1, 2), date(2026, 5, 21)),
        ("BNY", date(2026, 5, 21), None),
    ]


def test_bbby_is_two_companies(con: duckdb.DuckDBPyConnection) -> None:
    # Bed Bath & Beyond (CIK 886158) delisted 2023; Beyond Inc (ex-Overstock, CIK 1130713) took
    # the ticker on 2025-08-29 and renamed to NXH on 2026-08-17
    bed_bath = sid(con, "BBBY", date(2021, 1, 4))
    beyond = sid(con, "BBBY", date(2025, 9, 2))
    assert ids(con, "BBBY")["BBBY"] == {bed_bath, beyond}
    assert chain(con, bed_bath) == [("BBBY", date(2020, 1, 2), date(2023, 6, 1))]
    assert chain(con, beyond) == [
        ("OSTK", date(2020, 1, 2), date(2023, 11, 6)),
        ("BYON", date(2023, 11, 6), date(2025, 8, 29)),
        ("BBBY", date(2025, 8, 29), date(2026, 8, 17)),
        ("NXH", date(2026, 8, 17), None),
    ]
    master = visible_master(con, ROOT, NOW).to_pylist()
    ciks = {r["security_id"]: r["cik"] for r in master if r["symbol"] == "BBBY"}
    assert ciks == {bed_bath: 886158, beyond: 1130713}


def test_share_classes_with_one_cik_stay_separate(con: duckdb.DuckDBPyConnection) -> None:
    found = ids(con, "GOOG", "GOOGL", "GME", "GME.WS", "BRK.A", "BRK.B")
    assert all(len(found[s]) == 1 for s in found)
    assert found["GOOG"] != found["GOOGL"]
    assert found["GME"] != found["GME.WS"]
    assert found["BRK.A"] != found["BRK.B"]
    master = visible_master(con, ROOT, NOW).to_pylist()
    cik = {r["symbol"]: r["cik"] for r in master if r["symbol"] in found}
    assert cik["GOOG"] == cik["GOOGL"] == 1652044
    assert cik["GME"] == cik["GME.WS"] == 1326380


def test_brk_b_starts_on_the_first_session_of_2020(con: duckdb.DuckDBPyConnection) -> None:
    # 4b-2 dropped BRK.B's first five months of 2020 at write time (the SEC spelled it BRKB
    # until June); the 4b-2b refetch, which keeps everything Alpaca returns, brought them back
    days = bars(con, NOW, ["BRK.B"]).column("session_date").to_pylist()
    assert min(days) == date(2020, 1, 2)


def test_renamed_watchlist_names_kept_their_identity(con: duckdb.DuckDBPyConnection) -> None:
    # Pure Storage -> P (2026-04-17) and EchoStar SATS -> ECHO (2026-06-24): renames, not exits
    assert chain(con, sid(con, "PSTG", date(2021, 1, 4))) == [
        ("PSTG", date(2020, 1, 2), date(2026, 4, 17)),
        ("P", date(2026, 4, 17), None),
    ]
    assert chain(con, sid(con, "SATS", date(2021, 1, 4))) == [
        ("SATS", date(2020, 1, 2), date(2026, 6, 24)),
        ("ECHO", date(2026, 6, 24), None),
    ]


def test_ffr_and_para_were_reused_by_unrelated_securities(con: duckdb.DuckDBPyConnection) -> None:
    # FFR: First Trust's ETF renamed to DTRE on 2022-10-03 and Ritter (RTTR -> QLGN -> AIXC ->
    # FFR) took the label in 2026. Alpaca then rewrote FFR's past with Ritter's chain: the 2020-
    # 2022 bars under FFR close at 0.27-0.29 (Ritter's level, First Trust's ETF traded near 40)
    # and are OHLC-identical to QLGN / AIXC / RTTR. Since 6-b-2 they belong to nobody and the
    # label has one security; DTRE's own history before the record is a known loss (I1) until
    # a second source supplies First Trust's FFR bars (the PARA path).
    ritter = sid(con, "FFR", date(2026, 10, 1))
    assert ids(con, "FFR")["FFR"] == {ritter}
    assert [s for s, _, _ in chain(con, ritter)] == ["RTTR", "QLGN", "AIXC", "FFR"]
    dtre = sid(con, "DTRE", date(2023, 1, 3))
    assert dtre != ritter
    assert chain(con, dtre) == [("DTRE", date(2022, 10, 3), None)]
    assert bars(con, NOW, ["FFR"], start=ny(2021, 1, 4), end=ny(2021, 1, 5)).num_rows == 0
    # Paramount (VIAC -> PARA, merged away 2025-08-07) and Banzai (VII -> BNZI -> PARA)
    paramount = sid(con, "PARA", date(2023, 1, 3))
    banzai = sid(con, "PARA", date(2026, 10, 1))
    assert ids(con, "PARA")["PARA"] == {paramount, banzai}
    assert chain(con, paramount) == [
        # CBS -> VIAC was filed 2020-02-13 for a rename of 2019-12-05; the label lives from
        # its first stored bar (rule V, 2026-10-16)
        ("VIAC", date(2020, 1, 2), date(2022, 2, 17)),
        ("PARA", date(2022, 2, 17), date(2025, 8, 7)),
    ]
    assert [s for s, _, _ in chain(con, banzai)] == ["VII", "BNZI", "PARA"]
    # VII reused again from 2026-07-16 by an unrelated listing
    assert ids(con, "VII")["VII"] == {banzai, sid(con, "VII", date(2026, 8, 3))}


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
    assert set(late.column("security_id").to_pylist()) == {sid(con, "NKLA", date(2020, 7, 1))}


def test_bbuc_bars_before_the_filing_belong_to_the_old_holder(
    con: duckdb.DuckDBPyConnection,
) -> None:
    # Brookfield Business Corp ended under BBUC on 2026-03-31, filed 2026-04-21; Brookfield
    # Business Partners (BBU) took the symbol that day
    window = {"start": ny(2026, 4, 1), "end": ny(2026, 4, 10)}
    then = bars(con, ny(2026, 4, 10), ["BBUC"], **window)
    now = bars(con, NOW, ["BBUC"], **window)
    corp, partners = sid(con, "BBUC", date(2026, 3, 2)), sid(con, "BBUC", date(2026, 5, 1))
    assert corp != partners
    assert set(then.column("security_id").to_pylist()) == {corp}
    assert set(now.column("security_id").to_pylist()) == {partners}
    assert then.num_rows == now.num_rows == 6


def test_arnc_two_events_on_one_day_leave_no_overlap(con: duckdb.DuckDBPyConnection) -> None:
    # 2020-04-01: Arconic Inc renamed itself Howmet (HWM) and the spun-off Arconic Corp took ARNC
    arconic_inc = sid(con, "ARNC", date(2020, 1, 2))
    arconic_corp = sid(con, "ARNC", date(2020, 4, 1))
    assert segments(con, ny(2020, 3, 31, 12), "ARNC", "HWM") == [
        (arconic_inc, "ARNC", date(2020, 1, 2), None)
    ]
    assert segments(con, datetime(2020, 4, 1, 5, tzinfo=UTC), "ARNC", "HWM") == [
        (arconic_inc, "ARNC", date(2020, 1, 2), date(2020, 4, 1)),
        (arconic_inc, "HWM", date(2020, 4, 1), None),
        (arconic_corp, "ARNC", date(2020, 4, 1), None),
    ]
    # known imprecision: the SEC kept ARNC under Arconic Inc's CIK until 2020-07-10, and so does
    # the master; no check can see it because both sides agree
    master = visible_master(con, ROOT, NOW).to_pylist()
    arnc = [(r["valid_from"], r["cik"]) for r in master if r["security_id"] == arconic_corp]
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
        s: sid(con, s, date(2021, 3, 1)) for s in ("SIVB", "GME", "AMC", "BBBY")
    }
    assert "SIVB" not in tier(ny(2026, 10, 1))
    june_2022 = tier(ny(2022, 6, 1))
    assert june_2022.get("BK") == sid(con, "BK", date(2022, 6, 1)) and "BNY" not in june_2022
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
    min_traded = results["I7"].notes["min_traded_symbols"]
    assert isinstance(min_traded, int) and min_traded > 1_000
    i4 = {r["key"]: r["bars"] for r in results["I4"].rows.to_pylist()}
    assert i4["BBUC"] == 14
    s3 = {r["key"]: r for r in results["S3"].rows.to_pylist()}
    exeew = s3["EXEEW:2026-02-03"]
    assert (exeew["prev_close"], exeew["close"], exeew["next_close"]) == (105.0, 0.0101, 125.0)


# -- the second bar source -----------------------------------------------------------------------


def test_para_2022_2025_is_paramount_from_the_second_source(
    con: duckdb.DuckDBPyConnection,
) -> None:
    # Alpaca rewrote PARA with Banzai's chain (VII -> BNZI -> PARA) after the 2026-08-07 reuse;
    # Paramount Global's class B bars 2022-02-17 .. 2025-08-06 come from Yahoo (symbol SKYD),
    # imported as the first bar_imports entry on 2026-10-16. Checked by hand: 2024-06-03 close
    # 12.80 (Paramount), while Banzai's PARA row for that day closes at 0.1902.
    as_of = ny(2026, 10, 16)
    day = ny(2024, 6, 3)
    window = {"start": day, "end": ny(2024, 6, 4)}
    [row] = bars(con, as_of, ["PARA"], **window).to_pylist()
    assert row["source"] == "yahoo"
    assert row["close"] == pytest.approx(12.80, abs=1e-4)
    assert row["close"] != pytest.approx(0.1902, abs=1e-4)
    assert row["security_id"] == sid(con, "VIAC", date(2021, 6, 1))
    # the imported bar obeys the daily availability rule like any other
    assert bars(con, ny(2024, 6, 3, 19), ["PARA"], **window).num_rows == 0
    assert bars(con, ny(2024, 6, 3, 20), ["PARA"], **window).num_rows == 1
    # inside the imported range no Alpaca row for PARA is served any more
    imported = bars(
        con,
        as_of,
        ["PARA"],
        start=ny(2022, 2, 17),
        end=ny(2025, 8, 7),
        resolve=False,
        include_untraded=True,
    )
    assert imported.column("source").to_pylist().count("alpaca") == 0
    assert imported.num_rows == 869


def test_banzai_bars_under_para_do_not_resolve_to_paramount(
    con: duckdb.DuckDBPyConnection,
) -> None:
    # Measured 2026-10-16: 251 Alpaca PARA bars 2025-08-07 .. 2026-08-06 (Banzai, close
    # 0.22 .. 7.66) come back with Paramount's security_id. 6-b-2 rule F ends Paramount's PARA
    # segment on 2025-08-07, where the Banzai copies begin (its CIK 813828 leaves the SEC table
    # after the 2025-08-19 snapshot); the copies then belong to nobody.
    rows = bars(
        con, ny(2026, 10, 16), ["PARA"], start=ny(2025, 8, 7), end=ny(2026, 8, 7)
    ).to_pylist()
    paramount = sid(con, "VIAC", date(2021, 6, 1))
    assert all(r["security_id"] != paramount for r in rows)


# -- step 6-b rules (N, D, V, O, A), written 2026-10-16 to FAIL on the store built before them --
# Fake-world tests: tests/ingest/test_security_master_rules.py. Values below were read off the
# store on 2026-10-16 (bars, SEC rows, Alpaca records) and checked against the companies' own
# notices; the master rows they contradict are quoted in each case.


def test_kinzu_carries_the_cik_the_sec_wrote_under_kinz_u(con: duckdb.DuckDBPyConnection) -> None:
    # N: the SEC wrote KINZ.U once (2020-11-03) and KINZU 21 times after; the store has KINZU.
    # Master before the rule: KINZU [2021-01-04, None) bars_only, cik None.
    master = visible_master(con, ROOT, NOW).to_pylist()
    rows = [r for r in master if r["symbol"] == "KINZU"]
    assert rows and all(r["cik"] == 1820875 for r in rows), rows
    assert not any(r["symbol"] == "KINZ.U" for r in master)


def test_svmk_became_mntv_on_2021_06_15_although_alpaca_filed_it_in_august(
    con: duckdb.DuckDBPyConnection,
) -> None:
    # D: SVMK's last bar 2021-06-14, MNTV's first 06-15, Alpaca's record 08-09. Master before
    # the rule: SVMK to 08-09, MNTV [06-15, 08-09) + [08-09, None) -- an overlap.
    momentive = sid(con, "SVMK", date(2021, 1, 4))
    assert chain(con, momentive) == [
        ("SVMK", date(2020, 1, 2), date(2021, 6, 15)),
        ("MNTV", date(2021, 6, 15), None),
    ]
    assert ids(con, "SVMK", "MNTV") == {"SVMK": {momentive}, "MNTV": {momentive}}
    # the rename is knowable on 08-09: a reader on 07-01 sees SVMK open and no MNTV
    assert segments(con, ny(2021, 7, 1), "SVMK", "MNTV") == [
        (momentive, "SVMK", date(2020, 1, 2), None)
    ]


def test_viac_and_worx_start_at_their_first_bar_not_at_the_late_record(
    con: duckdb.DuckDBPyConnection,
) -> None:
    # V: CBS -> VIAC recorded 2020-02-13 with VIAC trading from 2020-01-02 and CBS never stored;
    # WORXD -> WORX recorded 2026-09-01 with WORX listed by the SEC since 2019 and trading
    # throughout. Master before the rule: VIAC from 02-13 (29 orphan bars), WORX cut in two.
    paramount = sid(con, "VIAC", date(2021, 6, 1))
    assert segments(con, NOW, "VIAC") == [(paramount, "VIAC", date(2020, 1, 2), date(2022, 2, 17))]
    assert sid(con, "VIAC", date(2020, 1, 15)) == paramount
    scworx = sid(con, "WORX", date(2021, 1, 4))
    assert segments(con, NOW, "WORX") == [(scworx, "WORX", date(2020, 1, 2), None)]


def test_jamf_and_intl_bars_after_the_vacated_placeholder_have_a_security(
    con: duckdb.DuckDBPyConnection,
) -> None:
    # O: JAMF -> JNMF (2020-01-13) vacated a pre-IPO placeholder; Jamf's real bars start
    # 2020-07-22 and the SEC lists JAMF under 1721947 from 2020-08-13. INTL -> SNEX (StoneX,
    # 2020-07-06); an ETF trades under INTL from 2022-12-02 and never reaches the SEC table.
    # Master before the rule: neither label has a segment on the dates below.
    jamf = sid(con, "JAMF", date(2021, 1, 4))
    assert jamf != sid(con, "JAMF", date(2020, 1, 2))
    master = visible_master(con, ROOT, NOW).to_pylist()
    assert {r["cik"] for r in master if r["security_id"] == jamf} == {1721947}
    etf = sid(con, "INTL", date(2023, 1, 3))
    stonex = sid(con, "INTL", date(2020, 3, 2))
    assert etf != stonex
    assert stonex == sid(con, "SNEX", date(2023, 1, 3))
    assert {r["cik"] for r in master if r["security_id"] == etf} == {None}


def test_honav_when_issued_is_not_hona(con: duckdb.DuckDBPyConnection) -> None:
    # A: HONAV traded when-issued 2026-06-15..06-26, HONAV -> HONA recorded 06-29. Master before
    # the rule: one security for both.
    hona = sid(con, "HONA", date(2026, 7, 1))
    assert sid(con, "HONAV", date(2026, 6, 16)) != hona
    assert segments(con, NOW, "HONA") == [(hona, "HONA", date(2026, 6, 29), None)]


# -- step 6-b-2 rules (B, E, F, Z, whole-chain relabel, imports in the build), written 2026-10-17
# to FAIL on the store built before them --
# Fake-world tests: tests/ingest/test_security_master_copies.py. Values below were read off the
# store on 2026-10-17 (bars, SEC rows, Alpaca records); the master rows they contradict are
# quoted in each case. Security ids are content hashes and shift on a rebuild, so every case
# names a security by (symbol, day).


def test_cycn_became_krsa_on_2026_09_09_without_a_record(con: duckdb.DuckDBPyConnection) -> None:
    # E. Every one of CYCN's 1,662 traded bars (2020-01-02 .. 2026-09-08; 17 more are volume-0
    # filler) is OHLC-identical to KRSA's; CYCN's last traded bar is 2026-09-08 (close 3.36),
    # KRSA trades on from 09-09 (open 21.20: a reverse split with the name change); no Alpaca
    # or SEC record. Master before the rule: CYCN [2020-01-02, None) 'sec' and KRSA
    # [2020-01-02, None) bars_only in two securities, conflict dup_unlinked "1609 days".
    cyclerion = sid(con, "CYCN", date(2021, 1, 4))
    assert segments(con, NOW, "CYCN", "KRSA") == [
        (cyclerion, "CYCN", date(2020, 1, 2), date(2026, 9, 9)),
        (cyclerion, "KRSA", date(2026, 9, 9), None),
    ]
    master = visible_master(con, ROOT, NOW).to_pylist()
    [krsa] = [r for r in master if r["symbol"] == "KRSA"]
    assert krsa["evidence"] == "bars"
    # knowable with KRSA's first post-run bar (20:00 New York of 2026-09-09), not before
    first_bar = datetime(2026, 9, 10, 0, 0, tzinfo=UTC)
    assert krsa["available_at"] == first_bar
    assert segments(con, first_bar - timedelta(microseconds=1), "CYCN", "KRSA") == [
        (cyclerion, "CYCN", date(2020, 1, 2), None)
    ]
    assert segments(con, first_bar, "CYCN", "KRSA") == [
        (cyclerion, "CYCN", date(2020, 1, 2), date(2026, 9, 9)),
        (cyclerion, "KRSA", date(2026, 9, 9), None),
    ]
    # one timeline: the identical days are served under CYCN only
    days = bars(con, NOW, ["CYCN", "KRSA"], start=ny(2026, 8, 1)).column("session_date")
    assert len(days) == len(set(days.to_pylist()))


def test_cuk_converted_into_ccl_on_2026_05_07_and_ccl_is_untouched(
    con: duckdb.DuckDBPyConnection,
) -> None:
    # B1. Carnival plc's ADR (CUK, CIK 1125259) converted into Carnival Corp (CCL, CIK 815097):
    # record 2026-05-08, CUK's last bar 05-06 (27.47 vs CCL 27.52), 1,594 parallel days all
    # OHLC-different, CCL under 815097 in every snapshot through 2026-09-08. Master before the
    # rule: CUK [2020-01-02, 2026-05-07) and CCL cut in two at 05-07, all one security.
    carnival_plc = sid(con, "CUK", date(2021, 1, 4))
    carnival_corp = sid(con, "CCL", date(2021, 1, 4))
    assert carnival_plc != carnival_corp
    assert segments(con, NOW, "CUK") == [(carnival_plc, "CUK", date(2020, 1, 2), date(2026, 5, 7))]
    assert segments(con, NOW, "CCL") == [(carnival_corp, "CCL", date(2020, 1, 2), None)]


def test_cot_became_prmw_in_2020_although_the_two_traded_in_parallel(
    con: duckdb.DuckDBPyConnection,
) -> None:
    # B1 negative (guard: passes on today's master, must keep passing). Cott (CIK 884713)
    # bought Primo Water and took its ticker: record COT -> PRMW 2020-03-03, 40 parallel
    # differing days, PRMW 14.00 -> 14.78 across the record. The PRMW label's CIK before the
    # record is old Primo's 1365101 -- rows the build drops as a zombie in the 2020-07-10 SEC
    # correction, so the CIK reads UNKNOWN, not "different"; either way not a conversion. COT
    # and PRMW from 03-03 are one security; nothing of old Primo's PRMW belongs to it.
    cott = sid(con, "COT", date(2020, 2, 3))
    assert sid(con, "PRMW", date(2020, 6, 1)) == cott
    assert segments(con, NOW, "COT") == [(cott, "COT", date(2020, 1, 2), date(2020, 3, 3))]
    before = [s for s in segments(con, NOW, "PRMW") if s[2] < date(2020, 3, 3)]
    assert all(s[0] != cott and s[3] == date(2020, 3, 3) for s in before), before


def test_iivip_preferred_moved_to_the_vacated_iivi_label_in_2023(
    con: duckdb.DuckDBPyConnection,
) -> None:
    # B1 negative (guard: passes on today's master, must keep passing). IIVIP -> IIVI recorded
    # 2023-02-23: II-VI's preferred took the common's vacated ticker after IIVI -> COHR
    # 2022-10-18 (IIVIP's last bar 02-22 at 187.51, IIVI from 02-23 at 195.01). The 465 parallel
    # differing days are the common's, a year earlier: no differing day in the five sessions
    # before the record, so the record is a rename and the preferred keeps one security across
    # the label move. II-VI's common (IIVI before 2022-10-18) is another security.
    preferred = sid(con, "IIVIP", date(2021, 1, 4))
    assert segments(con, NOW, "IIVIP") == [
        (preferred, "IIVIP", date(2020, 8, 7), date(2023, 2, 23))
    ]
    assert sid(con, "IIVI", date(2023, 3, 1)) == preferred
    assert sid(con, "IIVI", date(2021, 1, 4)) != preferred  # II-VI's common, COHR since 2022


def test_axnx_ends_where_the_mimi_copies_begin_after_the_2025_gap(
    con: duckdb.DuckDBPyConnection,
) -> None:
    # F, gap variant. Axonics' last own bar 2024-11-14 (70.98; acquired by Boston Scientific),
    # no traded bar for 57 days, then 245 bars from 2025-01-10 (4.11) identical to MIMI's, whose
    # SEC rows start 2024-11-28. Master before the rule: AXNX [2020-01-02, None) owns them;
    # I2 reports AXNX/MIMI.
    axonics = sid(con, "AXNX", date(2021, 1, 4))
    assert segments(con, NOW, "AXNX") == [(axonics, "AXNX", date(2020, 1, 2), date(2025, 1, 10))]
    assert bars(con, NOW, ["AXNX"], start=ny(2025, 1, 10)).num_rows == 0
    assert sid(con, "MIMI", date(2025, 6, 2)) != axonics
    # the end is knowable with the first tail bar (20:00 New York of 2025-01-10 = 01:00 UTC)
    end_known = datetime(2025, 1, 11, 1, 0, tzinfo=UTC)
    assert segments(con, end_known - timedelta(microseconds=1), "AXNX") == [
        (axonics, "AXNX", date(2020, 1, 2), None)
    ]
    assert segments(con, end_known, "AXNX") == [
        (axonics, "AXNX", date(2020, 1, 2), date(2025, 1, 10))
    ]


def test_vjet_ends_on_2024_03_25_where_the_vtak_copies_begin(
    con: duckdb.DuckDBPyConnection,
) -> None:
    # F, CIK-leaves variant. Voxeljet's own bars stop 2024-03-22 (0.6401); from 03-25 (0.501)
    # VJET's 634 bars are VTAK's; VJET's SEC rows (CIK 1582581) end 2024-04-01 and the filer
    # reappears as VJTTY from 05-02. No gap in the traded bars. Master before the rule: VJET
    # [2020-01-02, 2024-05-02); the copies from 03-25 to 05-01 are Voxeljet's, the rest nobody's.
    voxeljet = sid(con, "VJET", date(2021, 1, 4))
    assert segments(con, NOW, "VJET") == [(voxeljet, "VJET", date(2020, 1, 2), date(2024, 3, 25))]
    assert bars(con, NOW, ["VJET"], start=ny(2024, 3, 25)).num_rows == 0
    assert sid(con, "VTAK", date(2024, 6, 3)) != voxeljet


def test_btln_old_holder_segment_is_gone_and_its_bars_live_under_vrar(
    con: duckdb.DuckDBPyConnection,
) -> None:
    # Z / whole-chain relabel. The Glimpse Group (CIK 1854445): VRAR 2021-07-01 .. 2026-02-18,
    # GGRP 2026-02-19 .. 08-19, BTLN from 2026-08-20. The 1,289 BTLN bars before the takeover
    # are exact copies of VRAR (1,163) + GGRP (126); the old BTLN filer (CIK 1425289) was listed
    # 2021-06..2022-01 only. Master before the rule: BTLN [2021-07-01, 2026-08-20) under the old
    # CIK (GGRP alone covers 10% of its days), conflict reuse_looks_continuous GGRP/BTLN.
    glimpse = sid(con, "BTLN", date(2026, 9, 1))
    assert segments(con, NOW, "BTLN") == [(glimpse, "BTLN", date(2026, 8, 20), None)]
    assert chain(con, glimpse) == [
        ("VRAR", date(2021, 7, 1), date(2026, 2, 19)),
        ("GGRP", date(2026, 2, 19), date(2026, 8, 20)),
        ("BTLN", date(2026, 8, 20), None),
    ]
    window = {"start": ny(2023, 6, 1), "end": ny(2023, 6, 2)}
    assert bars(con, NOW, ["BTLN"], **window).num_rows == 0
    [vrar] = bars(con, NOW, ["VRAR"], **window).to_pylist()
    assert vrar["security_id"] == glimpse


def test_para_2023_bar_is_paramounts_through_a_build_that_reads_the_imports(
    con: duckdb.DuckDBPyConnection,
) -> None:
    # Build reads bars_1d_ext. PARA 2023-06-01: Yahoo (SKYD) close 15.39, Alpaca's Banzai row
    # 10.45. The build must see the imported row, or the whole-chain relabel / F comparison
    # runs on Banzai's copies and misjudges Paramount's piece. After F the piece ends
    # 2025-08-07 (Paramount's CIK 813828 leaves the SEC table after 2025-08-19; the Banzai
    # copies under PARA begin 08-07). Master before 6-b-2: PARA [2022-02-17, 2026-08-07).
    paramount = sid(con, "VIAC", date(2021, 6, 1))
    window = {"start": ny(2023, 6, 1), "end": ny(2023, 6, 2)}
    [row] = bars(con, NOW, ["PARA"], **window).to_pylist()
    assert row["source"] == "yahoo"
    assert row["close"] == pytest.approx(15.39, abs=1e-4)
    assert row["security_id"] == paramount
    assert (paramount, "PARA", date(2022, 2, 17), date(2025, 8, 7)) in segments(con, NOW, "PARA")
