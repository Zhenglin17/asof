"""Guards against bars being attributed to the wrong company: a reused ticker (BBBY, FFR) must
split into different securities, a renamed one (BK -> BNY) must stay one, two share classes
with one CIK must not be merged, identical price series must be reported but never merged
automatically, and no segment may be "known" before the event that created it.

Target API (asof.ingest.security_master, new module). Signatures are a proposal; the
implementation may regroup them as long as ``build_security_master`` keeps its contract.

    normalize_spelling(sec_history: pa.Table) -> pa.Table
        Same CIK and same ticker once dots are removed -> one spelling (the dotted one).
    edges_from_name_changes(name_changes: pa.Table) -> pa.Table
    edges_from_sec(sec_history: pa.Table) -> pa.Table
    merge_edges(alpaca_edges: pa.Table, sec_edges: pa.Table) -> pa.Table
        Edge columns: old_symbol, new_symbol, date (the cut), date_lower (nullable; the
        exclusive lower bound of a snapshot interval for SEC-only edges), evidence
        ('name_change' | 'sec' | 'both'), available_at.
    ohlcv_duplicates(bars: pa.Table) -> pa.Table      columns symbol_a, symbol_b, days
    detect_conflicts(master, dup_pairs, bars) -> pa.Table
    apply_overrides(master: pa.Table, overrides: Sequence[Mapping]) -> pa.Table
    build_security_master(*, bars, sec_history, name_changes, overrides=()) -> (master, conflicts)
        master columns: security_id, symbol, valid_from, valid_to, cik, cusip, evidence,
        available_at (design 4b-2b §4.1). conflicts columns: kind, symbols, dates, detail.
        bars columns: symbol, session_date, open, high, low, close, volume, available_at.
        overrides: list of mappings, e.g. {"merge": ["CYCN", "KRSA"], "reason": ..., "date": ...}

Fake data provenance (probe 2026-10-06, ``alpaca-corporate-actions-probe.md``, unless noted):
- BK -> BNY 2026-05-21, CUSIP 064058100 unchanged; SEC CIK 1390777 on both names (real).
  Alpaca serves BK's history under BNY too, so BNY carries relabelled duplicates (real
  mechanism, DECISIONS 2026-10-05; the one-day overlap seen in 4a becomes the whole history
  once bars are written raw).
- MMC -> MRSH 2026-01-14 (real); SEC still lists MMC in its last snapshot (real lag).
- GMGI -> MRDN: no Alpaca record (real miss), SEC CIK 1451448 switches ticker between two
  snapshots; 992 days of identical bars (real).
- BBBY: Bed Bath & Beyond CIK 886158 until 2023-05; Beyond Inc CIK 1130713 took the ticker
  2025-08-29 via OSTK -> BYON -> BBBY -> NXH (2023-11-06 / 2025-08-29 / 2026-08-17), CUSIP
  690370101 throughout (real).
- GOOG / GOOGL: Google Inc CIK 1288776 -> Alphabet CIK 1652044 in 2019 with continuous
  prices (SEC filings; prices ASSUMED flat, checked on real data by golden case 5).
- BRK.A / BRK.B: one CIK 1067983, prices ~1,500x apart (real); older SEC spelling ``BRKB``
  vs ``BRK.B`` (real: ~50 such pairs).
- FFR: 33736N101 renamed FFR -> DTRE 2022-10-03; AIXC -> FFR 2026-09-30 (74754R301); Alpaca
  files AIXC's history under FFR as well (real, probe §6 #1).
- HCTI reverse split 2025-08-01 249:1, old_cusip 42227W207 != new_cusip 42227W306 (real).
- CYCN / KRSA: 1,609 days identical OHLCV, no record anywhere (real).
- SPY: has bars, in no SEC company table (real: ~1,131 such symbols).
- DISC: a CIK switch where prices jump 3x: ASSUMPTION (no real example in hand); audit A2
  reports this kind on real data.
"""

import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pyarrow as pa
import pytest

from asof.ingest import security_master as sm
from asof.ingest.security_master import (
    build_security_master,
    edges_from_name_changes,
    edges_from_sec,
    merge_edges,
    normalize_spelling,
    ohlcv_duplicates,
)

ET = ZoneInfo("America/New_York")
MASTER_COLUMNS = [
    "security_id",
    "symbol",
    "valid_from",
    "valid_to",
    "cik",
    "cusip",
    "evidence",
    "available_at",
]
CONFLICT_COLUMNS = ["kind", "symbols", "dates", "detail"]


# =============================================================================================
# builders
# =============================================================================================


def et_midnight(day: date) -> datetime:
    return datetime.combine(day, time(0, 0), tzinfo=ET).astimezone(UTC)


def bar_available_at(day: date) -> datetime:
    return datetime.combine(day, time(20, 0), tzinfo=ET).astimezone(UTC)


def business_days(start: date, end: date) -> list[date]:
    out: list[date] = []
    day = start
    while day <= end:
        if day.weekday() < 5:
            out.append(day)
        day += timedelta(days=1)
    return out


def bar_rows(
    symbol: str, start: date, end: date, base: float, *, volume: int = 10_000
) -> list[dict[str, Any]]:
    """Business-day bars with a slowly drifting close so no two symbols collide by accident."""
    rows = []
    for i, day in enumerate(business_days(start, end)):
        close = round(base * (1 + 0.0002 * i), 6)
        rows.append(
            {
                "symbol": symbol,
                "session_date": day,
                "open": round(close * 0.995, 6),
                "high": round(close * 1.01, 6),
                "low": round(close * 0.99, 6),
                "close": close,
                "volume": volume + i,
                "available_at": bar_available_at(day),
            }
        )
    return rows


def relabel(rows: Iterable[dict[str, Any]], symbol: str) -> list[dict[str, Any]]:
    """What Alpaca does: the same bars, filed under another name."""
    return [{**r, "symbol": symbol} for r in rows]


def bars_table(rows: Sequence[dict[str, Any]]) -> pa.Table:
    return pa.table(
        {
            "symbol": pa.array([r["symbol"] for r in rows], pa.string()),
            "session_date": pa.array([r["session_date"] for r in rows], pa.date32()),
            "open": pa.array([r["open"] for r in rows], pa.float64()),
            "high": pa.array([r["high"] for r in rows], pa.float64()),
            "low": pa.array([r["low"] for r in rows], pa.float64()),
            "close": pa.array([r["close"] for r in rows], pa.float64()),
            "volume": pa.array([r["volume"] for r in rows], pa.int64()),
            "available_at": pa.array(
                [r["available_at"] for r in rows], pa.timestamp("us", tz="UTC")
            ),
        }
    )


Life = tuple[str, int, date, date | None]  # ticker, cik, first snapshot, last snapshot (incl.)


def sec_table(snapshots: Sequence[date], lives: Sequence[Life]) -> pa.Table:
    rows = []
    for snap in snapshots:
        for ticker, cik, first, last in lives:
            if first <= snap and (last is None or snap <= last):
                rows.append((snap, ticker, cik, f"{ticker} CO"))
    rows.sort()
    return pa.table(
        {
            "snapshot_date": pa.array([r[0] for r in rows], pa.date32()),
            "ticker": pa.array([r[1] for r in rows], pa.string()),
            "cik": pa.array([r[2] for r in rows], pa.int64()),
            "name": pa.array([r[3] for r in rows], pa.string()),
        }
    )


Rename = tuple[str, str, date, str, str]  # old, new, process_date, old_cusip, new_cusip


def name_changes_table(rows: Sequence[Rename]) -> pa.Table:
    return pa.table(
        {
            "old_symbol": pa.array([r[0] for r in rows], pa.string()),
            "new_symbol": pa.array([r[1] for r in rows], pa.string()),
            "process_date": pa.array([r[2] for r in rows], pa.date32()),
            "old_cusip": pa.array([r[3] for r in rows], pa.string()),
            "new_cusip": pa.array([r[4] for r in rows], pa.string()),
            "available_at": pa.array(
                [et_midnight(r[2]) for r in rows], pa.timestamp("us", tz="UTC")
            ),
        }
    )


# =============================================================================================
# the world: every case in one consistent set of inputs
# =============================================================================================

SNAPSHOTS = [
    date(2019, 5, 1),
    date(2019, 7, 1),
    date(2020, 1, 13),
    date(2022, 9, 6),
    date(2022, 11, 1),
    date(2023, 4, 3),
    date(2023, 12, 1),
    date(2024, 5, 1),
    date(2024, 6, 3),
    date(2025, 9, 2),
    date(2026, 5, 4),
    date(2026, 6, 4),
    date(2026, 9, 8),
]
FIRST = SNAPSHOTS[0]

LIVES: list[Life] = [
    ("BK", 1390777, FIRST, date(2026, 5, 4)),
    ("BNY", 1390777, date(2026, 6, 4), None),
    ("MMC", 571748, FIRST, None),  # SEC lag: MRSH never appears
    ("GMGI", 1451448, FIRST, date(2024, 5, 1)),
    ("MRDN", 1451448, date(2024, 6, 3), None),
    ("BBBY", 886158, FIRST, date(2023, 4, 3)),
    ("OSTK", 1130713, FIRST, date(2023, 4, 3)),
    ("BYON", 1130713, date(2023, 12, 1), date(2024, 6, 3)),
    ("BBBY", 1130713, date(2025, 9, 2), date(2026, 6, 4)),
    ("NXH", 1130713, date(2026, 9, 8), None),
    ("GOOG", 1288776, FIRST, date(2019, 5, 1)),
    ("GOOG", 1652044, date(2019, 7, 1), None),
    ("GOOGL", 1288776, FIRST, date(2019, 5, 1)),
    ("GOOGL", 1652044, date(2019, 7, 1), None),
    ("BRK.A", 1067983, FIRST, None),
    ("BRKB", 1067983, FIRST, date(2020, 1, 13)),  # old SEC spelling
    ("BRK.B", 1067983, date(2022, 9, 6), None),
    ("FFR", 1000001, FIRST, date(2022, 9, 6)),
    ("DTRE", 1000001, date(2022, 11, 1), date(2023, 4, 3)),
    ("AIXC", 1000002, date(2024, 5, 1), None),  # SEC lag: the new FFR never appears
    ("HCTI", 1000003, FIRST, None),
    ("CYCN", 1000004, FIRST, None),
    ("KRSA", 1000005, FIRST, None),
    ("DISC", 1000006, FIRST, date(2024, 5, 1)),
    ("DISC", 1000007, date(2024, 6, 3), None),
]

RENAMES: list[Rename] = [
    ("BK", "BNY", date(2026, 5, 21), "064058100", "064058100"),
    ("MMC", "MRSH", date(2026, 1, 14), "571748102", "571748102"),
    ("OSTK", "BYON", date(2023, 11, 6), "690370101", "690370101"),
    ("BYON", "BBBY", date(2025, 8, 29), "690370101", "690370101"),
    ("BBBY", "NXH", date(2026, 8, 17), "690370101", "690370101"),
    ("FFR", "DTRE", date(2022, 10, 3), "33736N101", "33736N101"),
    ("AIXC", "FFR", date(2026, 9, 30), "74754R301", "74754R301"),
]


def world_bars() -> list[dict[str, Any]]:
    bk = bar_rows("BK", date(2020, 1, 2), date(2026, 5, 20), 50.0)
    mmc = bar_rows("MMC", date(2020, 1, 2), date(2026, 1, 13), 200.0)
    gmgi = bar_rows("GMGI", date(2020, 1, 2), date(2024, 5, 20), 3.0)
    aixc = bar_rows("AIXC", date(2024, 1, 2), date(2026, 9, 29), 2.0)
    cycn = bar_rows("CYCN", date(2024, 1, 2), date(2024, 2, 13), 7.0, volume=5_000)
    return [
        *bk,
        *relabel(bk, "BNY"),
        *bar_rows("BNY", date(2026, 5, 21), date(2026, 10, 5), 51.0),
        *mmc,
        *relabel(mmc, "MRSH"),
        *bar_rows("MRSH", date(2026, 1, 14), date(2026, 10, 5), 201.0),
        *gmgi,
        *relabel(gmgi, "MRDN"),
        *bar_rows("MRDN", date(2024, 5, 21), date(2026, 10, 5), 3.1),
        *bar_rows("BBBY", date(2020, 1, 2), date(2023, 5, 3), 5.0),
        *bar_rows("OSTK", date(2020, 1, 2), date(2023, 11, 3), 20.0),
        *bar_rows("BYON", date(2023, 11, 6), date(2025, 8, 28), 21.0),
        *bar_rows("BBBY", date(2025, 8, 29), date(2026, 8, 14), 22.0),
        *bar_rows("NXH", date(2026, 8, 17), date(2026, 10, 5), 23.0),
        *bar_rows("GOOG", date(2019, 1, 2), date(2019, 12, 31), 1100.0),
        *bar_rows("GOOGL", date(2019, 1, 2), date(2019, 12, 31), 1105.0),
        *bar_rows("BRK.A", date(2020, 1, 2), date(2020, 6, 30), 300_000.0),
        *bar_rows("BRK.B", date(2020, 1, 2), date(2020, 6, 30), 200.0),
        *bar_rows("FFR", date(2020, 1, 2), date(2022, 9, 30), 5.5),
        *bar_rows("DTRE", date(2022, 10, 3), date(2023, 6, 30), 6.0),
        *aixc,
        *relabel(aixc, "FFR"),
        *bar_rows("FFR", date(2026, 9, 30), date(2026, 10, 5), 2.1),
        *bar_rows("HCTI", date(2025, 7, 1), date(2025, 7, 31), 0.02),
        *bar_rows("HCTI", date(2025, 8, 1), date(2025, 8, 29), 3.0),  # 249:1 reverse split
        *cycn,
        *relabel(cycn, "KRSA"),
        *bar_rows("SPY", date(2020, 1, 2), date(2020, 3, 31), 320.0),
        *bar_rows("DISC", date(2024, 1, 2), date(2024, 5, 31), 10.0),
        *bar_rows("DISC", date(2024, 6, 3), date(2024, 12, 31), 30.0),  # 3x across CIK switch
    ]


@pytest.fixture(scope="module")
def bars() -> pa.Table:
    return bars_table(world_bars())


@pytest.fixture(scope="module")
def sec_history() -> pa.Table:
    return sec_table(SNAPSHOTS, LIVES)


@pytest.fixture(scope="module")
def name_changes() -> pa.Table:
    return name_changes_table(RENAMES)


@pytest.fixture(scope="module")
def built(
    bars: pa.Table, sec_history: pa.Table, name_changes: pa.Table
) -> tuple[pa.Table, pa.Table]:
    return build_security_master(bars=bars, sec_history=sec_history, name_changes=name_changes)


@pytest.fixture(scope="module")
def master(built: tuple[pa.Table, pa.Table]) -> pa.Table:
    return built[0]


@pytest.fixture(scope="module")
def conflicts(built: tuple[pa.Table, pa.Table]) -> pa.Table:
    return built[1]


# --- query helpers ------------------------------------------------------------------------------


def segments(master: pa.Table, symbol: str) -> list[dict[str, Any]]:
    rows = [r for r in master.to_pylist() if r["symbol"] == symbol]
    return sorted(rows, key=lambda r: r["valid_from"])


def segment_at(master: pa.Table, symbol: str, day: date) -> dict[str, Any]:
    hits = [
        r
        for r in segments(master, symbol)
        if r["valid_from"] <= day and (r["valid_to"] is None or day < r["valid_to"])
    ]
    assert len(hits) == 1, f"{symbol} @ {day}: {hits}"
    return hits[0]


def sids(master: pa.Table, symbol: str) -> set[str]:
    return {r["security_id"] for r in segments(master, symbol)}


def only_sid(master: pa.Table, symbol: str) -> str:
    [sid] = sids(master, symbol)
    return sid


def conflicts_of(conflicts: pa.Table, kind: str) -> list[dict[str, Any]]:
    return [r for r in conflicts.to_pylist() if r["kind"] == kind]


# =============================================================================================
# shape
# =============================================================================================


def test_master_and_conflicts_have_the_documented_columns(
    master: pa.Table, conflicts: pa.Table
) -> None:
    assert master.schema.names == MASTER_COLUMNS
    assert master.schema.field("valid_from").type == pa.date32()
    assert master.schema.field("valid_to").type == pa.date32()
    assert master.schema.field("cik").type == pa.int64()
    assert master.schema.field("available_at").type == pa.timestamp("us", tz="UTC")
    assert conflicts.schema.names == CONFLICT_COLUMNS
    assert pa.types.is_list(conflicts.schema.field("symbols").type)
    assert all(re.fullmatch(r"S\d{6}", s) for s in master.column("security_id").to_pylist())


def test_segments_of_one_symbol_never_overlap(master: pa.Table) -> None:
    for symbol in set(master.column("symbol").to_pylist()):
        segs = segments(master, symbol)
        for earlier, later in zip(segs, segs[1:], strict=False):
            assert earlier["valid_to"] is not None, (symbol, earlier)
            assert earlier["valid_to"] <= later["valid_from"], (symbol, earlier, later)


# =============================================================================================
# normal path: renames
# =============================================================================================


def test_pure_rename_with_both_sources_is_one_security_in_two_segments(master: pa.Table) -> None:
    [bk] = segments(master, "BK")
    [bny] = segments(master, "BNY")

    assert bk["security_id"] == bny["security_id"]
    assert (bk["valid_from"], bk["valid_to"]) == (date(2020, 1, 2), date(2026, 5, 21))
    assert (bny["valid_from"], bny["valid_to"]) == (date(2026, 5, 21), None)
    assert bk["cik"] == bny["cik"] == 1390777
    assert bk["cusip"] == bny["cusip"] == "064058100"
    assert bny["evidence"] == "both"
    assert bk["evidence"] == "both"


def test_rename_known_only_to_alpaca_uses_the_exact_date(master: pa.Table) -> None:
    [mmc] = segments(master, "MMC")
    [mrsh] = segments(master, "MRSH")

    assert mmc["security_id"] == mrsh["security_id"]
    assert mmc["valid_to"] == mrsh["valid_from"] == date(2026, 1, 14)
    assert mrsh["valid_to"] is None
    assert mrsh["evidence"] == "name_change"
    assert mrsh["cusip"] == "571748102"


def test_rename_known_only_to_sec_cuts_the_day_after_the_old_names_last_bar(
    master: pa.Table,
) -> None:
    # GMGI was in the 2024-05-01 snapshot, MRDN (same CIK) in the 2024-06-03 one, nothing in
    # Alpaca: the switch happened somewhere in (2024-05-01, 2024-06-03]. Alpaca stops filing
    # bars under the old name on the rename day (real: MMC's bars end 2026-01-13, MRSH's
    # rename is 2026-01-14), so GMGI's last bar, 2024-05-20, pins the cut at 2024-05-21 --
    # weeks before the snapshot bound, which would have hidden MRDN's first weeks.
    [gmgi] = segments(master, "GMGI")
    [mrdn] = segments(master, "MRDN")

    assert gmgi["security_id"] == mrdn["security_id"]
    assert gmgi["valid_to"] == mrdn["valid_from"] == date(2024, 5, 21)
    assert mrdn["evidence"] == "sec"
    assert mrdn["cik"] == 1451448


def test_sec_only_edge_records_the_snapshot_interval(sec_history: pa.Table) -> None:
    edges = edges_from_sec(normalize_spelling(sec_history)).to_pylist()

    [edge] = [e for e in edges if (e["old_symbol"], e["new_symbol"]) == ("GMGI", "MRDN")]
    assert edge["date"] == date(2024, 6, 3)
    assert edge["date_lower"] == date(2024, 5, 1)
    assert edge["evidence"] == "sec"


def test_relabelled_history_under_the_new_name_gets_no_segment(master: pa.Table) -> None:
    # Alpaca filed BK's 2020-2026 bars under BNY as well. The BNY segment starts 2026-05-21;
    # a BNY bar from 2022 falls into no segment and so is invisible to visible_bars(resolve).
    assert all(seg["valid_from"] >= date(2026, 5, 21) for seg in segments(master, "BNY"))
    assert all(seg["valid_from"] >= date(2026, 1, 14) for seg in segments(master, "MRSH"))


# =============================================================================================
# edges helpers
# =============================================================================================


def test_edges_from_name_changes_carry_date_cusips_and_availability(
    name_changes: pa.Table,
) -> None:
    edges = edges_from_name_changes(name_changes).to_pylist()

    assert {(e["old_symbol"], e["new_symbol"]) for e in edges} == {
        (old, new) for old, new, *_ in RENAMES
    }
    bk = next(e for e in edges if e["old_symbol"] == "BK")
    assert bk["date"] == date(2026, 5, 21)
    assert bk["date_lower"] in (None, date(2026, 5, 21))
    assert bk["evidence"] == "name_change"
    assert bk["available_at"] == datetime(2026, 5, 21, 4, 0, tzinfo=UTC)


def test_cusip_only_change_is_not_an_edge() -> None:
    table = name_changes_table([("AMAM", "AMAM", date(2023, 10, 12), "02290A102", "641871108")])

    assert edges_from_name_changes(table).num_rows == 0


def test_edges_from_sec_ignore_delistings_and_cik_switches(sec_history: pa.Table) -> None:
    edges = edges_from_sec(normalize_spelling(sec_history)).to_pylist()
    pairs = {(e["old_symbol"], e["new_symbol"]) for e in edges}

    assert ("BBBY", "BBBY") not in pairs  # CIK switch, not a rename
    assert ("GOOG", "GOOG") not in pairs
    assert not any(old == "BBBY" and new != "NXH" for old, new in pairs)  # 886158 just vanished
    assert ("BYON", "BBBY") in pairs
    assert ("FFR", "DTRE") in pairs


def test_merge_edges_prefers_alpaca_dates_and_marks_both(
    sec_history: pa.Table, name_changes: pa.Table
) -> None:
    merged = merge_edges(
        edges_from_name_changes(name_changes), edges_from_sec(normalize_spelling(sec_history))
    ).to_pylist()
    by_pair = {(e["old_symbol"], e["new_symbol"]): e for e in merged}

    assert len(merged) == len(by_pair), "one row per (old, new)"
    assert by_pair[("BK", "BNY")]["date"] == date(2026, 5, 21)
    assert by_pair[("BK", "BNY")]["evidence"] == "both"
    assert by_pair[("MMC", "MRSH")]["evidence"] == "name_change"
    assert by_pair[("GMGI", "MRDN")]["evidence"] == "sec"
    assert by_pair[("GMGI", "MRDN")]["date"] == date(2024, 6, 3)
    assert by_pair[("AIXC", "FFR")]["evidence"] == "name_change"


# =============================================================================================
# boundaries: reuse, CIK switches, share classes, spelling, splits
# =============================================================================================


def test_reused_ticker_bbby_is_two_securities(master: pa.Table) -> None:
    assert len(sids(master, "BBBY")) == 2

    old = segment_at(master, "BBBY", date(2021, 6, 1))
    new = segment_at(master, "BBBY", date(2026, 1, 5))
    assert old["cik"] == 886158
    assert new["cik"] == 1130713
    assert old["security_id"] != new["security_id"]
    assert old["valid_to"] is not None and old["valid_to"] <= date(2025, 8, 29)
    assert new["security_id"] == only_sid(master, "OSTK") == only_sid(master, "BYON")
    assert new["security_id"] == only_sid(master, "NXH")
    assert min(s["valid_from"] for s in segments(master, "BBBY") if s["cik"] == 1130713) == (
        date(2025, 8, 29)
    )


def test_cik_switch_with_continuous_prices_stays_one_security(master: pa.Table) -> None:
    # Google Inc -> Alphabet Inc: the SEC CIK changes between the 2019-05-01 and 2019-07-01
    # snapshots while the close barely moves, so the two segments are the same security.
    goog = segments(master, "GOOG")

    assert {s["cik"] for s in goog} == {1288776, 1652044}
    assert len({s["security_id"] for s in goog}) == 1
    assert goog[0]["valid_to"] == goog[1]["valid_from"] == date(2019, 7, 1)


def test_cik_switch_with_a_price_gap_or_jump_is_not_bridged(
    master: pa.Table, conflicts: pa.Table
) -> None:
    # BBBY: two years without a bar between the two CIKs. DISC: bars on both sides but the
    # close triples. Neither passes the 0.8-1.25 continuity test; DISC is reported.
    assert len(sids(master, "BBBY")) == 2
    assert len(sids(master, "DISC")) == 2
    assert any("DISC" in r["symbols"] for r in conflicts.to_pylist())


def test_share_classes_with_one_cik_are_separate_securities(master: pa.Table) -> None:
    assert only_sid(master, "GOOG") != only_sid(master, "GOOGL")
    assert only_sid(master, "BRK.A") != only_sid(master, "BRK.B")
    assert segment_at(master, "BRK.A", date(2020, 3, 2))["cik"] == 1067983
    assert segment_at(master, "BRK.B", date(2020, 3, 2))["cik"] == 1067983


def test_sec_spelling_variant_is_not_a_rename(master: pa.Table) -> None:
    # SEC wrote BRKB until 2020, BRK.B afterwards, same CIK: one symbol, one segment.
    [seg] = segments(master, "BRK.B")

    assert "BRKB" not in set(master.column("symbol").to_pylist())
    assert (seg["valid_from"], seg["valid_to"]) == (date(2020, 1, 2), None)
    assert seg["cik"] == 1067983


def test_normalize_spelling_merges_dotless_twins_with_the_same_cik_only() -> None:
    table = sec_table(
        [date(2020, 1, 13), date(2022, 9, 6)],
        [
            ("BRKB", 1067983, date(2020, 1, 13), date(2020, 1, 13)),
            ("BRK.B", 1067983, date(2022, 9, 6), None),
            ("AIGWS", 5272, date(2020, 1, 13), date(2020, 1, 13)),
            ("AIG.WS", 5272, date(2022, 9, 6), None),
            ("ABC", 1, date(2020, 1, 13), None),
            ("AB.C", 2, date(2020, 1, 13), None),  # same letters, other CIK: untouched
        ],
    )

    out = normalize_spelling(table)

    got = sorted(zip(out.column("ticker").to_pylist(), out.column("cik").to_pylist(), strict=True))
    assert got == sorted(
        [
            ("BRK.B", 1067983),
            ("BRK.B", 1067983),
            ("AIG.WS", 5272),
            ("AIG.WS", 5272),
            ("ABC", 1),
            ("ABC", 1),
            ("AB.C", 2),
            ("AB.C", 2),
        ]
    )
    assert out.num_rows == table.num_rows


def test_vacated_and_reused_ticker_ffr_is_two_securities(master: pa.Table) -> None:
    first, second = segments(master, "FFR")

    assert (first["valid_from"], first["valid_to"]) == (date(2020, 1, 2), date(2022, 10, 3))
    assert (second["valid_from"], second["valid_to"]) == (date(2026, 9, 30), None)
    assert first["security_id"] == only_sid(master, "DTRE")
    assert second["security_id"] == only_sid(master, "AIXC")
    assert first["security_id"] != second["security_id"]
    assert first["cusip"] == "33736N101"
    assert second["cusip"] == "74754R301"
    # AIXC's history relabelled FFR (2024-01..2026-09-29) lies in neither segment.
    assert len(segments(master, "FFR")) == 2


def test_reverse_split_that_changes_cusip_does_not_cut_the_segment(master: pa.Table) -> None:
    [seg] = segments(master, "HCTI")

    assert (seg["valid_from"], seg["valid_to"]) == (date(2025, 7, 1), None)
    assert seg["cik"] == 1000003


def test_symbol_known_only_from_bars_gets_a_single_bars_only_segment(master: pa.Table) -> None:
    [spy] = segments(master, "SPY")

    assert (spy["valid_from"], spy["valid_to"]) == (date(2020, 1, 2), None)
    assert spy["cik"] is None
    assert spy["cusip"] is None
    assert spy["evidence"] == "bars_only"
    assert spy["available_at"] == bar_available_at(date(2020, 1, 2))


# =============================================================================================
# OHLCV duplicates: report, never merge; overrides merge
# =============================================================================================


def test_ohlcv_duplicates_finds_identical_series_above_the_volume_floor(bars: pa.Table) -> None:
    pairs = {
        frozenset((r["symbol_a"], r["symbol_b"])): r["days"]
        for r in ohlcv_duplicates(bars).to_pylist()
    }

    assert pairs[frozenset(("CYCN", "KRSA"))] == len(
        business_days(date(2024, 1, 2), date(2024, 2, 13))
    )
    assert frozenset(("BK", "BNY")) in pairs  # relabelled history is a duplicate too
    assert frozenset(("GOOG", "GOOGL")) not in pairs


def test_ohlcv_duplicates_ignore_volume_at_or_below_1000() -> None:
    # bar_rows adds the day index to the volume: 979 + 0..21 keeps every day at or below 1000.
    thin = bar_rows("AAAA", date(2024, 1, 2), date(2024, 1, 31), 1.0, volume=979)
    table = bars_table(thin + relabel(thin, "BBBB"))

    assert ohlcv_duplicates(table).num_rows == 0


def test_unlinked_duplicates_are_reported_not_merged(master: pa.Table, conflicts: pa.Table) -> None:
    assert only_sid(master, "CYCN") != only_sid(master, "KRSA")
    [row] = [
        r for r in conflicts_of(conflicts, "dup_unlinked") if set(r["symbols"]) == {"CYCN", "KRSA"}
    ]
    assert row["detail"]


def test_duplicates_inside_one_chain_are_not_conflicts(conflicts: pa.Table) -> None:
    flagged = [set(r["symbols"]) for r in conflicts_of(conflicts, "dup_unlinked")]

    assert {"BK", "BNY"} not in flagged
    assert {"MMC", "MRSH"} not in flagged
    assert {"GMGI", "MRDN"} not in flagged
    assert {"AIXC", "FFR"} not in flagged


def test_override_merges_two_chains(
    bars: pa.Table, sec_history: pa.Table, name_changes: pa.Table
) -> None:
    overrides: list[Mapping[str, Any]] = [
        {
            "merge": ["CYCN", "KRSA"],
            "reason": "1,609 days of identical OHLCV, no record in Alpaca or SEC",
            "date": "2026-10-07",
        }
    ]

    master, conflicts = build_security_master(
        bars=bars, sec_history=sec_history, name_changes=name_changes, overrides=overrides
    )

    assert only_sid(master, "CYCN") == only_sid(master, "KRSA")
    assert "override" in {
        s["evidence"] for s in segments(master, "CYCN") + segments(master, "KRSA")
    }
    assert not any(
        set(r["symbols"]) == {"CYCN", "KRSA"} for r in conflicts_of(conflicts, "dup_unlinked")
    )
    assert only_sid(master, "BK") != only_sid(master, "CYCN")  # nothing else moved


# =============================================================================================
# stability of security_id
# =============================================================================================


def test_same_inputs_give_the_same_master(
    bars: pa.Table, sec_history: pa.Table, name_changes: pa.Table, master: pa.Table
) -> None:
    again, _ = build_security_master(bars=bars, sec_history=sec_history, name_changes=name_changes)

    assert again.equals(master)


def test_adding_a_new_symbol_keeps_existing_ids(
    bars: pa.Table, sec_history: pa.Table, name_changes: pa.Table, master: pa.Table
) -> None:
    extra = bars_table(world_bars() + bar_rows("ZZZZ", date(2026, 9, 1), date(2026, 10, 5), 9.0))

    grown, _ = build_security_master(bars=extra, sec_history=sec_history, name_changes=name_changes)

    def keyed(table: pa.Table) -> set[tuple[str, str, date, date | None]]:
        return {
            (r["security_id"], r["symbol"], r["valid_from"], r["valid_to"])
            for r in table.to_pylist()
        }

    assert keyed(master) <= keyed(grown)
    assert len(keyed(grown) - keyed(master)) == 1
    assert only_sid(grown, "ZZZZ") not in set(master.column("security_id").to_pylist())


# =============================================================================================
# leakage: a segment is known no earlier than the event that created it
# =============================================================================================


def test_segment_created_by_a_rename_is_known_at_midnight_new_york_of_that_day(
    master: pa.Table,
) -> None:
    [bny] = segments(master, "BNY")
    [bk] = segments(master, "BK")

    assert bny["available_at"] == datetime(2026, 5, 21, 4, 0, tzinfo=UTC)  # 00:00 EDT
    assert bny["available_at"] == et_midnight(date(2026, 5, 21))
    assert bk["available_at"] == bar_available_at(date(2020, 1, 2))  # first bar, 20:00 ET
    assert bk["available_at"] == datetime(2020, 1, 3, 1, 0, tzinfo=UTC)


def test_segment_created_by_an_sec_snapshot_is_not_known_before_the_snapshot(
    master: pa.Table,
) -> None:
    [mrdn] = segments(master, "MRDN")

    assert mrdn["available_at"] >= et_midnight(date(2024, 6, 3))


def test_no_segment_is_known_before_its_own_valid_from(master: pa.Table) -> None:
    for row in master.to_pylist():
        assert row["available_at"] >= et_midnight(row["valid_from"]), row


def test_bk_segment_end_is_not_knowable_before_the_rename(master: pa.Table) -> None:
    # A reader at as_of 2022-03-01 filters segments by available_at: the BNY segment (known
    # 2026-05-21) is absent, and the BK row it sees must be treated as open-ended. The master
    # gives the reader what it needs: BK's valid_to equals BNY's valid_from, so the reader can
    # reopen BK whenever BNY is invisible.
    as_of = datetime(2022, 3, 1, tzinfo=UTC)
    visible = [r for r in master.to_pylist() if r["available_at"] <= as_of]
    names = {r["symbol"] for r in visible}

    assert "BK" in names
    assert "BNY" not in names
    [bk_now] = segments(master, "BK")
    [bny_now] = segments(master, "BNY")
    assert bk_now["valid_to"] == bny_now["valid_from"]


# =============================================================================================
# rules measured on the real store (2026-10-07): stale SEC table (R1), mass table correction
# (R2), vacate before the first bar (R3), relabelled pre-takeover piece (R4), same-CIK
# takeover (R5)
# =============================================================================================
#
# API these tests expect (implementation pending):
#   build_security_master(..., sec_history_start: date | None = None)
#       SEC snapshots dated before ``sec_history_start`` are ignored entirely (no edges, CIK
#       switches, presence or segment CIKs come from them); None = use everything.
#   SEC_HISTORY_START = date(2019, 10, 2)        module constant, passed by build_from_store
#   TABLE_CORRECTION_MIN_SWITCHES = 50          module constant
#
# Every world below is self-contained (own bars / SEC snapshots / name changes) so each rule
# can be read off its inputs. Bars start 2020-01-02 as in the real store.


def assert_nothing_known_before_its_valid_from(master: pa.Table) -> None:
    """Leakage guard reused on every new world: merging or dropping a piece must not let any
    segment become knowable before the day it starts."""
    for row in master.to_pylist():
        assert row["available_at"] >= et_midnight(row["valid_from"]), row


def visible_at(master: pa.Table, as_of: datetime) -> list[dict[str, Any]]:
    return [r for r in master.to_pylist() if r["available_at"] <= as_of]


# --- R1: the SEC table before 2019-10-02 is frozen and stale --------------------------------
# Real (2026-10-07): the company_tickers.json snapshots before 2019-10-02 are one frozen file:
# 6,259 rows, 5,530 (ticker, cik) pairs survive the 2019-10-02 refresh, 597 tickers vanish,
# 132 switch CIK in that single pair, and none of the 132 has a bar before 2019-10-02 (bars
# start 2020-01-02). MRNA sat there as "Marina Biotech" CIK 737207 (Moderna is 1682852), BUD
# under the old entity CIK 1140467 (now 1668717); the first real build killed BUD with a fake
# vacate edge dated 2019-10-02 (no segment at all).

R1_CUTOFF = date(2019, 10, 2)
R1_SNAPSHOTS = [date(2019, 5, 1), date(2019, 7, 1), R1_CUTOFF, date(2020, 6, 1), date(2021, 1, 4)]
R1_LIVES: list[Life] = [
    # stale file
    ("MRNX", 737207, date(2019, 5, 1), date(2019, 5, 1)),  # constructed: an earlier label on
    #   the stale CIK, so the stale pair yields a fake SEC edge MRNX -> MRNA (a "start" for MRNA)
    ("MRNA", 737207, date(2019, 7, 1), date(2019, 7, 1)),  # real: Marina Biotech row
    ("BUD", 1140467, date(2019, 5, 1), date(2019, 7, 1)),  # real: old entity
    # refreshed table
    ("MRNA", 1682852, R1_CUTOFF, None),  # real: Moderna
    ("BUD", 1668717, R1_CUTOFF, None),  # real
    ("ABIX", 1140467, R1_CUTOFF, None),  # constructed: the old BUD CIK carries another label
    #   after the refresh, so the (last stale, first good) pair yields a fake vacate BUD -> ABIX
    #   at 2019-10-02 -- the shape that killed BUD in the first real build
]


def r1_inputs() -> tuple[pa.Table, pa.Table, pa.Table]:
    bars = bars_table(
        bar_rows("MRNA", date(2020, 1, 2), date(2020, 12, 31), 20.0)
        + bar_rows("BUD", date(2020, 1, 2), date(2020, 12, 31), 60.0)
    )
    return bars, sec_table(R1_SNAPSHOTS, R1_LIVES), name_changes_table([])


def test_r1_sec_history_start_is_the_first_refreshed_snapshot() -> None:
    assert sm.SEC_HISTORY_START == date(2019, 10, 2)


def test_r1_snapshots_before_sec_history_start_are_ignored_entirely() -> None:
    bars, sec, changes = r1_inputs()

    master, conflicts = build_security_master(
        bars=bars, sec_history=sec, name_changes=changes, sec_history_start=R1_CUTOFF
    )

    # MRNA: one segment from its first bar, the good CIK, no CIK switch, no fake start edge.
    assert len(segments(master, "MRNA")) == 1
    [mrna] = segments(master, "MRNA")
    assert (mrna["valid_from"], mrna["valid_to"]) == (date(2020, 1, 2), None)
    assert mrna["cik"] == 1682852
    assert mrna["available_at"] == bar_available_at(date(2020, 1, 2))  # not the edge's midnight
    # BUD: the fake vacate came from the stale side of the (2019-07-01, 2019-10-02] pair.
    assert len(segments(master, "BUD")) == 1
    [bud] = segments(master, "BUD")
    assert (bud["valid_from"], bud["valid_to"]) == (date(2020, 1, 2), None)
    assert bud["cik"] == 1668717
    # Nothing from the stale file survives anywhere.
    symbols = set(master.column("symbol").to_pylist())
    assert "MRNX" not in symbols and "ABIX" not in symbols
    assert 737207 not in set(master.column("cik").to_pylist())
    assert 1140467 not in set(master.column("cik").to_pylist())
    assert conflicts_of(conflicts, "cik_switch_discontinuous") == []
    assert_nothing_known_before_its_valid_from(master)


def test_r1_filtering_the_stale_snapshots_removes_the_fake_edges() -> None:
    # edges_from_sec has no cutoff parameter; the cutoff is applied to the table it is given.
    _, sec, _ = r1_inputs()

    kept = sec.filter(pa.array([d >= R1_CUTOFF for d in sec.column("snapshot_date").to_pylist()]))
    all_edges = edges_from_sec(normalize_spelling(sec)).to_pylist()
    kept_edges = edges_from_sec(normalize_spelling(kept)).to_pylist()

    assert {(e["old_symbol"], e["new_symbol"]) for e in all_edges} == {
        ("MRNX", "MRNA"),
        ("BUD", "ABIX"),
    }
    assert kept_edges == []


def test_r1_without_a_cutoff_the_stale_table_leaves_a_trace() -> None:
    # Default None keeps every snapshot (the shared world above starts 2019-05-01 and must keep
    # working). Then the stale rows are believed: at least one of MRNA / BUD shows a CIK switch
    # at the refresh (a segment boundary at 2019-10-02) or the stale CIK itself. Deliberately
    # loose: it only pins that the parameter does work, not how the stale rows are read.
    bars, sec, changes = r1_inputs()

    master, _ = build_security_master(bars=bars, sec_history=sec, name_changes=changes)

    rows = segments(master, "MRNA") + segments(master, "BUD")
    trace = {r["cik"] for r in rows} | {r["valid_from"] for r in rows}
    assert {737207, 1140467, R1_CUTOFF} & trace, rows


# --- R2: mass table correction between two consecutive snapshots ----------------------------
# Real (2026-10-07): between the 2020-06-01 and 2020-07-10 snapshots 142 tickers switched CIK
# (a normal consecutive pair: at most 14). 126 of the old CIKs never appear again (zombie rows:
# AFC listed as "Allied Capital Corp" CIK 3906, acquired 2010, actual holder Ares Capital CIK
# 1287750; APPB "Applebee's", delisted 2007). 16 old CIKs live on under another ticker (ARNC's
# old CIK 4281 "Arconic Inc." continues as HWM; the new ARNC is Arconic Corp CIK 1790982).
# Rule: >= TABLE_CORRECTION_MIN_SWITCHES switches in one pair -> a table correction; a switching
# ticker whose old CIK is in no snapshot at/after the correction loses its rows before it.

R2_SNAPSHOTS = [date(2020, 1, 13), date(2020, 6, 1), date(2020, 7, 10), date(2021, 1, 4)]
R2_CORRECTION = date(2020, 7, 10)
R2_N_MASS = 60  # >= 50 together with AFC and ARNC


def r2_lives(*, mass: bool) -> list[Life]:
    lives: list[Life] = [
        ("AFC", 3906, date(2020, 1, 13), date(2020, 6, 1)),  # real: zombie row
        ("AFC", 1287750, R2_CORRECTION, None),  # real: Ares Capital
        ("ARNC", 4281, date(2020, 1, 13), date(2020, 6, 1)),  # real: Arconic Inc.
        ("ARNC", 1790982, R2_CORRECTION, None),  # real: Arconic Corp
        # ASSUMPTION: HWM already sat under 4281 next to the stale ARNC row before the
        # correction (Howmet renamed 2020-04-01). The rule only needs 4281 to appear at or
        # after the correction; checked on real data by the ARNC golden case.
        ("HWM", 4281, date(2020, 6, 1), None),
    ]
    if mass:
        for i in range(R2_N_MASS):
            ticker = f"T{i:03d}"  # constructed zombies, like the other 124 real ones
            lives.append((ticker, 900_000 + i, date(2020, 1, 13), date(2020, 6, 1)))
            lives.append((ticker, 800_000 + i, R2_CORRECTION, None))
    return lives


def r2_bars(*, mass: bool) -> list[dict[str, Any]]:
    rows = [
        *bar_rows("AFC", date(2020, 1, 2), date(2020, 12, 31), 15.0),  # continuous through
        # ARNC: Arconic Corp is a different company (spin-off); its price level is made 3x so
        # the switch is reported rather than bridged. ASSUMPTION on the size of the jump.
        *bar_rows("ARNC", date(2020, 1, 2), date(2020, 7, 9), 25.0),
        *bar_rows("ARNC", R2_CORRECTION, date(2020, 12, 31), 75.0),
        *bar_rows("HWM", date(2020, 4, 1), date(2020, 12, 31), 18.0),
    ]
    if mass:
        for i in range(R2_N_MASS):  # a few weeks each, distinct base prices
            rows += bar_rows(f"T{i:03d}", date(2020, 1, 2), date(2020, 1, 31), 100.0 + i)
    return rows


@pytest.fixture(scope="module")
def r2_mass() -> tuple[pa.Table, pa.Table]:
    return build_security_master(
        bars=bars_table(r2_bars(mass=True)),
        sec_history=sec_table(R2_SNAPSHOTS, r2_lives(mass=True)),
        name_changes=name_changes_table([]),
    )


@pytest.fixture(scope="module")
def r2_few() -> tuple[pa.Table, pa.Table]:
    return build_security_master(
        bars=bars_table(r2_bars(mass=False)),
        sec_history=sec_table(R2_SNAPSHOTS, r2_lives(mass=False)),
        name_changes=name_changes_table([]),
    )


def test_r2_table_correction_threshold_constant() -> None:
    assert sm.TABLE_CORRECTION_MIN_SWITCHES == 50


def test_r2_mass_correction_drops_zombie_rows(r2_mass: tuple[pa.Table, pa.Table]) -> None:
    master, conflicts = r2_mass

    assert len(segments(master, "AFC")) == 1
    [afc] = segments(master, "AFC")
    assert (afc["valid_from"], afc["valid_to"]) == (date(2020, 1, 2), None)
    assert afc["cik"] == 1287750
    assert not any("AFC" in r["symbols"] for r in conflicts.to_pylist())
    for i in range(R2_N_MASS):
        ticker = f"T{i:03d}"
        assert len(segments(master, ticker)) == 1, ticker
        [seg] = segments(master, ticker)
        assert (seg["valid_from"], seg["valid_to"], seg["cik"]) == (
            date(2020, 1, 2),
            None,
            800_000 + i,
        ), ticker


def test_r2_mass_correction_keeps_switches_whose_old_cik_lives_on(
    r2_mass: tuple[pa.Table, pa.Table],
) -> None:
    master, conflicts = r2_mass

    old, new = segments(master, "ARNC")
    assert (old["valid_from"], old["valid_to"], old["cik"]) == (
        date(2020, 1, 2),
        R2_CORRECTION,
        4281,
    )
    assert (new["valid_from"], new["valid_to"], new["cik"]) == (R2_CORRECTION, None, 1790982)
    assert old["security_id"] != new["security_id"]  # 3x jump: not bridged, reported
    assert any(
        r["symbols"] == ["ARNC"] for r in conflicts_of(conflicts, "cik_switch_discontinuous")
    )
    [hwm] = segments(master, "HWM")
    assert hwm["cik"] == 4281


def test_r2_a_handful_of_switches_is_not_a_correction(r2_few: tuple[pa.Table, pa.Table]) -> None:
    # Same AFC / ARNC / HWM rows, two switches in the pair: the zombie rule does not apply and
    # AFC gets the ordinary CIK cut (bridged by continuous prices).
    master, conflicts = r2_few

    old, new = segments(master, "AFC")
    assert (old["valid_from"], old["valid_to"], old["cik"]) == (
        date(2020, 1, 2),
        R2_CORRECTION,
        3906,
    )
    assert (new["valid_from"], new["valid_to"], new["cik"]) == (R2_CORRECTION, None, 1287750)
    assert old["security_id"] == new["security_id"]
    assert len(sids(master, "ARNC")) == 2
    assert any(
        r["symbols"] == ["ARNC"] for r in conflicts_of(conflicts, "cik_switch_discontinuous")
    )


def test_r2_nothing_is_known_before_its_valid_from(r2_mass: tuple[pa.Table, pa.Table]) -> None:
    assert_nothing_known_before_its_valid_from(r2_mass[0])


# --- R3: a vacate dated on/before the first bar with no later start -------------------------
# Real (2026-10-07): BUD was vacated by a fake edge dated 2019-10-02 (stale SEC table), its
# bars start 2020-01-02 and nobody took the label afterwards, so the first real build gave it
# no segment at all. Rule: such a label is alive from its first bar. The vacate edge is given
# here as a name_change row to isolate R3 from R1; the mechanism (an "end" edge before the
# first bar, no later "start") is the same.

R3_SNAPSHOTS = [date(2019, 10, 2), date(2020, 6, 1), date(2021, 1, 4), date(2023, 6, 1)]


@pytest.mark.parametrize("vacated_on", [date(2019, 10, 2), date(2020, 1, 2)])
def test_r3_vacate_before_first_bar_without_a_later_start_is_ignored(vacated_on: date) -> None:
    # Second parameter: the edge dated exactly on the first bar day (the rule says on/before).
    master, conflicts = build_security_master(
        bars=bars_table(bar_rows("BUD", date(2020, 1, 2), date(2020, 12, 31), 60.0)),
        sec_history=sec_table(R3_SNAPSHOTS, [("BUD", 1668717, date(2019, 10, 2), None)]),
        name_changes=name_changes_table([("BUD", "ABIX", vacated_on, "03524A108", "03524A108")]),
    )

    assert len(segments(master, "BUD")) == 1
    [bud] = segments(master, "BUD")
    assert (bud["valid_from"], bud["valid_to"]) == (date(2020, 1, 2), None)
    assert bud["cik"] == 1668717
    assert segment_at(master, "BUD", date(2020, 3, 2))["security_id"] == bud["security_id"]
    assert "ABIX" not in set(master.column("symbol").to_pylist())
    assert conflicts.num_rows == 0
    assert_nothing_known_before_its_valid_from(master)


def test_r3_vacate_before_first_bar_followed_by_a_takeover_stays_dead_until_taken() -> None:
    # Must still hold: X -> Y at 2019-11-15 (before any bar), X's bars from 2020-01-02 are all
    # Z's (Alpaca relabelling, real mechanism: AIXC's history filed under FFR), Z -> X at
    # 2023-03-01. X's only segment starts at the takeover; the relabelled years get none.
    z = bar_rows("Z", date(2020, 1, 2), date(2023, 2, 28), 4.0)
    bars = bars_table(
        [
            *z,
            *relabel(z, "X"),
            *bar_rows("X", date(2023, 3, 1), date(2023, 12, 29), 4.1),
            *bar_rows("Y", date(2020, 1, 2), date(2020, 6, 30), 9.0),
        ]
    )
    sec = sec_table(
        R3_SNAPSHOTS,
        [
            ("X", 2001, date(2019, 10, 2), date(2019, 10, 2)),
            ("Y", 2001, date(2020, 6, 1), None),
            ("Z", 2002, date(2019, 10, 2), date(2021, 1, 4)),
            ("X", 2002, date(2023, 6, 1), None),
        ],
    )
    changes = name_changes_table(
        [
            ("X", "Y", date(2019, 11, 15), "200100100", "200100100"),
            ("Z", "X", date(2023, 3, 1), "200200100", "200200100"),
        ]
    )

    master, conflicts = build_security_master(bars=bars, sec_history=sec, name_changes=changes)

    assert len(segments(master, "X")) == 1
    [x] = segments(master, "X")
    assert (x["valid_from"], x["valid_to"]) == (date(2023, 3, 1), None)
    assert x["security_id"] == only_sid(master, "Z")
    assert x["security_id"] != only_sid(master, "Y")
    assert len(segments(master, "Y")) == 1
    assert segment_at(master, "Y", date(2020, 3, 2))["cik"] == 2001
    assert conflicts_of(conflicts, "dup_unlinked") == []  # X / Z duplicates sit in one chain
    assert_nothing_known_before_its_valid_from(master)


# --- R4: a pre-takeover piece made of the taker's own bars is relabelling -------------------
# Real (2026-10-07): the SEC listed BNY under CIK 1176197 until the 2020-08-13 snapshot;
# Alpaca files BK's whole 2020-2026 history under BNY too; the first real build produced a BNY
# segment [2020-01-02, 2020-08-13) cik 1176197 whose bars are all BK's. Rule: when a label is
# taken over while a previous piece exists and that piece's bars equal the taker's old
# symbol's bars on >= 90% of its days, the piece is relabelling and gets no segment.

R4_SNAPSHOTS = [
    date(2020, 1, 13),
    date(2020, 6, 1),
    date(2020, 8, 13),
    date(2021, 1, 4),
    date(2022, 1, 3),
    date(2026, 5, 4),
    date(2026, 6, 4),
]
R4_LIVES: list[Life] = [
    ("BK", 1390777, date(2020, 1, 13), date(2026, 5, 4)),  # real
    ("BNY", 1176197, date(2020, 1, 13), date(2020, 8, 13)),  # real: another registrant's row
    ("BNY", 1390777, date(2026, 6, 4), None),  # real
    # PRV / TKR: constructed boundary for the 90% threshold, see the test.
    ("PRV", 3001, date(2020, 1, 13), date(2021, 1, 4)),
    ("TKR", 3002, date(2020, 1, 13), date(2021, 1, 4)),
    ("PRV", 3002, date(2022, 1, 3), None),
]
R4_RENAMES: list[Rename] = [
    ("BK", "BNY", date(2026, 5, 21), "064058100", "064058100"),  # real
    ("TKR", "PRV", date(2022, 1, 3), "300100100", "300100100"),  # constructed
]


def r4_bars() -> list[dict[str, Any]]:
    bk = bar_rows("BK", date(2020, 1, 2), date(2026, 5, 20), 50.0)
    tkr = bar_rows("TKR", date(2020, 1, 2), date(2021, 12, 31), 30.0)
    return [
        *bk,
        *relabel(bk, "BNY"),  # real mechanism: the taker's history under the new label
        *bar_rows("BNY", date(2026, 5, 21), date(2026, 10, 5), 51.0),
        # PRV: a genuine previous holder for 2020, then (ASSUMPTION, no real example) half of
        # its pre-takeover days identical to TKR's -- about 50%, below the 90% threshold.
        *bar_rows("PRV", date(2020, 1, 2), date(2020, 12, 31), 8.0),
        *relabel([r for r in tkr if r["session_date"] >= date(2021, 1, 1)], "PRV"),
        *tkr,
        *bar_rows("PRV", date(2022, 1, 3), date(2022, 12, 30), 90.0),  # 3x jump: plain reuse
    ]


@pytest.fixture(scope="module")
def r4_built() -> tuple[pa.Table, pa.Table]:
    return build_security_master(
        bars=bars_table(r4_bars()),
        sec_history=sec_table(R4_SNAPSHOTS, R4_LIVES),
        name_changes=name_changes_table(R4_RENAMES),
    )


def test_r4_relabelled_pre_takeover_piece_gets_no_segment(
    r4_built: tuple[pa.Table, pa.Table],
) -> None:
    master, conflicts = r4_built

    assert len(segments(master, "BNY")) == 1
    [bny] = segments(master, "BNY")
    assert (bny["valid_from"], bny["valid_to"]) == (date(2026, 5, 21), None)
    assert bny["security_id"] == only_sid(master, "BK")
    assert bny["cik"] == 1390777
    assert 1176197 not in set(master.column("cik").to_pylist())
    for kind in ("reuse_looks_continuous", "dup_unlinked"):
        assert not any("BNY" in r["symbols"] for r in conflicts_of(conflicts, kind)), kind


def test_r4_previous_holder_with_mostly_different_bars_keeps_its_segment(
    r4_built: tuple[pa.Table, pa.Table],
) -> None:
    # ASSUMPTION (threshold boundary, no real example in hand): ~50% identical days is not
    # relabelling. Needs a real-data check: the identical-day ratio of every pre-takeover piece
    # the build drops vs keeps should be bimodal (near 100% vs near 0%), not spread out.
    master, _ = r4_built

    old, new = segments(master, "PRV")
    assert (old["valid_from"], old["valid_to"], old["cik"]) == (
        date(2020, 1, 2),
        date(2022, 1, 3),
        3001,
    )
    assert (new["valid_from"], new["valid_to"], new["cik"]) == (date(2022, 1, 3), None, 3002)
    assert new["security_id"] == only_sid(master, "TKR")
    assert old["security_id"] != new["security_id"]


def test_r4_dropped_piece_is_not_visible_at_an_earlier_as_of(
    r4_built: tuple[pa.Table, pa.Table],
) -> None:
    # Leakage: the relabelled 2020-2026 BNY bars belong to no segment, so a reader at 2023
    # sees no BNY at all (the only BNY segment becomes knowable 2026-05-21), and nothing is
    # knowable before its own valid_from.
    master, _ = r4_built

    visible = visible_at(master, datetime(2023, 1, 1, tzinfo=UTC))
    assert "BNY" not in {r["symbol"] for r in visible}
    assert "BK" in {r["symbol"] for r in visible}
    assert_nothing_known_before_its_valid_from(master)


# --- R5: takeover of a label still in use: same CIK + continuous prices = one security -----
# Real (2026-10-07): 437 reuse_looks_continuous conflicts in the first build, mostly temporary
# tickers (ORIC -> 1ORIC -> ORIC), SPAC completions (ACAC -> FOXX) and preferred exchanges;
# same CIK on both sides, close ratio within 0.8-1.25 across the cut. Rule: same CIK and
# continuous (CONTINUOUS, MAX_GAP_DAYS) -> bridged, no conflict; different CIK and continuous
# -> reuse_looks_continuous; different CIK and a jump -> plain reuse, no conflict.

R5_SNAPSHOTS = [
    date(2020, 1, 13),
    date(2021, 1, 4),
    date(2022, 1, 3),
    date(2024, 6, 3),
    date(2025, 1, 2),
]
R5_ORIC_DAY = date(2024, 3, 15)  # ASSUMPTION: the real round trip's date is not in hand
R5_TAKEOVER = date(2021, 7, 1)
R5_LIVES: list[Life] = [
    ("ORIC", 1796280, date(2020, 1, 13), None),  # real CIK; 1ORIC never reaches the SEC table
    # LBL: company A (4001) holds the label; B's old symbol BOLD (4002) is renamed onto it at a
    # continuous price level. Constructed (real analogue: the 437 conflicts' different-CIK tail).
    ("LBL", 4001, date(2020, 1, 13), date(2021, 1, 4)),
    ("BOLD", 4002, date(2020, 1, 13), date(2021, 1, 4)),
    ("LBL", 4002, date(2022, 1, 3), None),
    # LBJ: same shape, ~5x price jump across the takeover. Constructed.
    ("LBJ", 4003, date(2020, 1, 13), date(2021, 1, 4)),
    ("BJOLD", 4004, date(2020, 1, 13), date(2021, 1, 4)),
    ("LBJ", 4004, date(2022, 1, 3), None),
]
R5_RENAMES: list[Rename] = [
    ("ORIC", "1ORIC", R5_ORIC_DAY, "68622P109", "68622P109"),  # real pattern, same day
    ("1ORIC", "ORIC", R5_ORIC_DAY, "68622P109", "68622P109"),
    ("BOLD", "LBL", R5_TAKEOVER, "400200100", "400200100"),
    ("BJOLD", "LBJ", R5_TAKEOVER, "400400100", "400400100"),
]


def r5_bars() -> list[dict[str, Any]]:
    return [
        *bar_rows("ORIC", date(2020, 1, 2), date(2026, 10, 5), 12.0),  # continuous through
        *bar_rows("LBL", date(2020, 1, 2), date(2021, 6, 30), 10.0),
        *bar_rows("BOLD", date(2020, 1, 2), date(2021, 6, 30), 10.1),
        *bar_rows("LBL", R5_TAKEOVER, date(2021, 12, 31), 10.2),  # ~0.95x: continuous
        *bar_rows("LBJ", date(2020, 1, 2), date(2021, 6, 30), 7.0),  # distinct from LBL
        *bar_rows("BJOLD", date(2020, 1, 2), date(2021, 6, 30), 40.0),
        *bar_rows("LBJ", R5_TAKEOVER, date(2021, 12, 31), 40.5),  # ~5x: a jump
    ]


@pytest.fixture(scope="module")
def r5_built() -> tuple[pa.Table, pa.Table]:
    return build_security_master(
        bars=bars_table(r5_bars()),
        sec_history=sec_table(R5_SNAPSHOTS, R5_LIVES),
        name_changes=name_changes_table(R5_RENAMES),
    )


def test_r5_same_cik_continuous_takeover_is_one_security(
    r5_built: tuple[pa.Table, pa.Table],
) -> None:
    # One or two ORIC segments are both acceptable; what matters is one security_id and that
    # every bar day on both sides of the round trip resolves.
    master, conflicts = r5_built

    assert len(sids(master, "ORIC")) == 1
    before = segment_at(master, "ORIC", R5_ORIC_DAY - timedelta(days=1))
    after = segment_at(master, "ORIC", R5_ORIC_DAY + timedelta(days=7))
    assert before["security_id"] == after["security_id"]
    assert before["cik"] == after["cik"] == 1796280
    assert segments(master, "ORIC")[0]["valid_from"] == date(2020, 1, 2)
    assert segments(master, "ORIC")[-1]["valid_to"] is None
    assert not any("ORIC" in r["symbols"] for r in conflicts.to_pylist())


def test_r5_different_cik_continuous_takeover_is_reported_not_merged(
    r5_built: tuple[pa.Table, pa.Table],
) -> None:
    master, conflicts = r5_built

    old = segment_at(master, "LBL", date(2020, 6, 1))
    new = segment_at(master, "LBL", date(2021, 9, 1))
    assert (old["cik"], new["cik"]) == (4001, 4002)
    assert old["security_id"] != new["security_id"]
    assert new["security_id"] == only_sid(master, "BOLD")
    assert new["valid_from"] == R5_TAKEOVER
    flagged = [
        r for r in conflicts_of(conflicts, "reuse_looks_continuous") if "LBL" in r["symbols"]
    ]
    assert len(flagged) == 1
    assert R5_TAKEOVER in flagged[0]["dates"]


def test_r5_different_cik_with_a_price_jump_is_a_plain_reuse(
    r5_built: tuple[pa.Table, pa.Table],
) -> None:
    master, conflicts = r5_built

    old = segment_at(master, "LBJ", date(2020, 6, 1))
    new = segment_at(master, "LBJ", date(2021, 9, 1))
    assert (old["cik"], new["cik"]) == (4003, 4004)
    assert old["security_id"] != new["security_id"]
    assert new["security_id"] == only_sid(master, "BJOLD")
    assert not any("LBJ" in r["symbols"] for r in conflicts.to_pylist())


def test_r5_merged_pieces_are_not_known_before_their_valid_from(
    r5_built: tuple[pa.Table, pa.Table],
) -> None:
    # Leakage: bridging the ORIC round trip must not back-date anything. A reader a year before
    # the round trip sees an ORIC segment covering 2020-06-01 and nothing that starts on the
    # round-trip day; whatever starts on that day is knowable at its midnight, not earlier.
    master, _ = r5_built

    assert_nothing_known_before_its_valid_from(master)
    visible = visible_at(master, et_midnight(R5_ORIC_DAY) - timedelta(days=365))
    oric = [r for r in visible if r["symbol"] == "ORIC"]
    assert any(r["valid_from"] <= date(2020, 6, 1) for r in oric)
    assert not any(r["valid_from"] == R5_ORIC_DAY for r in oric)
    for seg in segments(master, "ORIC"):
        if seg["valid_from"] == R5_ORIC_DAY:
            assert seg["available_at"] == et_midnight(R5_ORIC_DAY)


# =============================================================================================
# F: SEC listing flicker, label handover, duplicate records (measured on the real store,
# 2026-10-07, after the R1-R5 build)
# =============================================================================================
#
# Rules these tests pin (implementation pending):
#   F1 edges_from_sec drops an edge old -> new when (a) old is listed under the same CIK in any
#      snapshot after the edge's date, or (b) new was listed under the same CIK in any snapshot
#      before date_lower. Real: of 639 SEC-only edges touching symbols with bars, 194 have the
#      old ticker back later and 181 had the new ticker earlier -- the file flips between two
#      labels (VIAC vanished from the 2020-07-10 snapshot while CBS reappeared under CIK 813828:
#      a fake VIAC -> CBS killed VIAC's segment for 2020-07..2022-02, 428 bars; SPN/SPNV flip
#      2019-11 -> 2020-07; SPAC units vs common APXTU/APXT, LATNU/LATN).
#   F2 merge_edges re-dates an SEC-only edge X -> L to the date of an Alpaca edge Y -> L when
#      that date lies in (date_lower, date]; evidence stays 'sec'. Real: HWM.WI -> HWM at
#      2020-04-01 (Alpaca) and ARNC -> HWM under CIK 4281 between 2020-06-01 and 2020-07-10
#      (SEC) are one event (Arconic Inc became Howmet); today HWM gets two takeovers and a
#      spurious reuse_looks_continuous.
#   F3 a CIK switch on a dead label (vacated by a rename, nobody took it) starts a new piece
#      at the switch's snapshot date, owned by the new CIK, its own security. Real: Arconic Inc
#      vacated ARNC (-> HWM) and the SEC lists ARNC under the new CIK 1790982 from 2020-07-10;
#      Inhibrx INXB -> INBX 2024-05-31, then INBX moves to CIK 2007919 while the old CIK
#      1739614 shows INBXV. Today the new holders' bars (ARNC to 2023-08-17, INBX to 2026) have
#      no segment.
#   F4 a CIK cut whose previous piece was ended by a rename is never bridged: continuous prices
#      -> reuse_looks_continuous (symbols [L]), otherwise nothing. Real: ARNC's pre-2020-04
#      piece ends with ARNC -> HWM; the label's continuation (Arconic Corp) is another company
#      although the closes are close (14.77 -> 15.33).
#   F5 the chain link for an edge uses the (new_symbol, date) piece whatever edge created it.
#      Today it requires piece.start is edge, so when two edges start one label on one day the
#      other edge's history is cut off (Arconic Inc's ARNC never joins HWM).
#   F6 duplicate Alpaca (old, new) rows keep the EARLIEST process_date. Real: BIGT -> MAGS is
#      filed twice (2023-11-09 and 2025-02-03; also ARNC.WI -> ARNC at 2020-04-01 and
#      2020-04-02); the last one won, MAGS (bars 2023-04-11..2024-03-28) got no segment.


# --- F1: SEC listing flicker is not a rename ------------------------------------------------

F1_SNAPSHOTS = [
    date(2019, 10, 2),
    date(2020, 1, 13),
    date(2020, 6, 1),
    date(2020, 7, 10),  # real: the snapshot VIAC was missing from
    date(2020, 8, 13),
    date(2021, 1, 4),
]
F1_FLICKER = date(2020, 7, 10)
F1_LIVES: list[Life] = [
    # CIK 813828 (real): {CBS} -> {VIAC} -> {VIAC} -> {CBS} -> {VIAC} -> {VIAC}. The shape
    # must be label A alone, then B alone, then A alone again: a snapshot listing both would
    # yield no edge at all today (gone and came are not both of size 1), so it would not
    # reproduce the fake edge. Real: VIAC vanished from 2020-07-10 while CBS reappeared.
    ("CBS", 813828, date(2019, 10, 2), date(2019, 10, 2)),
    ("VIAC", 813828, date(2020, 1, 13), date(2020, 6, 1)),
    ("CBS", 813828, F1_FLICKER, F1_FLICKER),
    ("VIAC", 813828, date(2020, 8, 13), None),
    # Genuine rename whose old name never returns and whose new name never appeared before
    # (real mechanism: GMGI -> MRDN, CIK 1451448; dates moved into this world).
    ("GMGI", 1451448, date(2019, 10, 2), date(2020, 6, 1)),
    ("MRDN", 1451448, F1_FLICKER, None),
]
# Real event: CBS became VIAC on 2019-12-05 (ViacomCBS merger). ASSUMPTION: Alpaca carries
# the record; whether it does was not checked (the shared world uses the same mechanism).
F1_RENAMES: list[Rename] = [("CBS", "VIAC", date(2019, 12, 5), "124857202", "92556H206")]


def f1_bars() -> list[dict[str, Any]]:
    return [
        # VIAC: bars on every business day of the window; CBS has none (renamed before the
        # store starts). ASSUMPTION on the end date (VIAC -> PARA in 2022-02 is not modelled).
        *bar_rows("VIAC", date(2020, 1, 2), date(2022, 2, 15), 30.0),
        *bar_rows("GMGI", date(2020, 1, 2), date(2020, 6, 15), 3.0),
        *bar_rows("MRDN", date(2020, 6, 16), date(2021, 6, 30), 3.1),
    ]


@pytest.fixture(scope="module")
def f1_sec() -> pa.Table:
    return sec_table(F1_SNAPSHOTS, F1_LIVES)


@pytest.fixture(scope="module")
def f1_built(f1_sec: pa.Table) -> tuple[pa.Table, pa.Table]:
    return build_security_master(
        bars=bars_table(f1_bars()),
        sec_history=f1_sec,
        name_changes=name_changes_table(F1_RENAMES),
    )


def test_f1_flicker_pairs_are_not_sec_edges(f1_sec: pa.Table) -> None:
    # F1. Today edges_from_sec yields CBS -> VIAC (2020-01-13), VIAC -> CBS (2020-07-10) and
    # CBS -> VIAC again (2020-08-13) for CIK 813828. All three fail rule (a) or (b): CBS is
    # listed after 2020-01-13, VIAC is listed after 2020-07-10 and before 2020-06-01, VIAC
    # was listed before 2020-07-10. The real CBS -> VIAC rename is Alpaca's to provide.
    pairs = {
        (e["old_symbol"], e["new_symbol"])
        for e in edges_from_sec(normalize_spelling(f1_sec)).to_pylist()
    }

    assert ("VIAC", "CBS") not in pairs
    assert ("CBS", "VIAC") not in pairs


def test_f1_rename_whose_old_name_never_returns_survives(
    f1_sec: pa.Table, f1_built: tuple[pa.Table, pa.Table]
) -> None:
    # F1 guard (passes today and must keep passing): GMGI is never listed again and MRDN was
    # never listed before, so the edge is kept with its snapshot interval, and the master
    # still chains the two names.
    edges = edges_from_sec(normalize_spelling(f1_sec)).to_pylist()
    [edge] = [e for e in edges if (e["old_symbol"], e["new_symbol"]) == ("GMGI", "MRDN")]
    assert (edge["date"], edge["date_lower"]) == (F1_FLICKER, date(2020, 6, 1))

    master, _ = f1_built
    assert only_sid(master, "GMGI") == only_sid(master, "MRDN")


def test_f1_flicker_leaves_viac_one_open_ended_segment(
    f1_built: tuple[pa.Table, pa.Table],
) -> None:
    # F1. Today the fake VIAC -> CBS edge ends VIAC's segment at 2020-07-10 and every later
    # bar (real: 428 of them) belongs to no security.
    master, conflicts = f1_built

    assert len(segments(master, "VIAC")) == 1
    [viac] = segments(master, "VIAC")
    assert viac["valid_to"] is None
    assert viac["valid_from"] <= date(2020, 1, 2)  # covers the first bar
    assert viac["cik"] == 813828
    for day in (
        date(2020, 3, 2),  # before the snapshot VIAC was missing from
        F1_FLICKER,  # the snapshot day itself
        date(2020, 7, 20),  # between that snapshot and the one listing VIAC again
        date(2021, 3, 1),  # after
        date(2022, 2, 15),  # last bar
    ):
        assert segment_at(master, "VIAC", day)["security_id"] == viac["security_id"], day
    assert "CBS" not in set(master.column("symbol").to_pylist())  # no bars, no segment
    assert not any("VIAC" in r["symbols"] for r in conflicts.to_pylist())


def test_f1_no_segment_boundary_appears_at_the_flicker_snapshot(
    f1_built: tuple[pa.Table, pa.Table],
) -> None:
    # Leakage: a reader the day after the flicker snapshot must see exactly one VIAC row and
    # it must be open-ended -- the snapshot created no event, so nothing becomes knowable at
    # its midnight. And nothing anywhere is knowable before its own valid_from.
    master, _ = f1_built

    seen = [
        r
        for r in visible_at(master, et_midnight(F1_FLICKER + timedelta(days=1)))
        if r["symbol"] == "VIAC"
    ]
    assert len(seen) == 1
    assert seen[0]["valid_to"] is None
    assert not any(r["valid_from"] == F1_FLICKER for r in segments(master, "VIAC"))
    assert_nothing_known_before_its_valid_from(master)


# --- F2: one event, two records (Alpaca HWM.WI -> HWM, SEC ARNC -> HWM) --------------------
# Real file: Alpaca HWM.WI -> HWM at 2020-04-01; the SEC pair (2020-06-01, 2020-07-10] shows
# ARNC -> HWM under 4281, i.e. the SEC still listed ARNC on 06-01 (one snapshot of lag).
# _refine_sec_edges cannot pull the SEC edge back to 04-01 because ARNC keeps trading
# (Arconic Corp took the label), so HWM gets two starts. NOTE: the rule as stated needs the
# Alpaca date inside (date_lower, date]; with the real lag it is not (04-01 < 06-01). The
# world below shortens the lag so the stated rule applies (ARNC last listed 01-13, HWM first
# listed 06-01) -- ASSUMPTION flagged for the author: the real shape needs the rule to allow
# one snapshot of lag, or the real-data basis is a different edge.

F2_SNAPSHOTS = [date(2020, 1, 13), date(2020, 6, 1), date(2020, 7, 10), date(2021, 1, 4)]
F2_RENAME_DAY = date(2020, 4, 1)  # real: Alpaca HWM.WI -> HWM process_date
F2_SEC_DATE = date(2020, 6, 1)  # the SEC edge's date before re-dating (first snapshot with HWM)
F2_LIVES: list[Life] = [
    ("ARNC", 4281, date(2020, 1, 13), date(2020, 1, 13)),  # Arconic Inc (lag shortened, above)
    ("HWM", 4281, F2_SEC_DATE, None),  # real CIK
    ("ARNC", 1790982, date(2020, 7, 10), None),  # real: Arconic Corp takes the label (F3)
]
# Real record. HWM.WI has no bars in this world: the rule only needs the edge; whether Alpaca
# serves when-issued bars was not checked.
F2_RENAMES: list[Rename] = [("HWM.WI", "HWM", F2_RENAME_DAY, "443201108", "443201108")]


def f2_bars() -> list[dict[str, Any]]:
    return [
        # ARNC trades all year (real: the label continued under Arconic Corp to 2023-08-17),
        # which is what defeats _refine_sec_edges. ASSUMPTION on levels.
        *bar_rows("ARNC", date(2020, 1, 2), date(2020, 12, 31), 25.0),
        *bar_rows("HWM", F2_RENAME_DAY, date(2020, 12, 31), 18.0),
    ]


@pytest.fixture(scope="module")
def f2_built() -> tuple[pa.Table, pa.Table]:
    return build_security_master(
        bars=bars_table(f2_bars()),
        sec_history=sec_table(F2_SNAPSHOTS, F2_LIVES),
        name_changes=name_changes_table(F2_RENAMES),
    )


def test_f2_sec_edge_into_a_label_alpaca_started_inside_its_interval_takes_alpacas_date() -> None:
    # F2 on merge_edges alone: SEC ARNC -> HWM in (2020-01-13, 2020-06-01], Alpaca
    # HWM.WI -> HWM at 2020-04-01 inside it -> the SEC edge is re-dated, evidence stays 'sec'.
    sec = sec_table(F2_SNAPSHOTS[:2], F2_LIVES[:2])
    merged = merge_edges(
        edges_from_name_changes(name_changes_table(F2_RENAMES)),
        edges_from_sec(normalize_spelling(sec)),
    ).to_pylist()
    by_pair = {(e["old_symbol"], e["new_symbol"]): e for e in merged}

    assert set(by_pair) == {("ARNC", "HWM"), ("HWM.WI", "HWM")}
    assert by_pair[("ARNC", "HWM")]["date"] == F2_RENAME_DAY
    assert by_pair[("ARNC", "HWM")]["evidence"] == "sec"
    assert by_pair[("HWM.WI", "HWM")]["date"] == F2_RENAME_DAY


def test_f2_alpaca_edge_after_the_sec_edges_date_leaves_the_sec_edge_alone() -> None:
    # F2 boundary: an Alpaca rename into the same label dated *after* the SEC edge's date is
    # a later, separate handover (constructed): the SEC edge keeps its own date.
    later = name_changes_table([("HWM.WI", "HWM", date(2021, 3, 1), "443201108", "443201108")])
    merged = merge_edges(
        edges_from_name_changes(later),
        edges_from_sec(normalize_spelling(sec_table(F2_SNAPSHOTS[:2], F2_LIVES[:2]))),
    ).to_pylist()
    by_pair = {(e["old_symbol"], e["new_symbol"]): e for e in merged}

    assert by_pair[("ARNC", "HWM")]["date"] == F2_SEC_DATE
    assert by_pair[("ARNC", "HWM")]["date_lower"] == date(2020, 1, 13)
    assert by_pair[("HWM.WI", "HWM")]["date"] == date(2021, 3, 1)


def test_f2_one_rename_seen_twice_gives_hwm_one_segment_chained_to_arnc(
    f2_built: tuple[pa.Table, pa.Table],
) -> None:
    # F2 end to end. Today: HWM [04-01, 06-01) + HWM [06-01, None), two security_ids, and a
    # reuse_looks_continuous [ARNC, HWM]. The shared-id assertion also needs F5: after
    # re-dating, two edges start HWM on 04-01 and only one of them can be the piece's start.
    # ARNC's later segment (Arconic Corp, F3) is not asserted here: only its first one.
    master, conflicts = f2_built

    assert len(segments(master, "HWM")) == 1
    [hwm] = segments(master, "HWM")
    assert (hwm["valid_from"], hwm["valid_to"]) == (F2_RENAME_DAY, None)
    assert hwm["cik"] == 4281
    arnc = segments(master, "ARNC")[0]
    assert (arnc["valid_from"], arnc["valid_to"]) == (date(2020, 1, 2), F2_RENAME_DAY)
    assert arnc["security_id"] == hwm["security_id"]
    assert not any("HWM" in r["symbols"] for r in conflicts.to_pylist())


def test_f2_nothing_about_hwm_becomes_knowable_at_the_later_snapshot(
    f2_built: tuple[pa.Table, pa.Table],
) -> None:
    # Leakage: the SEC snapshot of 2020-06-01 adds no event, so a reader on 2020-05-01 already
    # sees HWM's whole (only) segment, and no HWM row starts on the snapshot day.
    master, _ = f2_built

    seen = [r for r in visible_at(master, et_midnight(date(2020, 5, 1))) if r["symbol"] == "HWM"]
    assert len(seen) == 1 and seen[0]["valid_from"] == F2_RENAME_DAY
    assert not any(r["valid_from"] == F2_SEC_DATE for r in segments(master, "HWM"))
    assert_nothing_known_before_its_valid_from(master)


# --- F3: a dead label taken by a new filer (ARNC -> Arconic Corp, CIK 1790982) ---------------

F3_SNAPSHOTS = [date(2020, 1, 13), date(2020, 6, 1), date(2020, 7, 10), date(2021, 1, 4)]
F3_VACATED = date(2020, 4, 1)  # d1: Arconic Inc leaves the label (real day)
F3_NEW_CIK_SNAPSHOT = date(2020, 7, 10)  # d2: first snapshot listing ARNC under 1790982 (real)
F3_LIVES: list[Life] = [
    ("ARNC", 4281, date(2020, 1, 13), date(2020, 6, 1)),  # real
    # ASSUMPTION (as in R2): HWM already next to ARNC under 4281 on 06-01, so no SEC edge
    # ARNC -> HWM exists and F3 is isolated from F2 / F4.
    ("HWM", 4281, date(2020, 6, 1), None),
    ("ARNC", 1790982, F3_NEW_CIK_SNAPSHOT, None),  # real: Arconic Corp
]
# ASSUMPTION: the vacate is given as a direct Alpaca ARNC -> HWM record (the real one is
# HWM.WI -> HWM plus the SEC pair, see F2) to keep the world about the dead label only.
F3_RENAMES: list[Rename] = [("ARNC", "HWM", F3_VACATED, "03965L100", "443201108")]


def f3_bars() -> list[dict[str, Any]]:
    return [
        *bar_rows("ARNC", date(2020, 1, 2), date(2020, 3, 31), 25.0),  # Arconic Inc
        # Arconic Corp keeps the label with bars from the vacate day on. ASSUMPTION on the
        # level: 3x so continuity never comes into it (that is F4's question).
        *bar_rows("ARNC", F3_VACATED, date(2020, 12, 31), 75.0),
        *bar_rows("HWM", F3_VACATED, date(2020, 12, 31), 18.0),
    ]


@pytest.fixture(scope="module")
def f3_built() -> tuple[pa.Table, pa.Table]:
    return build_security_master(
        bars=bars_table(f3_bars()),
        sec_history=sec_table(F3_SNAPSHOTS, F3_LIVES),
        name_changes=name_changes_table(F3_RENAMES),
    )


def test_f3_cik_switch_on_a_dead_label_starts_the_new_holders_segment(
    f3_built: tuple[pa.Table, pa.Table],
) -> None:
    # F3. Today ARNC has one segment [2020-01-02, 2020-04-01) and the CIK switch at 07-10
    # finds no live piece to cut, so Arconic Corp's bars (real: to 2023-08-17) have no segment.
    master, _ = f3_built

    assert len(segments(master, "ARNC")) == 2
    old, new = segments(master, "ARNC")
    assert (old["valid_from"], old["valid_to"], old["cik"]) == (date(2020, 1, 2), F3_VACATED, 4281)
    assert old["security_id"] == only_sid(master, "HWM")
    assert (new["valid_from"], new["valid_to"], new["cik"]) == (F3_NEW_CIK_SNAPSHOT, None, 1790982)
    assert new["security_id"] != old["security_id"]
    for day in (F3_NEW_CIK_SNAPSHOT, date(2020, 8, 3), date(2020, 12, 31)):
        assert segment_at(master, "ARNC", day)["security_id"] == new["security_id"], day
    # ASSUMPTION about [d1, d2): the SEC first shows the new filer on d2, so the build starts
    # the new piece there and the Arconic Corp bars of 2020-04-01..07-09 stay uncovered; the
    # gap is accepted and deliberately not asserted either way.


def test_f3_new_holders_segment_is_not_knowable_before_its_snapshot(
    f3_built: tuple[pa.Table, pa.Table],
) -> None:
    # Leakage: the new segment exists because of the 07-10 snapshot; a reader on 2020-05-01
    # sees only Arconic Inc's ARNC row.
    master, _ = f3_built

    assert len(segments(master, "ARNC")) == 2
    _, new = segments(master, "ARNC")
    assert new["available_at"] == et_midnight(F3_NEW_CIK_SNAPSHOT)
    seen = [r for r in visible_at(master, et_midnight(date(2020, 5, 1))) if r["symbol"] == "ARNC"]
    assert [r["valid_from"] for r in seen] == [date(2020, 1, 2)]
    assert_nothing_known_before_its_valid_from(master)


# --- F4: no bridging across a rename-ended piece ---------------------------------------------
# Sibling of F3 shaped like the real file: no Alpaca record, the SEC pair (06-01, 07-10] shows
# ARNC -> HWM under 4281 and ARNC under 1790982 from 07-10. The rename edge is dated at the
# snapshot (ARNC has bars through the interval, so _refine_sec_edges does not move it) and the
# CIK switch falls on the same day, so the piece before the cut was ended by a rename.
# GOOG-style bridging (CIK switch inside a label nobody renamed, continuous prices) is covered
# by test_cik_switch_with_continuous_prices_stays_one_security and must keep passing.

F4_CUT = date(2020, 7, 10)
F4_LIVES: list[Life] = [
    ("ARNC", 4281, date(2020, 1, 13), date(2020, 6, 1)),  # real
    ("HWM", 4281, F4_CUT, None),  # real
    ("ARNC", 1790982, F4_CUT, None),  # real
]


def f4_inputs(*, continuous: bool) -> tuple[pa.Table, pa.Table]:
    if continuous:
        # Real: 14.77 -> 15.33 across the handover; one drifting series reproduces the ratio.
        arnc = bar_rows("ARNC", date(2020, 1, 2), date(2020, 12, 31), 15.0)
    else:
        arnc = bar_rows("ARNC", date(2020, 1, 2), date(2020, 7, 9), 15.0) + bar_rows(
            "ARNC",
            F4_CUT,
            date(2020, 12, 31),
            75.0,  # ASSUMPTION: a 5x jump
        )
    # HWM bars from 2020-04-01 (real); without an HWM.WI record in this world its 04-01..07-09
    # bars are the F2 gap and are not under test here.
    hwm = bar_rows("HWM", date(2020, 4, 1), date(2020, 12, 31), 18.0)
    return build_security_master(
        bars=bars_table(arnc + hwm),
        sec_history=sec_table(F3_SNAPSHOTS, F4_LIVES),
        name_changes=name_changes_table([]),
    )


def test_f4_continuous_prices_across_a_rename_ended_piece_are_reported_not_bridged() -> None:
    # F4. Today the cut at 07-10 sees the rename-ended piece on its left and continuous closes,
    # bridges them, and Arconic Inc + Arconic Corp become one security.
    master, conflicts = f4_inputs(continuous=True)

    assert len(sids(master, "ARNC")) == 2
    old, new = segments(master, "ARNC")
    assert old["valid_to"] == new["valid_from"] == F4_CUT
    assert old["security_id"] == only_sid(master, "HWM")
    assert (old["cik"], new["cik"]) == (4281, 1790982)
    flagged = [
        r for r in conflicts_of(conflicts, "reuse_looks_continuous") if "ARNC" in r["symbols"]
    ]
    assert len(flagged) == 1
    assert set(flagged[0]["symbols"]) == {"ARNC"}
    assert F4_CUT in flagged[0]["dates"]
    assert conflicts_of(conflicts, "cik_switch_discontinuous") == []
    assert_nothing_known_before_its_valid_from(master)


def test_f4_discontinuous_prices_across_a_rename_ended_piece_are_two_securities_quietly() -> None:
    # F4, other branch: the label changed hands by rename and the price jumped: a plain reuse,
    # two securities, nothing to report.
    master, conflicts = f4_inputs(continuous=False)

    assert len(sids(master, "ARNC")) == 2
    old, new = segments(master, "ARNC")
    assert old["valid_to"] == new["valid_from"] == F4_CUT
    assert not any("ARNC" in r["symbols"] for r in conflicts.to_pylist())
    assert_nothing_known_before_its_valid_from(master)


# --- F5: same-day double start ---------------------------------------------------------------


def test_f5_two_edges_starting_one_label_on_one_day_both_link() -> None:
    # F5. ASSUMPTION about the world: two labels merging into one on one day is the simplest
    # shape; the real case is F2's (HWM.WI -> HWM and the re-dated ARNC -> HWM, both 04-01),
    # whose shared-id assertion depends on this rule. Today the piece's start is one of the two
    # edges and the other edge's chain link is dropped (A stays its own security).
    day = date(2020, 7, 1)
    master, conflicts = build_security_master(
        bars=bars_table(
            bar_rows("A", date(2020, 1, 2), date(2020, 6, 30), 10.0)
            + bar_rows("B", date(2020, 1, 2), date(2020, 6, 30), 20.0)
            + bar_rows("L", day, date(2020, 12, 31), 10.2)
        ),
        sec_history=sec_table([], []),
        name_changes=name_changes_table(
            [("A", "L", day, "000000100", "000000300"), ("B", "L", day, "000000200", "000000300")]
        ),
    )

    assert len(segments(master, "L")) == 1
    assert only_sid(master, "A") == only_sid(master, "B") == only_sid(master, "L")
    assert segments(master, "A")[0]["valid_to"] == segments(master, "B")[0]["valid_to"] == day
    assert conflicts.num_rows == 0
    assert_nothing_known_before_its_valid_from(master)


# --- F6: duplicate Alpaca records keep the earliest date ------------------------------------

F6_FIRST = date(2023, 11, 9)  # real process_date of the first BIGT -> MAGS row
F6_SECOND = date(2025, 2, 3)  # real process_date of the duplicate
F6_RENAMES: list[Rename] = [
    ("BIGT", "MAGS", F6_FIRST, "089500100", "559500100"),
    ("BIGT", "MAGS", F6_SECOND, "089500100", "559500100"),
]


def f6_inputs() -> tuple[pa.Table, pa.Table]:
    # Real spans: BIGT 2023-04-11..2023-11-08, MAGS 2023-04-11..2024-03-28 (BIGT's history
    # relabelled under MAGS, then its own bars). No SEC rows: neither name reached the table
    # in this world (ASSUMPTION; the rule is about the Alpaca feed alone).
    bigt = bar_rows("BIGT", date(2023, 4, 11), date(2023, 11, 8), 5.0)
    return build_security_master(
        bars=bars_table(
            bigt + relabel(bigt, "MAGS") + bar_rows("MAGS", F6_FIRST, date(2024, 3, 28), 5.1)
        ),
        sec_history=sec_table([], []),
        name_changes=name_changes_table(F6_RENAMES),
    )


def test_f6_duplicate_alpaca_rows_merge_to_the_earliest_date() -> None:
    # F6. Today the later row overwrites the earlier one in merge_edges.
    merged = merge_edges(
        edges_from_name_changes(name_changes_table(F6_RENAMES)), edges_from_sec(sec_table([], []))
    ).to_pylist()

    assert len(merged) == 1
    assert merged[0]["date"] == F6_FIRST
    assert merged[0]["evidence"] == "name_change"


def test_f6_mags_gets_its_segment_from_the_first_record() -> None:
    # F6 end to end. Today the rename is dated 2025-02-03, after MAGS's last bar, so MAGS's
    # only piece has no bars and is dropped: no segment at all, and BIGT/MAGS become a
    # dup_unlinked conflict.
    master, conflicts = f6_inputs()

    assert len(segments(master, "MAGS")) == 1
    [mags] = segments(master, "MAGS")
    assert mags["valid_from"] == F6_FIRST  # valid_to None or the duplicate's date: either
    assert segment_at(master, "MAGS", date(2024, 1, 2))["security_id"] == mags["security_id"]
    [bigt] = segments(master, "BIGT")
    assert (bigt["valid_from"], bigt["valid_to"]) == (date(2023, 4, 11), F6_FIRST)
    assert bigt["security_id"] == mags["security_id"]
    assert not any(
        set(r["symbols"]) == {"BIGT", "MAGS"} for r in conflicts_of(conflicts, "dup_unlinked")
    )


def test_f6_relabelled_mags_history_is_in_no_segment_and_nothing_leaks() -> None:
    # Leakage both ways: MAGS's relabelled 2023-04..11 bars resolve to nothing (segment_at
    # must find no hit), the segment is knowable at the first record's midnight -- so a reader
    # on 2024-01-02 does see it (with the later date it would have been invisible for a year
    # although the rename had happened) -- and nothing is knowable before its valid_from.
    master, _ = f6_inputs()

    early = [
        r
        for r in segments(master, "MAGS")
        if r["valid_from"] <= date(2023, 6, 1)
        and (r["valid_to"] is None or date(2023, 6, 1) < r["valid_to"])
    ]
    assert early == []
    assert len(segments(master, "MAGS")) == 1
    [mags] = segments(master, "MAGS")
    assert mags["available_at"] == et_midnight(F6_FIRST)
    assert "MAGS" in {r["symbol"] for r in visible_at(master, et_midnight(date(2024, 1, 2)))}
    assert_nothing_known_before_its_valid_from(master)


# --- F7: an SEC edge across a listing gap is not a rename -----------------------------------
# Real (2026-10-07): CIK 1872195 was listed as BULL up to the 2023-06-01 snapshot, absent from
# every snapshot for 26 months, then listed as BLSH from 2025-08-19. edges_from_sec pairs the
# consecutive snapshots in which the CIK is present, so it emitted BULL -> BLSH in
# (2023-06-01, 2025-08-19]; the cut stayed at 2025-08-19 (BULL had bars right up to it) because
# Webull had taken BULL meanwhile (CIK 1866364, SKGR -> BULL 2025-04-11): Webull's BULL segment
# was ended at 2025-08-19 and 284 bars lost. Rule: an SEC edge whose interval contains two or
# more intermediate global snapshots (present in the table, this CIK absent) is dropped; one
# missing snapshot is tolerated (the 2020-07-10 purge dropped many tickers for one month:
# HTF -> HTFA 2020-06-01 -> 2020-08-13 is real).

F7_SNAPSHOTS = [  # monthly; the 26-month gap is compressed to three snapshots
    date(2025, 1, 2),
    date(2025, 2, 3),
    date(2025, 3, 3),
    date(2025, 4, 1),
    date(2025, 5, 1),
    date(2025, 6, 2),
    date(2025, 8, 19),  # real: first snapshot listing BLSH
    date(2025, 9, 2),
]
F7_WEBULL_DAY = date(2025, 4, 11)  # real: Alpaca SKGR -> BULL
F7_LIVES: list[Life] = [
    ("BULL", 1872195, date(2025, 1, 2), date(2025, 3, 3)),  # real CIK; absent 04-01..06-02
    ("BLSH", 1872195, date(2025, 8, 19), None),  # real
    ("SKGR", 1866364, date(2025, 1, 2), date(2025, 4, 1)),  # real CIK (Webull's SPAC)
    ("BULL", 1866364, date(2025, 5, 1), None),  # real: Webull holds BULL
]
F7_RENAMES: list[Rename] = [("SKGR", "BULL", F7_WEBULL_DAY, "783760104", "947833108")]


def f7_bars() -> list[dict[str, Any]]:
    return [
        # ASSUMPTION on the first holder's last bar and on all price levels; the 2.2x level
        # between the two BULL holders keeps the takeover a plain reuse (not F4's question).
        *bar_rows("BULL", date(2025, 1, 2), date(2025, 3, 14), 10.0),
        *bar_rows("SKGR", date(2025, 1, 2), date(2025, 4, 10), 20.0),
        *bar_rows("BULL", F7_WEBULL_DAY, date(2025, 12, 31), 22.0),  # real: bars continue
        *bar_rows("BLSH", date(2025, 8, 19), date(2025, 12, 31), 40.0),
    ]


@pytest.fixture(scope="module")
def f7_sec() -> pa.Table:
    return sec_table(F7_SNAPSHOTS, F7_LIVES)


@pytest.fixture(scope="module")
def f7_built(f7_sec: pa.Table) -> tuple[pa.Table, pa.Table]:
    return build_security_master(
        bars=bars_table(f7_bars()),
        sec_history=f7_sec,
        name_changes=name_changes_table(F7_RENAMES),
    )


def f7_gap_sec(missing: int) -> pa.Table:
    """OLD listed on the first snapshot, the CIK absent on the next ``missing`` global
    snapshots (a filler CIK keeps them in the table), NEW listed from then on."""
    snaps = F7_SNAPSHOTS[: missing + 2]
    return sec_table(
        snaps,
        [
            ("OLD", 7001, snaps[0], snaps[0]),
            ("NEW", 7001, snaps[-1], None),
            ("FILL", 7002, snaps[0], None),
        ],
    )


def test_f7_sec_edge_across_a_listing_gap_is_dropped(f7_sec: pa.Table) -> None:
    # F7. Today the pair (2025-03-03, 2025-08-19] for CIK 1872195 yields BULL -> BLSH.
    pairs = {
        (e["old_symbol"], e["new_symbol"])
        for e in edges_from_sec(normalize_spelling(f7_sec)).to_pylist()
    }

    assert ("BULL", "BLSH") not in pairs
    assert pairs == {("SKGR", "BULL")}  # Webull's own rename (same CIK) survives


@pytest.mark.parametrize("missing", [2, 3])
def test_f7_two_or_more_missing_snapshots_break_the_edge(missing: int) -> None:
    # F7 threshold: two intermediate global snapshots without the CIK already mean "delisted
    # and relisted", not a rename.
    edges = edges_from_sec(normalize_spelling(f7_gap_sec(missing))).to_pylist()

    assert [(e["old_symbol"], e["new_symbol"]) for e in edges] == []


def test_f7_one_missing_snapshot_is_tolerated() -> None:
    # F7 guard (passes today and must keep passing). Real: HTF -> HTFA, listed 2020-06-01,
    # absent from the 2020-07-10 purge snapshot, HTFA from 2020-08-13. The edge keeps the NEW
    # snapshot as its date and the last snapshot showing OLD as date_lower.
    sec = sec_table(
        [date(2020, 6, 1), date(2020, 7, 10), date(2020, 8, 13)],
        [
            ("HTF", 1000010, date(2020, 6, 1), date(2020, 6, 1)),
            ("HTFA", 1000010, date(2020, 8, 13), None),
            ("FILL", 1000011, date(2020, 6, 1), None),  # keeps 07-10 a global snapshot
        ],
    )
    [edge] = edges_from_sec(normalize_spelling(sec)).to_pylist()

    assert (edge["old_symbol"], edge["new_symbol"]) == ("HTF", "HTFA")
    assert (edge["date"], edge["date_lower"]) == (date(2020, 8, 13), date(2020, 6, 1))


def test_f7_the_labels_current_holder_is_not_cut_by_the_dropped_edge(
    f7_built: tuple[pa.Table, pa.Table],
) -> None:
    # F7 end to end. Today Webull's BULL segment is [2025-04-11, 2025-08-19) and its later
    # bars (real: 284) belong to no security; BLSH is chained to the first BULL holder.
    master, conflicts = f7_built

    assert len(segments(master, "BULL")) == 2
    first, webull = segments(master, "BULL")
    assert (first["valid_from"], first["valid_to"], first["cik"]) == (
        date(2025, 1, 2),
        F7_WEBULL_DAY,
        1872195,
    )
    assert (webull["valid_from"], webull["valid_to"], webull["cik"]) == (
        F7_WEBULL_DAY,
        None,
        1866364,
    )
    assert webull["security_id"] == only_sid(master, "SKGR")
    for day in (date(2025, 8, 19), date(2025, 10, 1), date(2025, 12, 31)):
        assert segment_at(master, "BULL", day)["security_id"] == webull["security_id"], day
    assert len(segments(master, "BLSH")) == 1
    assert only_sid(master, "BLSH") not in sids(master, "BULL")
    assert not any("BULL" in r["symbols"] for r in conflicts.to_pylist())


def test_f7_nothing_about_bull_becomes_knowable_at_the_relisting_snapshot(
    f7_built: tuple[pa.Table, pa.Table],
) -> None:
    # Leakage: the 2025-08-19 snapshot creates no event for BULL, so a reader on 2025-06-01
    # already sees Webull's segment open-ended, and no BULL row ends or starts on that day.
    master, _ = f7_built

    seen = [r for r in visible_at(master, et_midnight(date(2025, 6, 1))) if r["symbol"] == "BULL"]
    assert any(r["valid_from"] == F7_WEBULL_DAY and r["valid_to"] is None for r in seen)
    assert not any(
        date(2025, 8, 19) in (r["valid_from"], r["valid_to"]) for r in segments(master, "BULL")
    )
    assert_nothing_known_before_its_valid_from(master)


# =============================================================================================
# F9/F10: late Alpaca dates, CIK switch next to a rename (measured on the real store,
# 2026-10-07, after the F1-F7 build)
# =============================================================================================
#
# Rules these tests pin (implementation pending):
#   F9  For ANY edge old -> new (not only SEC-only ones, which _refine_sec_edges already moves):
#       if old's last bar is before ``date``, the day after that last bar lies within
#       MAX_GAP_DAYS (10 calendar days) before ``date``, and new has at least one bar in
#       [last_bar + 1, date), the cut moves to last_bar + 1. The segment's ``available_at``
#       stays the edge's ORIGINAL availability (midnight of Alpaca's process_date): the rename
#       was only knowable when Alpaca filed it, so a segment may be knowable later than midnight
#       of its own valid_from. Real: after F1-F7 the uncovered bars with the largest turnover are
#       the first days of SPAC completions -- NKLA 2020-06-04/05 (VTIQ -> NKLA), FSR
#       2020-10-30..11-06 (6 days), LOTZ 2021-01-22..27 (4), WE 2021-10-21..26 (4), ME
#       2021-06-17..21 (3), GNOG 2021-01-04/05, RIDE 2020-10-26. In each the old symbol's bars
#       stop the day before the new label's first bar and Alpaca's record is dated a few days
#       later.
#   F10 A CIK switch in the same snapshot interval as a rename into the label is suppressed
#       today ("the rename explains the switch"). It is explained only if the rename's OLD
#       symbol was ever SEC-listed under one of the CIKs the label carries at the switch's
#       upper snapshot; otherwise the switch stays an event (F3: a new piece on a dead label,
#       a cut on a live one). Real: Inhibrx -- Alpaca INXB -> INBX 2024-05-31; in the SEC pair
#       (2024-05-02, 2024-06-06] INBX moves from CIK 1739614 (Inhibrx Inc) to 2007919 (Inhibrx
#       Biosciences, the new holding company) while 1739614 continues as INBXV. The build gave
#       INBX [2024-05-29, 05-31) and [05-31, 06-06) and nothing after, though it trades to
#       2026-10.


# --- F9: Alpaca's rename date lags the first trading day under the new label ----------------

F9_SNAPSHOTS = [date(2020, 1, 13), date(2020, 7, 10), date(2021, 1, 4)]
F9_NKLA_FIRST_BAR = date(2020, 6, 4)  # real: first NKLA bar in the store
F9_ALPACA_DAY = date(2020, 6, 8)  # real: the first day NKLA's bars are covered today, i.e. the
#   process_date of Alpaca's VTIQ -> NKLA record (the uncovered bars are exactly 06-04 and 06-05)
F9_HALT_DAY = date(2020, 6, 8)  # constructed: same record date, old bars stop 05-01
F9_RELABEL_DAY = date(2020, 6, 9)  # constructed: a Tuesday, so old's last bar is date - 1
F9_LIVES: list[Life] = [
    # VectoIQ Acquisition (CIK 1731289) became Nikola: same CIK on both labels (real).
    ("VTIQ", 1731289, date(2020, 1, 13), date(2020, 1, 13)),
    ("NKLA", 1731289, date(2020, 7, 10), None),
    # HLTO -> HLTN: a long halt, constructed (no real example in hand: see the test).
    ("HLTO", 9001, date(2020, 1, 13), date(2020, 1, 13)),
    ("HLTN", 9001, date(2020, 7, 10), None),
    # VIAC listed throughout; CBS renamed before the first snapshot of this world (real event
    # 2019-12-05; CBS never appears here, which keeps F1's flicker out of this world).
    ("VIAC", 813828, date(2020, 1, 13), None),
    # RLBO -> RLBN: constructed, the taker's history relabelled under the new name (real
    # mechanism: BK's bars under BNY, AIXC's under FFR).
    ("RLBO", 9002, date(2020, 1, 13), date(2020, 1, 13)),
    ("RLBN", 9002, date(2020, 7, 10), None),
    # SOLD -> SNEW: SEC-only edge, constructed guard for _refine_sec_edges (real analogue:
    # GMGI -> MRDN in the shared world).
    ("SOLD", 9003, date(2020, 1, 13), date(2020, 1, 13)),
    ("SNEW", 9003, date(2020, 7, 10), None),
]
F9_RENAMES: list[Rename] = [
    ("VTIQ", "NKLA", F9_ALPACA_DAY, "92243N108", "654110105"),  # CUSIPs not checked
    ("HLTO", "HLTN", F9_HALT_DAY, "900100100", "900100200"),
    # Real event date (ViacomCBS merger). ASSUMPTION as in F1: Alpaca carries the record.
    ("CBS", "VIAC", date(2019, 12, 5), "124857202", "92556H206"),
    ("RLBO", "RLBN", F9_RELABEL_DAY, "900200100", "900200200"),
]


def f9_bars() -> list[dict[str, Any]]:
    rlbo = bar_rows("RLBO", date(2020, 1, 2), date(2020, 6, 8), 40.0)
    return [
        # Real shape: VTIQ's bars stop 06-03, NKLA's start 06-04, Alpaca's record is 06-08.
        # Price levels: ASSUMPTION (a SPAC completion trades at the trust level, ~10).
        *bar_rows("VTIQ", date(2020, 1, 2), date(2020, 6, 3), 10.0),
        *bar_rows("NKLA", F9_NKLA_FIRST_BAR, date(2020, 12, 31), 10.1),
        # Long halt: 05-02 .. 06-08 is 37 days, well over MAX_GAP_DAYS. Constructed.
        *bar_rows("HLTO", date(2020, 1, 2), date(2020, 5, 1), 20.0),
        *bar_rows("HLTN", date(2020, 5, 4), date(2020, 12, 31), 20.1),
        # CBS has no bars at all (real: renamed before the store starts). VIAC from day one.
        *bar_rows("VIAC", date(2020, 1, 2), date(2020, 12, 31), 30.0),
        # RLBO trades through 06-08 (= date - 1); RLBN carries RLBO's history plus its own.
        *rlbo,
        *relabel(rlbo, "RLBN"),
        *bar_rows("RLBN", F9_RELABEL_DAY, date(2020, 12, 31), 40.5),
        # SEC-only edge in (01-13, 07-10]: SOLD's last bar 03-13 is four months before the
        # snapshot, far outside MAX_GAP_DAYS -- the SEC rule has no gap limit and must keep
        # cutting at last_bar + 1.
        *bar_rows("SOLD", date(2020, 1, 2), date(2020, 3, 13), 50.0),
        *bar_rows("SNEW", date(2020, 3, 16), date(2020, 12, 31), 50.1),
    ]


@pytest.fixture(scope="module")
def f9_built() -> tuple[pa.Table, pa.Table]:
    return build_security_master(
        bars=bars_table(f9_bars()),
        sec_history=sec_table(F9_SNAPSHOTS, F9_LIVES),
        name_changes=name_changes_table(F9_RENAMES),
    )


def test_f9_late_alpaca_date_cut_moves_to_the_new_labels_first_bar(
    f9_built: tuple[pa.Table, pa.Table],
) -> None:
    # F9 (1), the NKLA shape. Today the cut sits on Alpaca's 06-08 and the 06-04 / 06-05 bars
    # (real: NKLA's first two days, the largest uncovered turnover in the store) resolve to
    # nothing. VTIQ's last bar 06-03 + 1 = 06-04, 4 days before 06-08, and NKLA has bars there.
    master, conflicts = f9_built

    [vtiq] = segments(master, "VTIQ")
    [nkla] = segments(master, "NKLA")
    assert vtiq["valid_to"] == nkla["valid_from"] == F9_NKLA_FIRST_BAR
    assert nkla["valid_to"] is None
    assert vtiq["security_id"] == nkla["security_id"]
    assert len(sids(master, "NKLA")) == 1
    for day in (F9_NKLA_FIRST_BAR, date(2020, 6, 5), F9_ALPACA_DAY, date(2020, 9, 1)):
        assert segment_at(master, "NKLA", day)["security_id"] == nkla["security_id"], day
    assert nkla["cik"] == vtiq["cik"] == 1731289
    assert not any("NKLA" in r["symbols"] or "VTIQ" in r["symbols"] for r in conflicts.to_pylist())


def test_f9_moved_cut_keeps_the_records_availability(
    f9_built: tuple[pa.Table, pa.Table],
) -> None:
    # Leakage. The segment starts 06-04 but the rename was filed 06-08: a reader on 06-05
    # must not see NKLA at all (it sees VTIQ open-ended, as test_bk_segment_end_... explains),
    # a reader at midnight 06-08 sees the whole segment. ``available_at`` is therefore later
    # than midnight of valid_from here; assert_nothing_known_before_its_valid_from is only a
    # lower bound and still holds, the sharp assertion is the record's own midnight.
    master, _ = f9_built

    [nkla] = segments(master, "NKLA")
    assert nkla["valid_from"] == F9_NKLA_FIRST_BAR
    assert nkla["available_at"] == et_midnight(F9_ALPACA_DAY)
    assert nkla["available_at"] > et_midnight(nkla["valid_from"])

    day_after = visible_at(master, et_midnight(F9_NKLA_FIRST_BAR + timedelta(days=1)))
    assert "NKLA" not in {r["symbol"] for r in day_after}
    assert [r["valid_to"] for r in day_after if r["symbol"] == "VTIQ"] == [F9_NKLA_FIRST_BAR]
    on_record_day = [
        r for r in visible_at(master, et_midnight(F9_ALPACA_DAY)) if r["symbol"] == "NKLA"
    ]
    assert [r["valid_from"] for r in on_record_day] == [F9_NKLA_FIRST_BAR]
    assert_nothing_known_before_its_valid_from(master)


def test_f9_gap_longer_than_max_gap_days_keeps_the_alpaca_date(
    f9_built: tuple[pa.Table, pa.Table],
) -> None:
    # F9 (2), guard (passes today, must keep passing). ASSUMPTION: a halt of more than
    # MAX_GAP_DAYS between the old name's last bar and Alpaca's date is not a late record but
    # something else (a long suspension); HLTN's 05-04..06-05 bars stay uncovered on purpose
    # and are not asserted either way. Needs a real-data check: the lag between the old
    # name's last bar and Alpaca's date over all name_change rows should be a few days, with
    # no mass between 10 days and months.
    master, _ = f9_built

    assert sm.MAX_GAP_DAYS == 10
    [hlto] = segments(master, "HLTO")
    [hltn] = segments(master, "HLTN")
    assert hltn["valid_from"] == F9_HALT_DAY
    assert hlto["valid_to"] == F9_HALT_DAY
    assert hltn["available_at"] == et_midnight(F9_HALT_DAY)
    assert hlto["security_id"] == hltn["security_id"]


def test_f9_old_symbol_without_bars_keeps_the_alpaca_date(
    f9_built: tuple[pa.Table, pa.Table],
) -> None:
    # F9 (3), guard. CBS -> VIAC: the old symbol has no bars in the store at all (real), so
    # there is no last bar to refine from; VIAC's segment starts on the Alpaca date. Covering
    # the pre-rename history of such labels is the universe re-pull's job, not this rule's.
    master, _ = f9_built

    [viac] = segments(master, "VIAC")
    assert viac["valid_from"] == date(2019, 12, 5)
    assert viac["valid_to"] is None
    assert viac["cik"] == 813828
    assert "CBS" not in set(master.column("symbol").to_pylist())


def test_f9_relabelled_history_under_the_new_label_does_not_move_the_cut(
    f9_built: tuple[pa.Table, pa.Table],
) -> None:
    # F9 (4), guard. RLBN has bars before 06-09, but they are RLBO's own bars relabelled, and
    # RLBO itself trades through 06-08 = date - 1: last_bar + 1 == date, nothing to move. A
    # rule that looked only at the new label's bars would pull the cut back to 2020-01-02.
    master, conflicts = f9_built

    [rlbo] = segments(master, "RLBO")
    [rlbn] = segments(master, "RLBN")
    assert rlbn["valid_from"] == rlbo["valid_to"] == F9_RELABEL_DAY
    assert rlbo["security_id"] == rlbn["security_id"]
    assert not any("RLBN" in r["symbols"] for r in conflicts_of(conflicts, "dup_unlinked"))


def test_f9_sec_only_refinement_is_unchanged(f9_built: tuple[pa.Table, pa.Table]) -> None:
    # Guard (passes today, must keep passing): the SEC-only rule of _refine_sec_edges has no
    # gap limit -- SOLD's last bar 03-13 is four months before the 07-10 snapshot and the cut
    # is still 03-14 (same rule as test_rename_known_only_to_sec_cuts_the_day_after_the_old_
    # names_last_bar in the shared world). F9's MAX_GAP_DAYS applies to Alpaca-dated edges.
    master, _ = f9_built

    [sold] = segments(master, "SOLD")
    [snew] = segments(master, "SNEW")
    assert sold["valid_to"] == snew["valid_from"] == date(2020, 3, 14)
    assert snew["evidence"] == "sec"
    assert snew["available_at"] == et_midnight(date(2020, 7, 10))
    assert sold["security_id"] == snew["security_id"]


# --- F10: a CIK switch next to a rename into the label --------------------------------------
# World shaped like the real file. NOTE on the SEC rows: the measured failure (INBX ends at
# 2024-06-06 with no later piece) needs an SEC edge INBX -> INBXV, i.e. the SEC listed INBX
# under 1739614 up to 05-02 (real: Inhibrx Inc traded as INBX for years) and never listed
# INXB (a temporary ticker Alpaca knows, the SEC does not). With INXB listed under 1739614
# instead of INBX, CIK 1739614 would lose two tickers at once, no edge would exist and the
# build already gives INBX one open-ended segment -- that shape reproduces nothing.

F10_SNAPSHOTS = [
    date(2024, 4, 1),
    date(2024, 5, 2),  # real: last snapshot with INBX under 1739614
    date(2024, 6, 6),  # real: first snapshot with INBX under 2007919 and INBXV under 1739614
    date(2024, 7, 3),
    date(2025, 1, 2),
]
F10_RENAME_DAY = date(2024, 5, 31)  # real: Alpaca INXB -> INBX
F10_SWITCH_SNAPSHOT = date(2024, 6, 6)
F10_OLD_CIK, F10_NEW_CIK = 1739614, 2007919  # real
F10_LIVES: list[Life] = [
    ("INBX", F10_OLD_CIK, date(2024, 4, 1), date(2024, 5, 2)),
    ("INBX", F10_NEW_CIK, F10_SWITCH_SNAPSHOT, None),
    ("INBXV", F10_OLD_CIK, F10_SWITCH_SNAPSHOT, None),
]
F10_RENAMES: list[Rename] = [("INXB", "INBX", F10_RENAME_DAY, "45719W109", "45719W208")]  # CUSIPs
#   not checked


def f10_bars() -> list[dict[str, Any]]:
    return [
        # INXB's bars stop the day before the rename, INBX's start on it, at one price level
        # (real spans; levels ASSUMED). INBXV: two when-issued bars (ASSUMPTION; whatever they
        # are, INBXV's own segment is not under test here).
        *bar_rows("INXB", date(2020, 1, 2), date(2024, 5, 30), 30.0),
        *bar_rows("INBX", F10_RENAME_DAY, date(2025, 12, 31), 30.5),
        *bar_rows("INBXV", date(2024, 5, 28), date(2024, 5, 29), 2.0),
    ]


@pytest.fixture(scope="module")
def f10_built() -> tuple[pa.Table, pa.Table]:
    return build_security_master(
        bars=bars_table(f10_bars()),
        sec_history=sec_table(F10_SNAPSHOTS, F10_LIVES),
        name_changes=name_changes_table(F10_RENAMES),
    )


def test_f10_cik_switch_not_explained_by_the_rename_stays_an_event(
    f10_built: tuple[pa.Table, pa.Table],
) -> None:
    # F10 (1). INXB was never SEC-listed, so it was not listed under 2007919 (INBX's CIK at
    # the switch's upper snapshot): the rename does not explain the switch. Today the switch
    # is swallowed, the SEC edge INBX -> INBXV ends INBX's piece at 06-06 and the bars from
    # 06-06 to 2025-12-31 (real: to 2026-10) belong to no security.
    master, conflicts = f10_built

    for day in (F10_SWITCH_SNAPSHOT, date(2024, 8, 1), date(2025, 6, 2), date(2025, 12, 31)):
        seg = segment_at(master, "INBX", day)
        assert seg["cik"] == F10_NEW_CIK, day
        assert seg["valid_from"] == F10_SWITCH_SNAPSHOT, day
    # The first INBX piece is the renamed INXB.
    first = segment_at(master, "INBX", date(2024, 6, 3))
    assert (first["valid_from"], first["valid_to"]) == (F10_RENAME_DAY, F10_SWITCH_SNAPSHOT)
    assert first["security_id"] == only_sid(master, "INXB")
    # The 06-06 piece joins the first one. With F11 the SEC vacate INBX -> INBXV is dated at
    # the takeover (05-31), so it ends the OLD filer's label, not INXB's piece: the piece before
    # the 06-06 cut is INXB's own (started by a rename, not ended by one), F4 does not apply,
    # and the cut is the SEC catching up with the new filer's CIK -- bridged on continuous
    # prices. Real world: Inhibrx Biosciences is one security from 05-31 on; the SEC listed
    # its CIK a week later. Rewritten 2026-10-07: an earlier version demanded a split plus a
    # conflict here, which predates F11 and described the wrong outcome.
    later = segment_at(master, "INBX", date(2024, 8, 1))
    assert later["security_id"] == first["security_id"], (first, later)
    assert not any("INBX" in r["symbols"] for r in conflicts.to_pylist()), conflicts.to_pylist()


def test_f10_new_holders_piece_is_not_knowable_before_the_switch_snapshot(
    f10_built: tuple[pa.Table, pa.Table],
) -> None:
    # Leakage: the piece from 06-06 exists because of the 06-06 snapshot; a reader on 06-01
    # sees INBX's renamed piece and nothing starting 06-06. Nothing here is a late record
    # (INXB's last bar is the day before the rename), so the general guard applies in full.
    master, _ = f10_built

    later = segment_at(master, "INBX", date(2024, 8, 1))
    assert later["available_at"] == et_midnight(F10_SWITCH_SNAPSHOT)
    seen = [r for r in visible_at(master, et_midnight(date(2024, 6, 1))) if r["symbol"] == "INBX"]
    assert [r["valid_from"] for r in seen] == [F10_RENAME_DAY]
    assert_nothing_known_before_its_valid_from(master)


def test_f10_switch_explained_by_the_rename_is_still_suppressed(master: pa.Table) -> None:
    # F10 (2), guard on the shared world (passes today, must keep passing). BYON -> BBBY
    # 2025-08-29 with BYON listed under 1130713, and BBBY switching 886158 -> 1130713 in the
    # pair (2023-04-03, 2025-09-02]: BYON was listed under the CIK BBBY carries at 2025-09-02,
    # so the rename explains the switch and the 09-02 snapshot creates no piece. The chain
    # itself is test_reused_ticker_bbby_is_two_securities.
    beyond = [
        s
        for s in segments(master, "BBBY")
        if s["cik"] == 1130713 and s["valid_from"] < date(2026, 8, 17)
    ]
    assert len(beyond) == 1
    assert beyond[0]["valid_from"] == date(2025, 8, 29)
    assert not any(s["valid_from"] in set(SNAPSHOTS) for s in segments(master, "BBBY"))
    assert segment_at(master, "BBBY", date(2025, 9, 2))["valid_from"] == date(2025, 8, 29)


# =============================================================================================
# F9b/F11: zero-volume filler bars, SEC vacate dated by Alpaca's takeover (measured on the real
# store, 2026-10-07, after the F9/F10 build)
# =============================================================================================
#
# Rules these tests pin (implementation pending):
#   F9b After a rename Alpaca keeps writing bars under the OLD label with volume 0 and
#       open = high = low = close frozen at the last real close. Real: VTIQ's real bars end
#       2020-06-03 (close 33.97); every VTIQ bar from 06-04 to 2022-12-09 is 33.97 x4 / 0.
#       SPAQ: real to 2020-10-29 (8.96), filler to 2022-05-02. The relabelled history under the
#       new label (NKLA from 2020-06-01, FSR) copies the old label's REAL bars only. Because of
#       the filler "the old name's last bar" is years after the rename, so F9 never fires on the
#       real file (NKLA still starts 06-08, FSR 11-09). Rule: the old name's last bar means its
#       last bar WITH VOLUME > 0, in both _refine_alpaca_edges and _refine_sec_edges.
#   F11 Mirror of F2 in merge_edges. An SEC-only edge old -> new whose OLD label has an Alpaca
#       start edge INTO it (X -> old) dated at or before the SEC edge's date and within
#       SEC_LAG_MAX_DAYS is the old filer leaving the label at the moment the new filer took it:
#       the SEC edge takes that Alpaca date (evidence stays 'sec', date_lower None). Real:
#       Inhibrx. SEC: INBX under 1739614 (Inhibrx Inc) through 2024-05-02; on 06-06 INBX and
#       INXB under 2007919 (Inhibrx Biosciences; INXB its temporary ticker) and INBXV under
#       1739614. Alpaca: INXB -> INBX 2024-05-31. Today the SEC-only INBX -> INBXV found in
#       (05-02, 06-06] is dated at the snapshot (INBX trades through the interval), ends INBX's
#       piece at 06-06 and nothing follows; the CIK switch is explained by INXB -> INBX (INXB
#       sits under 2007919) so F3/F10 create nothing.


def covering(master: pa.Table, symbol: str, day: date) -> list[dict[str, Any]]:
    """Segments of ``symbol`` that contain ``day`` (may be empty, unlike segment_at)."""
    return [
        r
        for r in segments(master, symbol)
        if r["valid_from"] <= day and (r["valid_to"] is None or day < r["valid_to"])
    ]


def filler_rows(last_real: dict[str, Any], start: date, end: date) -> list[dict[str, Any]]:
    """Alpaca's zero-volume filler under a vacated label: one bar per business day with
    open = high = low = close frozen at the label's last real close and volume 0.

    Real (store, 2026-10-07): VTIQ's last real bar is 2020-06-03, close 33.97; 06-04, 06-05,
    06-08, 06-09, ... through 2022-12-09 are all (33.97, 33.97, 33.97, 33.97, 0). SPAQ: 8.96
    from 2020-10-30 to 2022-05-02. ``available_at`` follows the bar's own day (ASSUMPTION: the
    filler is written like any other bar; irrelevant to the rule, which only reads volume).
    """
    close = last_real["close"]
    return [
        {
            "symbol": last_real["symbol"],
            "session_date": day,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 0,
            "available_at": bar_available_at(day),
        }
        for day in business_days(start, end)
    ]


# --- F9b: filler bars must not count as the old name's last bar ------------------------------

F9B_LAST_REAL = date(2020, 6, 3)  # real: VTIQ's last bar with volume
F9B_FILLER_END = date(2020, 12, 31)  # real: 2022-12-09; shortened
F9B_THIN_DAY = date(2020, 2, 6)  # real: VTIQ traded 0 shares that day (close 10.36), label alive
F9B_LIVES: list[Life] = [
    ("VTIQ", 1731289, date(2020, 1, 13), date(2020, 1, 13)),  # real CIK on both labels
    ("NKLA", 1731289, date(2020, 7, 10), None),
]
F9B_RENAMES: list[Rename] = [("VTIQ", "NKLA", F9_ALPACA_DAY, "92243N108", "654110105")]  # CUSIPs
#   not checked


def f9b_bars() -> list[dict[str, Any]]:
    # Real shape (levels assumed: VTIQ 28.7 on 05-29, NKLA 33.75 on 06-04 in the store, one
    # drifting series here): VTIQ real to 06-03, filler from 06-04; NKLA carries VTIQ's REAL
    # 06-01..06-03 bars relabelled (real: identical OHLCV on those three days, none of the
    # filler) and its own bars from 06-04. Alpaca's record is dated 06-08.
    vtiq = bar_rows("VTIQ", date(2020, 1, 2), F9B_LAST_REAL, 28.0)
    [thin] = [r for r in vtiq if r["session_date"] == F9B_THIN_DAY]
    thin["volume"] = 0  # a zero-volume day INSIDE the run is not filler (real, see above)
    relabelled = relabel([r for r in vtiq if r["session_date"] >= date(2020, 6, 1)], "NKLA")
    return [
        *vtiq,
        *filler_rows(vtiq[-1], F9B_LAST_REAL + timedelta(days=1), F9B_FILLER_END),
        *relabelled,
        *bar_rows("NKLA", F9_NKLA_FIRST_BAR, F9B_FILLER_END, 28.3),
    ]


@pytest.fixture(scope="module")
def f9b_built() -> tuple[pa.Table, pa.Table]:
    return build_security_master(
        bars=bars_table(f9b_bars()),
        sec_history=sec_table(F9_SNAPSHOTS, F9B_LIVES),
        name_changes=name_changes_table(F9B_RENAMES),
    )


def test_f9b_filler_bars_do_not_hide_the_old_names_last_traded_bar(
    f9b_built: tuple[pa.Table, pa.Table],
) -> None:
    # F9b (1), the real NKLA shape. Today VTIQ's "last bar before 06-08" is the 06-05 filler,
    # the weekend leaves NKLA no bar in (06-05, 06-08) and the cut stays on 06-08: the
    # 06-04 / 06-05 NKLA bars resolve to nothing, exactly what the real build shows. The last
    # bar WITH VOLUME is 06-03, so the cut is 06-04. The zero-volume day inside VTIQ's run
    # (02-06) is before the last traded bar and changes nothing.
    master, conflicts = f9b_built

    [vtiq] = segments(master, "VTIQ")
    [nkla] = segments(master, "NKLA")
    assert vtiq["valid_to"] == nkla["valid_from"] == F9_NKLA_FIRST_BAR
    assert vtiq["valid_from"] == date(2020, 1, 2)
    assert covering(master, "VTIQ", F9B_THIN_DAY) == [vtiq]
    assert nkla["valid_to"] is None
    assert vtiq["security_id"] == nkla["security_id"]
    assert len(sids(master, "NKLA")) == 1
    for day in (F9_NKLA_FIRST_BAR, date(2020, 6, 5), F9_ALPACA_DAY, date(2020, 9, 1)):
        assert segment_at(master, "NKLA", day)["security_id"] == nkla["security_id"], day
    assert nkla["cik"] == vtiq["cik"] == 1731289
    assert not any("NKLA" in r["symbols"] or "VTIQ" in r["symbols"] for r in conflicts.to_pylist())


def test_f9b_filler_bars_and_relabelled_days_fall_in_no_segment(
    f9b_built: tuple[pa.Table, pa.Table],
) -> None:
    # F9b (2), boundary on both sides of the cut. VTIQ's filler (06-04 is the first filler
    # day; 09-01 is deep inside it) belongs to no segment -- the label was vacated, those bars
    # are not trades. NKLA's 06-01..06-03 bars are VTIQ's real bars relabelled and belong to no
    # NKLA segment (same rule as test_relabelled_history_under_the_new_name_gets_no_segment).
    # Today VTIQ's segment runs to 06-08 and swallows the 06-04 / 06-05 filler.
    master, _ = f9b_built

    assert covering(master, "VTIQ", F9B_LAST_REAL) != []
    assert covering(master, "VTIQ", F9_NKLA_FIRST_BAR) == []
    assert covering(master, "VTIQ", date(2020, 9, 1)) == []
    assert covering(master, "VTIQ", F9B_FILLER_END) == []
    assert covering(master, "NKLA", date(2020, 6, 2)) == []
    assert covering(master, "NKLA", F9_NKLA_FIRST_BAR) != []


def test_f9b_moved_cut_keeps_the_records_availability_despite_filler(
    f9b_built: tuple[pa.Table, pa.Table],
) -> None:
    # Leakage, as test_f9_moved_cut_keeps_the_records_availability: the segment starts 06-04
    # but Alpaca filed the rename on 06-08. A reader at midnight 06-05 sees no NKLA row and a
    # VTIQ row whose valid_to it must treat as open; a reader at midnight 06-08 sees the whole
    # NKLA segment. The filler bars (available daily from 06-04 on) must not make the rename
    # knowable earlier: their presence says nothing a reader may act on.
    master, _ = f9b_built

    [nkla] = segments(master, "NKLA")
    assert nkla["valid_from"] == F9_NKLA_FIRST_BAR
    assert nkla["available_at"] == et_midnight(F9_ALPACA_DAY)
    assert nkla["available_at"] > et_midnight(nkla["valid_from"])

    day_after = visible_at(master, et_midnight(F9_NKLA_FIRST_BAR + timedelta(days=1)))
    assert "NKLA" not in {r["symbol"] for r in day_after}
    assert [r["valid_to"] for r in day_after if r["symbol"] == "VTIQ"] == [F9_NKLA_FIRST_BAR]
    on_record_day = [
        r for r in visible_at(master, et_midnight(F9_ALPACA_DAY)) if r["symbol"] == "NKLA"
    ]
    assert [r["valid_from"] for r in on_record_day] == [F9_NKLA_FIRST_BAR]
    assert_nothing_known_before_its_valid_from(master)


# SEC-only variant: the filler must not drag _refine_sec_edges to the snapshot date either.
# Constructed: the filler was measured only under labels Alpaca has a record for (VTIQ, SPAQ);
# that it also appears under labels Alpaca missed is an ASSUMPTION, checked on real data by the
# uncovered-bars audit (traded bars in no segment).
F9B_SEC_LAST_REAL = date(2020, 3, 13)  # a Friday; FNEW's first bar is Monday 03-16
F9B_SEC_CUT = date(2020, 3, 14)  # last traded bar + 1, the rule of _refine_sec_edges (SOLD/SNEW
#   in F9 and GMGI/MRDN in the shared world cut there too); the weekend is not skipped
F9B_SEC_LIVES: list[Life] = [
    ("FOLD", 9004, date(2020, 1, 13), date(2020, 1, 13)),
    ("FNEW", 9004, date(2020, 7, 10), None),
]


def f9b_sec_bars() -> list[dict[str, Any]]:
    fold = bar_rows("FOLD", date(2020, 1, 2), F9B_SEC_LAST_REAL, 50.0)
    return [
        *fold,
        *filler_rows(fold[-1], date(2020, 3, 16), F9B_FILLER_END),
        *bar_rows("FNEW", date(2020, 3, 16), F9B_FILLER_END, 50.1),
    ]


def test_f9b_sec_only_cut_ignores_filler_bars() -> None:
    # F9b (3). Today FOLD's last bar inside (01-13, 07-10] is the 07-10 filler, so the SEC edge
    # keeps the snapshot date, FOLD's segment covers four months of filler and FNEW's bars from
    # 03-16 to 07-09 belong to nothing. With volume > 0 the last bar is 03-13 and the cut 03-14.
    master, conflicts = build_security_master(
        bars=bars_table(f9b_sec_bars()),
        sec_history=sec_table(F9_SNAPSHOTS, F9B_SEC_LIVES),
        name_changes=name_changes_table([]),
    )

    [fold] = segments(master, "FOLD")
    [fnew] = segments(master, "FNEW")
    assert fold["valid_to"] == fnew["valid_from"] == F9B_SEC_CUT
    assert fnew["valid_to"] is None
    assert fold["security_id"] == fnew["security_id"]
    assert fnew["evidence"] == "sec"
    assert fnew["available_at"] == et_midnight(date(2020, 7, 10))  # the snapshot, not the cut
    assert segment_at(master, "FNEW", date(2020, 3, 16))["valid_from"] == F9B_SEC_CUT
    assert covering(master, "FOLD", date(2020, 3, 16)) == []
    assert covering(master, "FOLD", date(2020, 9, 1)) == []
    assert not any("FOLD" in r["symbols"] or "FNEW" in r["symbols"] for r in conflicts.to_pylist())
    assert_nothing_known_before_its_valid_from(master)


# --- F11: an SEC vacate dated by Alpaca's takeover of the same label -------------------------

F11_SNAPSHOTS = [
    date(2024, 4, 1),
    date(2024, 5, 2),  # real: last snapshot with INBX under 1739614
    date(2024, 6, 6),  # real: INBX + INXB under 2007919, INBXV under 1739614
    date(2024, 7, 3),
    date(2025, 1, 2),
]
F11_RENAME_DAY = date(2024, 5, 31)  # real: Alpaca INXB -> INBX
F11_SEC_DATE = date(2024, 6, 6)  # the SEC edge INBX -> INBXV before re-dating
F11_SEC_LOWER = date(2024, 5, 2)
F11_OLD_CIK, F11_NEW_CIK = 1739614, 2007919  # real
F11_LIVES: list[Life] = [
    ("INBX", F11_OLD_CIK, date(2024, 4, 1), F11_SEC_LOWER),
    ("INBX", F11_NEW_CIK, F11_SEC_DATE, None),
    ("INXB", F11_NEW_CIK, F11_SEC_DATE, F11_SEC_DATE),  # real: listed in that one snapshot
    ("INBXV", F11_OLD_CIK, F11_SEC_DATE, None),
]
# Omitted on purpose: the real file also lists INXB under an unrelated stale CIK 1518205 in
# 2019-10..2020-06; this world starts in 2024 and those rows take part in no edge (a CIK that
# simply disappears) -- left out to keep the world about the one event.
F11_RENAMES: list[Rename] = [("INXB", "INBX", F11_RENAME_DAY, "45719W109", "45719W208")]  # CUSIPs
#   not checked


def f11_sec_edges(lives: Sequence[Life] = F11_LIVES[:1] + F11_LIVES[3:]) -> pa.Table:
    """The one SEC edge of this world, INBX -> INBXV in (05-02, 06-06]."""
    return edges_from_sec(normalize_spelling(sec_table(F11_SNAPSHOTS[:3], lives)))


def f11_bars() -> list[dict[str, Any]]:
    # Real spans and shape; levels follow the store's first days (INBX 05-29 13.92, 05-30
    # 15.86, 05-31 16.25, 06-03 18.02 -- one continuous series across 05-31; the drift after
    # that is the builder's). INXB has exactly one bar, 05-30, identical to INBX's 05-30 bar
    # (real: Alpaca filed the when-issued day under both labels). INBXV: 05-28 31.25, 05-29
    # 30.85 (real: the old filer's shares in their last two days).
    inbx = bar_rows("INBX", date(2024, 5, 29), date(2025, 12, 31), 13.9)
    [when_issued] = [r for r in inbx if r["session_date"] == date(2024, 5, 30)]
    return [
        *inbx,
        *relabel([when_issued], "INXB"),
        *bar_rows("INBXV", date(2024, 5, 28), date(2024, 5, 29), 31.0),
    ]


@pytest.fixture(scope="module")
def f11_built() -> tuple[pa.Table, pa.Table]:
    return build_security_master(
        bars=bars_table(f11_bars()),
        sec_history=sec_table(F11_SNAPSHOTS, F11_LIVES),
        name_changes=name_changes_table(F11_RENAMES),
    )


def test_f11_sec_vacate_takes_the_date_alpaca_gave_the_old_label() -> None:
    # F11 on merge_edges alone. SEC INBX -> INBXV in (05-02, 06-06]; Alpaca INXB -> INBX at
    # 05-31, i.e. somebody took INBX inside the interval: the SEC edge moves to 05-31, keeps
    # evidence 'sec', loses its date_lower (the date is exact now). The Alpaca edge is untouched.
    sec = f11_sec_edges()
    assert [
        (e["old_symbol"], e["new_symbol"], e["date"], e["date_lower"]) for e in sec.to_pylist()
    ] == [("INBX", "INBXV", F11_SEC_DATE, F11_SEC_LOWER)]
    merged = merge_edges(edges_from_name_changes(name_changes_table(F11_RENAMES)), sec).to_pylist()
    by_pair = {(e["old_symbol"], e["new_symbol"]): e for e in merged}

    assert set(by_pair) == {("INBX", "INBXV"), ("INXB", "INBX")}
    assert by_pair[("INBX", "INBXV")]["date"] == F11_RENAME_DAY
    assert by_pair[("INBX", "INBXV")]["evidence"] == "sec"
    assert by_pair[("INBX", "INBXV")]["date_lower"] is None
    assert by_pair[("INBX", "INBXV")]["available_at"] == et_midnight(F11_SEC_DATE)
    assert by_pair[("INXB", "INBX")]["date"] == F11_RENAME_DAY
    assert by_pair[("INXB", "INBX")]["evidence"] == "name_change"


@pytest.mark.parametrize(
    "alpaca_day",
    [
        F11_SEC_DATE + timedelta(days=1),  # a takeover AFTER the snapshot: a later event
        F11_SEC_DATE - timedelta(days=sm.SEC_LAG_MAX_DAYS + 1),  # too long before it
    ],
)
def test_f11_takeover_after_the_sec_date_or_beyond_the_lag_leaves_the_sec_edge_alone(
    alpaca_day: date,
) -> None:
    # F11 boundary (constructed, mirrors test_f2_alpaca_edge_after_...). The rule needs the
    # Alpaca start into the OLD label dated <= the SEC date and within SEC_LAG_MAX_DAYS of it;
    # otherwise the SEC edge keeps its snapshot date and interval.
    assert sm.SEC_LAG_MAX_DAYS == 183
    alpaca = name_changes_table([("INXB", "INBX", alpaca_day, "45719W109", "45719W208")])
    merged = merge_edges(edges_from_name_changes(alpaca), f11_sec_edges()).to_pylist()
    by_pair = {(e["old_symbol"], e["new_symbol"]): e for e in merged}

    assert by_pair[("INBX", "INBXV")]["date"] == F11_SEC_DATE
    assert by_pair[("INBX", "INBXV")]["date_lower"] == F11_SEC_LOWER
    assert by_pair[("INBX", "INBXV")]["evidence"] == "sec"
    assert by_pair[("INXB", "INBX")]["date"] == alpaca_day


def test_f11_inbx_new_filer_segment_runs_from_alpacas_date(
    f11_built: tuple[pa.Table, pa.Table],
) -> None:
    # F11 end to end. Today: INBX [05-29, 05-31) + [05-31, 06-06) and nothing after, though
    # the label trades to 2026-10 (real). With the vacate re-dated to 05-31 both edges cut on
    # one day: the old filer's piece ends at 05-31, the new filer's starts there, open-ended.
    master, conflicts = f11_built

    assert len(segments(master, "INBX")) == 2
    first, later = segments(master, "INBX")
    assert (first["valid_from"], first["valid_to"]) == (date(2024, 5, 29), F11_RENAME_DAY)
    assert first["cik"] == F11_OLD_CIK  # what the SEC rows before 05-31 say about the label
    assert (later["valid_from"], later["valid_to"]) == (F11_RENAME_DAY, None)
    assert later["cik"] == F11_NEW_CIK
    assert later["evidence"] == "name_change"
    for day in (
        F11_RENAME_DAY,
        F11_SEC_DATE,
        date(2024, 8, 1),
        date(2025, 6, 2),
        date(2025, 12, 31),
    ):
        assert segment_at(master, "INBX", day) == later, day
    # Old filer vs new filer: two securities. INXB (the temporary ticker) belongs to the new one.
    assert first["security_id"] != later["security_id"]
    assert only_sid(master, "INXB") == later["security_id"]
    assert not any(s["valid_from"] == F11_SEC_DATE for s in segments(master, "INBX"))
    # No CIK-cut event is left (the switch is explained by INXB -> INBX), so no conflict about
    # INBX is a CIK-switch report and none is dated at the 06-06 snapshot. The takeover check
    # may still report the 05-31 cut as reuse_looks_continuous (real: 15.86 -> 16.25 across
    # CIKs 1739614 -> 2007919) -- that report is a correct question for a human (the 05-29/30
    # INBX bars are INXB's when-issued days), not a leftover of the snapshot-dated vacate.
    assert conflicts_of(conflicts, "cik_switch_discontinuous") == []
    for r in conflicts.to_pylist():
        if "INBX" in r["symbols"]:
            assert r["kind"] == "reuse_looks_continuous", r
            assert F11_SEC_DATE not in r["dates"], r
            assert F11_RENAME_DAY in r["dates"], r
    assert conflicts_of(conflicts, "dup_unlinked") == []  # INXB's bar sits in the same chain


def test_f11_new_filer_segment_is_knowable_at_alpacas_record_not_the_snapshot(
    f11_built: tuple[pa.Table, pa.Table],
) -> None:
    # Leakage. The open-ended INBX segment exists because of Alpaca's 05-31 record, so it is
    # knowable at midnight 05-31 -- not earlier (05-30: only the pre-rename row, whose end the
    # reader must treat as open) and not later (the 06-06 snapshot adds no event). On 05-15
    # INBX has no bar yet and no row is visible at all.
    master, _ = f11_built

    [later] = [s for s in segments(master, "INBX") if s["valid_from"] == F11_RENAME_DAY]
    assert later["valid_to"] is None
    assert later["available_at"] == et_midnight(F11_RENAME_DAY)
    assert later["available_at"] < et_midnight(F11_SEC_DATE)

    assert [
        r for r in visible_at(master, et_midnight(date(2024, 5, 15))) if r["symbol"] == "INBX"
    ] == []
    before_record = [
        r for r in visible_at(master, et_midnight(date(2024, 5, 30))) if r["symbol"] == "INBX"
    ]
    assert [(r["valid_from"], r["valid_to"]) for r in before_record] == [
        (date(2024, 5, 29), F11_RENAME_DAY)
    ]
    at_record = [
        r for r in visible_at(master, et_midnight(F11_RENAME_DAY)) if r["symbol"] == "INBX"
    ]
    assert sorted(r["valid_from"] for r in at_record) == [date(2024, 5, 29), F11_RENAME_DAY]
    assert_nothing_known_before_its_valid_from(master)


# =============================================================================================
# G: reviewer fixes (2026-10-07)
# =============================================================================================
#
# Four defects found by review of the F9b/F11 build, each pinned by a world of its own:
#   G1 merge_edges: a duplicate Alpaca row for an edge the SEC also shows must keep the earliest
#      date (F6 covered the Alpaca-only shape; the "both" branch overwrote unconditionally, so
#      BIGT -> MAGS filed 2023-11-09 and 2025-02-03 came out as 2025-02-03).
#   G2 _closes_around: continuity across a CIK cut is judged on traded bars (volume > 0) only.
#      Real: VTIQ's volume-0 filler at 33.97 from 2020-06-04 to 2022-12-09; 69 of 345 bridged
#      CIK cuts on the real store had a zero-volume bar on one side of the cut.
#   G3 _relabelled_pieces: on a same-day round trip ORIC -> 1ORIC -> ORIC (real: 2025-02-19)
#      the label's own pre-trip history must not be dropped as "relabelled" because the
#      temporary ticker carries a copy of it.
#   G4 (test_corporate_actions.py) download_all refuses to overwrite a non-empty raw year
#      file with an empty response.


# --- G1: duplicate Alpaca rows for an edge the SEC also shows ------------------------------------

G1_SNAPSHOTS = [date(2023, 6, 1), date(2023, 12, 1), date(2024, 6, 3)]
G1_LIVES: list[Life] = [
    # Constructed CIK: the real MAGS (Roundhill Magnificent Seven ETF) is a fund and never
    # reaches the company table; the rule is about the merge, not about who files.
    ("BIGT", 1900001, date(2023, 6, 1), date(2023, 6, 1)),
    ("MAGS", 1900001, date(2023, 12, 1), None),
]


@pytest.mark.parametrize(
    "rows",
    [F6_RENAMES, list(reversed(F6_RENAMES))],
    ids=["earliest_row_first", "latest_row_first"],
)
def test_g1_duplicate_alpaca_rows_with_an_sec_edge_keep_the_earliest_date(
    rows: list[Rename],
) -> None:
    # G1. The SEC pair (06-01, 12-01] shows BIGT -> MAGS as well; Alpaca files it twice (real
    # dates). Old code: every Alpaca row matching an SEC edge overwrote the merged row, so the
    # LAST row's date won (2025-02-03, after MAGS's last bar, and MAGS got no segment). The
    # second parameter feeds the rows in the other order: the raw store is read year by year
    # (2023 before 2025), but the rule must not depend on that.
    merged = merge_edges(
        edges_from_name_changes(name_changes_table(rows)),
        edges_from_sec(normalize_spelling(sec_table(G1_SNAPSHOTS, G1_LIVES))),
    ).to_pylist()

    assert [(e["old_symbol"], e["new_symbol"]) for e in merged] == [("BIGT", "MAGS")]
    [edge] = merged
    assert edge["date"] == F6_FIRST
    assert edge["evidence"] == "both"
    assert edge["available_at"] == et_midnight(F6_FIRST)


def test_g1_end_to_end_mags_segment_starts_at_the_first_record_with_sec_backing() -> None:
    # G1 end to end: same world as F6 plus the SEC rows. MAGS's own bars (11-09..2024-03-28)
    # must resolve; the relabelled BIGT history under MAGS must not.
    bigt = bar_rows("BIGT", date(2023, 4, 11), date(2023, 11, 8), 5.0)
    master, conflicts = build_security_master(
        bars=bars_table(
            bigt + relabel(bigt, "MAGS") + bar_rows("MAGS", F6_FIRST, date(2024, 3, 28), 5.1)
        ),
        sec_history=sec_table(G1_SNAPSHOTS, G1_LIVES),
        name_changes=name_changes_table(F6_RENAMES),
    )

    [mags] = segments(master, "MAGS")
    assert mags["valid_from"] == F6_FIRST
    assert mags["evidence"] == "both"
    assert mags["cik"] == 1900001
    assert covering(master, "MAGS", date(2024, 1, 2)) == [mags]
    assert covering(master, "MAGS", date(2023, 6, 1)) == []
    [bigt_seg] = segments(master, "BIGT")
    assert (bigt_seg["valid_from"], bigt_seg["valid_to"]) == (date(2023, 4, 11), F6_FIRST)
    assert bigt_seg["security_id"] == mags["security_id"]
    assert not any(
        set(r["symbols"]) == {"BIGT", "MAGS"} for r in conflicts_of(conflicts, "dup_unlinked")
    )
    # Leakage: the segment is knowable at the first record's midnight, not the duplicate's.
    assert mags["available_at"] == et_midnight(F6_FIRST)
    assert "MAGS" in {r["symbol"] for r in visible_at(master, et_midnight(date(2024, 1, 2)))}
    assert_nothing_known_before_its_valid_from(master)


# --- G2: a CIK cut is judged on traded bars; filler bars do not bridge it -----------------------
# Shape: label L, held by CIK A, stops trading; Alpaca keeps writing volume-0 bars frozen at the
# last close (real mechanism: VTIQ 33.97 x 0 from 2020-06-04 to 2022-12-09); the SEC later lists
# L under CIK B, whose own bars start at a close near the frozen one. No rename record anywhere
# (ASSUMPTION: a delisting followed by a fresh listing under the same label; the real analogue
# is the 69 bridged cuts with a zero-volume bar on one side, whose true identities are unknown).

G2_OLD_CIK, G2_NEW_CIK = 5001, 5002


def g2_inputs(*, last_traded: date, cut: date, new_base: float) -> tuple[pa.Table, pa.Table]:
    """L traded 2020-01-02..``last_traded`` at ~20, filler from the next business day up to
    the day before ``cut``, the new holder's bars from ``cut`` at ``new_base``. The SEC shows
    the switch in the pair (2020-01-13, ``cut``]. ``cut`` is both the snapshot and the new
    holder's first bar: a holder that traded before the snapshot would sit left of the cut
    with real bars and the closes on both sides would be its own -- not this rule's question."""
    old = bar_rows("L", date(2020, 1, 2), last_traded, 20.0)
    filler = filler_rows(old[-1], last_traded + timedelta(days=1), cut - timedelta(days=1))
    bars = bars_table(old + filler + bar_rows("L", cut, date(2020, 12, 31), new_base))
    sec = sec_table(
        [date(2020, 1, 13), cut, date(2021, 1, 4)],
        [("L", G2_OLD_CIK, date(2020, 1, 13), date(2020, 1, 13)), ("L", G2_NEW_CIK, cut, None)],
    )
    return build_security_master(bars=bars, sec_history=sec, name_changes=name_changes_table([]))


G2_LAST_TRADED = date(2020, 3, 31)  # 64th business day: close 20 * 1.0126 = 20.252
G2_CUT = date(2020, 7, 1)
G2_NEW_BASE = 20.3  # 20.3 / 20.252 = 1.002: continuous if the filler is believed


def test_g2_filler_bars_do_not_make_a_cik_cut_look_continuous() -> None:
    # G2. Old code joined every bar: the last bar before 07-01 was the 06-30 filler at 20.252,
    # the first at/after it 20.3, ratio 1.002, one day apart -> bridged into one security. With
    # volume > 0 the last traded bar is 03-31: 92 days to 07-01 > MAX_GAP_DAYS -> not
    # continuous -> two securities and the cut is reported.
    master, conflicts = g2_inputs(last_traded=G2_LAST_TRADED, cut=G2_CUT, new_base=G2_NEW_BASE)

    assert len(sids(master, "L")) == 2
    old, new = segments(master, "L")
    assert (old["valid_from"], old["valid_to"], old["cik"]) == (
        date(2020, 1, 2),
        G2_CUT,
        G2_OLD_CIK,
    )
    assert (new["valid_from"], new["valid_to"], new["cik"]) == (G2_CUT, None, G2_NEW_CIK)
    assert old["security_id"] != new["security_id"]
    flagged = [
        r for r in conflicts_of(conflicts, "cik_switch_discontinuous") if "L" in r["symbols"]
    ]
    assert len(flagged) == 1
    assert G2_CUT in flagged[0]["dates"]
    assert G2_LAST_TRADED in flagged[0]["dates"]  # the last TRADED bar, not the last filler
    assert not any(d > G2_LAST_TRADED and d < G2_CUT for d in flagged[0]["dates"])
    assert conflicts_of(conflicts, "reuse_looks_continuous") == []


def test_g2_short_gap_between_traded_bars_is_still_bridged() -> None:
    # G2 guard: the rule only changes which bars are looked at. Old holder traded through 07-08
    # (Wednesday), one filler day 07-09, new holder from 07-10: 2 days between traded bars,
    # 20.3 / 20.536 = 0.989 -> continuous -> one security, nothing reported (GOOG-style).
    master, conflicts = g2_inputs(
        last_traded=date(2020, 7, 8), cut=date(2020, 7, 10), new_base=20.3
    )

    assert len(sids(master, "L")) == 1
    old, new = segments(master, "L")
    assert old["valid_to"] == new["valid_from"] == date(2020, 7, 10)
    assert (old["cik"], new["cik"]) == (G2_OLD_CIK, G2_NEW_CIK)
    assert not any("L" in r["symbols"] for r in conflicts.to_pylist())


def test_g2_filler_days_are_in_the_old_holders_piece_and_nothing_leaks() -> None:
    # Boundary + leakage. The filler sits inside [2020-01-02, 07-01): it resolves to the old
    # holder (the label was still that filer's as far as the SEC knew), the new holder's piece
    # exists because of the 07-01 snapshot and is knowable at its midnight, not before; a reader
    # on 2020-05-01 sees one open-ended L row of the old holder.
    master, _ = g2_inputs(last_traded=G2_LAST_TRADED, cut=G2_CUT, new_base=G2_NEW_BASE)

    old, new = segments(master, "L")
    assert covering(master, "L", date(2020, 5, 15)) == [old]
    assert covering(master, "L", G2_CUT) == [new]
    assert new["available_at"] == et_midnight(G2_CUT)
    seen = [r for r in visible_at(master, et_midnight(date(2020, 5, 1))) if r["symbol"] == "L"]
    assert [(r["valid_from"], r["cik"]) for r in seen] == [(date(2020, 1, 2), G2_OLD_CIK)]
    assert_nothing_known_before_its_valid_from(master)


# --- G3: a same-day round trip must not delete the label's own history ---------------------------
# Real: Alpaca ORIC -> 1ORIC and 1ORIC -> ORIC both dated 2025-02-19; the SEC lists ORIC under
# CIK 1796280 throughout; 1ORIC has no bars in today's store. Step ⑤ (the universe re-pull that
# adds every old name) is expected to make Alpaca serve ORIC's history under 1ORIC as well (the
# BK-under-BNY mechanism, DECISIONS 2026-10-05) -- that shape is ANTICIPATED, not yet observed,
# and is what this world builds: 1ORIC = ORIC's bars before the trip plus the trip day itself.
# Old code: ORIC's pre-trip piece was compared with 1ORIC's bars, found 100% identical and
# dropped as "relabelled", so ORIC's own 2020..2025-02 history ended up in no segment (silent).

G3_DAY = date(2025, 2, 19)  # real
G3_CIK = 1796280  # real
G3_SNAPSHOTS = [
    date(2020, 1, 13),
    date(2021, 1, 4),
    date(2022, 1, 3),
    date(2024, 6, 3),
    date(2025, 1, 2),
    date(2025, 9, 2),
]
G3_RENAMES: list[Rename] = [
    ("ORIC", "1ORIC", G3_DAY, "68622P109", "68622P109"),
    ("1ORIC", "ORIC", G3_DAY, "68622P109", "68622P109"),
]


def g3_bars() -> list[dict[str, Any]]:
    oric = bar_rows("ORIC", date(2020, 6, 1), date(2025, 12, 31), 12.0)
    copied = relabel([r for r in oric if r["session_date"] <= G3_DAY], "1ORIC")
    return [*oric, *copied]


@pytest.fixture(scope="module")
def g3_built() -> tuple[pa.Table, pa.Table]:
    return build_security_master(
        bars=bars_table(g3_bars()),
        sec_history=sec_table(G3_SNAPSHOTS, [("ORIC", G3_CIK, date(2020, 1, 13), None)]),
        name_changes=name_changes_table(G3_RENAMES),
    )


def test_g3_round_trip_keeps_the_labels_own_history(g3_built: tuple[pa.Table, pa.Table]) -> None:
    # G3. ORIC's piece before the trip is ended by ORIC -> 1ORIC, i.e. tied to the taker's old
    # name 1ORIC by an edge: it is the security's own history and must survive the relabelling
    # check; the trip is bridged (same CIK, continuous), so one security_id covers every day.
    master, conflicts = g3_built

    assert len(sids(master, "ORIC")) == 1
    assert covering(master, "ORIC", date(2021, 3, 1)) != []
    assert covering(master, "ORIC", G3_DAY - timedelta(days=1)) != []
    assert covering(master, "ORIC", G3_DAY) != []
    assert covering(master, "ORIC", date(2025, 6, 2)) != []
    assert segments(master, "ORIC")[0]["valid_from"] == date(2020, 6, 1)
    assert segments(master, "ORIC")[-1]["valid_to"] is None
    assert {s["cik"] for s in segments(master, "ORIC")} == {G3_CIK}
    assert not any(
        set(r["symbols"]) == {"ORIC", "1ORIC"} for r in conflicts_of(conflicts, "dup_unlinked")
    )
    assert conflicts_of(conflicts, "cik_switch_discontinuous") == []


def test_g3_relabelled_copy_under_the_temporary_ticker_gets_no_history_segment(
    g3_built: tuple[pa.Table, pa.Table],
) -> None:
    # G3 boundary, the other side of the same world. 1ORIC existed for one day; the 2020..2025-02
    # bars filed under it are ORIC's own bars relabelled (⑤-anticipated shape) and must resolve
    # to nothing -- otherwise visible_bars would hand out every pre-trip day twice (ORIC and
    # 1ORIC, one security). Whatever segment 1ORIC gets must start on the trip day.
    master, conflicts = g3_built

    assert covering(master, "1ORIC", date(2021, 3, 1)) == []
    assert covering(master, "1ORIC", G3_DAY - timedelta(days=1)) == []
    assert all(s["valid_from"] >= G3_DAY for s in segments(master, "1ORIC"))
    # A copy that gets a piece also gets a "takeover" of 1ORIC on the trip day and, with no SEC
    # row for 1ORIC, a reuse_looks_continuous [ORIC, 1ORIC] with "cik None -> None" -- noise a
    # human would have to dismiss. Same CIK on both labels of one security: nothing to report.
    assert not any("1ORIC" in r["symbols"] for r in conflicts.to_pylist()), conflicts.to_pylist()


def test_g3_nothing_about_the_trip_is_knowable_before_its_day(
    g3_built: tuple[pa.Table, pa.Table],
) -> None:
    # Leakage: a reader a year before the round trip sees ORIC's opening segment and nothing
    # that starts on the trip day; whatever starts on that day is knowable at its midnight.
    master, _ = g3_built

    visible = visible_at(master, et_midnight(G3_DAY) - timedelta(days=365))
    oric = [r for r in visible if r["symbol"] == "ORIC"]
    assert any(r["valid_from"] == date(2020, 6, 1) for r in oric)
    assert not any(r["valid_from"] == G3_DAY for r in visible)
    for seg in segments(master, "ORIC") + segments(master, "1ORIC"):
        if seg["valid_from"] == G3_DAY:
            assert seg["available_at"] == et_midnight(G3_DAY)
    assert_nothing_known_before_its_valid_from(master)
