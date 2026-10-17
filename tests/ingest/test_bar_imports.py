"""Guards against a second bar source quietly rewriting the Alpaca store, or a hand-kept import
list letting in an entry nobody can trace: every imported bar lands in its own directory
(``bars_1d_ext``), under the symbol the import entry names, with its source and the source's
own symbol beside it; the import list is validated entry by entry; and the same two write-time
rules as the Alpaca files hold (a bar not final at ``fetched_at`` is not stored; an existing row
is never replaced).

Target API (asof.ingest.bar_imports; implementation pending):

    EXT_DIR = "bars_1d_ext"
    EXT_BAR_SCHEMA = BAR_SCHEMA + source (string) + source_symbol (string)
    class BarImport(BaseModel, frozen=True): symbol, source, source_symbol, start, end, reason,
                                             added, by
    load_bar_imports(path) -> list[BarImport]
    class ImportReport: rows_fetched, rows_in_range, rows_added, rows_existing, first_session,
                        last_session
    import_bars(root, entry, bars, fetched_at) -> ImportReport

Fake data provenance:
- The PARA entry is the real first entry (2026-10-16): Alpaca rewrote PARA with Banzai's chain
  (VII -> BNZI -> PARA) after the 2026-08-07 reuse, so Paramount Global's class B bars
  2022-02-17 .. 2025-08-06 exist only at Yahoo, under the current symbol SKYD (measured).
- The SKYD prices are Yahoo's rows for 2022-02-16 .. 02-18 after rounding, equal to Alpaca's
  VIAC bars of those days (measured 2026-10-16); vwap is the typical-price proxy (project
  decision 2026-10-16); trade_count None (Yahoo has none).
- A daily bar is final at 20:00 New York of its session (docs: ingest.bars); an import that
  runs during a session must not store that session's bar.
- File layout, metadata key and the merge rule are those of ``write_partition`` (docs).

Real-data checks: after ``asof market import-bars`` (or the equivalent) on the real store,
``asof market audit`` C1 must show ext_bars 869 over 2022 .. 2025 for PARA and ext_overrides 869
(every imported key replaces a Banzai row Alpaca files under PARA), P1 0; the golden case
``test_para_2022_2025_is_paramount_from_the_second_source`` must pass.
"""

import textwrap
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from asof.ingest.alpaca import Bar
from asof.ingest.bar_imports import (
    EXT_BAR_SCHEMA,
    EXT_DIR,
    BarImport,
    ImportReport,
    import_bars,
    load_bar_imports,
)
from asof.ingest.bars import BAR_SCHEMA, FETCHED_AT_KEY, write_partition
from tests.ingest.fakes import daily

ET = ZoneInfo("America/New_York")
IMPORTED_AT = datetime(2026, 10, 16, 15, 0, tzinfo=ET)
PARA = BarImport(
    symbol="PARA",
    source="yahoo",
    source_symbol="SKYD",
    start=date(2022, 2, 17),
    end=date(2025, 8, 6),
    reason="Paramount Global class B; Alpaca rewrote PARA with Banzai's history after the "
    "2026-08-07 reuse",
    added=date(2026, 10, 16),
    by="greyson",
)
PARA_YAML = textwrap.dedent(
    """\
    - symbol: PARA
      source: yahoo
      source_symbol: SKYD
      start: 2022-02-17
      end: 2025-08-06
      reason: "Paramount Global class B; Alpaca rewrote PARA with Banzai's history after the
        2026-08-07 reuse"
      added: 2026-10-16
      by: greyson
    """
)


def ybar(
    day: date,
    *,
    open: float,
    high: float,
    low: float,
    close: float,
    volume: int,
    symbol: str = "SKYD",
) -> Bar:
    """A bar as the Yahoo client builds it: midnight New York, typical-price vwap, no count."""
    return Bar(
        symbol=symbol,
        t=datetime(day.year, day.month, day.day, tzinfo=ET),
        open=open,
        high=high,
        low=low,
        close=close,
        volume=volume,
        trade_count=None,
        vwap=round((high + low + close) / 3, 4),
    )


SKYD = [
    ybar(date(2022, 2, 16), open=30.24, high=30.33, low=27.84, close=29.58, volume=87606200),
    ybar(date(2022, 2, 17), open=29.51, high=30.45, low=29.1, close=29.35, volume=12100500),
    ybar(date(2022, 2, 18), open=29.0, high=29.64, low=27.62, close=28.38, volume=17020000),
]
EXT_2022_P = Path(EXT_DIR) / "year=2022" / "bucket=P.parquet"


def rows_of(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def write_yaml(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "bar_imports.yaml"
    path.write_text(text)
    return path


def entry_yaml(**overrides: Any) -> str:
    fields: dict[str, Any] = {
        "symbol": "PARA",
        "source": "yahoo",
        "source_symbol": "SKYD",
        "start": "2022-02-17",
        "end": "2025-08-06",
        "reason": '"Paramount Global class B"',
        "added": "2026-10-16",
        "by": "greyson",
    }
    fields.update(overrides)
    lines = [f"  {k}: {v}" for k, v in fields.items() if v is not None]
    lines[0] = "-" + lines[0][1:]
    return "\n".join(lines) + "\n"


# =============================================================================================
# schema and model
# =============================================================================================


def test_ext_schema_is_the_bar_schema_plus_source_columns() -> None:
    assert EXT_DIR == "bars_1d_ext"
    assert EXT_BAR_SCHEMA.names == [*BAR_SCHEMA.names, "source", "source_symbol"]
    for name in BAR_SCHEMA.names:
        assert EXT_BAR_SCHEMA.field(name).type == BAR_SCHEMA.field(name).type
    assert EXT_BAR_SCHEMA.field("source").type == pa.string()
    assert EXT_BAR_SCHEMA.field("source_symbol").type == pa.string()


def test_bar_import_is_frozen_and_keeps_its_fields() -> None:
    assert (PARA.symbol, PARA.source, PARA.source_symbol) == ("PARA", "yahoo", "SKYD")
    assert (PARA.start, PARA.end) == (date(2022, 2, 17), date(2025, 8, 6))
    assert (PARA.added, PARA.by) == (date(2026, 10, 16), "greyson")
    with pytest.raises((ValueError, TypeError, AttributeError)):
        PARA.symbol = "VIAC"  # type: ignore[misc]


@pytest.mark.parametrize(
    "overrides",
    [
        {"symbol": "para"},
        {"symbol": "X_DELISTED"},
        {"source_symbol": "abc def"},
        {"source": "alpaca"},
        {"source": "bloomberg"},
        {"start": date(2025, 8, 7)},  # after end
        {"reason": ""},
        {"reason": "   "},
        {"by": ""},
    ],
    ids=[
        "lowercase_symbol",
        "junk_symbol",
        "bad_source_symbol",
        "alpaca_is_not_an_import_source",
        "unknown_source",
        "start_after_end",
        "blank_reason",
        "whitespace_reason",
        "blank_by",
    ],
)
def test_bar_import_rejects_bad_fields(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        BarImport(**{**PARA.model_dump(), **overrides})


def test_bar_import_of_a_single_day_is_allowed() -> None:
    one = BarImport(**{**PARA.model_dump(), "start": date(2024, 6, 3), "end": date(2024, 6, 3)})
    assert one.start == one.end


# =============================================================================================
# load_bar_imports
# =============================================================================================


def test_load_the_real_first_entry(tmp_path: Path) -> None:
    assert load_bar_imports(write_yaml(tmp_path, PARA_YAML)) == [PARA]


def test_load_empty_list_and_missing_file(tmp_path: Path) -> None:
    assert load_bar_imports(write_yaml(tmp_path, "[]\n")) == []
    with pytest.raises(FileNotFoundError):
        load_bar_imports(tmp_path / "absent.yaml")


@pytest.mark.parametrize(
    ("text", "match"),
    [
        (entry_yaml(source_symbol=None), r"source_symbol"),
        (entry_yaml(reason=None), r"reason"),
        (entry_yaml(reason='""'), r"reason"),
        (entry_yaml(by=None), r"by"),
        (entry_yaml(added=None), r"added"),
        (entry_yaml(start="2022-02-30"), r"start"),
        (entry_yaml(end="not-a-date"), r"end"),
        (entry_yaml(start="2025-08-07"), r"start|end"),
        (entry_yaml(source="alpaca"), r"source"),
        (entry_yaml(symbol="para"), r"symbol"),
        ("symbol: PARA\n", r"list"),
        ("- PARA\n", r"0"),
    ],
    ids=[
        "missing_source_symbol",
        "missing_reason",
        "blank_reason",
        "missing_by",
        "missing_added",
        "impossible_start",
        "end_not_a_date",
        "start_after_end",
        "alpaca_source",
        "lowercase_symbol",
        "not_a_list",
        "entry_not_a_mapping",
    ],
)
def test_load_rejects_bad_entries_naming_the_field(tmp_path: Path, text: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        load_bar_imports(write_yaml(tmp_path, text))


def test_load_names_the_index_of_the_bad_entry(tmp_path: Path) -> None:
    text = PARA_YAML + entry_yaml(symbol="VIAC", start="2020-02-13", end="2022-02-16", by='""')

    with pytest.raises(ValueError, match=r"\b1\b.*\bby\b|\bby\b.*\b1\b"):
        load_bar_imports(write_yaml(tmp_path, text))


def test_load_rejects_overlapping_entries_for_one_symbol(tmp_path: Path) -> None:
    text = PARA_YAML + entry_yaml(start="2025-08-06", end="2025-12-31")

    with pytest.raises(ValueError, match="PARA"):
        load_bar_imports(write_yaml(tmp_path, text))


def test_load_accepts_adjacent_entries_and_overlaps_across_symbols(tmp_path: Path) -> None:
    text = (
        PARA_YAML
        + entry_yaml(start="2025-08-07", end="2025-12-31")
        + entry_yaml(symbol="VIAC", source_symbol="SKYD", start="2022-02-17", end="2025-08-06")
    )

    loaded = load_bar_imports(write_yaml(tmp_path, text))

    assert [(e.symbol, e.start, e.end) for e in loaded] == [
        ("PARA", date(2022, 2, 17), date(2025, 8, 6)),
        ("PARA", date(2025, 8, 7), date(2025, 12, 31)),
        ("VIAC", date(2022, 2, 17), date(2025, 8, 6)),
    ]


# =============================================================================================
# import_bars: normal path
# =============================================================================================


def test_import_writes_the_in_range_bars_under_the_entry_symbol(tmp_path: Path) -> None:
    report = import_bars(tmp_path, PARA, SKYD, IMPORTED_AT)

    assert isinstance(report, ImportReport)
    assert (report.rows_fetched, report.rows_in_range) == (3, 2)
    assert (report.rows_added, report.rows_existing) == (2, 0)
    assert (report.first_session, report.last_session) == (date(2022, 2, 17), date(2022, 2, 18))

    path = tmp_path / EXT_2022_P
    table = pq.read_table(path)
    assert table.schema.equals(EXT_BAR_SCHEMA, check_metadata=False)
    assert table.schema.metadata[FETCHED_AT_KEY] == IMPORTED_AT.astimezone(UTC).isoformat().encode()
    assert rows_of(path) == [
        {
            "symbol": "PARA",
            "t": datetime(2022, 2, 17, 5, 0, tzinfo=UTC),
            "session_date": date(2022, 2, 17),
            "open": 29.51,
            "high": 30.45,
            "low": 29.1,
            "close": 29.35,
            "volume": 12100500,
            "trade_count": None,
            "vwap": 29.6333,
            "available_at": datetime(2022, 2, 18, 1, 0, tzinfo=UTC),  # 20:00 New York
            "fetched_at": IMPORTED_AT.astimezone(UTC),
            "source": "yahoo",
            "source_symbol": "SKYD",
        },
        {
            "symbol": "PARA",
            "t": datetime(2022, 2, 18, 5, 0, tzinfo=UTC),
            "session_date": date(2022, 2, 18),
            "open": 29.0,
            "high": 29.64,
            "low": 27.62,
            "close": 28.38,
            "volume": 17020000,
            "trade_count": None,
            "vwap": 28.5467,
            "available_at": datetime(2022, 2, 19, 1, 0, tzinfo=UTC),
            "fetched_at": IMPORTED_AT.astimezone(UTC),
            "source": "yahoo",
            "source_symbol": "SKYD",
        },
    ]


def test_import_never_touches_the_alpaca_directory(tmp_path: Path) -> None:
    alpaca = tmp_path / "bars_1d" / "year=2022" / "bucket=P.parquet"
    write_partition(alpaca, [daily("PARA", date(2022, 2, 17), close=0.19)], "1Day", IMPORTED_AT)
    before = alpaca.read_bytes()

    import_bars(tmp_path, PARA, SKYD, IMPORTED_AT)

    assert alpaca.read_bytes() == before
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*.parquet")) == [
        Path("bars_1d") / "year=2022" / "bucket=P.parquet",
        EXT_2022_P,
    ]


def test_import_without_an_alpaca_store_creates_only_the_ext_directory(tmp_path: Path) -> None:
    import_bars(tmp_path, PARA, SKYD, IMPORTED_AT)

    assert not (tmp_path / "bars_1d").exists()
    assert (tmp_path / EXT_2022_P).exists()


def test_import_splits_bars_by_new_york_year(tmp_path: Path) -> None:
    bars = [
        ybar(date(2022, 12, 30), open=16.8, high=17.0, low=16.6, close=16.88, volume=1),
        ybar(date(2023, 1, 3), open=17.0, high=17.3, low=16.7, close=17.05, volume=2),
    ]

    report = import_bars(tmp_path, PARA, bars, IMPORTED_AT)

    assert (report.rows_added, report.first_session, report.last_session) == (
        2,
        date(2022, 12, 30),
        date(2023, 1, 3),
    )
    assert [r["session_date"] for r in rows_of(tmp_path / EXT_2022_P)] == [date(2022, 12, 30)]
    assert [
        r["session_date"] for r in rows_of(tmp_path / EXT_DIR / "year=2023" / "bucket=P.parquet")
    ] == [date(2023, 1, 3)]


def test_bucket_follows_the_entry_symbol_not_the_source_symbol(tmp_path: Path) -> None:
    import_bars(tmp_path, PARA, SKYD, IMPORTED_AT)

    assert not (tmp_path / EXT_DIR / "year=2022" / "bucket=S.parquet").exists()
    assert {r["symbol"] for r in rows_of(tmp_path / EXT_2022_P)} == {"PARA"}


# =============================================================================================
# import_bars: boundaries
# =============================================================================================


def test_reimporting_the_same_bars_adds_nothing_and_keeps_the_first_fetched_at(
    tmp_path: Path,
) -> None:
    first = import_bars(tmp_path, PARA, SKYD, IMPORTED_AT)
    later = IMPORTED_AT.replace(day=17)
    changed = [b.model_copy(update={"close": 99.0}) for b in SKYD]

    second = import_bars(tmp_path, PARA, changed, later)

    assert (first.rows_added, first.rows_existing) == (2, 0)
    assert (second.rows_added, second.rows_existing) == (0, 2)
    rows = rows_of(tmp_path / EXT_2022_P)
    assert [r["close"] for r in rows] == [29.35, 28.38]  # existing rows untouched
    assert {r["fetched_at"] for r in rows} == {IMPORTED_AT.astimezone(UTC)}
    assert pq.read_table(tmp_path / EXT_2022_P).schema.metadata[FETCHED_AT_KEY] == (
        later.astimezone(UTC).isoformat().encode()
    )


def test_merge_appends_only_new_keys_in_symbol_time_order(tmp_path: Path) -> None:
    import_bars(tmp_path, PARA, SKYD[1:2], IMPORTED_AT)
    later = IMPORTED_AT.replace(day=17)
    extra = ybar(date(2022, 2, 22), open=28.5, high=28.9, low=28.1, close=28.7, volume=3)

    report = import_bars(tmp_path, PARA, [extra, *SKYD], later)

    assert (report.rows_added, report.rows_existing) == (2, 1)
    rows = rows_of(tmp_path / EXT_2022_P)
    assert [(r["session_date"], r["fetched_at"]) for r in rows] == [
        (date(2022, 2, 17), IMPORTED_AT.astimezone(UTC)),
        (date(2022, 2, 18), later.astimezone(UTC)),
        (date(2022, 2, 22), later.astimezone(UTC)),
    ]


def test_bars_entirely_outside_the_entry_range_import_nothing(tmp_path: Path) -> None:
    report = import_bars(tmp_path, PARA, SKYD[:1], IMPORTED_AT)

    assert (report.rows_fetched, report.rows_in_range, report.rows_added) == (1, 0, 0)
    assert report.rows_existing == 0
    assert (report.first_session, report.last_session) == (None, None)
    assert not (tmp_path / "bars_1d").exists()


def test_range_edges_are_inclusive(tmp_path: Path) -> None:
    entry = BarImport(**{**PARA.model_dump(), "start": date(2022, 2, 17), "end": date(2022, 2, 18)})
    outside = ybar(date(2022, 2, 22), open=28.5, high=28.9, low=28.1, close=28.7, volume=3)

    report = import_bars(tmp_path, entry, [*SKYD, outside], IMPORTED_AT)

    assert report.rows_in_range == 2
    assert [r["session_date"] for r in rows_of(tmp_path / EXT_2022_P)] == [
        date(2022, 2, 17),
        date(2022, 2, 18),
    ]


def test_duplicate_keys_within_one_import_keep_the_last(tmp_path: Path) -> None:
    again = SKYD[1].model_copy(update={"close": 29.36})

    report = import_bars(tmp_path, PARA, [SKYD[1], again], IMPORTED_AT)

    assert report.rows_added == 1
    assert [r["close"] for r in rows_of(tmp_path / EXT_2022_P)] == [29.36]


def test_existing_file_with_another_schema_is_refused(tmp_path: Path) -> None:
    # an Alpaca-shaped file (no source columns) at the ext path: the import must not guess
    path = tmp_path / EXT_2022_P
    write_partition(path, [daily("PARA", date(2022, 2, 17))], "1Day", IMPORTED_AT)

    with pytest.raises(ValueError, match="bucket=P.parquet"):
        import_bars(tmp_path, PARA, SKYD, IMPORTED_AT)

    assert pq.read_table(path).schema.equals(BAR_SCHEMA, check_metadata=False)  # untouched


def test_naive_fetched_at_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fetched_at"):
        import_bars(tmp_path, PARA, SKYD, datetime(2026, 10, 16, 15, 0))

    assert not (tmp_path / EXT_DIR).exists()


def test_import_accepts_any_sequence_including_empty(tmp_path: Path) -> None:
    report = import_bars(tmp_path, PARA, (), IMPORTED_AT)

    assert (report.rows_fetched, report.rows_added, report.rows_existing) == (0, 0, 0)


# =============================================================================================
# import_bars: leakage
# =============================================================================================


def test_bar_not_final_at_fetched_at_is_not_stored(tmp_path: Path) -> None:
    # Imported during the 2022-02-18 session (19:00 New York, after the close but before the
    # post-market session ends): that day's bar is not final and must wait for a later run.
    during = datetime(2022, 2, 18, 19, 0, tzinfo=ET)

    report = import_bars(tmp_path, PARA, SKYD, during)

    assert report.rows_added == 1
    assert [r["session_date"] for r in rows_of(tmp_path / EXT_2022_P)] == [date(2022, 2, 17)]

    at_close = datetime(2022, 2, 18, 20, 0, tzinfo=ET)
    report = import_bars(tmp_path, PARA, SKYD, at_close)
    assert (report.rows_added, report.rows_existing) == (1, 1)
    assert [r["session_date"] for r in rows_of(tmp_path / EXT_2022_P)] == [
        date(2022, 2, 17),
        date(2022, 2, 18),
    ]


def test_every_stored_row_is_available_at_the_post_market_close_of_its_session(
    tmp_path: Path,
) -> None:
    bars = [
        ybar(date(2022, 2, 17), open=29.51, high=30.45, low=29.1, close=29.35, volume=1),
        ybar(date(2022, 6, 1), open=1.0, high=2.0, low=0.5, close=1.5, volume=2),  # EDT
    ]

    import_bars(tmp_path, PARA, bars, IMPORTED_AT)

    rows = rows_of(tmp_path / EXT_2022_P)
    assert [(r["session_date"], r["available_at"]) for r in rows] == [
        (date(2022, 2, 17), datetime(2022, 2, 18, 1, 0, tzinfo=UTC)),  # 20:00 EST
        (date(2022, 6, 1), datetime(2022, 6, 2, 0, 0, tzinfo=UTC)),  # 20:00 EDT
    ]
    assert all(r["available_at"] > r["t"] for r in rows)
    assert all(r["available_at"] <= r["fetched_at"] for r in rows)


# =============================================================================================
# reviewer 2026-10-16: guards added after the review
# =============================================================================================


def write_splits(root: Path, rows: list[tuple[str, date]]) -> None:
    """A splits table with one 2:1 forward split per (symbol, ex_date)."""
    from asof.ingest.corporate_actions import SPLIT_SCHEMA

    path = root / "corporate_actions" / "splits.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(
        [
            {
                "symbol": symbol,
                "ex_date": day,
                "old_rate": 1.0,
                "new_rate": 2.0,
                "factor": 2.0,
                "kind": "forward_split",
                "old_cusip": None,
                "new_cusip": None,
                "available_at": datetime(day.year, day.month, day.day, tzinfo=ET).astimezone(UTC),
            }
            for symbol, day in rows
        ],
        schema=SPLIT_SCHEMA,
    )
    pq.write_table(table, path)


@pytest.mark.parametrize(
    "split",
    [("PARA", date(2024, 6, 10)), ("SKYD", date(2026, 1, 5)), ("PARA", date(2022, 2, 18))],
    ids=["stored_symbol_inside_range", "source_symbol_after_range", "day_after_start"],
)
def test_import_refuses_a_lineage_with_a_split_after_the_start(
    tmp_path: Path, split: tuple[str, date]
) -> None:
    # measured 2026-10-16: Yahoo's chart prices and volumes are split-adjusted (NVDA
    # 2024-06-07 close 120.89 at Yahoo, 1208.88 raw at Alpaca), so any split after the
    # start of the range would have rewritten every earlier bar; the store keeps raw prices
    write_splits(tmp_path, [split])

    with pytest.raises(ValueError, match=split[0]):
        import_bars(tmp_path, PARA, SKYD, IMPORTED_AT)

    assert not (tmp_path / EXT_DIR).exists()


def test_import_allows_a_split_on_or_before_the_start(tmp_path: Path) -> None:
    write_splits(tmp_path, [("PARA", date(2022, 2, 17)), ("SKYD", date(2020, 1, 2))])

    report = import_bars(tmp_path, PARA, SKYD, IMPORTED_AT)

    assert report.rows_added == 2


def test_import_refuses_bars_of_another_symbol_than_the_source_symbol(tmp_path: Path) -> None:
    wrong = [b.model_copy(update={"symbol": "VIAC"}) for b in SKYD]

    with pytest.raises(ValueError, match="VIAC"):
        import_bars(tmp_path, PARA, wrong, IMPORTED_AT)

    assert not (tmp_path / EXT_DIR).exists()
