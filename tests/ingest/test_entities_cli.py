"""Guards against a half-built universe: `asof entities sync` writes every entity or none."""

import json
import re
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest
from sqlmodel import Session, select
from typer.testing import CliRunner

from asof.cli import app
from asof.store.db import init_db, make_engine
from asof.store.models import Entity

runner = CliRunner()

WATCHLIST = """\
groups:
  - id: funds
    label: Funds
    kind: etf
    added: 2026-10-05
    items:
      - {ticker: SPY, note: "S&P 500"}
      - {ticker: XLK, note: "Technology"}
      - {ticker: XLE, note: "Energy"}
      - {ticker: DRAM, note: "Memory chips"}
  - id: stocks
    label: Stocks
    kind: company
    added: 2026-10-05
    items:
      - {ticker: AAPL}
      - {ticker: MU}
      - {ticker: META}
  - id: coins
    label: Coins
    kind: crypto
    added: 2026-10-05
    items:
      - {ticker: BTC, name: Bitcoin}
"""
TICKERS = {"SPY", "XLK", "XLE", "DRAM", "AAPL", "MU", "META", "BTC"}
WITH_UNLISTED_COMPANY = WATCHLIST.replace("{ticker: META}", "{ticker: ZZZZ}")


def stored_entities(db_path: Path) -> list[Entity]:
    engine = make_engine(db_path)
    init_db(engine)
    try:
        with Session(engine) as session:
            return list(session.exec(select(Entity)).all())
    finally:
        engine.dispose()


def sync(watchlist: Path, sec_dir: Path, db_path: Path, *extra: str):
    args = ["entities", "sync", "--watchlist", str(watchlist), "--sec-dir", str(sec_dir)]
    return runner.invoke(app, [*args, "--db", str(db_path), *extra])


def summary_counts(output: str) -> list[int]:
    """Numbers on the line that reports both counts, in order of appearance."""
    lines = [
        line
        for line in output.splitlines()
        if "inserted" in line.lower() and "updated" in line.lower()
    ]
    assert len(lines) == 1, output
    return [int(n) for n in re.findall(r"\d+", lines[0])]


def test_sync_writes_every_watchlist_entity(
    write_watchlist: Callable[[str], Path], sec_dir: Path, tmp_path: Path
) -> None:
    db_path = tmp_path / "nested" / "meta.db"

    result = sync(write_watchlist(WATCHLIST), sec_dir, db_path)

    assert result.exit_code == 0, result.output
    rows = {e.ticker: e for e in stored_entities(db_path)}
    assert set(rows) == TICKERS
    assert (rows["AAPL"].kind, rows["AAPL"].cik, rows["AAPL"].name) == (
        "company",
        320193,
        "Apple Inc.",
    )
    assert rows["XLK"].cik == rows["XLE"].cik == 1064641
    assert (rows["XLK"].series_id, rows["XLE"].series_id) == ("S000006415", "S000006410")
    assert (rows["DRAM"].cik, rows["DRAM"].series_id) == (None, None)
    assert (rows["BTC"].kind, rows["BTC"].cik, rows["BTC"].name) == ("crypto", None, "Bitcoin")

    for ticker in TICKERS:
        assert ticker in result.output
    for name in ["Apple Inc.", "MICRON TECHNOLOGY INC", "SPDR S&P 500 ETF TRUST", "Bitcoin"]:
        assert name in result.output
    assert summary_counts(result.output) == [8, 0]


def test_sync_warns_about_funds_the_sec_does_not_list(
    write_watchlist: Callable[[str], Path], sec_dir: Path, tmp_path: Path
) -> None:
    result = sync(write_watchlist(WATCHLIST), sec_dir, tmp_path / "meta.db")

    assert result.exit_code == 0, result.output
    # DRAM appears once in its entity line and once more in the warning.
    assert sum("DRAM" in line for line in result.output.splitlines()) >= 2


def test_second_sync_changes_nothing(
    write_watchlist: Callable[[str], Path], sec_dir: Path, tmp_path: Path
) -> None:
    watchlist = write_watchlist(WATCHLIST)
    db_path = tmp_path / "meta.db"
    assert sync(watchlist, sec_dir, db_path).exit_code == 0
    first = {e.ticker: e.id for e in stored_entities(db_path)}

    result = sync(watchlist, sec_dir, db_path)

    assert result.exit_code == 0, result.output
    assert {e.ticker: e.id for e in stored_entities(db_path)} == first
    assert summary_counts(result.output) == [0, 0]


def test_unlisted_company_aborts_without_writing(
    write_watchlist: Callable[[str], Path], sec_dir: Path, tmp_path: Path
) -> None:
    db_path = tmp_path / "meta.db"

    result = sync(write_watchlist(WITH_UNLISTED_COMPANY), sec_dir, db_path)

    assert result.exit_code == 1, result.output
    assert "ZZZZ" in result.output
    assert stored_entities(db_path) == []


def test_unlisted_company_leaves_an_existing_database_untouched(
    write_watchlist: Callable[[str], Path], sec_dir: Path, tmp_path: Path
) -> None:
    db_path = tmp_path / "meta.db"
    assert sync(write_watchlist(WATCHLIST), sec_dir, db_path).exit_code == 0
    before = {(e.id, e.ticker, e.name) for e in stored_entities(db_path)}

    result = sync(write_watchlist(WITH_UNLISTED_COMPANY), sec_dir, db_path)

    assert result.exit_code == 1, result.output
    assert {(e.id, e.ticker, e.name) for e in stored_entities(db_path)} == before


@pytest.mark.parametrize("present", [[], ["company_tickers.json"], ["company_tickers_mf.json"]])
def test_missing_sec_files_point_to_download(
    write_watchlist: Callable[[str], Path], sec_dir: Path, tmp_path: Path, present: list[str]
) -> None:
    partial = tmp_path / "sec"
    partial.mkdir()
    for name in present:
        shutil.copy(sec_dir / name, partial / name)
    db_path = tmp_path / "meta.db"

    result = sync(write_watchlist(WATCHLIST), partial, db_path)

    assert result.exit_code == 1, result.output
    assert "--download" in result.output
    assert stored_entities(db_path) == []


def test_download_requires_a_user_agent(
    write_watchlist: Callable[[str], Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[object, ...]] = []

    def fake_download(*args: object, **kwargs: object) -> None:
        calls.append(args)

    monkeypatch.delenv("ASOF_SEC_USER_AGENT", raising=False)
    # Cover both import styles the command might use.
    monkeypatch.setattr("asof.ingest.sec_tickers.download_sec_listings", fake_download)
    monkeypatch.setattr("asof.cli.download_sec_listings", fake_download, raising=False)
    db_path = tmp_path / "meta.db"

    result = sync(write_watchlist(WATCHLIST), tmp_path / "sec", db_path, "--download")

    assert result.exit_code == 1, result.output
    assert "ASOF_SEC_USER_AGENT" in result.output
    assert calls == []
    assert stored_entities(db_path) == []


def test_sec_dir_and_db_default_to_the_data_dir(
    write_watchlist: Callable[[str], Path],
    sec_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    shutil.copytree(sec_dir, data_dir / "raw" / "sec")
    monkeypatch.setenv("ASOF_DATA_DIR", str(data_dir))

    result = runner.invoke(
        app, ["entities", "sync", "--watchlist", str(write_watchlist(WATCHLIST))]
    )

    assert result.exit_code == 0, result.output
    assert {e.ticker for e in stored_entities(data_dir / "meta.db")} == TICKERS


def test_fund_listed_for_two_registrants_aborts_without_writing(
    write_watchlist: Callable[[str], Path], sec_dir: Path, tmp_path: Path
) -> None:
    clashing = tmp_path / "sec"
    shutil.copytree(sec_dir, clashing)
    company_file = clashing / "company_tickers.json"
    table = json.loads(company_file.read_text())
    table["99"] = {"cik_str": 999, "ticker": "XLK", "title": "Unrelated Corp"}
    company_file.write_text(json.dumps(table))
    db_path = tmp_path / "meta.db"

    result = sync(write_watchlist(WATCHLIST), clashing, db_path)

    assert result.exit_code == 1, result.output
    assert "XLK" in result.output
    assert "cik" in result.output
    assert stored_entities(db_path) == []


def test_missing_or_malformed_watchlist_is_reported_in_one_line(
    write_watchlist: Callable[[str], Path], sec_dir: Path, tmp_path: Path
) -> None:
    db_path = tmp_path / "meta.db"

    absent = sync(tmp_path / "nowhere.yaml", sec_dir, db_path)
    malformed = sync(write_watchlist("groups: [oops"), sec_dir, db_path)

    for result in (absent, malformed):
        assert result.exit_code == 1, result.output
        assert isinstance(result.exception, SystemExit)
        assert "Traceback" not in result.output
    assert stored_entities(db_path) == []
