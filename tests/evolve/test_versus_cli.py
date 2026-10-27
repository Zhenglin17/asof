"""Guards against `asof gate compare` failing in ways a user cannot read: an unknown run or a run
without daily equity must end with exit code 1 and one line, never a traceback; a good run prints
the strategy and every protocol benchmark with their numbers, the exposure figures and the three
criteria as yes/no.

CLI under test (asof.cli):

    asof gate compare --run N [--data-dir DIR] [--db FILE]
        rows: strategy, then each benchmark; columns total, annualized, max dd, sharpe;
        then mean exposure, high-exposure share, and the three report-only criteria (yes/no).

The command reads the committed protocol (configs/evaluation.yaml: SPY, QQQ, a top-10 basket),
so the store holds SPY and QQQ besides tests/evolve/test_benchmark.py's stocks A-D (provenance
there). QQQ is SPY's path at twice the price, an ETF like SPY; made up. No as_of here: the
command forwards a run id to compare_run, whose leakage case is in test_versus.py.
"""

import re
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pytest
from sqlmodel import Session
from typer.testing import CliRunner

from asof.cli import app
from asof.store.models import EquityDaily, Run
from tests.evolve.test_benchmark import FEB_FIRST, LAST, SEGS, SESSIONS, bar, build, default_ohlcv
from tests.scan.world import close_of
from tests.store.test_universe import Seg

runner = CliRunner()
WINDOW = [d for d in SESSIONS if FEB_FIRST <= d <= LAST]
INITIAL = 100_000.0


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    bars = []
    for day in SESSIONS:
        for symbol in ("A", "B", "C", "D", "SPY"):
            bars.append(bar(symbol, day, default_ohlcv(symbol, day)))
        o, c, v = default_ohlcv("SPY", day)
        bars.append(bar("QQQ", day, (2 * o, 2 * c, v)))
    build(tmp_path, bars, [*SEGS, Seg("S_QQQ", "QQQ", cls="etf")])
    return tmp_path


def new_run(session: Session, equity: Sequence[float] | None) -> int:
    run = Run(
        strategy_id="earnings_gap",
        mode="replay",
        as_of=close_of(LAST),
        started_at=close_of(LAST),
        git_sha="test",
    )
    session.add(run)
    session.flush()
    assert run.id is not None
    for day, value in zip(WINDOW, equity or (), strict=False):
        session.add(
            EquityDaily(
                run_id=run.id,
                session_date=day,
                equity=value,
                realized=value - INITIAL,
                unrealized=0.0,
                gross_exposure=0.0,
                open_positions=0,
                available_at=close_of(day),
            )
        )
    session.commit()
    return run.id


def answers(text: str) -> list[str]:
    return re.findall(r"\b(yes|no)\b", text)


def compare(data_dir: Path, run_id: int) -> list[str]:
    return [
        "gate", "compare", "--run", str(run_id),
        "--data-dir", str(data_dir), "--db", str(data_dir / "meta.db"),
    ]  # fmt: skip


def strong_path() -> list[float]:
    """+1% every other session, never down: total 22%, beating SPY (3%) and the top-10 basket
    (7.6%, test_benchmark's size-10 case) with no drawdown and a far higher Sharpe."""
    out, value = [], INITIAL
    for i in range(len(WINDOW)):
        value *= 1.01 if i % 2 == 0 else 1.0
        out.append(value)
    return out


def test_prints_every_row_the_exposure_and_the_criteria(data_dir: Path, session: Session) -> None:
    run_id = new_run(session, strong_path())

    result = runner.invoke(app, compare(data_dir, run_id))

    assert result.exit_code == 0, result.output
    text = result.output.lower()
    assert "strategy" in text
    for name in ("spy", "qqq", "top-10"):
        assert name in text, name
    for column in ("total", "annualized", "sharpe"):
        assert column in text, column
    assert "dd" in text or "drawdown" in text
    assert "mean exposure" in text
    assert "high-exposure share" in text
    assert "22.02%" in result.output  # the strategy's total return
    # Every criterion holds for this path.
    assert answers(text) == ["yes", "yes", "yes"], result.output


def test_a_run_that_never_traded_prints_no_for_sharpe_and_total(
    data_dir: Path, session: Session
) -> None:
    run_id = new_run(session, [INITIAL] * len(WINDOW))

    result = runner.invoke(app, compare(data_dir, run_id))

    assert result.exit_code == 0, result.output
    assert "Traceback" not in result.output
    text = result.output.lower()
    # Drawdown 0 is not worse than the basket's; no Sharpe and no return pass nothing.
    assert sorted(answers(text)) == ["no", "no", "yes"], result.output


def test_an_unknown_run_exits_one(data_dir: Path, session: Session) -> None:
    result = runner.invoke(app, compare(data_dir, 999))

    assert result.exit_code == 1, result.output
    assert isinstance(result.exception, SystemExit)
    assert "Traceback" not in result.output
    assert result.output.strip()


def test_a_run_without_equity_rows_exits_one(data_dir: Path, session: Session) -> None:
    run_id = new_run(session, None)

    result = runner.invoke(app, compare(data_dir, run_id))

    assert result.exit_code == 1, result.output
    assert isinstance(result.exception, SystemExit)
    assert "Traceback" not in result.output
    assert result.output.strip()


def test_the_window_is_the_runs_own(data_dir: Path, session: Session) -> None:
    # Ten sessions from 2025-02-10: the output names the window it compared over.
    run_id = new_run(session, None)
    days = [d for d in WINDOW if d >= date(2025, 2, 10)][:10]
    for day in days:
        session.add(
            EquityDaily(
                run_id=run_id,
                session_date=day,
                equity=INITIAL,
                realized=0.0,
                unrealized=0.0,
                gross_exposure=0.0,
                open_positions=0,
                available_at=close_of(day),
            )
        )
    session.commit()

    result = runner.invoke(app, compare(data_dir, run_id))

    assert result.exit_code == 0, result.output
    assert days[0].isoformat() in result.output
    assert days[-1].isoformat() in result.output
