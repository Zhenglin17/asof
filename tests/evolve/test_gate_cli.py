"""Guards against `asof gate report` showing numbers a user cannot trust or read: a run replayed
under another strategy version measured with today's files without saying so, a pass or fail
slipped back in while strategies are explored, a run the gate cannot measure ending in a
traceback instead of one line and exit code 1, and a report missing the numbers.

CLI under test (asof.cli):

    asof gate report --run N [--run M ...] [--permutations K] [--seed S] [--current-strategy]
        [--strategies-dir DIR] [--data-dir DIR] [--db FILE] [--report PATH]
        version check as `asof gate null`; runs asof.evolve.gate.check with the protocol;
        writes a markdown report (default <data-dir>/reports/gate-<ids joined by ->.md); prints
        a one-line summary and the report path. Exit 0 when written, 1 cannot run.

Interface points this file pins down where the spec left a choice:
- The command reads the protocol through ``asof.cli.load_protocol`` (as `gate null` and `gate
  compare` do), so these tests replace that name with gate_world's GATE instead of writing the
  committed configs/evaluation.yaml.
- Neither the output nor the report says pass or fail.
- The candidate's Sharpe standard error is printed with two decimals.

World: tests/evolve/gate_world.py (provenance there). The command forwards dates and runs to
check(), whose leakage cases are in test_gate.py; here only the report's period slices touch the
holdout, and they are labelled as report-only rows, outside the dev measurements.
"""

import math
import re
from pathlib import Path

import duckdb
import pytest
from sqlmodel import Session
from typer.testing import CliRunner

import asof.cli
from asof.cli import app
from asof.evolve.protocol import EvaluationProtocol
from asof.evolve.versus import compare_run
from tests.evolve.gate_world import (
    A_LATE,
    A_LEGS,
    DEV_DAYS,
    GATE,
    HOLDOUT_DAYS,
    PERMUTATIONS,
    Runs,
    a_returns,
    extra_run,
    seed,
    store,
)
from tests.strategy.files import DEFAULTS, write_strategy

runner = CliRunner()


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    store(tmp_path)
    return tmp_path


@pytest.fixture
def gate_protocol(monkeypatch: pytest.MonkeyPatch) -> EvaluationProtocol:
    monkeypatch.setattr(asof.cli, "load_protocol", lambda *args, **kwargs: GATE)
    return GATE


def args(data_dir: Path, *run_ids: int, extra: tuple[str, ...] = ()) -> list[str]:
    out = ["gate", "report"]
    for run_id in run_ids:
        out += ["--run", str(run_id)]
    return [
        *out, "--permutations", str(PERMUTATIONS), "--seed", "0",
        "--strategies-dir", str(data_dir / "strategies"),
        "--data-dir", str(data_dir), "--db", str(data_dir / "meta.db"), *extra,
    ]  # fmt: skip


def default_report(data_dir: Path, *run_ids: int) -> Path:
    return data_dir / "reports" / f"gate-{'-'.join(str(r) for r in run_ids)}.md"


def cannot_run(output: str) -> None:
    assert "Traceback" not in output
    assert len(output.strip().splitlines()) == 1, output


# =============================================================================================
# normal path
# =============================================================================================


def test_a_family_exits_zero_and_writes_the_report(
    data_dir: Path, session: Session, gate_protocol: EvaluationProtocol
) -> None:
    runs = seed(session, data_dir)

    result = runner.invoke(app, args(data_dir, runs.a, runs.b))

    assert result.exit_code == 0, result.output
    report = default_report(data_dir, runs.a, runs.b)
    assert str(report) in result.output
    assert "candidate run" in result.output
    text = report.read_text()
    lower = text.lower()
    # Family table: every run, and the adjusted p.
    assert str(runs.a) in text and str(runs.b) in text
    assert "p adj" in lower
    # Walk-forward segments: the three test windows' edges.
    for day in ("2025-02-01", "2025-02-28", "2025-03-01", "2025-04-01", "2025-04-30"):
        assert day in text, day
    for word in ("thin", "mde", "spy", "top-2", "versions"):
        assert word in lower, word
    # The candidate's dev metrics with the Sharpe standard error: A's dev Sharpe 15.97 over 81
    # sessions (measured). Lo (2002) on daily returns, annualized: sqrt((1 + SR^2 / 504) / years)
    # (SR^2 / 2 is the yearly-observation form; over daily data it overstates the error).
    sharpe = 15.972198067614586
    se = math.sqrt((1 + sharpe**2 / (2 * 252)) / (len(DEV_DAYS) / 252))
    assert f"{se:.2f}" in text
    # Top legs by pnl: P1 +10% (entered 04-22) before P4 +9% (entered 02-11).
    assert "P1" in text and "P4" in text
    assert text.index("2025-04-22") < text.index("2025-02-11")


def test_the_report_slices_the_candidates_holdout_rows_without_changing_the_numbers(
    data_dir: Path, session: Session, gate_protocol: EvaluationProtocol
) -> None:
    runs = seed(session, data_dir, holdout=True)

    result = runner.invoke(app, args(data_dir, runs.a, runs.b))

    assert result.exit_code == 0, result.output
    text = default_report(data_dir, runs.a, runs.b).read_text()
    assert HOLDOUT_DAYS[0].isoformat() in text
    assert HOLDOUT_DAYS[-1].isoformat() in text


def test_report_goes_where_report_says(
    data_dir: Path, session: Session, gate_protocol: EvaluationProtocol, tmp_path: Path
) -> None:
    runs = seed(session, data_dir)
    target = tmp_path / "elsewhere" / "verdict.md"

    result = runner.invoke(app, args(data_dir, runs.a, runs.b, extra=("--report", str(target))))

    assert result.exit_code == 0, result.output
    assert target.is_file()
    assert str(target) in result.output


# =============================================================================================
# boundaries
# =============================================================================================


def test_nothing_is_scored_even_under_a_strict_protocol(
    data_dir: Path, session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A min_positive no family here can reach: still exit 0, and no pass or fail anywhere.
    strict = EvaluationProtocol.model_validate(
        {**GATE.model_dump(),
         "walk_forward": {"train_months": 1, "test_months": 1, "min_positive": 3}}
    )  # fmt: skip
    monkeypatch.setattr(asof.cli, "load_protocol", lambda *args, **kwargs: strict)
    runs = seed(session, data_dir)

    result = runner.invoke(app, args(data_dir, runs.a, runs.b))

    assert result.exit_code == 0, result.output
    text = default_report(data_dir, runs.a, runs.b).read_text()
    for out in (result.output, text):
        assert not re.search(r"\b(pass|passed|fail|failed|verdict)\b", out.lower()), out
    assert "2 positive of 3 evaluable windows" in text


def test_an_unknown_run_exits_one(
    data_dir: Path, session: Session, gate_protocol: EvaluationProtocol
) -> None:
    runs = seed(session, data_dir)

    result = runner.invoke(app, args(data_dir, runs.a, 9999))

    assert result.exit_code == 1, result.output
    assert isinstance(result.exception, SystemExit)
    cannot_run(result.output)


def test_a_run_without_dev_equity_rows_exits_one(
    data_dir: Path, session: Session, gate_protocol: EvaluationProtocol
) -> None:
    runs = seed(session, data_dir)
    no_rows = extra_run(session, runs, [], [], A_LEGS)

    result = runner.invoke(app, args(data_dir, runs.a, no_rows))

    assert result.exit_code == 1, result.output
    cannot_run(result.output)
    assert "replay" in result.output.lower()


def test_a_run_without_dev_legs_exits_one(
    data_dir: Path, session: Session, gate_protocol: EvaluationProtocol
) -> None:
    runs = seed(session, data_dir)
    late_only = extra_run(session, runs, DEV_DAYS, a_returns(DEV_DAYS), A_LATE)

    result = runner.invoke(app, args(data_dir, runs.a, late_only))

    assert result.exit_code == 1, result.output
    cannot_run(result.output)


def changed_strategy(runs: Runs) -> None:
    """Today's files are no longer the version A and B replayed (max_sessions 60 -> 45); the
    universe, scan thresholds and slippage stay as they were."""
    write_strategy(
        runs.strategy_dir.parent,
        {"execute": {**DEFAULTS["execute"], "exit": {"max_sessions": 45}}},
    )


def test_a_run_of_another_version_is_refused_unless_current_strategy_is_given(
    data_dir: Path, session: Session, gate_protocol: EvaluationProtocol
) -> None:
    runs = seed(session, data_dir)
    changed_strategy(runs)

    refused = runner.invoke(app, args(data_dir, runs.a, runs.b))
    allowed = runner.invoke(app, args(data_dir, runs.a, runs.b, extra=("--current-strategy",)))

    assert refused.exit_code == 1, refused.output
    assert "Traceback" not in refused.output
    assert "--current-strategy" in refused.output
    assert allowed.exit_code == 0, allowed.output
    assert default_report(data_dir, runs.a, runs.b).is_file()


def test_the_reports_difference_table_matches_the_measured_differences(
    data_dir: Path, session: Session, gate_protocol: EvaluationProtocol
) -> None:
    # Each row of "the strategy minus" table, cell by cell, against compare_run's differences:
    # a flipped sign or swapped column would still mention every benchmark.
    runs = seed(session, data_dir)

    result = runner.invoke(app, args(data_dir, runs.a, runs.b))

    assert result.exit_code == 0, result.output
    text = default_report(data_dir, runs.a, runs.b).read_text()
    con = duckdb.connect()
    try:
        versus = compare_run(
            session, con, data_dir / "market", runs.a, GATE, window=(GATE.dev.start, GATE.dev.end)
        )
    finally:
        con.close()
    for d in versus.differences:
        sharpe = f"{d.sharpe:+.2f}" if d.sharpe is not None else "-"
        row = (
            f"| {d.label} | {d.total_return * 100:+.1f} pts | {d.max_drawdown * 100:+.1f} pts | "
            f"{sharpe} |"
        )
        assert row in text, row
    spy = versus.differences[0]
    assert (
        f"Sharpe {spy.sharpe:+.2f}, max drawdown {spy.max_drawdown * 100:+.1f} pts, total return "  # pyright: ignore[reportOptionalOperand]
        f"{spy.total_return * 100:+.1f} pts" in text
    )


def null_args(data_dir: Path, *run_ids: int, extra: tuple[str, ...] = ()) -> list[str]:
    out = ["gate", "null"]
    for run_id in run_ids:
        out += ["--run", str(run_id)]
    return [
        *out, "--permutations", str(PERMUTATIONS), "--seed", "0",
        "--strategies-dir", str(data_dir / "strategies"),
        "--data-dir", str(data_dir), "--db", str(data_dir / "meta.db"), *extra,
    ]  # fmt: skip


def test_gate_null_still_checks_the_version_through_the_shared_helper(
    data_dir: Path, session: Session, gate_protocol: EvaluationProtocol
) -> None:
    # `gate null` and `gate report` share the version check; the null command keeps its table
    # and refuses changed files unless --current-strategy is given.
    runs = seed(session, data_dir)

    ok = runner.invoke(app, null_args(data_dir, runs.a, runs.b))
    changed_strategy(runs)
    refused = runner.invoke(app, null_args(data_dir, runs.a, runs.b))

    assert ok.exit_code == 0, ok.output
    assert "p adj" in ok.output
    assert str(runs.a) in ok.output and str(runs.b) in ok.output
    assert refused.exit_code == 1, refused.output
    assert "Traceback" not in refused.output
    assert "--current-strategy" in refused.output
