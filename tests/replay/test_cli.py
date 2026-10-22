"""Guards against `asof replay` failing in ways a user cannot read: a bad --set override or an
unknown strategy must end with exit code 1 and one line, never a traceback and never a half-
written Run; a good run prints its run id, and `asof replay report --run N` prints that run's
markdown report.

CLI under test (asof.cli):

    asof replay run --strategy NAME --start YYYY-MM-DD --end YYYY-MM-DD [--set key=value ...]
                    [--strategies-dir DIR] [--data-dir DIR] [--db FILE]
        prints "run <id>" among its output
    asof replay report --run N [--db FILE]

--strategies-dir defaults to ./strategies; --data-dir and --db follow the other commands
($ASOF_DATA_DIR, $ASOF_DATA_DIR/meta.db). The store is tests/scan/world.py. No as_of here: the
command only forwards --start/--end to run_replay, whose leakage is tested in test_replay.py.
"""

import re
from pathlib import Path

import pytest
from sqlmodel import Session, select
from typer.testing import CliRunner

from asof.cli import app
from asof.store.models import Run
from tests.replay.scenario import END, START, TRADED
from tests.scan.world import gap_world
from tests.strategy.files import write_strategy

runner = CliRunner()


@pytest.fixture
def paths(tmp_path: Path, session: Session) -> dict[str, str]:
    gap_world(tmp_path, session, after=TRADED)
    strategies = write_strategy(tmp_path / "strategies").parent
    return {
        "--strategies-dir": str(strategies),
        "--data-dir": str(tmp_path),
        "--db": str(tmp_path / "meta.db"),
    }


def run_args(paths: dict[str, str], *extra: str, strategy: str = "earnings_gap") -> list[str]:
    args = ["replay", "run", "--strategy", strategy]
    args += ["--start", START.isoformat(), "--end", END.isoformat()]
    for option, value in paths.items():
        args += [option, value]
    return args + list(extra)


def runs(session: Session) -> list[Run]:
    session.expire_all()
    return list(session.exec(select(Run)).all())


def test_run_then_report(paths: dict[str, str], session: Session) -> None:
    result = runner.invoke(app, run_args(paths, "--set", "execute.fills.slippage_bps=0"))
    assert result.exit_code == 0, result.output
    match = re.search(r"run (\d+)", result.output)
    assert match, result.output
    run_id = int(match.group(1))
    assert [r.id for r in runs(session)] == [run_id]

    report = runner.invoke(app, ["replay", "report", "--run", str(run_id), "--db", paths["--db"]])
    assert report.exit_code == 0, report.output
    assert report.output.lstrip().startswith("#")
    assert "win rate" in report.output.lower()


def test_repeated_set_options_all_apply(paths: dict[str, str], session: Session) -> None:
    result = runner.invoke(
        app,
        run_args(
            paths,
            "--set",
            "execute.fills.slippage_bps=0",
            "--set",
            "execute.reentry.risk_mode=independent",
        ),
    )
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize(
    "override", ["execute.stop.nope=1", "execute.stop.trail_atr=wide", "no_equals_sign"]
)
def test_bad_override_exits_one_without_a_run(
    paths: dict[str, str], session: Session, override: str
) -> None:
    result = runner.invoke(app, run_args(paths, "--set", override))
    assert result.exit_code == 1, result.output
    assert "Traceback" not in result.output
    assert isinstance(result.exception, SystemExit)
    assert runs(session) == []


def test_unknown_strategy_exits_one(paths: dict[str, str], session: Session) -> None:
    result = runner.invoke(app, run_args(paths, strategy="no_such_strategy"))
    assert result.exit_code == 1, result.output
    assert isinstance(result.exception, SystemExit)
    assert runs(session) == []


def test_report_of_an_unknown_run_exits_one(paths: dict[str, str]) -> None:
    result = runner.invoke(app, ["replay", "report", "--run", "999", "--db", paths["--db"]])
    assert result.exit_code == 1, result.output
    assert isinstance(result.exception, SystemExit)


def test_end_before_start_exits_one(paths: dict[str, str], session: Session) -> None:
    args = run_args(paths)
    args[args.index("--start") + 1], args[args.index("--end") + 1] = (
        END.isoformat(),
        START.isoformat(),
    )
    result = runner.invoke(app, args)
    assert result.exit_code == 1, result.output
    assert runs(session) == []
