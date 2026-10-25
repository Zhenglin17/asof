"""Guards against a replay looking at the sealed holdout by accident: a refused run that still
registers a parameter set (one more attempt nobody can see) or writes a Run, a refused run that
reads bars before refusing, an opening that is not recorded or is recorded against the wrong
version, a strategy that opens the holdout a second time through another parameter set, and a
CLI refusal that ends in a traceback or the misleading "the parameter set is registered" line.

API under test (asof.replay.run_replay, asof.cli):

    run_replay(session, con, root, strategy_dir, start, end, overrides=(), *,
               protocol: EvaluationProtocol | None = None, open_holdout: bool = False) -> run_id
        protocol None -> load_protocol(DEFAULT_PROTOCOL_PATH). Order: check_window (before any
        StrategyVersion or bar is touched) -> register_version -> record_opening if the check
        said so -> the replay.
    asof replay run ... [--open-holdout]: HoldoutError -> exit 1 with the message, no traceback.

World: tests/scan/world.py + tests/replay/scenario.py (provenance there); the replay covers
START 2026-02-12 .. END 2026-03-25, which lies inside the real holdout (2025-07-01 ..
2026-06-30). Tests that need another holdout build the protocol in code
(tests/evolve/protocols.py) instead of writing 2025 bars.
"""

import re
from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import pytest
from sqlmodel import Session, select
from typer.testing import CliRunner

import asof.replay.run as replay_module
from asof.cli import app
from asof.evolve.holdout import HoldoutError
from asof.replay import run_replay
from asof.replay.report import build_report
from asof.store.models import Decision, HoldoutOpening, Run, StrategyVersion, Trade
from asof.strategy import attempt_count
from tests.evolve.protocols import covering, sealed_after
from tests.replay.scenario import END, START, TRADED, X
from tests.scan.world import gap_world
from tests.strategy.files import DEFAULTS, write_strategy

ZERO_SLIP = ("execute.fills.slippage_bps=0",)
NEW = "gap_v2"
# The scenario's whole store (2026-01-05 .. 2026-06-30) is holdout under this protocol.
SEALED = covering(date(2026, 1, 1), date(2026, 6, 30))

runner = CliRunner()


@pytest.fixture
def new_strategy(tmp_path: Path) -> Path:
    """The earnings_gap defaults under a name that never saw the holdout."""
    return write_strategy(
        tmp_path / "strategies", {"strategy": {**DEFAULTS["strategy"], "name": NEW}}, name=NEW
    )


@pytest.fixture
def old_strategy(tmp_path: Path) -> Path:
    return write_strategy(tmp_path / "strategies")


def all_rows(session: Session, model: Any) -> list[Any]:
    session.expire_all()
    return list(session.exec(select(model)).all())


# =============================================================================================
# normal path
# =============================================================================================


def test_a_strategy_seen_before_the_protocol_replays_the_holdout_with_the_default_protocol(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, old_strategy: Path
) -> None:
    # Every existing replay test runs earnings_gap on 2026 dates: allowed only by the exemption.
    world = gap_world(tmp_path, session, after=TRADED)

    run_id = run_replay(session, con, world.root, old_strategy, START, END, ZERO_SLIP)

    assert session.get(Run, run_id) is not None
    assert all_rows(session, HoldoutOpening) == []


def test_a_new_strategy_replays_dev_dates_without_the_flag_and_records_nothing(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, new_strategy: Path
) -> None:
    world = gap_world(tmp_path, session, after=TRADED)

    run_id = run_replay(
        session, con, world.root, new_strategy, START, END, ZERO_SLIP, protocol=sealed_after(END)
    )

    assert session.get(Run, run_id) is not None
    assert len(all_rows(session, Trade)) == 2
    assert all_rows(session, HoldoutOpening) == []


def test_the_flag_outside_the_holdout_records_nothing(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, new_strategy: Path
) -> None:
    world = gap_world(tmp_path, session, after=TRADED)

    run_replay(
        session,
        con,
        world.root,
        new_strategy,
        START,
        END,
        ZERO_SLIP,
        protocol=sealed_after(END),
        open_holdout=True,
    )

    assert all_rows(session, HoldoutOpening) == []


def test_opening_the_holdout_runs_and_records_the_version_and_window(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, new_strategy: Path
) -> None:
    world = gap_world(tmp_path, session, after=TRADED)

    run_id = run_replay(
        session,
        con,
        world.root,
        new_strategy,
        START,
        END,
        ZERO_SLIP,
        protocol=SEALED,
        open_holdout=True,
    )

    replay = session.get(Run, run_id)
    assert replay is not None
    [opening] = all_rows(session, HoldoutOpening)
    assert opening.strategy_id == NEW
    assert opening.strategy_version_id == replay.strategy_version_id
    assert (opening.window_start, opening.window_end) == (START, END)
    assert len(all_rows(session, Trade)) == 2  # the run itself is the normal replay


# =============================================================================================
# boundaries
# =============================================================================================


def test_without_the_flag_the_replay_is_refused_and_writes_nothing(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, new_strategy: Path
) -> None:
    world = gap_world(tmp_path, session, after=TRADED)

    with pytest.raises(HoldoutError, match="--open-holdout"):
        run_replay(session, con, world.root, new_strategy, START, END, ZERO_SLIP, protocol=SEALED)

    assert attempt_count(session, NEW) == 0
    assert all_rows(session, StrategyVersion) == []
    assert all_rows(session, Run) == []
    assert all_rows(session, HoldoutOpening) == []


@pytest.mark.parametrize(
    ("holdout_start", "holdout_end"),
    [(END, date(2026, 6, 30)), (date(2026, 1, 1), START)],
    ids=["holdout_starts_on_the_last_replay_day", "holdout_ends_on_the_first_replay_day"],
)
def test_one_shared_day_is_enough_to_refuse(
    tmp_path: Path,
    session: Session,
    con: duckdb.DuckDBPyConnection,
    new_strategy: Path,
    holdout_start: date,
    holdout_end: date,
) -> None:
    world = gap_world(tmp_path, session, after=TRADED)

    with pytest.raises(HoldoutError):
        run_replay(
            session,
            con,
            world.root,
            new_strategy,
            START,
            END,
            ZERO_SLIP,
            protocol=covering(holdout_start, holdout_end),
        )
    assert all_rows(session, Run) == []


def test_a_second_opening_through_another_parameter_set_is_refused_before_registering(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, new_strategy: Path
) -> None:
    world = gap_world(tmp_path, session, after=TRADED)
    run_replay(
        session,
        con,
        world.root,
        new_strategy,
        START,
        END,
        ZERO_SLIP,
        protocol=SEALED,
        open_holdout=True,
    )
    attempts = attempt_count(session, NEW)

    with pytest.raises(HoldoutError):
        run_replay(
            session,
            con,
            world.root,
            new_strategy,
            START,
            END,
            ("execute.fills.slippage_bps=0", "execute.stop.trail_atr=1.0"),
            protocol=SEALED,
            open_holdout=True,
        )

    assert attempt_count(session, NEW) == attempts
    assert len(all_rows(session, Run)) == 1
    assert len(all_rows(session, HoldoutOpening)) == 1


# =============================================================================================
# leakage: holdout bars are in the store, a refused replay must not read any of them
# =============================================================================================


def test_a_refused_replay_reads_no_bar_and_runs_no_scan(
    tmp_path: Path,
    session: Session,
    con: duckdb.DuckDBPyConnection,
    new_strategy: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The holdout starts on X[1], the first traded day: its bars exist in the store, and the
    # replay window reaches into it. Refusing after a look would already have leaked.
    world = gap_world(tmp_path, session, after=TRADED)
    touched: list[str] = []

    def guard(name: str) -> Any:
        def refuse(*args: Any, **kwargs: Any) -> Any:
            touched.append(name)
            raise AssertionError(f"a refused replay called {name}")

        return refuse

    for name in ("visible_sessions", "visible_master", "visible_splits", "scan", "BarCache"):
        monkeypatch.setattr(replay_module, name, guard(name))

    with pytest.raises(HoldoutError):
        run_replay(
            session,
            con,
            world.root,
            new_strategy,
            START,
            END,
            ZERO_SLIP,
            protocol=covering(X[1], date(2026, 6, 30)),
        )

    assert touched == []
    assert all_rows(session, Decision) == []
    assert all_rows(session, Trade) == []


# =============================================================================================
# CLI: the real default protocol, whose holdout covers the scenario's dates
# =============================================================================================


def cli_args(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "replay", "run", "--strategy", NEW,
        "--start", START.isoformat(), "--end", END.isoformat(),
        "--strategies-dir", str(tmp_path / "strategies"),
        "--data-dir", str(tmp_path), "--db", str(tmp_path / "meta.db"),
        "--set", "execute.fills.slippage_bps=0",
        *extra,
    ]  # fmt: skip


def test_cli_refuses_with_one_readable_line_and_writes_nothing(
    tmp_path: Path, session: Session, new_strategy: Path
) -> None:
    gap_world(tmp_path, session, after=TRADED)

    result = runner.invoke(app, cli_args(tmp_path))

    assert result.exit_code == 1, result.output
    assert isinstance(result.exception, SystemExit)  # no traceback
    assert "--open-holdout" in result.output
    assert "2025-07-01" in result.output
    assert "parameter set is registered" not in result.output
    assert all_rows(session, Run) == []
    assert attempt_count(session, NEW) == 0


def test_cli_opens_the_holdout_with_the_flag(
    tmp_path: Path, session: Session, new_strategy: Path
) -> None:
    gap_world(tmp_path, session, after=TRADED)

    result = runner.invoke(app, cli_args(tmp_path, "--open-holdout"))

    assert result.exit_code == 0, result.output
    match = re.search(r"run (\d+)", result.output)
    assert match, result.output
    [opening] = all_rows(session, HoldoutOpening)
    assert opening.strategy_id == NEW

    again = runner.invoke(app, cli_args(tmp_path, "--open-holdout"))
    assert again.exit_code == 1, again.output
    assert isinstance(again.exception, SystemExit)
    assert len(all_rows(session, Run)) == 1


# =============================================================================================
# review fixes: identity, failed openings, the report
# =============================================================================================


def test_overriding_the_strategy_name_is_refused_before_anything_is_written(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, new_strategy: Path
) -> None:
    # Borrowing the exempt name would replay the holdout unguarded and file the attempt under it.
    world = gap_world(tmp_path, session, after=TRADED)
    with pytest.raises(ValueError, match="strategy section"):
        run_replay(
            session,
            con,
            world.root,
            new_strategy,
            START,
            END,
            (*ZERO_SLIP, "strategy.name=earnings_gap"),
            protocol=SEALED,
        )
    assert all_rows(session, StrategyVersion) == []
    assert all_rows(session, Run) == []


def test_a_directory_whose_name_differs_from_strategy_name_is_refused(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection
) -> None:
    # A copied directory still saying "earnings_gap" inside would inherit the exemption.
    renamed = write_strategy(tmp_path / "strategies", name="renamed")
    world = gap_world(tmp_path, session, after=TRADED)
    with pytest.raises(ValueError, match="directory name"):
        run_replay(session, con, world.root, renamed, START, END, ZERO_SLIP, protocol=SEALED)
    assert all_rows(session, Run) == []


def test_a_replay_that_fails_after_opening_leaves_no_opening_and_can_open_again(
    tmp_path: Path,
    session: Session,
    con: duckdb.DuckDBPyConnection,
    new_strategy: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A failed replay shows no result, so it must not use up the strategy's one opening.
    world = gap_world(tmp_path, session, after=TRADED)
    real = replay_module.BarCache

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("disk gone")

    monkeypatch.setattr(replay_module, "BarCache", broken)
    with pytest.raises(RuntimeError):
        run_replay(
            session,
            con,
            world.root,
            new_strategy,
            START,
            END,
            ZERO_SLIP,
            protocol=SEALED,
            open_holdout=True,
        )
    session.rollback()  # what the CLI's closing session does
    assert all_rows(session, HoldoutOpening) == []
    assert all_rows(session, Run) == []

    monkeypatch.setattr(replay_module, "BarCache", real)
    run_id = run_replay(
        session,
        con,
        world.root,
        new_strategy,
        START,
        END,
        ZERO_SLIP,
        protocol=SEALED,
        open_holdout=True,
    )
    [opening] = all_rows(session, HoldoutOpening)
    assert opening.run_id == run_id


def test_the_report_says_when_holdout_dates_are_not_unseen(
    tmp_path: Path,
    session: Session,
    con: duckdb.DuckDBPyConnection,
    old_strategy: Path,
    new_strategy: Path,
) -> None:
    world = gap_world(tmp_path, session, after=TRADED)
    exempt = covering(date(2026, 1, 1), date(2026, 6, 30), seen=["earnings_gap"])
    old_run = run_replay(session, con, world.root, old_strategy, START, END, ZERO_SLIP)
    new_run = run_replay(
        session,
        con,
        world.root,
        new_strategy,
        START,
        END,
        ZERO_SLIP,
        protocol=SEALED,
        open_holdout=True,
    )
    assert "before the evaluation protocol existed" in build_report(
        session, old_run, protocol=exempt
    )
    assert "one opening" in build_report(session, new_run, protocol=SEALED)
    assert "Holdout:" not in build_report(session, old_run, protocol=sealed_after(END))
