"""Guards against spending money on the sealed holdout by accident: a judge_window over holdout
dates that calls the model, writes the answer cache or even scans before refusing, a refusal the
CLI turns into "Judging stopped ... cached" (which reads as a partial success), and a judge run
that records a holdout opening (only the replay that uses the answers may record one).

API under test (asof.judge.window, asof.cli):

    judge_window(session, con, root, cfg, start, end, fetch, *, max_cost, limit=None,
                 protocol: EvaluationProtocol | None = None, open_holdout: bool = False)
        protocol None -> the default file. [start, end] overlapping the holdout raises
        HoldoutError unless open_holdout or cfg.strategy.name is in seen_before_protocol.
        Never writes HoldoutOpening.
    asof judge run ... [--open-holdout]: HoldoutError -> exit 1, the message, nothing sent.

World: tests/replay/two_gaps.py (GAPX gaps on X[0] and X[6], provenance there). The model is
``tests.judge.fakes.Script`` (reply shape measured, see fakes.py); httpx is guarded.
"""

from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import httpx
import pytest
from asof.evolve.holdout import HoldoutError
from sqlmodel import Session, select
from typer.testing import CliRunner

import asof.judge.window as window_module
from asof.cli import app
from asof.judge import client
from asof.judge.window import judge_window
from asof.store.models import HoldoutOpening, StrategyVersion
from tests.evolve.protocols import covering, sealed_after
from tests.judge.fakes import LLM_JUDGE, Script, costed
from tests.replay.scenario import END, START, X
from tests.replay.test_replay_llm import LLM, config, good_answer
from tests.replay.two_gaps import TwoGaps, two_gaps
from tests.strategy.files import DEFAULTS, write_strategy

NEW = "gap_v2"
SEALED = covering(date(2026, 1, 1), date(2026, 6, 30))

runner = CliRunner()


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def send(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the judge used the network")

    monkeypatch.setattr(httpx.Client, "send", send)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


@pytest.fixture
def new_strategy(tmp_path: Path) -> Path:
    return write_strategy(
        tmp_path / "strategies",
        {
            "strategy": {**DEFAULTS["strategy"], "name": NEW},
            "judge": {**LLM_JUDGE, "kind": "take_all"},
        },
        name=NEW,
    )


@pytest.fixture
def world(tmp_path: Path, session: Session) -> TwoGaps:
    return two_gaps(tmp_path, session)


def replies(world: TwoGaps) -> Script:
    return Script(
        costed(good_answer(world.first_chunks, "take", 0.8), 0.04852),
        costed(good_answer(world.second_chunks, "pass", 0.6), 0.03),
    )


def cache_files(root: Path) -> list[Path]:
    return sorted((root / "llm_cache").rglob("*.json"))


def openings(session: Session) -> list[HoldoutOpening]:
    session.expire_all()
    return list(session.exec(select(HoldoutOpening)).all())


# =============================================================================================
# normal path
# =============================================================================================


def test_dev_dates_are_judged_without_the_flag(
    session: Session, con: duckdb.DuckDBPyConnection, world: TwoGaps, new_strategy: Path
) -> None:
    cfg = config(new_strategy, LLM)
    result = judge_window(
        session, con, world.world.root, cfg, START, END, replies(world),
        max_cost=5.0, protocol=sealed_after(END),
    )  # fmt: skip

    assert len(result.outcomes) == 2
    assert openings(session) == []


def test_the_flag_judges_holdout_dates_and_records_no_opening(
    session: Session, con: duckdb.DuckDBPyConnection, world: TwoGaps, new_strategy: Path
) -> None:
    cfg = config(new_strategy, LLM)
    result = judge_window(
        session, con, world.world.root, cfg, START, END, replies(world),
        max_cost=5.0, protocol=SEALED, open_holdout=True,
    )  # fmt: skip

    assert len(result.outcomes) == 2
    assert openings(session) == []
    assert list(session.exec(select(StrategyVersion)).all()) == []


# =============================================================================================
# boundaries
# =============================================================================================


def test_holdout_dates_without_the_flag_are_refused_before_any_call(
    session: Session, con: duckdb.DuckDBPyConnection, world: TwoGaps, new_strategy: Path
) -> None:
    cfg = config(new_strategy, LLM)
    script = Script()  # any call fails the test

    with pytest.raises(HoldoutError, match="--open-holdout"):
        judge_window(
            session, con, world.world.root, cfg, START, END, script,
            max_cost=5.0, protocol=SEALED,
        )  # fmt: skip

    assert script.bodies == []
    assert cache_files(world.world.root) == []


def test_one_shared_day_at_the_window_end_is_refused(
    session: Session, con: duckdb.DuckDBPyConnection, world: TwoGaps, new_strategy: Path
) -> None:
    cfg = config(new_strategy, LLM)
    with pytest.raises(HoldoutError):
        judge_window(
            session, con, world.world.root, cfg, START, END, Script(),
            max_cost=5.0, protocol=covering(END, date(2026, 6, 30)),
        )  # fmt: skip


def test_a_strategy_seen_before_the_protocol_is_judged_without_the_flag(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, world: TwoGaps
) -> None:
    # The existing judge tests rely on this: earnings_gap on 2026 dates, default protocol.
    old = write_strategy(tmp_path / "old", {"judge": {**LLM_JUDGE, "kind": "take_all"}})
    result = judge_window(
        session, con, world.world.root, config(old, LLM), START, END, replies(world),
        max_cost=5.0,
    )  # fmt: skip

    assert len(result.outcomes) == 2
    assert openings(session) == []


# =============================================================================================
# leakage: holdout bars and filings are in the store; a refused window must not scan them
# =============================================================================================


def test_a_refused_window_reads_no_session_and_runs_no_scan(
    session: Session,
    con: duckdb.DuckDBPyConnection,
    world: TwoGaps,
    new_strategy: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    touched: list[str] = []

    def guard(name: str) -> Any:
        def refuse(*args: Any, **kwargs: Any) -> Any:
            touched.append(name)
            raise AssertionError(f"a refused window called {name}")

        return refuse

    for name in ("visible_sessions", "scan", "judge_candidate"):
        monkeypatch.setattr(window_module, name, guard(name))

    with pytest.raises(HoldoutError):
        judge_window(
            session, con, world.world.root, config(new_strategy, LLM), START, END, Script(),
            max_cost=5.0, protocol=covering(X[1], date(2026, 6, 30)),
        )  # fmt: skip

    assert touched == []


# =============================================================================================
# CLI: the real default protocol, whose holdout covers the two_gaps dates
# =============================================================================================


def test_cli_refuses_holdout_dates_without_sending_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, world: TwoGaps, new_strategy: Path
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")
    sent: list[dict[str, Any]] = []

    def complete(self: Any, body: dict[str, Any]) -> dict[str, Any]:
        sent.append(body)
        raise AssertionError("a refused judge run called the model")

    monkeypatch.setattr(client.OpenRouterClient, "complete", complete)
    args = [
        "judge", "run", "--strategy", NEW,
        "--start", START.isoformat(), "--end", END.isoformat(), "--max-cost", "5",
        "--strategies-dir", str(tmp_path / "strategies"),
        "--data-dir", str(tmp_path), "--db", str(tmp_path / "meta.db"),
    ]  # fmt: skip

    result = runner.invoke(app, args)

    assert result.exit_code == 1, result.output
    assert isinstance(result.exception, SystemExit)
    assert "--open-holdout" in result.output
    assert "cached" not in result.output  # not the "Judging stopped ... cached" line
    assert sent == []
    assert cache_files(world.world.root) == []
