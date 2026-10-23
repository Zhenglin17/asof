"""Guards against `asof judge run` spending money it was not allowed to or hiding what it did: a
missing API key that still sends a request, a cap that ends the run with exit code 0 (a script
would not notice), --set overrides that do not reach the judge, a run that judges with take_all,
and output that does not say how many calls were paid for and what they cost.

CLI under test (asof.cli):

    asof judge run --strategy NAME --start YYYY-MM-DD --end YYYY-MM-DD --max-cost USD
                   [--limit N] [--set key=value ...]
                   [--strategies-dir DIR] [--data-dir DIR] [--db FILE]
        forces judge.kind=llm, reads OPENROUTER_API_KEY (missing: exit non-zero, one line,
        nothing sent), fetches through OpenRouterClient(key).complete into
        <market root>/llm_cache, prints one count per verdict (take, pass, no_call), the fresh
        and cached call counts and the spend, each as "<label> ... <number>" on its own line,
        e.g. "take 1", "fresh calls 2", "spent $0.0785"; exit 0, or exit 1 if the cap stopped it.

Strategy options follow `asof replay run` (--strategy NAME under --strategies-dir) so one set of
paths serves both commands. No as_of here: the command forwards --start/--end to judge_window,
whose leakage is tested in tests/judge/test_window.py. The model is a scripted
OpenRouterClient.complete (replies: tests/judge/fakes.py, provenance there); httpx is guarded.
"""

import re
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlmodel import Session
from typer.testing import CliRunner

from asof.cli import app
from asof.judge import client
from tests.judge.fakes import LLM_JUDGE, Script, answer, claim, costed
from tests.replay.scenario import END, START, X
from tests.replay.test_replay_llm import good_answer
from tests.replay.two_gaps import TwoGaps, two_gaps
from tests.strategy.files import write_strategy

runner = CliRunner()
KEY = "sk-or-test-key"


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def send(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("asof judge run used the network")

    monkeypatch.setattr(httpx.Client, "send", send)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


@pytest.fixture
def world(tmp_path: Path, session: Session) -> TwoGaps:
    return two_gaps(tmp_path, session)


@pytest.fixture
def paths(tmp_path: Path, world: TwoGaps) -> dict[str, str]:
    # judge.yaml as committed: take_all, every llm field present. The command must force llm.
    strategies = write_strategy(
        tmp_path / "strategies", {"judge": {**LLM_JUDGE, "kind": "take_all"}}
    ).parent
    return {
        "--strategies-dir": str(strategies),
        "--data-dir": str(tmp_path),
        "--db": str(tmp_path / "meta.db"),
    }


class Model:
    """The scripted model: ``script`` answers complete(); ``keys`` records every key a client was
    built with."""

    def __init__(self) -> None:
        self.script = Script()
        self.keys: list[str] = []


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> Model:
    fake = Model()
    original = client.OpenRouterClient.__init__

    def init(self: Any, api_key: str, **kwargs: Any) -> None:
        fake.keys.append(api_key)
        original(self, api_key, **kwargs)

    def complete(self: Any, body: dict[str, Any]) -> dict[str, Any]:
        return fake.script(body)

    monkeypatch.setattr(client.OpenRouterClient, "__init__", init)
    monkeypatch.setattr(client.OpenRouterClient, "complete", complete)
    return fake


def judge_args(paths: dict[str, str], *extra: str, max_cost: str = "5") -> list[str]:
    args = ["judge", "run", "--strategy", "earnings_gap"]
    args += ["--start", START.isoformat(), "--end", END.isoformat(), "--max-cost", max_cost]
    for option, value in paths.items():
        args += [option, value]
    return args + list(extra)


def number(output: str, label: str) -> float:
    match = re.search(rf"^\s*{label}\b[^\d\n]*(\d+(?:\.\d+)?)", output, re.MULTILINE)
    assert match, f"no {label!r} line in:\n{output}"
    return float(match.group(1))


def cache_files(tmp_path: Path) -> list[Path]:
    return sorted((tmp_path / "market" / "llm_cache").rglob("*.json"))


def replies(world: TwoGaps, cost: float = 0.04852) -> Script:
    return Script(
        costed(good_answer(world.first_chunks, "take", 0.8), cost),
        costed(good_answer(world.second_chunks, "pass", 0.6), 0.03),
    )


# =============================================================================================
# normal path
# =============================================================================================


def test_judge_run_fills_the_cache_that_an_offline_llm_replay_then_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    paths: dict[str, str],
    world: TwoGaps,
    model: Model,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
    script = model.script = replies(world)

    result = runner.invoke(app, judge_args(paths))

    assert result.exit_code == 0, result.output
    assert len(script.bodies) == 2  # the strategy says take_all; the command judged with llm
    assert model.keys == [KEY]
    assert number(result.output, "take") == 1
    assert number(result.output, "pass") == 1
    assert number(result.output, "no_call") == 0
    assert number(result.output, "fresh") == 2
    assert number(result.output, "cached") == 0
    assert number(result.output, "spent") == pytest.approx(0.04852 + 0.03, abs=0.005)
    assert len(cache_files(tmp_path)) == 2

    replay = ["replay", "run", "--strategy", "earnings_gap"]
    replay += ["--start", START.isoformat(), "--end", END.isoformat()]
    replay += ["--set", "judge.kind=llm"]
    for option, value in paths.items():
        replay += [option, value]
    replayed = runner.invoke(app, replay)
    assert replayed.exit_code == 0, replayed.output
    assert len(script.bodies) == 2  # the replay asked the model nothing


def test_a_second_judge_run_reads_the_cache_and_pays_nothing(
    monkeypatch: pytest.MonkeyPatch, paths: dict[str, str], world: TwoGaps, model: Model
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
    model.script = replies(world)
    assert runner.invoke(app, judge_args(paths)).exit_code == 0
    model.script = Script()  # any call now fails the test

    result = runner.invoke(app, judge_args(paths))

    assert result.exit_code == 0, result.output
    assert number(result.output, "fresh") == 0
    assert number(result.output, "cached") == 2
    assert number(result.output, "spent") == 0


# =============================================================================================
# boundaries
# =============================================================================================


def test_set_overrides_reach_the_judge(
    monkeypatch: pytest.MonkeyPatch, paths: dict[str, str], world: TwoGaps, model: Model
) -> None:
    # With no retries a rejected answer is a no_call after one call; with the default 2 the
    # script would run out and fail the test.
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
    bad = answer("take", 0.9, [claim(world.first_chunks[0], "not in the filing at all", "x")])
    script = model.script = Script(costed(bad, 0.01))
    args = judge_args(paths, "--set", "judge.max_retries=0")
    args[args.index("--end") + 1] = X[0].isoformat()  # GAPX on D only

    result = runner.invoke(app, args)

    assert result.exit_code == 0, result.output
    assert len(script.bodies) == 1
    assert number(result.output, "no_call") == 1


def test_the_cap_stops_the_run_with_exit_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    paths: dict[str, str],
    world: TwoGaps,
    model: Model,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
    script = model.script = replies(world, cost=0.05)

    result = runner.invoke(app, judge_args(paths, max_cost="0.04"))

    assert result.exit_code == 1, result.output
    assert len(script.bodies) == 1  # X6 was never asked
    assert number(result.output, "fresh") == 1
    assert number(result.output, "spent") == pytest.approx(0.05, abs=0.005)
    assert len(cache_files(tmp_path)) == 1  # what was paid for is kept


def test_limit_judges_the_first_candidates_and_is_not_a_failure(
    monkeypatch: pytest.MonkeyPatch, paths: dict[str, str], world: TwoGaps, model: Model
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
    script = model.script = replies(world)

    result = runner.invoke(app, judge_args(paths, "--limit", "1"))

    assert result.exit_code == 0, result.output
    assert len(script.bodies) == 1
    assert number(result.output, "take") == 1


def test_a_failed_call_exits_one_and_still_prints_the_spend(
    monkeypatch: pytest.MonkeyPatch, paths: dict[str, str], world: TwoGaps, model: Model
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
    first = costed(good_answer(world.first_chunks, "take", 0.8), 0.05)
    calls: list[dict[str, Any]] = []

    def complete(self: Any, body: dict[str, Any]) -> dict[str, Any]:
        calls.append(body)
        if len(calls) == 1:
            return first
        raise client.JudgeCallError("OpenRouter HTTP 502: bad gateway")

    monkeypatch.setattr(client.OpenRouterClient, "complete", complete)

    result = runner.invoke(app, judge_args(paths))

    assert result.exit_code == 1, result.output
    assert number(result.output, "fresh") == 1
    assert number(result.output, "spent") == pytest.approx(0.05, abs=0.005)
    assert "502" in result.output


def test_a_missing_api_key_exits_nonzero_and_sends_nothing(
    tmp_path: Path, paths: dict[str, str], world: TwoGaps, model: Model
) -> None:
    script = model.script = replies(world)

    result = runner.invoke(app, judge_args(paths))

    assert result.exit_code != 0
    assert isinstance(result.exception, SystemExit)
    assert "OPENROUTER_API_KEY" in result.output
    assert "Traceback" not in result.output
    assert script.bodies == []
    assert cache_files(tmp_path) == []


@pytest.mark.parametrize("key", ["", "   "], ids=["empty", "blank"])
def test_an_empty_api_key_is_a_missing_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    paths: dict[str, str],
    world: TwoGaps,
    model: Model,
    key: str,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", key)
    script = model.script = replies(world)

    result = runner.invoke(app, judge_args(paths))

    assert result.exit_code != 0
    assert isinstance(result.exception, SystemExit)
    assert "OPENROUTER_API_KEY" in result.output
    assert script.bodies == []
    assert cache_files(tmp_path) == []
