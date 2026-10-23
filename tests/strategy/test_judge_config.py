"""Guards against an llm judge running on a config nobody can reproduce: a missing model, effort or
training cutoff silently defaulted, a conviction threshold or retry count out of range, or a typo
in judge.yaml accepted.

API under test (asof.strategy.JudgeConfig):

    kind: "take_all" | "llm"
    provider: "openrouter", model: str, pin_provider: str,
    effort: "low" | "medium" | "high" | "xhigh" | "max", max_retries: int >= 0,
    min_conviction: float in [0, 1], max_tokens: int > 0, training_cutoff: date,
    cutoff_source: str, prompt: str (non-empty)
        all required when kind == "llm"; allowed and ignored when kind == "take_all".

No leakage case: config loading has no as_of. Assumptions only real data can confirm:
- ``training_cutoff`` and ``cutoff_source`` in strategies/earnings_gap/judge.yaml match the
  pinned model's published model card (check by hand against the URL in cutoff_source).
- The ``effort`` value in judge.yaml is one the pinned provider accepts for that model: a
  cached real reply (``asof`` llm_cache file) must show reasoning tokens in its usage block.
- Replays never start before ``training_cutoff`` without being flagged: check the replay
  report of the first llm replay for its window versus the cutoff.
"""

import shutil
from datetime import date
from pathlib import Path
from typing import Any

import pytest
import yaml

from asof.strategy import apply_overrides, load_strategy
from tests.judge.fakes import LLM_FIELDS, LLM_JUDGE
from tests.strategy.files import write_strategy

REPO_STRATEGY = Path(__file__).resolve().parents[2] / "strategies" / "earnings_gap"


def repo_copy(tmp_path: Path, **judge_changes: Any) -> Path:
    """The committed strategy directory copied to tmp_path, judge.yaml edited; a value of None
    removes the key."""
    directory = tmp_path / "earnings_gap"
    shutil.copytree(REPO_STRATEGY, directory)
    path = directory / "judge.yaml"
    data = yaml.safe_load(path.read_text())
    for key, value in judge_changes.items():
        if value is None:
            assert key in data, f"judge.yaml has no {key!r} to remove"
            del data[key]
        else:
            data[key] = value
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return directory


# =============================================================================================
# normal path
# =============================================================================================


def test_repository_judge_defaults_to_take_all_and_carries_the_llm_fields() -> None:
    cfg = load_strategy(REPO_STRATEGY)
    assert cfg.judge.kind == "take_all"
    raw = yaml.safe_load((REPO_STRATEGY / "judge.yaml").read_text())
    assert set(LLM_FIELDS) <= set(raw), sorted(set(LLM_FIELDS) - set(raw))


def test_override_switches_the_repository_judge_to_llm() -> None:
    cfg = apply_overrides(load_strategy(REPO_STRATEGY), ["judge.kind=llm"])
    judge = cfg.judge
    assert judge.kind == "llm"
    assert judge.provider == "openrouter"
    assert judge.model
    assert judge.pin_provider
    assert judge.effort in {"low", "medium", "high", "xhigh", "max"}
    assert judge.max_retries is not None
    assert judge.min_conviction is not None
    assert judge.max_tokens is not None
    assert judge.prompt is not None
    assert judge.max_retries >= 0
    assert 0 <= judge.min_conviction <= 1
    assert judge.max_tokens > 0
    assert isinstance(judge.training_cutoff, date)
    assert judge.cutoff_source
    assert judge.prompt.strip()


def test_llm_fields_load_from_a_written_strategy(tmp_path: Path) -> None:
    cfg = load_strategy(write_strategy(tmp_path, {"judge": LLM_JUDGE}))
    assert cfg.judge.kind == "llm"
    assert cfg.judge.model == LLM_FIELDS["model"]
    assert cfg.judge.training_cutoff == date(2025, 1, 31)
    assert cfg.judge.min_conviction == 0.5


def test_take_all_with_llm_fields_present_still_loads(tmp_path: Path) -> None:
    cfg = load_strategy(write_strategy(tmp_path, {"judge": {**LLM_JUDGE, "kind": "take_all"}}))
    assert cfg.judge.kind == "take_all"


def test_take_all_without_llm_fields_still_loads(tmp_path: Path) -> None:
    cfg = load_strategy(write_strategy(tmp_path, {"judge": {"kind": "take_all"}}))
    assert cfg.judge.kind == "take_all"


# =============================================================================================
# boundaries: missing and out-of-range values
# =============================================================================================


@pytest.mark.parametrize("missing", sorted(LLM_FIELDS))
def test_llm_judge_missing_a_field_is_rejected(tmp_path: Path, missing: str) -> None:
    # Control: the same copy with every field loads as llm.
    assert load_strategy(repo_copy(tmp_path / "full", kind="llm")).judge.kind == "llm"
    with pytest.raises(ValueError):
        load_strategy(repo_copy(tmp_path / "cut", kind="llm", **{missing: None}))


def test_llm_judge_without_a_model_is_rejected(tmp_path: Path) -> None:
    # The case named in the spec, kept on its own so its failure reads plainly.
    assert load_strategy(repo_copy(tmp_path / "full", kind="llm")).judge.model
    with pytest.raises(ValueError):
        load_strategy(repo_copy(tmp_path / "cut", kind="llm", model=None))


@pytest.mark.parametrize(
    "override",
    [
        "judge.min_conviction=1.5",
        "judge.min_conviction=-0.1",
        "judge.max_retries=-1",
        "judge.max_retries=1.5",
        "judge.max_tokens=0",
        "judge.effort=extreme",
        "judge.provider=openai",
        "judge.prompt=''",
        "judge.kind=gpt",
        "judge.training_cutoff=soon",
    ],
)
def test_llm_judge_value_out_of_range_is_rejected(tmp_path: Path, override: str) -> None:
    base = load_strategy(write_strategy(tmp_path, {"judge": LLM_JUDGE}))
    with pytest.raises(ValueError):
        apply_overrides(base, [override])


@pytest.mark.parametrize(
    ("override", "field", "value"),
    [
        ("judge.min_conviction=0", "min_conviction", 0.0),
        ("judge.min_conviction=1", "min_conviction", 1.0),
        ("judge.max_retries=0", "max_retries", 0),
        ("judge.max_tokens=1", "max_tokens", 1),
        ("judge.effort=xhigh", "effort", "xhigh"),
        ("judge.effort=max", "effort", "max"),
    ],
)
def test_llm_judge_edge_values_are_accepted(
    tmp_path: Path, override: str, field: str, value: Any
) -> None:
    base = load_strategy(write_strategy(tmp_path, {"judge": LLM_JUDGE}))
    assert getattr(apply_overrides(base, [override]).judge, field) == value


def test_unknown_judge_key_is_still_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_strategy(write_strategy(tmp_path, {"judge": {**LLM_JUDGE, "temperature": 0}}))
    base = load_strategy(write_strategy(tmp_path / "ok", {"judge": LLM_JUDGE}))
    with pytest.raises(ValueError):
        apply_overrides(base, ["judge.temperature=0"])
