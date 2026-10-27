"""Guards against a pre-registered evaluation protocol that is silently wrong: overlapping or
reversed windows, a holdout that runs into the clean period, a significance level or attempt
budget that makes the gate meaningless, unknown keys that would be ignored instead of refused,
and a default path that depends on the working directory.

API under test (asof.evolve.protocol):

    Window(start: date, end: date); ThinEdge(drop_top: int, extra_cost_bps: float);
    Basket(size: int, lookback_sessions: int, rebalance: Literal["monthly"]);
    Benchmarks(symbols: list[str], basket: Basket);
    Comparison(risk_free_rate: float >= 0, high_exposure: float in (0, 1]);
    WalkForward(train_months: int >= 1, test_months: int >= 1, min_positive: int >= 1);
    EvaluationProtocol(dev, holdout, clean_after, seen_before_protocol, attempt_budget, alpha,
                       permutations, thin_edge, benchmarks, comparison, walk_forward,
                       mde_power: float in (0, 1))
                       -- every key required, extra keys forbidden everywhere
    load_protocol(path: Path) -> EvaluationProtocol
    DEFAULT_PROTOCOL_PATH: <repo root>/configs/evaluation.yaml, resolved from the package file

No as_of here: the protocol is a static config, so there is no leakage case. These assumptions
cannot be checked by fixtures and must be checked on the real store:
- ``earnings_gap`` really was replayed over holdout dates before the protocol existed (that is
  what justifies exempting it). Check:
  sqlite3 $ASOF_DATA_DIR/meta.db "select strategy_id, min(started_at), max(as_of) from run
  group by strategy_id" -> earnings_gap rows with as_of >= 2025-07-01 started before the
  protocol commit; no other strategy_id has a run with as_of >= 2025-07-01.
- Holdout and dev edges sit on the real exchange calendar as intended: 2025-06-30 and
  2025-07-01 are both sessions, 2026-06-30 is a session and the first session after it is
  2026-07-01 (visible_sessions on the real market store), so no session falls between windows.
- The committed configs/evaluation.yaml is the protocol that was registered: its holdout
  2025-07-01 .. 2026-06-30 is asserted below, and `git log configs/evaluation.yaml` shows one
  commit before any holdout opening.
"""

from datetime import date
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from asof.evolve import protocol as protocol_module
from asof.evolve.protocol import (
    DEFAULT_PROTOCOL_PATH,
    EvaluationProtocol,
    Window,
    load_protocol,
)
from tests.evolve.protocols import REAL, protocol, raw

REPO_ROOT = Path(__file__).resolve().parents[2]


def write(tmp_path: Path, content: dict[str, Any]) -> Path:
    path = tmp_path / "evaluation.yaml"
    path.write_text(yaml.safe_dump(content, sort_keys=False))
    return path


# =============================================================================================
# normal path
# =============================================================================================


def test_load_protocol_reads_every_field(tmp_path: Path) -> None:
    loaded = load_protocol(write(tmp_path, REAL))

    assert loaded.dev == Window(start=date(2020, 1, 1), end=date(2025, 6, 30))
    assert loaded.holdout == Window(start=date(2025, 7, 1), end=date(2026, 6, 30))
    assert loaded.clean_after == date(2026, 7, 1)
    assert loaded.seen_before_protocol == ["earnings_gap"]
    assert loaded.attempt_budget == 20
    assert loaded.alpha == pytest.approx(0.05)
    assert loaded.permutations == 1000
    assert loaded.thin_edge.drop_top == 3
    assert loaded.thin_edge.extra_cost_bps == pytest.approx(10.0)
    assert loaded.benchmarks.symbols == ["SPY", "QQQ"]
    basket = loaded.benchmarks.basket
    assert (basket.size, basket.lookback_sessions, basket.rebalance) == (10, 60, "monthly")
    assert loaded.comparison.risk_free_rate == 0.0
    assert loaded.comparison.high_exposure == pytest.approx(0.70)
    wf = loaded.walk_forward
    assert (wf.train_months, wf.test_months, wf.min_positive) == (24, 6, 4)
    assert loaded.mde_power == pytest.approx(0.80)


def test_dates_written_as_quoted_strings_still_load_as_dates(tmp_path: Path) -> None:
    path = tmp_path / "evaluation.yaml"
    text = yaml.safe_dump(REAL, sort_keys=False).replace("2025-07-01", "'2025-07-01'")
    path.write_text(text)

    assert load_protocol(path).holdout.start == date(2025, 7, 1)


def test_the_committed_default_file_parses_with_the_registered_holdout() -> None:
    loaded = load_protocol(DEFAULT_PROTOCOL_PATH)

    assert loaded.holdout == Window(start=date(2025, 7, 1), end=date(2026, 6, 30))
    assert loaded.dev.end < loaded.holdout.start
    assert loaded.clean_after == date(2026, 7, 1)
    assert "earnings_gap" in loaded.seen_before_protocol
    # The comparison block was appended after registration; its values are part of the protocol.
    assert loaded.comparison.risk_free_rate == 0.0
    assert loaded.comparison.high_exposure == pytest.approx(0.70)
    # So were the walk-forward windows and the power the MDE is reported at.
    wf = loaded.walk_forward
    assert (wf.train_months, wf.test_months, wf.min_positive) == (24, 6, 4)
    assert loaded.mde_power == pytest.approx(0.80)


def test_default_path_is_the_repo_config_not_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    assert DEFAULT_PROTOCOL_PATH.is_absolute()
    assert DEFAULT_PROTOCOL_PATH == REPO_ROOT / "configs" / "evaluation.yaml"
    assert DEFAULT_PROTOCOL_PATH.is_file()
    # Derived from the package file, so it would follow the code, not the shell.
    assert Path(protocol_module.__file__).resolve().is_relative_to(REPO_ROOT)


# =============================================================================================
# boundaries: windows
# =============================================================================================


def test_a_one_day_window_is_allowed() -> None:
    one = Window(start=date(2025, 7, 1), end=date(2025, 7, 1))
    assert one.start == one.end


def test_a_window_ending_before_it_starts_is_refused() -> None:
    with pytest.raises(ValidationError):
        Window(start=date(2025, 7, 2), end=date(2025, 7, 1))


def test_holdout_starting_the_day_after_dev_ends_is_allowed() -> None:
    p = protocol()
    assert (p.holdout.start - p.dev.end).days == 1


def test_dev_ending_on_the_holdout_start_is_refused() -> None:
    with pytest.raises(ValidationError):
        protocol(dev={"start": date(2020, 1, 1), "end": date(2025, 7, 1)})


def test_dev_after_the_holdout_is_refused() -> None:
    with pytest.raises(ValidationError):
        protocol(dev={"start": date(2026, 7, 1), "end": date(2026, 12, 31)})


def test_clean_period_starting_on_the_last_holdout_day_is_refused() -> None:
    with pytest.raises(ValidationError):
        protocol(clean_after=date(2026, 6, 30))


def test_clean_period_starting_the_day_after_the_holdout_is_allowed() -> None:
    assert protocol(clean_after=date(2026, 7, 1)).clean_after == date(2026, 7, 1)


# =============================================================================================
# boundaries: numbers
# =============================================================================================


@pytest.mark.parametrize(
    "changes",
    [
        {"attempt_budget": 0},
        {"alpha": 0.0},
        {"alpha": 1.0},
        {"alpha": -0.05},
        {"permutations": 99},
        {"thin_edge": {"drop_top": -1, "extra_cost_bps": 10}},
        {"thin_edge": {"drop_top": 3, "extra_cost_bps": -0.5}},
        {
            "benchmarks": {
                "symbols": ["SPY"],
                "basket": {"size": 0, "lookback_sessions": 60, "rebalance": "monthly"},
            }
        },
        {
            "benchmarks": {
                "symbols": ["SPY"],
                "basket": {"size": 10, "lookback_sessions": 0, "rebalance": "monthly"},
            }
        },
        {
            "benchmarks": {
                "symbols": ["SPY"],
                "basket": {"size": 10, "lookback_sessions": 60, "rebalance": "weekly"},
            }
        },
    ],
    ids=[
        "budget_0",
        "alpha_0",
        "alpha_1",
        "alpha_negative",
        "permutations_99",
        "drop_top_negative",
        "extra_cost_negative",
        "basket_size_0",
        "lookback_0",
        "rebalance_weekly",
    ],
)
def test_out_of_range_numbers_are_refused(changes: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        protocol(**changes)


def test_smallest_allowed_numbers_load() -> None:
    p = protocol(
        attempt_budget=1,
        permutations=100,
        thin_edge={"drop_top": 0, "extra_cost_bps": 0},
        benchmarks={
            "symbols": ["SPY"],
            "basket": {"size": 1, "lookback_sessions": 1, "rebalance": "monthly"},
        },
    )
    assert (p.attempt_budget, p.permutations, p.thin_edge.drop_top) == (1, 100, 0)
    assert p.benchmarks.basket.size == 1


def test_an_empty_exemption_list_is_allowed() -> None:
    assert protocol(seen_before_protocol=[]).seen_before_protocol == []


# =============================================================================================
# boundaries: unknown or missing keys
# =============================================================================================


def test_an_unknown_top_level_key_is_refused(tmp_path: Path) -> None:
    # A typo such as "holduot" must not leave the real holdout unguarded with defaults.
    with pytest.raises(ValidationError):
        load_protocol(write(tmp_path, raw(holduot={"start": "2025-07-01", "end": "2026-06-30"})))


def test_an_unknown_nested_key_is_refused() -> None:
    with pytest.raises(ValidationError):
        protocol(thin_edge={"drop_top": 3, "extra_cost_bps": 10, "drop_bottom": 1})
    with pytest.raises(ValidationError):
        protocol(holdout={"start": date(2025, 7, 1), "end": date(2026, 6, 30), "note": "x"})


def test_a_missing_key_is_refused() -> None:
    content = raw()
    del content["holdout"]
    with pytest.raises(ValidationError):
        EvaluationProtocol.model_validate(content)


@pytest.mark.parametrize(
    "comparison",
    [
        {"risk_free_rate": -0.01, "high_exposure": 0.70},
        {"risk_free_rate": 0.0, "high_exposure": 0.0},
        {"risk_free_rate": 0.0, "high_exposure": 1.01},
    ],
    ids=["risk_free_negative", "high_exposure_0", "high_exposure_above_1"],
)
def test_out_of_range_comparison_numbers_are_refused(comparison: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        protocol(comparison=comparison)


def test_comparison_edges_that_are_allowed_load() -> None:
    p = protocol(comparison={"risk_free_rate": 0.0, "high_exposure": 1.0})
    assert (p.comparison.risk_free_rate, p.comparison.high_exposure) == (0.0, 1.0)
    other = protocol(comparison={"risk_free_rate": 0.04, "high_exposure": 0.5})
    assert other.comparison.model_dump() == {"risk_free_rate": 0.04, "high_exposure": 0.5}


def test_a_missing_comparison_block_is_refused(tmp_path: Path) -> None:
    content = raw()
    del content["comparison"]
    with pytest.raises(ValidationError):
        load_protocol(write(tmp_path, content))


@pytest.mark.parametrize(
    "comparison",
    [
        {"risk_free_rate": 0.0},
        {"high_exposure": 0.70},
        {"risk_free_rate": 0.0, "high_exposur": 0.70},
        {"risk_free_rate": 0.0, "high_exposure": 0.70, "benchmark": "SPY"},
    ],
    ids=["no_high_exposure", "no_risk_free_rate", "misspelled", "unknown_key"],
)
def test_a_missing_or_misspelled_comparison_key_is_refused(comparison: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        protocol(comparison=comparison)


def test_benchmarks_need_at_least_one_symbol() -> None:
    # The first symbol is the SPY of the comparison; without one the basket would stand in.
    with pytest.raises(ValidationError):
        protocol(
            benchmarks={
                "symbols": [],
                "basket": {"size": 10, "lookback_sessions": 60, "rebalance": "monthly"},
            }
        )


def test_a_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_protocol(tmp_path / "nope.yaml")


# =============================================================================================
# boundaries: walk-forward and MDE power
# =============================================================================================


@pytest.mark.parametrize(
    "walk_forward",
    [
        {"train_months": 0, "test_months": 6, "min_positive": 4},
        {"train_months": 24, "test_months": 0, "min_positive": 4},
        {"train_months": 24, "test_months": 6, "min_positive": 0},
        {"train_months": 2.5, "test_months": 6, "min_positive": 4},
    ],
    ids=["train_0", "test_0", "min_positive_0", "train_fractional"],
)
def test_out_of_range_walk_forward_numbers_are_refused(walk_forward: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        protocol(walk_forward=walk_forward)


@pytest.mark.parametrize("power", [0.0, 1.0, -0.2, 1.2], ids=["0", "1", "negative", "above_1"])
def test_an_mde_power_outside_zero_one_is_refused(power: float) -> None:
    with pytest.raises(ValidationError):
        protocol(mde_power=power)


def test_smallest_walk_forward_and_an_mde_power_inside_zero_one_load() -> None:
    p = protocol(walk_forward={"train_months": 1, "test_months": 1, "min_positive": 1},
                 mde_power=0.5)  # fmt: skip
    assert p.walk_forward.model_dump() == {"train_months": 1, "test_months": 1, "min_positive": 1}
    assert p.mde_power == 0.5


@pytest.mark.parametrize("key", ["walk_forward", "mde_power"])
def test_a_missing_walk_forward_or_mde_power_is_refused(tmp_path: Path, key: str) -> None:
    content = raw()
    del content[key]
    with pytest.raises(ValidationError):
        load_protocol(write(tmp_path, content))


@pytest.mark.parametrize(
    "walk_forward",
    [
        {"train_months": 24, "test_months": 6},
        {"train_months": 24, "min_positive": 4},
        {"test_months": 6, "min_positive": 4},
        {"train_month": 24, "test_months": 6, "min_positive": 4},
        {"train_months": 24, "test_months": 6, "min_positive": 4, "step_months": 6},
    ],
    ids=["no_min_positive", "no_test_months", "no_train_months", "misspelled", "unknown_key"],
)
def test_a_missing_or_misspelled_walk_forward_key_is_refused(walk_forward: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        protocol(walk_forward=walk_forward)


def test_a_misspelled_mde_power_is_refused(tmp_path: Path) -> None:
    content = raw()
    content["mde_powr"] = content.pop("mde_power")
    with pytest.raises(ValidationError):
        load_protocol(write(tmp_path, content))
