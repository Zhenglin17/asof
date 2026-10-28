"""Guards against the spin-off wait being a number hidden in code: how many parent sessions a held
leg may stay frozen waiting for a spin-off's factor is a strategy parameter
(``execute.data.spinoff_max_wait``), so changing it is a new StrategyVersion and an attempt.

API under test (asof.strategy): DataConfig gains ``spinoff_max_wait: int``; the repository's
strategies/earnings_gap/execute.yaml sets it to 5 (the ex-date session is wait session 0).

No as_of: config loading reads no market information. Assumption to check on real data: 5
sessions cover the real lags between an ex-date and the first child bar of priced events
(store probe 2026-10-27: 2-13 sessions for the listed children of tests/store/test_spinoffs.py;
LEN -> MRP took 13, so it ends unpriced under 5). Check: for every adjust row of spinoff_events
at today's as_of, count the parent sessions from ex_date to known_on; the share above 5 is what
the replay treats as restart.
"""

import copy
from pathlib import Path

import pytest

from asof.strategy import apply_overrides, load_strategy
from tests.strategy.files import DEFAULTS, write_strategy

REPO_STRATEGY = Path(__file__).resolve().parents[2] / "strategies" / "earnings_gap"


def with_wait(tmp_path: Path, wait: object) -> Path:
    execute = copy.deepcopy(DEFAULTS["execute"])
    execute["data"] = {**execute["data"], "spinoff_max_wait": wait}
    return write_strategy(tmp_path / "strategies", {"execute": execute})


def test_the_repository_strategy_waits_five_sessions() -> None:
    assert load_strategy(REPO_STRATEGY).execute.data.spinoff_max_wait == 5


def test_the_wait_is_read_and_overridable(tmp_path: Path) -> None:
    cfg = load_strategy(with_wait(tmp_path, 3))
    assert cfg.execute.data.spinoff_max_wait == 3
    changed = apply_overrides(cfg, ["execute.data.spinoff_max_wait=0"])
    assert changed.execute.data.spinoff_max_wait == 0


def test_a_non_integer_wait_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_strategy(with_wait(tmp_path, "soon"))
