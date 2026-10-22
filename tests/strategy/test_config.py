"""Guards against a strategy running under a config nobody can identify later: a typo'd key that
is silently ignored, an override that mutates the base config, a content hash that changes with
file mtimes or override order (so one parameter set counts as two attempts), or a version table
that undercounts attempts and so loosens the attempts-adjusted gate.

API under test (asof.strategy):

    load_strategy(path) -> StrategyConfig          sections .strategy .scan .rules .judge
                                                   .execute .attribution, Pydantic, extra=forbid
    apply_overrides(cfg, ["execute.stop.trail_atr=1.0", ...]) -> StrategyConfig
    content_hash(path, overrides=()) -> str
    register_version(session, path, overrides=(), proposed_by="human") -> StrategyVersion
    attempt_count(session, strategy_id) -> int

No leakage case: config loading has no as_of (a strategy directory is not market information).
Assumptions that only real data can confirm -- check them on the real repository, not here:
- The repository's ``strategies/earnings_gap`` holds exactly the six YAML files and nothing the
  hash should ignore; ``content_hash`` of the committed directory is the hash of the approved
  StrategyVersion row (``SELECT content_hash FROM strategy_version WHERE strategy_id =
  'earnings_gap' AND status = 'approved'`` equals ``content_hash(Path('strategies/earnings_gap'))``
  after the first registration).
- ``attempt_count`` after a real replay sweep equals the number of distinct ``--set`` parameter
  sets tried plus one (the base), compared against the sweep log.
"""

import json
import os
import shutil
from pathlib import Path

import pytest
from sqlmodel import Session, col, select

from asof.store.models import StrategyVersion
from asof.strategy import (
    apply_overrides,
    attempt_count,
    content_hash,
    load_strategy,
    register_version,
)
from tests.strategy.files import DEFAULTS, FILES, write_strategy

REPO_STRATEGY = Path(__file__).resolve().parents[2] / "strategies" / "earnings_gap"


def dumped(model: object) -> object:
    """A section as plain JSON values, so tuples compare equal to the YAML lists."""
    return json.loads(model.model_dump_json())  # type: ignore[attr-defined]


@pytest.fixture
def strategy_dir(tmp_path: Path) -> Path:
    return write_strategy(tmp_path / "strategies")


# =============================================================================================
# load_strategy
# =============================================================================================


def test_load_reads_every_section(strategy_dir: Path) -> None:
    cfg = load_strategy(strategy_dir)
    for section in FILES:
        assert dumped(getattr(cfg, section)) == DEFAULTS[section], section
    assert cfg.execute.stop.trail_atr == 1.5


def test_repository_earnings_gap_has_the_approved_defaults() -> None:
    # The committed config, not a fixture: these are the values the user approved for v1.
    cfg = load_strategy(REPO_STRATEGY)
    assert dumped(cfg.strategy) == {
        "name": "earnings_gap",
        "cadence": "daily",
        "universe": "watchlist_with_documents",
        "direction": "long",
    }
    assert dumped(cfg.scan) == {
        "min_gap_pct": 4.0,
        "min_rvol": 2.0,
        "rvol_days": 20,
        "atr_days": 14,
        "min_price": 5.0,
        "min_avg_dollar_volume": 20_000_000,
        "dollar_volume_days": 20,
        "earnings_items": ["2.02"],
    }
    assert dumped(cfg.rules) == {"direction": "follow_gap"}
    assert dumped(cfg.judge) == {"kind": "take_all"}
    assert dumped(cfg.execute) == {
        "entry": {"kind": "next_open", "key": "gap_day_low"},
        "stop": {
            "kind": "atr_zone_stop",
            "zone_atr": 0.25,
            "zone_cap_pct": 1.0,
            "hard_atr": 1.0,
            "observe_days": 3,
            "trail_atr": 1.5,
        },
        "reentry": {
            "key": "pre_earnings_close",
            "window_sessions": 20,
            "max_count": 1,
            "risk_mode": "shared",
        },
        "exit": {"max_sessions": 60},
        "sizing": {
            "kind": "risk_based",
            "risk_pct": 0.5,
            "max_position_pct": 10.0,
            "max_gross_pct": 100.0,
        },
        "fills": {"slippage_bps": 5.0},
        "account": {"initial_equity": 100_000},
        "data": {"max_missing_sessions": 5, "max_jump_ratio": 1.8},
    }
    assert dumped(cfg.attribution) == {"metrics": ["r_multiple", "exit_reason", "holding_sessions"]}


@pytest.mark.parametrize("missing", FILES)
def test_load_refuses_a_directory_missing_a_file(strategy_dir: Path, missing: str) -> None:
    (strategy_dir / f"{missing}.yaml").unlink()
    with pytest.raises((FileNotFoundError, ValueError)):
        load_strategy(strategy_dir)


def test_load_refuses_an_unknown_top_level_key(tmp_path: Path) -> None:
    # A typo ("min_gap_pc") must not fall back silently to the default.
    scan = {**DEFAULTS["scan"], "min_gap_pc": 3.0}
    with pytest.raises(ValueError):
        load_strategy(write_strategy(tmp_path, {"scan": scan}))


def test_load_refuses_an_unknown_nested_key(tmp_path: Path) -> None:
    execute = json.loads(json.dumps(DEFAULTS["execute"]))
    execute["stop"]["trail"] = 1.0
    with pytest.raises(ValueError):
        load_strategy(write_strategy(tmp_path, {"execute": execute}))


def test_load_refuses_a_value_of_the_wrong_type(tmp_path: Path) -> None:
    scan = {**DEFAULTS["scan"], "min_rvol": "high"}
    with pytest.raises(ValueError):
        load_strategy(write_strategy(tmp_path, {"scan": scan}))


# =============================================================================================
# apply_overrides
# =============================================================================================


def test_override_changes_one_value_and_leaves_the_original_alone(strategy_dir: Path) -> None:
    base = load_strategy(strategy_dir)
    changed = apply_overrides(base, ["execute.stop.trail_atr=1.0", "scan.min_gap_pct=6"])
    assert changed.execute.stop.trail_atr == 1.0
    assert changed.scan.min_gap_pct == 6.0
    assert base.execute.stop.trail_atr == 1.5
    assert base.scan.min_gap_pct == 4.0
    # everything else is untouched
    assert dumped(changed.execute.sizing) == dumped(base.execute.sizing)


def test_no_overrides_is_the_same_config(strategy_dir: Path) -> None:
    base = load_strategy(strategy_dir)
    assert dumped(apply_overrides(base, [])) == dumped(base)


def test_override_values_are_typed(strategy_dir: Path) -> None:
    cfg = apply_overrides(
        load_strategy(strategy_dir),
        ["execute.reentry.risk_mode=independent", "execute.exit.max_sessions=40"],
    )
    assert cfg.execute.reentry.risk_mode == "independent"
    assert cfg.execute.exit.max_sessions == 40
    assert isinstance(cfg.execute.exit.max_sessions, int)


@pytest.mark.parametrize(
    "override",
    [
        "execute.stop.nope=1.0",  # unknown leaf
        "execute.nope.trail_atr=1.0",  # unknown branch
        "execute.stop.trail_atr=wide",  # not a number
        "execute.exit.max_sessions=1.5",  # not an integer
        "execute.stop.trail_atr",  # no value
        "=1.0",  # no path
    ],
)
def test_bad_override_raises(strategy_dir: Path, override: str) -> None:
    with pytest.raises(ValueError):
        apply_overrides(load_strategy(strategy_dir), [override])


# =============================================================================================
# content_hash
# =============================================================================================


def test_hash_is_stable_and_independent_of_location_and_mtime(
    strategy_dir: Path, tmp_path: Path
) -> None:
    first = content_hash(strategy_dir)
    assert first == content_hash(strategy_dir)
    for path in strategy_dir.iterdir():
        os.utime(path, (1_000_000_000, 1_000_000_000))
    assert content_hash(strategy_dir) == first
    copy = tmp_path / "elsewhere" / "earnings_gap"
    shutil.copytree(strategy_dir, copy)
    assert content_hash(copy) == first


@pytest.mark.parametrize("changed", FILES)
def test_hash_changes_when_any_file_changes(strategy_dir: Path, changed: str) -> None:
    before = content_hash(strategy_dir)
    path = strategy_dir / f"{changed}.yaml"
    path.write_text(path.read_text() + "edited: 1\n")  # changes raw bytes and parsed content
    assert content_hash(strategy_dir) != before


def test_hash_depends_on_overrides_but_not_their_order(strategy_dir: Path) -> None:
    base = content_hash(strategy_dir)
    a = "execute.stop.trail_atr=1.0"
    b = "scan.min_gap_pct=6.0"
    one = content_hash(strategy_dir, [a])
    assert one != base
    assert content_hash(strategy_dir, [a, b]) == content_hash(strategy_dir, [b, a])
    assert content_hash(strategy_dir, [a, b]) != one
    assert content_hash(strategy_dir, ["execute.stop.trail_atr=1.25"]) != one


# =============================================================================================
# register_version / attempt_count
# =============================================================================================


def versions(session: Session) -> list[StrategyVersion]:
    return list(session.exec(select(StrategyVersion).order_by(col(StrategyVersion.id))).all())


def test_base_config_registers_as_an_approved_new_strategy(
    session: Session, strategy_dir: Path
) -> None:
    version = register_version(session, strategy_dir)
    assert version.id is not None
    assert version.strategy_id == "earnings_gap"
    assert version.content_hash == content_hash(strategy_dir)
    assert version.status == "approved"
    assert version.change_kind == "new_strategy"
    assert version.proposed_by == "human"
    assert version.parent_id is None
    assert version.registered_at.tzinfo is not None


def test_registration_is_idempotent(session: Session, strategy_dir: Path) -> None:
    first = register_version(session, strategy_dir)
    again = register_version(session, strategy_dir)
    assert again.id == first.id
    assert len(versions(session)) == 1


def test_overrides_register_a_proposed_config_child_of_the_base(
    session: Session, strategy_dir: Path
) -> None:
    base = register_version(session, strategy_dir)
    child = register_version(
        session, strategy_dir, ["execute.stop.trail_atr=1.0"], proposed_by="agent"
    )
    assert child.id != base.id
    assert child.status == "proposed"
    assert child.change_kind == "config"
    assert child.parent_id == base.id
    assert child.proposed_by == "agent"
    assert child.content_hash == content_hash(strategy_dir, ["execute.stop.trail_atr=1.0"])


def test_attempt_count_counts_each_distinct_parameter_set_once(
    session: Session, strategy_dir: Path, tmp_path: Path
) -> None:
    assert attempt_count(session, "earnings_gap") == 0
    register_version(session, strategy_dir)
    register_version(session, strategy_dir, ["execute.stop.trail_atr=1.0"])
    register_version(session, strategy_dir, ["execute.stop.trail_atr=1.0"])  # a re-run
    register_version(session, strategy_dir, ["scan.min_gap_pct=6.0", "execute.stop.trail_atr=1.0"])
    register_version(
        session, strategy_dir, ["execute.stop.trail_atr=1.0", "scan.min_gap_pct=6.0"]
    )  # same set, other order
    assert attempt_count(session, "earnings_gap") == 3
    # another strategy's versions do not count
    other = write_strategy(
        tmp_path / "others", {"strategy": {**DEFAULTS["strategy"], "name": "other"}}, name="other"
    )
    register_version(session, other)
    assert attempt_count(session, "earnings_gap") == 3
    assert attempt_count(session, "other") == 1


def test_bad_override_registers_nothing(session: Session, strategy_dir: Path) -> None:
    register_version(session, strategy_dir)
    with pytest.raises(ValueError):
        register_version(session, strategy_dir, ["execute.stop.nope=1.0"])
    session.rollback()
    assert attempt_count(session, "earnings_gap") == 1
