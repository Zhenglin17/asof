# Guards the bx_trender strategy directory against a config nobody can identify later: a second
# strategy kind that silently changes how earnings_gap loads or hashes (its attempt count would
# restart), a typo'd key or an unknown exit name taken as a default, a hash blind to one of the
# four files or to an override, and a pre-registered S1 that drifts from the agreed values.
"""Target API (asof.strategy):

    strategy.yaml may carry ``kind: earnings_gap | bx_trender``; absent means earnings_gap.
    load_strategy(path) -> StrategyConfig | BxStrategyConfig
    BxStrategyConfig        sections .strategy (name, kind="bx_trender", cadence="daily",
                            universe="liquid_common", direction="long"), .scan (BxScanConfig of
                            asof.scan.bx_trender), .execute (BxExecuteConfig), .attribution
                            (AttributionConfig)
    BxExecuteConfig         entry {kind: next_open}; exits: list of literal names daily_red /
                            weekly_down; sizing {risk_pct, atr_days, atr_mult, max_position_pct,
                            max_gross_pct}; fills {slippage_bps}; account {initial_equity};
                            data (the shared DataConfig)
    content_hash(path, overrides) over the four bx files (+ canonical overrides)
    apply_overrides(cfg, overrides) on a BxStrategyConfig; ``strategy.*`` refused

No leakage case: config loading has no as_of (a strategy directory is not market information).
Assumptions that only the real repository can confirm (check there, not here):
- ``strategies/bx_trender`` holds exactly strategy / scan / execute / attribution .yaml and no
  file the hash should cover but ignores: ``ls strategies/bx_trender`` shows four files, and
  ``content_hash(Path("strategies/bx_trender"))`` equals the content_hash of the approved
  bx_trender StrategyVersion row after its first registration.
- ``strategies/earnings_gap`` is byte-identical after the bx change: its content_hash stays
  0b36fb14...8734 (pinned below), so its existing attempt count still applies.
"""

import copy
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml
from sqlmodel import Session

from asof.scan.bx_trender import BxScanConfig
from asof.strategy import (
    AttributionConfig,
    BxExecuteConfig,
    BxStrategyConfig,
    StrategyConfig,
    apply_overrides,
    attempt_count,
    content_hash,
    load_strategy,
    register_version,
)
from tests.strategy.files import DEFAULTS, write_strategy

REPO = Path(__file__).resolve().parents[2]
REPO_EARNINGS_GAP = REPO / "strategies" / "earnings_gap"
REPO_BX = REPO / "strategies" / "bx_trender"
EARNINGS_GAP_HASH = "0b36fb143459c7cc1f1fcd0db69b2a28657083dfdc11507e1a0f5c8920ee8734"

BX_FILES = ("strategy", "scan", "execute", "attribution")
BX_DEFAULTS: dict[str, dict[str, Any]] = {
    "strategy": {
        "name": "bx_trender",
        "kind": "bx_trender",
        "cadence": "daily",
        "universe": "liquid_common",
        "direction": "long",
    },
    "scan": {
        "variant": "mid",
        "universe": {"min_adv_usd": 50_000_000, "lookback_sessions": 20, "classes": ["common"]},
        "bx": {"l1": 5, "l2": 20, "l3": 5},
        "warmup": {"daily": 40, "weekly": 44, "monthly": 36},
        "daily_min": 0.0,
        "require_monthly_up": False,
        "market_filter": None,
    },
    "execute": {
        "entry": {"kind": "next_open"},
        "exits": ["daily_red", "weekly_down"],
        "sizing": {
            "risk_pct": 0.5,
            "atr_days": 14,
            "atr_mult": 2.0,
            "max_position_pct": 10.0,
            "max_gross_pct": 80.0,
        },
        "fills": {"slippage_bps": 5.0},
        "account": {"initial_equity": 100_000},
        "data": {"max_missing_sessions": 5, "max_jump_ratio": 1.8, "spinoff_max_wait": 5},
    },
    "attribution": {"metrics": ["r_multiple", "exit_reason", "holding_sessions"]},
}


def write_bx(
    parent: Path, sections: Mapping[str, Mapping[str, Any]] | None = None, name: str = "bx_trender"
) -> Path:
    """Write ``parent/<name>/<file>.yaml`` for the four bx files; ``sections`` replaces files."""
    content = copy.deepcopy(BX_DEFAULTS)
    for key, value in (sections or {}).items():
        content[key] = dict(value)
    directory = parent / name
    directory.mkdir(parents=True, exist_ok=True)
    for key in BX_FILES:
        (directory / f"{key}.yaml").write_text(yaml.safe_dump(content[key], sort_keys=False))
    return directory


def edited(section: str, edit: Any) -> dict[str, Any]:
    """A deep copy of one default section after ``edit(section_dict)``."""
    raw = copy.deepcopy(BX_DEFAULTS[section])
    edit(raw)
    return raw


def dumped(model: object) -> object:
    return json.loads(model.model_dump_json())  # type: ignore[attr-defined]


@pytest.fixture
def bx_dir(tmp_path: Path) -> Path:
    return write_bx(tmp_path / "strategies")


# =============================================================================================
# normal path
# =============================================================================================


def test_a_bx_directory_loads_into_its_four_sections(bx_dir: Path) -> None:
    cfg = load_strategy(bx_dir, BxStrategyConfig)

    assert isinstance(cfg, BxStrategyConfig)
    assert isinstance(cfg.scan, BxScanConfig)
    assert isinstance(cfg.execute, BxExecuteConfig)
    assert isinstance(cfg.attribution, AttributionConfig)
    assert dumped(cfg.strategy) == BX_DEFAULTS["strategy"]
    assert dumped(cfg.execute) == BX_DEFAULTS["execute"]
    assert dumped(cfg.attribution) == BX_DEFAULTS["attribution"]
    scan = dumped(cfg.scan)
    assert isinstance(scan, dict)
    for key, value in BX_DEFAULTS["scan"].items():
        assert scan[key] == value, key
    assert tuple(cfg.execute.exits) == ("daily_red", "weekly_down")
    assert cfg.execute.sizing.atr_mult == 2.0


def test_the_repository_bx_directory_is_the_preregistered_s1() -> None:
    # The committed directory, not a fixture: S1 = weekly up + daily up and daily >= 0.
    cfg = load_strategy(REPO_BX, BxStrategyConfig)

    assert isinstance(cfg, BxStrategyConfig)
    assert sorted(p.name for p in REPO_BX.iterdir() if p.suffix == ".yaml") == sorted(
        f"{name}.yaml" for name in BX_FILES
    )
    assert cfg.strategy.name == "bx_trender"
    assert cfg.strategy.kind == "bx_trender"
    assert cfg.scan.variant == "mid"
    assert cfg.scan.daily_min == 0
    assert cfg.scan.require_monthly_up is False
    assert cfg.scan.market_filter is None
    ex = cfg.execute
    assert ex.entry.kind == "next_open"
    assert tuple(ex.exits) == ("daily_red", "weekly_down")
    s = ex.sizing
    assert (s.risk_pct, s.atr_days, s.atr_mult, s.max_position_pct, s.max_gross_pct) == (
        0.5,
        14,
        2.0,
        10.0,
        80.0,
    )
    assert ex.fills.slippage_bps == 5.0
    assert ex.account.initial_equity == 100_000


def test_a_directory_without_kind_is_still_earnings_gap(tmp_path: Path) -> None:
    cfg = load_strategy(write_strategy(tmp_path / "strategies"))
    assert isinstance(cfg, StrategyConfig)
    assert not isinstance(cfg, BxStrategyConfig)
    assert cfg.execute.stop.trail_atr == 1.5


def test_kind_earnings_gap_written_out_loads_the_same_config(tmp_path: Path) -> None:
    plain = load_strategy(write_strategy(tmp_path / "a"))
    explicit = load_strategy(
        write_strategy(
            tmp_path / "b", {"strategy": {**DEFAULTS["strategy"], "kind": "earnings_gap"}}
        )
    )
    assert isinstance(explicit, StrategyConfig)
    for section in ("scan", "rules", "judge", "execute", "attribution"):
        assert dumped(getattr(explicit, section)) == dumped(getattr(plain, section)), section


def test_the_earnings_gap_directory_and_its_hash_are_unchanged() -> None:
    # Attempt counts of earnings_gap are keyed by this hash; the bx change must not touch it.
    assert isinstance(load_strategy(REPO_EARNINGS_GAP), StrategyConfig)
    assert content_hash(REPO_EARNINGS_GAP) == EARNINGS_GAP_HASH


def test_overrides_reach_the_new_scan_fields_and_the_sizing(bx_dir: Path) -> None:
    base = load_strategy(bx_dir, BxStrategyConfig)
    changed = apply_overrides(
        base,
        [
            "scan.require_monthly_up=true",
            "scan.market_filter={symbol: SPY, sma_sessions: 200}",
            "execute.sizing.atr_mult=3",
        ],
    )

    assert isinstance(changed, BxStrategyConfig)
    assert changed.scan.require_monthly_up is True
    assert changed.scan.market_filter is not None
    assert (changed.scan.market_filter.symbol, changed.scan.market_filter.sma_sessions) == (
        "SPY",
        200,
    )
    assert changed.execute.sizing.atr_mult == 3.0
    # the base is left alone
    assert base.scan.require_monthly_up is False
    assert base.scan.market_filter is None
    assert base.execute.sizing.atr_mult == 2.0


def test_register_version_counts_bx_attempts(session: Session, bx_dir: Path) -> None:
    base = register_version(session, bx_dir)
    child = register_version(session, bx_dir, ["scan.require_monthly_up=true"])
    again = register_version(session, bx_dir, ["scan.require_monthly_up=true"])

    assert base.strategy_id == "bx_trender"
    assert base.content_hash == content_hash(bx_dir)
    assert child.parent_id == base.id
    assert again.id == child.id
    assert attempt_count(session, "bx_trender") == 2


# =============================================================================================
# boundaries
# =============================================================================================


@pytest.mark.parametrize("missing", BX_FILES)
def test_a_bx_directory_missing_a_file_is_refused(bx_dir: Path, missing: str) -> None:
    (bx_dir / f"{missing}.yaml").unlink()
    with pytest.raises(FileNotFoundError):
        load_strategy(bx_dir)


@pytest.mark.parametrize(
    ("section", "edit"),
    [
        ("execute", lambda raw: raw.update(stop={"kind": "atr_zone_stop"})),  # no stop in bx
        ("execute", lambda raw: raw["sizing"].update(kind="risk_based")),
        ("execute", lambda raw: raw.update(exits=["daily_red", "price_stop"])),
        ("execute", lambda raw: raw["entry"].update(kind="close")),
        ("scan", lambda raw: raw.update(min_gap_pct=4.0)),
        ("scan", lambda raw: raw.update(market_filter={"symbol": "SPY", "sma_days": 200})),
        ("strategy", lambda raw: raw.update(universe="watchlist_with_documents")),
        ("strategy", lambda raw: raw.update(kind="swing")),
        ("attribution", lambda raw: raw.update(colour="red")),
    ],
    ids=[
        "execute_stop_section",
        "sizing_kind",
        "unknown_exit_name",
        "entry_kind",
        "scan_earnings_key",
        "market_filter_key",
        "strategy_universe",
        "unknown_kind",
        "attribution_key",
    ],
)
def test_unknown_keys_and_values_are_refused(tmp_path: Path, section: str, edit: Any) -> None:
    path = write_bx(tmp_path / "strategies", {section: edited(section, edit)})
    with pytest.raises(ValueError):
        load_strategy(path)


def test_the_name_must_be_the_directory_name(tmp_path: Path) -> None:
    path = write_bx(tmp_path / "strategies", name="bx_copy")
    with pytest.raises(ValueError):
        load_strategy(path)


def test_the_strategy_section_cannot_be_overridden(bx_dir: Path) -> None:
    base = load_strategy(bx_dir, BxStrategyConfig)
    for override in ("strategy.name=other", "strategy.kind=earnings_gap"):
        with pytest.raises(ValueError):
            apply_overrides(base, [override])


@pytest.mark.parametrize(
    "override",
    [
        "execute.stop.trail_atr=1.0",  # an earnings_gap path
        "execute.exits=[daily_red, price_stop]",
        "scan.market_filter={symbol: SPY}",
        "execute.sizing.atr_mult=wide",
    ],
)
def test_bad_overrides_are_refused(bx_dir: Path, override: str) -> None:
    with pytest.raises(ValueError):
        apply_overrides(load_strategy(bx_dir, BxStrategyConfig), [override])


def test_the_hash_reads_all_four_files_and_nothing_else(tmp_path: Path, bx_dir: Path) -> None:
    first = content_hash(bx_dir)
    copy_dir = tmp_path / "elsewhere" / "bx_trender"
    shutil.copytree(bx_dir, copy_dir)
    assert content_hash(copy_dir) == first  # location does not matter
    for name in BX_FILES:
        file = bx_dir / f"{name}.yaml"
        original = file.read_bytes()
        file.write_bytes(original + b"# edited\n")
        assert content_hash(bx_dir) != first, name
        file.write_bytes(original)
    assert content_hash(bx_dir) == first


def test_the_hash_depends_on_the_override_set_but_not_its_order(bx_dir: Path) -> None:
    base = content_hash(bx_dir)
    a = "scan.require_monthly_up=true"
    b = "scan.market_filter={symbol: SPY, sma_sessions: 200}"
    one = content_hash(bx_dir, [a])

    assert one != base
    assert content_hash(bx_dir, [a, b]) == content_hash(bx_dir, [b, a])
    assert content_hash(bx_dir, [a, b]) != one
    # "=3" and "=3.0" on a float setting are one parameter set
    assert content_hash(bx_dir, ["execute.sizing.atr_mult=3"]) == content_hash(
        bx_dir, ["execute.sizing.atr_mult=3.0"]
    )
    assert content_hash(bx_dir, ["execute.sizing.atr_mult=3"]) != base
