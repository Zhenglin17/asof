"""A strategy directory written from Python, for tests that need a config without depending on
the repository's own ``strategies/earnings_gap`` (whose values only the evolution flow may change).

``DEFAULTS`` mirrors the approved earnings_gap defaults, one dict per YAML file; each file holds
its section's keys at the top level (``scan.yaml`` starts with ``min_gap_pct: 4.0``).
"""

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

FILES = ("strategy", "scan", "rules", "judge", "execute", "attribution")

DEFAULTS: dict[str, dict[str, Any]] = {
    "strategy": {
        "name": "earnings_gap",
        "cadence": "daily",
        "universe": "watchlist_with_documents",
        "direction": "long",
    },
    "scan": {
        "min_gap_pct": 4.0,
        "min_rvol": 2.0,
        "rvol_days": 20,
        "atr_days": 14,
        "min_price": 5.0,
        "min_avg_dollar_volume": 20_000_000,
        "dollar_volume_days": 20,
        "earnings_items": ["2.02"],
    },
    "rules": {"direction": "follow_gap"},
    "judge": {"kind": "take_all"},
    "execute": {
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
    },
    "attribution": {"metrics": ["r_multiple", "exit_reason", "holding_sessions"]},
}


def write_strategy(
    parent: Path,
    sections: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    name: str = "earnings_gap",
) -> Path:
    """Write ``parent/<name>/<file>.yaml`` for every file; ``sections`` replaces whole files."""
    content = copy.deepcopy(DEFAULTS)
    for key, value in (sections or {}).items():
        content[key] = dict(value)
    directory = parent / name
    directory.mkdir(parents=True, exist_ok=True)
    for key in FILES:
        (directory / f"{key}.yaml").write_text(yaml.safe_dump(content[key], sort_keys=False))
    return directory
