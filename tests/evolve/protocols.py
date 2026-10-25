"""Evaluation protocols built in code, for tests that must not depend on configs/evaluation.yaml.

``REAL`` mirrors the pre-registered protocol's values (the file itself is checked once, in
test_protocol.py). ``protocol(**changes)`` replaces whole top-level keys. ``covering(...)``
moves the holdout onto the test store's dates: tests/scan/world.py has sessions 2026-01-05 ..
2026-06-30, which happen to sit inside the real holdout too.
"""

import copy
from datetime import date
from typing import Any

from asof.evolve.protocol import EvaluationProtocol

REAL: dict[str, Any] = {
    "dev": {"start": date(2020, 1, 1), "end": date(2025, 6, 30)},
    "holdout": {"start": date(2025, 7, 1), "end": date(2026, 6, 30)},
    "clean_after": date(2026, 7, 1),
    "seen_before_protocol": ["earnings_gap"],
    "attempt_budget": 20,
    "alpha": 0.05,
    "permutations": 1000,
    "thin_edge": {"drop_top": 3, "extra_cost_bps": 10},
    "benchmarks": {
        "symbols": ["SPY", "QQQ"],
        "basket": {"size": 10, "lookback_sessions": 60, "rebalance": "monthly"},
    },
}


def raw(**changes: Any) -> dict[str, Any]:
    out = copy.deepcopy(REAL)
    out.update(changes)
    return out


def protocol(**changes: Any) -> EvaluationProtocol:
    return EvaluationProtocol.model_validate(raw(**changes))


def covering(start: date, end: date, *, seen: list[str] | None = None) -> EvaluationProtocol:
    """A protocol whose holdout is exactly [start, end]; nothing exempt unless ``seen`` says so."""
    return protocol(
        dev={"start": date(2010, 1, 1), "end": date.fromordinal(start.toordinal() - 1)},
        holdout={"start": start, "end": end},
        clean_after=date.fromordinal(end.toordinal() + 1),
        seen_before_protocol=seen or [],
    )


def sealed_after(day: date) -> EvaluationProtocol:
    """A protocol whose holdout starts the day after ``day``: everything up to it is dev."""
    first = date.fromordinal(day.toordinal() + 1)
    return covering(first, date(first.year + 1, first.month, 1), seen=[])
