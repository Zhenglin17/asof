# Guards against reading a strategy version's overrides back wrongly from its stored summary: a
# value holding ", " (a mapping, a list) cut in two, or a None read back as the string "None",
# so the gate measures a family under other settings than it replayed, or refuses it.
"""API under test (asof.strategy):

    overrides_of_summary(summary: str | None) -> list[str]
        the overrides ``register_version`` joined into ``StrategyVersion.summary`` ("k=v, k=v",
        values as Python prints them), as ``a.b=value`` strings ``apply_overrides`` reads to the
        same typed values: ``content_hash(path, overrides_of_summary(v.summary))`` is the
        version's own hash.

Measured case: run 10 (S3, 2026-10-28) stored ``scan.market_filter={'symbol': 'SPY',
'sma_sessions': 200}``; ``gate report`` split it at ", " and failed to parse the halves.
"""

from pathlib import Path

import pytest
from sqlmodel import Session

from asof.strategy import content_hash, overrides_of_summary, register_version
from tests.strategy.test_bx_strategy_config import write_bx

CASES = [
    ("scan.require_monthly_up=true",),
    ("scan.market_filter={symbol: SPY, sma_sessions: 200}",),
    ("scan.market_filter={symbol: SPY, sma_sessions: 200}", "execute.sizing.risk_pct=0.25"),
    ("scan.daily_min=null",),
    ("execute.exits=[daily_red, weekly_down]", "scan.bx.l1=4"),
    ("scan.tht_filter={length: 33, min_crossings: 20, median_window: null}",),
]


@pytest.mark.parametrize("overrides", CASES, ids=lambda o: " & ".join(o))
def test_a_summary_reads_back_to_the_version_it_was_written_for(
    session: Session, tmp_path: Path, overrides: tuple[str, ...]
) -> None:
    path = write_bx(tmp_path / "strategies")
    version = register_version(session, path, overrides)

    back = overrides_of_summary(version.summary)

    assert len(back) == len(overrides)
    assert content_hash(path, back) == version.content_hash


def test_no_summary_is_no_override() -> None:
    assert overrides_of_summary(None) == []
    assert overrides_of_summary("") == []


def test_the_stored_tht_summary_reads_none_inside_a_mapping_as_null() -> None:
    # measured: run 11 (2026-10-29) stored this; "None" inside the mapping read back as the
    # string 'None' and failed median_window's int | None
    summary = "scan.tht_filter={'length': 33, 'min_crossings': 20, 'median_window': None}"
    assert overrides_of_summary(summary) == [
        "scan.tht_filter={'length': 33, 'min_crossings': 20, 'median_window': null}"
    ]


def test_the_stored_s3_summary_parses() -> None:
    summary = "scan.market_filter={'symbol': 'SPY', 'sma_sessions': 200}"
    assert overrides_of_summary(summary) == [summary]
