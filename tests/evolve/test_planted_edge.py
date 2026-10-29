"""Guards against a gate that cannot find an edge that is there, or finds one that is not: a
synthetic market where the B-Xtrender signal is followed by a planted drift is run through the
real BX replay and the real gate.check, and must come out clearly ahead of random entries after
the family correction and in most walk-forward windows; the same market built from the same
random numbers without the drift must not.

World: tests/evolve/planted_world.py (provenance and the self-consistency argument there).
Planted drift MU = 0.5% a session for H = 10 sessions after each planting night. Measured while
writing these tests (pool of all 30 stocks, 500 permutations, null built from the null module's
parts): with mu = 0.3% the edge was z 2.1 / p 0.024, short of the 0.01 asserted here, so MU is
0.5% (z 4.2, p 0.002; the control z -1.0, p 0.84).

What "known answer" means here: a real leg holds the planted drift of its own security over its
dates; a random draw holds, on average, the mean planted drift of the pool over the same dates
(the runs cover about half of all stock-sessions, so a draw often drifts too). The planted
excess the gate should report is therefore the difference of the two, ``planted_advantage``,
not the real legs' planted drift alone (that one is ~2.5 times larger and no unbiased null can
give it back).

Protocol PLANTED: dev 2023-03-01 .. 2024-03-28 (the BX values are warm from February 2023),
holdout 2024-04-01 .. 2024-06-28; walk-forward 2 + 2 months (five pairs); benchmark SPY; basket
of 5 over 20 sessions; 500 permutations. Variant 2 of the family is the same strategy with half
the gross cap, so the family correction has two variants to correct for.

Assumptions these fixtures cannot settle; check them on the real store:
- A real BX run's legs are all inside the pool the null draws from (see test_gate_bx.py).
- The planted world is easy on purpose (one strong signal, no costs beyond 5 bps, prices
  without gaps or corporate actions); passing it says the machinery can see an edge, not that
  the real S1 has one.
"""

import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pytest
from sqlalchemy import Engine
from sqlmodel import Session
from typer.testing import CliRunner

import asof.cli
from asof.cli import app
from asof.evolve.gate import GateResult, check
from asof.evolve.null import Span, spans_of_run
from asof.evolve.protocol import EvaluationProtocol
from asof.replay import run_replay
from asof.scan.bx_trender import pool_range
from asof.store.db import init_db, make_engine
from asof.store.market import market_root
from asof.strategy import BxStrategyConfig, load_any_strategy
from tests.evolve.planted_world import (
    World,
    first_planted_night,
    generate,
    held_drift,
    signals_through,
    write_store,
)
from tests.evolve.protocols import protocol
from tests.replay.bx_replay_world import bx_strategy
from tests.scan.bx_world import CAL

MU = 0.005
DEV_START, DEV_END = date(2023, 3, 1), date(2024, 3, 28)
PERMUTATIONS = 500
PLANTED: EvaluationProtocol = protocol(
    dev={"start": DEV_START, "end": DEV_END},
    holdout={"start": date(2024, 4, 1), "end": date(2024, 6, 28)},
    clean_after=date(2024, 7, 1),
    seen_before_protocol=[],
    permutations=PERMUTATIONS,
    benchmarks={
        "symbols": ["SPY"],
        "basket": {"size": 5, "lookback_sessions": 20, "rebalance": "monthly"},
    },
    walk_forward={"train_months": 2, "test_months": 2, "min_positive": 3},
)
HALF_CAP = "execute.sizing.max_gross_pct=40.0"
runner = CliRunner()


@dataclass(frozen=True)
class Store:
    """A world written to disk and replayed over dev, one run per variant."""

    world: World
    data_dir: Path
    root: Path
    strategy_dir: Path
    engine: Engine
    run_ids: tuple[int, ...]
    replay_seconds: float  # wall time of the first replay

    @property
    def cfg(self) -> BxStrategyConfig:
        cfg = load_any_strategy(self.strategy_dir)
        assert isinstance(cfg, BxStrategyConfig)
        return cfg


def _store(data_dir: Path, world: World, variants: tuple[tuple[str, ...], ...]) -> Iterator[Store]:
    root = market_root(data_dir)
    write_store(root, world)
    strategy_dir = bx_strategy(data_dir / "strategies")
    engine = make_engine(data_dir / "meta.db")
    init_db(engine)
    con = duckdb.connect()
    run_ids, seconds = [], []
    with Session(engine) as session:
        for overrides in variants:
            began = time.perf_counter()
            run_ids.append(
                run_replay(
                    session,
                    con,
                    root,
                    strategy_dir,
                    DEV_START,
                    DEV_END,
                    overrides,
                    protocol=PLANTED,
                )  # fmt: skip
            )
            seconds.append(time.perf_counter() - began)
    con.close()
    yield Store(world, data_dir, root, strategy_dir, engine, tuple(run_ids), seconds[0])
    engine.dispose()


def _measure(store: Store) -> GateResult:
    con = duckdb.connect()
    try:
        with Session(store.engine) as session:
            return check(
                session, con, store.root, list(store.run_ids), cfg=store.cfg, protocol=PLANTED,
                permutations=PERMUTATIONS, seed=0,
            )  # fmt: skip
    finally:
        con.close()


@pytest.fixture(scope="module")
def planted_world() -> World:
    return generate(MU)


@pytest.fixture(scope="module")
def control_world() -> World:
    return generate(0.0)


@pytest.fixture(scope="module")
def planted(tmp_path_factory: pytest.TempPathFactory, planted_world: World) -> Iterator[Store]:
    yield from _store(tmp_path_factory.mktemp("planted"), planted_world, ((), (HALF_CAP,)))


@pytest.fixture(scope="module")
def control(tmp_path_factory: pytest.TempPathFactory, control_world: World) -> Iterator[Store]:
    yield from _store(tmp_path_factory.mktemp("control"), control_world, ((),))


@pytest.fixture(scope="module")
def planted_result(planted: Store) -> GateResult:
    return _measure(planted)


@pytest.fixture(scope="module")
def control_result(control: Store) -> GateResult:
    return _measure(control)


def dev_spans(engine: Engine, run_id: int) -> list[Span]:
    with Session(engine) as session:
        return [
            s for s in spans_of_run(session, run_id)
            if s.entry_session >= DEV_START and s.exit_session <= DEV_END
        ]  # fmt: skip


def security_of(span: Span) -> str:
    return span.key.split(":")[0]


def planted_advantage(world: World, spans: list[Span]) -> tuple[float, float, float]:
    """Mean planted drift held by the real legs, by an average draw from the whole pool (every
    stock, see the pool guard) over the same dates, and their difference."""
    ids = [s.security_id for s in world.stocks]
    real = [held_drift(world, security_of(s), s.entry_session, s.exit_session, s.exit_at)
            for s in spans]  # fmt: skip
    drawn = [
        float(np.mean([held_drift(world, i, s.entry_session, s.exit_session, s.exit_at)
                       for i in ids]))
        for s in spans
    ]  # fmt: skip
    return float(np.mean(real)), float(np.mean(drawn)), float(np.mean(real) - np.mean(drawn))


# =============================================================================================
# fixture guards (pass today)
# =============================================================================================


def test_fixture_the_planted_world_trades_enough_dev_legs(planted: Store) -> None:
    for run_id in planted.run_ids:
        assert len(dev_spans(planted.engine, run_id)) > 80, run_id
    print(f"planted replay {DEV_START} .. {DEV_END}: {planted.replay_seconds:.1f} s")


def test_fixture_control_signals_equal_planted_ones_through_the_first_planted_night(
    planted_world: World, control_world: World
) -> None:
    first = first_planted_night(planted_world)
    assert first < DEV_START  # planting starts before dev: the dev legs see it
    assert signals_through(control_world, first) == signals_through(planted_world, first)
    later = CAL[CAL.index(first) + 1 :]
    assert any(  # and the drift does change later signals
        signals_through(control_world, d) != signals_through(planted_world, d) for d in later[:40]
    )
    assert not any(s.drift.any() for s in control_world.stocks)


def test_fixture_every_stock_is_in_the_dev_pools(planted: Store) -> None:
    con = duckdb.connect()
    nights = [CAL[CAL.index(DEV_START) - 1], date(2023, 9, 1), date(2024, 3, 27)]
    every = frozenset(s.security_id for s in planted.world.stocks)
    for night in nights:
        assert pool_range(con, planted.root, night, night, planted.cfg.scan)[night] == every, night
    con.close()


def test_fixture_the_real_legs_hold_more_planted_drift_than_a_draw(planted: Store) -> None:
    for run_id in planted.run_ids:
        real, drawn, advantage = planted_advantage(planted.world, dev_spans(planted.engine, run_id))
        assert advantage > 0.01, run_id
        assert real > drawn > 0, run_id


# =============================================================================================
# planted: the gate finds it
# =============================================================================================


def test_the_planted_edge_survives_the_family_correction(
    planted: Store, planted_result: GateResult
) -> None:
    assert set(planted_result.variants) == set(planted.run_ids)
    for run_id in planted.run_ids:
        v = planted_result.variants[run_id]
        assert v.p_adjusted <= 0.01, (run_id, v)
        assert v.excess > 0


def test_the_gate_recovers_at_least_half_of_the_planted_advantage(
    planted: Store, planted_result: GateResult
) -> None:
    for run_id in planted.run_ids:
        spans = dev_spans(planted.engine, run_id)
        real, drawn, advantage = planted_advantage(planted.world, spans)
        v = planted_result.variants[run_id]
        assert v.trades == len(spans)
        assert v.excess >= 0.5 * advantage, (run_id, v.excess, real, drawn, advantage)
        # and not more than the planted advantage plus noise: a gate whose random legs earn
        # too little would overstate any edge
        sd = abs(v.excess / v.z)
        assert v.excess <= advantage + 3 * sd, (run_id, v.excess, advantage, sd)


def test_walk_forward_is_positive_in_most_scorable_windows(planted_result: GateResult) -> None:
    r = planted_result
    assert len(r.segments) == 5  # test windows May-Jun 2023 .. Jan-Feb 2024; Mar-Apr ends late
    assert r.walk_forward_scorable >= 3
    assert r.walk_forward_positive > r.walk_forward_scorable / 2


def test_asof_gate_report_writes_the_planted_report(
    planted: Store, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(asof.cli, "load_protocol", lambda *args, **kwargs: PLANTED)
    report = tmp_path / "planted.md"
    args = ["gate", "report"]
    for run_id in planted.run_ids:
        args += ["--run", str(run_id)]
    args += [
        "--permutations", "99", "--seed", "0",
        "--strategies-dir", str(planted.data_dir / "strategies"),
        "--data-dir", str(planted.data_dir), "--db", str(planted.data_dir / "meta.db"),
        "--report", str(report),
    ]  # fmt: skip

    result = runner.invoke(app, args)

    assert result.exit_code == 0, result.output
    text = report.read_text()
    for run_id in planted.run_ids:
        assert str(run_id) in text


# =============================================================================================
# control: no edge, nothing found
# =============================================================================================


def test_the_control_world_shows_no_edge(control: Store, control_result: GateResult) -> None:
    [run_id] = control.run_ids
    v = control_result.variants[run_id]
    assert v.p_raw > 0.05, v
    assert v.z != 0
    sd = abs(v.excess / v.z)  # sd of the per-permutation means
    assert abs(v.excess) < 2 * sd, v


def test_fixture_the_control_world_has_no_planted_drift_to_find(control: Store) -> None:
    [run_id] = control.run_ids
    _, _, advantage = planted_advantage(control.world, dev_spans(control.engine, run_id))
    assert advantage == 0.0
