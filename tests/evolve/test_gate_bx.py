"""Guards against a gate that cannot measure a B-Xtrender family, or measures it against the wrong
luck: random entries drawn from securities the BX scanner could not have bought the evening
before entry (listed or liquid only later, an ETF, a thin name), drawn legs priced with another
slippage, the two halves of one position drawing different securities, holdout prices or late
records reaching a dev number, and an earnings-gap run silently measured in a BX family.

API under test (asof.evolve.null / gate, asof.cli), extended to BX:

    build_panel(session, con, root, spans, *, cfg: StrategyConfig | BxStrategyConfig, as_of)
        BX: pools[d] = asof.scan.bx_trender.pool(con, root, 20:00 New York of the session before
        d, cfg.scan), i.e. pool_range(con, root, prev, prev, cfg.scan)[prev].
    null_test(..., cfg=<BxStrategyConfig>, ...), gate.check(..., cfg=<BxStrategyConfig>, ...)
        slippage cfg.execute.fills.slippage_bps; spin-offs under cfg.execute.data.
    asof gate null / asof gate report: a BX run's strategy directory is loaded with
        load_any_strategy; a family mixing kinds exits 1 with one line.

Interface points this file pins down where the spec left a choice:
- An entry whose evening pool is empty (BX values not yet defined, or the market filter off)
  raises ValueError from build_panel or from null_returns, matching "no pool" or "no
  security"; which of the two raises is not pinned.
- The mixed-family message names both strategy kinds ("earnings_gap" and "bx_trender").
- A BX config with an unknown universe (only constructible through model_copy) raises
  ValueError in build_panel, as an earnings_gap one does.

World (made up; store writers and provenance in tests/scan/bx_world.py and
tests/replay/bx_replay_world.py): AAA, BBB (the replay tests' paths), CCC, DDD (more liquid
walks, seeds 303 / 404), KEEP (thin all along), SPY (an ETF, the benchmark), JMP (thin until a
400M-share day on JUMP_DAY, then inside the $50M tier for 20 sessions: ADV (19 * 1,000 * ~50 +
4e8 * ~50) / 20 ~ $1B), NEW (listed on LIST_DAY). One BX replay over CAL[205] .. CAL[280]
(2023-10-26 .. 2024-02-14), BX_GATE's dev 2023-10-01 .. 2024-02-29, holdout 2024-03-01 ..
2024-06-28. The ``future`` world is the same store with every price from 2024-03-01 on times
ten, every volume from then on times a hundred, and a 2:1 AAA split dated inside the replay
window (CAL[240]) that went on file only on 2024-04-15 (AAA's prices never split).

Assumptions these fixtures cannot settle; check them on the real store:
- Every Trade of a real BX run is inside its own pool: for run 7 (S1 2024-01-02 .. 04-30),
  every Trade.security_id is in build_panel(...).pools[entry_session]; a traded name outside its
  pool means the null draws from a different set than the scanner bought from.
- The BX pool on a real night is the size the scan reports: for 2024-01-10 the pool is the
  tier's common stocks with defined, warm BX values (a few thousand, not tens and not every
  security in the master).
"""

import shutil
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pytest
from sqlalchemy import Engine
from sqlmodel import Session, col, select
from typer.testing import CliRunner

import asof.cli
from asof.cli import app
from asof.evolve.gate import check
from asof.evolve.null import (
    Panel,
    Span,
    build_panel,
    max_statistic,
    null_returns,
    null_test,
    spans_of_run,
)
from asof.evolve.protocol import EvaluationProtocol
from asof.evolve.versus import compare_run
from asof.replay import run_replay
from asof.scan.bx_trender import pool, pool_range
from asof.store.db import init_db, make_engine
from asof.store.market import market_root
from asof.store.models import Trade
from asof.strategy import BxStrategyConfig, load_any_strategy
from tests.evolve.gate_world import new_run, version
from tests.evolve.protocols import protocol
from tests.evolve.test_null import leg_return
from tests.replay.bx_replay_world import AAA, BBB, KEEP, SCAN, build_world, bx_strategy
from tests.scan.bx_world import CAL, LIQUID_VOLUME, THIN_VOLUME, Sec, close_of, split_row
from tests.strategy.files import write_strategy
from tests.strategy.test_bx_strategy_config import BX_DEFAULTS, write_bx

runner = CliRunner()
ET = ZoneInfo("America/New_York")

START, END = CAL[205], CAL[280]
DEV_START, DEV_END = date(2023, 10, 1), date(2024, 2, 29)
FUTURE = date(2024, 3, 1)
BX_GATE: EvaluationProtocol = protocol(
    dev={"start": DEV_START, "end": DEV_END},
    holdout={"start": FUTURE, "end": date(2024, 6, 28)},
    clean_after=date(2024, 7, 1),
    seen_before_protocol=[],
    benchmarks={
        "symbols": ["SPY"],
        "basket": {"size": 2, "lookback_sessions": 5, "rebalance": "monthly"},
    },
    walk_forward={"train_months": 1, "test_months": 1, "min_positive": 2},
)
PERMUTATIONS = 39
BPS = BX_DEFAULTS["execute"]["fills"]["slippage_bps"]

JUMP_DAY = CAL[150]  # 2023-08-09: JMP's first liquid session
LIST_DAY = CAL[152]  # NEW's first bar
CCC = Sec("S_CCC", "CCC", seed=303)
DDD = Sec("S_DDD", "DDD", seed=404)
SPY = Sec("S_SPY", "SPY", seed=606, start=400.0, cls="etf", volume=10 * LIQUID_VOLUME)
JMP = Sec("S_JMP", "JMP", seed=505, volume=THIN_VOLUME, volumes={JUMP_DAY: 400_000_000})
NEW = Sec("S_NEW", "NEW", seed=707, first=CAL.index(LIST_DAY))
SECS = [AAA, BBB, CCC, DDD, KEEP, SPY, JMP, NEW]
LATE_SPLIT = split_row("AAA", CAL[240], 1.0, 2.0, available_at=close_of(date(2024, 4, 15)))


def _future(sec: Sec) -> Sec:
    after = {d: sec.volumes.get(d, sec.volume) * 100 for d in sec.days if d >= FUTURE}
    return Sec(
        sec.security_id,
        sec.symbol,
        seed=sec.seed,
        start=sec.start,
        cls=sec.cls,
        volume=sec.volume,
        first=sec.first,
        last=sec.last,
        old_symbol=sec.old_symbol,
        raw_scale=((FUTURE, None, 10.0),),
        volumes={**sec.volumes, **after},
    )


@dataclass(frozen=True)
class BxWorld:
    data_dir: Path
    root: Path
    strategy_dir: Path
    engine: Engine
    run_id: int

    @property
    def cfg(self) -> BxStrategyConfig:
        cfg = load_any_strategy(self.strategy_dir)
        assert isinstance(cfg, BxStrategyConfig)
        return cfg


def _world(data_dir: Path, *, future: bool) -> Iterator[BxWorld]:
    root = market_root(data_dir)
    secs = [_future(s) for s in SECS] if future else SECS
    build_world(root, secs, splits=[LATE_SPLIT] if future else ())
    strategy_dir = bx_strategy(data_dir / "strategies")
    engine = make_engine(data_dir / "meta.db")
    init_db(engine)
    con = duckdb.connect()
    with Session(engine) as session:
        run_id = run_replay(session, con, root, strategy_dir, START, END, protocol=BX_GATE)
    con.close()
    yield BxWorld(data_dir, root, strategy_dir, engine, run_id)
    engine.dispose()


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> Iterator[BxWorld]:
    yield from _world(tmp_path_factory.mktemp("bx_gate"), future=False)


@pytest.fixture(scope="module")
def future_world(tmp_path_factory: pytest.TempPathFactory) -> Iterator[BxWorld]:
    yield from _world(tmp_path_factory.mktemp("bx_gate_future"), future=True)


@pytest.fixture
def session(world: BxWorld) -> Iterator[Session]:
    with Session(world.engine) as s:
        yield s


@pytest.fixture
def gate_protocol(monkeypatch: pytest.MonkeyPatch) -> EvaluationProtocol:
    monkeypatch.setattr(asof.cli, "load_protocol", lambda *args, **kwargs: BX_GATE)
    return BX_GATE


def trades(session: Session, run_id: int) -> list[Trade]:
    stmt = select(Trade).where(Trade.run_id == run_id).order_by(col(Trade.entry_at), col(Trade.id))
    return list(session.exec(stmt).all())


def prev_session(day: date) -> date:
    return CAL[CAL.index(day) - 1]


def last_close(spans: list[Span]) -> datetime:
    return close_of(max(s.exit_session for s in spans))


def panel_of(session: Session, con: duckdb.DuckDBPyConnection, w: BxWorld) -> Panel:
    spans = spans_of_run(session, w.run_id)
    return build_panel(session, con, w.root, spans, cfg=w.cfg, as_of=last_close(spans))


def cli_args(
    w: BxWorld, command: str, *run_ids: int, db: Path, extra: tuple[str, ...] = ()
) -> list[str]:
    out = ["gate", command]
    for run_id in run_ids:
        out += ["--run", str(run_id)]
    return [
        *out, "--permutations", str(PERMUTATIONS), "--seed", "0",
        "--strategies-dir", str(w.data_dir / "strategies"),
        "--data-dir", str(w.data_dir), "--db", str(db), *extra,
    ]  # fmt: skip


def one_line_failure(output: str) -> None:
    assert "Traceback" not in output
    assert len(output.strip().splitlines()) == 1, output


# =============================================================================================
# fixture guards (pass today: a failure further down is not a fixture error)
# =============================================================================================


def test_fixture_the_run_trades_several_names_and_splits_a_position_in_two(
    session: Session, world: BxWorld
) -> None:
    rows = trades(session, world.run_id)
    assert len(rows) >= 4
    assert len({t.security_id for t in rows}) >= 2
    assert all(t.exit_at.astimezone(ET).date() <= END for t in rows)
    by_decision: dict[int, list[Trade]] = {}
    for t in rows:
        by_decision.setdefault(t.decision_id, []).append(t)
    assert any(len(legs) == 2 for legs in by_decision.values())
    assert not {t.security_id for t in rows} & {"S_SPY", "S_KEEP"}


def test_fixture_the_evening_pools_hold_several_names(
    con: duckdb.DuckDBPyConnection, world: BxWorld
) -> None:
    cfg = world.cfg.scan
    tonight = pool(con, world.root, close_of(CAL[230]), cfg)
    assert {"S_AAA", "S_BBB", "S_CCC", "S_DDD"} <= tonight
    assert not tonight & {"S_SPY", "S_KEEP"}
    # JMP: thin the evening before its jump, in the tier the evening of it
    assert "S_JMP" not in pool(con, world.root, close_of(prev_session(JUMP_DAY)), cfg)
    assert "S_JMP" in pool(con, world.root, close_of(JUMP_DAY), cfg)


def test_fixture_the_future_world_replays_the_same_trades(
    session: Session, world: BxWorld, future_world: BxWorld
) -> None:
    with Session(future_world.engine) as other:
        a = [(t.security_id, t.leg, t.entry_at, t.exit_at, t.shares, t.pnl)
             for t in trades(session, world.run_id)]  # fmt: skip
        b = [(t.security_id, t.leg, t.entry_at, t.exit_at, t.shares, t.pnl)
             for t in trades(other, future_world.run_id)]  # fmt: skip
    assert a == b


# =============================================================================================
# normal path
# =============================================================================================


def test_bx_pools_are_the_scan_pool_at_the_close_before_each_entry(
    session: Session, con: duckdb.DuckDBPyConnection, world: BxWorld
) -> None:
    spans = spans_of_run(session, world.run_id)
    panel = panel_of(session, con, world)
    cfg = world.cfg.scan
    entries = sorted({s.entry_session for s in spans})
    assert entries
    for entry in entries:
        prev = prev_session(entry)
        expected = pool(con, world.root, close_of(prev), cfg)
        assert panel.pools[entry] == expected, entry
        assert panel.pools[entry] == pool_range(con, world.root, prev, prev, cfg)[prev], entry
    for t in trades(session, world.run_id):
        entry = t.entry_at.astimezone(ET).date()
        assert t.security_id in panel.pools[entry]  # a real leg is inside its own pool


def test_bx_drawn_legs_are_priced_like_the_real_bx_legs(
    session: Session, con: duckdb.DuckDBPyConnection, world: BxWorld
) -> None:
    # With the pool narrowed to the traded security the draw is the real leg: same open, same
    # exit fill, the strategy's own 5 bps both ways.
    spans = spans_of_run(session, world.run_id)
    panel = panel_of(session, con, world)
    for s in spans:
        sid = s.key.split(":")[0]
        own = Panel(panel.sessions, panel.open, panel.close, panel.splits,
                    {s.entry_session: frozenset({sid})})  # fmt: skip
        out = null_returns([s], own, permutations=1, seed=0, slippage_bps=BPS)
        assert out[0, 0] == pytest.approx(s.ret, rel=1e-9, abs=1e-12), s.key


def test_null_test_accepts_a_bx_config_and_equals_its_parts(
    session: Session, con: duckdb.DuckDBPyConnection, world: BxWorld
) -> None:
    results = null_test(
        session, con, world.root, [world.run_id], cfg=world.cfg,
        permutations=PERMUTATIONS, seed=0,
    )  # fmt: skip

    spans = spans_of_run(session, world.run_id)
    panel = panel_of(session, con, world)
    null = null_returns(spans, panel, permutations=PERMUTATIONS, seed=0, slippage_bps=BPS)
    expected = max_statistic({"r": np.array([s.ret for s in spans])}, {"r": null})["r"]
    assert list(results) == [world.run_id]
    assert results[world.run_id] == expected
    assert results[world.run_id].trades == len(spans)


def test_gate_check_measures_a_bx_run(
    session: Session, con: duckdb.DuckDBPyConnection, world: BxWorld
) -> None:
    result = check(
        session, con, world.root, [world.run_id], cfg=world.cfg, protocol=BX_GATE,
        permutations=PERMUTATIONS, seed=0,
    )  # fmt: skip

    dev_spans = [
        s for s in spans_of_run(session, world.run_id)
        if s.entry_session >= DEV_START and s.exit_session <= DEV_END
    ]  # fmt: skip
    assert result.candidate == world.run_id
    assert result.variants[world.run_id].trades == len(dev_spans)
    assert result.versus == compare_run(
        session, con, world.root, world.run_id, BX_GATE, window=(DEV_START, DEV_END)
    )
    assert len(result.segments) == 4  # (Oct, Nov) .. (Jan, Feb)
    assert result.walk_forward_scorable >= 3
    assert result.versions_on_dev == 1


def test_the_two_halves_of_one_position_draw_the_same_security(
    session: Session, con: duckdb.DuckDBPyConnection, world: BxWorld
) -> None:
    # Both halves are Trade leg 1 with the same entry: one key, so one draw per permutation.
    by_decision: dict[int, list[Trade]] = {}
    for t in trades(session, world.run_id):
        by_decision.setdefault(t.decision_id, []).append(t)
    pair = next(legs for legs in by_decision.values() if len(legs) == 2)
    spans = spans_of_run(session, world.run_id)
    keyed = [
        s for s in spans
        if s.key.startswith(f"{pair[0].security_id}:")
        and s.entry_session == pair[0].entry_at.astimezone(ET).date()
    ]  # fmt: skip
    assert len(keyed) == 2 and keyed[0].key == keyed[1].key

    panel = panel_of(session, con, world)
    out = null_returns(keyed, panel, permutations=PERMUTATIONS, seed=0, slippage_bps=BPS)
    members = sorted(panel.pools[keyed[0].entry_session])
    assert len(members) >= 3
    for k in range(PERMUTATIONS):
        firsts = {
            m
            for m in members
            if leg_return(panel, m, keyed[0], BPS) == pytest.approx(out[k, 0], abs=1e-12)
        }
        seconds = {
            m
            for m in members
            if leg_return(panel, m, keyed[1], BPS) == pytest.approx(out[k, 1], abs=1e-12)
        }
        assert len(firsts) == 1, (k, firsts)  # fixture: returns tell the members apart
        assert firsts == seconds, k


def test_gate_report_cli_writes_a_report_for_a_bx_run(
    world: BxWorld, gate_protocol: EvaluationProtocol, tmp_path: Path
) -> None:
    report = tmp_path / "gate.md"
    result = runner.invoke(
        app,
        cli_args(world, "report", world.run_id, db=world.data_dir / "meta.db",
                 extra=("--report", str(report))),
    )  # fmt: skip

    assert result.exit_code == 0, result.output
    assert str(report) in result.output
    text = report.read_text()
    assert str(world.run_id) in text
    assert "p adj" in text.lower()
    assert "spy" in text.lower()


def test_gate_null_cli_prints_a_bx_run(world: BxWorld, gate_protocol: EvaluationProtocol) -> None:
    result = runner.invoke(
        app, cli_args(world, "null", world.run_id, db=world.data_dir / "meta.db")
    )

    assert result.exit_code == 0, result.output
    rows = [line.split() for line in result.output.splitlines()[1:] if line.strip()]
    assert [int(r[0]) for r in rows] == [world.run_id]


# =============================================================================================
# boundaries
# =============================================================================================


def test_an_entry_on_the_first_session_of_the_store_has_no_evening_to_decide_on(
    session: Session, con: duckdb.DuckDBPyConnection, world: BxWorld
) -> None:
    first = Span("S_AAA:2023-01-03:1", CAL[0], CAL[3], "open", 0.0)
    with pytest.raises(ValueError, match="no session before"):
        build_panel(session, con, world.root, [first], cfg=world.cfg, as_of=close_of(CAL[3]))


def test_an_entry_before_any_bx_value_is_defined_has_nothing_to_draw(
    session: Session, con: duckdb.DuckDBPyConnection, world: BxWorld
) -> None:
    # The evening of CAL[2] no security has three daily bars of history: nothing is eligible.
    early = Span("S_AAA:2023-01-06:1", CAL[3], CAL[6], "open", 0.0)
    with pytest.raises(ValueError, match="no pool|no security"):
        panel = build_panel(
            session, con, world.root, [early], cfg=world.cfg, as_of=close_of(CAL[6])
        )
        assert panel.pools[CAL[3]] == frozenset()
        null_returns([early], panel, permutations=PERMUTATIONS, seed=0, slippage_bps=BPS)


def test_a_night_the_market_filter_is_off_has_nothing_to_draw(
    session: Session, con: duckdb.DuckDBPyConnection, world: BxWorld, tmp_path: Path
) -> None:
    # SPY never has 10,000 closes: the filter reads False every night, the pool is empty.
    scan = {**SCAN, "market_filter": {"symbol": "SPY", "sma_sessions": 10_000}}
    closed = load_any_strategy(write_bx(tmp_path / "strategies", {"scan": scan}))
    assert isinstance(closed, BxStrategyConfig)
    assert pool(con, world.root, close_of(CAL[230]), closed.scan) == frozenset()  # fixture
    spans = spans_of_run(session, world.run_id)
    with pytest.raises(ValueError, match="no pool|no security"):
        panel = build_panel(session, con, world.root, spans, cfg=closed, as_of=last_close(spans))
        null_returns(spans, panel, permutations=PERMUTATIONS, seed=0, slippage_bps=BPS)


def test_an_unknown_bx_universe_raises(
    session: Session, con: duckdb.DuckDBPyConnection, world: BxWorld
) -> None:
    cfg = world.cfg
    other = cfg.model_copy(update={"strategy": cfg.strategy.model_copy(update={"universe": "x"})})
    spans = spans_of_run(session, world.run_id)
    with pytest.raises(ValueError):
        build_panel(session, con, world.root, spans, cfg=other, as_of=last_close(spans))


@pytest.mark.parametrize("command", ["report", "null"])
def test_a_family_mixing_an_earnings_gap_run_and_a_bx_run_exits_one(
    world: BxWorld, gate_protocol: EvaluationProtocol, tmp_path: Path, command: str
) -> None:
    db = tmp_path / "meta.db"
    # the shared database stays untouched; the backup API also carries what is still in the WAL
    with sqlite3.connect(world.data_dir / "meta.db") as src, sqlite3.connect(db) as dst:
        src.backup(dst)
    strategies = tmp_path / "strategies"
    shutil.copytree(world.data_dir / "strategies", strategies)
    gap_dir = write_strategy(strategies)
    engine = make_engine(db)
    with Session(engine) as s:
        gap_run = new_run(s, version(s, gap_dir, None), "earnings_gap")
    engine.dispose()

    extra = ("--report", str(tmp_path / "r.md")) if command == "report" else ()
    args = cli_args(world, command, world.run_id, gap_run, db=db, extra=extra)
    args[args.index("--strategies-dir") + 1] = str(strategies)
    result = runner.invoke(app, args)

    assert result.exit_code == 1, result.output
    one_line_failure(result.output)
    assert "earnings_gap" in result.output and "bx_trender" in result.output
    assert not (tmp_path / "r.md").exists()


# =============================================================================================
# leakage
# =============================================================================================


def test_a_security_that_turns_liquid_on_the_entry_session_is_not_in_that_pool(
    session: Session, con: duckdb.DuckDBPyConnection, world: BxWorld
) -> None:
    # JMP's 400M-share day is JUMP_DAY itself: its bar is on disk at the panel's as_of, but the
    # evening before JUMP_DAY it was a $50k-a-day name.
    after = CAL[CAL.index(JUMP_DAY) + 1]
    spans = [
        Span(
            f"S_AAA:{JUMP_DAY.isoformat()}:1", JUMP_DAY, CAL[CAL.index(JUMP_DAY) + 3], "open", 0.0
        ),
        Span(f"S_AAA:{after.isoformat()}:1", after, CAL[CAL.index(JUMP_DAY) + 3], "open", 0.0),
    ]
    panel = build_panel(
        session, con, world.root, spans, cfg=world.cfg, as_of=close_of(CAL[CAL.index(JUMP_DAY) + 3])
    )
    assert "S_JMP" not in panel.pools[JUMP_DAY]
    assert "S_JMP" in panel.pools[after]


def test_a_security_listed_on_the_entry_session_is_not_in_that_pool(
    session: Session, con: duckdb.DuckDBPyConnection, world: BxWorld
) -> None:
    exit_ = CAL[CAL.index(LIST_DAY) + 2]
    listed = Span(f"S_AAA:{LIST_DAY.isoformat()}:1", LIST_DAY, exit_, "close", 0.0)
    panel = build_panel(session, con, world.root, [listed], cfg=world.cfg, as_of=close_of(exit_))
    assert "S_NEW" not in panel.pools[LIST_DAY]
    assert "S_AAA" in panel.pools[LIST_DAY]  # fixture: the pool itself is not empty


def test_the_bx_panel_reads_no_bar_after_its_as_of(
    session: Session, con: duckdb.DuckDBPyConnection, future_world: BxWorld
) -> None:
    with Session(future_world.engine) as s:
        spans = spans_of_run(s, future_world.run_id)
        as_of = last_close(spans)
        panel = build_panel(s, con, future_world.root, spans, cfg=future_world.cfg, as_of=as_of)
    for by_day in (*panel.open.values(), *panel.close.values()):
        assert all(d <= as_of.date() for d in by_day)
    assert not list(panel.splits.get("S_AAA", ()))  # the late split record is not on file yet


def test_holdout_prices_and_late_records_change_no_gate_number(
    session: Session, con: duckdb.DuckDBPyConnection, world: BxWorld, future_world: BxWorld
) -> None:
    def measure(s: Session, w: BxWorld) -> tuple[str, str]:
        result = check(s, con, w.root, [w.run_id], cfg=w.cfg, protocol=BX_GATE,
                       permutations=PERMUTATIONS, seed=0)  # fmt: skip
        null = null_test(s, con, w.root, [w.run_id], cfg=w.cfg, permutations=PERMUTATIONS, seed=0)
        return repr(result), repr(null)

    clean = measure(session, world)
    with Session(future_world.engine) as s:
        wild = measure(s, future_world)

    # run ids agree (one run per database); repr compares NaN-safe
    assert clean == wild


# =============================================================================================
# a family whose variants scan differently: each draws from its own pool
# =============================================================================================
# The pre-registered BX variants differ in scan settings (S2 requires the monthly bar, S3 the
# SPY trend), and so in what their scanners could buy: S2's pool lacks every stock without 36
# monthly bars. Measured 2026-10-28: the family check refused runs 8 / 9 / 10 for differing in
# scan. Here the variant only lengthens the daily warm-up: NEW (listed on LIST_DAY) is outside its
# pool until it has 120 daily bars, inside the plain variant's from the start.

LONG_WARMUP = "scan.warmup.daily=120"


@pytest.fixture(scope="module")
def long_run(world: BxWorld) -> int:
    con = duckdb.connect()
    with Session(world.engine) as s:
        run_id = run_replay(
            s, con, world.root, world.strategy_dir, START, END, (LONG_WARMUP,), protocol=BX_GATE
        )
    con.close()
    return run_id


def long_cfg(w: BxWorld) -> BxStrategyConfig:
    from asof.strategy import apply_overrides

    return apply_overrides(w.cfg, [LONG_WARMUP])


def own_null(
    session: Session, con: duckdb.DuckDBPyConnection, w: BxWorld, run_id: int,
    cfg: BxStrategyConfig, as_of: datetime,
) -> np.ndarray:  # fmt: skip
    spans = spans_of_run(session, run_id)
    panel = build_panel(session, con, w.root, spans, cfg=cfg, as_of=as_of)
    return null_returns(spans, panel, permutations=PERMUTATIONS, seed=0, slippage_bps=BPS)


def test_fixture_the_two_variants_have_different_pools_on_an_entry(
    session: Session, con: duckdb.DuckDBPyConnection, world: BxWorld, long_run: int
) -> None:
    spans = spans_of_run(session, long_run)
    assert spans
    plain = build_panel(session, con, world.root, spans, cfg=world.cfg, as_of=last_close(spans))
    longer = build_panel(
        session, con, world.root, spans, cfg=long_cfg(world), as_of=last_close(spans)
    )
    differ = [e for e in longer.pools if longer.pools[e] != plain.pools[e]]
    assert differ
    assert all("S_NEW" in plain.pools[e] - longer.pools[e] for e in differ)


def test_null_test_draws_each_variant_from_its_own_pool(
    session: Session, con: duckdb.DuckDBPyConnection, world: BxWorld, long_run: int
) -> None:
    cfgs = {world.run_id: world.cfg, long_run: long_cfg(world)}
    results = null_test(
        session, con, world.root, [world.run_id, long_run], cfg=cfgs,
        permutations=PERMUTATIONS, seed=0,
    )  # fmt: skip

    every = [*spans_of_run(session, world.run_id), *spans_of_run(session, long_run)]
    as_of = max([last_close(every), *(t.available_at for r in cfgs for t in trades(session, r))])
    null = {str(r): own_null(session, con, world, r, c, as_of) for r, c in cfgs.items()}
    real = {str(r): np.array([s.ret for s in spans_of_run(session, r)]) for r in cfgs}
    expected = max_statistic(real, null)
    assert results == {r: expected[str(r)] for r in cfgs}


def test_gate_check_draws_each_variant_from_its_own_pool(
    session: Session, con: duckdb.DuckDBPyConnection, world: BxWorld, long_run: int
) -> None:
    cfgs = {world.run_id: world.cfg, long_run: long_cfg(world)}
    mapped = check(
        session, con, world.root, [world.run_id, long_run], cfg=cfgs, protocol=BX_GATE,
        permutations=PERMUTATIONS, seed=0,
    )  # fmt: skip
    shared = check(
        session, con, world.root, [world.run_id, long_run], cfg=world.cfg, protocol=BX_GATE,
        permutations=PERMUTATIONS, seed=0,
    )  # fmt: skip
    # the plain variant is measured alike either way; the long one only on its own pool
    assert mapped.variants[world.run_id].null_mean == shared.variants[world.run_id].null_mean
    assert mapped.variants[long_run].null_mean != shared.variants[long_run].null_mean


def test_gate_report_cli_accepts_variants_that_scan_differently(
    world: BxWorld, long_run: int, gate_protocol: EvaluationProtocol, tmp_path: Path
) -> None:
    report = tmp_path / "family.md"
    result = runner.invoke(
        app,
        cli_args(world, "report", world.run_id, long_run, db=world.data_dir / "meta.db",
                 extra=("--report", str(report))),
    )  # fmt: skip

    assert result.exit_code == 0, result.output
    assert report.exists()
