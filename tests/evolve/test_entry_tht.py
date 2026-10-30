# Guards the entry test's choice of null pool: draws for a "tht" cell taken from securities that
# were not THT-green the evening before entry (or from a pool that saw a later bar), a "tht" cell
# silently measured for a run whose scan never required green, plain and tht cells corrected as
# separate families, and the default call changing what it measured before.
"""Target API (asof.evolve.entry / null, acceptance point v2-b-2):

    entry_test(..., null_pools: Sequence[Literal["plain", "tht"]] = ("plain",))
        every (run, hold, null_pool) is one cell of one max-statistic family
    HoldResult.null_pool: str   ("plain" or "tht")
    build_panel(..., pool_tht: bool = False)
        BX: pools from pool_range(..., cfg.scan, pool_tht=pool_tht)

Interface points this file pins down where the spec left a choice:
- Results are sorted by (run_id, hold, null_pool); "plain" sorts before "tht".
- A "tht" cell with any run of the family whose scan has no tht_filter raises ValueError before
  anything is drawn; so do an empty, duplicated or unknown null_pools.
- Within one (run, hold) the plain and tht cells hold the same real entries and returns (the
  run's entries are THT-green, so inside both pools); only the draws differ.

Fake data: tests/evolve/test_gate_bx.py's B-Xtrender world (provenance there), with one more
replay over the same window whose scan sets tht_filter = {length: 33, min_crossings: 20,
median_window: null} (the defaults of asof.features.tht). Measured on the walk prices
(2026-10-29, tht_features directly): AAA has a colour from CAL[179] on and is green on most of
its plain signal nights CAL[211 .. 243]; BBB's first colour is CAL[257], CCC's CAL[268], so on
the evenings before most dev entries the THT pool is a strict part of the plain pool.
The ``future`` world (prices x10 from 2024-03-01, a 2:1 AAA split on file only 2024-04-15)
is the leakage case: nothing in it is visible at dev's last close.

Assumptions a real-data check must confirm:
- A real THT run's entries are each inside that evening's THT pool (fixed_hold raises
  otherwise): run ``asof gate entry`` with the tht null pool on the real THT run; a ValueError
  naming a security means the replay's colour and the pool's colour disagree.
- The THT pool on a real evening is a large share of the plain pool, not a handful (a tiny pool
  makes the draws nearly fixed): print both sizes for one real evening.
"""

from collections.abc import Iterator, Mapping, Sequence
from datetime import date, datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pytest
from sqlmodel import Session

from asof.evolve.entry import FixedHold, HoldResult, cluster_ci, entry_test, fixed_hold
from asof.evolve.null import Panel, build_panel, max_statistic, null_returns
from asof.evolve.protocol import Window
from asof.replay import run_replay
from asof.scan.bx_trender import bx_panel, pool_range
from asof.store.market import visible_sessions
from asof.strategy import BxStrategyConfig, apply_overrides
from tests.evolve.test_entry import dev_entries, entry_spans
from tests.evolve.test_gate_bx import BX_GATE, DEV_END, DEV_START, END, START, BxWorld, _world
from tests.scan.bx_world import CAL, close_of

ET = ZoneInfo("America/New_York")
THT = "scan.tht_filter={length: 33, min_crossings: 20, median_window: null}"
DEV = Window(start=DEV_START, end=DEV_END)
PERMUTATIONS = 39
HOLDS = (3, 7)


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> Iterator[BxWorld]:
    yield from _world(tmp_path_factory.mktemp("entry_tht"), future=False)


@pytest.fixture(scope="module")
def future_world(tmp_path_factory: pytest.TempPathFactory) -> Iterator[BxWorld]:
    yield from _world(tmp_path_factory.mktemp("entry_tht_future"), future=True)


@pytest.fixture(scope="module")
def tht_run(world: BxWorld) -> int:
    con = duckdb.connect()
    with Session(world.engine) as s:
        run_id = run_replay(s, con, world.root, world.strategy_dir, START, END, (THT,),
                            protocol=BX_GATE)  # fmt: skip
    con.close()
    return run_id


def tht_cfg(w: BxWorld) -> BxStrategyConfig:
    return apply_overrides(w.cfg, [THT])


@pytest.fixture
def session(world: BxWorld) -> Iterator[Session]:
    with Session(world.engine) as s:
        yield s


def dev_close() -> datetime:
    return datetime.combine(DEV.end, time(20), tzinfo=ET)


def prev_session(day: date) -> date:
    return CAL[CAL.index(day) - 1]


def green_at(con: duckdb.DuckDBPyConnection, root: Path, cfg: BxStrategyConfig, day: date,
             ids: Sequence[str]) -> set[str]:  # fmt: skip
    """Securities whose THT colour on ``day`` is green, read at that evening's close."""
    assert cfg.scan.tht_filter is not None
    panel = bx_panel(con, root, close_of(day), list(ids), tht=cfg.scan.tht_filter)
    rows = panel[[str(d)[:10] == day.isoformat() for d in panel["session_date"]]]
    return {str(s) for s, c in zip(rows["security_id"], rows["tht_color"], strict=True)
            if c == "green"}  # fmt: skip


def composed(
    session: Session,
    con: duckdb.DuckDBPyConnection,
    root: Path,
    cfgs: Mapping[int, BxStrategyConfig],
    holds: Sequence[int],
    pools: Sequence[str],
) -> list[HoldResult]:
    """entry_test written out from its parts, one cell per (run, hold, null pool), one family."""
    as_of = dev_close()
    calendar = visible_sessions(con, root, "1Day", as_of)
    index = {d: i for i, d in enumerate(calendar)}
    real: dict[str, np.ndarray] = {}
    null: dict[str, np.ndarray] = {}
    parts: dict[tuple[int, int, str], FixedHold] = {}
    for run_id, cfg in cfgs.items():
        entries = dev_entries(session, con, root, run_id, DEV)
        bps = cfg.execute.fills.slippage_bps
        for kind in pools:
            panel = build_panel(session, con, root, entry_spans(entries), cfg=cfg, as_of=as_of,
                                pool_tht=kind == "tht")  # fmt: skip
            for hold in holds:
                fh = fixed_hold(entries, panel, calendar, hold=hold, last=DEV.end,
                                slippage_bps=bps)  # fmt: skip
                key = f"{run_id}:{hold}:{kind}"
                real[key] = np.array([s.ret for s in fh.spans])
                null[key] = null_returns(fh.spans, panel, permutations=PERMUTATIONS, seed=0,
                                         slippage_bps=bps)  # fmt: skip
                parts[(run_id, hold, kind)] = fh
    family = max_statistic(real, null)
    out = []
    for run_id, hold, kind in sorted(parts):
        key, fh = f"{run_id}:{hold}:{kind}", parts[(run_id, hold, kind)]
        out.append(
            HoldResult(
                run_id=run_id,
                hold=hold,
                null_pool=kind,
                result=family[key],
                ci=cluster_ci(
                    real[key] - null[key].mean(axis=0),
                    [index[s.entry_session] // hold for s in fh.spans],
                    draws=2000,
                    seed=0,
                ),  # fmt: skip
                sd_ratio=float(np.std(real[key]) / np.std(null[key])),
                beyond_window=fh.beyond_window,
                unpriced=fh.unpriced,
            )
        )
    return out


def cells(results: Sequence[HoldResult]) -> list[tuple[int, int, str]]:
    return [(r.run_id, r.hold, r.null_pool) for r in results]


def assert_results(got: Sequence[HoldResult], expected: Sequence[HoldResult]) -> None:
    assert cells(got) == cells(expected)
    for g, e in zip(got, expected, strict=True):
        assert g.result == e.result, cells([g])
        assert g.ci == pytest.approx(e.ci, rel=1e-12, abs=1e-15), cells([g])
        assert g.sd_ratio == pytest.approx(e.sd_ratio, rel=1e-9), cells([g])
        assert (g.beyond_window, g.unpriced) == (e.beyond_window, e.unpriced)


def run(
    session: Session,
    con: duckdb.DuckDBPyConnection,
    root: Path,
    run_ids: Sequence[int],
    cfg: BxStrategyConfig | Mapping[int, BxStrategyConfig],
    **kw: Any,
) -> list[HoldResult]:
    return entry_test(
        session, con, root, run_ids, cfg=cfg, holds=HOLDS, dev=DEV,
        permutations=PERMUTATIONS, seed=0, **kw,
    )  # fmt: skip


# ---------------------------------------------------------------------------------------------
# fixture guards
# ---------------------------------------------------------------------------------------------


def test_fixture_the_tht_run_enters_inside_dev_and_its_pools_are_narrower(
    session: Session, con: duckdb.DuckDBPyConnection, world: BxWorld, tht_run: int
) -> None:
    entries = dev_entries(session, con, world.root, tht_run, DEV)
    assert len(entries) >= 2
    spans = entry_spans(entries)
    plain = build_panel(session, con, world.root, spans, cfg=tht_cfg(world), as_of=dev_close())
    tht = build_panel(session, con, world.root, spans, cfg=tht_cfg(world), as_of=dev_close(),
                      pool_tht=True)  # fmt: skip
    assert all(tht.pools[e] for e in tht.pools)
    assert any(tht.pools[e] < plain.pools[e] for e in tht.pools)


# ---------------------------------------------------------------------------------------------
# normal path
# ---------------------------------------------------------------------------------------------


def test_the_default_call_measures_the_plain_pool_only(
    session: Session, con: duckdb.DuckDBPyConnection, world: BxWorld, tht_run: int
) -> None:
    cfg = tht_cfg(world)

    default = run(session, con, world.root, [tht_run], cfg)
    plain = run(session, con, world.root, [tht_run], cfg, null_pools=("plain",))

    assert cells(default) == [(tht_run, h, "plain") for h in HOLDS]
    assert_results(default, plain)
    assert_results(default, composed(session, con, world.root, {tht_run: cfg}, HOLDS, ["plain"]))


def test_plain_and_tht_cells_are_one_family(
    session: Session, con: duckdb.DuckDBPyConnection, world: BxWorld, tht_run: int
) -> None:
    cfg = tht_cfg(world)

    got = run(session, con, world.root, [tht_run], cfg, null_pools=("tht", "plain"))

    assert cells(got) == [(tht_run, h, p) for h in HOLDS for p in ("plain", "tht")]
    assert_results(got, composed(session, con, world.root, {tht_run: cfg}, HOLDS, ["plain", "tht"]))
    # four cells: the correction reaches across pools
    alone = run(session, con, world.root, [tht_run], cfg)
    for a in alone:
        [same] = [r for r in got if (r.hold, r.null_pool) == (a.hold, "plain")]
        assert same.result.p_raw == a.result.p_raw
        assert same.result.p_adjusted >= a.result.p_adjusted


def test_plain_and_tht_cells_share_the_real_entries(
    session: Session, con: duckdb.DuckDBPyConnection, world: BxWorld, tht_run: int
) -> None:
    got = run(session, con, world.root, [tht_run], tht_cfg(world), null_pools=("plain", "tht"))

    for h in HOLDS:
        [p] = [r for r in got if (r.hold, r.null_pool) == (h, "plain")]
        [t] = [r for r in got if (r.hold, r.null_pool) == (h, "tht")]
        assert (p.result.trades, p.result.mean) == (t.result.trades, t.result.mean)
        assert (p.beyond_window, p.unpriced) == (t.beyond_window, t.unpriced)
    assert any(r.result.null_mean != s.result.null_mean for r in got for s in got
               if r.hold == s.hold and r.null_pool != s.null_pool)  # fmt: skip


def test_tht_draws_come_only_from_securities_green_the_evening_before(
    session: Session, con: duckdb.DuckDBPyConnection, world: BxWorld, tht_run: int
) -> None:
    cfg = tht_cfg(world)
    entries = dev_entries(session, con, world.root, tht_run, DEV)
    spans = entry_spans(entries)
    plain = build_panel(session, con, world.root, spans, cfg=cfg, as_of=dev_close())

    tht = build_panel(session, con, world.root, spans, cfg=cfg, as_of=dev_close(), pool_tht=True)

    assert set(tht.pools) == set(plain.pools)
    for entry, members in tht.pools.items():
        evening = prev_session(entry)
        green = green_at(con, world.root, cfg, evening, sorted(plain.pools[entry]))
        assert members == frozenset(plain.pools[entry] & green), entry
        assert members == pool_range(con, world.root, evening, evening, cfg.scan,
                                     pool_tht=True)[evening]  # fmt: skip
    # and the draws are those of that pool: the tht cell is the null of a panel holding it
    fh = fixed_hold(entries, tht, visible_sessions(con, world.root, "1Day", dev_close()), hold=3,
                    last=DEV.end, slippage_bps=cfg.execute.fills.slippage_bps)  # fmt: skip
    swapped = Panel(plain.sessions, plain.open, plain.close, plain.splits, tht.pools,
                    plain.spinoffs, plain.restarts)  # fmt: skip
    a = null_returns(fh.spans, tht, permutations=PERMUTATIONS, seed=0,
                     slippage_bps=cfg.execute.fills.slippage_bps)  # fmt: skip
    b = null_returns(fh.spans, swapped, permutations=PERMUTATIONS, seed=0,
                     slippage_bps=cfg.execute.fills.slippage_bps)  # fmt: skip
    np.testing.assert_array_equal(a, b)


def test_build_panel_defaults_to_the_plain_pool(
    session: Session, con: duckdb.DuckDBPyConnection, world: BxWorld, tht_run: int
) -> None:
    cfg = tht_cfg(world)
    spans = entry_spans(dev_entries(session, con, world.root, tht_run, DEV))

    a = build_panel(session, con, world.root, spans, cfg=cfg, as_of=dev_close())
    b = build_panel(session, con, world.root, spans, cfg=cfg, as_of=dev_close(), pool_tht=False)
    c = build_panel(session, con, world.root, spans, cfg=world.cfg, as_of=dev_close())

    assert a.pools == b.pools == c.pools


# ---------------------------------------------------------------------------------------------
# boundaries
# ---------------------------------------------------------------------------------------------


def test_a_tht_cell_for_a_run_without_the_tht_filter_raises(
    session: Session, con: duckdb.DuckDBPyConnection, world: BxWorld, tht_run: int
) -> None:
    assert world.cfg.scan.tht_filter is None
    with pytest.raises(ValueError, match="tht"):
        run(session, con, world.root, [world.run_id], world.cfg, null_pools=("tht",))
    # one such run in a family is enough
    cfgs = {world.run_id: world.cfg, tht_run: tht_cfg(world)}
    with pytest.raises(ValueError, match="tht"):
        run(session, con, world.root, [world.run_id, tht_run], cfgs, null_pools=("plain", "tht"))
    # the plain cell alone is fine for it
    assert run(session, con, world.root, [world.run_id], world.cfg, null_pools=("plain",))


@pytest.mark.parametrize("null_pools", [(), ("plain", "plain"), ("both",)])
def test_bad_null_pools_raise(
    session: Session, con: duckdb.DuckDBPyConnection, world: BxWorld, tht_run: int,
    null_pools: tuple[str, ...],
) -> None:  # fmt: skip
    with pytest.raises(ValueError):
        run(session, con, world.root, [tht_run], tht_cfg(world), null_pools=null_pools)


# ---------------------------------------------------------------------------------------------
# leakage
# ---------------------------------------------------------------------------------------------


def test_prices_and_records_after_dev_change_no_tht_cell(
    session: Session, con: duckdb.DuckDBPyConnection, world: BxWorld, future_world: BxWorld,
    tht_run: int,
) -> None:  # fmt: skip
    # the same decisions, read against a store whose later prices are ten times higher and that
    # holds a split record on file only after dev
    cfg = tht_cfg(world)
    spans = entry_spans(dev_entries(session, con, world.root, tht_run, DEV))
    a = build_panel(session, con, world.root, spans, cfg=cfg, as_of=dev_close(), pool_tht=True)
    b = build_panel(session, con, future_world.root, spans, cfg=cfg, as_of=dev_close(),
                    pool_tht=True)  # fmt: skip
    assert a.pools == b.pools

    clean = run(session, con, world.root, [tht_run], cfg, null_pools=("plain", "tht"))
    wild = run(session, con, future_world.root, [tht_run], cfg, null_pools=("plain", "tht"))

    assert repr(clean) == repr(wild)
