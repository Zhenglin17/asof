"""Guards against an entry test that credits the entry with what the exit did, or measures it
against the wrong luck: a fixed hold priced on another day or another price than the null's
legs, a split read as a crash, the two halves of one position counted as two entries, an entry
dropped because its position was still open when the replay ended (selection on the future
path), the same entry drawing different securities for different holds (so the holds no longer
compare), a family correction that forgets a (run, hold) cell, an interval that treats
overlapping holds as independent, a fixed exit after the development window priced from bars
the window could not see, decisions made after as_of, and draws from a pool the scanner did not
have the evening before entry.

API under test (asof.evolve.entry, new module):

    Entry(security_id, entry_session)
    entries_of(spans) -> list[Entry]
        one per (security_id, entry_session); security id parsed from Span.key
        (f"{security_id}:{entry_session}:{leg}"); sorted by (entry_session, security_id)
    entries_of_run(session, run_id, calendar, *, as_of) -> list[Entry]
        visible_decisions(session, as_of, kind="trade", strategy_id=run.strategy_id) of the run,
        long, without "cancelled" in the payload, security_id set; entry_session = the first
        calendar session strictly after the New York date of decision.as_of. Every closed Trade
        of the run must be among them (else ValueError naming it); unknown run: LookupError.
    FixedHold(spans, beyond_window, unpriced)
    fixed_hold(entries, panel, calendar, *, hold, last, slippage_bps) -> FixedHold
        span key f"{security_id}:{entry_session}:fixed", exit_session = calendar[i + hold],
        exit_at "open", ret = asof.evolve.null._leg_return of the real security; a real
        security without the entry open raises (the replay bought there), without the exit
        open is ``unpriced``
    cluster_ci(excess, clusters: Sequence[Hashable], *, draws, seed, level=0.90) -> (low, high)
    HoldResult(run_id, hold, result, ci, sd_ratio, beyond_window, unpriced)
    entry_test(session, con, root, run_ids, *, cfg, holds, dev, permutations, seed,
               bootstrap=2000) -> list[HoldResult]
    asof gate entry --run N [--run M ...] --hold H [--hold G ...] [--permutations K] [--seed S]
        [--current-strategy] [--strategies-dir DIR] [--data-dir DIR] [--db FILE]

Interface points this file pins down where the spec left a choice:
- ``entry_test`` is exactly its parts: the calendar ``visible_sessions(con, root, "1Day",
  as_of)`` at as_of = 20:00 New York of dev.end; per run, ``entries_of_run`` at that as_of
  whose entry session is within dev (any exit, or none yet); ``build_panel`` of those entry
  sessions with the run's own config at that as_of; ``fixed_hold`` with ``last=dev.end`` and
  the config's slippage; ``null_returns`` of the fixed spans with the given seed;
  ``max_statistic`` over every (run, hold) cell at once; ``cluster_ci`` of real - per-entry
  mean of the draws, clustered in blocks of ``hold`` sessions (label = index of the entry
  session in that calendar // hold), with ``draws=bootstrap`` and the same ``seed``;
  ``sd_ratio = np.std(real) / np.std(null)`` (numpy's ddof=0, over every draw of the cell).
- ``entry_test`` measures B-Xtrender runs only: a config whose universe is not
  "liquid_common" raises ValueError.
- Results are sorted by (run id, hold), whatever order the holds were given in.
- An entry whose fixed exit is beyond the window is counted in ``beyond_window`` even when its
  prices are missing too (the window is checked first).
- ``beyond_window + unpriced + len(spans) == len(entries)``.
- An empty ``entries`` gives ``FixedHold([], 0, 0)``.
- cluster_ci's interval is nested across levels for one seed (it resamples the same way
  whatever ``level`` is) and with one cluster is (mean, mean) up to float rounding.
- ``entries_of_run`` is sorted by (entry_session, security_id), as ``entries_of``.
- CLI rows: whitespace-separated tokens; the first three are run, hold, entries (integers),
  the last two are beyond_window and unpriced; p adj is printed with four decimals. The header
  contains "hold", "entries", "ci low", "ci high", "sd ratio" and "p adj". The last line is
  "family:" followed by the cells as run:hold, sorted. A failure is one line, exit 1.

Fake data provenance:
- Pure-layer prices are hand-picked numbers (tests.evolve.test_null.panel_of builders); split
  ratio = new_rate / old_rate, applied to ex-dates in (entry, exit] on raw prices (documented in
  asof.replay.run._split_ratio and asof.evolve.null._leg_return).
- Store-backed tests reuse tests/evolve/test_gate_bx.py's B-Xtrender world (provenance there):
  one BX replay over 2023-10-26 .. 2024-02-14, dev 2023-10-01 .. 2024-02-29, and a ``future``
  copy whose prices from 2024-03-01 on are times ten and whose AAA has a 2:1 split dated
  2023-12-11 (CAL[240]) that went on file only on 2024-04-15.
- Measured on that world (2026-10-29): the run has 14 trade decisions, each long, none
  cancelled, made at 20:00 New York of a signal night; 13 have two Trade rows (both leg 1, the
  same entry: the next session's open); the 14th (S_DDD, decided 2024-02-14, the replay's last
  night) has no Trade row: its position would open on 2024-02-15, after the replay.
- A daily bar is visible from 20:00 New York of its session (asof.ingest.bars, documented); a
  BX decision's as_of is 20:00 New York of its signal night and the replay buys the next
  session's open (asof.replay.run, documented).

Assumptions these fixtures cannot settle; check them on the real store:
- Every entry of a real BX run is inside its own pool at the fixed hold too (fixed_hold raises
  otherwise): run ``asof gate entry --run 8 --hold 7`` on the real store; a ValueError naming a
  security means the replay bought outside the scan's pool.
- A fixed exit at the open N sessions later is priced for nearly every real entry: on the real
  store the printed ``unpriced`` should be zero or a handful (halts, delistings) per run; tens
  mean the panel is missing bars.
- Every closed Trade of a real BX run has a decision whose next session is its entry (the
  cross-check raises otherwise): ``asof gate entry --run 8 --hold 7`` and ``--run 9`` on the real
  store must not raise; and the count of decisions without any Trade row is the number of
  positions open at the replay's end (a handful, all entered in its last weeks).
- The sessions of the real calendar at dev.end are the NYSE sessions (no missing day in the
  daily bars): ``visible_sessions`` over 2020-01-01 .. 2025-06-30 has the NYSE session count
  of those years; a missing day would shift every fixed exit after it by one session.
"""

import shutil
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pytest
from sqlmodel import Session, col, delete, select
from typer.testing import Result

import asof.cli
from asof.cli import app
from asof.evolve.entry import (
    Entry,
    FixedHold,
    HoldResult,
    cluster_ci,
    entries_of,
    entries_of_run,
    entry_test,
    fixed_hold,
)
from asof.evolve.null import Panel, Span, build_panel, max_statistic, null_returns, spans_of_run
from asof.evolve.protocol import EvaluationProtocol, Window
from asof.replay import run_replay
from asof.scan.bx_trender import pool
from asof.store.db import make_engine
from asof.store.market import visible_sessions
from asof.store.models import Decision, Trade
from asof.strategy import AnyStrategyConfig
from tests.evolve.gate_world import new_run
from tests.evolve.protocols import protocol
from tests.evolve.test_gate_bx import (
    BX_GATE,
    DEV_END,
    DEV_START,
    JUMP_DAY,
    LONG_WARMUP,
    PERMUTATIONS,
    BxWorld,
    _world,
    cli_args,
    long_cfg,
    one_line_failure,
    prev_session,
    runner,
    trades,
)
from tests.evolve.test_null import panel_of, span, weekdays
from tests.scan.bx_world import CAL, close_of

ET = ZoneInfo("America/New_York")
DAYS = weekdays(10)  # 2024-01-01 .. 2024-01-12
DEV = Window(start=DEV_START, end=DEV_END)


def ret(exit_px: float, entry_px: float, bps: float, factor: float = 1.0) -> float:
    """A fixed-hold return written out independently of the module."""
    b = bps / 10_000
    return exit_px * factor * (1 - b) / (entry_px * (1 + b)) - 1


def fixed(sid: str, entry: date, exit_: date, value: float) -> Span:
    return Span(f"{sid}:{entry.isoformat()}:fixed", entry, exit_, "open", value)


def assert_spans(got: Sequence[Span], expected: Sequence[Span]) -> None:
    assert [(s.key, s.entry_session, s.exit_session, s.exit_at) for s in got] == [
        (s.key, s.entry_session, s.exit_session, s.exit_at) for s in expected
    ]
    assert [s.ret for s in got] == pytest.approx([s.ret for s in expected], rel=1e-12)


# =============================================================================================
# entries_of
# =============================================================================================


def test_entries_of_merges_the_legs_of_one_entry_and_sorts_by_session_then_security() -> None:
    spans = [
        span("S_B:2024-01-03:1", DAYS[2], DAYS[5]),
        span("S_A:2024-01-03:1", DAYS[2], DAYS[6]),
        span("S_A:2024-01-03:1", DAYS[2], DAYS[8]),  # the second half of one BX position
        span("S_A:2024-01-02:2", DAYS[1], DAYS[4]),  # an earnings-gap re-entry, same session
        span("S_A:2024-01-02:1", DAYS[1], DAYS[3]),
    ]

    assert entries_of(spans) == [
        Entry("S_A", DAYS[1]),
        Entry("S_A", DAYS[2]),
        Entry("S_B", DAYS[2]),
    ]


def test_entries_of_keeps_a_re_entry_on_a_later_session_and_of_nothing_is_empty() -> None:
    spans = [span("S_A:2024-01-02:1", DAYS[1], DAYS[2]), span("S_A:2024-01-05:1", DAYS[4], DAYS[6])]

    assert entries_of(spans) == [Entry("S_A", DAYS[1]), Entry("S_A", DAYS[4])]
    assert entries_of([]) == []


# =============================================================================================
# fixed_hold: normal path
# =============================================================================================


def split_panel() -> Panel:
    """X: open 100 on DAYS[1], a 2:1 split ex DAYS[3], open 60 on DAYS[4] (120 pre-split).
    Y: open 40 every day."""
    x_opens = [100.0, 100.0, 100.0, 50.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0]
    return panel_of(
        {
            "X": {d: (o, o) for d, o in zip(DAYS, x_opens, strict=True)},
            "Y": {d: (40.0, 41.0) for d in DAYS},
        },
        splits={"X": [(DAYS[3], 2.0)]},
    )


def test_a_fixed_hold_buys_the_entry_open_and_sells_the_open_hold_sessions_later() -> None:
    panel = split_panel()

    out = fixed_hold([Entry("X", DAYS[1])], panel, DAYS, hold=3, last=DAYS[-1], slippage_bps=10)

    assert isinstance(out, FixedHold)
    # 60 after the split is 120 before it: +20% before slippage, not -40%
    assert_spans(out.spans, [fixed("X", DAYS[1], DAYS[4], ret(60.0, 100.0, 10, factor=2.0))])
    assert out.spans[0].ret == pytest.approx(1.2 * 0.999 / 1.001 - 1, abs=1e-12)  # 0.197602
    assert (out.beyond_window, out.unpriced) == (0, 0)


def test_the_random_draws_of_a_fixed_hold_are_priced_the_same_way() -> None:
    panel = split_panel()
    out = fixed_hold([Entry("X", DAYS[1])], panel, DAYS, hold=3, last=DAYS[-1], slippage_bps=10)

    drawn = null_returns(out.spans, panel, permutations=50, seed=0, slippage_bps=10)

    x, y = ret(60.0, 100.0, 10, factor=2.0), ret(40.0, 40.0, 10)
    values = drawn[:, 0]
    assert all(v == pytest.approx(x, abs=1e-12) or v == pytest.approx(y, abs=1e-12)
               for v in values)  # fmt: skip
    assert np.any(np.isclose(values, x)) and np.any(np.isclose(values, y))


def test_hold_counts_calendar_sessions_not_weekdays() -> None:
    # DAYS[2] is a closure: two sessions after DAYS[1] is DAYS[4]
    calendar = [d for d in DAYS if d != DAYS[2]]
    panel = split_panel()

    out = fixed_hold([Entry("Y", DAYS[1])], panel, calendar, hold=2, last=DAYS[-1], slippage_bps=0)

    assert [s.exit_session for s in out.spans] == [DAYS[4]]


def test_the_same_entry_draws_the_same_securities_for_every_hold() -> None:
    days = weekdays(20)
    members = [f"M{i}" for i in range(6)]
    # every member grows at its own rate: a return tells the member apart for either hold
    prices = {
        m: {d: (100 * (1 + 0.01 * (i + 1)) ** j, 1.0) for j, d in enumerate(days)}
        for i, m in enumerate(members)
    }
    panel = panel_of(prices, sessions=days)
    entry = [Entry("M0", days[2])]

    short = fixed_hold(entry, panel, days, hold=7, last=days[-1], slippage_bps=5)
    long = fixed_hold(entry, panel, days, hold=11, last=days[-1], slippage_bps=5)

    assert short.spans[0].key == long.spans[0].key == f"M0:{days[2].isoformat()}:fixed"
    assert (short.spans[0].exit_session, long.spans[0].exit_session) == (days[9], days[13])

    def picked(fh: FixedHold, hold: int) -> list[str]:
        drawn = null_returns(fh.spans, panel, permutations=40, seed=3, slippage_bps=5)
        by_return = {
            m: ret(prices[m][days[2 + hold]][0], prices[m][days[2]][0], 5) for m in members
        }
        out = []
        for value in drawn[:, 0]:
            [m] = [m for m, r in by_return.items() if r == pytest.approx(value, abs=1e-12)]
            out.append(m)
        return out

    a, b = picked(short, 7), picked(long, 11)
    assert len(set(a)) > 2  # fixture: the draws do vary
    assert a == b


# =============================================================================================
# fixed_hold: boundaries
# =============================================================================================


def test_a_fixed_exit_after_last_or_past_the_calendar_is_dropped_as_beyond_the_window() -> None:
    panel = split_panel()
    entries = [Entry("Y", DAYS[1]), Entry("Y", DAYS[3]), Entry("Y", DAYS[8])]

    # DAYS[3] + 3 = DAYS[6] is after last (its prices are in the panel: they must not be read);
    # DAYS[8] + 3 is past the calendar's end
    out = fixed_hold(entries, panel, DAYS, hold=3, last=DAYS[5], slippage_bps=0)

    assert_spans(out.spans, [fixed("Y", DAYS[1], DAYS[4], 0.0)])
    assert (out.beyond_window, out.unpriced) == (2, 0)


def test_a_fixed_exit_on_last_itself_is_kept() -> None:
    out = fixed_hold([Entry("Y", DAYS[2])], split_panel(), DAYS, hold=3, last=DAYS[5],
                     slippage_bps=0)  # fmt: skip

    assert [s.exit_session for s in out.spans] == [DAYS[5]]
    assert out.beyond_window == 0


def test_a_real_security_without_the_exit_open_is_dropped_as_unpriced() -> None:
    prices: dict[str, dict[date, tuple[float | None, float | None]]] = {
        "X": {d: (100.0, 100.0) for d in DAYS},
        "Y": {d: (40.0, 41.0) for d in DAYS},
    }
    prices["X"][DAYS[4]] = (None, 99.0)  # a close but no open on the fixed exit (a halt)
    panel = panel_of(prices)
    entries = [Entry("X", DAYS[1]), Entry("X", DAYS[6])]

    out = fixed_hold(entries, panel, DAYS, hold=3, last=DAYS[-1], slippage_bps=0)

    assert_spans(out.spans, [fixed("X", DAYS[6], DAYS[9], 0.0)])
    assert (out.beyond_window, out.unpriced) == (0, 1)


def test_a_real_security_without_the_entry_open_raises() -> None:
    # The replay bought at that open: a panel without it is a bug, not a halt to count.
    prices: dict[str, dict[date, tuple[float | None, float | None]]] = {
        "X": {d: (100.0, 100.0) for d in DAYS},
        "Y": {d: (40.0, 41.0) for d in DAYS},
    }
    prices["X"][DAYS[5]] = (None, 99.0)
    panel = panel_of(prices)

    with pytest.raises(ValueError, match="X"):
        fixed_hold([Entry("X", DAYS[5])], panel, DAYS, hold=3, last=DAYS[-1], slippage_bps=0)


def test_beyond_the_window_is_checked_before_the_prices() -> None:
    prices = {"X": {d: (100.0, 100.0) for d in DAYS[:8]}}  # no bar from DAYS[8] on
    out = fixed_hold([Entry("X", DAYS[6])], panel_of(prices), DAYS, hold=3, last=DAYS[7],
                     slippage_bps=0)  # fmt: skip

    assert (len(out.spans), out.beyond_window, out.unpriced) == (0, 1, 0)


def test_draws_skip_pool_members_without_the_fixed_exit_open() -> None:
    # Z has the exit session's close but not its open: a fixed-hold draw never lands on it
    prices: dict[str, dict[date, tuple[float | None, float | None]]] = {
        "X": {d: (100.0, 100.0) for d in DAYS},
        "Y": {d: (40.0, 40.0) for d in DAYS},
        "Z": {d: (10.0, 10.0) for d in DAYS},
    }
    prices["X"][DAYS[4]] = (110.0, 110.0)
    prices["Z"][DAYS[4]] = (None, 30.0)
    panel = panel_of(prices)
    out = fixed_hold([Entry("X", DAYS[1])], panel, DAYS, hold=3, last=DAYS[-1], slippage_bps=0)

    drawn = null_returns(out.spans, panel, permutations=60, seed=1, slippage_bps=0)

    assert set(np.round(drawn[:, 0], 12)) == {0.1, 0.0}


def test_no_entries_give_an_empty_fixed_hold() -> None:
    out = fixed_hold([], split_panel(), DAYS, hold=3, last=DAYS[-1], slippage_bps=0)

    assert out == FixedHold(spans=[], beyond_window=0, unpriced=0)


@pytest.mark.parametrize("hold", [0, -1])
def test_a_hold_under_one_session_raises(hold: int) -> None:
    with pytest.raises(ValueError):
        fixed_hold([Entry("X", DAYS[1])], split_panel(), DAYS, hold=hold, last=DAYS[-1],
                   slippage_bps=0)  # fmt: skip


def test_a_real_security_outside_its_own_pool_raises() -> None:
    panel = panel_of(
        {"X": {d: (1.0, 1.0) for d in DAYS}, "Y": {d: (1.0, 1.0) for d in DAYS}},
        pools={DAYS[1]: frozenset({"Y"})},
    )

    with pytest.raises(ValueError, match="X"):
        fixed_hold([Entry("X", DAYS[1])], panel, DAYS, hold=3, last=DAYS[-1], slippage_bps=0)


def test_an_entry_session_without_a_pool_raises() -> None:
    panel = panel_of({"X": {d: (1.0, 1.0) for d in DAYS}}, pools={DAYS[2]: frozenset({"X"})})

    with pytest.raises(ValueError):
        fixed_hold([Entry("X", DAYS[1])], panel, DAYS, hold=3, last=DAYS[-1], slippage_bps=0)


# =============================================================================================
# cluster_ci
# =============================================================================================


def test_one_cluster_gives_its_mean_on_both_sides() -> None:
    low, high = cluster_ci(np.array([0.1, 0.2, 0.6]), [DAYS[0]] * 3, draws=500, seed=0)

    assert low == pytest.approx(0.3, abs=1e-12)
    assert high == pytest.approx(0.3, abs=1e-12)


def test_whole_clusters_are_resampled_not_single_entries() -> None:
    # Two sessions of fifty entries each, all 0 on one and all 1 on the other. Resampling the
    # two sessions, a quarter of the means are 0 and a quarter 1: the 90% interval is (0, 1).
    # Resampling single entries would give a tight interval around 0.5.
    excess = np.array([0.0] * 50 + [1.0] * 50)
    clusters = [DAYS[0]] * 50 + [DAYS[1]] * 50

    assert cluster_ci(excess, clusters, draws=2000, seed=0) == pytest.approx((0.0, 1.0))


def test_cluster_ci_is_deterministic_for_a_seed_and_nested_across_levels() -> None:
    rng = np.random.default_rng(11)
    excess = rng.normal(0.01, 0.05, 60)
    clusters = [DAYS[i % 10] for i in range(60)]

    a = cluster_ci(excess, clusters, draws=1000, seed=5)
    b = cluster_ci(excess, clusters, draws=1000, seed=5)
    other = cluster_ci(excess, clusters, draws=1000, seed=6)
    narrow = cluster_ci(excess, clusters, draws=1000, seed=5, level=0.5)
    wide = cluster_ci(excess, clusters, draws=1000, seed=5, level=0.99)

    assert a == b
    assert a != other
    assert a[0] < float(np.mean(excess)) < a[1]
    assert wide[0] <= a[0] <= narrow[0] <= narrow[1] <= a[1] <= wide[1]


def test_cluster_ci_refuses_mismatched_or_empty_input() -> None:
    with pytest.raises(ValueError):
        cluster_ci(np.array([0.1, 0.2]), [DAYS[0]], draws=100, seed=0)
    with pytest.raises(ValueError):
        cluster_ci(np.array([]), [], draws=100, seed=0)


# =============================================================================================
# entry_test on the B-Xtrender world (tests/evolve/test_gate_bx.py)
# =============================================================================================


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> Iterator[BxWorld]:
    yield from _world(tmp_path_factory.mktemp("entry_bx"), future=False)


@pytest.fixture(scope="module")
def future_world(tmp_path_factory: pytest.TempPathFactory) -> Iterator[BxWorld]:
    yield from _world(tmp_path_factory.mktemp("entry_bx_future"), future=True)


@pytest.fixture(scope="module")
def long_run(world: BxWorld) -> int:
    con = duckdb.connect()
    with Session(world.engine) as s:
        run_id = run_replay(
            s, con, world.root, world.strategy_dir, CAL[205], CAL[280], (LONG_WARMUP,),
            protocol=BX_GATE,
        )  # fmt: skip
    con.close()
    return run_id


@pytest.fixture
def session(world: BxWorld) -> Iterator[Session]:
    with Session(world.engine) as s:
        yield s


def entering(session: Session, run_id: int, dev: Window) -> list[Span]:
    return [s for s in spans_of_run(session, run_id) if dev.start <= s.entry_session <= dev.end]


def dev_close(dev: Window) -> datetime:
    return datetime.combine(dev.end, time(20), tzinfo=ET)


def dev_entries(
    session: Session, con: duckdb.DuckDBPyConnection, root: Path, run_id: int, dev: Window
) -> list[Entry]:
    as_of = dev_close(dev)
    calendar = visible_sessions(con, root, "1Day", as_of)
    return [e for e in entries_of_run(session, run_id, calendar, as_of=as_of)
            if dev.start <= e.entry_session <= dev.end]  # fmt: skip


def entry_spans(entries: Sequence[Entry]) -> list[Span]:
    """What build_panel needs of an entry: its session (the pools are keyed by it)."""
    return [Span(f"{e.security_id}:{e.entry_session.isoformat()}:fixed", e.entry_session,
                 e.entry_session, "open", 0.0) for e in entries]  # fmt: skip


def composed(
    session: Session,
    con: duckdb.DuckDBPyConnection,
    root: Path,
    cfgs: Mapping[int, AnyStrategyConfig],
    holds: Sequence[int],
    dev: Window,
    *,
    permutations: int = PERMUTATIONS,
    seed: int = 0,
    bootstrap: int = 2000,
    pools: Mapping[date, frozenset[str]] | None = None,
    clusters: str = "blocks",
) -> list[HoldResult]:
    """entry_test written out from its parts (see the module docstring); ``pools`` replaces the
    panel's pools when given; ``clusters="sessions"`` labels each entry by its own session
    instead of its block of ``hold`` sessions."""
    as_of = dev_close(dev)
    calendar = visible_sessions(con, root, "1Day", as_of)
    index = {d: i for i, d in enumerate(calendar)}
    real: dict[str, np.ndarray] = {}
    null: dict[str, np.ndarray] = {}
    parts: dict[tuple[int, int], tuple[FixedHold, np.ndarray]] = {}
    for run_id, cfg in cfgs.items():
        entries = dev_entries(session, con, root, run_id, dev)
        panel = build_panel(session, con, root, entry_spans(entries), cfg=cfg, as_of=as_of)
        if pools is not None:
            panel = Panel(panel.sessions, panel.open, panel.close, panel.splits, dict(pools),
                          panel.spinoffs, panel.restarts)  # fmt: skip
        bps = cfg.execute.fills.slippage_bps
        for hold in holds:
            fh = fixed_hold(entries, panel, calendar, hold=hold, last=dev.end,
                            slippage_bps=bps)  # fmt: skip
            drawn = null_returns(fh.spans, panel, permutations=permutations, seed=seed,
                                 slippage_bps=bps)  # fmt: skip
            real[f"{run_id}:{hold}"] = np.array([s.ret for s in fh.spans])
            null[f"{run_id}:{hold}"] = drawn
            parts[(run_id, hold)] = (fh, drawn)
    family = max_statistic(real, null)
    out = []
    for run_id, hold in sorted(parts):
        fh, drawn = parts[(run_id, hold)]
        values = real[f"{run_id}:{hold}"]
        out.append(
            HoldResult(
                run_id=run_id,
                hold=hold,
                result=family[f"{run_id}:{hold}"],
                ci=cluster_ci(
                    values - drawn.mean(axis=0),
                    [
                        index[s.entry_session] // hold if clusters == "blocks" else s.entry_session
                        for s in fh.spans
                    ],  # fmt: skip
                    draws=bootstrap,
                    seed=seed,
                ),  # fmt: skip
                sd_ratio=float(np.std(values) / np.std(drawn)),
                beyond_window=fh.beyond_window,
                unpriced=fh.unpriced,
            )
        )
    return out


def assert_results(got: Sequence[HoldResult], expected: Sequence[HoldResult]) -> None:
    assert [(r.run_id, r.hold) for r in got] == [(r.run_id, r.hold) for r in expected]
    for g, e in zip(got, expected, strict=True):
        assert g.result == e.result, (g.run_id, g.hold)
        assert g.ci == pytest.approx(e.ci, rel=1e-12, abs=1e-15), (g.run_id, g.hold)
        assert g.sd_ratio == pytest.approx(e.sd_ratio, rel=1e-9), (g.run_id, g.hold)
        assert (g.beyond_window, g.unpriced) == (e.beyond_window, e.unpriced)


def run_entry(
    session: Session, con: duckdb.DuckDBPyConnection, w: BxWorld, run_ids: Sequence[int],
    holds: Sequence[int], *, cfg: AnyStrategyConfig | Mapping[int, AnyStrategyConfig] | None = None,
    dev: Window = DEV, permutations: int = PERMUTATIONS,
) -> list[HoldResult]:  # fmt: skip
    return entry_test(
        session, con, w.root, run_ids, cfg=cfg if cfg is not None else w.cfg, holds=holds,
        dev=dev, permutations=permutations, seed=0,
    )  # fmt: skip


# ---------------------------------------------------------------------------------------------
# fixture guards (pass today: a failure further down is not a fixture error)
# ---------------------------------------------------------------------------------------------


def test_fixture_a_long_hold_runs_past_dev_for_the_last_entries(
    session: Session, world: BxWorld
) -> None:
    entries = entries_of(entering(session, world.run_id, DEV))
    last = max(i for i, d in enumerate(CAL) if d <= DEV_END)
    assert len(entries) >= 4
    # the last decision's position never opened inside the replay: no Trade row
    decisions = session.exec(select(Decision).where(Decision.run_id == world.run_id)).all()
    traded = {t.decision_id for t in trades(session, world.run_id)}
    assert [d.security_id for d in decisions if d.id not in traded] == ["S_DDD"]
    assert any(CAL.index(e.entry_session) + 19 > last for e in entries)
    assert any(CAL.index(e.entry_session) + 7 <= last for e in entries)
    # one BX position is two Trade rows with one key: entries are fewer than legs
    assert len(entries) < len(spans_of_run(session, world.run_id))


# ---------------------------------------------------------------------------------------------
# normal path
# ---------------------------------------------------------------------------------------------


def test_entry_test_is_its_parts(
    session: Session, con: duckdb.DuckDBPyConnection, world: BxWorld
) -> None:
    got = run_entry(session, con, world, [world.run_id], (7, 11))

    assert all(isinstance(r, HoldResult) for r in got)
    assert_results(got, composed(session, con, world.root, {world.run_id: world.cfg}, (7, 11), DEV))


def test_each_run_draws_from_its_own_pool_and_results_are_sorted_by_run_then_hold(
    session: Session, con: duckdb.DuckDBPyConnection, world: BxWorld, long_run: int
) -> None:
    cfgs = {world.run_id: world.cfg, long_run: long_cfg(world)}

    got = run_entry(session, con, world, [long_run, world.run_id], (11, 7), cfg=cfgs)

    expected = composed(session, con, world.root, cfgs, (7, 11), DEV)
    assert [(r.run_id, r.hold) for r in got] == sorted(
        [(world.run_id, 7), (world.run_id, 11), (long_run, 7), (long_run, 11)]
    )
    assert_results(got, expected)


def test_every_cell_is_corrected_over_the_whole_family_and_matches_its_parts(
    session: Session, con: duckdb.DuckDBPyConnection, world: BxWorld, long_run: int
) -> None:
    cfgs = {world.run_id: world.cfg, long_run: long_cfg(world)}
    one = run_entry(session, con, world, [world.run_id], (7,), cfg=cfgs)
    more = run_entry(session, con, world, [world.run_id, long_run], (7, 11, 19), cfg=cfgs)

    assert_results(more, composed(session, con, world.root, cfgs, (7, 11, 19), DEV))
    for r in more:
        assert r.result.p_adjusted >= r.result.p_raw, (r.run_id, r.hold)
    # six cells: the correction bites somewhere (a cell corrected alone has p adj == p raw)
    assert any(r.result.p_adjusted > r.result.p_raw for r in more)
    [alone] = one
    [same] = [r for r in more if (r.run_id, r.hold) == (world.run_id, 7)]
    assert same.result.p_raw == alone.result.p_raw  # the same draws
    assert same.result.p_adjusted >= alone.result.p_adjusted  # more cells never lower it


def test_legs_exiting_after_dev_still_count_as_entries(
    session: Session, con: duckdb.DuckDBPyConnection, world: BxWorld
) -> None:
    # Cut dev the session before a real exit that comes three or more sessions after its entry:
    # that leg's entry is measured at hold 1, though its real exit is outside dev.
    rows = trades(session, world.run_id)
    leg = next(t for t in rows
               if CAL.index(t.exit_at.astimezone(ET).date())
               - CAL.index(t.entry_at.astimezone(ET).date()) >= 3)  # fmt: skip
    cut = Window(start=DEV_START, end=CAL[CAL.index(leg.exit_at.astimezone(ET).date()) - 1])
    legs = entering(session, world.run_id, cut)
    assert any(s.exit_session > cut.end for s in legs)  # fixture

    [r] = run_entry(session, con, world, [world.run_id], (1,), dev=cut)

    last = CAL.index(cut.end)
    entries = dev_entries(session, con, world.root, world.run_id, cut)
    beyond = sum(CAL.index(e.entry_session) + 1 > last for e in entries)
    assert r.beyond_window == beyond
    assert r.result.trades + r.beyond_window + r.unpriced == len(entries)
    assert r.result.trades > sum(s.exit_session <= cut.end for s in legs if s.key.endswith(":1"))
    assert_results([r], composed(session, con, world.root, {world.run_id: world.cfg}, (1,), cut))


# ---------------------------------------------------------------------------------------------
# entries come from the run's decisions, not its closed legs
# ---------------------------------------------------------------------------------------------


def copied(world: BxWorld, tmp_path: Path) -> Path:
    """A copy of the world's database (the backup API also carries what is still in the WAL)."""
    db = tmp_path / "meta.db"
    with sqlite3.connect(world.data_dir / "meta.db") as src, sqlite3.connect(db) as dst:
        src.backup(dst)
    return db


def next_session(decided: datetime) -> date:
    night = decided.astimezone(ET).date()
    return next(d for d in CAL if d > night)


def expected_entries(session: Session, run_id: int, as_of: datetime) -> list[Entry]:
    """The run's long, uncancelled trade decisions made by as_of, each bought the next open."""
    stmt = select(Decision).where(Decision.run_id == run_id, col(Decision.as_of) <= as_of)
    found = {
        Entry(d.security_id, next_session(d.as_of))
        for d in session.exec(stmt).all()
        if d.kind == "trade" and d.direction == "long" and d.security_id is not None
        and "cancelled" not in d.payload
    }  # fmt: skip
    return sorted(found, key=lambda e: (e.entry_session, e.security_id))


def test_entries_are_the_runs_decisions_including_a_position_open_at_the_replays_end(
    session: Session, con: duckdb.DuckDBPyConnection, world: BxWorld
) -> None:
    as_of = dev_close(DEV)
    calendar = visible_sessions(con, world.root, "1Day", as_of)

    entries = entries_of_run(session, world.run_id, calendar, as_of=as_of)

    assert entries == expected_entries(session, world.run_id, as_of)
    # decided on the replay's last night, never filled inside it: still an entry
    assert Entry("S_DDD", date(2024, 2, 15)) in entries
    legs = {
        (t.security_id, t.entry_at.astimezone(ET).date()) for t in trades(session, world.run_id)
    }
    assert legs < {(e.security_id, e.entry_session) for e in entries}

    [r] = run_entry(session, con, world, [world.run_id], (1,))
    assert r.result.trades + r.beyond_window + r.unpriced == len(entries)
    assert len(entries) > len(entries_of(entering(session, world.run_id, DEV)))


def test_an_open_position_without_trade_rows_changes_no_number(
    con: duckdb.DuckDBPyConnection, world: BxWorld, session: Session, tmp_path: Path
) -> None:
    # Delete the Trade rows of the last filled position, as if the replay had ended while it
    # was open: the entry test, which never reads the real exit, must not notice.
    last = max(trades(session, world.run_id), key=lambda t: t.entry_at)
    db = copied(world, tmp_path)
    engine = make_engine(db)
    with Session(engine) as s:
        s.execute(delete(Trade).where(col(Trade.decision_id) == last.decision_id))
        s.commit()
        assert all(t.decision_id != last.decision_id for t in trades(s, world.run_id))  # fixture
        open_ = run_entry(s, con, world, [world.run_id], (7, 11))
    engine.dispose()

    assert repr(open_) == repr(run_entry(session, con, world, [world.run_id], (7, 11)))


def test_a_decision_whose_next_session_is_not_its_trades_entry_raises(
    con: duckdb.DuckDBPyConnection, world: BxWorld, session: Session, tmp_path: Path
) -> None:
    # Move S_AAA's first decision one night earlier: its derived entry is no longer the session
    # its legs entered, so a leg has no entry to belong to.
    first = min((t for t in trades(session, world.run_id) if t.security_id == "S_AAA"),
                key=lambda t: t.entry_at)  # fmt: skip
    db = copied(world, tmp_path)
    engine = make_engine(db)
    as_of = dev_close(DEV)
    calendar = visible_sessions(con, world.root, "1Day", as_of)
    with Session(engine) as s:
        decision = s.get(Decision, first.decision_id)
        assert decision is not None
        night = decision.as_of.astimezone(ET).date()
        decision.as_of = close_of(prev_session(night))
        s.add(decision)
        s.commit()
        with pytest.raises(ValueError, match="S_AAA"):
            entries_of_run(s, world.run_id, calendar, as_of=as_of)
        with pytest.raises(ValueError, match="S_AAA"):
            run_entry(s, con, world, [world.run_id], (7,))
    engine.dispose()


def test_cancelled_and_not_long_decisions_are_not_entries(
    con: duckdb.DuckDBPyConnection, world: BxWorld, session: Session, tmp_path: Path
) -> None:
    as_of = dev_close(DEV)
    calendar = visible_sessions(con, world.root, "1Day", as_of)
    before = entries_of_run(session, world.run_id, calendar, as_of=as_of)
    traded = {t.decision_id for t in trades(session, world.run_id)}
    db = copied(world, tmp_path)
    engine = make_engine(db)
    with Session(engine) as s:
        [unfilled] = [d for d in s.exec(select(Decision).where(Decision.run_id == world.run_id))
                      if d.id not in traded]  # fmt: skip
        unfilled.payload = {**unfilled.payload, "cancelled": "no_entry"}
        s.add(unfilled)
        for direction in ("flat", "short"):
            s.add(Decision(run_id=world.run_id, kind="trade", as_of=close_of(date(2024, 1, 22)),
                           strategy_id="bx_trender", security_id="S_CCC", symbol="CCC",
                           direction=direction))  # fmt: skip
        s.commit()
        after = entries_of_run(s, world.run_id, calendar, as_of=as_of)
    engine.dispose()

    assert after == [e for e in before if e != Entry("S_DDD", date(2024, 2, 15))]
    assert Entry("S_CCC", date(2024, 1, 23)) not in after


def test_a_decision_after_as_of_is_not_read(session: Session, world: BxWorld) -> None:
    # At 2024-01-31 the position decided on 2024-02-02 is not decided yet: its leg entering
    # 2024-02-05 is not checked against it, and it is no entry.
    as_of = close_of(date(2024, 1, 31))

    entries = entries_of_run(session, world.run_id, CAL, as_of=as_of)

    assert entries == expected_entries(session, world.run_id, as_of)
    assert entries and max(e.entry_session for e in entries) <= date(2024, 2, 1)
    assert len(entries) < len(expected_entries(session, world.run_id, dev_close(DEV)))


def test_entries_of_an_unknown_run_raise_lookup_error(session: Session) -> None:
    with pytest.raises(LookupError):
        entries_of_run(session, 9999, CAL, as_of=dev_close(DEV))


def test_entry_test_measures_b_xtrender_runs_only(
    session: Session, con: duckdb.DuckDBPyConnection, world: BxWorld
) -> None:
    cfg = world.cfg
    gap = cfg.model_copy(
        update={
            "strategy": cfg.strategy.model_copy(update={"universe": "watchlist_with_documents"})
        }
    )
    with pytest.raises(ValueError):
        run_entry(session, con, world, [world.run_id], (7,), cfg=gap)


def test_the_interval_resamples_blocks_of_hold_sessions(
    session: Session, con: duckdb.DuckDBPyConnection, world: BxWorld
) -> None:
    # Entries a few sessions apart hold overlapping fixed windows: they are one block, not
    # independent draws.
    cfgs = {world.run_id: world.cfg}
    [got] = run_entry(session, con, world, [world.run_id], (11,))
    [blocks] = composed(session, con, world.root, cfgs, (11,), DEV)
    [sessions] = composed(session, con, world.root, cfgs, (11,), DEV, clusters="sessions")

    assert blocks.ci != sessions.ci  # fixture: the labels make a difference here
    assert got.ci == pytest.approx(blocks.ci, rel=1e-12, abs=1e-15)


# ---------------------------------------------------------------------------------------------
# boundaries
# ---------------------------------------------------------------------------------------------


def test_a_fixed_exit_past_dev_is_counted_and_not_priced(
    session: Session, con: duckdb.DuckDBPyConnection, world: BxWorld
) -> None:
    entries = dev_entries(session, con, world.root, world.run_id, DEV)
    last = max(i for i, d in enumerate(CAL) if d <= DEV_END)

    got = {r.hold: r for r in run_entry(session, con, world, [world.run_id], (7, 19))}

    for hold, r in got.items():
        beyond = sum(CAL.index(e.entry_session) + hold > last for e in entries)
        assert r.beyond_window == beyond, hold
        assert r.result.trades + r.beyond_window + r.unpriced == len(entries), hold
    assert got[19].beyond_window > 0


@pytest.mark.parametrize("holds", [(7, 7), (), (0,), (7, -3)])
def test_bad_holds_raise(
    session: Session, con: duckdb.DuckDBPyConnection, world: BxWorld, holds: tuple[int, ...]
) -> None:
    with pytest.raises(ValueError):
        run_entry(session, con, world, [world.run_id], holds)


def test_a_run_without_an_entry_in_dev_raises(
    session: Session, con: duckdb.DuckDBPyConnection, world: BxWorld
) -> None:
    before = Window(start=date(2023, 3, 1), end=date(2023, 9, 29))
    with pytest.raises(ValueError):
        run_entry(session, con, world, [world.run_id], (7,), dev=before)


def test_an_unknown_run_raises_lookup_error(
    session: Session, con: duckdb.DuckDBPyConnection, world: BxWorld
) -> None:
    with pytest.raises(LookupError):
        run_entry(session, con, world, [world.run_id, 9999], (7,))


# ---------------------------------------------------------------------------------------------
# leakage
# ---------------------------------------------------------------------------------------------


def test_bars_and_records_after_dev_change_no_number(
    session: Session, con: duckdb.DuckDBPyConnection, world: BxWorld, future_world: BxWorld
) -> None:
    # The future world prices every bar from 2024-03-01 at ten times and files a 2:1 AAA split
    # inside the replay on 2024-04-15. Hold 19 has fixed exits after 2024-02-29: they are
    # dropped, not priced from the wild bars; the late split is not on file at dev's end.
    clean = run_entry(session, con, world, [world.run_id], (7, 19))
    with Session(future_world.engine) as s:
        wild = run_entry(s, con, future_world, [future_world.run_id], (7, 19), cfg=future_world.cfg)

    assert [r.beyond_window for r in clean if r.hold == 19] != [0]  # fixture: the case is hit
    assert repr(clean) == repr(wild)


@pytest.fixture
def jump_run(world: BxWorld, tmp_path: Path) -> tuple[Path, int]:
    """A copy of the database with one more BX run whose real AAA legs enter on JUMP_DAY (JMP
    thin the evening before, inside the tier that evening) and on the session after it, each
    with its decision the evening before."""
    db = tmp_path / "meta.db"
    with sqlite3.connect(world.data_dir / "meta.db") as src, sqlite3.connect(db) as dst:
        src.backup(dst)
    engine = make_engine(db)
    with Session(engine) as s:
        template = trades(s, world.run_id)[0].model_dump(exclude={"id"})
        run_id = new_run(s, None, "bx_trender")
        i = CAL.index(JUMP_DAY)
        for entry in (CAL[i], CAL[i + 1]):
            decision = Decision(
                run_id=run_id, kind="trade", as_of=close_of(prev_session(entry)),
                strategy_id="bx_trender", security_id="S_AAA", symbol="AAA", direction="long",
            )  # fmt: skip
            s.add(decision)
            s.flush()
            row = dict(template)
            row.update(
                decision_id=decision.id,
                run_id=run_id,
                security_id="S_AAA",
                symbol="AAA",
                entry_at=datetime.combine(entry, time(9, 30), tzinfo=ET),
                exit_at=datetime.combine(CAL[CAL.index(entry) + 5], time(9, 30), tzinfo=ET),
                available_at=close_of(CAL[CAL.index(entry) + 5]),
            )
            s.add(Trade(**row))
        s.commit()
    engine.dispose()
    return db, run_id


def test_draws_come_from_the_scan_pool_of_the_evening_before_entry(
    con: duckdb.DuckDBPyConnection, world: BxWorld, jump_run: tuple[Path, int]
) -> None:
    db, run_id = jump_run
    i = CAL.index(JUMP_DAY)
    window = Window(start=CAL[i - 5], end=CAL[i + 15])
    cfg = world.cfg
    evening = {e: pool(con, world.root, close_of(prev_session(e)), cfg.scan)
               for e in (CAL[i], CAL[i + 1])}  # fmt: skip
    assert "S_JMP" not in evening[CAL[i]] and "S_JMP" in evening[CAL[i + 1]]  # fixture
    assert "S_AAA" in evening[CAL[i]] and len(evening[CAL[i]]) >= 3  # fixture

    engine = make_engine(db)
    with Session(engine) as s:
        got = run_entry(s, con, world, [run_id], (3,), dev=window)
        expected = composed(s, con, world.root, {run_id: cfg}, (3,), window, pools=evening)
    engine.dispose()

    assert_results(got, expected)


# =============================================================================================
# CLI: asof gate entry
# =============================================================================================


@pytest.fixture
def gate_protocol(monkeypatch: pytest.MonkeyPatch) -> EvaluationProtocol:
    monkeypatch.setattr(asof.cli, "load_protocol", lambda *args, **kwargs: BX_GATE)
    return BX_GATE


def holds_args(*holds: int) -> tuple[str, ...]:
    return tuple(a for h in holds for a in ("--hold", str(h)))


def test_gate_entry_prints_one_row_per_run_and_hold(
    session: Session, con: duckdb.DuckDBPyConnection, world: BxWorld,
    gate_protocol: EvaluationProtocol,
) -> None:  # fmt: skip
    result = runner.invoke(
        app,
        cli_args(world, "entry", world.run_id, db=world.data_dir / "meta.db",
                 extra=holds_args(11, 7)),
    )  # fmt: skip

    assert result.exit_code == 0, result.output
    lines = [line for line in result.output.splitlines() if line.strip()]
    header, rows = lines[0].lower(), [line.split() for line in lines[1:]]
    for column in ("run", "hold", "entries", "ci low", "ci high", "sd ratio", "p adj"):
        assert column in header, column
    assert lines[-1].split() == ["family:", f"{world.run_id}:7", f"{world.run_id}:11"]
    rows = rows[:-1]
    expected = run_entry(session, con, world, [world.run_id], (7, 11))
    assert [(int(r[0]), int(r[1])) for r in rows] == [(world.run_id, 7), (world.run_id, 11)]
    for row, e in zip(rows, expected, strict=True):
        assert int(row[2]) == e.result.trades
        assert f"{e.result.p_adjusted:.4f}" in row
        assert (int(row[-2]), int(row[-1])) == (e.beyond_window, e.unpriced)


def test_gate_entry_reads_dev_from_the_protocol(
    world: BxWorld, monkeypatch: pytest.MonkeyPatch
) -> None:
    # a protocol whose dev ends before the run's first entry: nothing to measure, one line
    early = protocol(
        dev={"start": date(2023, 3, 1), "end": date(2023, 9, 29)},
        holdout={"start": date(2023, 10, 2), "end": date(2024, 6, 28)},
        clean_after=date(2024, 7, 1),
        seen_before_protocol=[],
    )
    monkeypatch.setattr(asof.cli, "load_protocol", lambda *args, **kwargs: early)

    result = runner.invoke(
        app, cli_args(world, "entry", world.run_id, db=world.data_dir / "meta.db",
                      extra=holds_args(7))
    )  # fmt: skip

    assert result.exit_code == 1, result.output
    one_line_failure(result.output)


@pytest.mark.parametrize("holds", [(7, 7), (0,)])
def test_gate_entry_refuses_bad_holds_in_one_line(
    world: BxWorld, gate_protocol: EvaluationProtocol, holds: tuple[int, ...]
) -> None:
    result = runner.invoke(
        app, cli_args(world, "entry", world.run_id, db=world.data_dir / "meta.db",
                      extra=holds_args(*holds))
    )  # fmt: skip

    assert result.exit_code == 1, result.output
    one_line_failure(result.output)


def test_gate_entry_refuses_an_unknown_run_in_one_line(
    world: BxWorld, gate_protocol: EvaluationProtocol
) -> None:
    result = runner.invoke(
        app, cli_args(world, "entry", world.run_id, 9999, db=world.data_dir / "meta.db",
                      extra=holds_args(7))
    )  # fmt: skip

    assert result.exit_code == 1, result.output
    one_line_failure(result.output)


def test_gate_entry_checks_the_strategy_version(
    world: BxWorld, gate_protocol: EvaluationProtocol, tmp_path: Path
) -> None:
    # Today's files differ from the version the run replayed: refused unless --current-strategy.
    strategies = tmp_path / "strategies"
    shutil.copytree(world.data_dir / "strategies", strategies)
    bx_yaml = next(strategies.glob("*/execute.yaml"))
    bx_yaml.write_text(bx_yaml.read_text() + "\n# edited\n")

    def invoke(*extra: str) -> Result:
        args = cli_args(world, "entry", world.run_id, db=world.data_dir / "meta.db",
                        extra=(*holds_args(7), *extra))  # fmt: skip
        args[args.index("--strategies-dir") + 1] = str(strategies)
        return runner.invoke(app, args)

    refused, allowed = invoke(), invoke("--current-strategy")

    assert refused.exit_code == 1, refused.output
    assert "--current-strategy" in refused.output
    assert allowed.exit_code == 0, allowed.output
