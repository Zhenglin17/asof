"""Guards against a random-entry null that lets a strategy pass the gate for the wrong reason: a
null drawn from securities the strategy could not have picked that day (filed later, too thin
or too cheap for the scanner, a warrant, no price),
a split read as a crash, draws that change with span order or the Python process, a max-statistic
correction that is too loose (a family of 20 pure-luck variants passing more than about 5% of
the time) or that a duplicate variant can game, and prices or filings from after the window.

API under test (asof.evolve.null, new module):

    Span(key, entry_session, exit_session, exit_at, ret)
    Panel(sessions, open, close, splits, pools)
    null_returns(spans, panel, *, permutations, seed, slippage_bps) -> np.ndarray (K, n)
    VariantResult(trades, mean, null_mean, excess, z, p_raw, p_adjusted)
    max_statistic(real, null) -> dict[str, VariantResult]
    spans_of_run(session, run_id) -> list[Span]
    build_panel(session, con, root, spans, *, cfg: StrategyConfig, as_of) -> Panel
        pools[d] = asof.scan.earnings_gap.pool(..., 20:00 New York of the session before d, cfg)
    null_test(session, con, root, run_ids, *, cfg: StrategyConfig, permutations, seed)
        -> dict[int, VariantResult]; slippage from cfg.execute.fills.slippage_bps

Interface points this file pins down where the spec left a choice:
- ``z`` uses the population standard deviation of the K per-permutation means (numpy's default
  ``ddof=0``). p-values do not depend on this choice; only ``z`` does.
- ``max_statistic`` raises ValueError when ``real`` and ``null`` have different variant names or
  a null's column count differs from its real's length.
- ``spans_of_run`` of a run that exists but has no trades returns ``[]`` (``null_test`` is the
  one that refuses it).
- ``build_panel``'s ``pools`` has a key for every span's entry session (it may have more);
  ``sessions`` covers every session from the first entry to the last exit and none after
  ``as_of`` (where it starts is not pinned: it may reach back for the pool's lookback).
- An unknown ``cfg.strategy.universe`` (only constructible through ``model_copy``) raises
  ValueError in ``build_panel`` and ``null_test``.

Fake data provenance:
- Pure-layer prices are random walks or hand-picked numbers; nothing in them is meant to look
  like real data. Calibration worlds: 30 securities, 150 sessions, every security in every pool.
- Split ratio = new_rate / old_rate, applied to ex-dates in (entry, exit], prices raw (not
  adjusted): documented in asof.replay.run._split_ratio and asof.evolve.benchmark.
- Trade times: entry 09:30 New York; exits at 09:30 (break, timeouts, gap stops), 16:00 (an
  intraday hard stop) or 20:00 (data-gap exit): documented in asof.replay.run._fill_time.
- A daily bar is visible from 20:00 New York of its session (asof.ingest.bars); a split record
  from 00:00 New York of its ex-date (asof.ingest.corporate_actions); a document from its
  available_at (asof.store.queries).
- Calendar: weekdays 2025-01-02 .. 2025-02-21 minus 2025-01-09, 2025-01-20 and 2025-02-17, the
  real NYSE closures. Prices, volumes and filing times are made up so that each name sits
  clearly on one side of the approved $5 / $20M thresholds (tests/strategy/files.py), except H,
  whose volume jumps on 02-06 to show the pool's average is read the evening before entry.

Assumptions these fixtures cannot settle; check them on the real store:
- Every real Trade's entry_at is 09:30 New York and every exit_at is 09:30, 16:00 or 20:00 New
  York. Check: SELECT the distinct New York clock times of trade.entry_at / trade.exit_at over
  all replay runs; anything else must be zero.
- Trade.pnl / (shares * entry_price) is the leg's return net of the replay's slippage, so the
  null must use the same slippage_bps the replay used. Check: for a recorded run, recompute one
  leg from its bars (exit * split factor * (1 - bps) / (entry_open * (1 + bps)) - 1) and compare
  with the stored pnl ratio.
- A security's entry-session open in the panel equals the price the replay filled at before
  slippage. Check: for every trade of one real run, panel.open[security_id][entry_session]
  times (1 + bps) equals Trade.entry_price.
- The pool (scan() minus the gap and rvol filters, decided the evening before entry) contains
  every security the replay actually traded. Check: every Trade.security_id of a real run is in
  panel.pools[entry_session]; a real leg outside its own pool means the null draws from the
  wrong set.
"""

import json
import math
import os
import subprocess
import sys
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sqlmodel import Session, select

from asof.evolve.null import (
    Panel,
    Span,
    VariantResult,
    build_panel,
    max_statistic,
    null_returns,
    null_test,
    spans_of_run,
)
from asof.ingest.alpaca import Bar
from asof.ingest.corporate_actions import SPLIT_SCHEMA
from asof.store.market import market_root
from asof.store.models import Entity, Trade
from asof.strategy import StrategyConfig, apply_overrides, load_strategy
from tests.ingest.fakes import daily
from tests.store.factories import (
    add,
    new_document,
    new_entity,
    new_run,
    new_source,
    new_trade_decision,
    pk,
)
from tests.store.test_filer_securities import CikSeg, write_cik_master
from tests.store.test_universe import store_daily
from tests.strategy.files import write_strategy

ET = ZoneInfo("America/New_York")
ExitAt = Literal["open", "close"]


# =============================================================================================
# pure-layer builders
# =============================================================================================


def weekdays(n: int, start: date = date(2024, 1, 1)) -> tuple[date, ...]:
    days: list[date] = []
    day = start
    while len(days) < n:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return tuple(days)


DAYS = weekdays(10)  # 2024-01-01 .. 2024-01-12


def panel_of(
    prices: Mapping[str, Mapping[date, tuple[float | None, float | None]]],
    *,
    sessions: Sequence[date] = DAYS,
    pools: Mapping[date, frozenset[str]] | None = None,
    splits: Mapping[str, Sequence[tuple[date, float]]] | None = None,
) -> Panel:
    """``prices[sid][day] = (open, close)``; None leaves that price out. Default pools: every
    priced security on every session."""
    opens = {s: {d: o for d, (o, _) in by.items() if o is not None} for s, by in prices.items()}
    closes = {s: {d: c for d, (_, c) in by.items() if c is not None} for s, by in prices.items()}
    if pools is None:
        pools = {d: frozenset(prices) for d in sessions}
    return Panel(
        sessions=tuple(sessions),
        open=opens,
        close=closes,
        splits=dict(splits or {}),
        pools=dict(pools),
    )


def flat(
    open_: float, close: float, days: Sequence[date] = DAYS
) -> dict[date, tuple[float, float]]:
    return {d: (open_, close) for d in days}


def span(key: str, entry: date, exit_: date, exit_at: ExitAt = "close", ret: float = 0.0) -> Span:
    return Span(key=key, entry_session=entry, exit_session=exit_, exit_at=exit_at, ret=ret)


def leg_return(panel: Panel, sid: str, s: Span, slippage_bps: float) -> float:
    """The spec's leg return, written out independently of the module."""
    bps = slippage_bps / 10_000
    entry = panel.open[sid][s.entry_session]
    exit_px = (panel.open if s.exit_at == "open" else panel.close)[sid][s.exit_session]
    factor = math.prod(
        r for d, r in panel.splits.get(sid, ()) if s.entry_session < d <= s.exit_session
    )
    return exit_px * factor * (1 - bps) / (entry * (1 + bps)) - 1


# =============================================================================================
# null_returns: normal path
# =============================================================================================


def test_one_eligible_security_gives_its_own_return_in_every_permutation() -> None:
    panel = panel_of({"X": {DAYS[0]: (100.0, 101.0), DAYS[3]: (108.0, 110.0)}})
    spans = [span("X:2024-01-01:1", DAYS[0], DAYS[3], "close", ret=0.07)]

    out = null_returns(spans, panel, permutations=50, seed=1, slippage_bps=10)

    assert out.shape == (50, 1)
    expected = 110 * (1 - 0.001) / (100 * (1 + 0.001)) - 1
    assert out[:, 0] == pytest.approx(np.full(50, expected))


def test_exit_at_open_prices_the_exit_at_that_sessions_open() -> None:
    panel = panel_of({"X": {DAYS[0]: (100.0, 90.0), DAYS[2]: (120.0, 80.0)}})

    at_open = null_returns(
        [span("k", DAYS[0], DAYS[2], "open")], panel, permutations=3, seed=0, slippage_bps=0
    )
    at_close = null_returns(
        [span("k", DAYS[0], DAYS[2], "close")], panel, permutations=3, seed=0, slippage_bps=0
    )

    # Entry is always the entry session's open (100), never its close (90).
    assert at_open[:, 0] == pytest.approx([0.2, 0.2, 0.2])
    assert at_close[:, 0] == pytest.approx([-0.2, -0.2, -0.2])


def test_columns_follow_span_order_and_each_value_is_an_eligible_securitys_return() -> None:
    # Four securities with distinct span returns, so a value names the security drawn.
    prices = {
        "A": flat(100.0, 101.0),
        "B": flat(100.0, 102.0),
        "C": flat(100.0, 103.0),
        "D": flat(100.0, 104.0),
    }
    panel = panel_of(prices)
    spans = [span("s1", DAYS[0], DAYS[1]), span("s2", DAYS[4], DAYS[8])]

    out = null_returns(spans, panel, permutations=400, seed=7, slippage_bps=0)

    assert out.shape == (400, 2)
    allowed = {0.01, 0.02, 0.03, 0.04}
    for value in out.ravel():
        assert min(abs(value - a) for a in allowed) < 1e-12
    # Every security turns up in 400 draws (each has a 1/4 chance per draw).
    for j in range(2):
        assert {round(v, 6) for v in out[:, j]} == allowed


def test_draws_are_uniform_over_the_eligible_set() -> None:
    prices = {f"S{i}": flat(100.0, 100.0 + i + 1) for i in range(5)}
    panel = panel_of(prices)

    out = null_returns(
        [span("u", DAYS[0], DAYS[5])], panel, permutations=10_000, seed=3, slippage_bps=0
    )

    counts = np.unique(np.round(out[:, 0], 6), return_counts=True)[1]
    assert len(counts) == 5
    # Expected 2000 each; binomial sd = 40, so +-200 is five sd.
    assert all(1800 <= c <= 2200 for c in counts)


# =============================================================================================
# null_returns: boundaries
# =============================================================================================


def test_split_inside_the_hold_multiplies_the_exit_price() -> None:
    # 4:1 split with ex-date DAYS[2]: raw prices drop 4x from that session on.
    prices = {"X": {DAYS[0]: (100.0, 100.0), DAYS[2]: (25.0, 25.5), DAYS[3]: (25.5, 26.0)}}
    panel = panel_of(prices, splits={"X": [(DAYS[2], 4.0)]})

    out = null_returns(
        [span("k", DAYS[0], DAYS[3], "close")], panel, permutations=2, seed=0, slippage_bps=0
    )

    assert out[:, 0] == pytest.approx([0.04, 0.04])  # 26 * 4 / 100 - 1


def test_split_on_the_exit_session_applies_and_on_the_entry_session_does_not() -> None:
    prices = {
        "X": {DAYS[0]: (100.0, 100.0), DAYS[1]: (50.0, 50.0), DAYS[2]: (55.0, 55.0)},
    }
    panel = panel_of(prices, splits={"X": [(DAYS[1], 2.0)]})

    # Ex-date == exit session, exiting at the open: that open is already post-split.
    on_exit = null_returns(
        [span("k", DAYS[0], DAYS[1], "open")], panel, permutations=1, seed=0, slippage_bps=0
    )
    # Ex-date == entry session: the entry open is already post-split, nothing to undo.
    on_entry = null_returns(
        [span("k", DAYS[1], DAYS[2], "close")], panel, permutations=1, seed=0, slippage_bps=0
    )

    assert on_exit[0, 0] == pytest.approx(0.0)  # 50 * 2 / 100 - 1
    assert on_entry[0, 0] == pytest.approx(0.1)  # 55 / 50 - 1


def test_splits_outside_the_hold_are_ignored_and_two_inside_multiply() -> None:
    prices = {"X": {d: (100.0, 100.0) for d in DAYS}}
    splits = {"X": [(DAYS[0], 3.0), (DAYS[2], 2.0), (DAYS[4], 1.5), (DAYS[8], 10.0)]}
    panel = panel_of(prices, splits=splits)

    out = null_returns(
        [span("k", DAYS[0], DAYS[5], "close")], panel, permutations=1, seed=0, slippage_bps=0
    )

    assert out[0, 0] == pytest.approx(100 * 2.0 * 1.5 / 100 - 1)


def test_a_split_dated_on_a_non_session_day_inside_the_hold_still_applies() -> None:
    friday, monday = DAYS[4], DAYS[5]  # 2024-01-05, 2024-01-08
    saturday = friday + timedelta(days=1)
    panel = panel_of(
        {"X": {friday: (100.0, 100.0), monday: (50.0, 51.0)}}, splits={"X": [(saturday, 2.0)]}
    )

    out = null_returns(
        [span("k", friday, monday, "close")], panel, permutations=1, seed=0, slippage_bps=0
    )

    assert out[0, 0] == pytest.approx(0.02)


def test_eligibility_needs_the_pool_an_entry_open_and_the_right_exit_price() -> None:
    entry, exit_ = DAYS[0], DAYS[2]
    prices: dict[str, dict[date, tuple[float | None, float | None]]] = {
        "OK": {entry: (100.0, 100.0), exit_: (110.0, 120.0)},
        "NO_ENTRY_OPEN": {entry: (None, 100.0), exit_: (500.0, 500.0)},
        "NO_EXIT_CLOSE": {entry: (100.0, 100.0), exit_: (500.0, None)},
        "NO_EXIT_OPEN": {entry: (100.0, 100.0), exit_: (None, 300.0)},
        "OUT_OF_POOL": {entry: (100.0, 100.0), exit_: (900.0, 900.0)},
    }
    in_pool = frozenset(prices) - {"OUT_OF_POOL"}
    panel = panel_of(prices, pools={d: in_pool for d in DAYS})

    close_exit = null_returns(
        [span("c", entry, exit_, "close")], panel, permutations=300, seed=5, slippage_bps=0
    )
    open_exit = null_returns(
        [span("o", entry, exit_, "open")], panel, permutations=300, seed=5, slippage_bps=0
    )

    # Close exits: OK (0.2) and NO_EXIT_OPEN (2.0); open exits: OK (0.1) and NO_EXIT_CLOSE (4.0).
    assert {round(v, 9) for v in close_exit[:, 0]} == {0.2, 2.0}
    assert {round(v, 9) for v in open_exit[:, 0]} == {0.1, 4.0}


def test_a_span_with_no_eligible_security_raises_naming_its_key_and_date() -> None:
    panel = panel_of({"X": {DAYS[0]: (100.0, 100.0)}})  # no price on DAYS[3]

    with pytest.raises(ValueError, match=r"X:2024-01-01:1") as info:
        null_returns(
            [span("X:2024-01-01:1", DAYS[0], DAYS[3])],
            panel,
            permutations=5,
            seed=0,
            slippage_bps=0,
        )
    assert "2024-01-01" in str(info.value)


def test_an_empty_pool_raises() -> None:
    panel = panel_of({"X": flat(100.0, 100.0)}, pools={d: frozenset() for d in DAYS})

    with pytest.raises(ValueError, match="k1"):
        null_returns([span("k1", DAYS[0], DAYS[1])], panel, permutations=5, seed=0, slippage_bps=0)


def test_a_missing_pool_for_the_entry_session_raises() -> None:
    panel = panel_of({"X": flat(100.0, 100.0)}, pools={DAYS[1]: frozenset({"X"})})

    with pytest.raises(ValueError, match="2024-01-01"):
        null_returns([span("k", DAYS[0], DAYS[1])], panel, permutations=5, seed=0, slippage_bps=0)


@pytest.mark.parametrize("permutations", [0, -1])
def test_fewer_than_one_permutation_raises(permutations: int) -> None:
    panel = panel_of({"X": flat(100.0, 100.0)})

    with pytest.raises(ValueError):
        null_returns(
            [span("k", DAYS[0], DAYS[1])],
            panel,
            permutations=permutations,
            seed=0,
            slippage_bps=0,
        )


def test_no_spans_gives_k_rows_and_zero_columns() -> None:
    panel = panel_of({"X": flat(100.0, 100.0)})

    out = null_returns([], panel, permutations=4, seed=0, slippage_bps=0)

    assert out.shape == (4, 0)


# =============================================================================================
# null_returns: determinism
# =============================================================================================


def five_way_panel() -> Panel:
    return panel_of({f"S{i}": flat(100.0, 101.0 + i) for i in range(5)})


FIVE_SPANS = [
    span("S0:2024-01-01:1", DAYS[0], DAYS[2]),
    span("S3:2024-01-02:1", DAYS[1], DAYS[6], "close"),
    span("S1:2024-01-02:2", DAYS[1], DAYS[4]),
    span("S4:2024-01-08:1", DAYS[5], DAYS[9]),
]


def test_same_inputs_and_seed_give_identical_arrays() -> None:
    a = null_returns(FIVE_SPANS, five_way_panel(), permutations=64, seed=11, slippage_bps=5)
    b = null_returns(FIVE_SPANS, five_way_panel(), permutations=64, seed=11, slippage_bps=5)

    assert np.array_equal(a, b)


def test_a_different_seed_gives_different_draws() -> None:
    a = null_returns(FIVE_SPANS, five_way_panel(), permutations=64, seed=11, slippage_bps=5)
    b = null_returns(FIVE_SPANS, five_way_panel(), permutations=64, seed=12, slippage_bps=5)

    assert not np.array_equal(a, b)


def test_reordering_spans_reorders_columns_only() -> None:
    order = [2, 0, 3, 1]
    a = null_returns(FIVE_SPANS, five_way_panel(), permutations=64, seed=11, slippage_bps=5)
    b = null_returns(
        [FIVE_SPANS[i] for i in order], five_way_panel(), permutations=64, seed=11, slippage_bps=5
    )

    assert np.array_equal(b, a[:, order])


def test_a_spans_draws_do_not_depend_on_which_other_spans_are_present() -> None:
    together = null_returns(FIVE_SPANS, five_way_panel(), permutations=64, seed=11, slippage_bps=5)
    for j, s in enumerate(FIVE_SPANS):
        alone = null_returns([s], five_way_panel(), permutations=64, seed=11, slippage_bps=5)
        assert np.array_equal(alone[:, 0], together[:, j])


def test_the_same_key_draws_the_same_securities_whatever_its_return() -> None:
    # Two runs record the same leg; the null must not care about the leg's recorded return.
    a = null_returns(
        [span("S0:2024-01-01:1", DAYS[0], DAYS[2], ret=0.5)],
        five_way_panel(),
        permutations=64,
        seed=11,
        slippage_bps=5,
    )
    b = null_returns(
        [span("S0:2024-01-01:1", DAYS[0], DAYS[2], ret=-0.3)],
        five_way_panel(),
        permutations=64,
        seed=11,
        slippage_bps=5,
    )

    assert np.array_equal(a, b)


def test_one_leg_with_two_exits_draws_the_same_security_where_both_are_priced() -> None:
    # Two variants hold the same leg but exit on different days; NOEXIT has no price on the
    # later exit, so only the later span skips it. Every security's return is distinct.
    prices: dict[str, dict[date, tuple[float | None, float | None]]] = {
        f"S{i}": {d: (100.0, 100.0 + i + 1 + 0.1 * j) for j, d in enumerate(DAYS)} for i in range(5)
    }
    prices["NOEXIT"] = {d: (100.0, 150.0) for d in DAYS if d != DAYS[6]}
    panel = panel_of(prices)
    early = span("S0:2024-01-01:1", DAYS[0], DAYS[2])
    late = span("S0:2024-01-01:1", DAYS[0], DAYS[6])

    a = null_returns([early], panel, permutations=500, seed=4, slippage_bps=0)[:, 0]
    b = null_returns([late], panel, permutations=500, seed=4, slippage_bps=0)[:, 0]

    def sid_early(v: float) -> str:
        return min(panel.close, key=lambda s: abs(leg_return(panel, s, early, 0) - v))

    def sid_late(v: float) -> str:
        return min(
            (s for s in panel.close if s != "NOEXIT"),
            key=lambda s: abs(leg_return(panel, s, late, 0) - v),
        )

    both = [k for k in range(500) if sid_early(a[k]) != "NOEXIT"]
    assert len(both) > 300
    assert all(sid_early(a[k]) == sid_late(b[k]) for k in both)


_CHILD = """
import json, sys
from datetime import date
from asof.evolve.null import Panel, Span, null_returns

spec = json.loads(sys.argv[1])
days = [date.fromisoformat(d) for d in spec["days"]]
sids = spec["sids"]
panel = Panel(
    sessions=tuple(days),
    open={s: {d: 100.0 for d in days} for s in sids},
    close={s: {d: 101.0 + i for d in days} for i, s in enumerate(sids)},
    splits={},
    pools={d: frozenset(sids) for d in days},
)
spans = [Span(k, days[0], days[3], "close", 0.0) for k in spec["keys"]]
out = null_returns(spans, panel, permutations=32, seed=99, slippage_bps=0)
print(json.dumps([[round(v, 9) for v in row] for row in out.tolist()]))
"""


def test_draws_are_stable_across_processes_with_different_hash_seeds() -> None:
    # Python salts str hashes per process; frozenset order and hash(key) both change with it.
    spec = {
        "days": [d.isoformat() for d in DAYS[:4]],
        "sids": [f"SEC{i}" for i in range(6)],
        "keys": ["SEC0:2024-01-01:1", "SEC5:2024-01-01:2", "ZZZ:2024-01-01:1"],
    }
    outputs = []
    for hash_seed in ("0", "1", "4242"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        done = subprocess.run(
            [sys.executable, "-c", _CHILD, json.dumps(spec)],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.append(json.loads(done.stdout))

    assert outputs[0] == outputs[1] == outputs[2]
    # The same computation in this process agrees too.
    days = DAYS[:4]
    sids = spec["sids"]
    panel = Panel(
        sessions=tuple(days),
        open={s: {d: 100.0 for d in days} for s in sids},
        close={s: {d: 101.0 + i for d in days} for i, s in enumerate(sids)},
        splits={},
        pools={d: frozenset(sids) for d in days},
    )
    here = null_returns(
        [Span(k, days[0], days[3], "close", 0.0) for k in spec["keys"]],
        panel,
        permutations=32,
        seed=99,
        slippage_bps=0,
    )
    assert [[round(v, 9) for v in row] for row in here.tolist()] == outputs[0]


# =============================================================================================
# max_statistic: normal path and properties
# =============================================================================================


def test_hand_computed_family_of_one() -> None:
    real = {"v": np.array([0.02, 0.04])}
    # Per-permutation means 0.0, 0.01, 0.02, 0.05: one of four reaches the real mean 0.03.
    null = {"v": np.array([[0.0, 0.0], [0.01, 0.01], [0.0, 0.04], [0.05, 0.05]])}

    got = max_statistic(real, null)

    sd = math.sqrt((0.02**2 + 0.01**2 + 0.0**2 + 0.03**2) / 4)  # ddof=0
    assert set(got) == {"v"}
    r = got["v"]
    assert isinstance(r, VariantResult)
    assert r.trades == 2
    assert r.mean == pytest.approx(0.03)
    assert r.null_mean == pytest.approx(0.02)
    assert r.excess == pytest.approx(0.01)
    assert r.z == pytest.approx(0.01 / sd)
    assert r.p_raw == pytest.approx(2 / 5)
    assert r.p_adjusted == pytest.approx(2 / 5)


def test_a_mean_above_every_permutation_gets_the_smallest_p() -> None:
    real = {"v": np.array([1.0, 1.0])}
    null = {"v": np.array([[0.0, 0.1], [0.2, 0.0], [0.0, 0.0]])}  # means 0.05, 0.1, 0

    r = max_statistic(real, null)["v"]

    assert r.p_raw == pytest.approx(1 / 4)
    assert r.p_adjusted == pytest.approx(1 / 4)


def random_family(
    rng: np.random.Generator, variants: int, k: int, shift: float = 0.0
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    real: dict[str, np.ndarray] = {}
    null: dict[str, np.ndarray] = {}
    for v in range(variants):
        n = int(rng.integers(3, 12))
        real[f"v{v}"] = rng.normal(shift, 0.05, n)
        null[f"v{v}"] = rng.normal(0.0, 0.05, (k, n))
    return real, null


def single_p(real: np.ndarray, null: np.ndarray) -> float:
    means = null.mean(axis=1)
    centre, sd = means.mean(), means.std()
    z = (real.mean() - centre) / sd
    z_k = (means - centre) / sd
    return float((1 + np.sum(z_k >= z)) / (1 + len(means)))


def test_family_of_one_p_adjusted_is_the_standardized_single_p() -> None:
    rng = np.random.default_rng(20261026)
    for _ in range(10):
        real, null = random_family(rng, 1, 199, shift=0.01)
        got = max_statistic(real, null)["v0"]
        assert got.p_adjusted == pytest.approx(single_p(real["v0"], null["v0"]))
        assert got.p_adjusted == pytest.approx(got.p_raw)


def test_p_adjusted_is_never_below_the_variants_own_standardized_p() -> None:
    rng = np.random.default_rng(1)
    real, null = random_family(rng, 8, 299, shift=0.01)

    got = max_statistic(real, null)

    for v, r in got.items():
        assert r.p_adjusted >= single_p(real[v], null[v]) - 1e-12
        assert r.p_adjusted >= r.p_raw - 1e-12
        assert 1 / 300 <= r.p_raw <= 1.0
        assert 1 / 300 <= r.p_adjusted <= 1.0
        assert r.trades == len(real[v])
        assert r.excess == pytest.approx(r.mean - r.null_mean)


def test_an_exact_duplicate_variant_changes_no_p_adjusted() -> None:
    rng = np.random.default_rng(2)
    real, null = random_family(rng, 5, 199, shift=0.01)
    before = max_statistic(real, null)

    after = max_statistic({**real, "copy": real["v2"]}, {**null, "copy": null["v2"]})

    for v in real:
        assert after[v].p_adjusted == before[v].p_adjusted
    assert after["copy"] == after["v2"]


def test_adding_an_independent_noise_variant_cannot_lower_any_p_adjusted() -> None:
    rng = np.random.default_rng(3)
    real, null = random_family(rng, 5, 199, shift=0.01)
    before = max_statistic(real, null)

    for trial in range(5):
        noise_real = rng.normal(0.0, 0.05, 6)
        noise_null = rng.normal(0.0, 0.05, (199, 6))
        after = max_statistic({**real, f"n{trial}": noise_real}, {**null, f"n{trial}": noise_null})
        for v in real:
            assert after[v].p_adjusted >= before[v].p_adjusted
            assert after[v].p_raw == before[v].p_raw  # a variant's own p is its own business


# =============================================================================================
# max_statistic: boundaries
# =============================================================================================


def test_unequal_permutation_counts_raise() -> None:
    with pytest.raises(ValueError):
        max_statistic(
            {"a": np.array([0.1, 0.2]), "b": np.array([0.1])},
            {"a": np.zeros((10, 2)) + np.arange(10)[:, None], "b": np.arange(11.0)[:, None]},
        )


def test_a_null_with_zero_spread_raises() -> None:
    with pytest.raises(ValueError):
        max_statistic({"a": np.array([0.1, 0.2])}, {"a": np.full((10, 2), 0.05)})


def test_an_empty_variant_raises() -> None:
    with pytest.raises(ValueError):
        max_statistic({"a": np.array([])}, {"a": np.zeros((10, 0))})


def test_mismatched_variant_names_or_columns_raise() -> None:
    null = np.arange(20.0).reshape(10, 2)
    with pytest.raises(ValueError):
        max_statistic({"a": np.array([0.1, 0.2])}, {"b": null})
    with pytest.raises(ValueError):
        max_statistic({"a": np.array([0.1, 0.2, 0.3])}, {"a": null})


# =============================================================================================
# calibration: pure-luck families pass about 5% of the time; a planted edge always passes
# =============================================================================================

N_SECURITIES = 30
N_SESSIONS = 150
VARIANTS = 20
BASE_LEGS = 60  # legs a world's variants draw from, so variants share some keys
LEGS_PER_VARIANT = 10
K = 200
WORLDS = 200
BPS = 5.0


def random_world(rng: np.random.Generator) -> Panel:
    """Log random walks, open and close alternating, 2% sd per half session."""
    sessions = weekdays(N_SESSIONS)
    sids = [f"S{i:02d}" for i in range(N_SECURITIES)]
    steps = rng.normal(0.0, 0.02, (N_SECURITIES, 2 * N_SESSIONS))
    paths = 50.0 * np.exp(np.cumsum(steps, axis=1))
    opens = {
        s: dict(zip(sessions, paths[i, 0::2].tolist(), strict=True)) for i, s in enumerate(sids)
    }
    closes = {
        s: dict(zip(sessions, paths[i, 1::2].tolist(), strict=True)) for i, s in enumerate(sids)
    }
    every = frozenset(sids)
    return Panel(
        sessions=sessions,
        open=opens,
        close=closes,
        splits={},
        pools={d: every for d in sessions},
    )


def random_legs(rng: np.random.Generator, panel: Panel, n: int, *, planted: bool) -> list[Span]:
    """``n`` distinct legs with random spans. The security is a uniform pick from the eligible
    set, or (planted) the one with the best return over the span."""
    sids = sorted(panel.open)
    legs: dict[str, Span] = {}
    while len(legs) < n:
        i = int(rng.integers(0, N_SESSIONS - 10))
        hold = int(rng.integers(0, 10))
        exit_at: ExitAt = "close" if hold == 0 else ("open" if rng.random() < 0.5 else "close")
        entry, exit_ = panel.sessions[i], panel.sessions[i + hold]
        leg = int(rng.integers(1, 3))
        probe = span("probe", entry, exit_, exit_at)
        if planted:
            sid = max(sids, key=lambda s: leg_return(panel, s, probe, BPS))
        else:
            sid = sids[int(rng.integers(0, len(sids)))]
        key = f"{sid}:{entry.isoformat()}:{leg}"
        ret = leg_return(panel, sid, probe, BPS)
        legs[key] = Span(key, entry, exit_, exit_at, ret)
    return list(legs.values())


def family_of(
    rng: np.random.Generator, panel: Panel, base: Sequence[Span], seed: int
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """VARIANTS variants, each LEGS_PER_VARIANT legs sampled from ``base``."""
    draws = null_returns(base, panel, permutations=K, seed=seed, slippage_bps=BPS)
    real: dict[str, np.ndarray] = {}
    null: dict[str, np.ndarray] = {}
    for v in range(VARIANTS):
        idx = sorted(rng.choice(len(base), LEGS_PER_VARIANT, replace=False).tolist())
        real[f"v{v}"] = np.array([base[j].ret for j in idx])
        null[f"v{v}"] = draws[:, idx]
    return real, null


def test_slicing_one_null_equals_drawing_each_variant_separately() -> None:
    # family_of draws the union once and slices; that is only fair if draws are per key.
    rng = np.random.default_rng(5)
    panel = random_world(rng)
    base = random_legs(rng, panel, 12, planted=False)

    union = null_returns(base, panel, permutations=K, seed=8, slippage_bps=BPS)
    part = null_returns(base[3:7], panel, permutations=K, seed=8, slippage_bps=BPS)

    assert np.array_equal(union[:, 3:7], part)


def test_real_legs_are_priced_like_the_null_prices_them() -> None:
    # The calibration below is only meaningful if real and null legs share one return formula.
    rng = np.random.default_rng(6)
    panel = random_world(rng)
    for s in random_legs(rng, panel, 10, planted=False):
        sid = s.key.split(":")[0]
        only = Panel(
            sessions=panel.sessions,
            open=panel.open,
            close=panel.close,
            splits={},
            pools={d: frozenset({sid}) for d in panel.sessions},
        )
        out = null_returns([s], only, permutations=1, seed=0, slippage_bps=BPS)
        assert out[0, 0] == pytest.approx(s.ret, rel=1e-12, abs=1e-15)


def test_twenty_pure_luck_variants_pass_at_5_percent_about_5_percent_of_the_time() -> None:
    rng = np.random.default_rng(20261026)
    passed = 0
    for world in range(WORLDS):
        panel = random_world(rng)
        base = random_legs(rng, panel, BASE_LEGS, planted=False)
        real, null = family_of(rng, panel, base, seed=1000 + world)
        results = max_statistic(real, null)
        if min(r.p_adjusted for r in results.values()) <= 0.05:
            passed += 1

    share = passed / WORLDS
    # Nominal 5%; binomial sd over 200 worlds is about 1.5 points.
    assert 0.01 <= share <= 0.10, f"family-wise false pass rate {share:.3f}"


def test_without_the_correction_twenty_lucky_variants_pass_far_more_often() -> None:
    # The fixture is meaningful: the uncorrected minimum p passes much more than 5% of worlds.
    rng = np.random.default_rng(20261027)
    passed = 0
    worlds = 60
    for world in range(worlds):
        panel = random_world(rng)
        base = random_legs(rng, panel, BASE_LEGS, planted=False)
        real, null = family_of(rng, panel, base, seed=5000 + world)
        if min(r.p_raw for r in max_statistic(real, null).values()) <= 0.05:
            passed += 1

    assert passed / worlds >= 0.25


def test_a_planted_edge_passes_the_corrected_gate() -> None:
    rng = np.random.default_rng(77)
    panel = random_world(rng)
    base = random_legs(rng, panel, BASE_LEGS, planted=False)
    real, null = family_of(rng, panel, base, seed=4)
    edge = random_legs(rng, panel, LEGS_PER_VARIANT, planted=True)
    real["edge"] = np.array([s.ret for s in edge])
    null["edge"] = null_returns(edge, panel, permutations=K, seed=4, slippage_bps=BPS)

    results = max_statistic(real, null)

    assert results["edge"].p_adjusted < 0.01
    assert results["edge"].excess > 0
    assert results["edge"].trades == LEGS_PER_VARIANT


# =============================================================================================
# DB layer: a seven-week store
# =============================================================================================


def _calendar(start: date, end: date, closed: set[date]) -> tuple[date, ...]:
    days, day = [], start
    while day <= end:
        if day.weekday() < 5 and day not in closed:
            days.append(day)
        day += timedelta(days=1)
    return tuple(days)


# Weekdays 2025-01-02 .. 2025-02-21 without 2025-01-09 (national day of mourning), 2025-01-20
# (MLK day) and 2025-02-17 (Presidents' Day), the real NYSE closures: 34 sessions, so the first
# pool (decided 02-04) already has 21 sessions behind it for the 20-session dollar volume.
CAL: tuple[date, ...] = _calendar(
    date(2025, 1, 2), date(2025, 2, 21), {date(2025, 1, 9), date(2025, 1, 20), date(2025, 2, 17)}
)
FEB = {d.day: d for d in CAL if d.month == 2}
WINDOW_END = FEB[13]  # the last exit session of the spans below
KNOWN_EARLY = datetime(2016, 1, 4, tzinfo=ET).astimezone(UTC)
EARLY_FILING = datetime(2024, 12, 15, 12, tzinfo=UTC)
SYMBOLS = ("A", "A2", "B", "C", "C.WS", "D", "E", "F", "G", "H")
GROWTH = {"A": 0.01, "A2": 0.01, "B": -0.01, "C": 0.02, "E": 0.005}  # others flat
BASE = {
    "A": 40.0,
    "A2": 40.0,
    "B": 60.0,
    "C": 20.0,
    "C.WS": 8.0,
    "D": 30.0,
    "E": 10.0,
    "F": 30.0,
    "G": 3.0,
    "H": 30.0,
}
VOLUME = 5_000_000  # $50M+ a day for every name above $10 ...
THIN_VOLUME = {"F": 100_000, "H": 30_000}  # ... except F ($3M) and H ($0.9M) ...
G_VOLUME = 50_000_000  # ... and G, $150M a day at $3 (under the $5 floor)
H_WAKES = FEB[6]  # H trades $600M a day from 02-06 on
B_SPLIT = FEB[12]  # B 2:1, ex-date 2025-02-12
A_RENAMED = FEB[10]  # S_A trades as A until 02-07 and as A2 from 02-10


def close_of(day: date) -> datetime:
    return datetime.combine(day, time(20), tzinfo=ET)


def at_open(day: date) -> datetime:
    return datetime.combine(day, time(9, 30), tzinfo=ET)


def midnight(day: date) -> datetime:
    return datetime.combine(day, time(0), tzinfo=ET).astimezone(UTC)


def px(symbol: str, day: date, *, future: bool = False) -> tuple[float, float]:
    """(open, close) of ``symbol`` on ``day``; raw, so B halves from its ex-date on."""
    i = CAL.index(day)
    close = BASE[symbol] * (1 + GROWTH.get(symbol, 0.0)) ** i
    if symbol == "B" and day >= B_SPLIT:
        close /= 2
    if future:
        close *= 10  # bars after the window: wild, so a leak shows
    return close * 0.995, close


def volume_of(symbol: str, day: date) -> int:
    if symbol == "G":
        return G_VOLUME
    if symbol == "H" and day >= H_WAKES:
        return 20_000_000
    return THIN_VOLUME.get(symbol, VOLUME)


def store_world(data_dir: Path, *, through: date, future_bars: bool = True) -> Path:
    root = market_root(data_dir)
    bars: list[Bar] = []
    for day in CAL:
        if day > through:
            break
        for symbol in SYMBOLS:
            if (symbol == "A" and day >= A_RENAMED) or (symbol == "A2" and day < A_RENAMED):
                continue
            o, c = px(symbol, day, future=future_bars and day > WINDOW_END)
            v = volume_of(symbol, day)
            bars.append(
                daily(symbol, day, open=o, high=max(o, c), low=min(o, c), close=c, volume=v, vwap=c)
            )
    store_daily(root, bars)
    write_cik_master(
        root,
        [
            CikSeg("S_A", "A", 101, valid_to=A_RENAMED, end_available_at=midnight(A_RENAMED)),
            CikSeg("S_A", "A2", 101, valid_from=A_RENAMED, available_at=midnight(A_RENAMED)),
            CikSeg("S_B", "B", 102),
            CikSeg("S_C", "C", 103),
            CikSeg("S_CW", "C.WS", 103),  # C's warrant: same CIK, never the scanner's pick
            CikSeg("S_D", "D", 104),
            CikSeg("S_E", "E", 105),
            CikSeg("S_F", "F", 106),
            CikSeg("S_G", "G", 107),
            CikSeg("S_H", "H", 108),
        ],
    )
    split_rows = [
        ("B", B_SPLIT, 1.0, 2.0, midnight(B_SPLIT)),
        ("B", B_SPLIT, 1.0, 2.0, midnight(B_SPLIT)),  # stored twice, one event
        ("C", FEB[19], 1.0, 3.0, midnight(FEB[19])),  # after the window
        ("C", FEB[11], 1.0, 5.0, close_of(FEB[20])),  # inside the window, on file only later
    ]
    write_split_rows(root, split_rows)
    return root


def write_split_rows(root: Path, rows: Sequence[tuple[str, date, float, float, datetime]]) -> Path:
    """``(symbol, ex_date, old_rate, new_rate, available_at)``: per-row availability, which
    tests.store.test_market.write_splits does not offer."""
    table = pa.table(
        {
            "symbol": pa.array([r[0] for r in rows], pa.string()),
            "ex_date": pa.array([r[1] for r in rows], pa.date32()),
            "old_rate": pa.array([r[2] for r in rows], pa.float64()),
            "new_rate": pa.array([r[3] for r in rows], pa.float64()),
            "factor": pa.array([r[3] / r[2] for r in rows], pa.float64()),
            "kind": pa.array(["forward" for _ in rows], pa.string()),
            "old_cusip": pa.array([None for _ in rows], pa.string()),
            "new_cusip": pa.array([None for _ in rows], pa.string()),
            "available_at": pa.array([r[4] for r in rows], SPLIT_SCHEMA.field("available_at").type),
        },
        schema=SPLIT_SCHEMA,
    )
    path = root / "corporate_actions" / "splits.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


def seed_filings(session: Session) -> None:
    """A, F, G and H filed long ago; C's 8-K lands exactly at 02-04 20:00 New York, B's one
    minute later; E's on Saturday 02-08; D files only 10-Qs."""
    source_id = pk(new_source(session))
    filings = {
        101: ("A", "8-K", EARLY_FILING),
        102: ("B", "8-K", close_of(FEB[4]) + timedelta(minutes=1)),
        103: ("C", "8-K", close_of(FEB[4])),
        104: ("D", "10-Q", EARLY_FILING),
        105: ("E", "8-K", datetime(2025, 2, 8, 10, tzinfo=ET)),
        106: ("F", "8-K", EARLY_FILING),
        107: ("G", "8-K", EARLY_FILING),
        108: ("H", "8-K", EARLY_FILING),
    }
    for cik, (ticker, doc_type, available) in filings.items():
        entity = new_entity(session, cik=cik, ticker=ticker, name=f"{ticker} Corp")
        new_document(
            session,
            source_id=source_id,
            entity_id=pk(entity),
            available_at=available,
            doc_type=doc_type,
        )


@pytest.fixture
def con() -> Iterator[duckdb.DuckDBPyConnection]:
    connection = duckdb.connect()
    yield connection
    connection.close()


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return store_world(tmp_path, through=CAL[-1])


@pytest.fixture
def filed(session: Session) -> Session:
    seed_filings(session)
    return session


@pytest.fixture
def cfg(tmp_path: Path) -> StrategyConfig:
    """The approved earnings_gap settings ($5 close, $20M over 20 sessions, 5 bps slippage),
    as mirrored by tests/strategy/files.py."""
    return load_strategy(write_strategy(tmp_path / "strategies"))


def book(
    session: Session,
    run_id: int,
    decision_id: int,
    security_id: str,
    entry: date,
    exit_at: datetime,
    *,
    leg: int = 1,
    entry_price: float = 50.0,
    shares: int = 10,
    pnl: float = 5.0,
) -> Trade:
    return add(
        session,
        Trade(
            decision_id=decision_id,
            run_id=run_id,
            strategy_id="earnings_gap",
            security_id=security_id,
            symbol=security_id.removeprefix("S_"),
            leg=leg,
            direction="long",
            entry_at=at_open(entry),
            entry_price=entry_price,
            shares=shares,
            key_level=entry_price * 0.98,
            hard_stop=entry_price * 0.95,
            risk_per_share=entry_price * 0.05,
            exit_at=exit_at,
            exit_price=entry_price + pnl / shares,
            exit_reason="break",
            pnl=pnl,
            r_multiple=1.0,
            holding_sessions=None,
            available_at=close_of(exit_at.astimezone(ET).date()),
        ),
    )


def trade_run(session: Session) -> tuple[int, int]:
    """A Run and one trade decision in it: (run_id, decision_id)."""
    run = new_run(session)
    entity = new_entity(
        session, cik=None, ticker=f"T{pk(run)}", kind="crypto", name="decision subject"
    )
    decision = new_trade_decision(session, run_id=pk(run), entity_id=pk(entity))
    return pk(run), pk(decision)


# =============================================================================================
# spans_of_run
# =============================================================================================


def test_spans_of_run_reads_sessions_exit_kind_and_return_in_entry_order(
    session: Session,
) -> None:
    run_id, decision_id = trade_run(session)
    other_run, other_decision = trade_run(session)
    # Inserted out of entry order; the 20:00 New York exit is the next UTC day (01:00 UTC).
    book(
        session,
        run_id,
        decision_id,
        "S_B",
        FEB[11],
        close_of(FEB[12]),
        leg=2,
        entry_price=40.0,
        shares=5,
        pnl=-4.0,
    )
    book(
        session,
        run_id,
        decision_id,
        "S_A",
        FEB[5],
        at_open(FEB[7]),
        pnl=25.0,
        entry_price=50.0,
        shares=10,
    )
    book(
        session,
        run_id,
        decision_id,
        "S_C",
        FEB[6],
        datetime.combine(FEB[10], time(16), tzinfo=ET),
        pnl=3.0,
        entry_price=30.0,
        shares=4,
    )
    book(session, other_run, other_decision, "S_D", FEB[5], at_open(FEB[6]))

    spans = spans_of_run(session, run_id)

    assert [s.key for s in spans] == [
        "S_A:2025-02-05:1",
        "S_C:2025-02-06:1",
        "S_B:2025-02-11:2",
    ]
    a, c, b = spans
    assert (a.entry_session, a.exit_session, a.exit_at) == (FEB[5], FEB[7], "open")
    assert (c.entry_session, c.exit_session, c.exit_at) == (FEB[6], FEB[10], "close")
    assert (b.entry_session, b.exit_session, b.exit_at) == (FEB[11], FEB[12], "close")
    assert a.ret == pytest.approx(25.0 / (10 * 50.0))
    assert c.ret == pytest.approx(3.0 / (4 * 30.0))
    assert b.ret == pytest.approx(-4.0 / (5 * 40.0))


def test_spans_with_the_same_entry_time_are_ordered_by_trade_id(session: Session) -> None:
    run_id, decision_id = trade_run(session)
    first = book(session, run_id, decision_id, "S_Z", FEB[5], at_open(FEB[6]))
    second = book(session, run_id, decision_id, "S_A", FEB[5], at_open(FEB[6]))
    assert pk(first) < pk(second)

    assert [s.key for s in spans_of_run(session, run_id)] == [
        "S_Z:2025-02-05:1",
        "S_A:2025-02-05:1",
    ]


def test_sessions_are_new_york_dates_across_the_dst_switch(session: Session) -> None:
    # 2025-03-07 20:00 EST = 03-08 01:00 UTC; 2025-03-10 20:00 EDT = 03-11 00:00 UTC.
    run_id, decision_id = trade_run(session)
    book(session, run_id, decision_id, "S_A", date(2025, 3, 7), close_of(date(2025, 3, 10)))

    [s] = spans_of_run(session, run_id)

    assert (s.entry_session, s.exit_session, s.exit_at) == (
        date(2025, 3, 7),
        date(2025, 3, 10),
        "close",
    )
    assert s.key == "S_A:2025-03-07:1"


def test_a_run_without_trades_has_no_spans_and_an_unknown_run_raises(session: Session) -> None:
    run_id, _ = trade_run(session)

    assert spans_of_run(session, run_id) == []
    with pytest.raises(LookupError):
        spans_of_run(session, run_id + 1000)


# =============================================================================================
# build_panel
# =============================================================================================

PANEL_SPANS = [
    span("S_A:2025-02-05:1", FEB[5], FEB[7], "close"),
    span("S_B:2025-02-06:1", FEB[6], FEB[13], "open"),  # crosses B's split
    span("S_C:2025-02-10:1", FEB[10], FEB[11], "close"),  # Monday: pool as of Friday 20:00
    span("S_E:2025-02-11:1", FEB[11], FEB[12], "close"),
]


def panel_at(
    session: Session, con: duckdb.DuckDBPyConnection, root: Path, cfg: StrategyConfig
) -> Panel:
    return build_panel(session, con, root, PANEL_SPANS, cfg=cfg, as_of=close_of(WINDOW_END))


def test_pools_are_the_scanners_reach_the_evening_before_entry(
    filed: Session, con: duckdb.DuckDBPyConnection, root: Path, cfg: StrategyConfig
) -> None:
    panel = panel_at(filed, con, root, cfg)

    # 02-05: C (filed exactly 02-04 20:00) is in, B (20:01) is not; D never files an 8-K;
    # F ($3M a day), G ($3 a share), H ($0.9M a day so far) and C's warrant never qualify.
    assert panel.pools[FEB[5]] == frozenset({"S_A", "S_C"})
    assert panel.pools[FEB[6]] == frozenset({"S_A", "S_B", "S_C"})
    # H's $600M day on 02-06 enters its average from 02-07 on. E filed on Saturday 02-08:
    # after Friday's close, so not in Monday's pool, in Tuesday's.
    assert panel.pools[FEB[10]] == frozenset({"S_A", "S_B", "S_C", "S_H"})
    assert panel.pools[FEB[11]] == frozenset({"S_A", "S_B", "S_C", "S_E", "S_H"})


def test_pools_follow_the_configured_thresholds(
    filed: Session, con: duckdb.DuckDBPyConnection, root: Path, cfg: StrategyConfig
) -> None:
    loose = apply_overrides(cfg, ["scan.min_price=1.0", "scan.min_avg_dollar_volume=500000"])

    panel = panel_at(filed, con, root, loose)

    # Under $1 / $0.5M, G ($3) and F ($3M) and H ($0.9M) are in; the warrant still is not.
    assert panel.pools[FEB[5]] == frozenset({"S_A", "S_C", "S_F", "S_G", "S_H"})


def test_prices_follow_a_security_across_its_rename_and_stay_raw(
    filed: Session, con: duckdb.DuckDBPyConnection, root: Path, cfg: StrategyConfig
) -> None:
    panel = panel_at(filed, con, root, cfg)

    assert panel.open["S_A"][FEB[5]] == pytest.approx(px("A", FEB[5])[0])
    assert panel.close["S_A"][FEB[11]] == pytest.approx(px("A2", FEB[11])[1])
    assert panel.close["S_B"][FEB[12]] == pytest.approx(px("B", FEB[12])[1])  # post-split raw
    assert list(panel.splits["S_B"]) == [(B_SPLIT, 2.0)]  # the duplicate row is one event
    assert list(panel.sessions) == sorted(panel.sessions)
    assert set(CAL[CAL.index(FEB[5]) : CAL.index(WINDOW_END) + 1]) <= set(panel.sessions)


def test_the_panel_prices_a_split_crossing_leg_like_the_formula(
    filed: Session, con: duckdb.DuckDBPyConnection, root: Path, cfg: StrategyConfig
) -> None:
    panel = panel_at(filed, con, root, cfg)
    only_b = Panel(
        panel.sessions, panel.open, panel.close, panel.splits, {FEB[6]: frozenset({"S_B"})}
    )

    out = null_returns([PANEL_SPANS[1]], only_b, permutations=1, seed=0, slippage_bps=0)

    raw_exit_open = px("B", FEB[13])[0]
    assert out[0, 0] == pytest.approx(raw_exit_open * 2.0 / px("B", FEB[6])[0] - 1)
    # Five sessions of -1% each; the split and the 0.995 open/close ratio cancel out.
    assert out[0, 0] == pytest.approx(0.99**5 - 1, rel=1e-9)


def test_unknown_universe_and_naive_as_of_raise(
    filed: Session, con: duckdb.DuckDBPyConnection, root: Path, cfg: StrategyConfig
) -> None:
    # StrategyConfig refuses other universes on load; model_copy skips validation.
    other = cfg.model_copy(update={"strategy": cfg.strategy.model_copy(update={"universe": "x"})})
    with pytest.raises(ValueError):
        build_panel(filed, con, root, PANEL_SPANS, cfg=other, as_of=close_of(WINDOW_END))
    with pytest.raises(ValueError):
        build_panel(filed, con, root, PANEL_SPANS, cfg=cfg, as_of=datetime(2025, 2, 13, 20))


def test_panel_sees_no_bar_or_split_after_as_of(
    filed: Session, con: duckdb.DuckDBPyConnection, root: Path, cfg: StrategyConfig
) -> None:
    # On disk: bars through 02-21 at ten times the price, C's 3:1 split on 02-19, and a C split
    # dated 02-11 that went on file only on 02-20 (C's prices never actually split).
    panel = panel_at(filed, con, root, cfg)

    assert max(panel.sessions) <= WINDOW_END
    for by_day in (*panel.open.values(), *panel.close.values()):
        assert all(d <= WINDOW_END for d in by_day)
    assert "S_C" in panel.open  # C is pooled, so its splits were looked up
    assert not list(panel.splits.get("S_C", ()))


def test_a_filing_after_the_evening_before_entry_is_not_in_that_pool(
    filed: Session, con: duckdb.DuckDBPyConnection, root: Path, cfg: StrategyConfig
) -> None:
    # B's 8-K (02-04 20:01) and E's (02-08) are in the database already when the panel is
    # built at 02-13 20:00; they must still not reach 02-05's or 02-10's pool.
    panel = panel_at(filed, con, root, cfg)

    assert "S_B" not in panel.pools[FEB[5]]
    assert "S_B" in panel.pools[FEB[6]]
    assert "S_E" not in panel.pools[FEB[10]]
    assert "S_E" in panel.pools[FEB[11]]


def test_volume_after_the_evening_before_entry_does_not_qualify_a_security(
    filed: Session, con: duckdb.DuckDBPyConnection, root: Path, cfg: StrategyConfig
) -> None:
    # At the panel's as_of (02-13) H has averaged far above $20M for days; on the evenings of
    # 02-04 and 02-05 its average was $0.9M and the $600M days were still ahead.
    panel = panel_at(filed, con, root, cfg)

    assert "S_H" not in panel.pools[FEB[5]]
    assert "S_H" not in panel.pools[FEB[6]]
    assert "S_H" in panel.pools[FEB[10]]


def test_panel_matches_a_store_that_ends_at_as_of(
    filed: Session,
    con: duckdb.DuckDBPyConnection,
    root: Path,
    tmp_path: Path,
    cfg: StrategyConfig,
) -> None:
    cut = store_world(tmp_path / "cut", through=WINDOW_END)

    full = panel_at(filed, con, root, cfg)
    short = panel_at(filed, con, cut, cfg)

    assert full.sessions == short.sessions
    assert full.pools == short.pools
    assert {s: dict(v) for s, v in full.open.items()} == {s: dict(v) for s, v in short.open.items()}
    assert {s: dict(v) for s, v in full.close.items()} == {
        s: dict(v) for s, v in short.close.items()
    }


# =============================================================================================
# null_test
# =============================================================================================


def two_runs(session: Session) -> tuple[int, int]:
    """Run 1: S_A 02-05 -> 02-07 close, S_B 02-06 -> 02-13 open. Run 2: the same S_A leg (one
    key, shared draws) and S_C 02-11 -> 02-12 hard stop."""
    r1, d1 = trade_run(session)
    r2, d2 = trade_run(session)
    book(session, r1, d1, "S_A", FEB[5], close_of(FEB[7]), pnl=8.0)
    book(session, r1, d1, "S_B", FEB[6], at_open(FEB[13]), pnl=-3.0)
    book(session, r2, d2, "S_A", FEB[5], close_of(FEB[7]), pnl=8.0)
    book(session, r2, d2, "S_C", FEB[11], datetime.combine(FEB[12], time(16), tzinfo=ET), pnl=12.0)
    return r1, r2


@pytest.mark.parametrize("bps", [None, 50.0])
def test_null_test_is_the_composition_of_its_parts_with_the_configs_slippage(
    filed: Session,
    con: duckdb.DuckDBPyConnection,
    root: Path,
    cfg: StrategyConfig,
    bps: float | None,
) -> None:
    if bps is not None:
        cfg = apply_overrides(cfg, [f"execute.fills.slippage_bps={bps}"])
    slippage = cfg.execute.fills.slippage_bps
    r1, r2 = two_runs(filed)

    got = null_test(filed, con, root, [r1, r2], cfg=cfg, permutations=60, seed=9)

    spans = {r: spans_of_run(filed, r) for r in (r1, r2)}
    panel = build_panel(
        filed, con, root, spans[r1] + spans[r2], cfg=cfg, as_of=close_of(WINDOW_END)
    )
    nulls = {
        str(r): null_returns(spans[r], panel, permutations=60, seed=9, slippage_bps=slippage)
        for r in (r1, r2)
    }
    reals = {str(r): np.array([s.ret for s in spans[r]]) for r in (r1, r2)}
    expected = max_statistic(reals, nulls)
    # The shared S_A leg draws the same securities in both runs.
    assert np.array_equal(nulls[str(r1)][:, 0], nulls[str(r2)][:, 0])

    assert set(got) == {r1, r2}
    for r in (r1, r2):
        want = expected[str(r)]
        assert got[r].trades == want.trades == 2
        for field in ("mean", "null_mean", "excess", "z", "p_raw", "p_adjusted"):
            assert getattr(got[r], field) == pytest.approx(getattr(want, field)), field


def test_more_slippage_in_the_config_lowers_the_null_mean(
    filed: Session, con: duckdb.DuckDBPyConnection, root: Path, cfg: StrategyConfig
) -> None:
    r1, r2 = two_runs(filed)
    costly = apply_overrides(cfg, ["execute.fills.slippage_bps=50.0"])

    cheap_result = null_test(filed, con, root, [r1, r2], cfg=cfg, permutations=60, seed=9)
    costly_result = null_test(filed, con, root, [r1, r2], cfg=costly, permutations=60, seed=9)

    for r in (r1, r2):
        assert costly_result[r].null_mean < cheap_result[r].null_mean
        assert costly_result[r].mean == cheap_result[r].mean  # real returns are as booked


def test_null_test_refuses_a_run_without_trades_an_unknown_run_and_universe(
    filed: Session, con: duckdb.DuckDBPyConnection, root: Path, cfg: StrategyConfig
) -> None:
    r1, _ = two_runs(filed)
    empty, _ = trade_run(filed)
    other = cfg.model_copy(update={"strategy": cfg.strategy.model_copy(update={"universe": "x"})})

    with pytest.raises(ValueError):
        null_test(filed, con, root, [r1, empty], cfg=cfg, permutations=20, seed=0)
    with pytest.raises(LookupError):
        null_test(filed, con, root, [r1, 9999], cfg=cfg, permutations=20, seed=0)
    with pytest.raises(ValueError):
        null_test(filed, con, root, [r1], cfg=other, permutations=20, seed=0)


def test_null_test_ignores_bars_and_filings_after_the_last_exit(
    filed: Session,
    con: duckdb.DuckDBPyConnection,
    root: Path,
    tmp_path: Path,
    cfg: StrategyConfig,
) -> None:
    r1, r2 = two_runs(filed)
    before = null_test(filed, con, root, [r1, r2], cfg=cfg, permutations=60, seed=9)
    # D files its first 8-K after the window; the cut store has no bar after 02-13.
    d_corp = filed.exec(select(Entity).where(Entity.cik == 104)).one()
    new_document(
        filed,
        source_id=pk(new_source(filed, name="sec_edgar_late")),
        entity_id=pk(d_corp),
        available_at=close_of(FEB[14]),
        doc_type="8-K",
    )
    cut = store_world(tmp_path / "cut", through=WINDOW_END)

    after_full = null_test(filed, con, root, [r1, r2], cfg=cfg, permutations=60, seed=9)
    after_cut = null_test(filed, con, cut, [r1, r2], cfg=cfg, permutations=60, seed=9)

    assert after_full == before
    assert after_cut == before
