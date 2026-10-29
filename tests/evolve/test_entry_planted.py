"""Guards against an entry test that cannot see an entry edge that is there, or sees one that is
not: in the planted B-Xtrender market (tests/evolve/planted_world.py) the stocks a signal picks
drift up for the next H = 10 sessions, so a real entry held a fixed 5 or 9 sessions must beat
random entries from the same pool, after the family correction over both holds and with a
cluster bootstrap interval above zero; in the control market built from the same random numbers
without the drift it must not.

API under test: asof.evolve.entry.entry_test (see tests/evolve/test_entry.py).

Interface points this file pins down where the spec left a choice:
- "Detects it" means p adjusted <= 0.01 (with 200 permutations the smallest p is 1/201) and the
  90% interval's low end above zero, for every hold; "does not" means p raw > 0.05 and an
  interval straddling zero.

World (made up; provenance and the self-consistency argument in planted_world.py): 30 stocks,
every one in every dev pool (guarded in test_planted_edge.py), drift MU = 0.5% a session over
the H = 10 sessions after each planting night, protocol PLANTED (dev 2023-03-01 .. 2024-03-28).
Holds 5 and 9 sit inside the planted run of an entry made the open after its signal night, so a
real entry holds up to 5 * 0.5% or 9 * 0.5% of drift; a random draw holds, on average, the
pool's drifting share of it (about half the stock-sessions drift).

Assumptions these fixtures cannot settle; check them on the real store:
- The planted world is easy on purpose (one strong signal, 5 bps, no gaps or corporate
  actions): detecting it says the machinery can see an entry edge, not that S1 has one.
- Entries cluster on few sessions in a real market sell-off or rally more than here; the
  interval's width on the real store (``asof gate entry --run 8 --hold 7``) is the check that
  clustering is not hiding a single session's luck.
"""

import time
from collections.abc import Iterator

import duckdb
import pytest
from sqlmodel import Session

from asof.evolve.entry import HoldResult, entry_test
from tests.evolve.planted_world import World, generate
from tests.evolve.test_planted_edge import MU, PLANTED, Store, _store

PERMUTATIONS = 200
HOLDS = (5, 9)


def _measure(store: Store) -> list[HoldResult]:
    con = duckdb.connect()
    try:
        with Session(store.engine) as session:
            return entry_test(
                session, con, store.root, list(store.run_ids), cfg=store.cfg, holds=HOLDS,
                dev=PLANTED.dev, permutations=PERMUTATIONS, seed=0,
            )  # fmt: skip
    finally:
        con.close()


@pytest.fixture(scope="module")
def planted(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Store]:
    world: World = generate(MU)
    yield from _store(tmp_path_factory.mktemp("entry_planted"), world, ((),))


@pytest.fixture(scope="module")
def control(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Store]:
    yield from _store(tmp_path_factory.mktemp("entry_control"), generate(0.0), ((),))


def test_the_planted_entry_edge_is_found_at_every_hold(planted: Store) -> None:
    began = time.perf_counter()
    results = _measure(planted)
    print(f"entry test, {len(HOLDS)} holds: {time.perf_counter() - began:.1f} s")

    assert [r.hold for r in results] == list(HOLDS)
    for r in results:
        assert r.result.trades > 80, r  # fixture: enough entries
        assert r.result.excess > 0, r
        assert r.result.p_adjusted <= 0.01, r
        assert r.ci[0] > 0, r


def test_the_control_market_shows_no_entry_edge(control: Store) -> None:
    for r in _measure(control):
        assert r.result.trades > 80, r  # fixture: enough entries
        assert r.result.p_raw > 0.05, r
        assert r.ci[0] < 0 < r.ci[1], r
