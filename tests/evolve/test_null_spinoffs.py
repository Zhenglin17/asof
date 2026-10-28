"""Guards against a random-entry null that prices a spin-off differently from the replay: a drawn
security's ex-date drop counted as a loss while the replay credits the child (the null would look
worse than it is and flatter every real leg), a child paid on a span that entered on or after the
ex-date, a child price or decision read from after the panel's as_of, and restart events (raw,
unadjustable prices) that silently pass as ordinary spans.

API under test (asof.evolve.null), as agreed for the spin-off bookkeeping change; the names
below are this file's choice where the design left one:

    SpinoffAdjust(ex_date, known_on, children: tuple[(ratio, sale price), ...])
        sale price: the child's open on its first session after known_on (before slippage),
        the same price the replay sells at.
    Panel(..., spinoffs: Mapping[security_id, Sequence[SpinoffAdjust]] = {},
               restarts: Mapping[security_id, Sequence[ex_date]] = {})
        both default to empty, so Panel(sessions, open, close, splits, pools) still builds.
    _leg_return / null_returns: a drawn span with entry_session < ex_date <= exit_session of an
        adjust event adds sum(ratio * sale * (1 - bps)) per share to the exit value; restart
        events leave the raw prices.
    restart_crossings(spans, panel, *, permutations, seed) -> int
        how many (permutation, span) draws of null_returns with the same seed landed on a
        security with a restart ex_date in (entry_session, exit_session].
    build_panel(...): spinoffs / restarts of pooled securities, from spinoff_events at as_of;
        only events with known_on <= as_of.

Fake data provenance:
- Pure layer: hand-picked prices, as in tests/evolve/test_null.py.
- Store layer: tests/evolve/test_null.py's seven-week world, plus spin-off records written by
  tests/store/test_spinoffs.spin_row (available_at 00:00 New York of the ex-date, ingest rule)
  and flat child bars visible at 20:00 New York of their session (ingest rule). E -> KE (1 per
  share), A2 -> KA (0.5 per share, KA first trading two sessions after the window) and H -> H
  (parent and child on one symbol, the real CR 2023-04-04 shape: restart) are constructed.

Assumption to check on real data: a real replay's child_proceeds / shares for a spin-off leg
equals the panel's per-share child value for that security and event (WDC S008254 2025-02-24:
(1/3) x SNDK 2025-02-25 open 48.30 = 16.10). Command: build_panel over the run's spans at its
last exit and compare with the Trade row; tests/replay/test_replay_spinoffs.py pins the same on
fake data.
"""

import math
from dataclasses import replace
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sqlmodel import Session

from asof.evolve.null import Panel, SpinoffAdjust, build_panel, null_returns, restart_crossings
from asof.ingest.security_master import MASTER_FILE
from asof.strategy import StrategyConfig
from tests.evolve.test_null import (
    CAL,
    DAYS,
    FEB,
    PANEL_SPANS,
    WINDOW_END,
    close_of,
    flat,
    midnight,
    panel_of,
    seed_filings,
    span,
    store_world,
)
from tests.evolve.test_null import (
    cfg as cfg,  # noqa: F401  (fixture)
)
from tests.store.test_filer_securities import CikSeg, write_cik_master
from tests.store.test_spinoffs import px as child_px
from tests.store.test_spinoffs import spin_row, write_bars, write_other_rows


def with_events(
    panel: Panel,
    spinoffs: dict[str, list[SpinoffAdjust]] | None = None,
    restarts: dict[str, list[date]] | None = None,
) -> Panel:
    return replace(panel, spinoffs=spinoffs or {}, restarts=restarts or {})


# =============================================================================================
# pure layer: normal path
# =============================================================================================


def test_an_adjust_inside_the_hold_adds_the_child_per_share() -> None:
    # 100 in, the parent drops to 78 after handing out 0.25 child worth 88 (22 per share)
    prices = {"X": {DAYS[0]: (100.0, 100.0), DAYS[2]: (79.0, 78.5), DAYS[4]: (78.2, 78.0)}}
    event = SpinoffAdjust(ex_date=DAYS[2], known_on=DAYS[2], children=((0.25, 88.0),))
    panel = with_events(panel_of(prices), spinoffs={"X": [event]})

    out = null_returns(
        [span("k", DAYS[0], DAYS[4], "close")], panel, permutations=3, seed=0, slippage_bps=0
    )
    assert out[:, 0] == pytest.approx([0.0, 0.0, 0.0])  # (78 + 22) / 100 - 1

    slipped = null_returns(
        [span("k", DAYS[0], DAYS[4], "close")], panel, permutations=1, seed=0, slippage_bps=10
    )
    assert slipped[0, 0] == pytest.approx((78.0 * 0.999 + 22.0 * 0.999) / (100.0 * 1.001) - 1)


def test_several_children_of_one_event_add_up() -> None:
    prices = {"X": {DAYS[0]: (100.0, 100.0), DAYS[3]: (80.0, 80.0)}}
    event = SpinoffAdjust(ex_date=DAYS[1], known_on=DAYS[1], children=((0.25, 32.0), (0.5, 8.0)))
    panel = with_events(panel_of(prices), spinoffs={"X": [event]})
    out = null_returns(
        [span("k", DAYS[0], DAYS[3], "close")], panel, permutations=1, seed=0, slippage_bps=0
    )
    assert out[0, 0] == pytest.approx((80.0 + 12.0) / 100.0 - 1)


def test_a_panel_without_events_still_builds_positionally() -> None:
    panel = Panel(DAYS, {}, {}, {}, {})
    assert dict(panel.spinoffs) == {}
    assert dict(panel.restarts) == {}


# =============================================================================================
# pure layer: boundaries
# =============================================================================================


def test_an_ex_date_on_the_exit_session_pays_and_on_the_entry_session_does_not() -> None:
    prices = {
        "X": {DAYS[0]: (100.0, 100.0), DAYS[2]: (80.0, 80.0), DAYS[4]: (84.0, 84.0)},
    }
    event = SpinoffAdjust(ex_date=DAYS[2], known_on=DAYS[2], children=((0.5, 40.0),))
    panel = with_events(panel_of(prices), spinoffs={"X": [event]})

    # Exit at the ex open (a pending exit): the parent shares are sold but the child is owed.
    on_exit = null_returns(
        [span("a", DAYS[0], DAYS[2], "open")], panel, permutations=1, seed=0, slippage_bps=0
    )
    # Entry at the ex open: bought without the entitlement.
    on_entry = null_returns(
        [span("b", DAYS[2], DAYS[4], "close")], panel, permutations=1, seed=0, slippage_bps=0
    )
    # Out before the ex-date: nothing.
    before = null_returns(
        [span("c", DAYS[0], DAYS[1], "close")],
        with_events(
            panel_of({"X": {DAYS[0]: (100.0, 100.0), DAYS[1]: (101.0, 101.0)}}),
            spinoffs={"X": [event]},
        ),
        permutations=1,
        seed=0,
        slippage_bps=0,
    )
    assert on_exit[0, 0] == pytest.approx((80.0 + 20.0) / 100.0 - 1)
    assert on_entry[0, 0] == pytest.approx(84.0 / 80.0 - 1)
    assert before[0, 0] == pytest.approx(0.01)


def test_a_restart_inside_the_hold_keeps_raw_prices_and_is_counted() -> None:
    entry, exit_ = DAYS[0], DAYS[4]
    prices = {
        "A": {entry: (100.0, 100.0), exit_: (101.0, 101.0)},  # +1%
        "R": {entry: (100.0, 100.0), exit_: (70.0, 70.0)},  # -30%, a restart on DAYS[2]
    }
    panel = with_events(panel_of(prices), restarts={"R": [DAYS[2]]})
    spans = [span("s", entry, exit_, "close")]

    out = null_returns(spans, panel, permutations=200, seed=4, slippage_bps=0)
    count = restart_crossings(spans, panel, permutations=200, seed=4)

    assert {round(v, 9) for v in out[:, 0]} == {0.01, -0.3}
    assert count == int(np.sum(np.isclose(out[:, 0], -0.3)))
    assert 0 < count < 200


def test_restarts_outside_a_span_are_not_counted() -> None:
    prices = {"R": {d: (100.0, 100.0) for d in DAYS}}
    panel = with_events(panel_of(prices), restarts={"R": [DAYS[0], DAYS[6]]})
    # DAYS[0] is the entry session (bought after it), DAYS[6] is after the exit
    spans = [span("s", DAYS[0], DAYS[5], "close")]
    assert restart_crossings(spans, panel, permutations=10, seed=0) == 0
    assert (
        restart_crossings([span("t", DAYS[0], DAYS[6], "close")], panel, permutations=10, seed=0)
        == 10
    )


def test_an_event_of_one_security_does_not_reach_another() -> None:
    prices = {"A": flat(100.0, 100.0), "B": flat(100.0, 100.0)}
    event = SpinoffAdjust(ex_date=DAYS[2], known_on=DAYS[2], children=((1.0, 50.0),))
    panel = with_events(panel_of(prices), spinoffs={"B": [event]})
    out = null_returns(
        [span("k", DAYS[0], DAYS[4], "close")], panel, permutations=300, seed=1, slippage_bps=0
    )
    # A: 0.0; B: (100 + 50) / 100 - 1 = 0.5 (its raw price is flat here, which is not the
    # point): each value names the security drawn and only B carries the child
    assert {round(v, 9) for v in out[:, 0]} == {0.0, 0.5}


# =============================================================================================
# store layer: E -> KE adjust, A2 -> KA known only after the window, H -> H restart
# =============================================================================================


def add_master(root: Path, segs: list[CikSeg]) -> None:
    """Append segments to the world's master (store_world writes the whole file once)."""
    path = root / MASTER_FILE
    old = pq.read_table(path)
    extra = pq.read_table(write_cik_master(root.parent / "extra_master", segs))
    pq.write_table(pa.concat_tables([old, extra.cast(old.schema)]), path)


KE_FIRST, KE_SALE = 1.0, 1.2  # KE's first close (prices the factor) and next open (the sale)
KA_FIRST_DAY = FEB[14]  # two sessions after WINDOW_END
KA_SALE_DAY = FEB[18]  # 02-17 is Presidents' Day


def spun(data_dir: Path, *, through: date) -> Path:
    root = store_world(data_dir, through=through)
    add_master(
        root,
        [
            CikSeg("S_KE", "KE", None, valid_from=FEB[12], available_at=midnight(FEB[12])),
            CikSeg(
                "S_KA", "KA", None, valid_from=KA_FIRST_DAY, available_at=midnight(KA_FIRST_DAY)
            ),
        ],
    )
    kids = [child_px("KE", FEB[12], KE_FIRST)]
    kids += [child_px("KE", d, KE_SALE) for d in CAL if FEB[12] < d <= through]
    kids += [child_px("KA", d, 2.0 if d == KA_FIRST_DAY else 2.2)
             for d in CAL if KA_FIRST_DAY <= d <= through]  # fmt: skip
    write_bars(root, kids)
    write_other_rows(
        root,
        [
            spin_row("E", "KE", FEB[12], 1.0),
            spin_row("A2", "KA", FEB[12], 0.5),
            spin_row("H", "H", FEB[11], 0.5),
        ],
    )
    return root


@pytest.fixture
def filed(session: Session) -> Session:
    seed_filings(session)
    return session


def test_the_panel_carries_adjust_and_restart_events_of_pooled_securities(
    tmp_path: Path, filed: Session, con: duckdb.DuckDBPyConnection, cfg: StrategyConfig
) -> None:
    root = spun(tmp_path, through=CAL[-1])
    panel = build_panel(filed, con, root, PANEL_SPANS, cfg=cfg, as_of=close_of(WINDOW_END))

    assert list(panel.spinoffs["S_E"]) == [
        SpinoffAdjust(ex_date=FEB[12], known_on=FEB[12], children=((1.0, KE_SALE),))
    ]
    assert list(panel.restarts["S_H"]) == [FEB[11]]


def test_the_null_adds_the_child_like_the_formula(
    tmp_path: Path, filed: Session, con: duckdb.DuckDBPyConnection, cfg: StrategyConfig
) -> None:
    root = spun(tmp_path, through=CAL[-1])
    panel = build_panel(filed, con, root, PANEL_SPANS, cfg=cfg, as_of=close_of(WINDOW_END))
    only_e = replace(panel, pools={FEB[11]: frozenset({"S_E"})})

    out = null_returns([PANEL_SPANS[3]], only_e, permutations=1, seed=0, slippage_bps=0)

    entry_open = panel.open["S_E"][FEB[11]]
    exit_close = panel.close["S_E"][FEB[12]]
    assert out[0, 0] == pytest.approx((exit_close + 1.0 * KE_SALE) / entry_open - 1)
    assert not math.isclose(out[0, 0], exit_close / entry_open - 1)


# =============================================================================================
# store layer: leakage
# =============================================================================================


def test_an_event_known_after_as_of_is_not_in_the_panel(
    tmp_path: Path, filed: Session, con: duckdb.DuckDBPyConnection, cfg: StrategyConfig
) -> None:
    # A2 -> KA: KA's first bar (02-14) is on disk but after the window's last close, so at
    # 02-13 20:00 the event is undecided. Read later, the same store has it.
    root = spun(tmp_path, through=CAL[-1])
    at_end = build_panel(filed, con, root, PANEL_SPANS, cfg=cfg, as_of=close_of(WINDOW_END))
    assert not list(at_end.spinoffs.get("S_A", ()))
    # undecided at as_of: a draw holding it is on raw prices, so it counts as a restart
    assert list(at_end.restarts.get("S_A", ())) == [FEB[12]]

    later = build_panel(filed, con, root, PANEL_SPANS, cfg=cfg, as_of=close_of(KA_SALE_DAY))
    [event] = later.spinoffs["S_A"]
    assert (event.ex_date, event.known_on) == (FEB[12], KA_FIRST_DAY)
    assert event.children == ((0.5, 2.2),)


def test_panel_events_match_a_store_that_ends_at_as_of(
    tmp_path: Path, filed: Session, con: duckdb.DuckDBPyConnection, cfg: StrategyConfig
) -> None:
    full = spun(tmp_path / "full", through=CAL[-1])
    cut = spun(tmp_path / "cut", through=WINDOW_END)
    a = build_panel(filed, con, full, PANEL_SPANS, cfg=cfg, as_of=close_of(WINDOW_END))
    b = build_panel(filed, con, cut, PANEL_SPANS, cfg=cfg, as_of=close_of(WINDOW_END))
    assert {s: list(v) for s, v in a.spinoffs.items()} == {
        s: list(v) for s, v in b.spinoffs.items()
    }
    assert {s: list(v) for s, v in a.restarts.items()} == {
        s: list(v) for s, v in b.restarts.items()
    }
    assert "S_E" in a.spinoffs  # not vacuous
