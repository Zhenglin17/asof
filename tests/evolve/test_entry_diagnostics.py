# Guards the entry diagnostics against counting what they claim to count: THT flips mixed
# across securities or broken by missing colours, green spells still running counted as short,
# re-entries matched to another security or to a close after (or on) the entry session, calendar
# days taken for sessions, a position's close read before it was on file, wrong tertile
# boundaries, and a group difference whose interval resamples single entries.
"""Target API (asof.evolve.entry_diagnostics, new module):

    FlipStats(flips_per_year: float, short_green_share: float, by_month: pd.DataFrame)
    flip_stats(colors, *, short=5) -> FlipStats
        colors: columns security_id, session_date, tht_color ("green" / "red" / missing).
        Per security in session order over the rows with a defined colour: a flip is a change
        from the previous defined colour (the first is none). flips_per_year = flips /
        (defined rows / 252). short_green_share = completed green spells (followed by a red)
        lasting < short defined sessions / completed green spells. by_month: month, flips
        (counted in the month of the session that shows the new colour), security_sessions
        (defined rows in that month).
    reentry_flags(entries, closes, calendar, *, within=10) -> list[bool]
        aligned with entries; True when the same security has a close x with
        0 < index(entry) - index(x) <= within on ``calendar``.
    closes_of_run(session, run_id, as_of) -> list[tuple[str, date]]
        per position (decision_id): the New York date of the latest leg exit visible at as_of.
    tertile_of(rank, n) -> int = floor(3 * (rank - 1) / n), 1 <= rank <= n, else ValueError
    group_difference(excess, flags, clusters, *, draws, seed, level=0.90) -> (diff, lo, hi)
        diff = mean(excess[flags]) - mean(excess[~flags]); (lo, hi) a percentile bootstrap
        resampling whole clusters, as asof.evolve.entry.cluster_ci does.

Interface points this file pins down where the spec left a choice:
- A missing colour may come as None or as NaN (tht_features writes a str column, pandas 3:
  measured 2026-10-29, ``df["c"] = ["green", None]`` reads back ``["green", nan]``); both are
  skipped alike.
- A spell's length counts defined sessions only: green, missing, green, red is a green spell
  of 2.
- by_month has one row per calendar month holding a defined colour, sorted by month; the month
  is compared as its first seven characters ("2024-01"), so a Period, a date or a string all do.
- closes_of_run's order is not pinned (compared sorted); an unknown run raises LookupError (as
  entries_of_run); a close's session is the New York date of exit_at (a 20:00 New York exit is
  00:00 UTC the next day).
- group_difference with one cluster gives (diff, diff, diff); mismatched lengths and an empty
  group raise ValueError.

Not pinned (left to the implementation; see the report): a position with a leg still open at
as_of (only its visible leg is on file), flip_stats with no completed green spell or with no
defined colour at all, a calendar missing an entry or close session, and how a bootstrap draw
without one of the two groups is handled.

Fake data: hand-made colours, sessions from tests/scan/bx_world.CAL (NYSE calendar, provenance
there), Decision / Trade rows written directly as asof.replay writes them (Trade.available_at =
20:00 New York of the exit session, documented in asof.store.models.Trade). No network.

No as_of in flip_stats, reentry_flags, tertile_of, group_difference: there is nothing to leak
through them; a leakage case is written for closes_of_run (a leg on file only later) and for
reentry_flags (a close after the entry must not count). What only real data can confirm:
- That tht_color in the real panel is None/NaN only before a series' first colour and after
  a blank spin-off window, never in the middle of a spell for another reason: print, for the
  liquid tier over 2023, the count of (security, session) with a missing colour between two
  defined ones; expect only spin-off blanks.
- That a real BX run's positions close in one or two legs that share one decision_id and that
  every closed leg's exit_at is 09:30 or 16:00 / 20:00 New York: query Trade for run 8 grouped
  by decision_id (legs per position 1 or 2) and by exit time of day.
- That real candidate ranks are 1..n without gaps on every night (tertile_of assumes it):
  check BxCandidate.rank of one real night's scan equals 1..len.
"""

import math
from collections.abc import Iterator
from datetime import UTC, date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pandas as pd
import pytest
from sqlmodel import Session

from asof.evolve.entry import Entry, cluster_ci, fixed_hold
from asof.evolve.entry_diagnostics import (
    FlipStats,
    closes_of_run,
    flip_stats,
    group_difference,
    pool_excess,
    rank_tertiles,
    reentries,
    reentry_flags,
    tertile_of,
)
from asof.evolve.null import Panel, Span, _leg_return, build_panel, null_returns
from asof.evolve.protocol import Window
from asof.scan.bx_trender import BxCandidate, scan_range
from asof.store.db import init_db, make_engine
from asof.store.market import market_root, visible_sessions
from asof.store.models import Decision, Trade
from asof.strategy import BxStrategyConfig, load_any_strategy
from tests.evolve.gate_world import new_run
from tests.evolve.test_entry import dev_entries, entry_spans
from tests.evolve.test_gate_bx import DEV_END, DEV_START, SECS, BxWorld, _world
from tests.replay.bx_replay_world import build_world, bx_strategy
from tests.scan.bx_world import CAL, close_of

ET = ZoneInfo("America/New_York")
G, R, N = "green", "red", None
JAN29 = CAL.index(date(2024, 1, 29))


def colours(by_security: dict[str, list[str | float | None]], start: int = JAN29) -> pd.DataFrame:
    rows = [
        {"security_id": sid, "session_date": CAL[start + i], "tht_color": c}
        for sid, seq in by_security.items()
        for i, c in enumerate(seq)
    ]
    return pd.DataFrame(rows, columns=["security_id", "session_date", "tht_color"])


def months(stats: FlipStats) -> list[tuple[str, int, int]]:
    m = stats.by_month
    return [
        (str(month)[:7], int(f), int(s))
        for month, f, s in zip(m["month"], m["flips"], m["security_sessions"], strict=True)
    ]


# =============================================================================================
# flip_stats
# =============================================================================================


def test_flip_stats_counts_flips_spells_and_months() -> None:
    # A over 2024-01-29 .. 02-05: G G R R R G -> flips on 01-31 and 02-05; one completed green
    # spell of 2 (short), the last one still running.
    # B over 2024-01-29 .. 02-08: - - R G G G G G R -> first colour 01-31 is no flip; flips on
    # 02-01 and 02-08; one completed green spell of 5 (not short: 5 < 5 is false).
    frame = colours({"A": [G, G, R, R, R, G], "B": [N, N, R, G, G, G, G, G, R]})
    assert [d.isoformat() for d in CAL[JAN29 : JAN29 + 9]] == [
        "2024-01-29", "2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02",
        "2024-02-05", "2024-02-06", "2024-02-07", "2024-02-08",
    ]  # fmt: skip

    stats = flip_stats(frame)

    assert stats.flips_per_year == pytest.approx(4 / (13 / 252), rel=1e-12)
    assert stats.short_green_share == pytest.approx(0.5, rel=1e-12)
    # Jan: A flips on 01-31; defined rows A 3 + B 1. Feb: A 02-05, B 02-01 and 02-08; A 3 + B 6.
    assert months(stats) == [("2024-01", 1, 4), ("2024-02", 3, 9)]


def test_flip_stats_takes_rows_in_any_order() -> None:
    frame = colours({"A": [G, G, R, R, R, G], "B": [N, N, R, G, G, G, G, G, R]})
    shuffled = frame.sample(frac=1.0, random_state=3).reset_index(drop=True)

    a, b = flip_stats(frame), flip_stats(shuffled)

    assert (a.flips_per_year, a.short_green_share) == (b.flips_per_year, b.short_green_share)
    assert months(a) == months(b)


def test_flips_never_cross_securities() -> None:
    # read as one series sorted by security, A's green run would end in B's red (a flip and a
    # completed green spell) and B's red would turn into C's green (another flip)
    frame = colours({"A": [G, G, G], "B": [R, R, R], "C": [G, R]})

    stats = flip_stats(frame)

    assert stats.flips_per_year == pytest.approx(1 / (8 / 252), rel=1e-12)
    assert stats.short_green_share == pytest.approx(1.0)  # C's spell of 1; A's never completed


@pytest.mark.parametrize("missing", [None, float("nan")])
def test_a_missing_colour_breaks_nothing_and_is_not_a_session(missing: float | None) -> None:
    # G G - R with short=3: one green spell of 2 defined sessions (short), one flip on the red;
    # counting the missing row would make the spell 3 long (not short)
    frame = colours({"A": [G, G, missing, R]})

    stats = flip_stats(frame, short=3)

    assert stats.flips_per_year == pytest.approx(1 / (3 / 252), rel=1e-12)
    assert stats.short_green_share == pytest.approx(1.0)
    assert sum(s for _, _, s in months(stats)) == 3


def test_the_short_bound_is_strict() -> None:
    # spells of 4 and 5 between reds: with short=5 only the first is short
    frame = colours({"A": [R, G, G, G, G, R, G, G, G, G, G, R]})

    assert flip_stats(frame, short=5).short_green_share == pytest.approx(0.5)
    assert flip_stats(frame, short=6).short_green_share == pytest.approx(1.0)
    assert flip_stats(frame, short=4).short_green_share == pytest.approx(0.0)


def test_a_first_colour_after_missing_rows_is_no_flip_and_red_spells_do_not_count() -> None:
    # - - R R G G G R R: flips on G and on the second R; the green spell of 3 is the only spell
    frame = colours({"A": [N, N, R, R, G, G, G, R, R]})

    stats = flip_stats(frame, short=5)

    assert stats.flips_per_year == pytest.approx(2 / (7 / 252), rel=1e-12)
    assert stats.short_green_share == pytest.approx(1.0)


# =============================================================================================
# reentry_flags
# =============================================================================================


def test_a_re_entry_is_a_close_of_the_same_security_up_to_within_sessions_before() -> None:
    entries = [Entry("A", CAL[120]), Entry("B", CAL[120]), Entry("A", CAL[200])]
    closes = [("A", CAL[115]), ("C", CAL[118])]

    assert reentry_flags(entries, closes, CAL) == [True, False, False]


def test_within_counts_sessions_inclusive() -> None:
    i = 120
    entry = [Entry("A", CAL[i])]
    assert reentry_flags(entry, [("A", CAL[i - 10])], CAL) == [True]
    assert reentry_flags(entry, [("A", CAL[i - 11])], CAL) == [False]
    assert reentry_flags(entry, [("A", CAL[i - 1])], CAL) == [True]
    assert reentry_flags(entry, [("A", CAL[i - 3])], CAL, within=3) == [True]
    assert reentry_flags(entry, [("A", CAL[i - 4])], CAL, within=3) == [False]


def test_within_is_sessions_not_calendar_days() -> None:
    # ten sessions back across the 2023 Christmas and New Year holidays is 15 calendar days
    i = CAL.index(date(2024, 1, 5))
    assert (CAL[i] - CAL[i - 10]).days > 14
    assert reentry_flags([Entry("A", CAL[i])], [("A", CAL[i - 10])], CAL) == [True]


def test_a_close_on_or_after_the_entry_session_does_not_count() -> None:
    # the position closing the day the new one opens, or later (a future close), is not what
    # the new entry could have followed
    i = 120
    entry = [Entry("A", CAL[i])]
    assert reentry_flags(entry, [("A", CAL[i])], CAL) == [False]
    assert reentry_flags(entry, [("A", CAL[i + 1]), ("A", CAL[i + 5])], CAL) == [False]


def test_any_qualifying_close_flags_and_another_securitys_close_never_does() -> None:
    i = 120
    entries = [Entry("A", CAL[i]), Entry("B", CAL[i])]
    closes = [("A", CAL[i - 30]), ("A", CAL[i - 2]), ("A", CAL[i + 1]), ("C", CAL[i - 1])]

    assert reentry_flags(entries, closes, CAL) == [True, False]
    assert reentry_flags([], closes, CAL) == []


# =============================================================================================
# closes_of_run
# =============================================================================================


def add_position(
    session: Session,
    run_id: int,
    sid: str,
    entry: date,
    exits: list[tuple[date, time]],
) -> int:
    """A long decision the evening before ``entry`` and one Trade per exit (leg 1, as the BX
    replay writes both halves), each on file from 20:00 New York of its exit session."""
    decided = close_of(CAL[CAL.index(entry) - 1])
    d = Decision(
        run_id=run_id, kind="trade", as_of=decided, strategy_id="bx_trender",
        security_id=sid, symbol=sid, direction="long",
    )  # fmt: skip
    session.add(d)
    session.flush()
    assert d.id is not None
    for day, at in exits:
        session.add(
            Trade(
                decision_id=d.id, run_id=run_id, strategy_id="bx_trender", security_id=sid,
                symbol=sid, leg=1, direction="long",
                entry_at=datetime.combine(entry, time(9, 30), tzinfo=ET), entry_price=50.0,
                shares=10, key_level=45.0, hard_stop=40.0, risk_per_share=10.0,
                exit_at=datetime.combine(day, at, tzinfo=ET), exit_price=51.0,
                exit_reason="test", pnl=5.0, r_multiple=0.1, available_at=close_of(day),
            )
        )  # fmt: skip
    session.commit()
    return d.id


OPEN, CLOSE = time(9, 30), time(20)


def test_closes_are_each_positions_last_leg_exit(session: Session) -> None:
    run = new_run(session, None, "bx_trender")
    other = new_run(session, None, "bx_trender")
    add_position(session, run, "A", CAL[100], [(CAL[104], OPEN), (CAL[109], OPEN)])
    add_position(session, run, "B", CAL[101], [(CAL[103], CLOSE)])  # sold at a close
    add_position(session, run, "A", CAL[120], [(CAL[125], OPEN)])  # A again: its own position
    add_position(session, run, "C", CAL[130], [])  # still open: no close
    add_position(session, other, "D", CAL[100], [(CAL[102], OPEN)])  # another run

    got = closes_of_run(session, run, close_of(CAL[-1]))

    assert sorted(got) == sorted([("A", CAL[109]), ("B", CAL[103]), ("A", CAL[125])])


def test_a_close_at_20_new_york_keeps_its_new_york_date(session: Session) -> None:
    run = new_run(session, None, "bx_trender")
    add_position(session, run, "B", CAL[101], [(CAL[103], CLOSE)])
    assert datetime.combine(CAL[103], CLOSE, tzinfo=ET).astimezone(UTC).date() != CAL[103]

    assert closes_of_run(session, run, close_of(CAL[-1])) == [("B", CAL[103])]


def test_a_leg_not_yet_on_file_is_not_read(session: Session) -> None:
    # the future exists in the database: legs exiting on CAL[109] and CAL[125] are on file only
    # from those evenings
    run = new_run(session, None, "bx_trender")
    add_position(session, run, "A", CAL[100], [(CAL[104], OPEN), (CAL[109], OPEN)])
    add_position(session, run, "E", CAL[120], [(CAL[125], OPEN)])
    as_of = close_of(CAL[106])

    got = closes_of_run(session, run, as_of)

    assert ("A", CAL[109]) not in got and ("E", CAL[125]) not in got
    assert all(day <= CAL[106] for _, day in got)
    # one second before CAL[109]'s close the second leg is still not there; at the close it is
    assert ("A", CAL[109]) in closes_of_run(session, run, close_of(CAL[109]))
    before = datetime.combine(CAL[109], time(19, 59, 59), tzinfo=ET)
    assert ("A", CAL[109]) not in closes_of_run(session, run, before)


def test_closes_of_an_unknown_run_raise_lookup_error(session: Session) -> None:
    with pytest.raises(LookupError):
        closes_of_run(session, 999_999, close_of(CAL[-1]))


# =============================================================================================
# tertile_of
# =============================================================================================


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (1, [0]),
        (2, [0, 1]),
        (3, [0, 1, 2]),
        (4, [0, 0, 1, 2]),
        (6, [0, 0, 1, 1, 2, 2]),
        (7, [0, 0, 0, 1, 1, 2, 2]),
        (10, [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]),
    ],
)
def test_tertile_is_floor_of_three_rank_minus_one_over_n(n: int, expected: list[int]) -> None:
    got = [tertile_of(rank, n) for rank in range(1, n + 1)]
    assert got == expected
    assert got == [(3 * (rank - 1)) // n for rank in range(1, n + 1)]


@pytest.mark.parametrize(("rank", "n"), [(0, 3), (4, 3), (-1, 3), (1, 0)])
def test_an_invalid_rank_raises(rank: int, n: int) -> None:
    with pytest.raises(ValueError):
        tertile_of(rank, n)


# =============================================================================================
# group_difference
# =============================================================================================


def test_the_difference_is_flagged_mean_minus_unflagged_mean() -> None:
    excess = np.array([0.10, 0.30, -0.05, 0.05, 0.00, 0.20])
    flags = [True, True, False, False, False, True]
    clusters = [1, 2, 1, 2, 3, 3]  # every cluster holds both groups

    diff, lo, hi = group_difference(excess, flags, clusters, draws=500, seed=0)

    assert diff == pytest.approx(0.20 - 0.0, abs=1e-12)
    assert lo <= hi


def test_one_cluster_gives_the_difference_on_both_sides() -> None:
    excess = np.array([0.4, 0.2, 0.1])
    diff, lo, hi = group_difference(excess, [True, False, False], ["d"] * 3, draws=300, seed=0)

    assert (diff, lo, hi) == pytest.approx((0.25, 0.25, 0.25), abs=1e-12)


def test_whole_clusters_are_resampled() -> None:
    # Two clusters of twenty: in c1 flagged 1 / unflagged 0, in c2 all 0. The difference is 0.5;
    # drawing c1 twice gives 1, c2 twice gives 0 (a quarter each), so the 90% interval is (0, 1).
    # Resampling single entries would be tight around 0.5.
    excess = np.array([1.0] * 10 + [0.0] * 10 + [0.0] * 20)
    flags = ([True] * 10 + [False] * 10) * 2
    clusters = ["c1"] * 20 + ["c2"] * 20

    diff, lo, hi = group_difference(excess, flags, clusters, draws=2000, seed=0)

    assert diff == pytest.approx(0.5)
    assert (lo, hi) == pytest.approx((0.0, 1.0))


def test_group_difference_is_deterministic_for_a_seed_and_nested_across_levels() -> None:
    rng = np.random.default_rng(7)
    excess = rng.normal(0.0, 0.05, 80)
    flags = [bool(i % 3 == 0) for i in range(80)]
    clusters = [i % 16 for i in range(80)]

    a = group_difference(excess, flags, clusters, draws=1000, seed=5)
    b = group_difference(excess, flags, clusters, draws=1000, seed=5)
    other = group_difference(excess, flags, clusters, draws=1000, seed=6)
    narrow = group_difference(excess, flags, clusters, draws=1000, seed=5, level=0.5)
    wide = group_difference(excess, flags, clusters, draws=1000, seed=5, level=0.99)

    assert a == b
    assert a[1:] != other[1:]
    assert a[0] == narrow[0] == wide[0]
    assert wide[1] <= a[1] <= narrow[1] <= narrow[2] <= a[2] <= wide[2]
    assert not math.isnan(a[1])


@pytest.mark.parametrize("flags", [[True] * 4, [False] * 4])
def test_an_empty_group_raises(flags: list[bool]) -> None:
    with pytest.raises(ValueError):
        group_difference(np.array([0.1, 0.2, 0.3, 0.4]), flags, [1, 1, 2, 2], draws=100, seed=0)


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError):
        group_difference(np.array([0.1, 0.2, 0.3]), [True, False], [1, 2, 3], draws=100, seed=0)
    with pytest.raises(ValueError):
        group_difference(np.array([0.1, 0.2]), [True, False], [1], draws=100, seed=0)


# =============================================================================================
# pool_excess, rank_tertiles, reentries: the orchestrators over a run's development dates
# =============================================================================================
#
# Fake data: tests/evolve/test_gate_bx.py's B-Xtrender world (provenance there): one BX replay
# over 2023-10-26 .. 2024-02-14, dev 2023-10-01 .. 2024-02-29. Measured on it (2026-10-29):
# the replay has 14 dev entries; scan_range over dev finds candidates on 82 nights, e.g.
# 2023-12-08 [AAA rank 1, BBB rank 2] while the replay entered nothing on 2023-12-11 (no room).
# The ``halt`` world is the same store without S_CCC's bar of 2024-01-09: CCC is a candidate
# on 2024-01-08 and has no open the next session. The ``future`` world (prices x10 from
# 2024-03-01, a 2:1 AAA split on file only 2024-04-15) is the leakage case for rank_tertiles.
#
# Assumptions only the real store can confirm:
# - The real pools are large (thousands), so the pool mean is a stable baseline; print the pool
#   size of one real dev evening beside rank_tertiles' output.
# - Real candidates without a next open are rare (halts): ``unopened`` on the real S1 dev
#   window should be a handful, not tens.

BX_DEV = Window(start=DEV_START, end=DEV_END)
HALT = date(2024, 1, 9)


@pytest.fixture(scope="module")
def bx_world(tmp_path_factory: pytest.TempPathFactory) -> Iterator[BxWorld]:
    yield from _world(tmp_path_factory.mktemp("diag_bx"), future=False)


@pytest.fixture(scope="module")
def bx_future_world(tmp_path_factory: pytest.TempPathFactory) -> Iterator[BxWorld]:
    yield from _world(tmp_path_factory.mktemp("diag_bx_future"), future=True)


@pytest.fixture(scope="module")
def halt_world(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Path, Session, Path]]:
    """The BX world's store without CCC's 2024-01-09 bar; no replay (rank_tertiles reads none)."""
    data_dir = tmp_path_factory.mktemp("diag_halt")
    root = market_root(data_dir)
    build_world(root, SECS, drop={"S_CCC": [HALT]})
    strategy_dir = bx_strategy(data_dir / "strategies")
    engine = make_engine(data_dir / "meta.db")
    init_db(engine)
    with Session(engine) as s:
        yield root, s, strategy_dir
    engine.dispose()


@pytest.fixture
def bx_session(bx_world: BxWorld) -> Iterator[Session]:
    with Session(bx_world.engine) as s:
        yield s


def bx_close() -> datetime:
    return datetime.combine(BX_DEV.end, time(20), tzinfo=ET)


def bps_of(cfg: BxStrategyConfig) -> float:
    return cfg.execute.fills.slippage_bps


def pool_mean(panel: Panel, span: Span, bps: float) -> tuple[float, float]:
    """Plain mean and population sd of _leg_return over the priced members of span's pool."""
    exit_prices = panel.open if span.exit_at == "open" else panel.close
    rets = [
        _leg_return(panel, m, span, bps / 10_000)
        for m in sorted(panel.pools[span.entry_session])
        if span.entry_session in panel.open.get(m, {})
        and span.exit_session in exit_prices.get(m, {})
    ]
    assert rets
    return float(np.mean(rets)), float(np.std(rets))


def test_pool_excess_is_the_real_return_minus_the_mean_of_the_pool(
    bx_session: Session, con: duckdb.DuckDBPyConnection, bx_world: BxWorld
) -> None:
    cfg, root, hold = bx_world.cfg, bx_world.root, 3
    entries = dev_entries(bx_session, con, root, bx_world.run_id, BX_DEV)
    calendar = visible_sessions(con, root, "1Day", bx_close())
    panel = build_panel(bx_session, con, root, entry_spans(entries), cfg=cfg, as_of=bx_close())
    fh = fixed_hold(entries, panel, calendar, hold=hold, last=BX_DEV.end,
                    slippage_bps=bps_of(cfg))  # fmt: skip
    assert len(fh.spans) >= 10

    kept, excess = pool_excess(entries, panel, calendar, hold=hold, last=BX_DEV.end,
                               slippage_bps=bps_of(cfg))  # fmt: skip

    assert [f"{e.security_id}:{e.entry_session}:fixed" for e in kept] == [s.key for s in fh.spans]
    real = np.array([s.ret for s in fh.spans])
    means = np.array([pool_mean(panel, s, bps_of(cfg))[0] for s in fh.spans])
    # exact: the plain mean over the priced pool (same arithmetic, so float-identical up to
    # summation order)
    np.testing.assert_allclose(excess, real - means, rtol=0, atol=1e-15)
    # and that is what the permutation null averages to: each column of null_returns is 4000
    # uniform draws from the same priced members, so its mean differs from the pool mean by
    # Monte Carlo error with standard error sd_pool / sqrt(4000). Five standard errors per
    # entry (two-sided p ~ 6e-7 each) is a bound no correct implementation should miss.
    null = null_returns(fh.spans, panel, permutations=4000, seed=0, slippage_bps=bps_of(cfg))
    sds = np.array([pool_mean(panel, s, bps_of(cfg))[1] for s in fh.spans])
    gap = np.abs(excess - (real - null.mean(axis=0)))
    assert (gap <= 5 * sds / np.sqrt(4000) + 1e-15).all(), (gap, sds)


def test_pool_excess_of_no_entry_is_empty(
    bx_session: Session, con: duckdb.DuckDBPyConnection, bx_world: BxWorld
) -> None:
    calendar = visible_sessions(con, bx_world.root, "1Day", bx_close())
    panel = build_panel(bx_session, con, bx_world.root, [], cfg=bx_world.cfg, as_of=bx_close())
    kept, excess = pool_excess([], panel, calendar, hold=3, last=BX_DEV.end, slippage_bps=5.0)
    assert kept == [] and len(excess) == 0


def assembled(
    session: Session, con: duckdb.DuckDBPyConnection, root: Path, cfg: BxStrategyConfig,
    hold: int,
) -> tuple[
    dict[date, list[BxCandidate]], list[Entry], dict[Entry, int], list[tuple[Entry, int, float]]
]:  # fmt: skip
    """rank_tertiles written out from scan_range, tertile_of, fixed_hold and the pool mean."""
    calendar = visible_sessions(con, root, "1Day", bx_close())
    found = scan_range(con, root, BX_DEV.start, BX_DEV.end, cfg.scan)
    entries: list[Entry] = []
    third: dict[Entry, int] = {}
    for night, cands in found.items():
        i = calendar.index(night)
        if i + 1 >= len(calendar):  # no next session visible at dev's last close
            continue
        for c in cands:
            e = Entry(c.security_id, calendar[i + 1])
            entries.append(e)
            third[e] = tertile_of(c.rank, len(cands))
    panel = build_panel(session, con, root, entry_spans(entries), cfg=cfg, as_of=bx_close())
    opened = [e for e in entries if e.entry_session in panel.open.get(e.security_id, {})]
    fh = fixed_hold(opened, panel, calendar, hold=hold, last=BX_DEV.end,
                    slippage_bps=bps_of(cfg))  # fmt: skip
    rows = []
    for s in fh.spans:
        e = Entry(s.key.split(":")[0], s.entry_session)
        rows.append((e, third[e], s.ret - pool_mean(panel, s, bps_of(cfg))[0]))
    return found, entries, third, rows


def test_rank_tertiles_equals_its_parts_and_counts_candidates_never_bought(
    bx_session: Session, con: duckdb.DuckDBPyConnection, bx_world: BxWorld
) -> None:
    cfg, root, hold = bx_world.cfg, bx_world.root, 3
    found, entries, _, rows = assembled(bx_session, con, root, cfg, hold)

    got = rank_tertiles(bx_session, con, root, cfg=cfg, dev=BX_DEV, hold=hold, bootstrap=500,
                        seed=0)  # fmt: skip

    assert got.nights == len(found)
    assert got.unopened == 0
    assert got.dropped == len(entries) - len(rows)
    calendar = visible_sessions(con, root, "1Day", bx_close())
    index = {d: i for i, d in enumerate(calendar)}
    for g in (0, 1, 2):
        part = [(e, x) for e, t, x in rows if t == g]
        assert got.tertiles[g].entries == len(part), g
        assert got.tertiles[g].excess == pytest.approx(np.mean([x for _, x in part]), abs=1e-15)
        assert got.tertiles[g].ci == pytest.approx(cluster_ci(
            np.array([x for _, x in part]), [index[e.entry_session] // hold for e, _ in part],
            draws=500, seed=0), abs=1e-15)  # fmt: skip
    ends = [(e, t, x) for e, t, x in rows if t != 1]
    assert got.top_minus_bottom == pytest.approx(group_difference(
        np.array([x for _, _, x in ends]), [t == 0 for _, t, _ in ends],
        [index[e.entry_session] // hold for e, _, _ in ends], draws=500, seed=0,
    ), abs=1e-15)  # fmt: skip

    # 2023-12-08's candidates (AAA rank 1, BBB rank 2) were never bought: the replay had no room
    bought = set(dev_entries(bx_session, con, root, bx_world.run_id, BX_DEV))
    unbought = {e for e, _, _ in rows} - bought
    assert Entry("S_BBB", date(2023, 12, 11)) in unbought
    assert Entry("S_AAA", date(2023, 12, 11)) in unbought
    assert len(rows) > len(bought)


def test_a_candidate_without_a_next_open_is_left_out_and_counted_unopened(
    con: duckdb.DuckDBPyConnection, halt_world: tuple[Path, Session, Path]
) -> None:
    root, session, strategy_dir = halt_world
    cfg = load_any_strategy(strategy_dir)
    assert isinstance(cfg, BxStrategyConfig)
    hold = 3
    found, entries, _, rows = assembled(session, con, root, cfg, hold)
    night = CAL[CAL.index(HALT) - 1]
    assert "S_CCC" in [c.security_id for c in found[night]]  # the fixture still signals there

    got = rank_tertiles(session, con, root, cfg=cfg, dev=BX_DEV, hold=hold, bootstrap=200, seed=0)

    assert got.unopened == 1
    assert Entry("S_CCC", HALT) in entries
    assert Entry("S_CCC", HALT) not in {e for e, _, _ in rows}
    assert sum(t.entries for t in got.tertiles) == len(rows)
    assert got.dropped == len(entries) - 1 - len(rows)
    for g in (0, 1, 2):
        part = [x for _, t, x in rows if t == g]
        assert got.tertiles[g].entries == len(part)
        assert got.tertiles[g].excess == pytest.approx(np.mean(part), abs=1e-15)


def test_rank_tertiles_ignores_prices_and_records_after_dev(
    bx_session: Session, con: duckdb.DuckDBPyConnection, bx_world: BxWorld,
    bx_future_world: BxWorld,
) -> None:  # fmt: skip
    # the future exists in the store: prices x10 from 2024-03-01 and a split on file 2024-04-15
    a = rank_tertiles(bx_session, con, bx_world.root, cfg=bx_world.cfg, dev=BX_DEV, hold=3,
                      bootstrap=300, seed=1)  # fmt: skip
    b = rank_tertiles(bx_session, con, bx_future_world.root, cfg=bx_world.cfg, dev=BX_DEV, hold=3,
                      bootstrap=300, seed=1)  # fmt: skip
    assert repr(a) == repr(b)


@pytest.mark.parametrize("within", [0])
def test_reentries_without_any_re_entry_give_nan(
    bx_session: Session, con: duckdb.DuckDBPyConnection, bx_world: BxWorld, within: int
) -> None:
    # within=0: no close can be 1..0 sessions before an entry, so no entry is a re-entry
    got = reentries(bx_session, con, bx_world.root, bx_world.run_id, cfg=bx_world.cfg,
                    dev=BX_DEV, hold=3, within=within, bootstrap=200, seed=0)  # fmt: skip

    assert got.reentries == 0
    assert got.entries > 0
    assert all(math.isnan(x) for x in got.difference)
    # with a wide window the same run does have re-entries and a real difference
    wide = reentries(bx_session, con, bx_world.root, bx_world.run_id, cfg=bx_world.cfg,
                     dev=BX_DEV, hold=3, within=10, bootstrap=200, seed=0)  # fmt: skip
    assert 0 < wide.reentries < wide.entries == got.entries
    assert not math.isnan(wide.difference[0])
