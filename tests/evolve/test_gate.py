"""Guards against a gate that passes a strategy for the wrong reason: holdout trades, rows or bars
reaching a development verdict, a leg that exits after the development window counted, a
walk-forward pick made with the test window's own results, a candidate chosen on rows outside
dev, a thin edge that only survives because its best trades or the extra cost were left out,
and a failing criterion missing from the reasons.

API under test (asof.evolve.gate, new module):

    walk_forward_windows(dev: Window, train_months, test_months)
        -> list[((train_start, train_end), (test_start, test_end))]
        tests start at dev.start + train_months and step test_months (calendar months); each
        test window ends the day before start + test_months; kept while it ends <= dev.end;
        train = [test_start - train_months, test_start - 1 day].
    Segment(train, test, picked: int | None, legs: int, mean: float | None,
            null_mean: float | None, positive: bool)
    GateResult(dev, candidate, variants: dict[int, VariantResult], segments: list[Segment],
               walk_forward_positive, versus: Versus, mde, thin_excess, thin_edge_ok,
               versions_on_dev, passed, reasons: list[str])
    check(session, con, root, run_ids, *, cfg, protocol, permutations, seed) -> GateResult

Interface points this file pins down where the spec left a choice:
- ``reasons`` names: "null" (candidate p_adjusted >= alpha), "walk_forward" (fewer positive
  segments than min_positive), "thin_edge", and the three Versus field names
  "drawdown_not_worse_than_basket", "sharpe_at_least_best", "beats_spy_and_basket_total".
  Compared as sets: the order is not pinned.
- ``result.dev`` has ``start`` and ``end`` (the protocol's Window).
- An unknown run raises LookupError (as null_test does); no dev legs or no dev equity rows
  raise ValueError, the latter mentioning "replay".
- A segment that picks nothing, or whose pick has no leg entering in the test window, has
  legs 0, mean None and null_mean None.
- ``versus`` equals compare_run(..., window=(dev.start, dev.end)) of the candidate.
- MDE uses the one-sided z(1 - alpha).

World: tests/evolve/gate_world.py (provenance there). Expected values are rebuilt here from the
null module's parts (spans_of_run, build_panel, null_returns, max_statistic), whose own tests
are in test_null.py.

Assumptions these fixtures cannot settle; check them on the real store:
- A replay's equity rows cover every session of its window, so the dev slice lines up with the
  benchmarks' sessions. Check: for each run passed to `asof gate check`, count equity_daily
  rows inside dev against visible_sessions over the same dates; any difference must be zero.
- The MDE reference: on the real earnings_gap run (sd of dev leg returns about 14.3%, n 170)
  the report shows an MDE of about 2.7%.
"""

import dataclasses
import math
import statistics
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pytest
from sqlmodel import Session

from asof.evolve.gate import GateResult, Segment, check, walk_forward_windows
from asof.evolve.metrics import curve_metrics
from asof.evolve.null import Span, build_panel, max_statistic, null_returns, spans_of_run
from asof.evolve.protocol import EvaluationProtocol, Window
from asof.evolve.versus import compare_run
from asof.strategy import StrategyConfig, load_strategy
from tests.evolve.gate_world import (
    A_LATE,
    A_LEGS,
    B_LEGS,
    DEV_DAYS,
    DEV_END,
    DEV_START,
    GATE,
    HOLDOUT_DAYS,
    PERMUTATIONS,
    PRE_DEV_DAYS,
    Leg,
    Runs,
    a_returns,
    add_equity,
    add_holdout,
    alternating,
    at_open,
    book,
    close_of,
    d,
    extra_run,
    month,
    new_run,
    seed,
    store,
    version,
)

FEB, MAR, APR = month(2), month(3), month(4)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return store(tmp_path / "full")


@pytest.fixture
def runs(session: Session, tmp_path: Path) -> Runs:
    return seed(session, tmp_path)


@pytest.fixture
def cfg(runs: Runs) -> StrategyConfig:
    return load_strategy(runs.strategy_dir)


def gate(
    session: Session,
    con: duckdb.DuckDBPyConnection,
    root: Path,
    run_ids: Sequence[int],
    cfg: StrategyConfig,
    protocol: EvaluationProtocol = GATE,
    *,
    permutations: int = PERMUTATIONS,
) -> GateResult:
    return check(
        session, con, root, list(run_ids), cfg=cfg, protocol=protocol,
        permutations=permutations, seed=0,
    )  # fmt: skip


def with_(**changes: object) -> EvaluationProtocol:
    return GATE.model_copy(update=changes)


def dev_spans(session: Session, run_id: int, dev: Window = GATE.dev) -> list[Span]:
    return [
        s
        for s in spans_of_run(session, run_id)
        if s.entry_session >= dev.start and s.exit_session <= dev.end
    ]


def reference(
    session: Session,
    con: duckdb.DuckDBPyConnection,
    root: Path,
    run_ids: Sequence[int],
    cfg: StrategyConfig,
) -> tuple[dict[int, list[Span]], dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Dev spans, real returns and null matrices, read as known at the last dev exit."""
    spans = {r: dev_spans(session, r) for r in run_ids}
    every = [s for legs in spans.values() for s in legs]
    panel = build_panel(
        session, con, root, every, cfg=cfg, as_of=close_of(max(s.exit_session for s in every))
    )
    nulls = {
        r: null_returns(legs, panel, permutations=PERMUTATIONS, seed=0,
                        slippage_bps=cfg.execute.fills.slippage_bps)
        for r, legs in spans.items()
    }  # fmt: skip
    reals = {r: np.array([s.ret for s in legs]) for r, legs in spans.items()}
    return spans, reals, nulls


# =============================================================================================
# walk_forward_windows
# =============================================================================================


def test_the_real_dev_window_gives_seven_half_years_after_two_years_of_training() -> None:
    dev = Window(start=date(2020, 1, 1), end=date(2025, 6, 30))

    pairs = walk_forward_windows(dev, 24, 6)

    assert len(pairs) == 7
    assert pairs[0] == (
        (date(2020, 1, 1), date(2021, 12, 31)),
        (date(2022, 1, 1), date(2022, 6, 30)),
    )
    assert pairs[1] == (
        (date(2020, 7, 1), date(2022, 6, 30)),
        (date(2022, 7, 1), date(2022, 12, 31)),
    )
    assert pairs[-1][1] == (date(2025, 1, 1), date(2025, 6, 30))
    assert pairs[-1][0] == (date(2023, 1, 1), date(2024, 12, 31))
    # Test windows tile dev from the first one on, without gaps or overlaps.
    for (_, (_, end)), (_, (start, _)) in zip(pairs, pairs[1:], strict=False):
        assert start.toordinal() == end.toordinal() + 1


@pytest.mark.parametrize(
    ("end", "count"),
    [(date(2025, 3, 30), 1), (date(2025, 3, 31), 2), (date(2025, 4, 29), 2)],
    ids=["one_day_short", "exactly", "almost_another"],
)
def test_a_test_window_is_kept_only_if_it_ends_inside_dev(end: date, count: int) -> None:
    pairs = walk_forward_windows(Window(start=date(2025, 1, 1), end=end), 1, 1)

    assert len(pairs) == count
    assert pairs[0] == (
        (date(2025, 1, 1), date(2025, 1, 31)),
        (date(2025, 2, 1), date(2025, 2, 28)),
    )
    assert all(test_end <= end for _, (_, test_end) in pairs)


def test_a_mid_month_start_steps_by_calendar_months() -> None:
    pairs = walk_forward_windows(Window(start=date(2020, 3, 15), end=date(2020, 8, 14)), 2, 1)

    assert pairs == [
        ((date(2020, 3, 15), date(2020, 5, 14)), (date(2020, 5, 15), date(2020, 6, 14))),
        ((date(2020, 4, 15), date(2020, 6, 14)), (date(2020, 6, 15), date(2020, 7, 14))),
        ((date(2020, 5, 15), date(2020, 7, 14)), (date(2020, 7, 15), date(2020, 8, 14))),
    ]


def test_windows_across_a_leap_day() -> None:
    pairs = walk_forward_windows(Window(start=date(2023, 3, 1), end=date(2025, 2, 28)), 12, 12)

    assert pairs == [((date(2023, 3, 1), date(2024, 2, 29)), (date(2024, 3, 1), date(2025, 2, 28)))]


def test_a_dev_window_shorter_than_train_plus_test_has_no_windows() -> None:
    assert walk_forward_windows(Window(start=date(2025, 1, 1), end=date(2025, 1, 31)), 1, 1) == []


# =============================================================================================
# check: normal path
# =============================================================================================


def test_null_is_the_family_over_dev_legs_read_as_known_at_the_last_dev_exit(
    session: Session, con: duckdb.DuckDBPyConnection, root: Path, runs: Runs, cfg: StrategyConfig
) -> None:
    result = gate(session, con, root, [runs.a, runs.b], cfg)

    _, reals, nulls = reference(session, con, root, [runs.a, runs.b], cfg)
    want = max_statistic({str(r): reals[r] for r in reals}, {str(r): nulls[r] for r in nulls})
    assert set(result.variants) == {runs.a, runs.b}
    for r in (runs.a, runs.b):
        for field in ("trades", "mean", "null_mean", "excess", "z", "p_raw", "p_adjusted"):
            assert getattr(result.variants[r], field) == pytest.approx(
                getattr(want[str(r)], field)
            ), field
    # Measured on this world: A beats every one of the 39 permutations.
    assert result.variants[runs.a].trades == 7
    assert result.variants[runs.b].trades == 3
    assert result.variants[runs.a].p_adjusted == pytest.approx(1 / 40)


def test_the_candidate_is_the_run_with_the_highest_dev_sharpe(
    session: Session, con: duckdb.DuckDBPyConnection, root: Path, runs: Runs, cfg: StrategyConfig
) -> None:
    # A: Sharpe 15.97 over dev; B: 4.50 (measured). The order the runs are given in is no hint.
    result = gate(session, con, root, [runs.b, runs.a], cfg)

    assert result.candidate == runs.a
    assert (result.dev.start, result.dev.end) == (DEV_START, DEV_END)


def test_walk_forward_picks_on_the_train_month_and_scores_the_test_month(
    session: Session, con: duckdb.DuckDBPyConnection, root: Path, runs: Runs, cfg: StrategyConfig
) -> None:
    # January Sharpe: B 23.2 > A 15.5, so (Jan, Feb) trusts B, whose February leg lost 4%.
    # February and March: A 16.3 and 14.8 against B 0.8 and -0.7 (measured).
    result = gate(session, con, root, [runs.a, runs.b], cfg)

    spans, reals, nulls = reference(session, con, root, [runs.a, runs.b], cfg)

    def expected(run_id: int, test: list[date]) -> tuple[int, float, float]:
        idx = [i for i, s in enumerate(spans[run_id]) if test[0] <= s.entry_session <= test[-1]]
        return len(idx), float(reals[run_id][idx].mean()), float(nulls[run_id][:, idx].mean())

    assert [(s.train, s.test) for s in result.segments] == [
        ((date(2025, 1, 1), date(2025, 1, 31)), (date(2025, 2, 1), date(2025, 2, 28))),
        ((date(2025, 2, 1), date(2025, 2, 28)), (date(2025, 3, 1), date(2025, 3, 31))),
        ((date(2025, 3, 1), date(2025, 3, 31)), (date(2025, 4, 1), date(2025, 4, 30))),
    ]
    assert [s.picked for s in result.segments] == [runs.b, runs.a, runs.a]
    for seg, (run_id, test) in zip(
        result.segments, [(runs.b, FEB), (runs.a, MAR), (runs.a, APR)], strict=True
    ):
        legs, mean, null_mean = expected(run_id, test)
        assert seg.legs == legs
        assert seg.mean == pytest.approx(mean)
        assert seg.null_mean == pytest.approx(null_mean)
    assert [s.legs for s in result.segments] == [1, 2, 3]  # April's last leg exits on DEV_END
    assert [s.mean for s in result.segments] == pytest.approx([-0.04, 0.06, 0.22 / 3])
    assert [s.positive for s in result.segments] == [False, True, True]
    assert result.walk_forward_positive == 2
    assert all(isinstance(s, Segment) for s in result.segments)


def test_thin_edge_drops_the_best_legs_and_charges_only_the_real_returns(
    session: Session, con: duckdb.DuckDBPyConnection, root: Path, runs: Runs, cfg: StrategyConfig
) -> None:
    result = gate(session, con, root, [runs.a, runs.b], cfg)

    _, reals, nulls = reference(session, con, root, [runs.a, runs.b], cfg)
    real, null = reals[runs.a], nulls[runs.a]
    keep = np.argsort(-real)[3:]  # drop +10%, +9%, +8%; returns are distinct
    c = 10 / 10_000
    charged = (1 + real[keep]) * (1 - c) / (1 + c) - 1
    assert sorted(real[keep].tolist()) == pytest.approx([0.04, 0.05, 0.06, 0.07])
    assert result.thin_excess == pytest.approx(float(charged.mean() - null[:, keep].mean()))
    assert result.thin_edge_ok is True


def test_the_mde_is_the_normal_approximation_over_the_candidates_dev_returns(
    session: Session, con: duckdb.DuckDBPyConnection, root: Path, runs: Runs, cfg: StrategyConfig
) -> None:
    result = gate(session, con, root, [runs.a, runs.b], cfg)

    z = statistics.NormalDist().inv_cdf
    returns = [leg.ret for leg in A_LEGS]
    factor = z(1 - GATE.alpha) + z(GATE.mde_power)
    assert result.mde == pytest.approx(factor * statistics.stdev(returns) / math.sqrt(7))
    # The real-data reference the report is read against: sd 14.3%, 170 legs -> about 2.7%.
    assert factor * 0.143 / math.sqrt(170) == pytest.approx(0.027, abs=0.0005)


def test_versus_is_the_candidates_comparison_over_dev(
    session: Session, con: duckdb.DuckDBPyConnection, root: Path, runs: Runs, cfg: StrategyConfig
) -> None:
    result = gate(session, con, root, [runs.a, runs.b], cfg)

    want = compare_run(session, con, root, runs.a, GATE, window=(DEV_START, DEV_END))
    assert result.versus == want
    assert (result.versus.start, result.versus.end) == (DEV_DAYS[0], DEV_DAYS[-1])
    # Measured: A total 50.4% and Sharpe 15.97, drawdown 0; SPY 3.8% / 2.96, basket 21.9% / 7.76
    # with a 0.25% drawdown.
    assert result.versus.drawdown_not_worse_than_basket is True
    assert result.versus.sharpe_at_least_best is True
    assert result.versus.beats_spy_and_basket_total is True


def test_a_run_meeting_every_criterion_passes_with_no_reasons(
    session: Session, con: duckdb.DuckDBPyConnection, root: Path, runs: Runs, cfg: StrategyConfig
) -> None:
    result = gate(session, con, root, [runs.a, runs.b], cfg)

    assert result.passed is True
    assert result.reasons == []


def test_versions_on_dev_counts_each_version_that_traded_or_marked_in_dev_once(
    session: Session, con: duckdb.DuckDBPyConnection, root: Path, runs: Runs, cfg: StrategyConfig
) -> None:
    # v1 (A) and v2 (B); a second v1 run with only dev rows adds nothing; v4 with one dev trade
    # and no rows counts; another strategy's version does not; v3 only in the holdout does not.
    extra_run(session, runs, DEV_DAYS[:5], [0.0] * 5, [])
    v4 = version(session, runs.strategy_dir, "execute.exit.max_sessions=50")
    v4_run = new_run(session, v4)
    book(session, v4_run, runs.decision, [B_LEGS[0]])
    other = version(session, runs.strategy_dir, None, strategy_id="other_strategy")
    extra_run(session, runs, DEV_DAYS, [0.001] * len(DEV_DAYS), [A_LEGS[0]], version_id=other,
              strategy_id="other_strategy")  # fmt: skip
    add_holdout(session, runs)

    result = gate(session, con, root, [runs.a, runs.b], cfg)

    assert result.versions_on_dev == 3


# =============================================================================================
# check: boundaries
# =============================================================================================


def test_a_leg_entering_before_dev_is_left_out(
    session: Session, con: duckdb.DuckDBPyConnection, root: Path, runs: Runs, cfg: StrategyConfig
) -> None:
    before = gate(session, con, root, [runs.a, runs.b], cfg)
    book(
        session, runs.a, runs.decision, [Leg("P1", d(12, 16, 2024), at_open(d(12, 19, 2024)), 0.4)]
    )

    after = gate(session, con, root, [runs.a, runs.b], cfg)

    assert after.variants[runs.a].trades == 7
    assert after == before


def test_a_run_starting_before_dev_is_measured_from_its_last_row_before_dev(
    session: Session, con: duckdb.DuckDBPyConnection, root: Path, runs: Runs, cfg: StrategyConfig
) -> None:
    # C lost heavily in December (alternating -3% / +1%) and then beat A every dev session
    # (+1% / +0.05%). Over dev alone C is the candidate; counted from its first row it would not
    # be, and its dev return would carry December's losses.
    pre = alternating(len(PRE_DEV_DAYS), -0.03, 0.01)
    c = new_run(session, runs.v1)
    december = add_equity(session, c, PRE_DEV_DAYS, pre)
    dev_values = add_equity(
        session, c, DEV_DAYS, alternating(len(DEV_DAYS), 0.01, 0.0005), start=december[-1]
    )
    book(session, c, runs.decision, A_LEGS)

    result = gate(session, con, root, [runs.a, runs.b, c], cfg)

    assert result.candidate == c
    want = curve_metrics(dev_values, base=december[-1])
    assert result.versus.strategy.total_return == pytest.approx(want.total_return)
    assert result.versus.strategy.sharpe == pytest.approx(want.sharpe)
    assert result.versus.start == DEV_DAYS[0]


def test_equal_sharpes_go_to_the_lowest_run_id(
    session: Session, con: duckdb.DuckDBPyConnection, root: Path, runs: Runs, cfg: StrategyConfig
) -> None:
    twin = extra_run(session, runs, DEV_DAYS, a_returns(DEV_DAYS), A_LEGS)
    assert twin > runs.a

    result = gate(session, con, root, [twin, runs.b, runs.a], cfg)

    assert result.candidate == runs.a
    assert [s.picked for s in result.segments] == [runs.b, runs.a, runs.a]


def test_a_run_without_a_sharpe_ranks_below_a_losing_one(
    session: Session, con: duckdb.DuckDBPyConnection, root: Path, runs: Runs, cfg: StrategyConfig
) -> None:
    flat = extra_run(session, runs, DEV_DAYS, [0.0] * len(DEV_DAYS), A_LEGS)
    losing = extra_run(session, runs, DEV_DAYS, alternating(len(DEV_DAYS), -0.005, 0.002), B_LEGS)

    result = gate(session, con, root, [flat, losing], cfg)

    assert result.candidate == losing
    assert [s.picked for s in result.segments] == [losing, losing, losing]


def test_the_train_base_is_the_row_before_the_window(
    session: Session, con: duckdb.DuckDBPyConnection, root: Path, runs: Runs, cfg: StrategyConfig
) -> None:
    # X is flat except +3% on February's first session: from the last January row that is a
    # Sharpe of 3.64 over February; from February's first row it is no Sharpe at all. Y swings
    # +0.5% / -0.4% (Sharpe 2.53). So the (Feb, Mar) pair must trust X.
    x_returns = [0.03 if day == FEB[0] else 0.0 for day in DEV_DAYS]
    x = extra_run(session, runs, DEV_DAYS, x_returns, A_LEGS)
    y = extra_run(session, runs, DEV_DAYS, alternating(len(DEV_DAYS), 0.005, -0.004), B_LEGS)

    result = gate(session, con, root, [x, y], cfg)

    assert result.segments[1].train == (date(2025, 2, 1), date(2025, 2, 28))
    assert result.segments[1].picked == x


def test_a_train_window_without_rows_picks_nothing(
    session: Session, con: duckdb.DuckDBPyConnection, root: Path, runs: Runs, cfg: StrategyConfig
) -> None:
    # Dev from 2024-12-01: neither A nor B has a December row.
    early = with_(dev=Window(start=date(2024, 12, 1), end=DEV_END))

    result = gate(session, con, root, [runs.a, runs.b], cfg, early)

    first = result.segments[0]
    assert first.train == (date(2024, 12, 1), date(2024, 12, 31))
    assert (first.picked, first.legs, first.mean, first.null_mean, first.positive) == (
        None, 0, None, None, False,
    )  # fmt: skip
    assert [s.picked for s in result.segments[1:]] == [runs.b, runs.a, runs.a]
    assert result.walk_forward_positive == 2


def test_a_pick_without_legs_in_the_test_window_is_not_positive(
    session: Session, con: duckdb.DuckDBPyConnection, root: Path, runs: Runs, cfg: StrategyConfig
) -> None:
    # C alone has December rows, so (Dec, Jan) trusts C; C never trades in January.
    early = with_(dev=Window(start=date(2024, 12, 1), end=DEV_END))
    days = [*PRE_DEV_DAYS, *DEV_DAYS]
    c = extra_run(session, runs, days, alternating(len(days), 0.002, -0.001), [A_LEGS[2]])

    result = gate(session, con, root, [runs.a, runs.b, c], cfg, early)

    first = result.segments[0]
    assert first.picked == c
    assert (first.legs, first.mean, first.null_mean, first.positive) == (0, None, None, False)


def test_dropping_every_leg_fails_the_thin_edge(
    session: Session, con: duckdb.DuckDBPyConnection, root: Path, runs: Runs, cfg: StrategyConfig
) -> None:
    result = gate(session, con, root, [runs.a, runs.b], cfg,
                  with_(thin_edge=GATE.thin_edge.model_copy(update={"drop_top": 7})))  # fmt: skip

    assert result.thin_edge_ok is False
    assert set(result.reasons) == {"thin_edge"}
    assert result.passed is False


def test_the_extra_cost_is_charged_to_real_returns_only(
    session: Session, con: duckdb.DuckDBPyConnection, root: Path, runs: Runs, cfg: StrategyConfig
) -> None:
    # 20% a side turns +7% into -28.7%; the null is not charged, so the edge is gone.
    extra = {"drop_top": 0, "extra_cost_bps": 2_000.0}
    costly = with_(thin_edge=GATE.thin_edge.model_copy(update=extra))

    result = gate(session, con, root, [runs.a, runs.b], cfg, costly)

    _, reals, nulls = reference(session, con, root, [runs.a, runs.b], cfg)
    charged = (1 + reals[runs.a]) * 0.8 / 1.2 - 1
    assert result.thin_excess == pytest.approx(float(charged.mean() - nulls[runs.a].mean()))
    assert result.thin_edge_ok is False


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"alpha": 0.02}, "null"),  # p adjusted 0.025
        (
            {"walk_forward": {"train_months": 1, "test_months": 1, "min_positive": 3}},
            "walk_forward",
        ),
    ],
    ids=["null", "walk_forward"],
)
def test_a_failing_protocol_criterion_is_the_only_reason(
    session: Session,
    con: duckdb.DuckDBPyConnection,
    root: Path,
    runs: Runs,
    cfg: StrategyConfig,
    changes: dict[str, object],
    reason: str,
) -> None:
    p = EvaluationProtocol.model_validate({**GATE.model_dump(), **changes})

    result = gate(session, con, root, [runs.a, runs.b], cfg, p)

    assert result.passed is False
    assert set(result.reasons) == {reason}


def test_a_soaring_spy_fails_the_sharpe_and_total_criteria(
    session: Session,
    con: duckdb.DuckDBPyConnection,
    tmp_path: Path,
    runs: Runs,
    cfg: StrategyConfig,
) -> None:
    # SPY alternating +3% / +1%: total and Sharpe above A's; its drawdown is not compared.
    hot = store(tmp_path / "hot", spy=(0.03, 0.01))

    result = gate(session, con, hot, [runs.a, runs.b], cfg)

    assert result.passed is False
    assert set(result.reasons) == {"sharpe_at_least_best", "beats_spy_and_basket_total"}


def test_a_candidate_deeper_in_drawdown_than_the_basket_fails_that_criterion(
    session: Session, con: duckdb.DuckDBPyConnection, root: Path, runs: Runs, cfg: StrategyConfig
) -> None:
    # A's path with one -3% session in March: drawdown 3% against the basket's 0.25%; its
    # Sharpe and total stay above SPY's and the basket's.
    dipped = [(-0.03 if day == MAR[5] else r) for day, r in zip(DEV_DAYS, a_returns(DEV_DAYS),
                                                                    strict=True)]  # fmt: skip
    a2 = extra_run(session, runs, DEV_DAYS, dipped, A_LEGS)

    result = gate(session, con, root, [a2, runs.b], cfg)

    assert result.candidate == a2
    assert set(result.reasons) == {"drawdown_not_worse_than_basket"}


def test_a_run_without_dev_legs_raises(
    session: Session, con: duckdb.DuckDBPyConnection, root: Path, runs: Runs, cfg: StrategyConfig
) -> None:
    late_only = extra_run(session, runs, DEV_DAYS, a_returns(DEV_DAYS), A_LATE)

    with pytest.raises(ValueError):
        gate(session, con, root, [runs.a, late_only], cfg)


def test_a_run_without_dev_equity_rows_raises_and_says_replay(
    session: Session, con: duckdb.DuckDBPyConnection, root: Path, runs: Runs, cfg: StrategyConfig
) -> None:
    holdout_rows = extra_run(session, runs, HOLDOUT_DAYS, [0.01] * len(HOLDOUT_DAYS), A_LEGS)
    no_rows = extra_run(session, runs, [], [], A_LEGS)

    for run_id in (holdout_rows, no_rows):
        with pytest.raises(ValueError, match="replay"):
            gate(session, con, root, [runs.a, run_id], cfg)


def test_an_unknown_run_raises_lookup_error(
    session: Session, con: duckdb.DuckDBPyConnection, root: Path, runs: Runs, cfg: StrategyConfig
) -> None:
    with pytest.raises(LookupError):
        gate(session, con, root, [runs.a, 9999], cfg)


# =============================================================================================
# check: leakage
# =============================================================================================


def test_holdout_trades_rows_bars_versions_and_filings_change_nothing(
    session: Session,
    con: duckdb.DuckDBPyConnection,
    tmp_path: Path,
    root: Path,
    runs: Runs,
    cfg: StrategyConfig,
) -> None:
    # Before: a store that ends on DEV_END and a database with nothing after it. After: the full
    # store (prices x10 and volume x100 after DEV_END, a P2 split dated in dev but filed in
    # May), holdout equity under which B would outrank A, holdout legs of +30%, a leg entering
    # in dev and exiting after it (+50%), run Z of a third version trading only in the holdout,
    # and a new filer's 8-K.
    cut = store(tmp_path / "cut", through=DEV_END)
    before = gate(session, con, cut, [runs.a, runs.b], cfg)
    add_holdout(session, runs)

    after = gate(session, con, root, [runs.a, runs.b], cfg)

    # Only the count of legs left out moves: the straddling leg is named, not used.
    assert (before.left_out, after.left_out) == (0, 1)
    assert after == dataclasses.replace(before, left_out=1)
    assert after.versions_on_dev == 2
    assert after.variants[runs.a].trades == 7


def test_a_leg_entering_in_dev_and_exiting_after_it_is_left_out(
    session: Session, con: duckdb.DuckDBPyConnection, root: Path, runs: Runs, cfg: StrategyConfig
) -> None:
    before = gate(session, con, root, [runs.a, runs.b], cfg)
    book(session, runs.a, runs.decision, [A_LATE[0]])  # 04-28 -> 05-02, +50%

    after = gate(session, con, root, [runs.a, runs.b], cfg)

    assert after.variants[runs.a].trades == 7
    assert after.segments[2].legs == 3
    assert (before.left_out, after.left_out) == (0, 1)
    assert after == dataclasses.replace(before, left_out=1)


def test_a_walk_forward_pick_ignores_the_test_windows_rows(
    session: Session, con: duckdb.DuckDBPyConnection, root: Path, runs: Runs, cfg: StrategyConfig
) -> None:
    # A2 is A with a February of +5% / +4% a day: over January and February together it would
    # outrank B, over January alone it does not. The (Jan, Feb) pair must still trust B.
    hot_feb = [
        (0.05 if i % 2 == 0 else 0.04) if day in FEB else r
        for i, (day, r) in enumerate(zip(DEV_DAYS, a_returns(DEV_DAYS), strict=True))
    ]
    a2 = extra_run(session, runs, DEV_DAYS, hot_feb, A_LEGS)

    plain = gate(session, con, root, [runs.a, runs.b], cfg)
    hot = gate(session, con, root, [a2, runs.b], cfg)

    assert plain.segments[0].picked == runs.b
    assert hot.segments[0].picked == runs.b
    assert hot.segments[0] == plain.segments[0]


# =============================================================================================
# review follow-ups
# =============================================================================================


def test_a_strategy_seen_before_the_protocol_never_passes_on_dev(
    session: Session, con: duckdb.DuckDBPyConnection, root: Path, runs: Runs, cfg: StrategyConfig
) -> None:
    # The passing family again, under a protocol that lists earnings_gap as replayed over these
    # dates before the protocol existed: every number stays, the verdict cannot be a pass.
    plain = gate(session, con, root, [runs.a, runs.b], cfg)
    seen = gate(
        session, con, root, [runs.a, runs.b], cfg, with_(seen_before_protocol=["earnings_gap"])
    )

    assert plain.passed is True and plain.seen_before_protocol is False
    assert seen.seen_before_protocol is True
    assert seen.passed is False
    assert seen.reasons == ["seen_before_protocol"]
    assert seen.variants == plain.variants and seen.segments == plain.segments


def test_too_few_scorable_windows_is_named_as_not_evaluable(
    session: Session, con: duckdb.DuckDBPyConnection, root: Path, runs: Runs, cfg: StrategyConfig
) -> None:
    # Dev from December: four windows, the first with no rows to pick on. Needing 4 positive
    # out of 3 scorable cannot happen; the reason says so instead of "no edge".
    p = EvaluationProtocol.model_validate(
        {**GATE.model_dump(), "dev": {"start": date(2024, 12, 1), "end": DEV_END},
         "walk_forward": {"train_months": 1, "test_months": 1, "min_positive": 4}}
    )  # fmt: skip

    result = gate(session, con, root, [runs.a, runs.b], cfg, p)

    assert len(result.segments) == 4
    assert sum(s.picked is not None for s in result.segments) == 3
    assert "walk_forward_not_evaluable" in result.reasons
    assert "walk_forward" not in result.reasons


def test_a_run_without_a_version_counts_as_one_attempt(
    session: Session, con: duckdb.DuckDBPyConnection, root: Path, runs: Runs, cfg: StrategyConfig
) -> None:
    # Two version-less runs that traded in dev: each is an attempt of its own (older than
    # versions, nothing says they were the same settings).
    for _ in range(2):
        legacy = new_run(session, None)
        book(session, legacy, runs.decision, [B_LEGS[0]])

    result = gate(session, con, root, [runs.a, runs.b], cfg)

    assert result.versions_on_dev == 4


def test_the_thin_edge_parts_are_reported_apart(
    session: Session, con: duckdb.DuckDBPyConnection, root: Path, runs: Runs, cfg: StrategyConfig
) -> None:
    result = gate(session, con, root, [runs.a, runs.b], cfg)

    _, reals, nulls = reference(session, con, root, [runs.a, runs.b], cfg)
    real, null = reals[runs.a], nulls[runs.a]
    keep = np.argsort(-real)[3:]
    c = 10 / 10_000
    assert result.thin_drop_excess == pytest.approx(float(real[keep].mean() - null[:, keep].mean()))
    charged = (1 + real) * (1 - c) / (1 + c) - 1
    assert result.thin_cost_excess == pytest.approx(float(charged.mean() - null.mean()))
