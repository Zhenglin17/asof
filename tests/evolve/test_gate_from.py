"""Guards against a gate measured "from a date" that still counts what came before it (a leg,
an equity row or a walk-forward window dated before the start), measures its first row against
the run's initial equity instead of the row before the start, lets the start move the sealed
end of dev into the holdout, or accepts a start outside dev without saying so.

API under test (new):

    EvaluationProtocol.from_date(start: date) -> EvaluationProtocol   (asof.evolve.protocol)
        a copy whose dev window is [start, dev.end]; every other key (holdout, clean_after,
        walk_forward, alpha, thin_edge, benchmarks, ...) unchanged; ValueError unless
        dev.start <= start <= dev.end.
    asof gate report ... --from YYYY-MM-DD                               (asof.cli)
        runs check() with load_protocol().from_date(start); a start outside dev exits 1 with
        one line; the report states the window it measured. `asof gate null` has no --from.

Interface points this file pins down where the spec left a choice:
- The helper is a method on EvaluationProtocol, ``from_date``, not a free function.
- ``check(..., protocol=GATE.from_date(DEV_START))`` equals ``check(..., protocol=GATE)``.
- A leg entering before the start and exiting after it is dropped, not counted in left_out.
- The report states the window on one line that holds both the start and dev's end.
- ``versions_on_dev`` under a later start is not pinned here.

World: tests/evolve/gate_world.py (provenance there): GATE has dev 2025-01-01 .. 2025-04-30,
holdout 2025-05-01 .. 2025-06-30, one-month train and test windows. Every leg of runs A and B
enters in February or later, so tests that need a dropped leg book January legs on A.
"""

import dataclasses
from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pytest
from sqlmodel import Session
from typer.testing import CliRunner

import asof.cli
from asof.cli import app
from asof.evolve.gate import GateResult, check
from asof.evolve.metrics import curve_metrics
from asof.evolve.null import Span, build_panel, max_statistic, null_returns, spans_of_run
from asof.evolve.protocol import EvaluationProtocol, Window
from asof.evolve.versus import compare_run
from asof.strategy import StrategyConfig, load_strategy
from tests.evolve.gate_world import (
    A_LEGS,
    DEV_DAYS,
    DEV_END,
    DEV_START,
    GATE,
    HOLDOUT_END,
    HOLDOUT_START,
    PERMUTATIONS,
    Leg,
    Runs,
    add_equity,
    add_holdout,
    alternating,
    at_open,
    book,
    close_of,
    d,
    month,
    new_run,
    seed,
    store,
)

FEB_1 = date(2025, 2, 1)  # a Saturday: the start need not be a session
JAN = month(1)
FROM_FEB = [x for x in DEV_DAYS if x >= FEB_1]

# January legs on A: one closed in January, one straddling the start (enters 01-28, exits
# 02-03). Both must drop out of a from-February measurement; plain dev counts both.
A_JANUARY = [
    Leg("P1", d(1, 13), at_open(d(1, 16)), 0.40),
    Leg("P2", d(1, 28), close_of(d(2, 3)), 0.35),
]


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
    run_ids: list[int],
    cfg: StrategyConfig,
    protocol: EvaluationProtocol,
) -> GateResult:
    return check(
        session, con, root, run_ids, cfg=cfg, protocol=protocol,
        permutations=PERMUTATIONS, seed=0,
    )  # fmt: skip


def window_spans(session: Session, run_id: int, dev: Window) -> list[Span]:
    return [
        s
        for s in spans_of_run(session, run_id)
        if s.entry_session >= dev.start and s.exit_session <= dev.end
    ]


def reference(
    session: Session,
    con: duckdb.DuckDBPyConnection,
    root: Path,
    run_ids: list[int],
    cfg: StrategyConfig,
    dev: Window,
) -> dict[str, Any]:
    """The family's max-statistic results over ``dev``'s legs, read as known at the last exit."""
    spans = {r: window_spans(session, r, dev) for r in run_ids}
    every = [s for legs in spans.values() for s in legs]
    panel = build_panel(
        session, con, root, every, cfg=cfg, as_of=close_of(max(s.exit_session for s in every))
    )
    nulls = {
        str(r): null_returns(legs, panel, permutations=PERMUTATIONS, seed=0,
                             slippage_bps=cfg.execute.fills.slippage_bps)
        for r, legs in spans.items()
    }  # fmt: skip
    reals = {str(r): np.array([s.ret for s in legs]) for r, legs in spans.items()}
    return max_statistic(reals, nulls)


# =============================================================================================
# normal path
# =============================================================================================


def test_from_date_moves_only_the_dev_start() -> None:
    later = GATE.from_date(FEB_1)

    assert isinstance(later, EvaluationProtocol)
    assert later.dev == Window(start=FEB_1, end=DEV_END)
    assert later.model_dump() == {**GATE.model_dump(), "dev": {"start": FEB_1, "end": DEV_END}}
    assert GATE.dev == Window(start=DEV_START, end=DEV_END)  # the original is untouched


def test_legs_before_the_start_drop_out_of_every_count(
    session: Session, con: duckdb.DuckDBPyConnection, root: Path, runs: Runs, cfg: StrategyConfig
) -> None:
    book(session, runs.a, runs.decision, A_JANUARY)
    later = GATE.from_date(FEB_1)

    plain = gate(session, con, root, [runs.a, runs.b], cfg, GATE)
    result = gate(session, con, root, [runs.a, runs.b], cfg, later)

    assert plain.variants[runs.a].trades == 9  # both January legs count over all of dev
    assert result.variants[runs.a].trades == 7
    assert result.variants[runs.b].trades == 3
    assert result.left_out == 0  # the straddling January leg is dropped, not "left out"
    # but it is reported: its February moves are in the equity rows the pick reads
    assert (plain.carried_in, result.carried_in) == (0, 1)
    assert result.dev == later.dev
    want = reference(session, con, root, [runs.a, runs.b], cfg, later.dev)
    for r in (runs.a, runs.b):
        for field in ("trades", "mean", "null_mean", "excess", "z", "p_raw", "p_adjusted"):
            assert getattr(result.variants[r], field) == pytest.approx(
                getattr(want[str(r)], field)
            ), field


def test_walk_forward_windows_start_at_the_start(
    session: Session, con: duckdb.DuckDBPyConnection, root: Path, runs: Runs, cfg: StrategyConfig
) -> None:
    # February and March Sharpe: A 16.3 and 14.8 against B 0.8 and -0.7 (measured in
    # test_gate.py), so both pairs pick A; the (Jan, Feb) pair that picked B is gone.
    result = gate(session, con, root, [runs.a, runs.b], cfg, GATE.from_date(FEB_1))

    assert [(s.train, s.test) for s in result.segments] == [
        ((date(2025, 2, 1), date(2025, 2, 28)), (date(2025, 3, 1), date(2025, 3, 31))),
        ((date(2025, 3, 1), date(2025, 3, 31)), (date(2025, 4, 1), date(2025, 4, 30))),
    ]
    assert [s.picked for s in result.segments] == [runs.a, runs.a]
    assert [s.legs for s in result.segments] == [2, 3]
    assert [s.mean for s in result.segments] == pytest.approx([0.06, 0.22 / 3])
    assert (result.walk_forward_positive, result.walk_forward_scorable) == (2, 2)


def test_versus_is_compare_run_over_the_start_to_dev_end(
    session: Session, con: duckdb.DuckDBPyConnection, root: Path, runs: Runs, cfg: StrategyConfig
) -> None:
    result = gate(session, con, root, [runs.a, runs.b], cfg, GATE.from_date(FEB_1))

    assert result.candidate == runs.a
    assert result.versus == compare_run(session, con, root, runs.a, GATE, window=(FEB_1, DEV_END))
    assert (result.versus.start, result.versus.end) == (FROM_FEB[0], DEV_END)


def test_the_candidate_is_picked_and_measured_from_the_last_row_before_the_start(
    session: Session, con: duckdb.DuckDBPyConnection, root: Path, runs: Runs, cfg: StrategyConfig
) -> None:
    # C lost heavily in January (-3% / +1%) and beat A on every later session (+1% / +0.05%):
    # over all of dev A is the candidate, from February C is, measured from its last January
    # row (counted from its initial equity, January's losses would be in its return).
    c = new_run(session, runs.v1)
    january = add_equity(session, c, JAN, alternating(len(JAN), -0.03, 0.01))
    later_values = add_equity(
        session, c, FROM_FEB, alternating(len(FROM_FEB), 0.01, 0.0005), start=january[-1]
    )
    book(session, c, runs.decision, A_LEGS)

    plain = gate(session, con, root, [runs.a, runs.b, c], cfg, GATE)
    result = gate(session, con, root, [runs.a, runs.b, c], cfg, GATE.from_date(FEB_1))

    assert plain.candidate == runs.a
    assert result.candidate == c
    want = curve_metrics(later_values, base=january[-1])
    assert result.versus.strategy.total_return == pytest.approx(want.total_return)
    assert result.versus.strategy.sharpe == pytest.approx(want.sharpe)
    assert result.versus.start == FROM_FEB[0]


# =============================================================================================
# boundaries
# =============================================================================================


def test_from_dev_start_is_the_plain_protocol(
    session: Session, con: duckdb.DuckDBPyConnection, root: Path, runs: Runs, cfg: StrategyConfig
) -> None:
    book(session, runs.a, runs.decision, A_JANUARY)

    assert GATE.from_date(DEV_START) == GATE
    plain = gate(session, con, root, [runs.a, runs.b], cfg, GATE)
    same = gate(session, con, root, [runs.a, runs.b], cfg, GATE.from_date(DEV_START))

    assert same == plain


def test_from_dev_end_is_allowed_but_a_family_without_legs_there_raises(
    session: Session, con: duckdb.DuckDBPyConnection, root: Path, runs: Runs, cfg: StrategyConfig
) -> None:
    # A's last leg enters 04-28: nothing enters on 04-30, so the existing no-leg error fires.
    last = GATE.from_date(DEV_END)

    assert last.dev == Window(start=DEV_END, end=DEV_END)
    with pytest.raises(ValueError, match="no leg entering and exiting within dev"):
        gate(session, con, root, [runs.a, runs.b], cfg, last)


@pytest.mark.parametrize(
    "start",
    [date(2024, 12, 31), date(2020, 1, 1), HOLDOUT_START, date(2025, 6, 2), date(2026, 1, 1)],
    ids=["day_before_dev", "long_before_dev", "holdout_start", "inside_holdout", "after_all"],
)
def test_a_start_outside_dev_raises(start: date) -> None:
    with pytest.raises(ValueError):
        GATE.from_date(start)


# =============================================================================================
# leakage
# =============================================================================================


@pytest.mark.parametrize("start", [DEV_START, FEB_1, date(2025, 3, 17), DEV_END])
def test_from_date_never_moves_the_sealed_end_or_the_holdout(start: date) -> None:
    later = GATE.from_date(start)

    assert later.dev.end == DEV_END
    assert later.holdout == Window(start=HOLDOUT_START, end=HOLDOUT_END)
    assert later.clean_after == GATE.clean_after


def test_holdout_trades_rows_bars_and_filings_change_nothing_from_a_later_start(
    session: Session,
    con: duckdb.DuckDBPyConnection,
    tmp_path: Path,
    root: Path,
    runs: Runs,
    cfg: StrategyConfig,
) -> None:
    # Before: a store ending on DEV_END and nothing after it in the database. After: the full
    # store (prices x10 after DEV_END, a P2 split dated in dev but filed in May), holdout
    # equity under which B would outrank A, holdout legs, a leg entering 04-28 and exiting in
    # May (+50%), run Z only in the holdout, a late filer. The future exists but is not visible.
    later = GATE.from_date(FEB_1)
    cut = store(tmp_path / "cut", through=DEV_END)
    before = gate(session, con, cut, [runs.a, runs.b], cfg, later)
    add_holdout(session, runs)

    after = gate(session, con, root, [runs.a, runs.b], cfg, later)

    assert (before.left_out, after.left_out) == (0, 1)
    assert after == dataclasses.replace(before, left_out=1)
    assert after.dev.end == DEV_END
    assert after.versus.end == DEV_END
    assert after.variants[runs.a].trades == 7


# =============================================================================================
# CLI
# =============================================================================================

runner = CliRunner()


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    store(tmp_path)
    return tmp_path


@pytest.fixture
def gate_protocol(monkeypatch: pytest.MonkeyPatch) -> EvaluationProtocol:
    monkeypatch.setattr(asof.cli, "load_protocol", lambda *args, **kwargs: GATE)
    return GATE


def report_args(data_dir: Path, *run_ids: int, extra: tuple[str, ...] = ()) -> list[str]:
    out = ["gate", "report"]
    for run_id in run_ids:
        out += ["--run", str(run_id)]
    return [
        *out, "--permutations", str(PERMUTATIONS), "--seed", "0",
        "--strategies-dir", str(data_dir / "strategies"),
        "--data-dir", str(data_dir), "--db", str(data_dir / "meta.db"), *extra,
    ]  # fmt: skip


def default_report(data_dir: Path, *run_ids: int) -> Path:
    return data_dir / "reports" / f"gate-{'-'.join(str(r) for r in run_ids)}.md"


def window_lines(text: str, start: str, end: str) -> list[str]:
    return [line for line in text.splitlines() if start in line and end in line]


def test_cli_from_measures_and_states_the_later_window(
    data_dir: Path, session: Session, gate_protocol: EvaluationProtocol
) -> None:
    runs = seed(session, data_dir)

    plain = runner.invoke(app, report_args(data_dir, runs.a, runs.b))
    assert plain.exit_code == 0, plain.output
    plain_text = default_report(data_dir, runs.a, runs.b).read_text()
    result = runner.invoke(
        app, report_args(data_dir, runs.a, runs.b, extra=("--from", "2025-02-01"))
    )

    assert result.exit_code == 0, result.output
    # the later report has its own default name: the plain one is not overwritten
    later = default_report(data_dir, runs.a, runs.b).with_name("gate-1-2-from-20250201.md")
    assert (runs.a, runs.b) == (1, 2)  # fixture: the name above
    text = later.read_text()
    assert default_report(data_dir, runs.a, runs.b).read_text() == plain_text
    # No line of the plain report holds both dates; the later report states its window.
    assert not window_lines(plain_text, "2025-02-01", "2025-04-30")
    assert window_lines(text, "2025-02-01", "2025-04-30")
    # The start reached check(): two walk-forward windows, both positive (see above).
    assert "walk-forward 2 of 2 evaluable windows positive" in result.output
    assert "2025-01-01 .. 2025-01-31" not in text


@pytest.mark.parametrize("start", ["2024-12-31", "2025-05-01"], ids=["before_dev", "in_holdout"])
def test_cli_from_outside_dev_exits_one_with_one_line(
    data_dir: Path, session: Session, gate_protocol: EvaluationProtocol, start: str
) -> None:
    runs = seed(session, data_dir)

    result = runner.invoke(app, report_args(data_dir, runs.a, runs.b, extra=("--from", start)))

    assert result.exit_code == 1, result.output
    assert isinstance(result.exception, SystemExit)
    assert "Traceback" not in result.output
    assert len(result.output.strip().splitlines()) == 1, result.output
    assert not default_report(data_dir, runs.a, runs.b).exists()


def test_gate_null_has_no_from_option(
    data_dir: Path, session: Session, gate_protocol: EvaluationProtocol
) -> None:
    runs = seed(session, data_dir)
    out = ["gate", "null", "--run", str(runs.a), "--permutations", str(PERMUTATIONS),
           "--strategies-dir", str(data_dir / "strategies"), "--data-dir", str(data_dir),
           "--db", str(data_dir / "meta.db"), "--from", "2025-02-01"]  # fmt: skip

    result = runner.invoke(app, out)

    assert result.exit_code == 2, result.output  # usage error: unknown option
