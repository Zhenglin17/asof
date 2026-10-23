"""Guards against a comparison of an llm replay with its take-all baseline that misstates what the
judge did: decisions matched across runs by the wrong key, passes scored by the llm run (where they
have no trades) instead of by what take-all earned on them, calibration bins that drop empty bins
or misplace boundary convictions, possibly contaminated decisions (before the model's training
cutoff) mixed silently with clean ones, and a decision present in only one run dropped instead of
listed.

API under test (asof.replay.compare):

    compare_runs(session, base_run_id, run_id) -> Comparison
    Comparison: base_run_id, run_id, cutoff (the llm run's model_training_cutoff),
        all / before_cutoff / after_cutoff: Section (as_of < cutoff, as_of >= cutoff),
        unmatched: list[Unmatched] sorted by (as_of, run_id)
    Section: base: RunSummary, run: RunSummary,
        verdicts: dict[str, VerdictSummary] with exactly the keys take, pass, no_call,
        calibration: list[CalibrationBin], always the five BINS in order
    RunSummary: decisions, traded (decisions with at least one trade), total_pnl,
        mean_r (mean over traded decisions of the sum of their legs' r_multiple; None if none)
    VerdictSummary: count (llm decisions with payload["verdict"] == key), base_outcomes (how
        many of them the base run traded), base_pnl, base_mean_r (None if base_outcomes == 0)
    CalibrationBin: low, high, count, base_outcomes, base_win_rate (share of base_outcomes
        whose base pnl > 0; None if base_outcomes == 0); BINS [0, .4) [.4, .5) [.5, .6)
        [.6, .7) [.7, 1.0], the last closed
    Unmatched: run_id (the run the decision is in), entity_id, as_of
    ValueError: base run not take_all, or llm run without model_training_cutoff
    LookupError: unknown run id

    asof replay compare --base ID --run ID [--db FILE] prints it, one line per unmatched
    decision with its ticker and as_of date; exit 1 with one line on an error above.

No as_of: the comparison reads two finished runs, whose decisions and trades were written by the
replay under its own as_of (leakage tested in test_replay.py / test_replay_llm.py). The
unmatched listing is this module's leakage-type guard. Assumptions only real runs can confirm
(check on the first real llm-vs-take_all comparison):
  1. Run.model_training_cutoff is the model's documented cutoff: compare it with the source in
     judge.cutoff_source (model card) by hand.
  2. Matching by (entity_id, as_of) is complete: both runs use the same calendar and stamp
     decisions at close_of(day). Every unmatched row of the real comparison must be explained
     by a held security (the llm passed and judged the security again while take-all held it,
     or the reverse); any other unmatched row is a bug.
  3. A pass is scored by take-all's trade on the same candidate. Take-all's portfolio differs,
     so share counts (and pnl) can differ when a sizing cap binds; R does not depend on size.
     Check on the real comparison that matched takes have equal R in both runs.
  4. payload["verdict"] is on every llm decision (written by the replay's llm path).

Data: hand-built Run/Decision/Trade rows for exact arithmetic (table below), and the real replay
over tests/replay/two_gaps.py for the matching and the cutoff taken from the Run row.

Hand-built (cutoff = 2024-01-16 20:00 New York, exactly L4's as_of; days are 2024-01):
    base (take_all)                            llm
    B1 e1 01-02 +300 R 1.5                     L1 e1 01-02 take 1.0   +300 R 1.5
    B2 e2 01-04 -200 R -1 / +500 R 2.5         L2 e2 01-04 pass 0.4
    B3 e3 01-08 -100 R -0.5                    L3 e3 01-08 pass 0.3
    B4 e4 01-16 no trades                      L4 e4 01-16 take 0.65  no trades
    B5 e5 01-18 +100 R 0.5                     L5 e5 01-18 no_call
    B6 e6 01-22 -400 R -2                      L6 e6 01-22 take 0.7   -400 R -2
    B7 e7 01-24 no trades (base only)          L8 e8 01-26 pass 0.55  (llm only)
  all: base 7 decisions, 5 traded, pnl 200, mean R 0.2; llm 7, 2, -100, -0.25
       take 3 (base 2: -100, R -0.25); pass 3 (base 2: 200, R 0.5); no_call 1 (base 1: 100, 0.5)
       bins: [0,.4) 1 (1, 0.0); [.4,.5) 1 (1, 1.0); [.5,.6) 1 (0, None); [.6,.7) 1 (0, None);
             [.7,1] 2 (2, 0.5)
  before: base 3, 3, 500, 2.5/3; llm 3, 1, 300, 1.5; take 1 (1: 300, 1.5); pass 2 (2: 200, 0.5);
          no_call 0 (0: 0, None); bins 1 (1, 0.0), 1 (1, 1.0), 0, 0, 1 (1, 1.0)
  after:  base 4, 2, -300, -0.75; llm 4, 1, -400, -2.0; take 2 (1: -400, -2.0);
          pass 1 (0: 0, None); no_call 1 (1: 100, 0.5); bins 0, 0, 1 (0, None), 1 (0, None),
          1 (1, 0.0)
"""

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import duckdb
import httpx
import pytest
from sqlmodel import Session
from typer.testing import CliRunner

from asof.cli import app
from asof.judge import client
from asof.replay import close_of, compare, run_replay
from asof.store.models import Decision, Entity, Run, Trade
from tests.judge.fakes import LLM_JUDGE
from tests.replay.scenario import END, START, X
from tests.replay.test_replay import trades
from tests.replay.test_replay_llm import LLM, TAKE_ALL, config, good_answer, remember
from tests.replay.test_report import at, make_trade
from tests.replay.two_gaps import SECOND, two_gaps
from tests.store.factories import add, pk
from tests.strategy.files import write_strategy

compare_runs = compare.compare_runs
runner = CliRunner()

BINS = [(0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.0)]
CUTOFF = at(date(2024, 1, 16))


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def complete(self: Any, body: Any) -> Any:
        raise AssertionError("a comparison test called the model")

    def send(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a comparison test used the network")

    monkeypatch.setattr(client.OpenRouterClient, "complete", complete)
    monkeypatch.setattr(httpx.Client, "send", send)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


# =============================================================================================
# hand-built runs
# =============================================================================================


def new_run(session: Session, *, llm: bool, cutoff: datetime | None = CUTOFF) -> Run:
    return add(
        session,
        Run(
            strategy_id="earnings_gap",
            mode="replay",
            as_of=at(date(2024, 1, 31)),
            started_at=datetime(2026, 10, 22, tzinfo=UTC),
            git_sha="0" * 40,
            model_id="vendor/model-under-test" if llm else None,
            model_training_cutoff=cutoff if llm else None,
        ),
    )


def new_decision(
    session: Session,
    run: Run,
    entity_id: int,
    as_of: datetime,
    verdict: str | None = None,
    conviction: float | None = None,
) -> Decision:
    payload: dict[str, Any] = {"judge": "take_all" if verdict is None else "llm"}
    if verdict is not None:
        payload["verdict"] = verdict
    return add(
        session,
        Decision(
            run_id=pk(run),
            kind="trade",
            as_of=as_of,
            strategy_id="earnings_gap",
            entity_id=entity_id,
            direction="long" if verdict in (None, "take") else "flat",
            conviction=conviction,
            payload=payload,
        ),
    )


def leg(session: Session, decision: Decision, n: int, pnl: float, r: float, day: int) -> Trade:
    return make_trade(session, decision, n, pnl, r, "break", date(2024, 2, day))


@dataclass
class Hand:
    base: Run
    llm: Run
    entities: dict[str, int]


def day(n: int, hh: int = 20) -> datetime:
    return at(date(2024, 1, n), hh)


@pytest.fixture
def hand(session: Session) -> Hand:
    e = {
        t: pk(add(session, Entity(kind="company", ticker=t, name=f"{t} Inc.", cik=900 + i)))
        for i, t in enumerate(["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"], start=1)
    }
    base = new_run(session, llm=False)
    llm = new_run(session, llm=True)

    b1 = new_decision(session, base, e["E1"], day(2))
    leg(session, b1, 1, 300.0, 1.5, 1)
    b2 = new_decision(session, base, e["E2"], day(4))
    leg(session, b2, 1, -200.0, -1.0, 2)
    leg(session, b2, 2, 500.0, 2.5, 3)
    b3 = new_decision(session, base, e["E3"], day(8))
    leg(session, b3, 1, -100.0, -0.5, 4)
    new_decision(session, base, e["E4"], day(16))
    b5 = new_decision(session, base, e["E5"], day(18))
    leg(session, b5, 1, 100.0, 0.5, 5)
    b6 = new_decision(session, base, e["E6"], day(22))
    leg(session, b6, 1, -400.0, -2.0, 6)
    new_decision(session, base, e["E7"], day(24, 12))  # noon: same date in New York and UTC

    l1 = new_decision(session, llm, e["E1"], day(2), "take", 1.0)
    leg(session, l1, 1, 300.0, 1.5, 1)
    new_decision(session, llm, e["E2"], day(4), "pass", 0.4)
    new_decision(session, llm, e["E3"], day(8), "pass", 0.3)
    new_decision(session, llm, e["E4"], day(16), "take", 0.65)
    new_decision(session, llm, e["E5"], day(18), "no_call", None)
    l6 = new_decision(session, llm, e["E6"], day(22), "take", 0.7)
    leg(session, l6, 1, -400.0, -2.0, 6)
    new_decision(session, llm, e["E8"], day(26, 12), "pass", 0.55)
    return Hand(base, llm, e)


def summary(s: Any) -> tuple[Any, ...]:
    return (s.decisions, s.traded, s.total_pnl, s.mean_r)


def verdicts(section: Any) -> dict[str, tuple[Any, ...]]:
    return {
        k: (v.count, v.base_outcomes, v.base_pnl, v.base_mean_r)
        for k, v in section.verdicts.items()
    }


def bins(section: Any) -> list[tuple[Any, ...]]:
    return [(b.low, b.high, b.count, b.base_outcomes, b.base_win_rate) for b in section.calibration]


def approx(values: Any) -> Any:
    """pytest.approx over nested tuples / dicts / lists that may hold None."""
    if isinstance(values, dict):
        return {k: approx(v) for k, v in values.items()}
    if isinstance(values, list | tuple):
        return type(values)(approx(v) for v in values)
    return values if values is None else pytest.approx(values)


# ---- normal path ----------------------------------------------------------------------------


def test_run_summaries_count_decisions_trades_pnl_and_mean_r_per_decision(
    session: Session, hand: Hand
) -> None:
    c = compare_runs(session, pk(hand.base), pk(hand.llm))
    assert c.base_run_id == pk(hand.base)
    assert c.run_id == pk(hand.llm)
    assert summary(c.all.base) == approx((7, 5, 200.0, 0.2))
    assert summary(c.all.run) == approx((7, 2, -100.0, -0.25))


def test_each_verdict_is_scored_by_the_base_runs_outcome_of_the_same_candidates(
    session: Session, hand: Hand
) -> None:
    c = compare_runs(session, pk(hand.base), pk(hand.llm))
    assert verdicts(c.all) == approx(
        {
            "take": (3, 2, -100.0, -0.25),
            "pass": (3, 2, 200.0, 0.5),  # what passing gave up: take-all made 200 on them
            "no_call": (1, 1, 100.0, 0.5),
        }
    )


def test_calibration_bins_score_convictions_by_the_base_outcome(
    session: Session, hand: Hand
) -> None:
    # 0.4 opens [0.4, 0.5); 0.7 opens the last bin; 1.0 is inside it. L8 (0.55) has no base
    # decision and L4 (0.65) a base decision without trades: counted, never scored.
    c = compare_runs(session, pk(hand.base), pk(hand.llm))
    assert bins(c.all) == approx(
        [
            (0.0, 0.4, 1, 1, 0.0),
            (0.4, 0.5, 1, 1, 1.0),
            (0.5, 0.6, 1, 0, None),
            (0.6, 0.7, 1, 0, None),
            (0.7, 1.0, 2, 2, 0.5),
        ]
    )


def test_every_figure_is_split_at_the_training_cutoff(session: Session, hand: Hand) -> None:
    c = compare_runs(session, pk(hand.base), pk(hand.llm))
    assert c.cutoff == CUTOFF

    before = c.before_cutoff
    assert summary(before.base) == approx((3, 3, 500.0, 2.5 / 3))
    assert summary(before.run) == approx((3, 1, 300.0, 1.5))
    assert verdicts(before) == approx(
        {"take": (1, 1, 300.0, 1.5), "pass": (2, 2, 200.0, 0.5), "no_call": (0, 0, 0.0, None)}
    )
    assert bins(before) == approx(
        [
            (0.0, 0.4, 1, 1, 0.0),
            (0.4, 0.5, 1, 1, 1.0),
            (0.5, 0.6, 0, 0, None),
            (0.6, 0.7, 0, 0, None),
            (0.7, 1.0, 1, 1, 1.0),
        ]
    )

    after = c.after_cutoff  # L4 / B4 sit exactly on the cutoff: on/after it
    assert summary(after.base) == approx((4, 2, -300.0, -0.75))
    assert summary(after.run) == approx((4, 1, -400.0, -2.0))
    assert verdicts(after) == approx(
        {"take": (2, 1, -400.0, -2.0), "pass": (1, 0, 0.0, None), "no_call": (1, 1, 100.0, 0.5)}
    )
    assert bins(after) == approx(
        [
            (0.0, 0.4, 0, 0, None),
            (0.4, 0.5, 0, 0, None),
            (0.5, 0.6, 1, 0, None),
            (0.6, 0.7, 1, 0, None),
            (0.7, 1.0, 1, 1, 0.0),
        ]
    )


# ---- boundaries -----------------------------------------------------------------------------


def test_a_decision_in_only_one_run_is_listed_not_dropped(session: Session, hand: Hand) -> None:
    c = compare_runs(session, pk(hand.base), pk(hand.llm))
    assert [(u.run_id, u.entity_id, u.as_of) for u in c.unmatched] == [
        (pk(hand.base), hand.entities["E7"], day(24, 12)),
        (pk(hand.llm), hand.entities["E8"], day(26, 12)),
    ]


def test_matching_needs_the_same_as_of_not_just_the_same_entity(session: Session) -> None:
    # The same company decided one session apart is two unmatched decisions, not a match.
    entity = pk(add(session, Entity(kind="company", ticker="SAME", name="Same Inc.", cik=77)))
    base = new_run(session, llm=False)
    llm = new_run(session, llm=True)
    b = new_decision(session, base, entity, day(2))
    leg(session, b, 1, 300.0, 1.5, 1)
    new_decision(session, llm, entity, day(3), "pass", 0.45)

    c = compare_runs(session, pk(base), pk(llm))

    assert [(u.run_id, u.as_of) for u in c.unmatched] == [(pk(base), day(2)), (pk(llm), day(3))]
    assert verdicts(c.all)["pass"] == approx((1, 0, 0.0, None))


def test_two_runs_without_decisions_compare_to_zeros_and_empty_bins(session: Session) -> None:
    base = new_run(session, llm=False)
    llm = new_run(session, llm=True)
    c = compare_runs(session, pk(base), pk(llm))
    assert summary(c.all.base) == (0, 0, 0.0, None)
    assert summary(c.all.run) == (0, 0, 0.0, None)
    assert set(c.all.verdicts) == {"take", "pass", "no_call"}
    assert [(b.low, b.high, b.count) for b in c.all.calibration] == [(lo, hi, 0) for lo, hi in BINS]
    assert c.unmatched == []


def test_a_base_run_that_is_not_take_all_is_refused(session: Session, hand: Hand) -> None:
    with pytest.raises(ValueError, match="take_all"):
        compare_runs(session, pk(hand.llm), pk(hand.llm))


def test_an_llm_run_without_a_training_cutoff_is_refused(session: Session, hand: Hand) -> None:
    uncut = new_run(session, llm=True, cutoff=None)
    with pytest.raises(ValueError, match="cutoff"):
        compare_runs(session, pk(hand.base), pk(uncut))


def test_runs_of_different_strategies_are_refused(session: Session, hand: Hand) -> None:
    other = new_run(session, llm=True)
    other.strategy_id = "another_strategy"
    session.add(other)
    session.commit()
    with pytest.raises(ValueError, match="strateg"):
        compare_runs(session, pk(hand.base), pk(other))


def test_runs_over_different_windows_are_refused(session: Session, hand: Hand) -> None:
    shorter = new_run(session, llm=True)
    shorter.as_of = at(date(2024, 1, 20))
    session.add(shorter)
    session.commit()
    with pytest.raises(ValueError, match="window"):
        compare_runs(session, pk(hand.base), pk(shorter))


def test_an_unknown_run_is_a_lookup_error(session: Session, hand: Hand) -> None:
    with pytest.raises(LookupError):
        compare_runs(session, pk(hand.base), 999)
    with pytest.raises(LookupError):
        compare_runs(session, 999, pk(hand.llm))


# =============================================================================================
# real replays over the same window
# =============================================================================================


@pytest.fixture
def strategy_dir(tmp_path: Path) -> Path:
    return write_strategy(tmp_path / "strategies", {"judge": {**LLM_JUDGE, "kind": "take_all"}})


def test_a_pass_that_frees_a_held_security_shows_as_unmatched_and_is_scored_by_take_all(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    # The llm passes GAPX on D, so on X6 it judges GAPX's second gap (and passes again); take-all
    # took D and still holds GAPX on X6, so it has no X6 decision. The cutoff, set between the
    # two days, comes from the llm Run row.
    tg = two_gaps(tmp_path, session)
    root = tg.world.root
    overrides = (*LLM, "judge.training_cutoff=2026-02-20")
    cfg = config(strategy_dir, overrides)
    first = good_answer(tg.first_chunks, "pass", 0.45)
    second = good_answer(tg.second_chunks, "pass", 0.55)
    remember(session, con, root, cfg, X[0], lambda _: first)
    remember(session, con, root, cfg, X[SECOND], lambda _: second)
    base_id = run_replay(session, con, root, strategy_dir, START, END, TAKE_ALL)
    llm_id = run_replay(session, con, root, strategy_dir, START, END, overrides)
    base_trades = trades(session, base_id)
    assert len(base_trades) == 2  # fixture: the scenario's two legs
    base_pnl = sum(t.pnl for t in base_trades)
    base_r = sum(t.r_multiple for t in base_trades)

    c = compare_runs(session, base_id, llm_id)

    assert [(u.run_id, u.entity_id, u.as_of) for u in c.unmatched] == [
        (llm_id, tg.world.gapx, close_of(X[SECOND]))
    ]
    assert summary(c.all.base) == approx((1, 1, base_pnl, base_r))
    assert summary(c.all.run) == approx((2, 0, 0.0, None))
    assert verdicts(c.all) == approx(
        {"take": (0, 0, 0.0, None), "pass": (2, 1, base_pnl, base_r), "no_call": (0, 0, 0.0, None)}
    )
    win = 1.0 if base_pnl > 0 else 0.0
    assert bins(c.all)[1] == approx((0.4, 0.5, 1, 1, win))
    assert bins(c.all)[2] == approx((0.5, 0.6, 1, 0, None))

    assert c.cutoff == datetime(2026, 2, 20, tzinfo=UTC)
    assert summary(c.before_cutoff.base) == approx((1, 1, base_pnl, base_r))
    assert verdicts(c.before_cutoff)["pass"] == approx((1, 1, base_pnl, base_r))
    assert summary(c.after_cutoff.base) == approx((0, 0, 0.0, None))
    assert verdicts(c.after_cutoff)["pass"] == approx((1, 0, 0.0, None))


def test_a_take_of_the_same_candidate_matches_with_equal_outcomes(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    tg = two_gaps(tmp_path, session)
    root = tg.world.root
    cfg = config(strategy_dir, LLM)
    remember(session, con, root, cfg, X[0], lambda _: good_answer(tg.first_chunks, "take", 0.8))
    base_id = run_replay(session, con, root, strategy_dir, START, END, TAKE_ALL)
    llm_id = run_replay(session, con, root, strategy_dir, START, END, LLM)

    c = compare_runs(session, base_id, llm_id)

    assert c.unmatched == []
    assert summary(c.all.run) == approx(summary(c.all.base))
    take = c.all.verdicts["take"]
    assert (take.count, take.base_outcomes) == (1, 1)
    assert take.base_pnl == pytest.approx(c.all.run.total_pnl)
    # The fake model's cutoff (2025-01-31) is before every 2026 decision: nothing is before it.
    assert summary(c.before_cutoff.run) == (0, 0, 0.0, None)
    assert summary(c.after_cutoff.run) == approx(summary(c.all.run))


# =============================================================================================
# CLI
# =============================================================================================


def cli_args(tmp_path: Path, base: int, run: int) -> list[str]:
    db = str(tmp_path / "meta.db")
    return ["replay", "compare", "--base", str(base), "--run", str(run), "--db", db]


def test_compare_prints_verdicts_and_every_unmatched_decision(
    tmp_path: Path, session: Session, hand: Hand
) -> None:
    result = runner.invoke(app, cli_args(tmp_path, pk(hand.base), pk(hand.llm)))
    assert result.exit_code == 0, result.output
    for word in ("take", "pass", "no_call"):
        assert word in result.output
    assert "unmatched" in result.output.lower()
    unmatched_lines = [line for line in result.output.splitlines() if re.search(r"\bE[78]\b", line)]
    assert any("2024-01-24" in line for line in unmatched_lines), result.output
    assert any("2024-01-26" in line for line in unmatched_lines), result.output


def test_compare_with_a_non_take_all_base_exits_one(
    tmp_path: Path, session: Session, hand: Hand
) -> None:
    result = runner.invoke(app, cli_args(tmp_path, pk(hand.llm), pk(hand.llm)))
    assert result.exit_code == 1, result.output
    assert isinstance(result.exception, SystemExit)
    assert "Traceback" not in result.output


def test_compare_of_an_unknown_run_exits_one(tmp_path: Path, session: Session, hand: Hand) -> None:
    result = runner.invoke(app, cli_args(tmp_path, pk(hand.base), 999))
    assert result.exit_code == 1, result.output
    assert isinstance(result.exception, SystemExit)
