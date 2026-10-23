"""Guards against filling the answer cache in a way the offline replay cannot use or should not
trust: a window that leaves a candidate the replay will ask about unjudged (CacheMiss later), a
request that holds a chunk published after its day's close, a day after ``end`` judged, money spent
past the cap or spent without being counted, and a re-run that pays again for cached answers.

API under test (asof.judge.window):

    @dataclass WindowResult: outcomes: list[JudgeOutcome]; fresh_calls: int; cached_calls: int;
        spent: float; stopped: bool
    judge_window(session, con, root, cfg, start, end, fetch, *, max_cost, limit=None)
        -> WindowResult

    The calendar is visible_sessions(con, root, "1Day", close_of(end)); for each session d in
    [start, end] it runs scan(session, con, root, close_of(d), cfg, sessions=...) as the replay
    does and judges every candidate, one per security, in security_id order, with
    judge_candidate(..., call) where call(body) = LLMCache(root / "llm_cache",
    offline=False).call(body, counted_fetch). It does not know the portfolio, so it also judges
    candidates the replay will skip because the security is held.
    Cost: usage.cost of every fresh reply is added to spent; a cached reply adds nothing and
    counts in cached_calls; a fresh reply without usage.cost is stored, then ValueError.
    Cap: before each candidate, spent >= max_cost stops the window (stopped=True); a candidate
    already started finishes its retries. limit: at most that many candidates, day then
    security_id order. cfg.judge.kind must be "llm" (ValueError otherwise).

World: tests/replay/two_gaps.py (GAPX on D and again on X[6], OTHR on D when asked). The model is
``tests.judge.fakes.Script``; costs are made up, their shape is measured (see fakes.py). The
network is guarded as in tests/replay/test_replay_llm.py.
"""

from datetime import UTC, date, timedelta
from pathlib import Path
from typing import Any

import duckdb
import httpx
import pytest
from sqlmodel import Session, col, select

from asof.judge import client, window
from asof.judge.cache import LLMCache, request_key
from asof.replay import close_of, run_replay
from asof.store.models import Decision
from tests.judge.fakes import LLM_JUDGE, Script, answer, claim, costed, response
from tests.replay.scenario import END, START, X
from tests.replay.test_replay_llm import (
    LLM,
    REVENUE,
    TAKE_ALL,
    candidates_on,
    config,
    good_answer,
)
from tests.replay.two_gaps import SECOND, two_gaps
from tests.store.factories import new_chunk, new_document, pk
from tests.strategy.files import write_strategy

judge_window = window.judge_window

D = X[0]
X6 = X[SECOND]


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    touched: list[str] = []

    def complete(self: Any, body: Any) -> Any:
        touched.append("OpenRouterClient.complete")
        raise AssertionError("judge_window called the real client instead of fetch")

    def send(self: Any, *args: Any, **kwargs: Any) -> Any:
        touched.append("httpx.Client.send")
        raise AssertionError("judge_window used the network")

    monkeypatch.setattr(client.OpenRouterClient, "complete", complete)
    monkeypatch.setattr(httpx.Client, "send", send)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    return touched


@pytest.fixture
def strategy_dir(tmp_path: Path) -> Path:
    return write_strategy(tmp_path / "strategies", {"judge": {**LLM_JUDGE, "kind": "take_all"}})


def bad_answer(ids: list[int]) -> str:
    # quote_not_in_chunk: the validator sends it back for a retry.
    return answer("take", 0.9, [claim(ids[0], "revenue was about $2 billion this year", "x")])


def where(result: window.WindowResult) -> list[tuple[date, str]]:
    return [(o.candidate.reaction_session, o.candidate.security_id) for o in result.outcomes]


def cache_files(root: Path) -> list[Path]:
    return sorted((root / "llm_cache").rglob("*.json"))


def llm_decisions(session: Session, run_id: int) -> list[Decision]:
    stmt = select(Decision).where(Decision.run_id == run_id).order_by(col(Decision.id))
    return list(session.exec(stmt).all())


# =============================================================================================
# normal path
# =============================================================================================


@pytest.mark.parametrize(
    ("first_action", "replay_judges"),
    [("take", [D]), ("pass", [D, X6])],
    ids=["take_holds_gapx", "pass_frees_gapx"],
)
def test_the_window_covers_every_request_the_offline_replay_makes(
    tmp_path: Path,
    session: Session,
    con: duckdb.DuckDBPyConnection,
    strategy_dir: Path,
    no_network: list[str],
    first_action: str,
    replay_judges: list[date],
) -> None:
    # The window judges GAPX on X6 even though a replay that took D will skip it (held), so one
    # cache serves whichever verdict the replay ends up acting on.
    tg = two_gaps(tmp_path, session)
    root = tg.world.root
    cfg = config(strategy_dir, LLM)
    fetch = Script(
        costed(good_answer(tg.first_chunks, first_action, 0.8), 0.04852),
        costed(good_answer(tg.second_chunks, "take", 0.6), 0.03),
    )

    result = judge_window(session, con, root, cfg, START, END, fetch, max_cost=5.0)

    assert where(result) == [(D, "S_GAPX"), (X6, "S_GAPX")]
    assert [o.verdict for o in result.outcomes] == [first_action, "take"]
    assert result.fresh_calls == 2
    assert result.cached_calls == 0
    assert result.spent == pytest.approx(0.04852 + 0.03)
    assert result.stopped is False
    assert len(cache_files(root)) == 2

    run_id = run_replay(session, con, root, strategy_dir, START, END, LLM)  # no CacheMiss

    decisions = llm_decisions(session, run_id)
    assert [d.as_of for d in decisions] == [close_of(day) for day in replay_judges]
    by_as_of = {close_of(o.candidate.reaction_session): o for o in result.outcomes}
    for decision in decisions:
        outcome = by_as_of[decision.as_of]
        assert decision.entity_id == outcome.candidate.entity_id
        assert decision.payload["verdict"] == outcome.verdict
        assert decision.direction == ("long" if outcome.verdict == "take" else "flat")
        assert decision.conviction == pytest.approx(outcome.conviction)
        assert decision.payload["request_keys"] == outcome.request_keys
    assert no_network == []


def test_same_day_candidates_are_judged_in_security_order_and_retries_are_counted(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    tg = two_gaps(tmp_path, session, othr_on_d=True)
    cfg = config(strategy_dir, LLM)
    fetch = Script(
        costed(bad_answer(tg.first_chunks), 0.01),
        costed(good_answer(tg.first_chunks, "take", 0.7), 0.02),
        costed(good_answer(tg.othr_chunks, "pass", 0.4), 0.03),
    )

    result = judge_window(session, con, tg.world.root, cfg, START, D, fetch, max_cost=5.0)

    assert where(result) == [(D, "S_GAPX"), (D, "S_OTHR")]
    assert [o.verdict for o in result.outcomes] == ["take", "pass"]
    assert [o.attempts for o in result.outcomes] == [2, 1]
    assert result.fresh_calls == 3
    assert result.spent == pytest.approx(0.06)
    # The request of each call is the one the judge built for that candidate and attempt.
    assert [request_key(b) for b in fetch.bodies] == [
        *result.outcomes[0].request_keys,
        *result.outcomes[1].request_keys,
    ]


def test_a_rerun_over_a_filled_cache_pays_nothing_and_gives_the_same_outcomes(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    tg = two_gaps(tmp_path, session, othr_on_d=True)
    root = tg.world.root
    cfg = config(strategy_dir, LLM)
    first = judge_window(
        session,
        con,
        root,
        cfg,
        START,
        END,
        Script(
            costed(bad_answer(tg.first_chunks), 0.01),
            costed(good_answer(tg.first_chunks, "take", 0.7), 0.02),
            costed(good_answer(tg.othr_chunks, "pass", 0.4), 0.03),
            costed(good_answer(tg.second_chunks, "pass", 0.55), 0.04),
        ),
        max_cost=5.0,
    )
    assert first.fresh_calls == 4

    again = judge_window(session, con, root, cfg, START, END, Script(), max_cost=5.0)

    assert again.fresh_calls == 0
    assert again.cached_calls == 4
    assert again.spent == 0  # the cached replies carry a cost; it was paid by the first run
    assert again.stopped is False
    assert [(o.candidate, o.verdict, o.conviction, o.request_keys) for o in again.outcomes] == [
        (o.candidate, o.verdict, o.conviction, o.request_keys) for o in first.outcomes
    ]


# =============================================================================================
# boundaries
# =============================================================================================


def test_a_candidate_in_progress_finishes_its_retries_past_the_cap_then_the_window_stops(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    tg = two_gaps(tmp_path, session, othr_on_d=True)
    cfg = config(strategy_dir, LLM)
    # Script fails the test on a third call: OTHR must not be asked.
    fetch = Script(
        costed(bad_answer(tg.first_chunks), 0.04),
        costed(good_answer(tg.first_chunks, "take", 0.7), 0.04),
    )

    result = judge_window(session, con, tg.world.root, cfg, START, END, fetch, max_cost=0.05)

    assert len(fetch.bodies) == 2
    assert where(result) == [(D, "S_GAPX")]
    assert result.outcomes[0].attempts == 2
    assert result.spent == pytest.approx(0.08)
    assert result.fresh_calls == 2
    assert result.stopped is True


@pytest.mark.parametrize(
    ("max_cost", "judged", "stopped"),
    [(0.04, 1, True), (0.041, 2, False)],
    ids=["spent_equals_cap_stops", "just_under_cap_goes_on"],
)
def test_the_cap_is_checked_with_greater_or_equal_before_each_candidate(
    tmp_path: Path,
    session: Session,
    con: duckdb.DuckDBPyConnection,
    strategy_dir: Path,
    max_cost: float,
    judged: int,
    stopped: bool,
) -> None:
    # Two candidates in the window (both on D); "stopped" means one was left unjudged.
    tg = two_gaps(tmp_path, session, othr_on_d=True)
    cfg = config(strategy_dir, LLM)
    replies = [
        costed(good_answer(tg.first_chunks, "take", 0.7), 0.04),
        costed(good_answer(tg.othr_chunks, "take", 0.7), 0.04),
    ]
    fetch = Script(*replies[:judged])

    result = judge_window(session, con, tg.world.root, cfg, START, D, fetch, max_cost=max_cost)

    assert len(result.outcomes) == judged
    assert result.stopped is stopped
    assert result.spent == pytest.approx(0.04 * judged)


def test_a_stopped_window_resumes_from_the_cache_and_pays_only_for_the_rest(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    tg = two_gaps(tmp_path, session, othr_on_d=True)
    root = tg.world.root
    cfg = config(strategy_dir, LLM)
    stopped = judge_window(
        session,
        con,
        root,
        cfg,
        START,
        D,
        Script(costed(good_answer(tg.first_chunks, "take", 0.7), 0.04)),
        max_cost=0.04,
    )
    assert stopped.stopped is True

    resumed = judge_window(
        session,
        con,
        root,
        cfg,
        START,
        D,
        Script(costed(good_answer(tg.othr_chunks, "pass", 0.3), 0.02)),
        max_cost=1.0,
    )

    assert where(resumed) == [(D, "S_GAPX"), (D, "S_OTHR")]
    assert resumed.cached_calls == 1
    assert resumed.fresh_calls == 1
    assert resumed.spent == pytest.approx(0.02)
    assert resumed.stopped is False


@pytest.mark.parametrize(
    ("limit", "expected"),
    [(1, [(D, "S_GAPX")]), (2, [(D, "S_GAPX"), (D, "S_OTHR")]), (3, None)],
    ids=["one", "two_same_day_first", "all"],
)
def test_limit_takes_candidates_in_day_then_security_order(
    tmp_path: Path,
    session: Session,
    con: duckdb.DuckDBPyConnection,
    strategy_dir: Path,
    limit: int,
    expected: list[tuple[date, str]] | None,
) -> None:
    tg = two_gaps(tmp_path, session, othr_on_d=True)
    cfg = config(strategy_dir, LLM)
    replies = [
        costed(good_answer(tg.first_chunks, "take", 0.7), 0.01),
        costed(good_answer(tg.othr_chunks, "take", 0.7), 0.01),
        costed(good_answer(tg.second_chunks, "take", 0.7), 0.01),
    ]
    every = [(D, "S_GAPX"), (D, "S_OTHR"), (X6, "S_GAPX")]
    want = expected if expected is not None else every

    fetch = Script(*replies[: len(want)])

    result = judge_window(
        session, con, tg.world.root, cfg, START, END, fetch, max_cost=5.0, limit=limit
    )

    assert where(result) == want
    assert result.stopped is False  # a limit is not the cap


def test_a_window_starting_after_the_first_gap_does_not_judge_it(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    tg = two_gaps(tmp_path, session)
    cfg = config(strategy_dir, LLM)
    fetch = Script(costed(good_answer(tg.second_chunks, "take", 0.7), 0.01))

    result = judge_window(session, con, tg.world.root, cfg, X[1], END, fetch, max_cost=5.0)

    assert where(result) == [(X6, "S_GAPX")]


def test_take_all_config_is_refused_before_any_call(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    tg = two_gaps(tmp_path, session)
    cfg = config(strategy_dir, TAKE_ALL)
    fetch = Script()
    with pytest.raises(ValueError, match="llm"):
        judge_window(session, con, tg.world.root, cfg, START, END, fetch, max_cost=5.0)
    assert fetch.bodies == []
    assert cache_files(tg.world.root) == []


def test_a_fresh_reply_without_a_cost_is_kept_but_stops_the_window(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    # Without usage.cost the cap cannot be enforced. The reply was paid for, so it is cached;
    # the error says what was spent before it, so the spend is not reported as zero.
    tg = two_gaps(tmp_path, session, othr_on_d=True)
    root = tg.world.root
    cfg = config(strategy_dir, LLM)
    fetch = Script(
        costed(good_answer(tg.first_chunks, "take", 0.7), 0.03),
        response(good_answer(tg.othr_chunks, "take", 0.7)),  # usage without cost
    )

    with pytest.raises(ValueError, match=r"cost") as raised:
        judge_window(session, con, root, cfg, START, D, fetch, max_cost=5.0)

    assert len(fetch.bodies) == 2
    assert "0.03" in str(raised.value)
    store = LLMCache(root / "llm_cache", offline=True)
    assert store.get(fetch.bodies[1]) is not None


def test_a_failed_call_ends_the_window_and_keeps_the_spend_so_far(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    # A timeout or provider error on the second candidate: what was paid for stays cached and
    # counted, the error is reported, nothing after it is asked.
    tg = two_gaps(tmp_path, session, othr_on_d=True)
    cfg = config(strategy_dir, LLM)
    calls: list[dict[str, Any]] = []

    def fetch(body: dict[str, Any]) -> dict[str, Any]:
        calls.append(body)
        if len(calls) == 1:
            return costed(good_answer(tg.first_chunks, "take", 0.7), 0.03)
        raise client.JudgeCallError("OpenRouter HTTP 502: bad gateway")

    result = judge_window(session, con, tg.world.root, cfg, START, END, fetch, max_cost=5.0)

    assert len(calls) == 2
    assert where(result) == [(D, "S_GAPX")]
    assert result.error is not None and "502" in result.error
    assert result.spent == pytest.approx(0.03)
    assert result.fresh_calls == 1
    assert len(cache_files(tg.world.root)) == 1


# =============================================================================================
# leakage
# =============================================================================================


def test_no_request_holds_a_chunk_published_after_its_day_but_before_the_window_end(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    # Each release is restated one hour after its reaction day's close, well before END: a window
    # that built every request at close_of(END) would put the restatements in.
    tg = two_gaps(tmp_path, session)
    world = tg.world
    for day, docs in ((D, world.earnings_docs), (X6, tg.second_docs)):
        revised_at = (close_of(day) + timedelta(hours=1)).astimezone(UTC)
        assert revised_at < close_of(END)
        for doc_id in docs:
            revision = new_document(
                session,
                source_id=world.source_id,
                entity_id=world.gapx,
                available_at=revised_at,
                doc_type="8-K/A",
                supersedes_id=doc_id,
            )
            new_chunk(
                session,
                document_id=pk(revision),
                text=f"Restated on {day.isoformat()}: revenue was $1,100.0 million.",
            )
    cfg = config(strategy_dir, LLM)
    fetch = Script(
        costed(good_answer(tg.first_chunks, "take", 0.7), 0.01),
        costed(good_answer(tg.second_chunks, "take", 0.7), 0.01),
    )

    result = judge_window(session, con, world.root, cfg, START, END, fetch, max_cost=5.0)

    assert where(result) == [(D, "S_GAPX"), (X6, "S_GAPX")]
    sent = [str(b["messages"]) for b in fetch.bodies]
    assert all("Restated" not in text for text in sent)
    assert REVENUE in sent[0]
    # Each request is built at its own day's close, not at the window's end.
    assert f"As of: {close_of(D).isoformat()}" in sent[0]
    assert f"As of: {close_of(X6).isoformat()}" in sent[1]


def test_a_candidate_the_day_after_the_window_end_is_not_judged(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    # The second release is public at X5 16:05 and GAPX gaps on X6; the X6 bar is in the store
    # but not final at close_of(X5), so a window ending on X5 sees D only.
    tg = two_gaps(tmp_path, session)
    cfg = config(strategy_dir, LLM)
    assert [c.security_id for c in candidates_on(session, con, tg.world.root, cfg, X6)] == [
        "S_GAPX"
    ]
    fetch = Script(costed(good_answer(tg.first_chunks, "take", 0.7), 0.01))

    result = judge_window(
        session, con, tg.world.root, cfg, START, X[SECOND - 1], fetch, max_cost=5.0
    )

    assert where(result) == [(D, "S_GAPX")]
    assert len(fetch.bodies) == 1
