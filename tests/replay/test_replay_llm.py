"""Guards against an llm replay that is not a replay: one that calls the model (or the network) when
an answer is missing instead of failing, acts on an answer whose citations the validator rejects
(a chunk not public at the decision's as_of), loses the citations or conviction behind a decision,
or lets the order of same-day candidates rather than conviction decide who gets the capital.

API under test (asof.replay.run_replay with overrides ["judge.kind=llm", ...]):

    answers come only from LLMCache(<root> / "llm_cache", offline=True); a miss raises CacheMiss
    take    -> Decision(kind trade, direction long, conviction) + one DecisionCitation per claim
               (evidence_kind "chunk", chunk_id, quote, claim) + trades as with take_all
    pass    -> Decision(direction flat, conviction), citations, no trades
    no_call -> Decision(direction flat, conviction None), no citations, payload has "no_call"
               and the failures
    Run.model_id = judge.model, Run.model_training_cutoff from judge.training_cutoff
    same-day candidates enter in order_for_entry order

Each test fills the cache itself: the exact request is rebuilt with build_messages + build_body
for the candidate the scanner finds, and stored with LLMCache(..., offline=False).call and a fake
fetch. The network is guarded: OpenRouterClient.complete and httpx.Client.send fail the test.

World: tests/scan/world.py (GAPX gaps on D = 2026-02-17 after an 8-K 2.02 released 02-13 16:08
New York; ATR 2, decision close 56 -> 125 shares; provenance there) plus hand-made chunk texts.
"""

from collections.abc import Callable
from datetime import UTC, date, timedelta
from pathlib import Path
from typing import Any

import duckdb
import httpx
import pytest
from sqlmodel import Session, col, select

from asof.judge import cache, client, llm
from asof.replay import close_of, replay_metrics, run_replay
from asof.scan.earnings_gap import Candidate, scan
from asof.store.models import Chunk, Decision, DecisionCitation, Run
from asof.strategy import StrategyConfig, apply_overrides, load_strategy
from tests.judge.fakes import LLM_JUDGE, answer, claim, response
from tests.replay.scenario import END, START, TRADED, X
from tests.replay.test_replay import rows, trade_key, trades
from tests.scan.world import GAP, RELEASE, filing, gap_world
from tests.store.factories import new_chunk, new_document, pk
from tests.strategy.files import write_strategy

CacheMiss = cache.CacheMiss
LLMCache = cache.LLMCache
OpenRouterClient = client.OpenRouterClient
build_body = client.build_body
build_messages = llm.build_messages
judge_candidate = llm.judge_candidate

LLM = ("judge.kind=llm", "execute.fills.slippage_bps=0")
TAKE_ALL = ("execute.fills.slippage_bps=0",)

REVENUE = "Fourth quarter revenue was $1,234.5 million, up 18% from the prior year."
REVENUE_QUOTE = "Fourth quarter revenue was $1,234.5 million, up 18% from the prior year"
OUTLOOK = "The company now expects full year revenue between $5.1 billion and $5.3 billion."
OUTLOOK_QUOTE = "expects full year revenue between $5.1 billion and $5.3 billion"
OUTLOOK_CLAIM = "Full year revenue is expected between $5.1 billion and $5.3 billion."


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    touched: list[str] = []

    def complete(self: Any, body: Any) -> Any:
        touched.append("OpenRouterClient.complete")
        raise AssertionError("an offline replay called the model")

    def send(self: Any, *args: Any, **kwargs: Any) -> Any:
        touched.append("httpx.Client.send")
        raise AssertionError("an offline replay used the network")

    monkeypatch.setattr(OpenRouterClient, "complete", complete)
    monkeypatch.setattr(httpx.Client, "send", send)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    return touched


@pytest.fixture
def strategy_dir(tmp_path: Path) -> Path:
    # judge.yaml as committed: take_all by default, every llm field present.
    return write_strategy(tmp_path / "strategies", {"judge": {**LLM_JUDGE, "kind": "take_all"}})


def config(strategy_dir: Path, overrides: tuple[str, ...]) -> StrategyConfig:
    return apply_overrides(load_strategy(strategy_dir), list(overrides))


def add_chunks(session: Session, document_ids: list[int]) -> None:
    """The release's text: one chunk on each exhibit, as the filing-text ingest splits them."""
    first, second = document_ids
    new_chunk(session, document_id=first, text=REVENUE, section_path="EX-99.1")
    new_chunk(session, document_id=second, text=OUTLOOK, section_path="EX-99.2")


def chunk_ids(session: Session, document_ids: list[int]) -> list[int]:
    stmt = select(Chunk).where(col(Chunk.document_id).in_(document_ids)).order_by(col(Chunk.id))
    return [pk(c) for c in session.exec(stmt).all()]


def candidates_on(
    session: Session, con: duckdb.DuckDBPyConnection, root: Path, cfg: StrategyConfig, day: date
) -> list[Candidate]:
    found = scan(session, con, root, close_of(day), cfg)
    assert found, f"fixture: no candidate on {day}"
    return found


def remember(
    session: Session,
    con: duckdb.DuckDBPyConnection,
    root: Path,
    cfg: StrategyConfig,
    day: date,
    content_for: Callable[[Candidate], str],
) -> None:
    """Store, for every candidate scanned on ``day``, the answer ``content_for`` gives it, under
    the exact request the judge will send."""
    store = LLMCache(root / "llm_cache", offline=False)
    as_of = close_of(day)
    for cand in candidates_on(session, con, root, cfg, day):
        body = build_body(cfg.judge, build_messages(session, cand, as_of, cfg.judge))
        reply = response(content_for(cand))
        store.call(body, _always(reply))


def _always(reply: dict[str, Any]) -> Callable[[dict[str, Any]], dict[str, Any]]:
    return lambda _: reply


def good_answer(ids: list[int], action: str = "take", conviction: float = 0.8) -> str:
    revenue, outlook = ids
    return answer(
        action,
        conviction,
        [
            claim(revenue, REVENUE_QUOTE, REVENUE),
            claim(outlook, OUTLOOK_QUOTE, OUTLOOK_CLAIM),
        ],
    )


def citations(session: Session, decision: Decision) -> list[DecisionCitation]:
    stmt = (
        select(DecisionCitation)
        .where(DecisionCitation.decision_id == decision.id)
        .order_by(col(DecisionCitation.id))
    )
    return list(session.exec(stmt).all())


# =============================================================================================
# normal path
# =============================================================================================


def test_a_cached_take_becomes_a_cited_long_with_the_take_all_trades(
    tmp_path: Path,
    session: Session,
    con: duckdb.DuckDBPyConnection,
    strategy_dir: Path,
    no_network: list[str],
) -> None:
    world = gap_world(tmp_path, session, after=TRADED)
    add_chunks(session, world.earnings_docs)
    ids = chunk_ids(session, world.earnings_docs)
    cfg = config(strategy_dir, LLM)
    remember(session, con, world.root, cfg, X[0], lambda _: good_answer(ids))

    run_id = run_replay(session, con, world.root, strategy_dir, START, END, LLM)

    [decision] = rows(session, Decision, run_id)
    assert decision.kind == "trade"
    assert decision.direction == "long"
    assert decision.conviction == pytest.approx(0.8)
    assert decision.entity_id == world.gapx
    assert decision.as_of == close_of(X[0])
    cited = citations(session, decision)
    assert [(c.evidence_kind, c.chunk_id, c.quote, c.claim) for c in cited] == [
        ("chunk", ids[0], REVENUE_QUOTE, REVENUE),
        ("chunk", ids[1], OUTLOOK_QUOTE, OUTLOOK_CLAIM),
    ]

    run = session.get(Run, run_id)
    assert run is not None
    assert run.model_id == cfg.judge.model
    assert run.model_training_cutoff is not None
    assert run.model_training_cutoff.astimezone(UTC).date() == cfg.judge.training_cutoff

    baseline = run_replay(session, con, world.root, strategy_dir, START, END, TAKE_ALL)
    assert [trade_key(t) for t in trades(session, run_id)] == [
        trade_key(t) for t in trades(session, baseline)
    ]
    assert len(trades(session, run_id)) == 2
    assert no_network == []


def test_a_cached_pass_is_flat_with_citations_and_no_trades(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    world = gap_world(tmp_path, session, after=TRADED)
    add_chunks(session, world.earnings_docs)
    ids = chunk_ids(session, world.earnings_docs)
    cfg = config(strategy_dir, LLM)
    remember(session, con, world.root, cfg, X[0], lambda _: good_answer(ids, "pass", 0.7))

    run_id = run_replay(session, con, world.root, strategy_dir, START, END, LLM)

    [decision] = rows(session, Decision, run_id)
    assert decision.kind == "trade"
    assert decision.direction == "flat"
    assert decision.conviction == pytest.approx(0.7)
    assert len(citations(session, decision)) == 2
    assert trades(session, run_id) == []
    # A pass never opens a position: the report must not count it as still open.
    assert replay_metrics(session, run_id).open_decisions == 0


def test_a_retry_answered_from_the_cache_replays_offline(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    # The retry body depends on the judge's own feedback text, so the cache is filled by running
    # the judge once online through the cache with a scripted model: bad answer, then good.
    world = gap_world(tmp_path, session, after=TRADED)
    add_chunks(session, world.earnings_docs)
    ids = chunk_ids(session, world.earnings_docs)
    cfg = config(strategy_dir, LLM)
    store = LLMCache(world.root / "llm_cache", offline=False)
    bad = answer("take", 0.9, [claim(ids[0], "revenue was about $2 billion this year", "x")])
    replies = [response(bad), response(good_answer(ids, "take", 0.65))]
    [cand] = candidates_on(session, con, world.root, cfg, X[0])
    online = judge_candidate(
        session,
        cand,
        close_of(X[0]),
        cfg.judge,
        lambda body: store.call(body, lambda _: replies.pop(0)),
    )
    assert online.attempts == 2

    run_id = run_replay(session, con, world.root, strategy_dir, START, END, LLM)
    [decision] = rows(session, Decision, run_id)
    assert decision.direction == "long"
    assert decision.conviction == pytest.approx(0.65)
    assert len(citations(session, decision)) == 2


# =============================================================================================
# boundaries
# =============================================================================================


def test_a_rejected_answer_with_no_retries_left_is_a_flat_no_call(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    world = gap_world(tmp_path, session, after=TRADED)
    add_chunks(session, world.earnings_docs)
    ids = chunk_ids(session, world.earnings_docs)
    overrides = (*LLM, "judge.max_retries=0")
    cfg = config(strategy_dir, overrides)
    bad = answer("take", 0.9, [claim(ids[0], "revenue was about $2 billion this year", "x")])
    remember(session, con, world.root, cfg, X[0], lambda _: bad)

    run_id = run_replay(session, con, world.root, strategy_dir, START, END, overrides)

    [decision] = rows(session, Decision, run_id)
    assert decision.direction == "flat"
    assert decision.conviction is None
    assert citations(session, decision) == []
    text = repr(decision.payload)
    assert "no_call" in text
    assert "quote_not_in_chunk" in text
    assert trades(session, run_id) == []


@pytest.mark.parametrize(
    ("gapx_conviction", "othr_conviction", "winner"),
    [(0.6, 0.9, "S_OTHR"), (0.9, 0.6, "S_GAPX"), (0.8, 0.8, "S_GAPX")],
    ids=["othr_higher", "gapx_higher", "tie_by_security_id"],
)
def test_same_day_candidates_enter_by_conviction(
    tmp_path: Path,
    session: Session,
    con: duckdb.DuckDBPyConnection,
    strategy_dir: Path,
    gapx_conviction: float,
    othr_conviction: float,
    winner: str,
) -> None:
    # GAPX and OTHR gap on D on the same path; each wants 125 shares * 56 = 7,000, and a 7% gross
    # cap on 100,000 leaves room for one. Take-all would always pick S_GAPX (security id order).
    othr_after = {X[i + 1]: bar for i, bar in enumerate(TRADED) if bar is not None}
    world = gap_world(tmp_path, session, after=TRADED, othr_bars={X[0]: GAP, **othr_after})
    othr_docs = filing(
        session,
        source_id=world.source_id,
        entity_id=world.othr,
        accession="0000002222-26-000009",
        available_at=RELEASE,
        exhibits=[
            ("ex991.htm", "8-K 2.02,9.01 EX-99.1 PRESS RELEASE"),
            ("ex992.htm", "8-K 2.02,9.01 EX-99.2 CFO COMMENTARY"),
        ],
    )
    add_chunks(session, world.earnings_docs)
    add_chunks(session, othr_docs)
    gapx_ids = chunk_ids(session, world.earnings_docs)
    othr_ids = chunk_ids(session, othr_docs)
    overrides = (*LLM, "execute.sizing.max_gross_pct=7")
    cfg = config(strategy_dir, overrides)

    def content_for(cand: Candidate) -> str:
        if cand.security_id == "S_GAPX":
            return good_answer(gapx_ids, "take", gapx_conviction)
        return good_answer(othr_ids, "take", othr_conviction)

    remember(session, con, world.root, cfg, X[0], content_for)
    run_id = run_replay(session, con, world.root, strategy_dir, START, END, overrides)

    assert len(rows(session, Decision, run_id)) == 2
    assert {t.security_id for t in trades(session, run_id)} == {winner}


def test_take_all_ignores_the_cache_and_the_llm_fields(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    world = gap_world(tmp_path, session, after=TRADED)
    add_chunks(session, world.earnings_docs)
    run_id = run_replay(session, con, world.root, strategy_dir, START, END, TAKE_ALL)
    [decision] = rows(session, Decision, run_id)
    assert decision.direction == "long"
    assert decision.conviction is None
    assert citations(session, decision) == []
    assert len(trades(session, run_id)) == 2
    cache_dir = world.root / "llm_cache"
    assert not cache_dir.exists() or not any(cache_dir.rglob("*.json"))


# =============================================================================================
# leakage
# =============================================================================================


def test_an_empty_cache_raises_and_never_calls_out(
    tmp_path: Path,
    session: Session,
    con: duckdb.DuckDBPyConnection,
    strategy_dir: Path,
    no_network: list[str],
) -> None:
    world = gap_world(tmp_path, session, after=TRADED)
    add_chunks(session, world.earnings_docs)
    with pytest.raises(CacheMiss):
        run_replay(session, con, world.root, strategy_dir, START, END, LLM)
    assert no_network == []


def test_an_answer_citing_a_chunk_published_later_is_not_acted_on(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    # A GAPX 10-Q published a week after D holds a sentence the model "knew"; the cached answer
    # quotes it verbatim. At D 20:00 that chunk is not public, so the decision is a no_call and
    # no citation row points at it.
    world = gap_world(tmp_path, session, after=TRADED)
    add_chunks(session, world.earnings_docs)
    later = new_document(
        session,
        source_id=world.source_id,
        entity_id=world.gapx,
        available_at=(close_of(X[0]) + timedelta(days=7)).astimezone(UTC),
        doc_type="10-Q",
    )
    text = "First quarter revenue was $1,500.0 million, a record for the company."
    future = new_chunk(session, document_id=pk(later), text=text, section_path="Item 2")
    overrides = (*LLM, "judge.max_retries=0")
    cfg = config(strategy_dir, overrides)
    leaked = answer(
        "take",
        0.95,
        [claim(pk(future), text.rstrip("."), "First quarter revenue was $1,500.0 million.")],
    )
    remember(session, con, world.root, cfg, X[0], lambda _: leaked)

    run_id = run_replay(session, con, world.root, strategy_dir, START, END, overrides)

    [decision] = rows(session, Decision, run_id)
    assert decision.direction == "flat"
    assert decision.conviction is None
    assert "not_visible" in repr(decision.payload)
    every = list(session.exec(select(DecisionCitation)).all())
    assert all(c.chunk_id != pk(future) for c in every)
    assert trades(session, run_id) == []


def test_the_request_never_holds_a_chunk_published_after_the_decision(
    tmp_path: Path, session: Session, con: duckdb.DuckDBPyConnection, strategy_dir: Path
) -> None:
    # The cache is keyed by the request: if the judge put tomorrow's text in it, the request
    # rebuilt here from visible chunks would differ and the replay would miss.
    world = gap_world(tmp_path, session, after=TRADED)
    add_chunks(session, world.earnings_docs)
    ids = chunk_ids(session, world.earnings_docs)
    cfg = config(strategy_dir, LLM)
    [cand] = candidates_on(session, con, world.root, cfg, X[0])
    as_of = close_of(X[0])
    revised_at = (as_of + timedelta(hours=1)).astimezone(UTC)
    for doc_id in world.earnings_docs:
        revision = new_document(
            session,
            source_id=world.source_id,
            entity_id=world.gapx,
            available_at=revised_at,
            doc_type="8-K/A",
            supersedes_id=doc_id,
        )
        new_chunk(session, document_id=pk(revision), text="Restated: revenue was $1,100.0 million.")
    content = build_messages(session, cand, as_of, cfg.judge)[1]["content"]
    assert "Restated" not in content
    assert REVENUE in content
    remember(session, con, world.root, cfg, X[0], lambda _: good_answer(ids))
    run_id = run_replay(session, con, world.root, strategy_dir, START, END, LLM)
    [decision] = rows(session, Decision, run_id)
    assert decision.direction == "long"
