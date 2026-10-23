"""Guards against the llm judge seeing or citing what was not public at as_of (a revision published
later, a filing of tomorrow, another company's text), against an answer whose citations fail the
validator being acted on, and against retries, refusals and conviction thresholds changing the
verdict in ways the config does not say.

API under test (asof.judge.llm):

    Claim(claim, chunk_id, quote, stance: "for" | "against")
    JudgeAnswer(action: "take" | "pass", conviction in [0, 1], claims: 1..8 Claim)
    build_messages(session, candidate, as_of, cfg) -> [system prompt, user content]
        user content: as-of time, symbol and numbers, every chunk of every document in
        candidate.document_ids read through visible_chunks(session, as_of, document_id=...),
        each wrapped as <chunk id="{id}" section="{section_path}"> ... </chunk>
    JudgeOutcome(candidate, verdict: "take" | "pass" | "no_call", conviction, claims, attempts,
                 failures, request_keys)
    judge_candidate(session, candidate, as_of, cfg, call) -> JudgeOutcome
    order_for_entry(outcomes) -> take outcomes by conviction desc, then security_id asc

``cfg`` is the JudgeConfig (strategy section ``judge``). ``call`` is a fake (tests/judge/fakes.py);
nothing reaches the network.

World (hand-made, shaped like the scan tests' GAPX): GAPX's 8-K 2.02 accepted Friday 2026-02-13
16:08 New York, judged at Tuesday 2026-02-17 20:00 New York (the reaction day's close). Chunk
texts are made up press-release sentences; each claim's quote was checked against the current
validator before these tests were written.
"""

import json
import re
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from sqlmodel import Session

from asof.judge import cache, client, llm
from asof.scan.earnings_gap import Candidate
from asof.store.models import Chunk, Source
from asof.store.queries import visible_chunks
from tests.judge.fakes import Script, answer, claim, judge_config, response
from tests.store.factories import new_chunk, new_document, new_entity, pk

request_key = cache.request_key
build_body = client.build_body
Claim = llm.Claim
JudgeAnswer = llm.JudgeAnswer
JudgeOutcome = llm.JudgeOutcome
build_messages = llm.build_messages
judge_candidate = llm.judge_candidate
order_for_entry = llm.order_for_entry

ET = ZoneInfo("America/New_York")
RELEASE = datetime(2026, 2, 13, 16, 8, tzinfo=ET)
AS_OF = datetime(2026, 2, 17, 20, 0, tzinfo=ET)
SECTION = "EX-99.1"

REVENUE = (
    "Fourth quarter revenue was $1,234.5 million, up 18% from the prior year. "
    "Gross margin was 61.2%."
)
OUTLOOK = "The company now expects full year revenue between $5.1 billion and $5.3 billion."
COSTS = "Operating expenses rose 25% as the company expanded its sales force."
# The same release revised after as_of: one figure changed.
REVISED = (
    "Fourth quarter revenue was $1,300.0 million, up 24% from the prior year. "
    "Gross margin was 61.2%."
)
FUTURE = "First quarter revenue was $1,500.0 million, a record for the company."
OTHER = "OTHR Inc. reported quarterly revenue of $9,999.0 million."

REVENUE_QUOTE = "Fourth quarter revenue was $1,234.5 million, up 18% from the prior year"
REVENUE_CLAIM = "Fourth quarter revenue was $1,234.5 million, up 18% from the prior year."
OUTLOOK_QUOTE = "expects full year revenue between $5.1 billion and $5.3 billion"
OUTLOOK_CLAIM = "Full year revenue is expected between $5.1 billion and $5.3 billion."
COSTS_QUOTE = "Operating expenses rose 25% as the company expanded its sales force"
COSTS_CLAIM = "Operating expenses rose 25%."
FUTURE_QUOTE = "First quarter revenue was $1,500.0 million, a record for the company"
FUTURE_CLAIM = "First quarter revenue was $1,500.0 million."


@dataclass
class World:
    gapx: int
    othr: int
    release: int  # GAPX's release, visible at AS_OF (later revised)
    revenue: Chunk
    outlook: Chunk
    costs: Chunk
    revision: int  # the release's revision, published after AS_OF
    revised: Chunk
    future_doc: int  # a GAPX filing published after AS_OF
    future: Chunk
    other_doc: int  # OTHR's filing, visible at AS_OF
    other: Chunk


def build_world(session: Session, source: Source) -> World:
    src = pk(source)
    gapx = pk(new_entity(session, cik=1111, ticker="GAPX", name="GAPX Inc."))
    othr = pk(new_entity(session, cik=2222, ticker="OTHR", name="OTHR Inc."))
    release = new_document(
        session, source_id=src, entity_id=gapx, available_at=RELEASE, doc_type="8-K"
    )
    chunks = [
        new_chunk(session, document_id=pk(release), ordinal=i, text=t, section_path=SECTION)
        for i, t in enumerate((REVENUE, OUTLOOK, COSTS))
    ]
    revision = new_document(
        session,
        source_id=src,
        entity_id=gapx,
        available_at=AS_OF + timedelta(hours=1),
        doc_type="8-K/A",
        supersedes_id=pk(release),
    )
    revised = new_chunk(session, document_id=pk(revision), text=REVISED, section_path=SECTION)
    future_doc = new_document(
        session, source_id=src, entity_id=gapx, available_at=AS_OF + timedelta(days=60)
    )
    future = new_chunk(session, document_id=pk(future_doc), text=FUTURE, section_path=SECTION)
    other_doc = new_document(
        session, source_id=src, entity_id=othr, available_at=RELEASE, doc_type="8-K"
    )
    other = new_chunk(session, document_id=pk(other_doc), text=OTHER, section_path=SECTION)
    return World(
        gapx=gapx,
        othr=othr,
        release=pk(release),
        revenue=chunks[0],
        outlook=chunks[1],
        costs=chunks[2],
        revision=pk(revision),
        revised=revised,
        future_doc=pk(future_doc),
        future=future,
        other_doc=pk(other_doc),
        other=other,
    )


@pytest.fixture
def world(session: Session, source: Source) -> World:
    return build_world(session, source)


def candidate(
    entity_id: int,
    document_ids: tuple[int, ...],
    *,
    security_id: str = "S_GAPX",
    symbol: str = "GAPX",
) -> Candidate:
    return Candidate(
        entity_id=entity_id,
        security_id=security_id,
        symbol=symbol,
        document_ids=document_ids,
        available_at=RELEASE.astimezone(UTC),
        reaction_session=date(2026, 2, 17),
        pre_earnings_close=50.5,
        gap_open=55.0,
        gap_high=57.0,
        gap_low=54.0,
        gap_close=56.25,
        gap=0.0891,
        rvol=3.0,
        atr=2.0,
        avg_dollar_volume=50_000_000.0,
    )


def gapx(world: World, *extra: int) -> Candidate:
    return candidate(world.gapx, (world.release, *extra))


def good_claims(world: World) -> list[dict[str, Any]]:
    return [
        claim(pk(world.revenue), REVENUE_QUOTE, REVENUE_CLAIM),
        claim(pk(world.outlook), OUTLOOK_QUOTE, OUTLOOK_CLAIM),
        claim(pk(world.costs), COSTS_QUOTE, COSTS_CLAIM, stance="against"),
    ]


def good(world: World, action: str = "take", conviction: float = 0.8) -> dict[str, Any]:
    return response(answer(action, conviction, good_claims(world)))


def bad_quote(world: World) -> dict[str, Any]:
    """Claim 1 quotes a sentence the chunk does not hold: quote_not_in_chunk."""
    claims = good_claims(world)
    claims[1] = claim(pk(world.outlook), "expects revenue of about $6 billion next year", "x")
    return response(answer("take", 0.8, claims))


def user_content(messages: list[dict[str, Any]]) -> str:
    [system, user] = messages
    assert system["role"] == "system"
    assert user["role"] == "user"
    return user["content"]


def judge(session: Session, cand: Candidate, script: Script, **cfg: Any) -> JudgeOutcome:
    return judge_candidate(session, cand, AS_OF, judge_config(**cfg), script)


# =============================================================================================
# answer schema
# =============================================================================================


def test_answer_schema_accepts_a_valid_answer(world: World) -> None:
    parsed = JudgeAnswer.model_validate_json(answer("take", 0.8, good_claims(world)))
    assert parsed.action == "take"
    assert parsed.conviction == 0.8
    assert parsed.claims[2] == Claim(
        claim=COSTS_CLAIM, chunk_id=pk(world.costs), quote=COSTS_QUOTE, stance="against"
    )


@pytest.mark.parametrize(
    "change",
    [
        {"conviction": 1.3},
        {"conviction": -0.01},
        {"claims": []},
        {"action": "short"},
    ],
)
def test_answer_schema_rejects_out_of_range(world: World, change: dict[str, Any]) -> None:
    raw = {"action": "take", "conviction": 0.8, "claims": good_claims(world), **change}
    with pytest.raises(ValueError):
        JudgeAnswer.model_validate(raw)


def test_answer_schema_claim_count_bounds(world: World) -> None:
    one = good_claims(world)[0]
    eight = JudgeAnswer.model_validate({"action": "pass", "conviction": 0.0, "claims": [one] * 8})
    assert len(eight.claims) == 8
    with pytest.raises(ValueError):
        JudgeAnswer.model_validate({"action": "pass", "conviction": 0.0, "claims": [one] * 9})


def test_answer_schema_rejects_an_unknown_stance(world: World) -> None:
    raw = good_claims(world)[0] | {"stance": "neutral"}
    with pytest.raises(ValueError):
        Claim.model_validate(raw)


# =============================================================================================
# build_messages: normal path
# =============================================================================================


def test_messages_are_the_prompt_and_the_candidate_with_its_chunks(
    session: Session, world: World
) -> None:
    cfg = judge_config()
    cand = gapx(world)
    messages = build_messages(session, cand, AS_OF, cfg)
    assert messages[0] == {"role": "system", "content": cfg.prompt}
    content = user_content(messages)
    assert "2026-02-17" in content
    assert "GAPX" in content
    assert "56.25" in content  # gap close
    assert "50.5" in content  # pre-earnings close
    for chunk in (world.revenue, world.outlook, world.costs):
        assert f'<chunk id="{pk(chunk)}" section="{SECTION}">' in content
        assert chunk.text in content
    assert content.count("<chunk ") == 3
    assert content.count("</chunk>") == 3


def test_the_prompt_names_the_ticker_not_todays_company_name(
    session: Session, world: World
) -> None:
    # Entity.name is today's name (SEC ticker table, watchlist). A company renamed after as_of
    # (a merger, a rebrand) would put that later event in the prompt; the filing text carries
    # the name it was filed under, and the ticker is the one of the reaction session.
    content = user_content(build_messages(session, gapx(world), AS_OF, judge_config()))
    assert "GAPX Inc." not in content
    assert "GAPX" in content


def test_chunks_come_in_document_and_ordinal_order(session: Session, world: World) -> None:
    content = user_content(build_messages(session, gapx(world), AS_OF, judge_config()))
    positions = [content.index(c.text) for c in (world.revenue, world.outlook, world.costs)]
    assert positions == sorted(positions)


def test_every_document_of_the_candidate_is_read(
    session: Session, source: Source, world: World
) -> None:
    # An 8-K with two exhibits is two documents (tests/scan/world.py); both are evidence.
    ex992 = new_document(
        session, source_id=pk(source), entity_id=world.gapx, available_at=RELEASE, doc_type="8-K"
    )
    commentary = new_chunk(
        session,
        document_id=pk(ex992),
        text="CFO commentary: free cash flow was $210.0 million.",
        section_path="EX-99.2",
    )
    content = user_content(build_messages(session, gapx(world, pk(ex992)), AS_OF, judge_config()))
    assert f'<chunk id="{pk(commentary)}" section="EX-99.2">' in content
    assert commentary.text in content
    assert content.count("<chunk ") == 4


def test_messages_are_deterministic(session: Session, world: World) -> None:
    cfg = judge_config()
    a = build_messages(session, gapx(world), AS_OF, cfg)
    b = build_messages(session, gapx(world), AS_OF, cfg)
    assert a == b
    assert request_key(build_body(cfg, a)) == request_key(build_body(cfg, b))


# =============================================================================================
# build_messages: boundaries
# =============================================================================================


def test_a_document_published_exactly_at_as_of_is_read(
    session: Session, source: Source, world: World
) -> None:
    on_time = new_document(
        session, source_id=pk(source), entity_id=world.gapx, available_at=AS_OF, doc_type="8-K"
    )
    chunk = new_chunk(session, document_id=pk(on_time), text="Published at the bell.")
    content = user_content(build_messages(session, gapx(world, pk(on_time)), AS_OF, judge_config()))
    assert chunk.text in content


def test_naive_as_of_is_refused(session: Session, world: World) -> None:
    with pytest.raises(ValueError, match="timezone"):
        build_messages(session, gapx(world), AS_OF.replace(tzinfo=None), judge_config())


# =============================================================================================
# build_messages: leakage
# =============================================================================================


def test_a_later_revision_does_not_reach_the_prompt(session: Session, world: World) -> None:
    # The revision exists in the store (published one hour after AS_OF) and even its id is
    # passed: the prompt holds the original text, never the revised one.
    content = user_content(
        build_messages(session, gapx(world, world.revision), AS_OF, judge_config())
    )
    assert REVENUE in content
    assert REVISED not in content
    assert "$1,300.0" not in content
    assert f'<chunk id="{pk(world.revised)}"' not in content


def test_once_the_revision_is_public_the_original_is_gone(session: Session, world: World) -> None:
    # Control for the test above: visibility, not the id list, decides.
    later = AS_OF + timedelta(hours=2)
    content = user_content(
        build_messages(session, gapx(world, world.revision), later, judge_config())
    )
    assert REVISED in content
    assert REVENUE not in content


def test_a_future_document_never_appears_even_if_listed(session: Session, world: World) -> None:
    content = user_content(
        build_messages(session, gapx(world, world.future_doc), AS_OF, judge_config())
    )
    assert FUTURE not in content
    assert f'<chunk id="{pk(world.future)}"' not in content
    assert content.count("<chunk ") == 3


def test_another_company_chunk_never_appears(session: Session, world: World) -> None:
    content = user_content(build_messages(session, gapx(world), AS_OF, judge_config()))
    assert OTHER not in content
    assert f'<chunk id="{pk(world.other)}"' not in content


def test_every_chunk_in_the_prompt_is_visible_at_as_of(session: Session, world: World) -> None:
    listed = gapx(world, world.revision, world.future_doc)
    content = user_content(build_messages(session, listed, AS_OF, judge_config()))
    visible = {
        pk(c) for d in listed.document_ids for c in visible_chunks(session, AS_OF, document_id=d)
    }
    shown = {int(m) for m in re.findall(r'<chunk id="(\d+)"', content)}
    assert shown == visible == {pk(world.revenue), pk(world.outlook), pk(world.costs)}


# =============================================================================================
# judge_candidate: normal path
# =============================================================================================


def test_a_valid_take_answer_is_a_take(session: Session, world: World) -> None:
    cfg = judge_config()
    cand = gapx(world)
    script = Script(good(world, "take", 0.8))
    outcome = judge_candidate(session, cand, AS_OF, cfg, script)

    expected = build_body(cfg, build_messages(session, cand, AS_OF, cfg))
    assert script.bodies == [expected]
    assert outcome.candidate == cand
    assert outcome.verdict == "take"
    assert outcome.conviction == 0.8
    assert outcome.claims == [Claim.model_validate(c) for c in good_claims(world)]
    assert outcome.attempts == 1
    assert outcome.failures == []
    assert outcome.request_keys == [request_key(expected)]


def test_a_valid_pass_answer_is_a_pass(session: Session, world: World) -> None:
    outcome = judge(session, gapx(world), Script(good(world, "pass", 0.7)))
    assert outcome.verdict == "pass"
    assert outcome.conviction == 0.7
    assert len(outcome.claims) == 3
    assert outcome.attempts == 1


# =============================================================================================
# judge_candidate: boundaries
# =============================================================================================


@pytest.mark.parametrize(
    ("conviction", "verdict"),
    [(0.5, "take"), (0.49, "pass"), (0.0, "pass"), (1.0, "take")],
)
def test_take_below_min_conviction_is_a_pass(
    session: Session, world: World, conviction: float, verdict: str
) -> None:
    outcome = judge(session, gapx(world), Script(good(world, "take", conviction)))
    assert outcome.verdict == verdict
    assert outcome.conviction == conviction
    assert len(outcome.claims) == 3


def bad_answers(world: World) -> dict[str, dict[str, Any]]:
    claims = good_claims(world)
    return {
        "quote_not_in_chunk": bad_quote(world),
        "invalid_json": response('{"action": "take", "conviction": 0.8, "claims": ['),
        "conviction_1.3": response(answer("take", 1.3, claims)),
        "no_claims": response(answer("take", 0.8, [])),
        "nine_claims": response(answer("take", 0.8, [claims[0]] * 9)),
        "bad_stance": response(answer("take", 0.8, [claims[0] | {"stance": "neutral"}])),
        "number_not_in_quote": response(
            answer(
                "take",
                0.8,
                [claim(pk(world.revenue), REVENUE_QUOTE, "Revenue was $1,235 million.")],
            )
        ),
    }


BAD = [
    "quote_not_in_chunk",
    "invalid_json",
    "conviction_1.3",
    "no_claims",
    "nine_claims",
    "bad_stance",
    "number_not_in_quote",
]


@pytest.mark.parametrize("bad", BAD)
def test_a_bad_answer_is_retried_with_feedback(session: Session, world: World, bad: str) -> None:
    first = bad_answers(world)[bad]
    script = Script(first, good(world, "take", 0.8))
    outcome = judge(session, gapx(world), script)

    assert outcome.verdict == "take"
    assert outcome.conviction == 0.8
    assert outcome.attempts == 2
    assert outcome.failures == []
    assert outcome.request_keys == [request_key(b) for b in script.bodies]
    assert outcome.request_keys[0] != outcome.request_keys[1]

    original, retry = script.bodies
    assert {k: v for k, v in retry.items() if k != "messages"} == {
        k: v for k, v in original.items() if k != "messages"
    }
    assert retry["messages"][:2] == original["messages"]
    assert retry["messages"][2] == {
        "role": "assistant",
        "content": first["choices"][0]["message"]["content"],
    }
    feedback = retry["messages"][3]
    assert feedback["role"] == "user"
    assert feedback["content"].strip()
    assert len(retry["messages"]) == 4


@pytest.mark.parametrize("reason", ["quote_not_in_chunk", "number_not_in_quote"])
def test_validator_feedback_names_the_failing_reason(
    session: Session, world: World, reason: str
) -> None:
    script = Script(bad_answers(world)[reason], good(world))
    judge(session, gapx(world), script)
    assert reason in script.bodies[1]["messages"][3]["content"]


def test_feedback_names_each_failing_claim(session: Session, world: World) -> None:
    # Claims 0 and 2 fail for different reasons, claim 1 passes: the feedback lists both.
    claims = good_claims(world)
    claims[0] = claim(pk(world.revenue), REVENUE_QUOTE, "Revenue was $1,235 million.")
    claims[2] = claim(pk(world.costs), "Operating expenses fell sharply this year", COSTS_CLAIM)
    script = Script(response(answer("take", 0.8, claims)), good(world))
    judge(session, gapx(world), script)
    text = script.bodies[1]["messages"][3]["content"]
    assert "number_not_in_quote" in text
    assert "quote_not_in_chunk" in text


def test_the_second_retry_carries_the_whole_conversation(session: Session, world: World) -> None:
    script = Script(bad_quote(world), bad_answers(world)["invalid_json"], good(world))
    outcome = judge(session, gapx(world), script, max_retries=2)
    assert outcome.verdict == "take"
    assert outcome.attempts == 3
    third = script.bodies[2]["messages"]
    assert third[:4] == script.bodies[1]["messages"]
    assert [m["role"] for m in third] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]


def test_retries_exhausted_is_no_call(session: Session, world: World) -> None:
    script = Script(bad_quote(world), bad_quote(world), bad_quote(world))
    outcome = judge(session, gapx(world), script, max_retries=2)
    assert outcome.verdict == "no_call"
    assert outcome.conviction is None
    assert outcome.claims == []
    assert outcome.attempts == 3
    assert len(outcome.request_keys) == 3
    assert outcome.failures
    assert any("quote_not_in_chunk" in f for f in outcome.failures)
    assert script.responses == []


def test_zero_retries_means_one_call(session: Session, world: World) -> None:
    script = Script(bad_quote(world))
    outcome = judge(session, gapx(world), script, max_retries=0)
    assert outcome.verdict == "no_call"
    assert outcome.attempts == 1
    assert len(script.bodies) == 1


def test_failures_are_those_of_the_last_attempt(session: Session, world: World) -> None:
    script = Script(bad_quote(world), bad_answers(world)["number_not_in_quote"])
    outcome = judge(session, gapx(world), script, max_retries=1)
    assert outcome.verdict == "no_call"
    assert any("number_not_in_quote" in f for f in outcome.failures)
    assert not any("quote_not_in_chunk" in f for f in outcome.failures)


@pytest.mark.parametrize(
    "reply",
    [
        response("I can't help with investment decisions.", finish_reason="refusal"),
        response(None, finish_reason="refusal"),
        response("", finish_reason="stop"),
        response(None, finish_reason="stop"),
    ],
    ids=["refusal_text", "refusal_none", "empty", "none"],
)
def test_refusal_or_empty_reply_is_no_call_without_retry(
    session: Session, world: World, reply: dict[str, Any]
) -> None:
    script = Script(reply, good(world))
    outcome = judge(session, gapx(world), script, max_retries=2)
    assert outcome.verdict == "no_call"
    assert outcome.conviction is None
    assert outcome.claims == []
    assert outcome.attempts == 1
    assert len(script.bodies) == 1


def test_a_refusal_in_the_native_finish_reason_is_no_call_without_retry(
    session: Session, world: World
) -> None:
    # Assumption (not yet seen on a real reply): a provider refusal surfaces only as
    # native_finish_reason "refusal" with finish_reason "stop". Real replies so far: "stop" /
    # "end_turn" / message.refusal null (smoke run 2026-10-22).
    reply = response("I can't help with that.", finish_reason="stop")
    reply["choices"][0]["native_finish_reason"] = "refusal"
    script = Script(reply, good(world))
    outcome = judge(session, gapx(world), script, max_retries=2)
    assert outcome.verdict == "no_call"
    assert outcome.attempts == 1


def test_same_inputs_same_bodies(session: Session, world: World) -> None:
    a, b = Script(bad_quote(world), good(world)), Script(bad_quote(world), good(world))
    first = judge(session, gapx(world), a)
    second = judge(session, gapx(world), b)
    assert a.bodies == b.bodies
    assert first.request_keys == second.request_keys


# A filing whose text did not survive conversion (image-only exhibit; real case AFRM
# 2026-08-28: one 46-character chunk "EX-99.1 2 affirmfq426shareholderle.htm EX-99.1").
IMAGE_ONLY = "EX-99.1 2 affirmfq426shareholderle.htm EX-99.1"


def image_only(session: Session, source: Source, world: World) -> Candidate:
    doc = new_document(
        session, source_id=pk(source), entity_id=world.gapx, available_at=RELEASE, doc_type="8-K"
    )
    new_chunk(session, document_id=pk(doc), text=IMAGE_ONLY, section_path=SECTION)
    return candidate(world.gapx, (pk(doc),))


def test_a_filing_without_readable_text_is_a_no_call_without_asking(
    session: Session, source: Source, world: World
) -> None:
    script = Script()  # any call fails the test
    outcome = judge(session, image_only(session, source, world), script, min_filing_chars=500)
    assert outcome.verdict == "no_call"
    assert outcome.conviction is None
    assert outcome.claims == []
    assert outcome.attempts == 0
    assert outcome.request_keys == []
    assert any("filing_too_short" in f and "46" in f for f in outcome.failures)
    assert script.bodies == []


def test_a_filing_exactly_at_the_minimum_is_judged(
    session: Session, source: Source, world: World
) -> None:
    cand = image_only(session, source, world)
    script = Script(response(answer("pass", 0.2, [claim(0, IMAGE_ONLY, "Exhibit 99.1.")])))
    outcome = judge(session, cand, script, min_filing_chars=len(IMAGE_ONLY), max_retries=0)
    assert outcome.attempts == 1
    assert len(script.bodies) == 1


def test_only_text_visible_at_as_of_counts_toward_the_minimum(
    session: Session, source: Source, world: World
) -> None:
    # A long restatement published after AS_OF exists, but at AS_OF the filing is still too short.
    cand = image_only(session, source, world)
    [doc_id] = cand.document_ids
    revision = new_document(
        session,
        source_id=pk(source),
        entity_id=world.gapx,
        available_at=AS_OF + timedelta(hours=1),
        doc_type="8-K/A",
        supersedes_id=doc_id,
    )
    new_chunk(session, document_id=pk(revision), text=REVENUE * 20, section_path=SECTION)
    script = Script()
    outcome = judge_candidate(
        session,
        candidate(world.gapx, (doc_id, pk(revision))),
        AS_OF,
        judge_config(min_filing_chars=500),
        script,
    )
    assert outcome.verdict == "no_call"
    assert script.bodies == []


# =============================================================================================
# judge_candidate: leakage
# =============================================================================================


def citing(chunk: Chunk, quote: str, text: str) -> dict[str, Any]:
    return response(answer("take", 0.9, [claim(pk(chunk), quote, text)]))


def test_a_future_chunk_cited_is_retried_and_ends_in_no_call(
    session: Session, world: World
) -> None:
    # The chunk exists (published 60 days later) and the quote is really in it: only as_of
    # stands between this answer and a cited take.
    reply = citing(world.future, FUTURE_QUOTE, FUTURE_CLAIM)
    script = Script(reply, reply, reply)
    outcome = judge(session, gapx(world), script, max_retries=2)
    assert outcome.verdict == "no_call"
    assert outcome.claims == []
    assert outcome.attempts == 3
    assert any("not_visible" in f for f in outcome.failures)
    assert "not_visible" in script.bodies[1]["messages"][3]["content"]


def test_a_future_chunk_next_to_good_claims_still_fails(session: Session, world: World) -> None:
    claims = [*good_claims(world), claim(pk(world.future), FUTURE_QUOTE, FUTURE_CLAIM)]
    reply = response(answer("take", 0.9, claims))
    outcome = judge(session, gapx(world), Script(reply), max_retries=0)
    assert outcome.verdict == "no_call"
    assert outcome.claims == []


def test_the_later_revision_cannot_be_cited(session: Session, world: World) -> None:
    quote = "Fourth quarter revenue was $1,300.0 million, up 24% from the prior year"
    reply = citing(world.revised, quote, "Fourth quarter revenue was $1,300.0 million.")
    outcome = judge(session, gapx(world), Script(reply), max_retries=0)
    assert outcome.verdict == "no_call"
    assert any("not_visible" in f for f in outcome.failures)


def test_another_company_chunk_cannot_be_cited(session: Session, world: World) -> None:
    quote = "OTHR Inc. reported quarterly revenue of $9,999.0 million"
    reply = citing(world.other, quote, "Quarterly revenue was $9,999.0 million.")
    outcome = judge(session, gapx(world), Script(reply), max_retries=0)
    assert outcome.verdict == "no_call"
    assert any("entity_mismatch" in f for f in outcome.failures)


def test_no_accepted_claim_cites_an_invisible_chunk(session: Session, world: World) -> None:
    # Whatever the path to an accepted verdict, its claims cite only chunks visible at AS_OF.
    future = citing(world.future, FUTURE_QUOTE, FUTURE_CLAIM)
    outcome = judge(session, gapx(world), Script(future, good(world)), max_retries=1)
    visible = {pk(c) for c in visible_chunks(session, AS_OF)}
    assert outcome.verdict == "take"
    assert {c.chunk_id for c in outcome.claims} <= visible
    assert pk(world.future) not in {c.chunk_id for c in outcome.claims}


# =============================================================================================
# order_for_entry
# =============================================================================================


def made(
    security_id: str, verdict: str, conviction: float | None, world_entity: int = 1
) -> JudgeOutcome:
    return JudgeOutcome(
        candidate=candidate(world_entity, (1,), security_id=security_id, symbol=security_id),
        verdict=verdict,  # type: ignore[arg-type]
        conviction=conviction,
        claims=[],
        attempts=1,
        failures=[],
        request_keys=["k"],
    )


def test_order_for_entry_keeps_takes_by_conviction() -> None:
    outcomes = [
        made("S_A", "take", 0.6),
        made("S_B", "pass", 0.95),
        made("S_C", "take", 0.9),
        made("S_D", "no_call", None),
        made("S_E", "take", 0.7),
    ]
    ordered = order_for_entry(outcomes)
    assert [o.candidate.security_id for o in ordered] == ["S_C", "S_E", "S_A"]


def test_order_for_entry_breaks_ties_by_security_id() -> None:
    outcomes = [made("S_Z", "take", 0.8), made("S_B", "take", 0.8), made("S_M", "take", 0.8)]
    assert [o.candidate.security_id for o in order_for_entry(outcomes)] == ["S_B", "S_M", "S_Z"]


def test_order_for_entry_does_not_depend_on_input_order() -> None:
    outcomes = [made("S_A", "take", 0.6), made("S_B", "take", 0.9), made("S_C", "take", 0.6)]
    expected = [o.candidate.security_id for o in order_for_entry(outcomes)]
    assert [o.candidate.security_id for o in order_for_entry(list(reversed(outcomes)))] == expected
    assert expected == ["S_B", "S_A", "S_C"]


def test_order_for_entry_of_nothing_is_nothing() -> None:
    assert order_for_entry([]) == []
    assert order_for_entry([made("S_A", "pass", 0.9), made("S_B", "no_call", None)]) == []


def test_outcome_is_a_plain_record() -> None:
    o = made("S_A", "take", 0.6)
    assert replace(o, conviction=0.7).conviction == 0.7
    assert json.dumps(o.request_keys) == '["k"]'
