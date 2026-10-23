"""The llm judge: a model reads the filing behind one candidate and answers take or pass, with a
conviction and claims that each quote a chunk word for word.

Code, not the model, decides what happens with the answer. The answer must parse into
``JudgeAnswer`` and pass the Citation Validator; if it does not, the failures are sent back and
the model answers again, up to ``max_retries`` times. An answer that still fails, a refusal or an
empty reply is a ``no_call``: nothing is acted on. A take below ``min_conviction`` is a pass.

What the model sees is built from ``visible_chunks`` at the decision's as_of, so a revision or a
filing published later can neither be read nor cited (the validator checks the citations against
the same as_of).
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlmodel import Session

from asof.evidence.validate import ClaimCitation, DecisionDraft, validate
from asof.judge.cache import request_key
from asof.judge.client import build_body, is_refusal
from asof.scan.earnings_gap import Candidate
from asof.store.models import Chunk, EvidenceKind
from asof.store.queries import visible_chunks
from asof.strategy import JudgeConfig

ET = ZoneInfo("America/New_York")
Call = Callable[[dict[str, Any]], dict[str, Any]]
Verdict = Literal["take", "pass", "no_call"]


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim: str
    chunk_id: int
    quote: str
    stance: Literal["for", "against"]


class JudgeAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["take", "pass"]
    # The model's probability that the long trade ends with a profit.
    conviction: float = Field(ge=0, le=1)
    claims: list[Claim] = Field(min_length=1, max_length=8)


@dataclass(frozen=True)
class JudgeOutcome:
    candidate: Candidate
    verdict: Verdict
    conviction: float | None
    # The validated claims; empty for a no_call.
    claims: list[Claim]
    attempts: int
    # Why the last attempt failed; empty when it passed.
    failures: list[str]
    # One per call, in order: the cache keys of the requests sent (none if not asked).
    request_keys: list[str]


def filing_chunks(session: Session, candidate: Candidate, as_of: datetime) -> list[Chunk]:
    """Every chunk of the candidate's documents that is public at ``as_of``, in order."""
    return [
        chunk
        for document_id in candidate.document_ids
        for chunk in visible_chunks(session, as_of, document_id=document_id)
    ]


def build_messages(
    session: Session, candidate: Candidate, as_of: datetime, cfg: JudgeConfig
) -> list[dict[str, Any]]:
    """The system prompt and one user message: the candidate's numbers and every chunk of its
    documents that is public at ``as_of``."""
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError(f"as_of must carry a timezone, got naive {as_of!r}")
    c = candidate
    lines = [
        f"As of: {as_of.astimezone(ET).isoformat()} (New York)",
        # The ticker of the reaction session, not Entity.name: that is today's name, and a later
        # rename would tell of a later event. The filing names the company as it filed.
        f"Ticker: {c.symbol}",
        f"Reaction session: {c.reaction_session.isoformat()}",
        f"Previous close {c.pre_earnings_close:.2f}; gap-day open {c.gap_open:.2f}, "
        f"high {c.gap_high:.2f}, low {c.gap_low:.2f}, close {c.gap_close:.2f}; "
        f"gap {c.gap * 100:.1f}%; relative volume {c.rvol:.1f}x; ATR {c.atr:.2f}",
        "",
        "Filing chunks:",
    ]
    for chunk in filing_chunks(session, candidate, as_of):
        lines.append(
            f'<chunk id="{chunk.id}" section="{chunk.section_path}">\n{chunk.text}\n</chunk>'
        )
    return [
        {"role": "system", "content": cfg.prompt},
        {"role": "user", "content": "\n".join(lines)},
    ]


def _refused(choice: dict[str, Any]) -> str | None:
    message = choice.get("message") or {}
    content = message.get("content")
    finish = choice.get("finish_reason")
    # Retrying a refusal would only ask the model to decline again.
    if is_refusal(choice):
        return f"refused: finish_reason={finish}"
    if not isinstance(content, str) or not content.strip():
        return f"empty reply: finish_reason={finish}"
    return None


def _check(
    session: Session, candidate: Candidate, as_of: datetime, content: str
) -> tuple[JudgeAnswer | None, list[str]]:
    try:
        answer = JudgeAnswer.model_validate_json(content)
    except ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or 'answer'}: {e['msg']}"
            for e in error.errors()[:5]
        )
        return None, [f"invalid_answer: {problems}"]
    draft = DecisionDraft(
        as_of=as_of,
        entity_id=candidate.entity_id,
        citations=[
            ClaimCitation(
                claim=c.claim, evidence_kind=EvidenceKind.CHUNK, chunk_id=c.chunk_id, quote=c.quote
            )
            for c in answer.claims
        ],
    )
    result = validate(session, draft)
    failures = [f"claim {f.index}: {f.reason.value}: {f.detail}" for f in result.failures]
    return answer, failures


def _feedback(failures: Sequence[str]) -> str:
    return (
        "The citation check rejected this answer:\n"
        + "\n".join(f"- {f}" for f in failures)
        + "\nReturn the full corrected answer. Fix or drop each rejected claim (claims are "
        "numbered from 0); every other rule still holds."
    )


def judge_candidate(
    session: Session, candidate: Candidate, as_of: datetime, cfg: JudgeConfig, call: Call
) -> JudgeOutcome:
    """Ask the model through ``call`` (the client, or a cache in front of it) and turn the answer
    into a verdict, retrying with the validator's failures."""
    assert cfg.max_retries is not None and cfg.min_conviction is not None
    assert cfg.min_filing_chars is not None
    text = sum(len(c.text) for c in filing_chunks(session, candidate, as_of))
    if text < cfg.min_filing_chars:
        short = f"filing_too_short: {text} characters of visible text < {cfg.min_filing_chars}"
        return JudgeOutcome(candidate, "no_call", None, [], 0, [short], [])
    messages = build_messages(session, candidate, as_of, cfg)
    keys: list[str] = []
    failures: list[str] = []
    for attempt in range(1, cfg.max_retries + 2):
        body = build_body(cfg, messages)
        keys.append(request_key(body))
        reply = call(body)
        choice = reply["choices"][0]
        refused = _refused(choice)
        if refused is not None:
            return JudgeOutcome(candidate, "no_call", None, [], attempt, [refused], keys)
        content: str = choice["message"]["content"]
        answer, failures = _check(session, candidate, as_of, content)
        if answer is not None and not failures:
            take = answer.action == "take" and answer.conviction >= cfg.min_conviction
            verdict: Verdict = "take" if take else "pass"
            claims = list(answer.claims)
            return JudgeOutcome(candidate, verdict, answer.conviction, claims, attempt, [], keys)
        messages = [
            *messages,
            {"role": "assistant", "content": content},
            {"role": "user", "content": _feedback(failures)},
        ]
    return JudgeOutcome(candidate, "no_call", None, [], cfg.max_retries + 1, failures, keys)


def one_per_security(candidates: list[Candidate]) -> list[Candidate]:
    """The candidates judged on a day, in the order they are judged: one per security (the
    replay and the cache-filling window must agree on it)."""
    chosen: dict[str, Candidate] = {}
    for candidate in sorted(candidates, key=lambda c: (c.security_id, c.entity_id)):
        chosen.setdefault(candidate.security_id, candidate)
    return list(chosen.values())


def order_for_entry(outcomes: Sequence[JudgeOutcome]) -> list[JudgeOutcome]:
    """The takes, most convinced first; equal convictions by security id."""
    takes = [o for o in outcomes if o.verdict == "take"]
    return sorted(takes, key=lambda o: (-(o.conviction or 0.0), o.candidate.security_id))
