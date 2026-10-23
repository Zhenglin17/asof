"""Fake model answers and an llm judge config for the judge and replay tests. Nothing here talks to
the network: a test hands ``Script.call`` to the judge in place of the real client.

Provenance of the fake responses:
- Response shape ``{"choices": [{"message": {"content": ...}, "finish_reason": ...}],
  "usage": {...}}``: measured. Smoke run 2026-10-22 (3 calls, the configured model pinned to
  its maker as provider): ``provider "Anthropic"``, ``finish_reason "stop"``, ``native_finish_reason
  "end_turn"``, ``message.refusal null``, ``usage.cost`` present (e.g. 0.160708), reasoning
  tokens present.
- A refusal (``finish_reason`` "refusal" / "content_filter", ``native_finish_reason``
  "refusal", or a non-empty ``message.refusal``): assumption, not yet seen on a real reply.
  Checked on real data by tallying finish reasons over every cached reply after a run (the
  pre-flight check in PROGRESS).
- Usage numbers are made up, except the shape of ``usage.cost`` in ``costed``: measured. Real
  OpenRouter replies to a body with ``usage: {include: true}`` carry ``usage.cost`` in USD as a
  float (measured 2026-10-22 on the pinned model, e.g. ``"cost": 0.04852``).
"""

import json
from collections.abc import Sequence
from datetime import date
from typing import Any

from asof.strategy import JudgeConfig

PROMPT = (
    "You review one earnings-gap candidate. Read the filing chunks, decide take or pass, and "
    "back every claim with a verbatim quote from one chunk."
)

# Every llm field, as judge.yaml would hold it. Model and provider names are placeholders.
LLM_FIELDS: dict[str, Any] = {
    "provider": "openrouter",
    "model": "vendor/model-under-test",
    "pin_provider": "provider-under-test",
    "effort": "high",
    "max_retries": 2,
    "min_conviction": 0.5,
    "max_tokens": 4000,
    # Below every fixture filing (the shortest, two chunks, is about 150 characters).
    "min_filing_chars": 50,
    "training_cutoff": date(2025, 1, 31),
    "cutoff_source": "model card of the placeholder model",
    "prompt": PROMPT,
}
LLM_JUDGE: dict[str, Any] = {"kind": "llm", **LLM_FIELDS}


def judge_config(**changes: Any) -> JudgeConfig:
    return JudgeConfig.model_validate({**LLM_JUDGE, **changes})


def claim(chunk_id: int, quote: str, text: str, stance: str = "for") -> dict[str, Any]:
    return {"claim": text, "chunk_id": chunk_id, "quote": quote, "stance": stance}


def answer(action: str, conviction: float, claims: Sequence[dict[str, Any]]) -> str:
    return json.dumps({"action": action, "conviction": conviction, "claims": list(claims)})


def response(content: str | None, finish_reason: str = "stop") -> dict[str, Any]:
    return {
        "id": "gen-fixture",
        "model": LLM_FIELDS["model"],
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 1200, "completion_tokens": 300, "total_tokens": 1500},
    }


def costed(content: str | None, cost: float, finish_reason: str = "stop") -> dict[str, Any]:
    """A reply whose usage carries what it cost, as OpenRouter reports it (measured, see top)."""
    reply = response(content, finish_reason)
    reply["usage"] = {**reply["usage"], "cost": cost}
    return reply


class Script:
    """A fake ``call``: returns the given responses in order and records every body it got.
    Running out of responses fails the test, so an extra call cannot go unnoticed."""

    def __init__(self, *responses: dict[str, Any]) -> None:
        self.responses = list(responses)
        self.bodies: list[dict[str, Any]] = []

    def __call__(self, body: dict[str, Any]) -> dict[str, Any]:
        self.bodies.append(json.loads(json.dumps(body)))  # a copy: later mutation cannot hide
        if not self.responses:
            raise AssertionError(f"unexpected call #{len(self.bodies)}")
        return self.responses.pop(0)
