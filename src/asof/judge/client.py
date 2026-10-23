"""The request the llm judge sends and the one place it is sent: OpenRouter chat completions.

The body pins everything that changes an answer: the model, the upstream provider (no fallback to
another host of the same model, whose weights or settings may differ), the reasoning effort, the
output limit and a strict JSON schema, so the reply is the answer's fields and nothing else.
OpenRouter forwards requests to the provider; ``require_parameters`` makes it refuse a provider
that would silently drop the schema or the effort instead of honouring them.
"""

import copy
from typing import Any

import httpx

from asof.strategy import JudgeConfig

URL = "https://openrouter.ai/api/v1/chat/completions"

# The answer as the model must write it; asof.judge.llm.JudgeAnswer checks the same fields and
# the bounds a strict schema cannot carry (conviction in [0, 1], 1 to 8 claims).
ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["action", "conviction", "claims"],
    "properties": {
        "action": {"type": "string", "enum": ["take", "pass"]},
        "conviction": {"type": "number"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["claim", "chunk_id", "quote", "stance"],
                "properties": {
                    "claim": {"type": "string"},
                    "chunk_id": {"type": "integer"},
                    "quote": {"type": "string"},
                    "stance": {"type": "string", "enum": ["for", "against"]},
                },
            },
        },
    },
}


class JudgeCallError(Exception):
    """The model could not be asked: refused key, network or HTTP error, or a reply that holds
    no answer. Raised rather than returned, so it is never cached: asking again may succeed."""


# A refusal is an answer (kept, so it is not paid for again); anything else without content is
# a failure of the call.
_REFUSALS = {"refusal", "content_filter"}


def is_refusal(choice: dict[str, Any]) -> bool:
    message = choice.get("message") or {}
    return (
        choice.get("finish_reason") in _REFUSALS
        or choice.get("native_finish_reason") in _REFUSALS
        or bool(message.get("refusal"))
    )


def build_body(cfg: JudgeConfig, messages: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "model": cfg.model,
        "messages": copy.deepcopy(messages),
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "judgement", "strict": True, "schema": ANSWER_SCHEMA},
        },
        "reasoning": {"effort": cfg.effort},
        "provider": {
            "order": [cfg.pin_provider],
            "allow_fallbacks": False,
            "require_parameters": True,
        },
        "max_tokens": cfg.max_tokens,
        "usage": {"include": True},
    }


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        *,
        transport: httpx.BaseTransport | None = None,
        # One answer at effort high can think for many minutes; a timeout is still billed.
        timeout: float = 1800.0,
    ) -> None:
        if not api_key or not api_key.strip():
            raise JudgeCallError("no OpenRouter API key (set OPENROUTER_API_KEY)")
        self._http = httpx.Client(
            headers={"Authorization": f"Bearer {api_key.strip()}"},
            timeout=timeout,
            transport=transport,
            follow_redirects=False,
        )

    def complete(self, body: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._http.post(URL, json=body)
        except httpx.HTTPError as error:
            raise JudgeCallError(f"OpenRouter request failed: {error!r}") from error
        if response.status_code != 200:
            raise JudgeCallError(f"OpenRouter HTTP {response.status_code}: {response.text[:500]}")
        reply = response.json()
        # A provider failure after the request was accepted comes back as 200 with an error.
        if "error" in reply:
            raise JudgeCallError(f"OpenRouter error: {reply['error']}")
        # Raised, not returned: a reply with no answer in it must never be cached as one.
        if not reply.get("choices"):
            raise JudgeCallError(f"OpenRouter reply without choices: {str(reply)[:500]}")
        choice = reply["choices"][0]
        if is_refusal(choice):
            return reply
        content = (choice.get("message") or {}).get("content")
        finish = choice.get("finish_reason")
        if finish in (None, "error") or not isinstance(content, str) or not content.strip():
            raise JudgeCallError(f"OpenRouter reply holds no answer: finish_reason={finish!r}")
        return reply
