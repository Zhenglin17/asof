"""Guards against a model call that cannot be reproduced or audited: a request routed to whatever
provider is up (different weights, different answers), a reply that is free text instead of the
schema, reasoning effort or token limits not pinned, an API error read as an answer, or a request
sent without the key.

API under test (asof.judge.client):

    build_body(cfg: JudgeConfig, messages: list[dict]) -> dict   OpenRouter chat completions body
    class JudgeCallError(Exception)
    OpenRouterClient(api_key: str, *, transport=None).complete(body) -> dict

All HTTP goes through ``httpx.MockTransport``; nothing reaches the network.

No leakage case: the client has no as_of; what it sends is built from visible chunks upstream
(test_llm.py). Assumptions only real data can confirm (documented by OpenRouter, not measured
here):
- ``POST https://openrouter.ai/api/v1/chat/completions`` with ``Authorization: Bearer <key>``
  is the endpoint and auth (documented); check with one real call recorded in the cache.
- ``provider.order`` + ``allow_fallbacks: false`` + ``require_parameters: true`` keeps the call
  on the pinned provider or fails it (documented); check the ``provider`` field of a real reply.
- The pinned provider honours ``response_format`` json_schema with ``strict: true`` and
  ``reasoning.effort`` at the configured level (assumption: provider support varies); check a
  real reply parses as JudgeAnswer and its usage shows reasoning tokens.
- An error reply is a 4xx with ``{"error": {"code", "message"}}`` (documented).
"""

import json
from typing import Any

import httpx
import pytest

from asof.judge import client
from tests.judge.fakes import LLM_FIELDS, judge_config, response

JudgeCallError = client.JudgeCallError
OpenRouterClient = client.OpenRouterClient
build_body = client.build_body

URL = "https://openrouter.ai/api/v1/chat/completions"
MESSAGES: list[dict[str, Any]] = [
    {"role": "system", "content": LLM_FIELDS["prompt"]},
    {"role": "user", "content": '<chunk id="1" section="EX-99.1">Revenue rose.</chunk>'},
]


# =============================================================================================
# build_body
# =============================================================================================


def test_body_pins_model_provider_schema_effort_and_tokens() -> None:
    cfg = judge_config()
    body = build_body(cfg, MESSAGES)
    assert body["model"] == cfg.model
    assert body["provider"] == {
        "order": [cfg.pin_provider],
        "allow_fallbacks": False,
        "require_parameters": True,
    }
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["reasoning"] == {"effort": cfg.effort}
    assert body["max_tokens"] == cfg.max_tokens
    assert body["usage"] == {"include": True}
    assert body["messages"] == MESSAGES


def test_body_schema_names_the_answer_fields() -> None:
    schema = build_body(judge_config(), MESSAGES)["response_format"]["json_schema"]
    text = json.dumps(schema)
    for field in ("action", "conviction", "claims", "chunk_id", "quote", "stance"):
        assert field in text, field


def test_body_is_deterministic_and_json() -> None:
    cfg = judge_config()
    a, b = build_body(cfg, MESSAGES), build_body(cfg, MESSAGES)
    assert a == b
    assert json.loads(json.dumps(a)) == a


def test_body_does_not_mutate_the_messages() -> None:
    before = json.loads(json.dumps(MESSAGES))
    build_body(judge_config(), MESSAGES)
    assert MESSAGES == before


@pytest.mark.parametrize(
    ("change", "path"),
    [
        ({"model": "vendor/other"}, ("model",)),
        ({"effort": "low"}, ("reasoning", "effort")),
        ({"max_tokens": 128}, ("max_tokens",)),
        ({"pin_provider": "other-provider"}, ("provider", "order")),
    ],
)
def test_body_follows_the_config(change: dict[str, Any], path: tuple[str, ...]) -> None:
    def at(body: dict[str, Any]) -> Any:
        for key in path:
            body = body[key]
        return body

    assert at(build_body(judge_config(**change), MESSAGES)) != at(
        build_body(judge_config(), MESSAGES)
    )


# =============================================================================================
# OpenRouterClient
# =============================================================================================


def test_complete_posts_with_bearer_key_and_returns_the_json() -> None:
    seen: list[httpx.Request] = []
    reply = response('{"action": "pass", "conviction": 0.2, "claims": []}')

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=reply)

    body = build_body(judge_config(), MESSAGES)
    api = OpenRouterClient("sk-test-key", transport=httpx.MockTransport(handler))
    assert api.complete(body) == reply

    [request] = seen
    assert request.method == "POST"
    assert str(request.url) == URL
    assert request.headers["Authorization"] == "Bearer sk-test-key"
    assert json.loads(request.content) == body


@pytest.mark.parametrize("status", [400, 401, 402, 429])
def test_error_reply_raises(status: int) -> None:
    # Documented error shape: {"error": {"code": <status>, "message": ...}}.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status, json={"error": {"code": status, "message": "No endpoints found"}}
        )

    api = OpenRouterClient("sk-test-key", transport=httpx.MockTransport(handler))
    with pytest.raises(JudgeCallError):
        api.complete(build_body(judge_config(), MESSAGES))


def _reply_with(choice: dict[str, Any]) -> dict[str, Any]:
    return {"id": "gen-x", "choices": [{"index": 0, **choice}], "usage": {"cost": 0.01}}


@pytest.mark.parametrize(
    "choice",
    [
        {"message": {"role": "assistant", "content": "{}"}, "finish_reason": "error"},
        {"message": {"role": "assistant", "content": "{}"}, "finish_reason": None},
        {"message": {"role": "assistant", "content": ""}, "finish_reason": "stop"},
        {"message": {"role": "assistant", "content": None}, "finish_reason": "length"},
    ],
    ids=["finish_error", "finish_none", "empty_stop", "empty_length"],
)
def test_a_reply_that_holds_no_answer_raises_so_it_is_never_cached(choice: dict[str, Any]) -> None:
    # A provider hiccup must not become a permanent no_call in the cache: raise, ask again later.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_reply_with(choice))

    api = OpenRouterClient("sk-test-key", transport=httpx.MockTransport(handler))
    with pytest.raises(JudgeCallError):
        api.complete(build_body(judge_config(), MESSAGES))


@pytest.mark.parametrize(
    "choice",
    [
        {"message": {"role": "assistant", "content": None}, "finish_reason": "refusal"},
        {
            "message": {"role": "assistant", "content": None, "refusal": "I can't help."},
            "finish_reason": "stop",
        },
        {
            "message": {"role": "assistant", "content": ""},
            "finish_reason": "stop",
            "native_finish_reason": "refusal",
        },
        {"message": {"role": "assistant", "content": None}, "finish_reason": "content_filter"},
    ],
    ids=["finish_refusal", "message_refusal", "native_refusal", "content_filter"],
)
def test_a_refusal_is_an_answer_and_is_returned(choice: dict[str, Any]) -> None:
    # A refusal is the model's answer to this request: kept (and cached) so it is not paid again.
    reply = _reply_with(choice)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=reply)

    api = OpenRouterClient("sk-test-key", transport=httpx.MockTransport(handler))
    assert api.complete(build_body(judge_config(), MESSAGES)) == reply


@pytest.mark.parametrize(
    "error",
    [httpx.ConnectError("connection refused"), httpx.ReadTimeout("timed out")],
    ids=["connect", "timeout"],
)
def test_a_network_failure_is_a_judge_call_error(error: Exception) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise error

    api = OpenRouterClient("sk-test-key", transport=httpx.MockTransport(handler))
    with pytest.raises(JudgeCallError):
        api.complete(build_body(judge_config(), MESSAGES))


@pytest.mark.parametrize("key", ["", "   "])
def test_missing_key_is_refused_before_any_request(key: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request may be sent without a key")

    with pytest.raises((ValueError, JudgeCallError)):
        OpenRouterClient(key, transport=httpx.MockTransport(handler))
