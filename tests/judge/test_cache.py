"""Guards against a replay that is not reproducible or quietly goes online: two requests that differ
in any input (model, effort, prompt, a chunk's text) sharing one cached answer, an offline replay
that calls the model on a miss, or a failed call leaving a half-written answer behind.

API under test (asof.judge.cache):

    request_key(body: dict) -> str          sha256 hex of canonical JSON (sorted keys)
    class CacheMiss(Exception)
    class LLMCache(root: Path, *, offline: bool)
        .call(body, fetch) -> dict          hit: stored response, fetch not called
                                            miss + offline: CacheMiss, fetch not called
                                            miss + online: fetch(body) once, stored, returned
        files: root / key[:2] / f"{key}.json" = {"request": body, "response": response}

The cache has no as_of: a key is a function of the request alone, and the request already holds
only what was visible (that is tested in test_llm.py). The leakage-relevant guard here is that a
chunk's text is part of the key, so a revised filing can never reuse the old answer.
Assumptions only real data can confirm:
- After a real llm replay, every file under ``<data root>/llm_cache`` satisfies
  ``request_key(file["request"]) == file name`` (a one-off audit over the directory).
- The real OpenRouter body is plain JSON (str / int / float / bool / list / dict only), so
  canonical JSON is a faithful key: check one cached real request round-trips through
  ``json.loads(json.dumps(...))`` unchanged.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from asof.judge import cache

CacheMiss = cache.CacheMiss
LLMCache = cache.LLMCache
request_key = cache.request_key

BODY: dict[str, Any] = {
    "model": "vendor/model-under-test",
    "provider": {"order": ["provider-under-test"], "allow_fallbacks": False},
    "reasoning": {"effort": "high"},
    "max_tokens": 4000,
    "messages": [
        {"role": "system", "content": "Decide take or pass."},
        {
            "role": "user",
            "content": '<chunk id="7" section="EX-99.1">Revenue was $1,234.5 million.</chunk>',
        },
    ],
}
RESPONSE: dict[str, Any] = {
    "choices": [{"message": {"content": '{"action": "pass"}'}, "finish_reason": "stop"}],
    "usage": {"total_tokens": 10},
}


def changed(path: tuple[Any, ...], value: Any) -> dict[str, Any]:
    body = json.loads(json.dumps(BODY))
    node = body
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return body


class Fetch:
    def __init__(self, reply: dict[str, Any] = RESPONSE) -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    def __call__(self, body: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(body)
        return self.reply


# =============================================================================================
# request_key
# =============================================================================================


def test_key_is_sha256_hex() -> None:
    key = request_key(BODY)
    assert len(key) == 64
    assert all(c in "0123456789abcdef" for c in key)
    assert request_key(BODY) == key


def test_key_ignores_key_insertion_order() -> None:
    reordered = {k: BODY[k] for k in reversed(list(BODY))}
    reordered["provider"] = {"allow_fallbacks": False, "order": ["provider-under-test"]}
    assert list(reordered) != list(BODY)
    assert request_key(reordered) == request_key(BODY)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("model",), "vendor/another-model"),
        (("reasoning", "effort"), "medium"),
        (("max_tokens",), 4001),
        (("provider", "order"), ["another-provider"]),
        (("messages", 0, "content"), "Decide take or pass!"),
    ],
    ids=["model", "effort", "max_tokens", "provider", "prompt"],
)
def test_key_changes_with_any_input(path: tuple[Any, ...], value: Any) -> None:
    assert request_key(changed(path, value)) != request_key(BODY)


def test_key_changes_when_a_chunk_text_changes() -> None:
    # Leakage guard: a revision of a filing with one digit changed must not hit the answer the
    # model gave to the original text.
    revised = changed(
        ("messages", 1, "content"),
        '<chunk id="7" section="EX-99.1">Revenue was $1,234.6 million.</chunk>',
    )
    assert request_key(revised) != request_key(BODY)


def test_list_order_is_part_of_the_key() -> None:
    # Messages are a conversation; their order is content, not layout.
    swapped = changed(("messages",), list(reversed(BODY["messages"])))
    assert request_key(swapped) != request_key(BODY)


# =============================================================================================
# LLMCache.call
# =============================================================================================


def test_online_miss_fetches_once_stores_and_then_hits(tmp_path: Path) -> None:
    store = LLMCache(tmp_path / "llm_cache", offline=False)
    fetch = Fetch()
    assert store.call(BODY, fetch) == RESPONSE
    assert fetch.calls == [BODY]

    key = request_key(BODY)
    path = tmp_path / "llm_cache" / key[:2] / f"{key}.json"
    assert path.is_file()
    assert json.loads(path.read_text()) == {"request": BODY, "response": RESPONSE}

    assert store.call(BODY, fetch) == RESPONSE
    assert len(fetch.calls) == 1


def test_offline_hit_returns_the_stored_response_without_fetching(tmp_path: Path) -> None:
    root = tmp_path / "llm_cache"
    LLMCache(root, offline=False).call(BODY, Fetch())
    fetch = Fetch({"choices": []})
    assert LLMCache(root, offline=True).call(BODY, fetch) == RESPONSE
    assert fetch.calls == []


def test_offline_miss_raises_and_never_fetches(tmp_path: Path) -> None:
    fetch = Fetch()
    with pytest.raises(CacheMiss):
        LLMCache(tmp_path / "llm_cache", offline=True).call(BODY, fetch)
    assert fetch.calls == []
    root = tmp_path / "llm_cache"
    assert not root.exists() or not any(root.rglob("*.json"))


def test_offline_miss_on_a_neighbouring_request(tmp_path: Path) -> None:
    # A cached answer for the original text does not serve the revised one.
    root = tmp_path / "llm_cache"
    LLMCache(root, offline=False).call(BODY, Fetch())
    revised = changed(("messages", 1, "content"), "Revenue was $1,300.0 million.")
    with pytest.raises(CacheMiss):
        LLMCache(root, offline=True).call(revised, Fetch())


def test_a_failed_fetch_stores_nothing(tmp_path: Path) -> None:
    root = tmp_path / "llm_cache"

    def broken(body: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("provider timed out")

    with pytest.raises(RuntimeError):
        LLMCache(root, offline=False).call(BODY, broken)
    assert not root.exists() or not any(root.rglob("*.json"))
    # and the next online call really fetches
    fetch = Fetch()
    assert LLMCache(root, offline=False).call(BODY, fetch) == RESPONSE
    assert len(fetch.calls) == 1


def test_two_requests_get_two_files(tmp_path: Path) -> None:
    root = tmp_path / "llm_cache"
    store = LLMCache(root, offline=False)
    other = changed(("model",), "vendor/another-model")
    store.call(BODY, Fetch())
    store.call(other, Fetch({"choices": [], "usage": {}}))
    assert len(list(root.rglob("*.json"))) == 2
    assert LLMCache(root, offline=True).call(other, Fetch()) == {"choices": [], "usage": {}}
