"""Model answers stored by request, so a replay reads the answers a model gave instead of asking
again: the same request gives the same decision, and an offline replay never reaches the network.

The key is the SHA-256 of the request body as canonical JSON. The body holds the model, effort,
prompt and every chunk's text, so a change to any of them (a revised filing, a new prompt) is a
new request that no old answer can serve. The cache has no as_of: what a request may contain is
decided where it is built (``judge.llm.build_messages``), from chunks visible at the decision.
"""

import hashlib
import json
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

Body = dict[str, Any]
Response = dict[str, Any]


class CacheMiss(Exception):
    """An offline cache has no answer for a request."""


def canonical(body: Body) -> str:
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def request_key(body: Body) -> str:
    return hashlib.sha256(canonical(body).encode()).hexdigest()


class LLMCache:
    """One JSON file per request: ``root/<key[:2]>/<key>.json`` = {"request", "response"}."""

    def __init__(self, root: Path, *, offline: bool) -> None:
        self.root = root
        self.offline = offline

    def path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, body: Body) -> Response | None:
        path = self.path(request_key(body))
        if not path.is_file():
            return None
        return json.loads(path.read_text())["response"]

    def call(self, body: Body, fetch: Callable[[Body], Response]) -> Response:
        """The stored answer to ``body``; on a miss, ``fetch`` it and store it, or raise
        CacheMiss when offline."""
        key = request_key(body)
        stored = self.get(body)
        if stored is not None:
            return stored
        if self.offline:
            raise CacheMiss(f"no cached answer for request {key}")
        response = fetch(body)
        path = self.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Written whole, then renamed: a crash never leaves half an answer under the key.
        tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps({"request": body, "response": response}, ensure_ascii=False))
        tmp.replace(path)
        return response
