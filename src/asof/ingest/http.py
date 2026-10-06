"""Minimal HTTP seam shared by the ingest downloaders.

Every network call in ``asof.ingest`` goes through a ``Fetch`` callable so tests can script
replies instead of hitting the network. ``make_fetch`` builds the real one on top of httpx with
a single long-lived connection pool.
"""

from collections.abc import Callable, Mapping
from typing import NamedTuple

import httpx


class FetchResult(NamedTuple):
    status: int
    headers: Mapping[str, str]
    body: bytes


Fetch = Callable[[str, Mapping[str, str]], FetchResult]
"""``fetch(url, query_params) -> FetchResult``. Auth headers are the caller's business."""


def make_fetch(
    headers: Mapping[str, str] | None = None,
    *,
    timeout: float = 60.0,
    follow_redirects: bool = False,
) -> Fetch:
    """A ``Fetch`` that sends ``headers`` on every request and reuses one connection pool.

    Redirects are off by default: httpx keeps custom auth headers across hosts, so a redirect
    away from Alpaca would carry the API keys with it.
    """
    client = httpx.Client(
        headers=dict(headers or {}), timeout=timeout, follow_redirects=follow_redirects
    )

    def fetch(url: str, params: Mapping[str, str]) -> FetchResult:
        response = client.get(url, params=dict(params))
        return FetchResult(response.status_code, dict(response.headers), response.content)

    return fetch
