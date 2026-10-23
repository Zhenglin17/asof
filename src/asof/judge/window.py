"""Fill the answer cache for a replay window: the only place the llm judge spends money.

For each session in the window, at its 20:00 New York close, the scanner runs exactly as the
replay runs it and the judge answers every candidate, one per security in security_id order.
The window does not know the portfolio, so it also judges candidates a replay will skip because
the security is still held; that costs a few calls and lets one cache serve every replay of the
window. Answers go through ``LLMCache``: a cached one is free, a fresh one adds the
``usage.cost`` OpenRouter reports to the spend. Before each candidate the spend is checked
against ``max_cost``; a candidate already started finishes its retries. The cap is per call of
this function: cached answers are free, so a rerun pays only for what is still missing. A failed
call (network, provider error) ends the window with the error recorded.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import duckdb
from sqlmodel import Session

from asof.judge.cache import LLMCache
from asof.judge.client import JudgeCallError
from asof.judge.llm import JudgeOutcome, judge_candidate, one_per_security
from asof.replay.run import close_of
from asof.scan.earnings_gap import scan
from asof.store.market import visible_sessions
from asof.strategy import StrategyConfig

Fetch = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class WindowResult:
    outcomes: list[JudgeOutcome] = field(default_factory=list)
    fresh_calls: int = 0
    cached_calls: int = 0
    spent: float = 0.0
    # The cap ended the window before every candidate was judged.
    stopped: bool = False
    # A call failed (network, provider): the window ended there; what was paid for is cached.
    error: str | None = None


def judge_window(
    session: Session,
    con: duckdb.DuckDBPyConnection,
    root: Path,
    cfg: StrategyConfig,
    start: date,
    end: date,
    fetch: Fetch,
    *,
    max_cost: float,
    limit: int | None = None,
) -> WindowResult:
    if cfg.judge.kind != "llm":
        raise ValueError(f"judge_window needs judge.kind llm, got {cfg.judge.kind!r}")
    cache = LLMCache(root / "llm_cache", offline=False)
    result = WindowResult()

    def counted(body: dict[str, Any]) -> dict[str, Any]:
        reply = fetch(body)
        result.fresh_calls += 1
        cost = (reply.get("usage") or {}).get("cost")
        if cost is not None:
            result.spent += float(cost)
        return reply

    def call(body: dict[str, Any]) -> dict[str, Any]:
        if cache.get(body) is not None:
            result.cached_calls += 1
            return cache.call(body, fetch)
        spent_before = result.spent
        reply = cache.call(body, counted)  # stored before any check below: it was paid for
        if (reply.get("usage") or {}).get("cost") is None:
            raise ValueError(
                f"a fresh reply carried no usage.cost, so the cap cannot be enforced; "
                f"${spent_before:.4f} was spent before it"
            )
        return reply

    calendar = visible_sessions(con, root, "1Day", close_of(end))
    for index, day in enumerate(calendar):
        if not start <= day <= end:
            continue
        as_of = close_of(day)
        candidates = scan(session, con, root, as_of, cfg, sessions=calendar[: index + 1])
        for candidate in one_per_security(candidates):
            if limit is not None and len(result.outcomes) >= limit:
                return result
            if result.spent >= max_cost:
                result.stopped = True
                return result
            try:
                outcome = judge_candidate(session, candidate, as_of, cfg.judge, call)
            except JudgeCallError as error:
                result.error = str(error)
                return result
            result.outcomes.append(outcome)
    return result
