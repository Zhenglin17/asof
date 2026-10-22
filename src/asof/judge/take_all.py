"""The take-all judge: every candidate becomes a long decision, in a fixed order.

It reads no evidence and makes no claim, so its decisions carry no citations; the filing that
triggered each one is kept in the decision payload. It is the baseline any evidence-reading
judge has to beat.
"""

from collections.abc import Sequence

from asof.scan.earnings_gap import Candidate
from asof.strategy import StrategyConfig


def judge(candidates: Sequence[Candidate], cfg: StrategyConfig) -> list[Candidate]:
    if cfg.judge.kind != "take_all":
        raise ValueError(f"judge kind {cfg.judge.kind!r} is not take_all")
    return sorted(candidates, key=lambda c: (c.security_id, c.entity_id))
