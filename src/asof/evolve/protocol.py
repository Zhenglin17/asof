"""The pre-registered evaluation protocol (configs/evaluation.yaml): which dates are for
development, which are sealed, and the gate's settings. Loaded strictly: a typo in a key must
fail, not leave a window unguarded with a default."""

from datetime import date
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

# src/asof/evolve/protocol.py -> repo root, so the path follows the code, not the shell.
DEFAULT_PROTOCOL_PATH = Path(__file__).resolve().parents[3] / "configs" / "evaluation.yaml"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Window(_Strict):
    start: date
    end: date

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.end < self.start:
            raise ValueError(f"window ends {self.end} before it starts {self.start}")
        return self


class ThinEdge(_Strict):
    drop_top: int = Field(ge=0)
    extra_cost_bps: float = Field(ge=0)


class Basket(_Strict):
    size: int = Field(ge=1)
    lookback_sessions: int = Field(ge=1)
    rebalance: Literal["monthly"]


class Benchmarks(_Strict):
    symbols: list[str] = Field(min_length=1)  # the first is the SPY of the comparison
    basket: Basket


class Comparison(_Strict):
    risk_free_rate: float = Field(ge=0)  # a year; replays earn nothing on idle cash
    high_exposure: float = Field(gt=0, le=1)  # report the share of sessions at least this invested


class WalkForward(_Strict):
    train_months: int = Field(ge=1)  # pick the best variant over this many months
    test_months: int = Field(ge=1)  # then score the pick over the next this many
    min_positive: int = Field(ge=1)  # test windows whose pick beat random entries


class EvaluationProtocol(_Strict):
    dev: Window
    holdout: Window
    clean_after: date
    seen_before_protocol: list[str]
    attempt_budget: int = Field(ge=1)
    alpha: float = Field(gt=0, lt=1)
    permutations: int = Field(ge=100)
    thin_edge: ThinEdge
    benchmarks: Benchmarks
    comparison: Comparison
    walk_forward: WalkForward
    mde_power: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def _in_order(self) -> Self:
        if self.dev.end >= self.holdout.start:
            raise ValueError(f"dev ends {self.dev.end}, not before the holdout starts")
        if self.clean_after <= self.holdout.end:
            raise ValueError(f"clean_after {self.clean_after} is not after the holdout")
        return self

    def from_date(self, start: date) -> "EvaluationProtocol":
        """The same protocol with development dates from ``start`` on: for comparing variants
        that cannot trade before it. The end of dev, and so the holdout, never moves."""
        if not self.dev.start <= start <= self.dev.end:
            raise ValueError(f"{start} is outside the development dates {self.dev.start} .. "
                             f"{self.dev.end}")  # fmt: skip
        return self.model_copy(update={"dev": Window(start=start, end=self.dev.end)})


def load_protocol(path: Path = DEFAULT_PROTOCOL_PATH) -> EvaluationProtocol:
    return EvaluationProtocol.model_validate(yaml.safe_load(path.read_text()))
