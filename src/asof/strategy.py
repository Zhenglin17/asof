"""A strategy is a directory of six YAML files: what to scan for, how to judge, how to trade.

The directory is the unit of versioning. ``content_hash`` identifies one exact parameter set (the
files plus any ``--set`` overrides), and ``register_version`` records each set as a
``StrategyVersion`` row once, so ``attempt_count`` -- the number of distinct sets ever tried --
is there for the attempts-adjusted gate (DECISIONS 2026-10-04). Unknown keys are errors: a typo
silently replaced by a default would run a strategy nobody wrote down.
"""

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from sqlmodel import Session, col, func, select

from asof.store.models import (
    ChangeKind,
    ProposedBy,
    Run,
    RunMode,
    StrategyVersion,
    VersionStatus,
)

FILES = ("strategy", "scan", "rules", "judge", "execute", "attribution")


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class StrategySection(_Section):
    name: str
    cadence: Literal["daily"]
    universe: Literal["watchlist_with_documents"]
    direction: Literal["long"]


class ScanConfig(_Section):
    min_gap_pct: float
    min_rvol: float
    rvol_days: int
    atr_days: int
    min_price: float
    min_avg_dollar_volume: float
    dollar_volume_days: int
    earnings_items: tuple[str, ...]


class RulesConfig(_Section):
    direction: Literal["follow_gap"]


_LLM_FIELDS = (
    "provider",
    "model",
    "pin_provider",
    "effort",
    "max_retries",
    "min_conviction",
    "max_tokens",
    "min_filing_chars",
    "training_cutoff",
    "cutoff_source",
    "prompt",
)


class JudgeConfig(_Section):
    """``take_all`` reads nothing; ``llm`` asks a model through OpenRouter and needs every field
    below. The model, its effort and the prompt are part of the parameter set, so changing any
    of them is a new StrategyVersion. ``training_cutoff`` is the model's documented knowledge
    cutoff: a replay before it is graded as possibly contaminated."""

    kind: Literal["take_all", "llm"]
    provider: Literal["openrouter"] | None = None
    model: str | None = None
    # The upstream provider OpenRouter must route to, with no fallback to another one.
    pin_provider: str | None = None
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    # Calls after the first, each carrying the validator's failures back to the model.
    max_retries: int | None = Field(default=None, ge=0)
    # A take below this conviction is a pass.
    min_conviction: float | None = Field(default=None, ge=0, le=1)
    max_tokens: int | None = Field(default=None, gt=0)
    # A filing with less visible text than this (an image-only exhibit) is a no_call without
    # asking: the model would have nothing to read, yet could cite a file name.
    min_filing_chars: int | None = Field(default=None, ge=0)
    training_cutoff: date | None = None
    cutoff_source: str | None = None
    prompt: str | None = None

    @model_validator(mode="after")
    def _llm_fields(self) -> "JudgeConfig":
        if self.kind != "llm":
            return self
        missing = [
            name
            for name in _LLM_FIELDS
            if getattr(self, name) is None
            or (isinstance(getattr(self, name), str) and not getattr(self, name).strip())
        ]
        if missing:
            raise ValueError(f"judge kind llm needs {', '.join(missing)}")
        return self


class EntryConfig(_Section):
    kind: Literal["next_open"]
    key: Literal["gap_day_low"]


class StopConfig(_Section):
    kind: Literal["atr_zone_stop"]
    zone_atr: float
    zone_cap_pct: float
    hard_atr: float
    observe_days: int
    trail_atr: float

    @model_validator(mode="after")
    def _hard_below_zone(self) -> "StopConfig":
        # The hard stop must sit below the zone, or the R unit (entry - hard) can reach zero.
        if self.hard_atr <= self.zone_atr:
            raise ValueError("hard_atr must be greater than zone_atr")
        return self


class ReentryConfig(_Section):
    key: Literal["pre_earnings_close"]
    window_sessions: int
    # Leg numbers are 1 and 2 in the trade table, so at most one re-entry.
    max_count: int = Field(ge=0, le=1)
    risk_mode: Literal["shared", "independent"]


class ExitConfig(_Section):
    max_sessions: int


class SizingConfig(_Section):
    kind: Literal["risk_based"]
    risk_pct: float
    max_position_pct: float
    max_gross_pct: float


class FillsConfig(_Section):
    slippage_bps: float


class AccountConfig(_Section):
    initial_equity: float


class DataConfig(_Section):
    max_missing_sessions: int
    # An open above this multiple of the last close with no split on record is taken for an
    # unrecorded reverse split: the leg leaves at the last close. Drops are left to the stops.
    max_jump_ratio: float
    # Sessions a leg held through a spin-off's ex-date (session 0) waits frozen for the event's
    # factor; still undecided at that close, it resumes on raw prices without the child.
    spinoff_max_wait: int


class ExecuteConfig(_Section):
    entry: EntryConfig
    stop: StopConfig
    reentry: ReentryConfig
    exit: ExitConfig
    sizing: SizingConfig
    fills: FillsConfig
    account: AccountConfig
    data: DataConfig


class AttributionConfig(_Section):
    metrics: tuple[str, ...]


class StrategyConfig(_Section):
    strategy: StrategySection
    scan: ScanConfig
    rules: RulesConfig
    judge: JudgeConfig
    execute: ExecuteConfig
    attribution: AttributionConfig


def _validate(raw: dict[str, Any]) -> StrategyConfig:
    try:
        return StrategyConfig.model_validate(raw)
    except ValidationError as error:
        raise ValueError(str(error)) from error


def load_strategy(path: Path) -> StrategyConfig:
    raw: dict[str, Any] = {}
    for name in FILES:
        file = path / f"{name}.yaml"
        if not file.is_file():
            raise FileNotFoundError(f"{file}: missing strategy file")
        raw[name] = yaml.safe_load(file.read_text()) or {}
    cfg = _validate(raw)
    # The name keys attempt counts and the holdout opening; it must not drift from the directory.
    if cfg.strategy.name != path.name:
        raise ValueError(f"{path}: strategy.name {cfg.strategy.name!r} is not the directory name")
    return cfg


def _parse_override(override: str) -> tuple[list[str], Any]:
    dotted, sep, value = override.partition("=")
    keys = dotted.split(".")
    if not sep or not dotted or any(not key for key in keys):
        raise ValueError(f"{override!r}: expected section.key=value")
    return keys, yaml.safe_load(value)


def apply_overrides(cfg: StrategyConfig, overrides: Sequence[str]) -> StrategyConfig:
    """A new config with each ``a.b.c=value`` applied; the value is read as YAML, then typed by
    the model. An unknown path or a value of the wrong type raises ValueError."""
    raw = cfg.model_dump(mode="json")
    for override in overrides:
        keys, value = _parse_override(override)
        if keys[0] == "strategy":
            # strategy.name keys attempts and the holdout; a new identity is a new directory.
            raise ValueError(f"{override!r}: the strategy section cannot be overridden")
        node = raw
        for key in keys[:-1]:
            if not isinstance(node.get(key), dict):
                raise ValueError(f"{override!r}: no section {key!r}")
            node = node[key]
        if keys[-1] not in node or isinstance(node[keys[-1]], dict):
            raise ValueError(f"{override!r}: no setting {keys[-1]!r}")
        node[keys[-1]] = value
    return _validate(raw)


def _canonical_overrides(path: Path, overrides: Sequence[str]) -> list[tuple[str, Any]]:
    """Each overridden path with its typed value, sorted: "=1" and "=1.0" on a float setting, or
    the same set in another order, are one parameter set."""
    cfg = apply_overrides(load_strategy(path), overrides)
    raw = cfg.model_dump(mode="json")
    found: dict[str, Any] = {}
    for override in overrides:
        keys, _ = _parse_override(override)
        node: Any = raw
        for key in keys:
            node = node[key]
        found[".".join(keys)] = node
    return sorted(found.items())


def content_hash(path: Path, overrides: Sequence[str] = ()) -> str:
    """SHA-256 over the six files' bytes and the canonical overrides. Independent of where the
    directory lives and of file times."""
    digest = hashlib.sha256()
    for name in FILES:
        digest.update(name.encode() + b"\0")
        digest.update((path / f"{name}.yaml").read_bytes() + b"\0")
    if overrides:
        canonical = _canonical_overrides(path, overrides)
        digest.update(json.dumps(canonical, sort_keys=True).encode())
    return digest.hexdigest()


def _find(session: Session, strategy_id: str, digest: str) -> StrategyVersion | None:
    stmt = select(StrategyVersion).where(
        StrategyVersion.strategy_id == strategy_id, StrategyVersion.content_hash == digest
    )
    return session.exec(stmt).first()


def register_version(
    session: Session,
    path: Path,
    overrides: Sequence[str] = (),
    proposed_by: str = ProposedBy.HUMAN.value,
) -> StrategyVersion:
    """The StrategyVersion row of this parameter set, created on first use. The committed
    directory is the approved base; any override set is a proposed child of it."""
    cfg = apply_overrides(load_strategy(path), overrides)  # raises before anything is written
    strategy_id = cfg.strategy.name
    digest = content_hash(path, overrides)
    existing = _find(session, strategy_id, digest)
    if existing is not None:
        return existing
    if overrides:
        parent = register_version(session, path, (), proposed_by)
        row = StrategyVersion(
            strategy_id=strategy_id,
            content_hash=digest,
            parent_id=parent.id,
            status=VersionStatus.PROPOSED.value,
            proposed_by=proposed_by,
            registered_at=datetime.now(UTC),
            change_kind=ChangeKind.CONFIG.value,
            summary=", ".join(f"{k}={v}" for k, v in _canonical_overrides(path, overrides)),
        )
    else:
        # Only a strategy's first version is approved by registering it; an edited directory
        # is a proposal until a person approves it.
        approved = session.exec(
            select(StrategyVersion)
            .where(
                StrategyVersion.strategy_id == strategy_id,
                StrategyVersion.status == VersionStatus.APPROVED.value,
            )
            .order_by(col(StrategyVersion.id).desc())
        ).first()
        row = StrategyVersion(
            strategy_id=strategy_id,
            content_hash=digest,
            parent_id=approved.id if approved else None,
            status=(VersionStatus.PROPOSED if approved else VersionStatus.APPROVED).value,
            proposed_by=proposed_by,
            registered_at=datetime.now(UTC),
            change_kind=(ChangeKind.CONFIG if approved else ChangeKind.NEW_STRATEGY).value,
        )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def replay_count(session: Session, strategy_id: str) -> int:
    """Replay runs of the strategy, any parameter set and window. Reported next to
    ``attempt_count``: a re-run on another window is a search too, even with the same set."""
    stmt = (
        select(func.count())
        .select_from(Run)
        .where(col(Run.strategy_id) == strategy_id, col(Run.mode) == RunMode.REPLAY.value)
    )
    return int(session.exec(stmt).one())


def attempt_count(session: Session, strategy_id: str) -> int:
    stmt = (
        select(func.count())
        .select_from(StrategyVersion)
        .where(col(StrategyVersion.strategy_id) == strategy_id)
    )
    return int(session.exec(stmt).one())
