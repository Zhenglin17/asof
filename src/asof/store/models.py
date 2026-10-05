"""Metadata tables.

Two rules hold for every table:
- Every time column is timezone-aware UTC (see `UTCDateTime`).
- Enum values and "required when" rules are SQLite CHECK constraints, so a bad row is
  rejected by the database no matter which code path tried to write it. SQLModel table
  classes do not validate on construction, so Python type hints alone would not stop it.
"""

import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import JSON, CheckConstraint, DateTime, Dialect, UniqueConstraint
from sqlalchemy.orm import declared_attr
from sqlalchemy.types import TypeDecorator
from sqlmodel import Field, SQLModel


class _Table(SQLModel):
    """Base for every table: the table name is the class name in snake_case."""

    # SQLModel's own __tablename__ carries the same ignore; its stubs do not type-check here.
    @declared_attr  # pyright: ignore[reportArgumentType]
    def __tablename__(cls) -> str:  # pyright: ignore[reportIncompatibleVariableOverride]
        return re.sub(r"(?<!^)(?=[A-Z])", "_", cls.__name__).lower()


class UTCDateTime(TypeDecorator[datetime]):
    """Refuses naive datetimes on write, stores UTC, returns aware UTC on read.

    SQLite has no timezone type. Without this, "16:05 New York" and "16:05 UTC" would be
    stored as the same string and compared as equal, which is a look-ahead leak.
    """

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"datetime must carry a timezone, got naive {value!r}")
        return value.astimezone(UTC).replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=UTC)


def _one_of(column: str, enum: type[StrEnum]) -> CheckConstraint:
    # NULL passes any CHECK in SQL, so whether NULL is allowed is decided by the column itself.
    values = ", ".join(f"'{member.value}'" for member in enum)
    return CheckConstraint(f'"{column}" IN ({values})', name=f"ck_{column}_enum")


class DecisionKind(StrEnum):
    TRADE = "trade"
    OBSERVATION = "observation"


class Direction(StrEnum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class EvidenceKind(StrEnum):
    CHUNK = "chunk"
    FACT = "fact"
    BAR = "bar"
    OPTION_SNAPSHOT = "option_snapshot"
    MACRO_OBS = "macro_obs"


class FailureCategory(StrEnum):
    TEMPORAL_MISMATCH = "temporal_mismatch"
    ENTITY_MISMATCH = "entity_mismatch"
    COMPARISON_HALLUCINATION = "comparison_hallucination"
    OTHER = "other"


class RunMode(StrEnum):
    LIVE = "live"
    REPLAY = "replay"


class VersionStatus(StrEnum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"
    RETIRED = "retired"


class ProposedBy(StrEnum):
    AGENT = "agent"
    HUMAN = "human"


class ChangeKind(StrEnum):
    CONFIG = "config"
    RULE = "rule"
    FEATURE = "feature"
    POLICY = "policy"
    NEW_STRATEGY = "new_strategy"


class FeedbackTarget(StrEnum):
    DECISION = "decision"
    STRATEGY_VERSION = "strategy_version"
    NONE = "none"


class FeedbackStatus(StrEnum):
    OPEN = "open"
    TURNED_INTO_PROPOSAL = "turned_into_proposal"
    DISMISSED = "dismissed"


class Source(_Table, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(unique=True)
    base_url: str | None = None
    notes: str | None = None


class Entity(_Table, table=True):
    # SEC Central Index Key. An int, so "0000320193" and "320193" cannot become two rows.
    # Tickers change and get reused; the CIK does not.
    cik: int = Field(primary_key=True, sa_column_kwargs={"autoincrement": False})
    ticker: str | None = Field(default=None, index=True)
    name: str


class Document(_Table, table=True):
    __table_args__ = (UniqueConstraint("source_id", "external_id"),)

    id: int | None = Field(default=None, primary_key=True)
    source_id: int = Field(foreign_key="source.id", index=True)
    entity_id: int | None = Field(default=None, foreign_key="entity.cik", index=True)
    doc_type: str
    external_id: str
    title: str | None = None
    url: str | None = None
    content_hash: str
    # The period the content describes, e.g. quarter end for a 10-Q. Never used to filter.
    source_time: datetime = Field(sa_type=UTCDateTime)
    # When the public could first see it. The only column as_of filters on.
    available_at: datetime = Field(sa_type=UTCDateTime, index=True)
    # When we downloaded it. Audit only.
    received_at: datetime = Field(sa_type=UTCDateTime)
    # A revision is a new row pointing at the row it replaces; rows are never updated.
    supersedes_id: int | None = Field(default=None, foreign_key="document.id", index=True)


class Chunk(_Table, table=True):
    __table_args__ = (
        UniqueConstraint("document_id", "ordinal"),
        CheckConstraint("char_start >= 0 AND char_end >= char_start", name="ck_char_span"),
    )

    id: int | None = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="document.id", index=True)
    ordinal: int
    section_path: str
    char_start: int
    char_end: int
    text: str


class Event(_Table, table=True):
    id: int | None = Field(default=None, primary_key=True)
    entity_id: int | None = Field(default=None, foreign_key="entity.cik", index=True)
    event_type: str
    event_time: datetime = Field(sa_type=UTCDateTime)
    available_at: datetime = Field(sa_type=UTCDateTime, index=True)
    document_id: int | None = Field(default=None, foreign_key="document.id")
    payload: dict[str, Any] = Field(default_factory=dict, sa_type=JSON)


class Run(_Table, table=True):
    __table_args__ = (_one_of("mode", RunMode),)

    id: int | None = Field(default=None, primary_key=True)
    # Strategy directory name, e.g. "momentum". The exact config is strategy_version_id.
    strategy_id: str | None = Field(default=None, index=True)
    strategy_version_id: int | None = Field(default=None, foreign_key="strategy_version.id")
    mode: str
    # Simulated "now" for this run; equals wall clock only in live mode.
    as_of: datetime = Field(sa_type=UTCDateTime)
    started_at: datetime = Field(sa_type=UTCDateTime)
    finished_at: datetime | None = Field(default=None, sa_type=UTCDateTime)
    model_id: str | None = None
    model_training_cutoff: datetime | None = Field(default=None, sa_type=UTCDateTime)
    git_sha: str


class Decision(_Table, table=True):
    __table_args__ = (
        _one_of("kind", DecisionKind),
        _one_of("direction", Direction),
        CheckConstraint(
            f"kind != '{DecisionKind.TRADE}' OR "
            "(entity_id IS NOT NULL AND direction IS NOT NULL AND strategy_id IS NOT NULL)",
            name="ck_trade_fields",
        ),
        # An observation without a check cannot be judged later, so it is only commentary.
        CheckConstraint(
            f"kind != '{DecisionKind.OBSERVATION}' OR "
            '(statement IS NOT NULL AND "check" IS NOT NULL)',
            name="ck_observation_fields",
        ),
    )

    id: int | None = Field(default=None, primary_key=True)
    run_id: int = Field(foreign_key="run.id", index=True)
    kind: str
    as_of: datetime = Field(sa_type=UTCDateTime, index=True)
    strategy_id: str | None = Field(default=None, index=True)
    entity_id: int | None = Field(default=None, foreign_key="entity.cik", index=True)
    direction: str | None = None
    topic: str | None = None
    statement: str | None = None
    implication: str | None = None
    check: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict, sa_type=JSON)


class DecisionCitation(_Table, table=True):
    """One row = one claim backed by one piece of evidence."""

    __table_args__ = (
        _one_of("evidence_kind", EvidenceKind),
        # Text evidence goes through a real foreign key ...
        CheckConstraint(
            f"evidence_kind != '{EvidenceKind.CHUNK}' OR chunk_id IS NOT NULL",
            name="ck_chunk_requires_chunk_id",
        ),
        # ... everything else lives in Parquet, which SQLite cannot reference, so it carries a
        # locator string that the Citation Validator checks in code.
        CheckConstraint(
            f"evidence_kind = '{EvidenceKind.CHUNK}' "
            "OR (chunk_id IS NULL AND evidence_ref IS NOT NULL)",
            name="ck_non_chunk_requires_ref",
        ),
    )

    id: int | None = Field(default=None, primary_key=True)
    decision_id: int = Field(foreign_key="decision.id", index=True)
    claim: str
    evidence_kind: str
    evidence_ref: str | None = None
    chunk_id: int | None = Field(default=None, foreign_key="chunk.id", index=True)


class Attribution(_Table, table=True):
    __table_args__ = (_one_of("failure_category", FailureCategory),)

    id: int | None = Field(default=None, primary_key=True)
    decision_id: int = Field(foreign_key="decision.id", index=True)
    strategy_id: str | None = Field(default=None, index=True)
    # When the outcome became knowable. Hides results from anyone looking at an earlier as_of.
    outcome_available_at: datetime = Field(sa_type=UTCDateTime, index=True)
    realized_return: float | None = None
    is_correct: bool | None = None
    failure_category: str | None = None
    notes: str | None = None


class StrategyVersion(_Table, table=True):
    __table_args__ = (
        UniqueConstraint("strategy_id", "content_hash"),
        _one_of("status", VersionStatus),
        _one_of("proposed_by", ProposedBy),
        _one_of("change_kind", ChangeKind),
    )

    id: int | None = Field(default=None, primary_key=True)
    strategy_id: str = Field(index=True)
    # Hash over the whole strategies/<name>/ directory.
    content_hash: str
    parent_id: int | None = Field(default=None, foreign_key="strategy_version.id")
    status: str = VersionStatus.PROPOSED.value
    proposed_by: str
    registered_at: datetime = Field(sa_type=UTCDateTime, index=True)
    change_kind: str
    summary: str | None = None


class Feedback(_Table, table=True):
    __table_args__ = (
        _one_of("target_kind", FeedbackTarget),
        _one_of("status", FeedbackStatus),
        # No foreign key: target_id points into decision or strategy_version depending on kind.
        CheckConstraint(
            f"target_kind = '{FeedbackTarget.NONE}' OR target_id IS NOT NULL",
            name="ck_target_id",
        ),
    )

    id: int | None = Field(default=None, primary_key=True)
    target_kind: str
    target_id: int | None = None
    body: str
    status: str = FeedbackStatus.OPEN.value
    created_at: datetime = Field(sa_type=UTCDateTime)
