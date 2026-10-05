"""Guards against invalid rows: dangling references, unknown enum values, duplicate companies."""

from typing import Any

import pytest
from sqlalchemy import Engine, inspect
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session

from asof.store.models import DecisionCitation, Entity, Source
from tests.store.factories import (
    T0,
    add,
    new_attribution,
    new_chunk,
    new_decision,
    new_document,
    new_entity,
    new_run,
    new_trade_decision,
    pk,
)

EXPECTED_TABLES = {
    "source",
    "entity",
    "document",
    "chunk",
    "event",
    "run",
    "decision",
    "decision_citation",
    "attribution",
    "strategy_version",
    "feedback",
}


def test_init_db_creates_exactly_the_eleven_tables(engine: Engine) -> None:
    assert set(inspect(engine).get_table_names()) == EXPECTED_TABLES


def test_foreign_keys_are_enforced(session: Session, apple: Entity) -> None:
    with pytest.raises(IntegrityError):
        new_document(session, source_id=999, available_at=T0)


def test_citation_to_missing_chunk_is_rejected(session: Session, apple: Entity) -> None:
    decision = new_trade_decision(session, run_id=pk(new_run(session)))
    citation = DecisionCitation(
        decision_id=pk(decision),
        claim="Revenue grew 5% year over year.",
        evidence_kind="chunk",
        chunk_id=999,
    )
    with pytest.raises(IntegrityError):
        add(session, citation)


def test_chunk_citation_without_chunk_id_is_rejected(session: Session, apple: Entity) -> None:
    decision = new_trade_decision(session, run_id=pk(new_run(session)))
    citation = DecisionCitation(
        decision_id=pk(decision),
        claim="Revenue grew 5% year over year.",
        evidence_kind="chunk",
    )
    with pytest.raises(IntegrityError):
        add(session, citation)


def test_citation_to_existing_chunk_is_accepted(
    session: Session, source: Source, apple: Entity
) -> None:
    doc = new_document(session, source_id=pk(source), available_at=T0)
    chunk = new_chunk(session, document_id=pk(doc))
    decision = new_trade_decision(session, run_id=pk(new_run(session)))
    add(
        session,
        DecisionCitation(
            decision_id=pk(decision),
            claim="Revenue grew 5% year over year.",
            evidence_kind="chunk",
            chunk_id=pk(chunk),
        ),
    )


@pytest.mark.parametrize("table", ["decision", "decision_citation", "attribution"])
def test_values_outside_enum_are_rejected(session: Session, apple: Entity, table: str) -> None:
    run_id = pk(new_run(session))
    with pytest.raises(IntegrityError):
        if table == "decision":
            new_decision(session, run_id=run_id, kind="opinion", statement="s", check="c")
        elif table == "decision_citation":
            decision = new_trade_decision(session, run_id=run_id)
            add(
                session,
                DecisionCitation(
                    decision_id=pk(decision),
                    claim="Someone said so.",
                    evidence_kind="tweet",
                    evidence_ref="x.com/123",
                ),
            )
        else:
            decision = new_trade_decision(session, run_id=run_id)
            new_attribution(
                session,
                decision_id=pk(decision),
                outcome_available_at=T0,
                failure_category="bad_luck",
            )


def test_duplicate_cik_is_rejected(engine: Engine) -> None:
    with Session(engine) as s:
        new_entity(s, cik=320193, ticker="AAPL")
    with Session(engine) as s, pytest.raises(IntegrityError):
        new_entity(s, cik=320193, ticker="AAPL")


@pytest.mark.parametrize(
    "fields",
    [
        # A trade must name a company.
        {"kind": "trade", "strategy_id": "momentum", "direction": "long"},
        # An observation without a check cannot be judged right or wrong later.
        {"kind": "observation", "topic": "energy flows", "statement": "Money flows into XLE."},
    ],
    ids=["trade_without_entity", "observation_without_check"],
)
def test_decision_kind_rules_reject_incomplete_rows(
    session: Session, apple: Entity, fields: dict[str, Any]
) -> None:
    with pytest.raises(IntegrityError):
        new_decision(session, run_id=pk(new_run(session)), **fields)


def test_observation_without_entity_is_accepted(session: Session) -> None:
    decision = new_decision(
        session,
        run_id=pk(new_run(session)),
        kind="observation",
        topic="energy flows",
        statement="Money is flowing into energy names.",
        implication="Momentum strategies may beat value strategies for now.",
        check="In two weeks, XLE relative to SPY is higher than today.",
    )
    assert decision.entity_id is None
    assert decision.direction is None
