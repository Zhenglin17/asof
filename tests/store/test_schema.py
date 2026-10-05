"""Guards against invalid rows: dangling references, unknown enum values, duplicate entities."""

from typing import Any

import pytest
from sqlalchemy import Engine, inspect
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session

from asof.store.models import Decision, DecisionCitation, Entity, EntityKind, Event, Source
from tests.store.factories import (
    APPLE_CIK,
    SECTOR_SPDR_CIK,
    T0,
    XLE_SERIES,
    XLK_SERIES,
    add,
    new_attribution,
    new_chunk,
    new_decision,
    new_document,
    new_entity,
    new_observation,
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
        new_document(session, source_id=999, entity_id=pk(apple), available_at=T0)


def test_citation_to_missing_chunk_is_rejected(session: Session, apple: Entity) -> None:
    decision = new_trade_decision(session, run_id=pk(new_run(session)), entity_id=pk(apple))
    citation = DecisionCitation(
        decision_id=pk(decision),
        claim="Revenue grew 5% year over year.",
        evidence_kind="chunk",
        chunk_id=999,
    )
    with pytest.raises(IntegrityError):
        add(session, citation)


def test_chunk_citation_without_chunk_id_is_rejected(session: Session, apple: Entity) -> None:
    decision = new_trade_decision(session, run_id=pk(new_run(session)), entity_id=pk(apple))
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
    doc = new_document(session, source_id=pk(source), entity_id=pk(apple), available_at=T0)
    chunk = new_chunk(session, document_id=pk(doc))
    decision = new_trade_decision(session, run_id=pk(new_run(session)), entity_id=pk(apple))
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
            decision = new_trade_decision(session, run_id=run_id, entity_id=pk(apple))
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
            decision = new_trade_decision(session, run_id=run_id, entity_id=pk(apple))
            new_attribution(
                session,
                decision_id=pk(decision),
                outcome_available_at=T0,
                failure_category="bad_luck",
            )


def test_duplicate_company_cik_is_rejected(engine: Engine) -> None:
    with Session(engine) as s:
        new_entity(s, cik=320193, ticker="AAPL")
    with Session(engine) as s, pytest.raises(IntegrityError):
        new_entity(s, cik=320193, ticker="AAPL")
    # A different ticker does not make it a different company.
    with Session(engine) as s, pytest.raises(IntegrityError):
        new_entity(s, cik=320193, ticker="APPL")


def test_entity_kind_values() -> None:
    assert {kind.value for kind in EntityKind} == {"company", "etf", "crypto"}


def test_entity_key_is_a_generated_id_not_the_cik(session: Session) -> None:
    apple = new_entity(session)
    micron = new_entity(session, cik=723125, ticker="MU", name="MICRON TECHNOLOGY INC")

    assert pk(apple) != pk(micron)
    assert APPLE_CIK not in {pk(apple), pk(micron)}
    session.expire_all()
    stored = session.get(Entity, pk(micron))
    assert stored is not None
    assert (stored.kind, stored.ticker, stored.cik, stored.series_id) == (
        "company",
        "MU",
        723125,
        None,
    )


def test_unknown_entity_kind_is_rejected(session: Session) -> None:
    with pytest.raises(IntegrityError):
        new_entity(session, cik=None, ticker="EURUSD", kind="forex", name="Euro / US dollar")


def test_company_without_cik_is_rejected(session: Session) -> None:
    with pytest.raises(IntegrityError):
        new_entity(session, cik=None)


@pytest.mark.parametrize("missing", ["ticker", "name"])
def test_entity_requires_ticker_and_name(session: Session, missing: str) -> None:
    fields: dict[str, Any] = {"kind": "crypto", "ticker": "BTC", "name": "Bitcoin"}
    del fields[missing]
    with pytest.raises(IntegrityError):
        add(session, Entity(**fields))


def test_funds_sharing_a_cik_with_different_series_are_accepted(session: Session) -> None:
    xlk = new_entity(
        session, SECTOR_SPDR_CIK, "XLK", "etf", name="Technology", series_id=XLK_SERIES
    )
    xle = new_entity(session, SECTOR_SPDR_CIK, "XLE", "etf", name="Energy", series_id=XLE_SERIES)

    assert pk(xlk) != pk(xle)
    assert xlk.cik == xle.cik == SECTOR_SPDR_CIK


def test_duplicate_series_id_is_rejected(session: Session) -> None:
    new_entity(session, SECTOR_SPDR_CIK, "XLK", "etf", name="Technology", series_id=XLK_SERIES)
    with pytest.raises(IntegrityError):
        new_entity(session, SECTOR_SPDR_CIK, "XLE", "etf", name="Energy", series_id=XLK_SERIES)


def test_funds_without_cik_or_series_are_accepted(session: Session) -> None:
    # Two NULL series ids must not count as the same series.
    first = new_entity(session, None, "DRAM", "etf", name="Memory chips")
    second = new_entity(session, None, "UFO", "etf", name="Space")

    assert pk(first) != pk(second)
    assert (first.cik, first.series_id) == (None, None)


def test_crypto_without_cik_is_accepted(session: Session) -> None:
    btc = new_entity(session, None, "BTC", "crypto", name="Bitcoin")
    assert btc.cik is None


@pytest.mark.parametrize("kind", ["crypto", "etf"])
def test_duplicate_ticker_without_series_is_rejected(session: Session, kind: str) -> None:
    new_entity(session, None, "BTC", kind, name="Bitcoin")
    with pytest.raises(IntegrityError):
        new_entity(session, None, "BTC", kind, name="Bitcoin again")


def test_same_ticker_under_different_kinds_is_accepted(session: Session) -> None:
    coin = new_entity(session, None, "BTC", "crypto", name="Bitcoin")
    fund = new_entity(session, None, "BTC", "etf", name="Bitcoin Mini Trust")
    assert pk(coin) != pk(fund)


@pytest.mark.parametrize("table", ["document", "event", "decision"])
def test_entity_reference_must_be_an_entity_id_not_a_cik(
    session: Session, source: Source, apple: Entity, table: str
) -> None:
    run_id = pk(new_run(session))
    assert pk(apple) != APPLE_CIK
    with pytest.raises(IntegrityError):
        if table == "document":
            new_document(session, source_id=pk(source), entity_id=APPLE_CIK, available_at=T0)
        elif table == "event":
            add(
                session,
                Event(entity_id=APPLE_CIK, event_type="earnings", event_time=T0, available_at=T0),
            )
        else:
            new_trade_decision(session, run_id=run_id, entity_id=APPLE_CIK)


def test_rows_referencing_an_entity_id_are_accepted(
    session: Session, source: Source, apple: Entity
) -> None:
    doc = new_document(session, source_id=pk(source), entity_id=pk(apple), available_at=T0)
    event = add(
        session,
        Event(entity_id=pk(apple), event_type="earnings", event_time=T0, available_at=T0),
    )
    decision = new_trade_decision(session, run_id=pk(new_run(session)), entity_id=pk(apple))

    assert doc.entity_id == event.entity_id == decision.entity_id == pk(apple)


@pytest.mark.parametrize(
    "fields",
    [
        # A trade must name an instrument.
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


@pytest.mark.parametrize("conviction", [None, 0.0, 0.7, 1.0])
def test_conviction_within_unit_interval_round_trips(
    session: Session, conviction: float | None
) -> None:
    decision = new_observation(session, run_id=pk(new_run(session)), conviction=conviction)
    session.expire_all()
    stored = session.get(Decision, pk(decision))
    assert stored is not None
    assert stored.conviction == conviction


def test_conviction_defaults_to_null(session: Session) -> None:
    decision = new_observation(session, run_id=pk(new_run(session)))
    session.expire_all()
    stored = session.get(Decision, pk(decision))
    assert stored is not None
    assert stored.conviction is None


@pytest.mark.parametrize("conviction", [-0.1, 1.5])
def test_conviction_outside_unit_interval_is_rejected(session: Session, conviction: float) -> None:
    with pytest.raises(IntegrityError):
        new_observation(session, run_id=pk(new_run(session)), conviction=conviction)


def test_series_id_is_only_for_funds(session: Session) -> None:
    with pytest.raises(IntegrityError):
        new_entity(session, APPLE_CIK, "AAPL", "company", series_id="S000000001")


def test_crypto_with_a_cik_is_rejected(session: Session) -> None:
    with pytest.raises(IntegrityError):
        new_entity(session, 2015034, "BTC", "crypto", name="Bitcoin")
