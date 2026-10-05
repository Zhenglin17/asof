"""Guards against duplicate identities: syncing the same instruments twice must not add rows."""

import pytest
from sqlalchemy import Engine
from sqlmodel import Session, col, select

from asof.store.entities import EntityConflict, EntitySpec, UpsertResult, upsert_entities
from asof.store.models import Entity
from tests.store.factories import APPLE_CIK, SECTOR_SPDR_CIK, XLE_SERIES, XLK_SERIES, pk

APPLE = EntitySpec(kind="company", ticker="AAPL", name="Apple Inc.", cik=APPLE_CIK)
XLK = EntitySpec(
    kind="etf", ticker="XLK", name="Technology", cik=SECTOR_SPDR_CIK, series_id=XLK_SERIES
)
XLE = EntitySpec(kind="etf", ticker="XLE", name="Energy", cik=SECTOR_SPDR_CIK, series_id=XLE_SERIES)
DRAM = EntitySpec(kind="etf", ticker="DRAM", name="Memory chips")
BTC = EntitySpec(kind="crypto", ticker="BTC", name="Bitcoin")
MIX = [APPLE, XLK, XLE, DRAM, BTC]


def all_entities(session: Session) -> list[Entity]:
    return list(session.exec(select(Entity).order_by(col(Entity.id))).all())


def as_spec(entity: Entity) -> EntitySpec:
    return EntitySpec(
        kind=entity.kind,
        ticker=entity.ticker,
        name=entity.name,
        cik=entity.cik,
        series_id=entity.series_id,
    )


def test_mixed_kinds_are_all_inserted(session: Session) -> None:
    result = upsert_entities(session, MIX)
    session.commit()

    assert result == UpsertResult(inserted=5, updated=0)
    assert [as_spec(e) for e in all_entities(session)] == MIX


def test_same_specs_twice_change_nothing(session: Session) -> None:
    upsert_entities(session, MIX)
    session.commit()
    ids = [pk(e) for e in all_entities(session)]

    result = upsert_entities(session, MIX)
    session.commit()

    assert result == UpsertResult(inserted=0, updated=0)
    assert [pk(e) for e in all_entities(session)] == ids


def test_empty_input_does_nothing(session: Session) -> None:
    assert upsert_entities(session, iter(())) == UpsertResult(inserted=0, updated=0)
    assert all_entities(session) == []


def test_company_with_new_ticker_updates_the_existing_row(session: Session) -> None:
    upsert_entities(session, [EntitySpec("company", "FB", "Facebook Inc", cik=1326801)])
    session.commit()
    original_id = pk(all_entities(session)[0])

    result = upsert_entities(
        session, [EntitySpec("company", "META", "Meta Platforms, Inc.", cik=1326801)]
    )
    session.commit()
    session.expire_all()

    assert result == UpsertResult(inserted=0, updated=1)
    (row,) = all_entities(session)
    assert pk(row) == original_id
    assert (row.ticker, row.name, row.cik) == ("META", "Meta Platforms, Inc.", 1326801)


def test_fund_is_matched_by_series_id_when_its_ticker_changes(session: Session) -> None:
    upsert_entities(session, [XLK, XLE])
    session.commit()

    renamed = EntitySpec("etf", "XLKK", "Technology", cik=SECTOR_SPDR_CIK, series_id=XLK_SERIES)
    result = upsert_entities(session, [renamed])
    session.commit()
    session.expire_all()

    assert result == UpsertResult(inserted=0, updated=1)
    assert [as_spec(e) for e in all_entities(session)] == [renamed, XLE]


def test_fund_without_series_is_matched_by_ticker_and_gains_its_cik(session: Session) -> None:
    upsert_entities(session, [EntitySpec("etf", "SPY", "S&P 500")])
    session.commit()

    listed = EntitySpec("etf", "SPY", "SPDR S&P 500 ETF TRUST", cik=884394)
    result = upsert_entities(session, [listed])
    session.commit()
    session.expire_all()

    assert result == UpsertResult(inserted=0, updated=1)
    assert [as_spec(e) for e in all_entities(session)] == [listed]


def test_same_ticker_under_another_kind_is_a_separate_row(session: Session) -> None:
    fund = EntitySpec("etf", "BTC", "Bitcoin Mini Trust")
    result = upsert_entities(session, [BTC, fund])
    session.commit()

    assert result == UpsertResult(inserted=2, updated=0)
    assert [as_spec(e) for e in all_entities(session)] == [BTC, fund]


def test_inserts_and_updates_are_counted_separately(session: Session) -> None:
    upsert_entities(session, [APPLE, BTC])
    session.commit()

    renamed = EntitySpec("crypto", "BTC", "Bitcoin (BTC)")
    result = upsert_entities(session, [APPLE, renamed, DRAM])
    session.commit()
    session.expire_all()

    assert result == UpsertResult(inserted=1, updated=1)
    assert [as_spec(e) for e in all_entities(session)] == [APPLE, renamed, DRAM]


def test_upsert_flushes_but_leaves_the_commit_to_the_caller(
    engine: Engine, session: Session
) -> None:
    upsert_entities(session, MIX)

    # Flushed: the open transaction already sees the rows and their generated ids.
    assert len(all_entities(session)) == 5
    session.rollback()

    with Session(engine) as other:
        assert all_entities(other) == []


def test_fund_stored_without_series_adopts_one_instead_of_duplicating(session: Session) -> None:
    upsert_entities(session, [EntitySpec("etf", "DRAM", "Memory chips")])
    session.commit()
    original_id = pk(all_entities(session)[0])

    listed = EntitySpec("etf", "DRAM", "Memory chips", cik=1234567, series_id="S000099999")
    result = upsert_entities(session, [listed])
    session.commit()
    session.expire_all()

    assert result == UpsertResult(inserted=0, updated=1)
    (row,) = all_entities(session)
    assert pk(row) == original_id
    assert as_spec(row) == listed


def test_stored_fund_is_never_moved_to_another_registrant(session: Session) -> None:
    upsert_entities(session, [EntitySpec("etf", "SPY", "SPDR S&P 500 ETF TRUST", cik=884394)])
    session.commit()

    with pytest.raises(EntityConflict, match="SPY"):
        upsert_entities(session, [EntitySpec("etf", "SPY", "Someone else", cik=999)])
    session.rollback()

    (row,) = all_entities(session)
    assert (row.name, row.cik) == ("SPDR S&P 500 ETF TRUST", 884394)


def test_fund_without_series_is_not_adopted_by_a_series_of_another_registrant(
    session: Session,
) -> None:
    upsert_entities(session, [EntitySpec("etf", "SPY", "SPDR S&P 500 ETF TRUST", cik=884394)])
    session.commit()

    other = EntitySpec("etf", "SPY", "Other", cik=999, series_id="S000000009")
    with pytest.raises(EntityConflict):
        upsert_entities(session, [other])


def test_two_tickers_of_one_company_in_a_batch_are_rejected(session: Session) -> None:
    googl = EntitySpec("company", "GOOGL", "Alphabet Inc.", cik=1652044)
    goog = EntitySpec("company", "GOOG", "Alphabet Inc.", cik=1652044)

    with pytest.raises(EntityConflict, match="GOOG"):
        upsert_entities(session, [googl, goog])
