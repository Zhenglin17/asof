"""Writing instruments into the entity table without creating duplicates."""

from collections.abc import Iterable
from dataclasses import dataclass

from sqlmodel import Session, col, select

from asof.store.models import Entity, EntityKind


@dataclass(frozen=True)
class EntitySpec:
    kind: str
    ticker: str
    name: str
    cik: int | None = None
    series_id: str | None = None


@dataclass(frozen=True)
class UpsertResult:
    inserted: int
    updated: int


class EntityConflict(ValueError):
    """Writing the spec would turn a stored instrument into a different one."""


def _find(session: Session, spec: EntitySpec) -> Entity | None:
    """Look the instrument up by the strongest identifier its kind has."""
    if spec.kind == EntityKind.COMPANY:
        by_cik = select(Entity).where(Entity.kind == spec.kind, Entity.cik == spec.cik)
        return session.exec(by_cik).first()

    # Several series may trade under one ticker over time; prefer the row without one.
    by_ticker = (
        select(Entity)
        .where(Entity.kind == spec.kind, Entity.ticker == spec.ticker)
        .order_by(col(Entity.series_id).is_not(None), col(Entity.id))
    )
    if spec.series_id is None:
        return session.exec(by_ticker).first()

    found = session.exec(select(Entity).where(Entity.series_id == spec.series_id)).first()
    if found is not None:
        return found
    # A fund stored before the SEC listed its series: adopt that row instead of adding a twin.
    return session.exec(by_ticker.where(col(Entity.series_id).is_(None))).first()


def _apply(row: Entity, spec: EntitySpec) -> bool:
    """Copy the spec onto the row. A missing identifier in the spec never erases a stored one."""
    wanted: dict[str, object] = {"ticker": spec.ticker, "name": spec.name}
    if spec.kind != EntityKind.COMPANY:
        if spec.cik is not None:
            wanted["cik"] = spec.cik
        if spec.series_id is not None:
            wanted["series_id"] = spec.series_id

    changed = False
    for field, value in wanted.items():
        if getattr(row, field) != value:
            setattr(row, field, value)
            changed = True
    return changed


def upsert_entities(session: Session, specs: Iterable[EntitySpec]) -> UpsertResult:
    """Insert new instruments and refresh known ones. Flushes; the caller commits."""
    inserted = updated = 0
    written: dict[int, EntitySpec] = {}
    for spec in specs:
        row = _find(session, spec)
        if row is not None:
            if row.id in written:
                raise EntityConflict(
                    f"{written[row.id].ticker} and {spec.ticker} are the same instrument"
                )
            if None not in (row.cik, spec.cik) and row.cik != spec.cik:
                raise EntityConflict(
                    f"{spec.ticker} is stored under CIK {row.cik}, not CIK {spec.cik}"
                )
        if row is None:
            row = Entity(
                kind=spec.kind,
                ticker=spec.ticker,
                name=spec.name,
                cik=spec.cik,
                series_id=spec.series_id,
            )
            session.add(row)
            inserted += 1
        elif _apply(row, spec):
            session.add(row)
            updated += 1
        # Later specs in the same batch must see this one.
        session.flush()
        assert row.id is not None
        written[row.id] = spec
    return UpsertResult(inserted=inserted, updated=updated)
