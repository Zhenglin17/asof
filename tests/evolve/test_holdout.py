"""Guards against a run reaching the sealed holdout by accident: an overlap missed at a window
edge, a refusal that does not say how to open it, a strategy that opens the holdout twice (the
second look would be one more unrecorded attempt), an exempt strategy that leaves an opening row
behind, and an opening row with a naive or local timestamp.

API under test (asof.evolve.holdout):

    class HoldoutError(ValueError)
    overlaps_holdout(protocol, start, end) -> bool          inclusive on both ends
    check_window(session, protocol, strategy_id, start, end, *, open_holdout) -> bool
        no overlap -> False; strategy in seen_before_protocol -> False;
        overlap without open_holdout -> HoldoutError naming the dates and "--open-holdout";
        overlap, open_holdout, a HoldoutOpening for strategy_id exists -> HoldoutError;
        overlap, open_holdout, no row -> True (the caller must record the opening)
    record_opening(session, strategy_id, strategy_version_id, start, end) -> HoldoutOpening
        opened_at = now UTC; flushed

    asof.store.models.HoldoutOpening (table holdout_opening): strategy_id unique.

Dates are the real protocol's (tests/evolve/protocols.py). These functions read no market data,
so the "future data present but invisible" case lives with the callers: see
tests/replay/test_replay_holdout.py, where a refused replay must not read a single bar.
"""

from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy import Engine, inspect
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from asof.evolve.holdout import HoldoutError, check_window, overlaps_holdout, record_opening
from asof.store.db import init_db
from asof.store.models import HoldoutOpening, Source, StrategyVersion
from tests.evolve.protocols import protocol
from tests.store.factories import add, new_source, pk

PROTOCOL = protocol()  # holdout 2025-07-01 .. 2026-06-30, earnings_gap exempt
NEW = "gap_v2"  # a strategy that never saw the holdout


def version(session: Session, strategy_id: str = NEW, digest: str = "h1") -> StrategyVersion:
    return add(
        session,
        StrategyVersion(
            strategy_id=strategy_id,
            content_hash=digest,
            proposed_by="human",
            registered_at=datetime.now(UTC),
            change_kind="new_strategy",
        ),
    )


def openings(session: Session) -> list[HoldoutOpening]:
    return list(session.exec(select(HoldoutOpening)).all())


# =============================================================================================
# overlaps_holdout: the edges, with the real protocol dates
# =============================================================================================


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        (date(2025, 1, 1), date(2025, 7, 1), True),  # ends on the first holdout day
        (date(2026, 6, 30), date(2026, 10, 16), True),  # starts on the last holdout day
        (date(2026, 7, 1), date(2026, 10, 16), False),  # the clean period
        (date(2020, 1, 1), date(2025, 6, 30), False),  # exactly dev
        (date(2025, 1, 1), date(2026, 10, 1), True),  # spans the whole holdout
        (date(2025, 9, 1), date(2025, 9, 30), True),  # inside it
        (date(2025, 6, 30), date(2025, 6, 30), False),  # one day, last dev day
        (date(2025, 7, 1), date(2025, 7, 1), True),  # one day, first holdout day
        (date(2026, 7, 1), date(2026, 7, 1), False),  # one day, first clean day
    ],
    ids=[
        "ends_2025_07_01",
        "starts_2026_06_30",
        "clean_period",
        "whole_dev",
        "spans_holdout",
        "inside",
        "last_dev_day",
        "first_holdout_day",
        "first_clean_day",
    ],
)
def test_overlap_is_inclusive_at_both_ends(start: date, end: date, expected: bool) -> None:
    assert overlaps_holdout(PROTOCOL, start, end) is expected


# =============================================================================================
# check_window: normal path
# =============================================================================================


def test_a_dev_window_is_allowed_and_needs_no_record(session: Session) -> None:
    assert (
        check_window(
            session, PROTOCOL, NEW, date(2023, 1, 3), date(2025, 6, 30), open_holdout=False
        )
        is False
    )
    assert openings(session) == []


def test_a_clean_window_is_allowed_even_with_the_flag(session: Session) -> None:
    # The flag alone does not make a run an opening: nothing to record outside the holdout.
    assert (
        check_window(
            session, PROTOCOL, NEW, date(2026, 7, 1), date(2026, 10, 16), open_holdout=True
        )
        is False
    )


def test_opening_the_holdout_once_returns_true_then_records(session: Session) -> None:
    v = version(session)
    start, end = date(2025, 7, 1), date(2026, 6, 30)

    assert check_window(session, PROTOCOL, NEW, start, end, open_holdout=True) is True
    row = record_opening(session, NEW, pk(v), start, end)

    assert row.id is not None  # flushed
    session.commit()
    session.expire_all()
    [stored] = openings(session)
    assert (stored.strategy_id, stored.strategy_version_id) == (NEW, v.id)
    assert (stored.window_start, stored.window_end) == (start, end)
    assert stored.opened_at.tzinfo is not None
    assert stored.opened_at.utcoffset() == timedelta(0)
    assert abs(datetime.now(UTC) - stored.opened_at) < timedelta(minutes=5)


# =============================================================================================
# check_window: refusals
# =============================================================================================


def test_overlap_without_the_flag_is_refused_with_a_readable_message(session: Session) -> None:
    with pytest.raises(HoldoutError) as caught:
        check_window(session, PROTOCOL, NEW, date(2025, 1, 1), date(2025, 7, 1), open_holdout=False)

    message = str(caught.value)
    assert "2025-07-01" in message
    assert "2026-06-30" in message
    assert "--open-holdout" in message
    assert isinstance(caught.value, ValueError)  # existing `except ValueError` paths still catch it
    assert openings(session) == []


def test_a_second_opening_for_the_same_strategy_is_refused_even_with_another_version(
    session: Session,
) -> None:
    first = version(session, digest="h1")
    version(session, digest="h2")  # an edited parameter set of the same strategy
    record_opening(session, NEW, pk(first), date(2025, 7, 1), date(2026, 6, 30))
    session.commit()

    with pytest.raises(HoldoutError):
        check_window(session, PROTOCOL, NEW, date(2025, 7, 1), date(2026, 6, 30), open_holdout=True)
    # A smaller window inside the holdout is still a second look.
    with pytest.raises(HoldoutError):
        check_window(session, PROTOCOL, NEW, date(2026, 1, 2), date(2026, 1, 30), open_holdout=True)
    assert len(openings(session)) == 1


def test_another_strategys_opening_does_not_block_this_one(session: Session) -> None:
    other = version(session, strategy_id="momentum", digest="m1")
    record_opening(session, "momentum", pk(other), date(2025, 7, 1), date(2026, 6, 30))
    session.commit()

    assert (
        check_window(session, PROTOCOL, NEW, date(2025, 7, 1), date(2026, 6, 30), open_holdout=True)
        is True
    )


def test_after_an_opening_dev_and_clean_windows_stay_allowed(session: Session) -> None:
    v = version(session)
    record_opening(session, NEW, pk(v), date(2025, 7, 1), date(2026, 6, 30))
    session.commit()

    for start, end in [
        (date(2020, 1, 1), date(2025, 6, 30)),
        (date(2026, 7, 1), date(2026, 9, 30)),
    ]:
        assert check_window(session, PROTOCOL, NEW, start, end, open_holdout=False) is False


def test_a_strategy_seen_before_the_protocol_is_allowed_and_leaves_no_row(
    session: Session,
) -> None:
    for flag in (False, True):
        assert (
            check_window(
                session,
                PROTOCOL,
                "earnings_gap",
                date(2025, 1, 1),
                date(2026, 10, 1),
                open_holdout=flag,
            )
            is False
        )
    assert openings(session) == []


# =============================================================================================
# the table
# =============================================================================================


def test_the_database_refuses_a_second_row_for_one_strategy(session: Session) -> None:
    # Even a caller that skips check_window cannot record two openings.
    first = version(session, digest="h1")
    second = version(session, digest="h2")
    record_opening(session, NEW, pk(first), date(2025, 7, 1), date(2026, 6, 30))
    with pytest.raises(IntegrityError):
        record_opening(session, NEW, pk(second), date(2025, 7, 1), date(2026, 6, 30))


def test_an_opening_must_point_at_a_real_version(session: Session) -> None:
    with pytest.raises(IntegrityError):
        record_opening(session, NEW, 999, date(2025, 7, 1), date(2026, 6, 30))


def test_init_db_adds_the_table_to_an_existing_database_and_keeps_its_rows(
    engine: Engine,
) -> None:
    with engine.begin() as conn:
        conn.exec_driver_sql("DROP TABLE holdout_opening")  # a database from before this table
    with Session(engine) as s:
        source_id = pk(new_source(s))
    assert "holdout_opening" not in inspect(engine).get_table_names()

    init_db(engine)

    assert "holdout_opening" in inspect(engine).get_table_names()
    with Session(engine) as s:
        assert [r.id for r in s.exec(select(Source)).all()] == [source_id]


def test_init_db_adds_run_id_to_an_opening_table_from_before_it_existed(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.exec_driver_sql("DROP TABLE holdout_opening")
        conn.exec_driver_sql(
            "CREATE TABLE holdout_opening (id INTEGER PRIMARY KEY, strategy_id VARCHAR UNIQUE, "
            "strategy_version_id INTEGER, window_start DATE, window_end DATE, opened_at DATETIME)"
        )
    init_db(engine)
    columns = {c["name"] for c in inspect(engine).get_columns("holdout_opening")}
    assert "run_id" in columns
