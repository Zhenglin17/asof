# Guards the decision table upgrade against losing or renumbering rows (a trade, attribution or
# citation would then point at another decision or at nothing), against a half-done rebuild left
# behind when a foreign key fails, and against a trade decision that names neither an entity nor
# a security getting in once the entity requirement is relaxed for securities.
"""API under test:

    Decision gains ``security_id: str | None`` (indexed) and ``symbol: str | None``.
    ck_trade_fields: kind != 'trade' OR ((entity_id IS NOT NULL OR security_id IS NOT NULL)
                     AND direction IS NOT NULL AND strategy_id IS NOT NULL).
    init_db -> _upgrade_decision_table: a decision table without ``security_id`` is rebuilt from
    the model inside init_db's all-or-nothing transaction with PRAGMA defer_foreign_keys=ON, rows
    copied back with their ids; a foreign-key violation left at commit fails the whole upgrade.

The old table is the exact DDL a fresh store created before the change (copied from
sqlite_master of a store built by init_db at commit 38758d4). Old rows are written with raw SQL:
the current model has columns the old table lacks.

No as_of: schema migration only. Assumptions to confirm on the real store (audit after the
first init_db on /data/asof/meta.db, with a copy taken before):
- ``SELECT count(*), min(id), max(id) FROM decision`` is the same before and after, and so are
  the counts of trade / attribution / decision_citation rows joining a decision on decision_id.
- ``PRAGMA foreign_key_check`` returns no row after the upgrade.
- ``SELECT count(*) FROM decision WHERE security_id IS NOT NULL`` is 0 right after it.
"""

import sqlite3
from pathlib import Path

import pytest
from sqlalchemy import Engine, inspect
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, SQLModel

from asof.store.db import init_db, make_engine
from asof.store.models import Decision
from tests.store.factories import T0, add, new_entity, new_run, pk

OLD_DECISION_DDL = """
CREATE TABLE decision (
	id INTEGER NOT NULL,
	run_id INTEGER NOT NULL,
	kind VARCHAR NOT NULL,
	as_of DATETIME NOT NULL,
	strategy_id VARCHAR,
	entity_id INTEGER,
	direction VARCHAR,
	conviction FLOAT,
	topic VARCHAR,
	statement VARCHAR,
	implication VARCHAR,
	"check" VARCHAR,
	payload JSON NOT NULL,
	PRIMARY KEY (id),
	CONSTRAINT ck_kind_enum CHECK ("kind" IN ('trade', 'observation')),
	CONSTRAINT ck_direction_enum CHECK ("direction" IN ('long', 'short', 'flat')),
	CONSTRAINT ck_trade_fields CHECK (kind != 'trade' OR (entity_id IS NOT NULL AND direction IS NOT NULL AND strategy_id IS NOT NULL)),
	CONSTRAINT ck_observation_fields CHECK (kind != 'observation' OR (statement IS NOT NULL AND "check" IS NOT NULL)),
	CONSTRAINT ck_conviction_range CHECK (conviction IS NULL OR (conviction >= 0 AND conviction <= 1)),
	FOREIGN KEY(run_id) REFERENCES run (id),
	FOREIGN KEY(entity_id) REFERENCES entity (id)
)
"""  # noqa: E501  (verbatim)
OLD_INDEXES = (
    "CREATE INDEX ix_decision_as_of ON decision (as_of)",
    "CREATE INDEX ix_decision_strategy_id ON decision (strategy_id)",
    "CREATE INDEX ix_decision_run_id ON decision (run_id)",
    "CREATE INDEX ix_decision_entity_id ON decision (entity_id)",
)
OLD_COLUMNS = (
    "id, run_id, kind, as_of, strategy_id, entity_id, direction, conviction, topic, statement, "
    'implication, "check", payload'
)
AS_OF = "2024-07-15 13:30:00.000000"  # T0 as the UTCDateTime column stores it
TRADE_ID, OBSERVATION_ID = 7, 3  # not 1, 2: a renumbering copy would show


def decision_sql(eng: Engine) -> str:
    with eng.connect() as conn:
        return conn.exec_driver_sql(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'decision'"
        ).scalar_one()


def decision_rows(eng: Engine) -> list[tuple[object, ...]]:
    with eng.connect() as conn:
        rows = conn.exec_driver_sql(f"SELECT {OLD_COLUMNS} FROM decision ORDER BY id")
        return [tuple(r) for r in rows]


def old_store(path: Path, *, orphan_run: bool = False) -> tuple[Engine, int, int]:
    """Every table from the current model except decision, which has the old DDL; a trade
    decision (TRADE_ID, with an entity) and an observation (OBSERVATION_ID). Returns the engine,
    the run id and the entity id. ``orphan_run``: the trade decision points at a run that does
    not exist (written with foreign keys off, as a broken store could hold)."""
    eng = make_engine(path)
    SQLModel.metadata.create_all(eng)
    with eng.begin() as conn:
        conn.exec_driver_sql("DROP TABLE decision")
        conn.exec_driver_sql(OLD_DECISION_DDL)
        for index in OLD_INDEXES:
            conn.exec_driver_sql(index)
    with Session(eng) as s:
        run_id, entity_id = pk(new_run(s)), pk(new_entity(s))
    trade_run = 999 if orphan_run else run_id
    with eng.begin() as conn:
        if orphan_run:
            conn.exec_driver_sql("PRAGMA foreign_keys=OFF")
        conn.exec_driver_sql(
            f"INSERT INTO decision ({OLD_COLUMNS}) VALUES "
            f"({TRADE_ID}, {trade_run}, 'trade', '{AS_OF}', 'earnings_gap', {entity_id}, 'long', "
            '0.7, NULL, NULL, NULL, NULL, \'{"judge": "take_all", "equity_at_decision": 1e5}\')'
        )
        conn.exec_driver_sql(
            f"INSERT INTO decision ({OLD_COLUMNS}) VALUES "
            f"({OBSERVATION_ID}, {run_id}, 'observation', '{AS_OF}', NULL, NULL, NULL, NULL, "
            "'energy flows', 'Money is flowing into energy names.', NULL, "
            "'In two weeks, XLE relative to SPY is higher than today.', '{}')"
        )
    if orphan_run:
        eng.dispose()  # pooled connections had foreign keys switched off; new ones turn them on
    return eng, run_id, entity_id


def add_children(eng: Engine, run_id: int) -> None:
    """A trade, an attribution and a citation of the trade decision, written with raw SQL."""
    with eng.begin() as conn:
        conn.exec_driver_sql(
            "INSERT INTO trade (decision_id, run_id, strategy_id, security_id, symbol, leg, "
            "direction, entry_at, entry_price, shares, key_level, hard_stop, risk_per_share, "
            "exit_at, exit_price, exit_reason, pnl, r_multiple, holding_sessions, available_at) "
            f"VALUES ({TRADE_ID}, {run_id}, 'earnings_gap', 'S_GAPX', 'GAPX', 1, 'long', "
            f"'{AS_OF}', 56.5, 125, 54.0, 52.0, 4.5, '{AS_OF}', 53.4, 'break', -387.5, "
            f"-0.6889, 3, '{AS_OF}')"
        )
        conn.exec_driver_sql(
            "INSERT INTO attribution (decision_id, strategy_id, outcome_available_at, "
            f"realized_return, is_correct) VALUES ({TRADE_ID}, 'earnings_gap', '{AS_OF}', "
            "-0.005075, 0)"
        )
        conn.exec_driver_sql(
            "INSERT INTO decision_citation (decision_id, claim, evidence_kind, evidence_ref) "
            f"VALUES ({TRADE_ID}, 'Closed higher.', 'bar', 'GAPX/2026-02-17')"
        )


def children_of(eng: Engine, decision_id: int) -> dict[str, int]:
    with eng.connect() as conn:
        return {
            table: conn.exec_driver_sql(
                f"SELECT count(*) FROM {table} JOIN decision ON decision.id = {table}.decision_id "
                f"WHERE decision.id = {decision_id}"
            ).scalar_one()
            for table in ("trade", "attribution", "decision_citation")
        }


def fk_violations(eng: Engine) -> list[tuple[object, ...]]:
    with eng.connect() as conn:
        return [tuple(r) for r in conn.exec_driver_sql("PRAGMA foreign_key_check")]


def test_the_fixture_is_the_old_shape(tmp_path: Path) -> None:
    # passes today: the old DDL is the one the current model creates
    eng, run_id, _ = old_store(tmp_path / "old.db")
    add_children(eng, run_id)
    assert "security_id" not in {c["name"] for c in inspect(eng).get_columns("decision")}
    assert children_of(eng, TRADE_ID) == {"trade": 1, "attribution": 1, "decision_citation": 1}
    assert fk_violations(eng) == []
    eng.dispose()


# =============================================================================================
# normal path
# =============================================================================================


def test_init_db_rebuilds_an_old_decision_table_keeping_ids_and_children(tmp_path: Path) -> None:
    eng, run_id, _ = old_store(tmp_path / "old.db")
    add_children(eng, run_id)
    before = decision_rows(eng)

    init_db(eng)

    columns = {c["name"]: c for c in inspect(eng).get_columns("decision")}
    for name in ("security_id", "symbol"):
        assert name in columns, name
        assert columns[name]["nullable"], name
    indexed = {col for ix in inspect(eng).get_indexes("decision") for col in ix["column_names"]}
    assert {"security_id", "as_of", "run_id", "entity_id", "strategy_id"} <= indexed
    assert decision_rows(eng) == before  # same ids, same fields
    with eng.connect() as conn:
        added = conn.exec_driver_sql(
            "SELECT id, security_id, symbol FROM decision ORDER BY id"
        ).all()
    assert [tuple(r) for r in added] == [(OBSERVATION_ID, None, None), (TRADE_ID, None, None)]
    assert children_of(eng, TRADE_ID) == {"trade": 1, "attribution": 1, "decision_citation": 1}
    assert fk_violations(eng) == []
    with Session(eng) as s:
        trade = s.get(Decision, TRADE_ID)
        assert trade is not None
        assert (trade.kind, trade.direction, trade.conviction) == ("trade", "long", 0.7)
        assert trade.payload == {"judge": "take_all", "equity_at_decision": 1e5}
        assert trade.as_of == T0
    eng.dispose()


def test_init_db_again_changes_nothing(tmp_path: Path) -> None:
    eng, run_id, _ = old_store(tmp_path / "old.db")
    add_children(eng, run_id)
    init_db(eng)
    sql, rows = decision_sql(eng), decision_rows(eng)

    init_db(eng)

    assert decision_sql(eng) == sql
    assert decision_rows(eng) == rows
    assert children_of(eng, TRADE_ID) == {"trade": 1, "attribution": 1, "decision_citation": 1}
    eng.dispose()


# =============================================================================================
# the relaxed trade check, on an upgraded and on a fresh store
# =============================================================================================


@pytest.fixture(params=["upgraded", "fresh"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> tuple[Engine, int, int]:
    if request.param == "upgraded":
        eng, run_id, entity_id = old_store(tmp_path / "old.db")
        init_db(eng)
        return eng, run_id, entity_id
    eng = make_engine(tmp_path / "new.db")
    init_db(eng)
    with Session(eng) as s:
        return eng, pk(new_run(s)), pk(new_entity(s))


def trade(run_id: int, **fields: object) -> Decision:
    base: dict[str, object] = {
        "run_id": run_id,
        "kind": "trade",
        "as_of": T0,
        "strategy_id": "bx_trender",
        "direction": "long",
    }
    return Decision.model_validate({**base, **fields})


def test_a_trade_decision_may_name_a_security_instead_of_an_entity(
    store: tuple[Engine, int, int],
) -> None:
    eng, run_id, entity_id = store
    assert {"security_id", "symbol"} <= set(Decision.model_fields)
    with Session(eng) as s:
        bx = add(s, trade(run_id, security_id="S_AAA", symbol="AAA"))
        both = add(s, trade(run_id, entity_id=entity_id, security_id="S_AAA", symbol="AAA"))
        old_style = add(s, trade(run_id, entity_id=entity_id))
        assert (bx.security_id, bx.symbol, bx.entity_id) == ("S_AAA", "AAA", None)
        assert both.entity_id == entity_id
        assert old_style.security_id is None
    eng.dispose()


@pytest.mark.parametrize(
    "fields",
    [
        {},  # neither an entity nor a security
        {"security_id": "S_AAA", "direction": None},
        {"security_id": "S_AAA", "strategy_id": None},
    ],
    ids=["no_entity_no_security", "no_direction", "no_strategy"],
)
def test_a_trade_decision_missing_a_required_field_is_rejected(
    store: tuple[Engine, int, int], fields: dict[str, object]
) -> None:
    eng, run_id, _ = store
    assert "security_id" in Decision.model_fields
    with Session(eng) as s, pytest.raises(IntegrityError):
        add(s, trade(run_id, **fields))
    eng.dispose()


def test_observation_rules_are_unchanged(store: tuple[Engine, int, int]) -> None:
    eng, run_id, _ = store
    assert "security_id" in Decision.model_fields
    observation = {
        "run_id": run_id,
        "kind": "observation",
        "as_of": T0,
        "topic": "energy flows",
        "statement": "Money is flowing into energy names.",
        "check": "In two weeks, XLE relative to SPY is higher than today.",
    }
    with Session(eng) as s:
        ok = add(s, Decision.model_validate(observation))
        assert ok.security_id is None
    with Session(eng) as s, pytest.raises(IntegrityError):
        add(s, Decision.model_validate({**observation, "check": None}))
    eng.dispose()


# =============================================================================================
# boundary: a foreign key broken at commit fails the whole upgrade
# =============================================================================================


def test_a_foreign_key_violation_fails_the_upgrade_and_changes_nothing(tmp_path: Path) -> None:
    eng, run_id, _ = old_store(tmp_path / "old.db", orphan_run=True)
    add_children(eng, run_id)
    sql, rows = decision_sql(eng), decision_rows(eng)
    assert fk_violations(eng) != []  # fixture: the trade decision's run does not exist

    with pytest.raises((IntegrityError, sqlite3.IntegrityError)):
        init_db(eng)

    # rolled back: the old table, its rows and its children exactly as they were
    assert decision_sql(eng) == sql
    assert "security_id" not in {c["name"] for c in inspect(eng).get_columns("decision")}
    assert decision_rows(eng) == rows
    assert children_of(eng, TRADE_ID) == {"trade": 1, "attribution": 1, "decision_citation": 1}
    eng.dispose()
