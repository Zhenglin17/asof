"""Guards against a schema upgrade that breaks or rewrites an existing trade table: a store whose
trade table predates the spin-off columns (child_proceeds, spinoff) must gain both as nullable
columns on init_db, keep every old row with NULL there, and survive init_db run twice.

No as_of: this is schema migration only. Assumption to confirm on real data (audit, after the
first init_db on /data/asof/meta.db): `SELECT count(*) FROM trade WHERE spinoff IS NULL AND
child_proceeds IS NULL` equals the number of trade rows written before the upgrade.
"""

from pathlib import Path

from sqlalchemy import inspect
from sqlmodel import Session, SQLModel, select

from asof.store.db import init_db, make_engine
from asof.store.models import Trade
from tests.store.factories import T0, add, new_entity, new_run, new_trade_decision, pk
from tests.store.test_trade_upgrade import trade_columns

NEW_COLUMNS = ("child_proceeds", "spinoff")


def test_init_db_adds_the_spinoff_columns_to_an_old_trade_table(tmp_path: Path) -> None:
    eng = make_engine(tmp_path / "old.db")
    SQLModel.metadata.create_all(eng)
    with Session(eng) as s:
        run = new_run(s)
        decision = new_trade_decision(s, run_id=pk(run), entity_id=pk(new_entity(s)))
        old = add(
            s,
            Trade(
                decision_id=pk(decision),
                run_id=pk(run),
                strategy_id="momentum",
                security_id="S_AAPL",
                symbol="AAPL",
                leg=1,
                direction="long",
                entry_at=T0,
                entry_price=100.0,
                shares=10,
                key_level=98.0,
                hard_stop=96.0,
                risk_per_share=4.0,
                exit_at=T0,
                exit_price=104.0,
                exit_reason="break",
                pnl=40.0,
                r_multiple=1.0,
                holding_sessions=4,
                available_at=T0,
            ),
        )
        old_id = pk(old)
    with eng.begin() as conn:
        for name in NEW_COLUMNS:
            if name in trade_columns(conn):
                conn.exec_driver_sql(f"ALTER TABLE trade DROP COLUMN {name}")
    assert not set(NEW_COLUMNS) & trade_columns(eng)

    init_db(eng)
    init_db(eng)

    columns = {c["name"]: c for c in inspect(eng).get_columns("trade")}
    for name in NEW_COLUMNS:
        assert name in columns, name
        assert columns[name]["nullable"], name
    with eng.connect() as conn:
        row = conn.exec_driver_sql("SELECT id, pnl, child_proceeds, spinoff FROM trade").one()
    assert tuple(row) == (old_id, 40.0, None, None)
    with Session(eng) as s:
        (back,) = s.exec(select(Trade)).all()
        assert (back.child_proceeds, back.spinoff) == (None, None)
    eng.dispose()


def test_a_fresh_store_has_the_spinoff_columns(tmp_path: Path) -> None:
    eng = make_engine(tmp_path / "new.db")
    init_db(eng)
    assert set(NEW_COLUMNS) <= trade_columns(eng)
    eng.dispose()
