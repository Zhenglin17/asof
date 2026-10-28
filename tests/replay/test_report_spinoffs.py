"""Guards against a replay report that hides how much of a run's result came from spin-offs:
legs whose pnl includes child proceeds, or that ran on raw prices after an event that could not
be priced (restart), mixed in with ordinary legs without a count.

API under test (asof.replay.build_report), new section, exact heading "## Spin-offs". Over the
run's trades whose ``spinoff`` is not NULL it lists, each on its own line:
    "- adjust: <n>"                  legs held through an adjusted event
    "- restart: <n>"                 legs flagged restart
    "- child proceeds: <sum, ,.2f>"  sum of child_proceeds over those legs
With no such leg the section says "no legs held through a spin-off" and lists no count.

No as_of: a report summarises one finished run. Assumption to check on the first real report
after this change: the adjust / restart counts equal SELECT spinoff, count(*) FROM trade WHERE
run_id = <run> GROUP BY spinoff.

Hand set (tests/replay/test_report.py builders): three spin-off legs and two ordinary ones.
    adjust   child 1500.00      adjust   child 1335.00      restart  child 0
  -> adjust: 2, restart: 1, child proceeds: 2,835.00
"""

from datetime import date

from sqlmodel import Session

from asof.replay import build_report
from asof.store.models import Entity
from tests.replay.test_report import make_decision, make_run, make_trade, section, version
from tests.store.factories import add, pk

SPINOFFS = "## Spin-offs"


def lines_of(body: str) -> list[str]:
    return [ln.strip() for ln in body.splitlines() if ln.strip()]


def spin_run(session: Session, rows: list[tuple[float, str | None, float]]) -> int:
    entity = add(session, Entity(kind="company", ticker="SPN", name="Spin Inc.", cik=995))
    run = make_run(session, pk(version(session, 0)))
    for i, (pnl, kind, child) in enumerate(rows):
        d = make_decision(session, run, entity, date(2024, 5, 1 + i))
        t = make_trade(session, d, 1, pnl, pnl / 200, "break", date(2024, 5, 10 + i))
        t.spinoff = kind
        t.child_proceeds = child
        session.add(t)
    session.commit()
    return pk(run)


def test_spinoff_section_counts_the_legs_and_sums_the_children(session: Session) -> None:
    run_id = spin_run(
        session,
        [
            (62.5, "adjust", 1500.0),
            (-352.5, "adjust", 1335.0),
            (-1025.0, "restart", 0.0),
            (100.0, None, 0.0),
            (-50.0, None, 0.0),
        ],
    )
    body = lines_of(section(build_report(session, run_id), SPINOFFS))
    assert "- adjust: 2" in body
    assert "- restart: 1" in body
    assert "- child proceeds: 2,835.00" in body


def test_a_run_without_spinoff_legs_says_so(session: Session) -> None:
    run_id = spin_run(session, [(100.0, None, 0.0), (-50.0, None, 0.0)])
    body = section(build_report(session, run_id), SPINOFFS)
    assert "no legs held through a spin-off" in body.lower()
    assert "- adjust:" not in body


def test_old_rows_without_the_columns_count_as_ordinary_legs(session: Session) -> None:
    # rows written before the columns existed read back NULL in both
    run_id = spin_run(session, [(100.0, None, None), (62.5, "adjust", 1500.0)])  # type: ignore[list-item]
    body = lines_of(section(build_report(session, run_id), SPINOFFS))
    assert "- adjust: 1" in body
    assert "- restart: 0" in body
    assert "- child proceeds: 1,500.00" in body
