"""A store and a database for the gate tests (tests/evolve/test_gate.py, test_gate_cli.py).

Dates: weekdays 2024-12-02 .. 2025-06-30 minus 2024-12-25, 2025-01-01, 2025-01-09 (national day
of mourning), 2025-01-20, 2025-02-17, 2025-04-18, 2025-05-26 and 2025-06-19: the real NYSE
closures of the period (documented, NYSE holiday calendar). ``GATE`` makes 2025-01 .. 2025-04
the development window and 2025-05 .. 2025-06 the holdout, with one-month train and test windows,
so the walk-forward has three pairs: (Jan, Feb), (Feb, Mar), (Mar, Apr).

Market (made up, chosen so that every name sits clearly on one side of the approved $5 / $20M
pool thresholds and the benchmark criteria have wide margins; numbers measured with the current
benchmark code on this store):
- P1..P4: common stocks, CIKs 201..204, each with one 8-K filed long before the window, so all
  four are in every pool. Closes start at 50 and alternate per session between g + 0.5% and
  g - 0.5% (g: P1 +0.4%, P2 +0.1%, P3 -0.1%, P4 -0.3%); each open is the previous close. Daily
  volume 3M / 2M / 1M / 1M shares ($40M+ a day), so the size-2 basket holds P1 and P2.
- SPY: an ETF (never a basket member or a pool member: no CIK), 400 at the start, alternating
  +0.3% / -0.2%, 10M shares.
- A daily bar is visible from 20:00 New York of its session (asof.ingest.bars, documented).
- ``future=True`` (the default) multiplies every price after DEV_END by ten and every volume by
  a hundred, and files a P2 split dated 2025-03-05 on 2025-05-15: a leak of holdout bars or
  records into anything the gate computes shows as a wild number. A split record is visible
  from its available_at (asof.ingest.corporate_actions, documented).

Database:
- Run A ("edge"): daily equity on every dev session, alternating +1% / 0%; seven legs with
  distinct returns +4% .. +10%, entered in February, March and April (the last exits on
  DEV_END itself, at the close).
- Run B ("luck"): equity alternating +1% / +0.2% in January (a better January Sharpe than A's)
  and +0.5% / -0.5% after; three legs of -3% .. -5%, one per test month.
  So the (Jan, Feb) pair picks B, the other two pick A, and A is the candidate.
- ``holdout=True`` adds what must not matter: A and B equity rows on every holdout session
  (A +5% / -4%, B +5% / +4.9%: counted, B would overtake A as the candidate), a leg of A
  entering in dev and exiting after DEV_END (+50%), legs entering in the holdout (+30%), run Z
  (a third version of the strategy that only traded in the holdout) and a new filer's 8-K.
- Trade returns are pnl / (shares * entry_price) (asof.evolve.null.spans_of_run, documented);
  entry 09:30 New York, exits at 09:30 or 20:00 New York (asof.replay.run._fill_time,
  documented). Equity rows: equity = 100,000 + realized + unrealized, available from 20:00 New
  York of the session (asof.store.models.EquityDaily, documented).
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from sqlmodel import Session, col, select

from asof.evolve.protocol import EvaluationProtocol
from asof.ingest.alpaca import Bar
from asof.store.market import market_root
from asof.store.models import EquityDaily, Run, StrategyVersion, Trade
from asof.strategy import content_hash
from tests.evolve.protocols import protocol
from tests.evolve.test_null import write_split_rows
from tests.ingest.fakes import daily
from tests.store.factories import add, new_document, new_entity, new_source, new_trade_decision, pk
from tests.store.test_filer_securities import CikSeg, write_cik_master
from tests.store.test_universe import Seg, store_daily, write_classes
from tests.strategy.files import write_strategy

ET = ZoneInfo("America/New_York")
INITIAL = 100_000.0
CLOSED = {
    date(2024, 12, 25),
    date(2025, 1, 1),
    date(2025, 1, 9),
    date(2025, 1, 20),
    date(2025, 2, 17),
    date(2025, 4, 18),
    date(2025, 5, 26),
    date(2025, 6, 19),
}


def _calendar(start: date, end: date) -> tuple[date, ...]:
    days, day = [], start
    while day <= end:
        if day.weekday() < 5 and day not in CLOSED:
            days.append(day)
        day += timedelta(days=1)
    return tuple(days)


CAL = _calendar(date(2024, 12, 2), date(2025, 6, 30))
DEV_START, DEV_END = date(2025, 1, 1), date(2025, 4, 30)
HOLDOUT_START, HOLDOUT_END = date(2025, 5, 1), date(2025, 6, 30)
DEV_DAYS = [d for d in CAL if DEV_START <= d <= DEV_END]
HOLDOUT_DAYS = [d for d in CAL if d > DEV_END]
PRE_DEV_DAYS = [d for d in CAL if d < DEV_START]


def month(m: int, year: int = 2025) -> list[date]:
    return [d for d in CAL if (d.year, d.month) == (year, m)]


GATE: EvaluationProtocol = protocol(
    dev={"start": DEV_START, "end": DEV_END},
    holdout={"start": HOLDOUT_START, "end": HOLDOUT_END},
    clean_after=date(2025, 7, 1),
    seen_before_protocol=[],
    benchmarks={
        "symbols": ["SPY"],
        "basket": {"size": 2, "lookback_sessions": 5, "rebalance": "monthly"},
    },
    walk_forward={"train_months": 1, "test_months": 1, "min_positive": 2},
)
PERMUTATIONS = 39  # the smallest p is 1/40 = 0.025, under alpha 0.05

# ---------------------------------------------------------------------------------------------
# market store
# ---------------------------------------------------------------------------------------------

STOCKS = {"P1": (0.004, 3_000_000), "P2": (0.001, 2_000_000), "P3": (-0.001, 1_000_000),
          "P4": (-0.003, 1_000_000)}  # fmt: skip
CIKS = {"P1": 201, "P2": 202, "P3": 203, "P4": 204}


def at_open(day: date) -> datetime:
    return datetime.combine(day, time(9, 30), tzinfo=ET)


def close_of(day: date) -> datetime:
    return datetime.combine(day, time(20), tzinfo=ET)


def _path(start: float, up: float, down: float) -> list[tuple[float, float]]:
    """(open, close) per CAL session; the move alternates up, down, up, ..."""
    out, close = [], start
    for i in range(len(CAL)):
        open_ = close
        close = close * (1 + (up if i % 2 == 0 else down))
        out.append((open_, close))
    return out


def store(
    data_dir: Path,
    *,
    through: date = CAL[-1],
    future: bool = True,
    spy: tuple[float, float] = (0.003, -0.002),
) -> Path:
    root = market_root(data_dir)
    paths = {s: (_path(50.0, g + 0.005, g - 0.005), v) for s, (g, v) in STOCKS.items()}
    paths["SPY"] = (_path(400.0, *spy), 10_000_000)
    bars: list[Bar] = []
    for symbol, (path, volume) in paths.items():
        for day, (o, c) in zip(CAL, path, strict=True):
            if day > through:
                break
            wild = future and day > DEV_END
            o, c, v = (o * 10, c * 10, volume * 100) if wild else (o, c, volume)
            bars.append(
                daily(symbol, day, open=o, high=max(o, c), low=min(o, c), close=c, volume=v, vwap=c)
            )
    store_daily(root, bars)
    write_cik_master(
        root,
        [*(CikSeg(f"S_{s}", s, CIKS[s]) for s in STOCKS), CikSeg("S_SPY", "SPY", None)],
    )
    write_classes(root, [*(Seg(f"S_{s}", s) for s in STOCKS), Seg("S_SPY", "SPY", cls="etf")])
    if future:
        # A 5:1 split of P2 dated inside dev that went on file only in the holdout (P2's prices
        # never split): read as known after DEV_END it would turn every random P2 leg across
        # 2025-03-05 into a +400% one.
        write_split_rows(root, [("P2", date(2025, 3, 5), 1.0, 5.0, close_of(date(2025, 5, 15)))])
    return root


# ---------------------------------------------------------------------------------------------
# database
# ---------------------------------------------------------------------------------------------

EARLY_FILING = datetime(2024, 6, 3, 12, tzinfo=UTC)


def seed_filers(session: Session) -> int:
    """P1..P4 each filed an 8-K long ago; returns an entity id the trade decisions can hang on."""
    source_id = pk(new_source(session))
    first = None
    for symbol, cik in CIKS.items():
        entity = new_entity(session, cik=cik, ticker=symbol, name=f"{symbol} Corp")
        new_document(
            session,
            source_id=source_id,
            entity_id=pk(entity),
            available_at=EARLY_FILING,
            doc_type="8-K",
        )
        first = first or pk(entity)
    assert first is not None
    return first


def late_filing(session: Session) -> None:
    """A fifth company, CIK 205 (no security in the store), whose first 8-K lands in the
    holdout."""
    entity = new_entity(session, cik=205, ticker="P5", name="P5 Corp")
    new_document(
        session,
        source_id=pk(new_source(session, name="sec_edgar_late")),
        entity_id=pk(entity),
        available_at=close_of(date(2025, 5, 15)),
        doc_type="8-K",
    )


def version(
    session: Session, strategy_dir: Path, summary: str | None, strategy_id: str = "earnings_gap"
) -> int:
    """A StrategyVersion whose hash matches ``strategy_dir`` with ``summary``'s overrides, as
    ``asof gate null`` checks it."""
    overrides = summary.split(", ") if summary else []
    return pk(
        add(
            session,
            StrategyVersion(
                strategy_id=strategy_id,
                content_hash=content_hash(strategy_dir, overrides),
                proposed_by="human",
                registered_at=datetime(2025, 1, 1, tzinfo=UTC),
                change_kind="config",
                summary=summary,
            ),
        )
    )


def new_run(session: Session, version_id: int | None, strategy_id: str = "earnings_gap") -> int:
    run = Run(
        strategy_id=strategy_id,
        strategy_version_id=version_id,
        mode="replay",
        as_of=close_of(CAL[-1]),
        started_at=close_of(CAL[-1]),
        git_sha="test",
    )
    return pk(add(session, run))


def alternating(n: int, first: float, second: float) -> list[float]:
    return [first if i % 2 == 0 else second for i in range(n)]


def add_equity(
    session: Session,
    run_id: int,
    days: Sequence[date],
    returns: Sequence[float],
    *,
    start: float = INITIAL,
) -> list[float]:
    """One row per day, each the previous value times (1 + return); realized = equity - INITIAL,
    so the first row's equity - realized - unrealized is INITIAL whatever ``start`` is."""
    values, value = [], start
    for day, r in zip(days, returns, strict=True):
        value *= 1 + r
        values.append(value)
        session.add(
            EquityDaily(
                run_id=run_id,
                session_date=day,
                equity=value,
                realized=value - INITIAL,
                unrealized=0.0,
                gross_exposure=0.0,
                open_positions=0,
                available_at=close_of(day),
            )
        )
    session.commit()
    return values


@dataclass(frozen=True)
class Leg:
    symbol: str
    entry: date
    exit_at: datetime
    ret: float


def book(session: Session, run_id: int, decision_id: int, legs: Iterable[Leg]) -> None:
    for leg in legs:
        session.add(
            Trade(
                decision_id=decision_id,
                run_id=run_id,
                strategy_id="earnings_gap",
                security_id=f"S_{leg.symbol}",
                symbol=leg.symbol,
                leg=1,
                direction="long",
                entry_at=at_open(leg.entry),
                entry_price=50.0,
                shares=100,
                key_level=49.0,
                hard_stop=47.5,
                risk_per_share=2.5,
                exit_at=leg.exit_at,
                exit_price=50.0 * (1 + leg.ret),
                exit_reason="break",
                pnl=leg.ret * 5_000.0,
                r_multiple=1.0,
                holding_sessions=None,
                available_at=close_of(leg.exit_at.astimezone(ET).date()),
            )
        )
    session.commit()


def d(m: int, day: int, year: int = 2025) -> date:
    out = date(year, m, day)
    assert out in CAL, out
    return out


A_LEGS = [
    Leg("P3", d(2, 4), at_open(d(2, 7)), 0.06),
    Leg("P4", d(2, 11), close_of(d(2, 14)), 0.09),
    Leg("P2", d(3, 4), at_open(d(3, 6)), 0.07),
    Leg("P3", d(3, 12), close_of(d(3, 18)), 0.05),
    Leg("P4", d(4, 2), at_open(d(4, 4)), 0.08),
    Leg("P1", d(4, 22), close_of(d(4, 25)), 0.10),
    Leg("P2", d(4, 28), close_of(DEV_END), 0.04),  # exits on the last dev day: counts
]
B_LEGS = [
    Leg("P1", d(2, 5), at_open(d(2, 10)), -0.04),
    Leg("P2", d(3, 5), close_of(d(3, 10)), -0.03),
    Leg("P3", d(4, 8), at_open(d(4, 10)), -0.05),
]
A_LATE = [
    Leg("P1", d(4, 28), at_open(d(5, 2)), 0.50),  # enters in dev, exits in the holdout
    Leg("P2", d(5, 6), close_of(d(5, 9)), 0.30),
]
B_LATE = [Leg("P4", d(6, 2), at_open(d(6, 5)), 0.30)]


def a_returns(days: Sequence[date]) -> list[float]:
    return alternating(len(days), 0.01, 0.0)


def b_returns(days: Sequence[date]) -> list[float]:
    jan = [x for x in days if x.month == 1]
    return alternating(len(jan), 0.01, 0.002) + alternating(len(days) - len(jan), 0.005, -0.005)


@dataclass(frozen=True)
class Runs:
    a: int
    b: int
    decision: int
    strategy_dir: Path
    v1: int
    v2: int


def seed(session: Session, tmp_path: Path, *, holdout: bool = False) -> Runs:
    """Filers, versions v1 (A) and v2 (B), runs A and B as described above."""
    strategy_dir = write_strategy(tmp_path / "strategies")
    entity_id = seed_filers(session)
    v1 = version(session, strategy_dir, None)
    v2 = version(session, strategy_dir, "execute.exit.max_sessions=40")
    a, b = new_run(session, v1), new_run(session, v2)
    decision = pk(new_trade_decision(session, run_id=a, entity_id=entity_id))
    add_equity(session, a, DEV_DAYS, a_returns(DEV_DAYS))
    add_equity(session, b, DEV_DAYS, b_returns(DEV_DAYS))
    book(session, a, decision, A_LEGS)
    book(session, b, decision, B_LEGS)
    runs = Runs(a, b, decision, strategy_dir, v1, v2)
    if holdout:
        add_holdout(session, runs)
    return runs


def extra_run(
    session: Session,
    runs: Runs,
    days: Sequence[date],
    returns: Sequence[float],
    legs: Iterable[Leg],
    *,
    version_id: int | None = None,
    strategy_id: str = "earnings_gap",
) -> int:
    """Another run with these equity rows and legs, under A's version unless told otherwise."""
    run_id = new_run(session, version_id or runs.v1, strategy_id)
    add_equity(session, run_id, days, returns)
    book(session, run_id, runs.decision, legs)
    return run_id


def add_holdout(session: Session, runs: Runs) -> None:
    paths = {runs.a: (0.05, -0.04), runs.b: (0.05, 0.049)}
    for run_id, (up, down) in paths.items():
        last = session.exec(
            select(EquityDaily)
            .where(EquityDaily.run_id == run_id)
            .order_by(col(EquityDaily.session_date).desc())
        ).first()
        assert last is not None
        add_equity(
            session,
            run_id,
            HOLDOUT_DAYS,
            alternating(len(HOLDOUT_DAYS), up, down),
            start=last.equity,
        )
    book(session, runs.a, runs.decision, A_LATE)
    book(session, runs.b, runs.decision, B_LATE)
    v3 = version(session, runs.strategy_dir, "execute.exit.max_sessions=30")
    z = new_run(session, v3)
    add_equity(session, z, HOLDOUT_DAYS, [0.02] * len(HOLDOUT_DAYS))
    book(session, z, runs.decision, [Leg("P1", d(5, 5), at_open(d(5, 8)), 0.2)])
    late_filing(session)
