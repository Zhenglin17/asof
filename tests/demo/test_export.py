"""Guards against a demo page that misstates a replay or shows what was not yet knowable: exported
trades that differ from the Trade rows, metrics or a drawdown that disagree with the replay
report, an equity curve stepped on UTC dates instead of New York sessions, claims shown in an
order that no longer lines up with their stances, a decision compared with the base run's
decision on another day, a filing timeline filtered on the period it describes (source_time)
rather than on when it became public (available_at), and local file paths leaking into the
published JSON.

API under test (asof.demo.export), read-only over the store:

    equity_curve(trades, equity0, start, end) -> [{"date", "equity", "drawdown"}]
        first point (start, equity0, 0.0); then one point per New York exit date (pnl of every
        leg exiting that date summed), sorted; drawdown = (peak - equity) / peak
    export_run(session, run_id, params) -> {"run_id", "judge", "params", "metrics", "curve"}
    export_decisions(session, run_id, base_run_id) -> one dict per Decision, by (as_of, id):
        id, as_of, entity_id, symbol, verdict, conviction, candidate, claims, trades, r, base_r
    export_chunks(session, chunk_ids, quotes) -> {chunk_id: {"document_id", "section_path",
        "text", "spans"}}
    export_timeline(session, entity_ids, start, end) -> documents with start <= available_at
        <= end, by (available_at, id): {"document_id", "entity_id", "doc_type", "title",
        "available_at"}

Hand-computed set (initial equity 100,000, the replay report's set from tests/replay):
    base run, take_all, entity XYZ
        D1 01-02  leg 1 +300 R 1.5 exit 01-10                     r 1.5
        D2 01-04  leg 1 -200 R -1.0 exit 01-12, leg 2 +500 R 2.5 exit 01-20   r 1.5
        D3 01-08  leg 1 -100 R -0.5 exit 01-15                    r -0.5
        D4 01-16  leg 1 +100 R 0.5 exit 01-25                     r 0.5
      curve 100,300 / 100,100 / 100,000 / 100,500 / 100,600; max drawdown 300 / 100,300
    llm run (inserted L1, L3, L2, L4 so id order differs from as_of order)
        L1 XYZ 01-02 take 0.7, claims [for, against], leg 1 +200 R 1.0     base_r 1.5 (D1)
        L2 XYZ 01-05 take 0.6, one claim, leg 1 +50 R 0.25                 base_r None (no
           base decision on 01-05; D2 is 01-04)
        L3 XYZ 01-08 pass 0.3, one claim, no trades -> r None              base_r -0.5 (D3)
        L4 ABC 01-16 no_call, no claims, no trades                         base_r None (D4 is
           XYZ on the same day: another entity)
    a third run's trade (+10,000) must appear nowhere.
"""

import dataclasses
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from sqlmodel import Session, col, select

from asof.demo.export import (
    equity_curve,
    export_chunks,
    export_decisions,
    export_run,
    export_timeline,
)
from asof.replay import replay_metrics
from asof.store.models import Chunk, Decision, DecisionCitation, Entity, Run, Trade
from tests.replay.test_report import at, make_run, make_trade, version
from tests.store.factories import add, new_chunk, new_document, new_source, pk

ET = ZoneInfo("America/New_York")
EQUITY0 = 100_000.0
TRADE_FIELDS = (
    "leg",
    "entry_at",
    "exit_at",
    "entry_price",
    "exit_price",
    "shares",
    "pnl",
    "r_multiple",
    "exit_reason",
)
# Must never reach the published JSON: the store keeps local paths in places a careless
# export would copy (a payload field here).
LOCAL_PATH = "/data/asof/llm_cache/ab/cdef0123.json"

# Hand-made chunk texts. Whitespace that differs between a chunk and a quote is a fallback case:
# measured on the 2026-07-01..10-21 demo grid, all 274 distinct cited quotes matched their chunk
# exactly, none needed the whitespace-insensitive match and none went without a span.
CHUNK_REVENUE = (
    "Fourth quarter revenue was\n  $1,234.5 million, up 18% from the prior year.\n"
    "Operating margin fell to 12.0% from 14.5%."
)
CHUNK_OUTLOOK = "The company now expects full year revenue between $5.1 billion and $5.3 billion."
QUOTE_EXACT = "Operating margin fell to 12.0% from 14.5%"
QUOTE_SPACED = "revenue was $1,234.5 million, up 18%"  # the text has a newline and two spaces
QUOTE_SPACED_ORIGINAL = "revenue was\n  $1,234.5 million, up 18%"
QUOTE_MISSING = "revenue was $9,999.9 million"
QUOTE_OUTLOOK = "expects full year revenue between $5.1 billion and $5.3 billion"


def candidate(entity: Entity, symbol: str, document_ids: Iterable[int] = ()) -> dict[str, Any]:
    """A candidate payload as run.py's _payload writes it (dates as ISO strings)."""
    return {
        "entity_id": pk(entity),
        "security_id": f"S_{symbol}",
        "symbol": symbol,
        "document_ids": list(document_ids),
        "available_at": "2024-01-01T21:08:00+00:00",
        "reaction_session": "2024-01-02",
        "pre_earnings_close": 50.0,
        "gap_open": 56.5,
        "gap_high": 57.0,
        "gap_low": 54.0,
        "gap_close": 56.0,
        "gap": 0.12,
        "rvol": 3.1,
        "atr": 2.0,
        "avg_dollar_volume": 40_000_000.0,
    }


def decide(
    session: Session,
    run: Run,
    entity: Entity,
    day: date,
    judge: str,
    *,
    verdict: str | None = None,
    conviction: float | None = None,
    stances: list[str] | None = None,
) -> Decision:
    payload: dict[str, Any] = {
        "judge": judge,
        "equity_at_decision": EQUITY0,
        "candidate": candidate(entity, entity.ticker),
    }
    if judge == "llm":
        payload |= {
            "verdict": verdict,
            "attempts": 1,
            "request_keys": ["k0"],
            "failures": [],
            "stances": stances or [],
            "cache_file": LOCAL_PATH,
        }
    direction = "long" if judge == "take_all" or verdict == "take" else "flat"
    return add(
        session,
        Decision(
            run_id=pk(run),
            kind="trade",
            as_of=at(day),
            strategy_id="earnings_gap",
            entity_id=pk(entity),
            direction=direction,
            conviction=conviction,
            payload=payload,
        ),
    )


def cite(session: Session, decision: Decision, claim: str, chunk: Chunk, quote: str) -> None:
    add(
        session,
        DecisionCitation(
            decision_id=pk(decision),
            claim=claim,
            evidence_kind="chunk",
            chunk_id=pk(chunk),
            quote=quote,
        ),
    )


@dataclass(frozen=True)
class World:
    base: int
    llm: int
    stray: int
    xyz: Entity
    abc: Entity
    revenue: Chunk
    outlook: Chunk


@pytest.fixture
def world(session: Session) -> World:
    xyz = add(session, Entity(kind="company", ticker="XYZ", name="XYZ Inc.", cik=999))
    abc = add(session, Entity(kind="company", ticker="ABC", name="ABC Corp.", cik=998))
    source = new_source(session)
    doc = new_document(
        session,
        source_id=pk(source),
        entity_id=pk(xyz),
        available_at=datetime(2024, 1, 1, 16, 8, tzinfo=ET),
        doc_type="8-K",
    )
    revenue = new_chunk(session, document_id=pk(doc), ordinal=0, text=CHUNK_REVENUE)
    outlook = new_chunk(
        session,
        document_id=pk(doc),
        ordinal=1,
        text=CHUNK_OUTLOOK,
        section_path="EX-99.1 > Outlook",
    )
    v0, v1 = version(session, 0), version(session, 1)

    base = make_run(session, pk(v0))
    d1, d2, d3, d4 = (
        decide(session, base, xyz, date(2024, 1, d), "take_all") for d in (2, 4, 8, 16)
    )
    make_trade(session, d1, 1, 300.0, 1.5, "break", date(2024, 1, 10))
    make_trade(session, d2, 1, -200.0, -1.0, "hard", date(2024, 1, 12))
    make_trade(session, d2, 2, 500.0, 2.5, "observe_timeout", date(2024, 1, 20))
    make_trade(session, d3, 1, -100.0, -0.5, "break", date(2024, 1, 15))
    make_trade(session, d4, 1, 100.0, 0.5, "max_sessions", date(2024, 1, 25))

    llm = make_run(session, pk(v1))
    l1 = decide(
        session, llm, xyz, date(2024, 1, 2), "llm",
        verdict="take", conviction=0.7, stances=["for", "against"],
    )  # fmt: skip
    # Citation-row order is not alphabetical: a sort by claim text would swap them.
    cite(session, l1, "Revenue grew 18% to $1,234.5 million.", revenue, QUOTE_SPACED)
    cite(session, l1, "Margin fell to 12.0% from 14.5%.", revenue, QUOTE_EXACT)
    make_trade(session, l1, 1, 200.0, 1.0, "break", date(2024, 1, 9))
    l3 = decide(
        session, llm, xyz, date(2024, 1, 8), "llm",
        verdict="pass", conviction=0.3, stances=["against"],
    )  # fmt: skip
    cite(session, l3, "Margin fell to 12.0% from 14.5%.", revenue, QUOTE_EXACT)
    l2 = decide(
        session, llm, xyz, date(2024, 1, 5), "llm",
        verdict="take", conviction=0.6, stances=["for"],
    )  # fmt: skip
    cite(session, l2, "Revenue is guided to $5.1 to $5.3 billion.", outlook, QUOTE_OUTLOOK)
    make_trade(session, l2, 1, 50.0, 0.25, "break", date(2024, 1, 11))
    decide(session, llm, abc, date(2024, 1, 16), "llm", verdict="no_call", conviction=None)

    stray = make_run(session, pk(v0))
    s1 = decide(session, stray, xyz, date(2024, 1, 2), "take_all")
    make_trade(session, s1, 1, 10_000.0, 9.0, "break", date(2024, 1, 11))
    return World(pk(base), pk(llm), pk(stray), xyz, abc, revenue, outlook)


def db_trades(session: Session, decision_id: int) -> list[Trade]:
    stmt = select(Trade).where(Trade.decision_id == decision_id).order_by(col(Trade.leg))
    return list(session.exec(stmt).all())


def run_decisions(session: Session, run_id: int) -> list[Decision]:
    stmt = select(Decision).where(Decision.run_id == run_id)
    return sorted(session.exec(stmt).all(), key=lambda d: (d.as_of, pk(d)))


def jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value))


def max_drawdown(curve: list[dict[str, Any]]) -> float:
    return max(point["drawdown"] for point in curve)


def by_day(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {datetime.fromisoformat(r["as_of"]).astimezone(ET).date().isoformat(): r for r in rows}


def leg(
    exit_at: datetime, pnl: float, *, leg_no: int = 1, decision_id: int = 1, run_id: int = 1
) -> Trade:
    """An in-memory Trade; equity_curve never touches the store."""
    entry = 100.0
    return Trade(
        decision_id=decision_id,
        run_id=run_id,
        strategy_id="earnings_gap",
        security_id="S_X",
        leg=leg_no,
        direction="long",
        entry_at=exit_at - timedelta(days=3),
        entry_price=entry,
        shares=50,
        key_level=98.0,
        hard_stop=96.0,
        risk_per_share=4.0,
        exit_at=exit_at,
        exit_price=entry + pnl / 50,
        exit_reason="break",
        pnl=pnl,
        r_multiple=pnl / 200.0,
        available_at=exit_at,
    )


# =============================================================================================
# normal path
# =============================================================================================


def test_exported_trades_equal_the_trade_rows(session: Session, world: World) -> None:
    rows = export_decisions(session, world.base, None)
    assert len(rows) == 4
    for row in rows:
        stored = db_trades(session, row["id"])
        assert [t["leg"] for t in row["trades"]] == [t.leg for t in stored]
        for exported, t in zip(row["trades"], stored, strict=True):
            assert set(exported) == set(TRADE_FIELDS)
            for name in TRADE_FIELDS:
                value = exported[name]
                if name in ("entry_at", "exit_at"):
                    assert isinstance(value, str)
                    assert datetime.fromisoformat(value) == getattr(t, name), name
                else:
                    assert value == getattr(t, name), name
        assert row["r"] == pytest.approx(sum(t.r_multiple for t in stored))


def test_take_all_decisions_are_takes_without_claims(session: Session, world: World) -> None:
    rows = export_decisions(session, world.base, None)
    assert [r["id"] for r in rows] == [pk(d) for d in run_decisions(session, world.base)]
    for row in rows:
        assert row["verdict"] == "take"
        assert row["claims"] == []
        assert row["base_r"] is None  # no base run given
        assert row["symbol"] == "XYZ"
        assert row["entity_id"] == pk(world.xyz)
        assert row["candidate"]["symbol"] == "XYZ"
        assert row["candidate"]["gap_low"] == 54.0
    assert [r["r"] for r in rows] == pytest.approx([1.5, 1.5, -0.5, 0.5])


def test_llm_decisions_carry_claims_in_citation_order(session: Session, world: World) -> None:
    rows = export_decisions(session, world.llm, world.base)
    # sorted by as_of, not by id (L3 was inserted before L2)
    days = [datetime.fromisoformat(r["as_of"]).astimezone(ET).date() for r in rows]
    assert days == [date(2024, 1, d) for d in (2, 5, 8, 16)]
    l1, l2, l3, l4 = rows
    assert [r["verdict"] for r in rows] == ["take", "take", "pass", "no_call"]
    assert [r["conviction"] for r in rows] == [0.7, 0.6, 0.3, None]
    assert l1["claims"] == [
        {
            "claim": "Revenue grew 18% to $1,234.5 million.",
            "stance": "for",
            "chunk_id": pk(world.revenue),
            "quote": QUOTE_SPACED,
        },
        {
            "claim": "Margin fell to 12.0% from 14.5%.",
            "stance": "against",
            "chunk_id": pk(world.revenue),
            "quote": QUOTE_EXACT,
        },
    ]
    assert [c["stance"] for c in l2["claims"]] == ["for"]
    assert l2["claims"][0]["chunk_id"] == pk(world.outlook)
    assert [c["stance"] for c in l3["claims"]] == ["against"]
    assert l4["claims"] == []
    assert l4["symbol"] == "ABC"
    assert l1["r"] == pytest.approx(1.0)
    assert l1["base_r"] == pytest.approx(1.5)  # D1: same entity, same as_of


def test_export_run_metrics_match_the_replay_report(session: Session, world: World) -> None:
    params = {"execute.stop.trail_atr": 1.5, "execute.sizing.risk_pct": 0.5}
    out = export_run(session, world.base, params)
    assert set(out) == {"run_id", "judge", "params", "metrics", "curve"}
    assert out["run_id"] == world.base
    assert out["judge"] == "take_all"
    assert out["params"] == params
    expected = dataclasses.asdict(replay_metrics(session, world.base))
    assert set(out["metrics"]) == set(expected)
    assert jsonable(out["metrics"]) == jsonable(expected)
    assert max_drawdown(out["curve"]) == pytest.approx(expected["max_drawdown"])
    assert expected["max_drawdown"] == pytest.approx(300 / 100_300)


def test_export_run_curve_steps_at_each_exit_day(session: Session, world: World) -> None:
    curve = export_run(session, world.base, {})["curve"]
    first, *steps = curve
    assert first["equity"] == EQUITY0
    assert first["drawdown"] == 0.0
    assert first["date"] <= steps[0]["date"]
    assert [p["date"] for p in steps] == [
        "2024-01-10",
        "2024-01-12",
        "2024-01-15",
        "2024-01-20",
        "2024-01-25",
    ]
    assert [p["equity"] for p in steps] == pytest.approx(
        [100_300.0, 100_100.0, 100_000.0, 100_500.0, 100_600.0]
    )
    assert [p["drawdown"] for p in steps] == pytest.approx(
        [0.0, 200 / 100_300, 300 / 100_300, 0.0, 0.0]
    )


def test_llm_run_reports_its_judge(session: Session, world: World) -> None:
    out = export_run(session, world.llm, {"judge.kind": "llm"})
    assert out["judge"] == "llm"
    assert max_drawdown(out["curve"]) == pytest.approx(
        replay_metrics(session, world.llm).max_drawdown
    )


def test_equity_curve_sums_pnl_and_tracks_the_peak() -> None:
    trades = [
        leg(datetime(2024, 3, 12, 13, 30, tzinfo=UTC), -300.0),  # given out of order
        leg(datetime(2024, 3, 5, 14, 30, tzinfo=UTC), 500.0),
        leg(datetime(2024, 3, 20, 14, 30, tzinfo=UTC), 100.0),
    ]
    curve = equity_curve(trades, 10_000.0, date(2024, 3, 1), date(2024, 3, 29))
    assert curve == [
        {"date": "2024-03-01", "equity": 10_000.0, "drawdown": 0.0},
        {"date": "2024-03-05", "equity": pytest.approx(10_500.0), "drawdown": 0.0},
        {
            "date": "2024-03-12",
            "equity": pytest.approx(10_200.0),
            "drawdown": pytest.approx(300 / 10_500),
        },
        {
            "date": "2024-03-20",
            "equity": pytest.approx(10_300.0),
            "drawdown": pytest.approx(200 / 10_500),
        },
    ]
    assert json.dumps(curve)


def test_every_export_is_json(session: Session, world: World) -> None:
    run = export_run(session, world.llm, {"judge.kind": "llm"})
    decisions = export_decisions(session, world.llm, world.base)
    chunks = export_chunks(
        session,
        [pk(world.revenue), pk(world.outlook)],
        {pk(world.revenue): [QUOTE_EXACT], pk(world.outlook): [QUOTE_OUTLOOK]},
    )
    timeline = export_timeline(
        session, [pk(world.xyz)], datetime(2024, 1, 1, tzinfo=ET), datetime(2024, 2, 1, tzinfo=ET)
    )
    for value in (run, decisions, chunks, timeline):
        json.dumps(value)


# =============================================================================================
# boundaries
# =============================================================================================


def test_equity_curve_without_trades_is_the_starting_point() -> None:
    curve = equity_curve([], 25_000.0, date(2024, 5, 1), date(2024, 5, 31))
    assert curve == [{"date": "2024-05-01", "equity": 25_000.0, "drawdown": 0.0}]


def test_a_run_without_trades(session: Session) -> None:
    entity = add(session, Entity(kind="company", ticker="NIL", name="Nil Inc.", cik=997))
    run = make_run(session, pk(version(session, 0)))
    decision = decide(
        session, run, entity, date(2024, 2, 6), "llm",
        verdict="pass", conviction=0.2, stances=[],
    )  # fmt: skip
    out = export_run(session, pk(run), {})
    assert out["judge"] == "llm"
    assert len(out["curve"]) == 1
    assert out["curve"][0]["equity"] == EQUITY0
    assert out["curve"][0]["drawdown"] == 0.0
    assert out["metrics"]["max_drawdown"] == 0.0
    assert out["metrics"]["total_return"] == 0.0
    (row,) = export_decisions(session, pk(run), None)
    assert row["id"] == pk(decision)
    assert row["trades"] == []
    assert row["r"] is None
    assert row["base_r"] is None


def test_a_decision_without_trades_has_no_r(session: Session, world: World) -> None:
    rows = by_day(export_decisions(session, world.llm, world.base))
    passed = rows["2024-01-08"]
    assert passed["verdict"] == "pass"
    assert passed["trades"] == []
    assert passed["r"] is None
    assert passed["base_r"] == pytest.approx(-0.5)  # D3 traded; L3 did not
    assert rows["2024-01-16"]["r"] is None


def test_two_legs_exiting_the_same_new_york_day_are_one_point() -> None:
    trades = [
        leg(datetime(2024, 4, 9, 13, 30, tzinfo=UTC), -200.0, leg_no=1),  # 09:30 New York
        leg(datetime(2024, 4, 9, 20, 0, tzinfo=UTC), 500.0, leg_no=2),  # 16:00 New York
    ]
    curve = equity_curve(trades, 100_000.0, date(2024, 4, 1), date(2024, 4, 30))
    assert len(curve) == 2
    assert curve[1]["date"] == "2024-04-09"
    assert curve[1]["equity"] == pytest.approx(100_300.0)
    # summed before the drawdown is taken: the intraday -200 is not a drawdown point
    assert curve[1]["drawdown"] == 0.0


def test_an_exit_late_in_utc_belongs_to_the_new_york_date() -> None:
    # 2024-01-10 23:30 UTC = 18:30 New York; 2024-01-11 03:00 UTC = 22:00 New York on 01-10.
    # By UTC date these are two days; by New York date one.
    trades = [
        leg(datetime(2024, 1, 10, 23, 30, tzinfo=UTC), 100.0),
        leg(datetime(2024, 1, 11, 3, 0, tzinfo=UTC), 50.0, decision_id=2),
    ]
    curve = equity_curve(trades, 100_000.0, date(2024, 1, 2), date(2024, 1, 31))
    assert [p["date"] for p in curve] == ["2024-01-02", "2024-01-10"]
    assert curve[-1]["equity"] == pytest.approx(100_150.0)


def test_a_data_gap_exit_at_20_00_new_york_keeps_its_session_date() -> None:
    # run.py stamps a data_gap exit at 20:00 New York: 01:00 UTC on the next calendar day.
    exit_at = datetime(2024, 1, 10, 20, 0, tzinfo=ET)
    assert exit_at.astimezone(UTC).date() == date(2024, 1, 11)
    curve = equity_curve([leg(exit_at, -80.0)], 100_000.0, date(2024, 1, 2), date(2024, 1, 31))
    assert curve[-1]["date"] == "2024-01-10"


def test_quotes_get_spans_in_the_original_text(session: Session, world: World) -> None:
    out = export_chunks(
        session,
        [pk(world.revenue), pk(world.outlook)],
        {pk(world.revenue): [QUOTE_EXACT, QUOTE_SPACED, QUOTE_MISSING]},
    )
    assert set(out) == {pk(world.revenue), pk(world.outlook)}
    revenue = out[pk(world.revenue)]
    assert set(revenue) == {"document_id", "section_path", "text", "spans"}
    assert revenue["text"] == CHUNK_REVENUE
    assert revenue["document_id"] == world.revenue.document_id
    assert revenue["section_path"] == world.revenue.section_path
    spans = [tuple(s) for s in revenue["spans"]]
    found = sorted(CHUNK_REVENUE[a:b] for a, b in spans)
    # exact match; the whitespace-insensitive one maps back onto the original characters;
    # the unfindable quote adds nothing
    assert found == sorted([QUOTE_EXACT, QUOTE_SPACED_ORIGINAL])
    assert len(spans) == 2
    exact = CHUNK_REVENUE.index(QUOTE_EXACT)
    assert (exact, exact + len(QUOTE_EXACT)) in spans
    # a chunk with no quotes is exported with no spans
    assert out[pk(world.outlook)]["spans"] == []
    assert out[pk(world.outlook)]["section_path"] == "EX-99.1 > Outlook"


def test_an_unfindable_quote_gives_no_span(session: Session, world: World) -> None:
    out = export_chunks(session, [pk(world.outlook)], {pk(world.outlook): [QUOTE_MISSING]})
    assert out[pk(world.outlook)]["spans"] == []


# =============================================================================================
# leakage: what was not yet public, or belongs to another run or day, stays out
# =============================================================================================


@dataclass(frozen=True)
class Docs:
    xyz: int
    abc: int
    start: datetime
    end: datetime
    ids: dict[str, int]


@pytest.fixture
def docs(session: Session) -> Docs:
    # Own entities and source, so a test may use this fixture together with ``world``.
    xyz = add(session, Entity(kind="company", ticker="TLX", name="Timeline X Inc.", cik=1999))
    abc = add(session, Entity(kind="company", ticker="TLA", name="Timeline A Corp.", cik=1998))
    source = new_source(session, name="sec_edgar_timeline")
    start = datetime(2024, 1, 2, 0, 0, tzinfo=ET)
    end = datetime(2024, 1, 31, 20, 0, tzinfo=ET)

    def doc(entity: Entity, available_at: datetime, source_time: datetime) -> int:
        return pk(
            new_document(
                session,
                source_id=pk(source),
                entity_id=pk(entity),
                available_at=available_at,
                source_time=source_time,
                doc_type="8-K",
            )
        )

    ids = {
        # describes a period inside the window and was public inside it
        "inside": doc(
            xyz, datetime(2024, 1, 5, 16, 10, tzinfo=ET), datetime(2024, 1, 3, tzinfo=ET)
        ),
        # describes a period inside the window but only became public after it: future
        "public_later": doc(
            xyz, datetime(2024, 2, 2, 8, 0, tzinfo=ET), datetime(2024, 1, 20, tzinfo=ET)
        ),
        # describes a period before the window, public inside it: visible
        "old_period": doc(
            xyz, datetime(2024, 1, 10, 7, 0, tzinfo=ET), datetime(2023, 12, 31, tzinfo=ET)
        ),
        "other_entity": doc(
            abc, datetime(2024, 1, 12, 9, 0, tzinfo=ET), datetime(2024, 1, 11, tzinfo=ET)
        ),
        "at_end": doc(xyz, end, datetime(2024, 1, 31, tzinfo=ET)),
        "at_start": doc(xyz, start, datetime(2023, 12, 29, tzinfo=ET)),
        "before": doc(
            xyz, datetime(2023, 12, 30, 9, 0, tzinfo=ET), datetime(2024, 1, 4, tzinfo=ET)
        ),
        # same available_at as "inside", inserted later: ordered after it by id
        "inside_tie": doc(
            xyz, datetime(2024, 1, 5, 16, 10, tzinfo=ET), datetime(2024, 1, 3, tzinfo=ET)
        ),
        # one microsecond after the end
        "just_after": doc(xyz, end + timedelta(microseconds=1), datetime(2024, 1, 25, tzinfo=ET)),
    }
    return Docs(pk(xyz), pk(abc), start, end, ids)


def test_timeline_filters_on_available_at_not_source_time(session: Session, docs: Docs) -> None:
    out = export_timeline(session, iter([docs.xyz]), docs.start, docs.end)
    ids = docs.ids
    assert [r["document_id"] for r in out] == [
        ids["at_start"],
        ids["inside"],
        ids["inside_tie"],
        ids["old_period"],
        ids["at_end"],
    ]
    assert ids["public_later"] not in {r["document_id"] for r in out}


def test_timeline_rows_have_no_source_time(session: Session, docs: Docs) -> None:
    out = export_timeline(session, [docs.xyz, docs.abc], docs.start, docs.end)
    assert docs.ids["other_entity"] in {r["document_id"] for r in out}
    for row in out:
        assert set(row) == {"document_id", "entity_id", "doc_type", "title", "available_at"}
        assert "source_time" not in row
        assert docs.start <= datetime.fromisoformat(row["available_at"]) <= docs.end
    assert "source_time" not in json.dumps(out)


def test_timeline_only_covers_the_entities_asked_for(session: Session, docs: Docs) -> None:
    out = export_timeline(session, [docs.abc], docs.start, docs.end)
    assert [r["document_id"] for r in out] == [docs.ids["other_entity"]]
    assert out[0]["entity_id"] == docs.abc
    assert export_timeline(session, [], docs.start, docs.end) == []


def test_base_r_never_pairs_another_day_or_entity(session: Session, world: World) -> None:
    rows = by_day(export_decisions(session, world.llm, world.base))
    # base has XYZ on 01-04, not 01-05: no pairing across days
    assert rows["2024-01-05"]["base_r"] is None
    assert rows["2024-01-05"]["r"] == pytest.approx(0.25)
    # base has XYZ on 01-16; this is ABC on 01-16: no pairing across entities
    assert rows["2024-01-16"]["base_r"] is None


def test_other_runs_trades_never_appear(session: Session, world: World) -> None:
    stray_ids = {
        pk(t) for t in session.exec(select(Trade).where(Trade.run_id == world.stray)).all()
    }
    assert stray_ids
    for run_id in (world.base, world.llm):
        rows = export_decisions(session, run_id, world.stray)
        own = {pk(d) for d in run_decisions(session, run_id)}
        assert {r["id"] for r in rows} == own
        for row in rows:
            for t in row["trades"]:
                assert t["pnl"] != 10_000.0
        out = export_run(session, run_id, {})
        assert max(p["equity"] for p in out["curve"]) < EQUITY0 + 10_000.0
    base_rows = export_decisions(session, world.base, None)
    assert sum(t["pnl"] for r in base_rows for t in r["trades"]) == pytest.approx(600.0)


def test_no_local_path_in_any_export(session: Session, world: World, docs: Docs) -> None:
    chunk_ids = [pk(world.revenue), pk(world.outlook)]
    dumps = [
        json.dumps(export_run(session, world.llm, {"judge.kind": "llm"})),
        json.dumps(export_run(session, world.base, {})),
        json.dumps(export_decisions(session, world.llm, world.base)),
        json.dumps(export_decisions(session, world.base, None)),
        json.dumps(export_chunks(session, chunk_ids, {pk(world.revenue): [QUOTE_SPACED]})),
        json.dumps(export_timeline(session, [docs.xyz, docs.abc], docs.start, docs.end)),
    ]
    for text in dumps:
        assert "/home/" not in text
        assert "/data/" not in text
        assert LOCAL_PATH not in text


def test_base_r_pairs_by_security_not_by_company(session: Session) -> None:
    # One company (one entity, one CIK) with two listed share classes: one earnings 8-K makes
    # both gap, and the replay decides each security on its own. Pairing by entity would let
    # one class take the other's base result.
    alpha = add(session, Entity(kind="company", ticker="GOOGL", name="Alpha Inc.", cik=1652))
    day = date(2024, 4, 26)
    base = make_run(session, pk(version(session, 0)))
    llm = make_run(session, pk(version(session, 1)))

    def on(run: Run, symbol: str, judge: str) -> Decision:
        d = decide(session, run, alpha, day, judge, verdict="take", conviction=0.6)
        d.payload = {**d.payload, "candidate": candidate(alpha, symbol)}
        return add(session, d)

    make_trade(session, on(base, "GOOGL", "take_all"), 1, 300.0, 1.5, "break", date(2024, 5, 3))
    make_trade(session, on(base, "GOOG", "take_all"), 1, -100.0, -0.5, "break", date(2024, 5, 3))
    on(llm, "GOOGL", "llm")
    on(llm, "GOOG", "llm")
    rows = {r["symbol"]: r for r in export_decisions(session, pk(llm), pk(base))}
    assert rows["GOOGL"]["base_r"] == pytest.approx(1.5)
    assert rows["GOOG"]["base_r"] == pytest.approx(-0.5)
