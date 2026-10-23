"""A store where GAPX reports twice inside one re-entry window, for the judge-window and comparison
tests (base store: tests/scan/world.py, traded path: tests/replay/scenario.py; provenance there).

GAPX gaps on D = X[0] after the release of 02-13 and trades the scenario path (out by X4). A
second earnings 8-K is accepted at X[5] 16:05 New York and GAPX gaps again on X[6]
(previous close 48 -> open 52, +8.3%; volume 5M against about 1.2M). X[6] is inside D's 20-session
re-entry window, so a replay that took D still holds GAPX and skips the second candidate, while a
replay that passed on D judges it. With ``othr_on_d`` OTHR also gaps on D on the same path, with its
own two-exhibit release, so D has two candidates (S_GAPX, S_OTHR).

Numbers are made up; the second gap is chosen well past every scan threshold.
"""

from dataclasses import dataclass
from pathlib import Path

from sqlmodel import Session

from tests.replay.scenario import TRADED, X
from tests.replay.test_replay_llm import add_chunks, chunk_ids
from tests.scan.world import EARNINGS, GAP, RELEASE, Ohlcv, World, et, filing, gap_world

SECOND = 6  # X[SECOND] is the second reaction day
SECOND_GAP: Ohlcv = (52.0, 53.0, 51.5, 52.5, 5_000_000)
SECOND_RELEASE = et(X[SECOND - 1], 16, 5)


@dataclass
class TwoGaps:
    world: World
    first_chunks: list[int]  # GAPX D release: [revenue, outlook]
    second_docs: list[int]
    second_chunks: list[int]
    othr_docs: list[int]
    othr_chunks: list[int]


def two_gaps(tmp_path: Path, session: Session, *, othr_on_d: bool = False) -> TwoGaps:
    after: list[Ohlcv | None] = list(TRADED)
    after[SECOND - 1] = SECOND_GAP  # after[i] is the bar of X[i + 1]
    othr_bars = None
    if othr_on_d:
        othr_after = {X[i + 1]: bar for i, bar in enumerate(TRADED) if bar is not None}
        othr_bars = {X[0]: GAP, **othr_after}
    world = gap_world(tmp_path, session, after=after, othr_bars=othr_bars)
    second = filing(
        session,
        source_id=world.source_id,
        entity_id=world.gapx,
        accession="0000001111-26-000011",
        available_at=SECOND_RELEASE,
        exhibits=EARNINGS,
    )
    othr_docs: list[int] = []
    if othr_on_d:
        othr_docs = filing(
            session,
            source_id=world.source_id,
            entity_id=world.othr,
            accession="0000002222-26-000009",
            available_at=RELEASE,
            exhibits=EARNINGS,
        )
    add_chunks(session, world.earnings_docs)
    add_chunks(session, second)
    if othr_docs:
        add_chunks(session, othr_docs)
    return TwoGaps(
        world=world,
        first_chunks=chunk_ids(session, world.earnings_docs),
        second_docs=second,
        second_chunks=chunk_ids(session, second),
        othr_docs=othr_docs,
        othr_chunks=chunk_ids(session, othr_docs) if othr_docs else [],
    )
