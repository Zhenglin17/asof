"""Build the demo page's data: real replays over a small parameter grid, exported as JSON.

Every replay runs on a private copy of the metadata database, so the sweep never adds strategy
versions or runs to the research store: the page's grid is a display, not a search, and must
not count as attempts against the significance gate. Each worker unit is one (trail, risk) pair:
its take-all replay and the three llm replays that differ only in ``min_conviction`` share one
copy, so every llm decision can be paired with the take-all decision on the same candidate. One
more unit replays take-all with the defaults over the long window.

Only derived numbers are written (equity, P&L, R, metrics, a benchmark index), never raw bars.
"""

import json
import shutil
import sqlite3
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from itertools import product
from pathlib import Path
from typing import Any

import duckdb
from sqlalchemy import func
from sqlmodel import Session, col, select

from asof.demo.export import (
    ET,
    export_chunks,
    export_decisions,
    export_run,
    export_timeline,
)
from asof.parallel import parallel_map
from asof.replay.run import close_of, run_replay
from asof.store.db import make_engine
from asof.store.market import visible_bars
from asof.store.models import Chunk, Document, Entity
from asof.strategy import attempt_count, load_strategy, replay_count

TRAILS = (1.0, 1.5, 2.0)
RISKS = (0.25, 0.5, 1.0)
CONVICTIONS = (0.0, 0.5, 0.6)
LONG = "long"


def _pair(trail: float, risk: float) -> list[str]:
    return [f"execute.stop.trail_atr={trail}", f"execute.sizing.risk_pct={risk}"]


def grid() -> list[list[str]]:
    """The override lists of the sweep: 9 take-all sets, then 27 llm sets."""
    take_all = [_pair(t, r) for t, r in product(TRAILS, RISKS)]
    llm = [
        ["judge.kind=llm", *_pair(t, r), f"judge.min_conviction={c}"]
        for t, r, c in product(TRAILS, RISKS, CONVICTIONS)
    ]
    return take_all + llm


def run_key(judge: str, trail: float, risk: float, conviction: float | None = None) -> str:
    """The page's name for one grid point, e.g. ``llm|1.5|0.5|0.6``."""
    parts = [judge, str(trail), str(risk)] + ([] if conviction is None else [str(conviction)])
    return "|".join(parts)


@dataclass(frozen=True)
class Context:
    db: Path
    root: Path
    strategy_dir: Path
    start: date
    end: date
    long_start: date
    scratch: Path


def _replay(session: Session, ctx: Context, start: date, overrides: list[str]) -> int:
    con = duckdb.connect()
    try:
        return run_replay(session, con, ctx.root, ctx.strategy_dir, start, ctx.end, overrides)
    finally:
        con.close()


def _snapshot(src: Path, dst: Path) -> None:
    """A consistent copy of a SQLite database, including commits still in its write-ahead log
    (a plain file copy would miss them, or catch a half-written page)."""
    source = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    target = sqlite3.connect(dst)
    try:
        source.backup(target)
    finally:
        target.close()
        source.close()


def _unit(ctx: Context, unit: tuple[float, float] | str) -> dict[str, Any]:
    """Replays of one unit on its own database copy, exported; the copy is deleted after. The
    copy is the one file a unit writes, private to it, so units stay independent."""
    name = unit if isinstance(unit, str) else f"{unit[0]}-{unit[1]}"
    copy = ctx.scratch / f"meta-{name}.db"
    _snapshot(ctx.db, copy)
    engine = make_engine(copy)
    runs: dict[str, Any] = {}
    decisions: dict[str, Any] = {}
    try:
        with Session(engine) as session:
            if isinstance(unit, str):
                run_id = _replay(session, ctx, ctx.long_start, [])
                runs[LONG] = export_run(
                    session, run_id, {"judge": "take_all"}, start=ctx.long_start
                )
                return {"runs": runs, "decisions": decisions}
            trail, risk = unit
            base = _replay(session, ctx, ctx.start, _pair(trail, risk))
            key = run_key("take_all", trail, risk)
            params = {"judge": "take_all", "trail_atr": trail, "risk_pct": risk}
            runs[key] = export_run(session, base, params, start=ctx.start)
            decisions[key] = export_decisions(session, base, None)
            for conviction in CONVICTIONS:
                overrides = ["judge.kind=llm", *_pair(trail, risk)]
                overrides.append(f"judge.min_conviction={conviction}")
                run_id = _replay(session, ctx, ctx.start, overrides)
                key = run_key("llm", trail, risk, conviction)
                runs[key] = export_run(
                    session,
                    run_id,
                    {**params, "judge": "llm", "min_conviction": conviction},
                    start=ctx.start,
                )
                decisions[key] = export_decisions(session, run_id, base)
    finally:
        engine.dispose()
        copy.unlink(missing_ok=True)
    return {"runs": runs, "decisions": decisions}


def _benchmark(root: Path, start: date, end: date, equity0: float) -> list[dict[str, Any]]:
    """SPY bought at the first session's open with the same starting equity, valued at each
    close (price only). An index of equity, not the bars themselves."""
    con = duckdb.connect()
    try:
        bars = visible_bars(
            con,
            root,
            "1Day",
            close_of(end),
            symbols=["SPY"],
            start=datetime.combine(start, time(0), tzinfo=ET),
            end=close_of(end),
        ).to_pylist()
    finally:
        con.close()
    if not bars:
        return []
    first = bars[0]["open"]
    return [
        {"date": b["t"].astimezone(ET).date().isoformat(), "equity": equity0 * b["close"] / first}
        for b in bars
    ]


def _code_version() -> str:
    """The commit the data was built from, ``-dirty`` if the working tree had changes."""
    try:
        out = subprocess.run(
            ["git", "describe", "--always", "--dirty", "--abbrev=40"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, separators=(",", ":"), sort_keys=True))


def build(
    db: Path,
    root: Path,
    strategy_dir: Path,
    out: Path,
    *,
    start: date,
    end: date,
    long_start: date,
    jobs: int = 1,
) -> dict[str, int]:
    """Run the grid and the long take-all replay, write the page's JSON files to ``out`` and
    return how many rows each file holds. Reads ``db`` but never writes it."""
    out.mkdir(parents=True, exist_ok=True)
    units: list[tuple[float, float] | str] = [LONG, *product(TRAILS, RISKS)]
    with tempfile.TemporaryDirectory(prefix="asof-demo-") as scratch:
        ctx = Context(db, root, strategy_dir, start, end, long_start, Path(scratch))
        results = parallel_map(_unit, ctx, units, jobs=jobs)
    runs = {k: v for r in results for k, v in r["runs"].items()}
    decisions = {k: v for r in results for k, v in r["decisions"].items()}
    if len(runs) != len(grid()) + 1:
        raise RuntimeError(f"{len(runs)} runs built, the grid has {len(grid())} plus the long one")

    cited: dict[int, list[str]] = {}
    for rows in decisions.values():
        for row in rows:
            for claim in row["claims"]:
                quotes = cited.setdefault(claim["chunk_id"], [])
                if claim["quote"] not in quotes:
                    quotes.append(claim["quote"])
    entity_ids = sorted({row["entity_id"] for rows in decisions.values() for row in rows})

    cfg = load_strategy(strategy_dir)
    equity0 = float(cfg.execute.account.initial_equity)
    snapshot = Path(tempfile.mkdtemp(prefix="asof-demo-")) / "meta.db"
    _snapshot(db, snapshot)
    engine = make_engine(snapshot)
    try:
        with Session(engine) as session:
            chunks = export_chunks(session, cited, cited)
            timeline = export_timeline(
                session,
                entity_ids,
                datetime.combine(start, time(0), tzinfo=ET),
                close_of(end),
            )
            entities = {
                e.id: {"ticker": e.ticker, "name": e.name}
                for e in session.exec(select(Entity).where(col(Entity.id).in_(entity_ids))).all()
            }
            counts = {
                "documents": session.exec(select(func.count()).select_from(Document)).one(),
                "chunks": session.exec(select(func.count()).select_from(Chunk)).one(),
            }
            # Each copy counted the grid's own sets; the page shows the research store's count.
            searched = {
                "attempt_count": attempt_count(session, cfg.strategy.name),
                "replay_count": replay_count(session, cfg.strategy.name),
            }
    finally:
        engine.dispose()
        shutil.rmtree(snapshot.parent, ignore_errors=True)
    for run in runs.values():
        run["metrics"].update(searched)

    meta = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "code_version": _code_version(),
        "strategy": cfg.strategy.name,
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "long_window": {"start": long_start.isoformat(), "end": end.isoformat()},
        "model": cfg.judge.model,
        "training_cutoff": _iso_date(cfg.judge.training_cutoff),
        "grid": {"trail_atr": TRAILS, "risk_pct": RISKS, "min_conviction": CONVICTIONS},
        "defaults": {
            "trail_atr": cfg.execute.stop.trail_atr,
            "risk_pct": cfg.execute.sizing.risk_pct,
        },
        "store": counts,
        # The grid is itself a search the page lets a reader make; the research count excludes it.
        "grid_points": len(grid()),
        "research_attempts": searched,
        "benchmark": "SPY bought at the first open, price only (dividends excluded)",
    }
    files = {
        "meta.json": meta,
        "runs.json": {
            "runs": runs,
            "benchmark": _benchmark(root, start, end, equity0),
            "long_benchmark": _benchmark(root, long_start, end, equity0),
        },
        "decisions.json": decisions,
        "chunks.json": chunks,
        "timeline.json": timeline,
        "entities.json": entities,
    }
    for name, value in files.items():
        _write(out / name, value)
    return {
        "runs": len(runs),
        "decisions": sum(len(v) for v in decisions.values()),
        "chunks": len(chunks),
        "spans_missing": sum(1 for c in chunks.values() if not c["spans"]),
        "timeline": len(timeline),
    }


def _iso_date(value: date | None) -> str | None:
    return None if value is None else value.isoformat()
