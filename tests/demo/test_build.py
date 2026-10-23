"""Guards against a demo parameter grid that is not the sweep it claims to be: a missing or
duplicated set, a take_all set that carries an llm-only knob (and so counts as a separate attempt
for nothing), or an override that the strategy loader rejects only when the sweep is half done.

API under test (asof.demo.build):

    grid() -> list[list[str]]   override lists for ``asof replay run --set``
        take_all: execute.stop.trail_atr in {1.0, 1.5, 2.0} x execute.sizing.risk_pct in
                  {0.25, 0.5, 1.0}, no judge.min_conviction                       (9)
        llm:      judge.kind=llm x the same 9 x judge.min_conviction in {0.0, 0.55, 0.6}  (27)

No as_of: the grid is a list of strings and holds no time information, so there is nothing to
leak here. Assumptions only the real store can confirm (check when the sweep is first run):
- every one of the 36 sets replays offline without CacheMiss: min_conviction is applied after
  the model call, so it does not change the request, but trail_atr and risk_pct change which
  securities are still held on a later day and so which candidates get judged. Check: run
  ``asof replay run --set ...`` for every grid entry over the demo window and require exit 0.
- each set registers as its own StrategyVersion, so attempt_count after the sweep equals the
  count before plus the number of new sets. Check: attempt_count before and after the sweep.
- the take_all set (trail 1.5, risk 0.5) equals the committed defaults, yet an override list
  hashes differently from no overrides (content_hash adds overrides when any are given), so
  it may count as an attempt distinct from the base run. Check: content_hash of that set vs
  the base StrategyVersion row.
"""

from itertools import product
from pathlib import Path

from asof.demo.build import grid
from asof.strategy import apply_overrides, load_strategy

REPO_STRATEGY = Path(__file__).resolve().parents[2] / "strategies" / "earnings_gap"
TRAILS = (1.0, 1.5, 2.0)
RISKS = (0.25, 0.5, 1.0)
CONVICTIONS = (0.0, 0.55, 0.6)


def keys(overrides: list[str]) -> list[str]:
    return [o.partition("=")[0] for o in overrides]


def test_grid_has_36_distinct_override_lists() -> None:
    sets = grid()
    assert len(sets) == 36
    assert all(isinstance(o, list) and all(isinstance(s, str) for s in o) for o in sets)
    assert len({frozenset(o) for o in sets}) == 36
    for overrides in sets:
        # one value per setting within a list
        assert len(keys(overrides)) == len(set(keys(overrides)))
        assert all("=" in o for o in overrides)


def test_every_list_applies_and_covers_the_product() -> None:
    base = load_strategy(REPO_STRATEGY)
    seen: set[tuple[str, float, float, float | None]] = set()
    for overrides in grid():
        cfg = apply_overrides(base, overrides)
        kind = cfg.judge.kind
        conviction = cfg.judge.min_conviction if kind == "llm" else None
        seen.add((kind, cfg.execute.stop.trail_atr, cfg.execute.sizing.risk_pct, conviction))
    expected = {("take_all", t, r, None) for t, r in product(TRAILS, RISKS)} | {
        ("llm", t, r, c) for t, r, c in product(TRAILS, RISKS, CONVICTIONS)
    }
    assert seen == expected


def test_take_all_sets_carry_no_llm_knob() -> None:
    base = load_strategy(REPO_STRATEGY)
    take_all = [o for o in grid() if apply_overrides(base, o).judge.kind == "take_all"]
    llm = [o for o in grid() if apply_overrides(base, o).judge.kind == "llm"]
    assert len(take_all) == 9
    assert len(llm) == 27
    for overrides in take_all:
        assert "judge.min_conviction" not in keys(overrides)
    for overrides in llm:
        assert "judge.kind=llm" in overrides
        assert "judge.min_conviction" in keys(overrides)
        assert {"execute.stop.trail_atr", "execute.sizing.risk_pct"} <= set(keys(overrides))
