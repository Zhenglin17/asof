"""The real-data truncation gate: ratchet what each leak changes for a backtest, not the leak.

``master_truncation.compare`` names every segment where the full security master differs from a
build from the inputs knowable at ``as_of`` (a diff key, ``SYM:YYYY-MM-DD:reason[:fields]``);
``truncation_impact.impact_at`` lists the price cells and symbol CIKs a reader then gets
differently. Most keys change neither (COHR / IIVI 2022-08 regroups identical prices), and an
exception for "this key" keeps passing when new data makes the same key start moving a price.
So the gate counts cells:

* a **price cell** is (security, day), a **cik cell** (symbol, day), one for each session of a
  leak impact row; delays are not counted (nothing is seen early).
* a cell is **covered** by every diff key of the same instant whose segment one of the cell's
  rows names (``full_segment`` or ``trunc_segment``). A cell is judged as a whole: the build
  filing SUNE's bars into JCS's security gives truncated-side rows that name only SUNE, while the
  full-side rows of the same cells name PEGY's grouping key (real store, 2026-10-20).
* a key's impact is the cells it covers summed over the instants, price and CIK apart: a key
  that starts reading one more wrong cell at any instant grows. (The largest instant alone would
  let a cell added at another one pass unseen.) A key whose cells change while their number
  does not still passes: the count holds no cell identity.

``configs/truncation_impact_baseline.txt`` holds the known impact per key, ``KEY price=N cik=M``,
added by hand and only ever lowered: a key found with impact and not listed is new, one above its
numbers grew, a cell no key covers is unmatched; all fail. On a full run a listed key without
impact is fixed (fails until its line goes) and one below its numbers shrank (a notice);
``--prune-baseline`` removes and tightens them. The numbers are sums over one list of instants,
recorded in the header as ``# instants fingerprint: <12 hex>``: a run on instants outside that
list can only tell new keys and unmatched cells, and once the list itself changes every default
run fails until ``--rebase-baseline`` re-sums the numbers (each change printed, in the diff).
"""

import hashlib
import os
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from asof.ingest.master_truncation import _KEY, diff_key
from asof.ingest.truncation_impact import InstantResult

_LINE = re.compile(r"^(\S+)\s+price=(\d+)\s+cik=(\d+)$")
_FINGERPRINT = re.compile(r"^#\s*instants fingerprint:\s*([0-9a-f]{12})\s*$")


@dataclass(frozen=True, order=True)
class ImpactCount:
    """Price cells and cik cells a key covers, summed over the instants."""

    price: int
    cik: int


@dataclass(frozen=True)
class UnmatchedCell:
    """A leak cell no diff key of its instant covers. ``owner``: a price cell's security name,
    a cik cell's symbol; ``segments``: the segments its leak rows name, sorted."""

    as_of: datetime
    what: str
    owner: str
    day: date
    segments: tuple[str, ...]


@dataclass(frozen=True)
class Attribution:
    counts: dict[str, ImpactCount]
    unmatched: list[UnmatchedCell]


def _segment(key: str) -> str:
    """``SYM:YYYY-MM-DD`` of ``SYM:YYYY-MM-DD:reason[:fields]``."""
    return ":".join(key.split(":")[:2])


def attribute(results: Sequence[InstantResult]) -> Attribution:
    """Each leak cell of each instant attributed to the leak keys of that instant's diff
    (module docstring): per key the cells it covers summed over the instants, (0, 0) left out;
    and every cell no key covers, sorted by (as_of, what, owner, day)."""
    total: defaultdict[str, dict[str, int]] = defaultdict(lambda: {"price": 0, "cik": 0})
    unmatched: list[UnmatchedCell] = []
    for result in sorted(results, key=lambda r: r.as_of):
        by_segment: defaultdict[str, set[str]] = defaultdict(set)
        for r in result.diff.to_pylist():
            if r["kind"] == "leak":
                key = diff_key(r)
                by_segment[_segment(key)].add(key)
        # (what, owner, day) -> (keys covering it, segments its rows name)
        cells: dict[tuple[str, str, date], tuple[set[str], set[str]]] = {}
        for r in result.impact.to_pylist():
            if r["kind"] != "leak":
                continue
            owner = r["security"] if r["what"] == "price" else r["symbol"]
            segments = {r["full_segment"], r["trunc_segment"]} - {None}
            keys = {key for segment in segments for key in by_segment.get(segment, ())}
            for day in r["sessions"]:
                covered, named = cells.setdefault((r["what"], owner, day), (set(), set()))
                covered |= keys
                named |= segments
        for (what, owner, day), (covered, named) in sorted(cells.items()):
            if not covered:
                cell = UnmatchedCell(result.as_of, what, owner, day, tuple(sorted(named)))
                unmatched.append(cell)
            for key in covered:
                total[key][what] += 1
    counts = {key: ImpactCount(n["price"], n["cik"]) for key, n in sorted(total.items())}
    return Attribution(counts=counts, unmatched=unmatched)


def _read(path: Path) -> tuple[list[str], dict[str, ImpactCount]]:
    """(header comment lines before the first entry, entries in file order), checked."""
    header: list[str] = []
    entries: dict[str, ImpactCount] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            if not entries and line.startswith("#"):
                header.append(raw)
            continue
        match = _LINE.match(line)
        key_match = None if match is None else _KEY.match(match.group(1))
        if match is None or key_match is None:
            raise ValueError(f"{path}: not 'KEY price=N cik=M' with a leak key: {line!r}")
        try:
            date.fromisoformat(key_match.group(1))
        except ValueError:
            raise ValueError(f"{path}: bad date in {line!r}") from None
        key, count = match.group(1), ImpactCount(int(match.group(2)), int(match.group(3)))
        if count == ImpactCount(0, 0):
            raise ValueError(f"{path}: {key} has no impact: it belongs in no baseline")
        if key in entries:
            raise ValueError(f"{path}: duplicated key {key}")
        entries[key] = count
    return header, entries


def load_impact_baseline(path: Path) -> dict[str, ImpactCount]:
    """``configs/truncation_impact_baseline.txt``: the known impact of each leak key, one
    ``KEY price=N cik=M`` per line, ``#`` comments. A missing file is empty (every impact is
    then new: a mistyped path fails the gate, never passes it)."""
    return _read(path)[1] if path.exists() else {}


def instants_fingerprint(instants: Iterable[datetime]) -> str:
    """12 hex digits naming a list of instants (UTC ISO, sorted, "|"-joined, SHA-256)."""
    joined = "|".join(t.isoformat() for t in sorted(instants))
    return hashlib.sha256(joined.encode()).hexdigest()[:12]


def baseline_fingerprint(path: Path) -> str | None:
    """The ``# instants fingerprint:`` of the baseline's header, None if it has none (or no
    file)."""
    if not path.exists():
        return None
    for line in _read(path)[0]:
        match = _FINGERPRINT.match(line.strip())
        if match:
            return match.group(1)
    return None


def prune_impact_baseline(
    path: Path,
    *,
    remove: Iterable[str],
    tighten: Mapping[str, ImpactCount],
    fingerprint: str | None = None,
) -> tuple[int, int]:
    """Remove the ``remove`` keys and lower the ``tighten`` ones to their new counts; keep the
    header, sort the rest. Never adds and never raises a number: anything else is refused and
    nothing is written. ``fingerprint``: the instants the counts were taken at, written into the
    header (replacing an earlier one). Returns (lines removed, lines tightened)."""
    header, entries = _read(path)
    if fingerprint is not None:
        if not re.fullmatch(r"[0-9a-f]{12}", fingerprint):
            raise ValueError(f"{path}: not a fingerprint: {fingerprint!r}")
        header = [h for h in header if not _FINGERPRINT.match(h.strip())]
        header.append(f"# instants fingerprint: {fingerprint}")
    drop = set(remove)
    unknown = sorted((drop | set(tighten)) - set(entries))
    if unknown:
        raise ValueError(f"{path}: not in the baseline: {', '.join(unknown)}")
    both = sorted(drop & set(tighten))
    if both:
        raise ValueError(f"{path}: both removed and tightened: {', '.join(both)}")
    for key, count in sorted(tighten.items()):
        old = entries[key]
        if count.price > old.price or count.cik > old.cik:
            raise ValueError(f"{path}: {key} would rise from {old} to {count}")
        if count == ImpactCount(0, 0):
            raise ValueError(f"{path}: {key} tightened to no impact: remove it instead")
    kept = {key: tighten.get(key, count) for key, count in entries.items() if key not in drop}
    lines = [*header, *(f"{k} price={c.price} cik={c.cik}" for k, c in sorted(kept.items()))]
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(f"{line}\n" for line in lines))
    os.replace(tmp, path)
    return len(drop), len(tighten)


def rebase_impact_baseline(
    path: Path, *, counts: Mapping[str, ImpactCount], fingerprint: str
) -> list[tuple[str, ImpactCount, ImpactCount | None]]:
    """Re-sum the baseline over a new instant list: each listed key takes its ``counts`` (up or
    down), a listed key without counts goes, the header names ``fingerprint``. Never adds a
    key: one in ``counts`` but not in the file is refused and nothing is written. Returns
    (key, old, new or None when removed) for every line that changed, sorted."""
    header, entries = _read(path)
    if not re.fullmatch(r"[0-9a-f]{12}", fingerprint):
        raise ValueError(f"{path}: not a fingerprint: {fingerprint!r}")
    unknown = sorted(set(counts) - set(entries))
    if unknown:
        raise ValueError(f"{path}: not in the baseline (add by hand): {', '.join(unknown)}")
    kept = {key: counts[key] for key in entries if key in counts}
    header = [h for h in header if not _FINGERPRINT.match(h.strip())]
    header.append(f"# instants fingerprint: {fingerprint}")
    lines = [*header, *(f"{k} price={c.price} cik={c.cik}" for k, c in sorted(kept.items()))]
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(f"{line}\n" for line in lines))
    os.replace(tmp, path)
    return [
        (key, old, kept.get(key)) for key, old in sorted(entries.items()) if kept.get(key) != old
    ]


@dataclass(frozen=True)
class GateResult:
    """``new``: keys with impact not in the baseline; ``grew``: above it in either component;
    ``shrank``: below it in some and above in none (a notice); ``fixed``: listed keys without
    impact. ``grew`` only from a run whose instants the baseline's are a superset of, the last
    two only from a full run; ``unmatched``: cells no key covers."""

    new: list[str]
    grew: list[str]
    shrank: list[str]
    fixed: list[str]
    unmatched: int

    @property
    def ok(self) -> bool:
        return not self.new and not self.grew and not self.fixed and self.unmatched == 0


def gate(
    found: Mapping[str, ImpactCount],
    *,
    baseline: Mapping[str, ImpactCount],
    unmatched: int,
    full_run: bool,
    comparable: bool = True,
) -> GateResult:
    """Judge a run's attribution against the baseline. A partial run on some of the baseline's
    instants sums fewer of them: a number above the baseline is growth, one below proves nothing
    (no ``shrank``, no ``fixed``). ``comparable`` False: the run's instants are not among the
    baseline's, its sums say nothing either way (no ``grew``); a full run is always comparable."""
    both = sorted(set(found) & set(baseline))
    grew = [
        k
        for k in both
        if (full_run or comparable)
        and (found[k].price > baseline[k].price or found[k].cik > baseline[k].cik)
    ]
    shrank = [k for k in both if k not in grew and found[k] != baseline[k]] if full_run else []
    return GateResult(
        new=sorted(set(found) - set(baseline)),
        grew=grew,
        shrank=shrank,
        fixed=sorted(set(baseline) - set(found)) if full_run else [],
        unmatched=unmatched,
    )
