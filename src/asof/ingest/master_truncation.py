"""Truncation test for the security master: does any row tell a reader at ``as_of`` more than the
inputs knowable at ``as_of`` could?

Several build rules read data after an event: rule B reads SEC snapshots up to 183 days after a
rename record, the copy rules read bars on both sides of a takeover. A row stamped knowable at the
event may carry what they found, and no unit test on hand-made data says whether it does. This
module rebuilds the master from the inputs knowable at ``as_of`` (the *truncated* master) and
compares it with the full master as a reader at ``as_of`` sees it (``mask_master``):

* **leak**: the full master shows something at ``as_of`` the truncated build could not produce
  -- a segment it lacks, a field it has otherwise, a different grouping into securities, or days
  it gives to another segment. Leaks fail ``asof market truncation``.
* **delay**: the truncated build shows a segment on days the full master leaves to nobody at
  ``as_of`` (the full master learned later that the guess was wrong and dates its correction
  later), or a CIK the full master still withholds (AFC's zombie 3906 before the SEC's 2020-07-10
  correction). Nothing is seen early; a replay merely loses those facts until the correction.
* **info**: only ``evidence`` differs. It records how the build learned of a segment (a record,
  later confirmed by the SEC: name_change -> both) and no reader decides anything on it.

The truncated build only simulates "had we been collecting since then, from the vendor's history
as served today". Imported bars (``configs/bar_imports.yaml``) count from their bars' own
``available_at``, not from the day we imported them, exactly as the read path serves them.
Histories the vendor rewrote later (AXNX, volume revisions) are the audit's business, not this
test's: it checks the build, not the inputs.
"""

import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import yaml

from asof.ingest.security_master import MASTER_SCHEMA, _midnight

MICRO = timedelta(microseconds=1)
# Instants a human has looked at (00:00 New York of each): VIAC's late record, NKLA's SPAC
# completion, PARA's end as Paramount, FATBW, and COLAR -> SAIQ, which rule B will judge again
# once the 2027-04 snapshot exists.
NAMED_DAYS = (
    date(2020, 2, 13),
    date(2020, 6, 8),
    date(2025, 8, 7),
    date(2025, 7, 25),
    date(2026, 10, 5),
)
DIFF_SCHEMA = pa.schema(
    [
        ("kind", pa.string()),  # leak | delay | info
        # only_in_full | fields_differ | grouping_differs | replaced (leaks); only_in_truncated |
        # cik_withheld (delays); evidence_differs (info)
        ("reason", pa.string()),
        ("symbol", pa.string()),
        ("valid_from", pa.date32()),
        ("fields", pa.string()),  # differing gated fields, sorted, "+"-joined; evidence never
        ("detail", pa.string()),
    ]
)
COMPARED = ("valid_to", "cik", "cik_available_at", "cusip", "available_at", "end_available_at")
CIK_FIELDS = {"cik", "cik_available_at"}

Key = tuple[str, date]  # (symbol, valid_from): one segment


def _aware(as_of: datetime) -> datetime:
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    return as_of.astimezone(UTC)


def _override_day(override: Mapping[str, Any]) -> date:
    day = override.get("date")
    if day is None:
        raise ValueError(f"override {dict(override)!r} has no date: when was it decided?")
    if isinstance(day, datetime):
        return day.date()
    return day if isinstance(day, date) else date.fromisoformat(str(day))


def truncate_inputs(
    *,
    bars: pa.Table,
    sec_history: pa.Table,
    name_changes: pa.Table,
    overrides: Iterable[Mapping[str, Any]],
    as_of: datetime,
) -> tuple[pa.Table, pa.Table, pa.Table, list[Mapping[str, Any]]]:
    """The build inputs a reader at ``as_of`` had: bars by ``available_at``, SEC snapshots and
    human overrides from 00:00 New York of their date, Alpaca records by ``available_at`` (00:00
    of ``process_date`` when null, as ``edges_from_name_changes`` reads them)."""
    t = _aware(as_of)
    con = duckdb.connect()
    try:
        con.execute("SET TimeZone = 'UTC'")
        con.register("_bars", bars)
        kept_bars = (
            con.execute("SELECT * FROM _bars WHERE available_at <= $t", {"t": t})
            .to_arrow_table()
            .cast(bars.schema)
        )
    finally:
        con.close()
    snapshots = sec_history.column("snapshot_date").to_pylist()
    kept_sec = sec_history.filter(pa.array([_midnight(d) <= t for d in snapshots], pa.bool_()))
    known = [
        (r["available_at"] or _midnight(r["process_date"])) <= t
        for r in name_changes.select(["process_date", "available_at"]).to_pylist()
    ]
    kept_changes = name_changes.filter(pa.array(known, pa.bool_()))
    kept_overrides = [o for o in overrides if _midnight(_override_day(o)) <= t]
    return kept_bars, kept_sec, kept_changes, kept_overrides


def mask_master(master: pa.Table, as_of: datetime) -> pa.Table:
    """The master as a reader at ``as_of`` sees it: segments known by then, an end not yet
    knowable shown as open (``valid_to`` and ``end_available_at`` null), a CIK not yet knowable
    shown as unknown (``cik`` and ``cik_available_at`` null). Ordered by security_id,
    valid_from, symbol. ``visible_master`` is this on the file, plus its checks."""
    t = _aware(as_of)
    rows = []
    for r in master.select(MASTER_SCHEMA.names).to_pylist():
        if r["available_at"] is None or r["available_at"] > t:
            continue
        if r["end_available_at"] is None or r["end_available_at"] > t:
            r = {**r, "valid_to": None, "end_available_at": None}
        if r["cik_available_at"] is None or r["cik_available_at"] > t:
            r = {**r, "cik": None, "cik_available_at": None}
        rows.append(r)
    rows.sort(key=lambda r: (r["security_id"], r["valid_from"], r["symbol"]))
    return pa.Table.from_pylist(rows, schema=MASTER_SCHEMA)


def _canonical(rows: Sequence[dict[str, Any]]) -> dict[Key, str]:
    """Each segment's security named by its earliest segment: ids are positional (S000001 ...),
    so two builds number the same securities differently."""
    first: dict[str, tuple[date, str]] = {}
    for r in rows:
        mark = (r["valid_from"], r["symbol"])
        if r["security_id"] not in first or mark < first[r["security_id"]]:
            first[r["security_id"]] = mark
    return {
        (r["symbol"], r["valid_from"]): "{1}@{0}".format(*first[r["security_id"]]) for r in rows
    }


def _overlaps(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Do two segments of one symbol share a day? Half-open ``[valid_from, valid_to)``."""
    a_end, b_end = a["valid_to"] or date.max, b["valid_to"] or date.max
    return a["valid_from"] < b_end and b["valid_from"] < a_end


def compare(full: pa.Table, truncated: pa.Table, as_of: datetime) -> pa.Table:
    """Diff the full master and a master built from the inputs knowable at ``as_of``, both as a
    reader at ``as_of`` sees them. One row per differing segment, ``DIFF_SCHEMA``."""
    f_rows = mask_master(full, as_of).to_pylist()
    t_rows = mask_master(truncated, as_of).to_pylist()
    f_by = {(r["symbol"], r["valid_from"]): r for r in f_rows}
    t_by = {(r["symbol"], r["valid_from"]): r for r in t_rows}
    f_sec, t_sec = _canonical(f_rows), _canonical(t_rows)

    out: list[dict[str, Any]] = []

    def add(kind: str, reason: str, key: Key, detail: str, fields: Sequence[str] = ()) -> None:
        out.append(
            {"kind": kind, "reason": reason, "symbol": key[0], "valid_from": key[1],
             "fields": "+".join(sorted(fields)), "detail": detail}
        )  # fmt: skip

    for key in sorted(f_by):
        f = f_by[key]
        if key not in t_by:
            add("leak", "only_in_full", key, f"full: {f['security_id']} evidence {f['evidence']}")
            continue
        t = t_by[key]
        differ = [n for n in COMPARED if f[n] != t[n]]
        shown = differ + (["evidence"] if f["evidence"] != t["evidence"] else [])
        detail = "; ".join(f"{n} full={f[n]} truncated={t[n]}" for n in shown)
        if differ and set(differ) <= CIK_FIELDS and _withholds(f, t):
            add("delay", "cik_withheld", key, detail, differ)
        elif differ:
            add("leak", "fields_differ", key, detail, differ)
        elif shown:
            add("info", "evidence_differs", key, detail)
        # the grouping is a fact of its own: a withheld CIK must not hide it
        if f_sec[key] != t_sec[key]:
            add("leak", "grouping_differs", key, f"full: {f_sec[key]}; truncated: {t_sec[key]}")
    for key in sorted(t_by):
        if key in f_by:
            continue
        t = t_by[key]
        hit = [f for f in f_rows if f["symbol"] == key[0] and _overlaps(f, t)]
        if hit:
            owners = ", ".join(f"{f['symbol']}@{f['valid_from']}" for f in hit)
            add("leak", "replaced", key, f"full gives these days to {owners}")
        else:
            add("delay", "only_in_truncated", key, f"truncated: evidence {t['evidence']}")
    return pa.Table.from_pylist(out, schema=DIFF_SCHEMA)


def _withholds(full: Mapping[str, Any], truncated: Mapping[str, Any]) -> bool:
    """The full master says less about the CIK than a build then could: none yet, or the same
    one from a later instant. Never a different CIK."""
    if full["cik"] is None:
        return True
    return (
        full["cik"] == truncated["cik"]
        and truncated["cik_available_at"] is not None
        and full["cik_available_at"] > truncated["cik_available_at"]
    )


def default_instants(full_master: pa.Table) -> list[datetime]:
    """The named instants plus the earliest, (lower) median and latest ``available_at`` of each
    evidence kind and of ``cik_available_at``; each also one microsecond before. Sorted, UTC.
    (Every CIK instant would be one per SEC snapshot, ~2 hours on the real store; pass them with
    ``--as-of`` for a full sweep.)"""
    picked = {_midnight(d) for d in NAMED_DAYS}
    by_evidence: dict[str, set[datetime]] = {}
    for r in full_master.select(["evidence", "available_at"]).to_pylist():
        if r["evidence"] is not None and r["available_at"] is not None:
            by_evidence.setdefault(r["evidence"], set()).add(r["available_at"])
    ciks = {t for t in full_master.column("cik_available_at").to_pylist() if t is not None}
    for values in [*by_evidence.values(), *([ciks] if ciks else [])]:
        ordered = sorted(values)
        picked |= {ordered[0], ordered[(len(ordered) - 1) // 2], ordered[-1]}
    return sorted({t.astimezone(UTC) for p in picked for t in (p, p - MICRO)})


def load_instants(path: Path) -> list[datetime]:
    """``configs/truncation_instants.txt``: the instants a default run checks, one ISO timestamp
    with its offset per line, ``#`` comments. Fixed by hand rather than recomputed per run: the
    computed spread moves whenever the master does, and a leak seen only at an instant that
    drifted away would look fixed. Sorted, UTC; a missing file is empty."""
    if not path.exists():
        return []
    seen: set[datetime] = set()
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            t = datetime.fromisoformat(line)
        except ValueError:
            raise ValueError(f"{path}: not an ISO timestamp: {line!r}") from None
        if t.tzinfo is None:
            raise ValueError(f"{path}: {line!r} has no UTC offset")
        t = t.astimezone(UTC)
        if t in seen:
            raise ValueError(f"{path}: duplicated instant {line}")
        seen.add(t)
    return sorted(seen)


def diff_key(row: Mapping[str, Any]) -> str:
    """``SYMBOL:YYYY-MM-DD:reason[:field+field]``, the shape of an entry in the exceptions file.
    The fields are part of the key: an excepted row that starts differing in one more field is
    a new leak, not the reviewed one."""
    key = f"{row['symbol']}:{row['valid_from'].isoformat()}:{row['reason']}"
    return f"{key}:{row['fields']}" if row.get("fields") else key


def load_truncation_exceptions(path: Path) -> dict[str, str]:
    """``configs/truncation_exceptions.yaml``: reviewed leaks, each ``key / reason / date``.
    Returns key -> reason; a missing file is empty."""
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text()) or []
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a list of exceptions")
    loaded: dict[str, str] = {}
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError(f"{path}: entry {item!r} is not a mapping")
        for name in ("key", "reason", "date"):
            if item.get(name) is None or str(item[name]).strip() == "":
                raise ValueError(f"{path}: entry {item!r} needs a {name}")
        loaded[str(item["key"])] = str(item["reason"])
    return loaded


LEAK_REASONS = ("only_in_full", "fields_differ", "grouping_differs", "replaced")
_KEY = re.compile(r"^[^:\s]+:(\d{4}-\d{2}-\d{2}):(" + "|".join(LEAK_REASONS) + r")(:[^:\s]+)?$")


@dataclass(frozen=True)
class GateResult:
    """``new``: leaks neither file knows. ``fixed``: baseline keys that no longer occur (their
    lines must go, so the count only shrinks). ``stale_exceptions``: reviewed leaks that no
    longer occur, reported but never failing. The last two only from a full run."""

    new: list[str]
    fixed: list[str]
    stale_exceptions: list[str]

    @property
    def ok(self) -> bool:
        return not self.new and not self.fixed


def gate(
    found: Iterable[str],
    *,
    baseline: Iterable[str],
    exceptions: Mapping[str, str],
    full_run: bool,
) -> GateResult:
    """Judge the leak keys of a run against the baseline (known, to be fixed) and the exceptions
    (reviewed, accepted). A partial run visits only some instants, so a key it did not find may
    still occur elsewhere: it reports no ``fixed`` and no ``stale_exceptions``."""
    seen, known, reviewed = set(found), set(baseline), set(exceptions)
    both = sorted(known & reviewed)
    if both:
        raise ValueError(f"keys both in the baseline and the exceptions: {', '.join(both)}")
    return GateResult(
        new=sorted(seen - known - reviewed),
        fixed=sorted(known - seen) if full_run else [],
        stale_exceptions=sorted(reviewed - seen) if full_run else [],
    )


def _read_baseline(path: Path) -> tuple[list[str], list[str]]:
    """(header comment lines before the first key, keys in file order), checked."""
    header: list[str] = []
    keys: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            if not keys and line.startswith("#"):
                header.append(raw)
            continue
        match = _KEY.match(line)
        if match is None:
            raise ValueError(f"{path}: not a leak key: {line!r}")
        try:
            date.fromisoformat(match.group(1))
        except ValueError:
            raise ValueError(f"{path}: bad date in {line!r}") from None
        if line in keys:
            raise ValueError(f"{path}: duplicated key {line}")
        keys.append(line)
    return header, keys


def load_baseline(path: Path) -> list[str]:
    """``configs/truncation_baseline.txt``: known leaks not fixed yet, one ``diff_key`` per
    line, ``#`` comments. Lines are added by hand only; ``prune_baseline`` removes them. A
    missing file is empty."""
    return _read_baseline(path)[1] if path.exists() else []


def prune_baseline(path: Path, fixed: Iterable[str]) -> int:
    """Remove the ``fixed`` keys, keep the header, sort the rest. Never adds: a key not in the
    file is refused and nothing is written. Returns how many lines went."""
    header, keys = _read_baseline(path)
    drop = set(fixed)
    unknown = sorted(drop - set(keys))
    if unknown:
        raise ValueError(f"{path}: not in the baseline: {', '.join(unknown)}")
    kept = sorted(k for k in keys if k not in drop)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(f"{line}\n" for line in [*header, *kept]))
    os.replace(tmp, path)
    return len(keys) - len(kept)
