"""What the truncation test's leaks change for a backtest at ``as_of``.

``master_truncation.compare`` lists every field where the full security master differs from one
built from the inputs knowable at ``as_of``. Most of those differences change nothing a backtest
reads: COHR / IIVI 2022-08 groups differently, yet both masters give II-VI's security the same
price rows (the full one through COHR, the build then through IIVI, Alpaca's identical copies).
A backtest reads exactly two things from the master:

* **price**: the daily rows a security shows on a day -- open, high, low, close, volume and the
  split recorded for the bar's symbol that day, compared as a set per (security, day); the
  symbol a row came through is not part of it.
* **cik**: the CIK a symbol shows on a day, which is how filings and news attach to it.

A **leak** is a price cell or CIK the full master shows that the build then could not (missing
or different there); a **delay** is one the build then had and the full master does not show
at ``as_of`` (it learned later and dates its correction later). Only leaks need fixing.

Securities are matched by name (``_canonical``: the earliest segment, ``SYM@YYYY-MM-DD``)
since ids are positional. ``liquid`` takes the full master's ids. Known limit: the instrument
class is a per-segment fact too and is not compared (the build has no class file).
"""

import tempfile
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import duckdb
import pyarrow as pa

from asof.ingest import security_master as sm
from asof.ingest.master_truncation import _aware, _canonical, compare, mask_master
from asof.parallel import parallel_map

IMPACT_SCHEMA = pa.schema(
    [
        ("kind", pa.string()),  # leak | delay
        ("what", pa.string()),  # price | cik
        ("security", pa.string()),  # price: the cell's security name; cik: null
        ("symbol", pa.string()),  # the symbol of the bars read
        ("full_segment", pa.string()),  # SYM:YYYY-MM-DD owning (symbol, day) in the full master
        ("trunc_segment", pa.string()),  # the same in the truncated master
        ("first_day", pa.date32()),
        ("last_day", pa.date32()),
        ("days", pa.int64()),  # distinct sessions, not the span
        ("detail", pa.string()),
        # the sorted distinct sessions of `days`: the gate counts and matches (security, day)
        # cells, and one bar can sit in rows of two cells
        ("sessions", pa.list_(pa.date32())),
    ]
)
_OWNER_SCHEMA = pa.schema(
    [
        ("symbol", pa.string()),
        ("valid_from", pa.date32()),
        ("valid_to", pa.date32()),
        # valid_to, or date.max while open: a join on "valid_to IS NULL OR day < valid_to" is
        # a nested loop over 19M bars x 15k segments (87 s); on this, a hash join (0.2 s)
        ("valid_until", pa.date32()),
        ("security", pa.string()),
        ("segment", pa.string()),
        ("cik", pa.int64()),
    ]
)
# Each worker's DuckDB: left alone it plans for 80% of the machine, four workers at once would
# each count on 25 GB. Above the limit joins and aggregates spill to disk.
DUCKDB_MEMORY_LIMIT = "3GB"
DUCKDB_THREADS = 4


@dataclass(frozen=True)
class InstantResult:
    """One instant of the truncation test: ``diff`` (``master_truncation.DIFF_SCHEMA``) and
    ``impact`` (``IMPACT_SCHEMA``) of the same build from the inputs knowable at ``as_of``."""

    as_of: datetime
    diff: pa.Table
    impact: pa.Table


def _owners(master: pa.Table, as_of: datetime) -> tuple[pa.Table, dict[str, str]]:
    """The master as a reader at ``as_of`` sees it, one row per segment with its security's
    name, and the name of each security id."""
    rows = mask_master(master, as_of).to_pylist()
    canon = _canonical(rows)
    owners = [
        {
            "symbol": r["symbol"],
            "valid_from": r["valid_from"],
            "valid_to": r["valid_to"],
            "valid_until": r["valid_to"] or date.max,
            "security": canon[(r["symbol"], r["valid_from"])],
            "segment": f"{r['symbol']}:{r['valid_from'].isoformat()}",
            "cik": r["cik"],
        }
        for r in rows
    ]
    names = {r["security_id"]: canon[(r["symbol"], r["valid_from"])] for r in rows}
    return pa.Table.from_pylist(owners, schema=_OWNER_SCHEMA), names


def impact_at(
    full: pa.Table,
    truncated: pa.Table,
    bars: pa.Table,
    splits: pa.Table,
    as_of: datetime,
    *,
    liquid: Collection[str] | None = None,
    exhaustive: bool = False,
) -> pa.Table:
    """Price cells and symbol CIKs a reader at ``as_of`` gets differently from the two masters
    (module docstring), grouped per (kind, what, security, symbol, full_segment,
    trunc_segment); ``IMPACT_SCHEMA``, ordered by what, kind, symbol, first_day, security.
    ``bars``: ``StoreInputs.bars``; only bars knowable at ``as_of`` with volume > 0 count, and
    only splits knowable then. ``liquid``: full-master security ids whose cells (and the CIKs
    of whose symbols) are kept, on either side; None keeps everything. A leak cell both sides
    have is reported through the bars of both, so every segment whose bars made it differ is
    named. Only cells holding a bar whose owner or CIK differs between the sides are built;
    ``exhaustive`` builds them all (the same result, for tests)."""
    t = _aware(as_of)
    full_owners, full_names = _owners(full, t)
    trunc_owners, _ = _owners(truncated, t)
    scope = None if liquid is None else sorted({full_names[i] for i in liquid if i in full_names})

    # where DuckDB spills past its memory limit (none by default in memory: it fails instead);
    # one per call, so workers share no files
    spill = tempfile.TemporaryDirectory(prefix="asof-impact-")
    con = duckdb.connect()
    try:
        con.execute("SET TimeZone = 'UTC'")
        con.execute("SET temp_directory = $dir", {"dir": spill.name})
        con.execute(f"SET memory_limit = '{DUCKDB_MEMORY_LIMIT}'")
        con.execute(f"SET threads = {DUCKDB_THREADS}")
        # Arrow scans carry no row counts (every one is estimated at ~1 row), so DuckDB may
        # flip "bars LEFT JOIN segments" into a RIGHT join that hashes millions of bars: past
        # 3 GB at 2023-10-03. Keep the hash on the right side as written: segments, cand.
        con.execute("SET disabled_optimizers = 'build_side_probe_side'")
        con.register("_bars", bars)
        con.register("_splits", splits)
        con.register("_f", full_owners)
        con.register("_t", trunc_owners)
        con.register("_scope", pa.table({"security": pa.array(scope or [], pa.string())}))
        in_scope = "TRUE" if scope is None else "{} IN (SELECT security FROM _scope)"
        # a cell's or a CIK's security outside the liquid tier on both sides is read by nobody
        # we test for; its symbols' bars need not even be joined
        symbols = (
            ""
            if scope is None
            else "AND b.symbol IN (SELECT symbol FROM _f WHERE security IN (SELECT security "
            "FROM _scope) UNION SELECT symbol FROM _t WHERE security IN (SELECT security "
            "FROM _scope))"
        )

        def owned(joins: str = "") -> str:
            """Counted bars with their owner on each side (and ``joins``), owned on one."""
            return f"""
                FROM _bars b
                LEFT JOIN _f f ON f.symbol = b.symbol AND b.session_date >= f.valid_from
                              AND b.session_date < f.valid_until
                LEFT JOIN _t t ON t.symbol = b.symbol AND b.session_date >= t.valid_from
                              AND b.session_date < t.valid_until
                {joins}
                WHERE b.available_at <= $t AND b.volume > 0 {symbols}
                  AND (f.security IS NOT NULL OR t.security IS NOT NULL)
                """

        differs = "(f.security IS DISTINCT FROM t.security OR f.cik IS DISTINCT FROM t.cik)"
        # A cell can differ only if one of its bars has another owner on the other side: both
        # sides read the same bar rows. One streaming pass finds those bars (few), and only
        # their cells are built and compared -- not 19M bars' worth on the real store.
        con.execute(
            f"""
            CREATE TEMP TABLE cand AS
            SELECT DISTINCT unnest([f.security, t.security]) AS security, b.session_date
            {owned()} AND {"TRUE" if exhaustive else differs}
            """,
            {"t": t},
        )
        # an equi-join, not EXISTS (... IN (f, t)): an OR inside a correlated subquery is not
        # turned into a hash join and scans cand once per bar. One side suffices: a bar outside
        # `differs` has the same owner on both.
        joins = """
            LEFT JOIN sp ON sp.symbol = b.symbol AND sp.ex_date = b.session_date
            LEFT JOIN cand cf ON cf.security = f.security AND cf.session_date = b.session_date
            """
        con.execute(
            f"""
            CREATE TEMP TABLE o AS
            WITH sp AS (
                SELECT symbol, ex_date, list_sort(list(factor)) AS factors
                FROM _splits WHERE available_at <= $t GROUP BY ALL
            )
            SELECT b.symbol, b.session_date,
                   struct_pack(o := b.open, h := b.high, l := b.low, c := b.close,
                               v := b.volume, s := sp.factors) AS row,
                   f.security AS f_security, f.segment AS f_segment, f.cik AS f_cik,
                   t.security AS t_security, t.segment AS t_segment, t.cik AS t_cik
            {owned(joins)}
              AND ({differs} OR cf.security IS NOT NULL)
            """,
            {"t": t},
        )
        for side in ("f", "t"):
            con.execute(
                f"""
                CREATE TEMP TABLE {side}c AS
                SELECT {side}_security AS security, session_date, list_sort(list(row)) AS cell
                FROM o WHERE {side}_security IS NOT NULL GROUP BY ALL
                """
            )
        found = con.execute(
            f"""
            WITH r AS (
                SELECT 'leak' AS kind, 'price' AS what, o.f_security AS security, o.symbol,
                       o.f_segment AS full_segment, o.t_segment AS trunc_segment,
                       o.session_date, tc.cell IS NULL AS missing, NULL AS f_cik, NULL AS t_cik
                FROM o
                JOIN fc ON fc.security = o.f_security AND fc.session_date = o.session_date
                LEFT JOIN tc ON tc.security = o.f_security AND tc.session_date = o.session_date
                WHERE (tc.cell IS NULL OR fc.cell <> tc.cell) AND {in_scope.format("o.f_security")}
                UNION ALL
                -- the other side of a cell both have: its bars made the difference too (a
                -- symbol the build filed into this security is named here, nowhere else)
                SELECT 'leak', 'price', o.t_security, o.symbol, o.f_segment, o.t_segment,
                       o.session_date, FALSE, NULL, NULL
                FROM o
                JOIN tc ON tc.security = o.t_security AND tc.session_date = o.session_date
                JOIN fc ON fc.security = o.t_security AND fc.session_date = o.session_date
                WHERE fc.cell <> tc.cell AND {in_scope.format("o.t_security")}
                UNION ALL
                SELECT 'delay', 'price', o.t_security, o.symbol, o.f_segment, o.t_segment,
                       o.session_date, TRUE, NULL, NULL
                FROM o
                JOIN tc ON tc.security = o.t_security AND tc.session_date = o.session_date
                LEFT JOIN fc ON fc.security = o.t_security AND fc.session_date = o.session_date
                WHERE fc.cell IS NULL AND {in_scope.format("o.t_security")}
                UNION ALL
                SELECT CASE WHEN f_cik IS NULL THEN 'delay' ELSE 'leak' END, 'cik', NULL,
                       symbol, f_segment, t_segment, session_date, FALSE, f_cik, t_cik
                FROM o
                WHERE f_cik IS DISTINCT FROM t_cik AND (f_cik IS NOT NULL OR t_cik IS NOT NULL)
                  AND ({in_scope.format("f_security")} OR {in_scope.format("t_security")})
            )
            SELECT kind, what, security, symbol, full_segment, trunc_segment,
                   min(session_date) AS first_day, max(session_date) AS last_day,
                   count(DISTINCT session_date) AS days,
                   CASE
                     WHEN what = 'cik' THEN
                       'full cik ' || coalesce(string_agg(DISTINCT f_cik::VARCHAR, ','), 'none')
                       || '; truncated cik '
                       || coalesce(string_agg(DISTINCT t_cik::VARCHAR, ','), 'none')
                     WHEN kind = 'leak' THEN
                       'full reads ' || security || ' through ' || symbol || '; truncated: '
                       || count(DISTINCT session_date) FILTER (WHERE missing)
                       || ' days no such row, '
                       || count(DISTINCT session_date) FILTER (WHERE NOT missing)
                       || ' days other rows'
                     ELSE
                       'truncated reads ' || security || ' through ' || symbol
                       || '; full has no such row'
                   END AS detail,
                   list_sort(list(DISTINCT session_date)) AS sessions
            FROM r
            GROUP BY kind, what, security, symbol, full_segment, trunc_segment
            ORDER BY what, kind, symbol, first_day, security NULLS FIRST,
                     full_segment NULLS FIRST, trunc_segment NULLS FIRST
            """
        ).to_arrow_table()
    finally:
        con.close()
        spill.cleanup()
    return found.cast(IMPACT_SCHEMA)


def _result_at(
    run: tuple[
        pa.Table, "sm.StoreInputs", list[Mapping[str, Any]], pa.Table, frozenset[str] | None
    ],
    t: datetime,
) -> InstantResult:
    full, inputs, overrides, splits, liquid = run
    truncated, _ = sm.build_from_inputs(inputs, overrides, as_of=t)
    return InstantResult(
        as_of=t,
        diff=compare(full, truncated, t),
        impact=impact_at(full, truncated, inputs.bars, splits, t, liquid=liquid),
    )


def run_instants(
    full: pa.Table,
    inputs: "sm.StoreInputs",
    overrides: Sequence[Mapping[str, Any]],
    splits: pa.Table,
    instants: Iterable[datetime],
    *,
    liquid: Collection[str] | None = None,
    jobs: int = 1,
    progress: Callable[[InstantResult], None] | None = None,
) -> list[InstantResult]:
    """One build from ``inputs`` truncated at each distinct instant, and of it both ``compare``
    with the full master and ``impact_at``; sorted by instant (UTC). ``jobs`` builds run at once
    in forked workers that inherit the inputs (``parallel_map``); the result never depends on
    it (``jobs=1`` runs in this process, the reference). ``progress`` sees each result in
    instant order as soon as it and all earlier ones are done. A failed build raises here once
    every earlier instant is done; queued builds are cancelled and running workers terminated."""
    if jobs < 1:
        raise ValueError(f"jobs must be at least 1, got {jobs}")
    ts = sorted({_aware(t) for t in instants})
    run = (full, inputs, list(overrides), splits, None if liquid is None else frozenset(liquid))
    return parallel_map(
        _result_at,
        run,
        ts,
        jobs=jobs,
        progress=None if progress is None else lambda _, result: progress(result),
    )
