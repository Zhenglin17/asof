"""The security-master scenario worlds of tests/ingest/test_security_master*.py as plain inputs
(bars, SEC history, Alpaca name changes, SEC cutoff), for tests that need to rebuild a world
under other conditions -- the truncation test rebuilds each one from only the inputs knowable at
an instant.

Two ways of getting the inputs, both without copying a world's data:

* worlds held by a module-scoped fixture: the fixture's arguments are repeated here from the
  same module constants and bar builders (a fixture cannot be called directly);
* worlds made by a builder function that calls ``build_security_master`` itself (``f4_inputs``,
  ``e_world``, ``in_build``, ...): the builder is run once with the build recorded, and the
  recorded keyword arguments are the inputs.

Worlds built inline inside a single test function are not listed. Every world is built without
overrides, as in its own module; ``sec_history_start`` is passed only where the module does
(R1).
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from types import ModuleType
from typing import Any

import pyarrow as pa

from tests.ingest import test_security_master as base
from tests.ingest import test_security_master_copies as copies
from tests.ingest import test_security_master_rules as rules
from tests.ingest import test_security_master_takeovers as takeovers
from tests.ingest.test_security_master import (
    Life,
    Rename,
    bar_rows,
    bars_table,
    name_changes_table,
    relabel,
    sec_table,
)


@dataclass(frozen=True)
class Inputs:
    """What ``build_security_master`` is called with for one world."""

    bars: pa.Table
    sec_history: pa.Table
    name_changes: pa.Table
    sec_history_start: date | None = None


def _inputs(
    bars: list[dict[str, Any]],
    snapshots: list[date],
    lives: list[Life],
    renames: list[Rename],
) -> Inputs:
    return Inputs(bars_table(bars), sec_table(snapshots, lives), name_changes_table(renames))


def _recorded(module: ModuleType, call: Callable[[], object]) -> Inputs:
    """Run a world builder of ``module`` and return the inputs of the one build it makes."""
    calls: list[dict[str, Any]] = []
    original = module.build_security_master

    def record(**kwargs: Any) -> tuple[pa.Table, pa.Table]:
        calls.append(kwargs)
        return original(**kwargs)

    setattr(module, "build_security_master", record)  # noqa: B010
    try:
        call()
    finally:
        setattr(module, "build_security_master", original)  # noqa: B010
    assert len(calls) == 1, f"{call}: expected one build, saw {len(calls)}"
    [kwargs] = calls
    assert not list(kwargs.get("overrides", ())), "scenario worlds are built without overrides"
    return Inputs(
        kwargs["bars"],
        kwargs["sec_history"],
        kwargs["name_changes"],
        kwargs.get("sec_history_start"),
    )


# -- tests/ingest/test_security_master.py --------------------------------------------------------


def _world() -> Inputs:  # fixtures bars / sec_history / name_changes / built
    return _inputs(base.world_bars(), base.SNAPSHOTS, base.LIVES, base.RENAMES)


def _r1() -> Inputs:
    bars, sec, changes = base.r1_inputs()
    return Inputs(bars, sec, changes, base.R1_CUTOFF)


def _r2(mass: bool) -> Inputs:  # fixtures r2_mass / r2_few
    return _inputs(base.r2_bars(mass=mass), base.R2_SNAPSHOTS, base.r2_lives(mass=mass), [])


SCENARIOS: dict[str, Callable[[], Inputs]] = {
    "world": _world,
    "r1_stale_sec_table": _r1,
    "r2_mass_correction": lambda: _r2(True),
    "r2_few_switches": lambda: _r2(False),
    "r4": lambda: _inputs(base.r4_bars(), base.R4_SNAPSHOTS, base.R4_LIVES, base.R4_RENAMES),
    "r5": lambda: _inputs(base.r5_bars(), base.R5_SNAPSHOTS, base.R5_LIVES, base.R5_RENAMES),
    "f1": lambda: _inputs(base.f1_bars(), base.F1_SNAPSHOTS, base.F1_LIVES, base.F1_RENAMES),
    "f2": lambda: _inputs(base.f2_bars(), base.F2_SNAPSHOTS, base.F2_LIVES, base.F2_RENAMES),
    "f3": lambda: _inputs(base.f3_bars(), base.F3_SNAPSHOTS, base.F3_LIVES, base.F3_RENAMES),
    "f4_continuous": lambda: _recorded(base, lambda: base.f4_inputs(continuous=True)),
    "f4_jump": lambda: _recorded(base, lambda: base.f4_inputs(continuous=False)),
    "f6": lambda: _recorded(base, base.f6_inputs),
    "f7": lambda: _inputs(base.f7_bars(), base.F7_SNAPSHOTS, base.F7_LIVES, base.F7_RENAMES),
    "f9": lambda: _inputs(base.f9_bars(), base.F9_SNAPSHOTS, base.F9_LIVES, base.F9_RENAMES),
    "f9b": lambda: _inputs(base.f9b_bars(), base.F9_SNAPSHOTS, base.F9B_LIVES, base.F9B_RENAMES),
    "f9b_sec_only": lambda: _inputs(base.f9b_sec_bars(), base.F9_SNAPSHOTS, base.F9B_SEC_LIVES, []),
    "f10": lambda: _inputs(base.f10_bars(), base.F10_SNAPSHOTS, base.F10_LIVES, base.F10_RENAMES),
    "f11": lambda: _inputs(base.f11_bars(), base.F11_SNAPSHOTS, base.F11_LIVES, base.F11_RENAMES),
    "g2": lambda: _recorded(
        base,
        lambda: base.g2_inputs(
            last_traded=base.G2_LAST_TRADED, cut=base.G2_CUT, new_base=base.G2_NEW_BASE
        ),
    ),
    "g3": lambda: _inputs(
        base.g3_bars(),
        base.G3_SNAPSHOTS,
        [("ORIC", base.G3_CIK, date(2020, 1, 13), None)],
        base.G3_RENAMES,
    ),
    "h2": lambda: _inputs(base.h2_bars(), base.H2_SNAPSHOTS, base.H2_LIVES, base.H2_RENAMES),
    # -- tests/ingest/test_security_master_rules.py ---------------------------------------------
    "n_unit_spelling": lambda: Inputs(rules.n_bars(), rules.n_sec(), name_changes_table([])),
    "d_late_record": lambda: _inputs(
        rules.d_bars(), rules.D_SNAPSHOTS, rules.D_LIVES, rules.D_RENAMES
    ),
    "d_handover_5_days": lambda: _recorded(rules, lambda: rules.d_handover_world(5)),
    "d_handover_6_days": lambda: _recorded(rules, lambda: rules.d_handover_world(6)),
    "v_viac": lambda: _inputs(
        bar_rows("VIAC", rules.V_VIAC_FIRST_BAR, date(2022, 2, 15), 42.0),
        rules.V_VIAC_SNAPSHOTS,
        [("VIAC", rules.V_VIAC_CIK, date(2020, 2, 16), None)],
        rules.V_VIAC_RENAMES,
    ),
    "o_resumed": lambda: _inputs(rules.o_bars(), rules.O_SNAPSHOTS, rules.O_LIVES, rules.O_RENAMES),
    "a_hona": lambda: _inputs(
        rules.a_hona_bars(), rules.A_HONA_SNAPSHOTS, rules.A_HONA_LIVES, rules.A_HONA_RENAMES
    ),
    "a_aan": lambda: _inputs(
        rules.a_aan_bars(), rules.A_AAN_SNAPSHOTS, rules.A_AAN_LIVES, rules.A_AAN_RENAMES
    ),
    # -- tests/ingest/test_security_master_copies.py --------------------------------------------
    "b_cuk": lambda: _inputs(
        bar_rows("CUK", copies.STORE_START, copies.B_CUK_LAST, 27.0)
        + bar_rows("CCL", copies.STORE_START, copies.S[-1], 27.1),
        copies.B_CUK_SNAPSHOTS,
        copies.B_CUK_LIVES,
        [("CUK", "CCL", copies.B_CUK_RECORD, "14365C103", "143658300")],
    ),
    "b_gold": lambda: _recorded(copies, lambda: copies.b_gold_world(sec_lag=False)),
    "b_gold_sec_lag": lambda: _recorded(copies, lambda: copies.b_gold_world(sec_lag=True)),
    "e_run": lambda: _recorded(copies, lambda: copies.e_world(run=130, lead=0)),
    "f_gap": lambda: _recorded(copies, copies.f_gap_world),
    "f_cik_leaves": lambda: _recorded(copies, lambda: copies.f_cik_world(date(2024, 5, 6))),
    "z_resumed_89": lambda: _recorded(copies, lambda: copies.z_resumed_world(89)),
    "z_resumed_90": lambda: _recorded(copies, lambda: copies.z_resumed_world(90)),
    "relabel_para": lambda: _inputs(
        copies.r_bars(), copies.R_SNAPSHOTS, copies.R_LIVES, copies.R_RENAMES
    ),
    "c_clipped_overlap": lambda: _inputs(
        copies.c_bars(),
        copies.C_SNAPSHOTS,
        copies.C_LIVES,
        [("OLDC", "NEWC", copies.C_RECORD, "000400100", "000400100")],
    ),
    "g_conflict_kinds": lambda: _inputs(
        copies.g_bars(),
        copies.G_SNAPSHOTS,
        copies.G_LIVES,
        [
            ("CBSX", "VIAX", copies.G_VIAX_RECORD, "124857202", "92556H206"),
            ("TAKR", "LBL", copies.G_SWITCH, "000700100", "000700100"),
        ],
    ),
    # -- tests/ingest/test_security_master_takeovers.py -----------------------------------------
    "l_xper": lambda: _inputs(
        takeovers.xper_bars(),
        takeovers.XPER_SNAPSHOTS,
        takeovers.XPER_LIVES,
        takeovers.XPER_RENAMES,
    ),
    "l_arnc": lambda: _inputs(
        bar_rows("ARNC", date(2020, 1, 2), date(2020, 3, 31), 25.0)
        + bar_rows("ARNC", takeovers.ARNC_DAY, date(2020, 12, 31), 75.0)
        + bar_rows("HWM", takeovers.ARNC_DAY, date(2020, 12, 31), 18.0),
        takeovers.ARNC_SNAPSHOTS,
        takeovers.ARNC_LIVES,
        takeovers.ARNC_RENAMES,
    ),
    "r_gold_b": lambda: _inputs(
        takeovers.gb_bars(), takeovers.GB_SNAPSHOTS, takeovers.GB_LIVES, takeovers.GB_RENAMES
    ),
    "r_inbx": lambda: _recorded(takeovers, takeovers.in_build),
    "r_inbx_without_inxb_copy": lambda: _recorded(
        takeovers, lambda: takeovers.in_build(inxb_copy=False)
    ),
    "r_fgmc": lambda: _inputs(takeovers.fi_bars(), takeovers.FI_SNAPSHOTS, takeovers.FI_LIVES, []),
    "r_disca_wbd": lambda: _inputs(
        bar_rows("DISCA", date(2020, 1, 2), date(2022, 4, 1), 20.0)
        + takeovers.closes("DISCA", takeovers.DW_DISCA_DAYS)
        + takeovers.closes("WBD", takeovers.DW_WI_DAYS)
        + relabel(takeovers.closes("WBD", takeovers.DW_WI_DAYS), "WBDWV")
        + bar_rows("WBD", takeovers.DW_CUT, date(2022, 12, 30), 24.78),
        takeovers.DW_SNAPSHOTS,
        takeovers.DW_LIVES,
        [("WBDWV", "WBD", takeovers.DW_CUT, "934423104", "934423104")],
    ),
    "r_bbu_bbuc": lambda: _inputs(
        bar_rows("BBU", date(2020, 1, 2), date(2026, 3, 30), 25.0)
        + takeovers.closes(
            "BBUC.WI", [(date(2022, 3, 4), 30.0, 5_000), (date(2022, 3, 9), 30.5, 4_000)]
        )
        + bar_rows("BBUC", date(2022, 3, 15), date(2026, 6, 30), 28.0),
        takeovers.BB_SNAPSHOTS,
        takeovers.BB_LIVES,
        [("BBUC.WI", "BBUC", date(2022, 3, 15), "11259V106", "11259V106")],
    ),
}
