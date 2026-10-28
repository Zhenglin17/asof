"""What one block of held shares has seen and is owed, across splits and spin-offs.

Every simulator needs this whatever its trading rules: the lowest and highest price seen while
held (for MAE / MFE), kept in the price unit of the latest bar when a split changes it or a
spin-off's factor becomes known, and the spin-off children owed to the shares held at an
ex-date. The rules that decide when to buy and sell live in the simulators.
"""

from dataclasses import dataclass


def _lower(a: float | None, b: float | None) -> float | None:
    return b if a is None else a if b is None else min(a, b)


def _higher(a: float | None, b: float | None) -> float | None:
    return b if a is None else a if b is None else max(a, b)


@dataclass
class Holding:
    # Extremes of the bars held before the exit bar; None until one such bar has been seen.
    low: float | None = None
    high: float | None = None
    # Held at a spin-off's ex-date: the extremes from before it wait for the event's factor,
    # and the MAE / MFE basis is entry_price x basis_factor.
    pre_low: float | None = None
    pre_high: float | None = None
    basis_factor: float = 1.0
    entitled: bool = False  # children owed and not yet paid
    entitled_shares: int = 0  # shares held at the ex-date: a later split adds no child
    spinoff: str | None = None  # "adjust" / "restart": held through a spin-off's ex-date

    def see(self, low: float, high: float) -> None:
        self.low = _lower(self.low, low)
        self.high = _higher(self.high, high)

    def split(self, ratio: float) -> None:
        """A ``ratio``-for-1 split: every price seen so far in the new unit."""
        if self.low is not None and self.high is not None:
            self.low /= ratio
            self.high /= ratio
        if self.pre_low is not None and self.pre_high is not None:
            self.pre_low /= ratio
            self.pre_high /= ratio

    def begin_spinoff(self, shares: int) -> None:
        """The next bar is an ex-date: ``shares`` are owed the children, and the extremes seen
        so far wait for the event's factor."""
        self.entitled = True
        self.entitled_shares = shares
        self.pre_low, self.pre_high = (
            _lower(self.pre_low, self.low),
            _higher(self.pre_high, self.high),
        )
        self.low = self.high = None

    def adjust(self, factor: float) -> None:
        """The factor is known: the extremes from before the ex-date join the later ones."""
        self.spinoff = "adjust"
        self.basis_factor *= factor
        self._merge_pre(factor)

    def restart(self) -> None:
        """The event cannot be priced: raw levels again, no child."""
        self.entitled = False
        self.spinoff = "restart"
        self._merge_pre(1.0)

    def _merge_pre(self, factor: float) -> None:
        if self.pre_low is not None and self.pre_high is not None:
            self.low = _lower(self.low, self.pre_low * factor)
            self.high = _higher(self.high, self.pre_high * factor)
        self.pre_low = self.pre_high = None

    def excursions(self, entry_price: float, exit_price: float) -> tuple[float, float]:
        """MAE and MFE in % of the entry: the extremes seen plus the exit price. The exit bar
        adds only the exit price: a daily bar does not say whether its high or low came before
        or after the fill. With nothing seen yet the entry price stands in."""
        basis = entry_price * self.basis_factor
        held_low, held_high = _lower(self.low, self.pre_low), _higher(self.high, self.pre_high)
        if held_low is None or held_high is None:
            held_low = held_high = basis
        low, high = min(held_low, exit_price), max(held_high, exit_price)
        return 100 * (low / basis - 1), 100 * (high / basis - 1)
