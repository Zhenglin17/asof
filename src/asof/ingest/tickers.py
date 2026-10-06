"""Ticker spelling. The canonical form is Alpaca's dot notation (``BRK.B``, ``F.PRD``).

SEC ticker tables spell the same securities with hyphens and their own suffixes; keeping two
spellings around would make one security look like two.
"""

import re

_VALID = re.compile(r"^[A-Z]+(\.[A-Z]+)?$")

# SEC suffix -> Alpaca suffix for the share types that are not plain classes.
_SUFFIXES = {"UN": "U", "WT": "WS", "W": "WS", "RT": "RT", "WI": "WI"}


def normalize_ticker(ticker: str) -> str:
    """Convert an SEC spelling to the Alpaca spelling; Alpaca spellings pass through."""
    symbol = ticker.strip().upper()
    if not symbol:
        raise ValueError("empty ticker")
    if symbol.count("-") > 1:
        raise ValueError(f"cannot normalize ticker {ticker!r}")
    if "-" not in symbol:
        return symbol
    base, suffix = symbol.split("-")
    if suffix in _SUFFIXES:
        return f"{base}.{_SUFFIXES[suffix]}"
    if suffix.startswith("PR"):  # preferred, already Alpaca-like: -PRA -> .PRA, -PR -> .PR
        return f"{base}.{suffix}"
    if suffix.startswith("P"):  # SEC short form: -PB -> .PRB, -P -> .PR
        return f"{base}.PR{suffix[1:]}"
    return f"{base}.{suffix}"


def is_valid_symbol(symbol: str) -> bool:
    """Letters, optionally one dot-suffix. Rejects Alpaca's ``X_DELISTED`` and CUSIP-like junk."""
    return _VALID.fullmatch(symbol) is not None
