"""Guards against one security living under two spellings (SEC "BRK-B" vs Alpaca "BRK.B")."""

import pytest

from asof.ingest.tickers import is_valid_symbol, normalize_ticker

# --- normal path -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("AAPL", "AAPL"),
        ("BRK-B", "BRK.B"),
        ("AAIC-PB", "AAIC.PRB"),
        ("XYZ-PRA", "XYZ.PRA"),
        ("ABC-P", "ABC.PR"),
        ("AAC-UN", "AAC.U"),
        ("AAC-WT", "AAC.WS"),
        ("AAC-W", "AAC.WS"),
        ("ABC-RT", "ABC.RT"),
        ("ABC-WI", "ABC.WI"),
    ],
)
def test_sec_hyphen_spellings_become_alpaca_dot_spellings(raw: str, expected: str) -> None:
    assert normalize_ticker(raw) == expected


@pytest.mark.parametrize("already", ["BRK.B", "F.PRD", "EVEX.WS", "AAPL"])
def test_alpaca_spellings_pass_through_unchanged(already: str) -> None:
    assert normalize_ticker(already) == already


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("aapl", "AAPL"), (" AAPL ", "AAPL"), ("brk-b", "BRK.B"), ("\tf.prd\n", "F.PRD")],
)
def test_whitespace_is_stripped_and_case_is_upper(raw: str, expected: str) -> None:
    assert normalize_ticker(raw) == expected


def test_normalize_is_idempotent() -> None:
    for raw in ["BRK-B", "AAIC-PB", "AAC-UN", "AAC-WT", "ABC-P", "AAPL"]:
        once = normalize_ticker(raw)
        assert normalize_ticker(once) == once


# --- boundaries --------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["", "   ", "A-B-C", "BRK-B-X"])
def test_empty_or_doubly_hyphenated_tickers_are_rejected(bad: str) -> None:
    with pytest.raises(ValueError):
        normalize_ticker(bad)


@pytest.mark.parametrize("symbol", ["AAPL", "BRK.B", "F.PRD", "EVEX.WS", "A", "ZZZZ.U"])
def test_valid_symbols_are_accepted(symbol: str) -> None:
    assert is_valid_symbol(symbol) is True


@pytest.mark.parametrize(
    "symbol", ["CIC_DELISTED", "384CNT069", "BRK-B", "", "aapl", "A.B.C", "AAPL.", ".B", "AA PL"]
)
def test_invalid_symbols_are_rejected(symbol: str) -> None:
    assert is_valid_symbol(symbol) is False


# --- leakage -----------------------------------------------------------------------------------


def test_normalization_does_not_invent_a_different_security() -> None:
    # A hyphen spelling and its dot spelling must map to one symbol, and nothing else may map
    # onto that symbol: otherwise bars of one security would be read as another's.
    assert normalize_ticker("BRK-B") == normalize_ticker("BRK.B")
    assert normalize_ticker("BRK-A") != normalize_ticker("BRK-B")
    assert normalize_ticker("ABC-P") != normalize_ticker("ABC-PRA")
    assert normalize_ticker("AAC-UN") != normalize_ticker("AAC-WT")
