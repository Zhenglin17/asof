"""The traded GAPX path shared by the replay and CLI tests (store: tests/scan/world.py).

ATR 2, key 54 (zone 53.5 .. 54.5, hard 52), pre-earnings close 50 (zone 49.5 .. 50.5, hard 48),
decision close 56 -> 125 shares. Hand-made numbers.
"""

from tests.scan.world import HISTORY, SESSIONS, Ohlcv

X = SESSIONS[HISTORY:]  # X[0] = D, the reaction day
START = SESSIONS[HISTORY - 2]  # the replay starts after the lookback begins
FILLER: Ohlcv = (48.0, 48.5, 47.5, 48.0, 1_000_000)
TRADED: list[Ohlcv | None] = [
    (56.5, 57.0, 56.0, 56.8, 1_200_000),  # X1: leg 1 in at 56.5
    (56.0, 56.2, 53.0, 53.4, 1_500_000),  # X2: close 53.4 < 53.5 -> break
    (53.4, 53.5, 50.2, 50.8, 1_500_000),  # X3: leg 1 out at 53.4; re-entry confirmed
    (51.0, 51.5, 47.0, 47.5, 1_500_000),  # X4: leg 2 in at 51.0, hard stop 48 hit
    *[FILLER] * 22,  # X5 .. X26
]
END = X[len(TRADED)]
