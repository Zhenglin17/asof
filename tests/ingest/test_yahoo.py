"""Guards against the second bar source handing back bars that do not look like Alpaca's: a
Yahoo chart row must become a ``Bar`` stamped the way Alpaca stamps a daily bar (midnight New
York of the session), priced to the cent, with the right session date across the DST change,
and nothing a replay would mistake for data (holiday padding, a session before ``start``, a
monthly series Yahoo silently substitutes for a daily one, an error body) may get through.

Target API (asof.ingest.yahoo; implementation pending):

    class YahooError(Exception)
    class YahooClient:
        __init__(self, *, fetch: Fetch | None = None, sleep: Callable[[float], None] = time.sleep)
        daily_bars(self, symbol: str, start: date, end: date) -> tuple[list[Bar], bytes]

and ``Bar.trade_count`` becomes ``int | None`` (Yahoo has no trade count).

Fake data provenance (all measured 2026-10-16 against Yahoo's chart API unless marked):
- Body shape: ``chart.result[0]`` with ``meta.dataGranularity``, ``timestamp`` (epoch seconds at
  the session open: 14:30 UTC in winter, 13:30 UTC in summer), ``indicators.quote[0]`` with
  open/high/low/close/volume lists, ``indicators.adjclose``; error form ``result: null`` with
  ``error.code`` / ``error.description`` (measured: "No data found, symbol may be delisted").
- SKYD rows 2022-02-16 .. 02-18 (``fixtures/yahoo_skyd_2022-02.json``) match Alpaca's VIAC bars
  of those days: open 30.24 / 29.51 / 29.00, close 29.58 / 29.35 / 28.38, volume about 5% lower
  than Alpaca's. Yahoo serialises 29.58 as 29.579999923706055 (measured), hence the 4-decimal
  rounding.
- Yahoo may return the session before ``period1`` (measured); rows outside ``[start, end]`` by
  session date are dropped.
- ``range=max`` silently downgrades the series to monthly (measured): a granularity other than
  "1d" is an error, never data.
- Without a ``User-Agent`` header Yahoo returns nothing (measured); the header is the client's
  business and is not asserted here.
- Holidays are padded with null rows (assumption): a row whose close is null is skipped.
- vwap = typical price (high + low + close) / 3 rounded to 4 decimals (project decision
  2026-10-16, no measurement behind it); trade_count = None.

No ``as_of`` lives in this module; the bars it returns are filtered by ``import_bars`` (not
final at ``fetched_at``) and by ``visible_bars`` (``available_at``) like any other bar. Real-data
checks for the assumptions above:
- On the overlapping days (2022-02-14 .. 02-16) Yahoo's SKYD open/close equal Alpaca's VIAC
  open/close to the cent and the volume ratio stays within 0.9 .. 1.0 (standard case after the
  import: compare ``visible_bars`` for VIAC and the raw Yahoo body).
- Null-padded rows: count rows with a null close in a one-year Yahoo body for SKYD; they must
  fall on NYSE holidays only.
- ``asof market audit`` P2 (impossible prices) and P6 (availability rule) must stay at 0 after
  the import.
"""

import json
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from asof.ingest.alpaca import Bar
from asof.ingest.yahoo import YahooClient, YahooError
from tests.ingest.fakes import FakeTime, ScriptedFetch, make_bar, raw_reply, reply

ET = ZoneInfo("America/New_York")
FIXTURE = Path(__file__).parent / "fixtures" / "yahoo_skyd_2022-02.json"
CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/SKYD"
# 09:30 New York in winter (EST) is 14:30 UTC; in summer (EDT) 13:30 UTC.
T_0216, T_0217, T_0218 = 1645021800, 1645108200, 1645194600
T_0601_EDT = 1654090200  # 2022-06-01 13:30 UTC


def ny(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, tzinfo=ET)


def chart(
    rows: Sequence[tuple[int, float | None, float | None, float | None, float | None, int | None]],
    granularity: str = "1d",
) -> dict[str, Any]:
    """A chart body from ``(timestamp, open, high, low, close, volume)`` rows."""
    return {
        "chart": {
            "result": [
                {
                    "meta": {"dataGranularity": granularity, "symbol": "SKYD"},
                    "timestamp": [r[0] for r in rows],
                    "indicators": {
                        "quote": [
                            {
                                "open": [r[1] for r in rows],
                                "high": [r[2] for r in rows],
                                "low": [r[3] for r in rows],
                                "close": [r[4] for r in rows],
                                "volume": [r[5] for r in rows],
                            }
                        ],
                        "adjclose": [{"adjclose": [r[4] for r in rows]}],
                    },
                }
            ],
            "error": None,
        }
    }


ERROR_BODY = {
    "chart": {
        "result": None,
        "error": {"code": "Not Found", "description": "No data found, symbol may be delisted"},
    }
}


def client(*results, clock: FakeTime | None = None) -> tuple[YahooClient, ScriptedFetch]:
    fetch = ScriptedFetch(results)
    return YahooClient(fetch=fetch, sleep=(clock or FakeTime()).sleep), fetch


def fields(bar: Bar) -> tuple:
    return (bar.symbol, bar.t, bar.open, bar.high, bar.low, bar.close, bar.volume, bar.vwap)


# =============================================================================================
# normal path
# =============================================================================================


def test_fixture_rows_become_bars_in_alpacas_daily_convention() -> None:
    body = FIXTURE.read_bytes()
    yahoo, fetch = client(raw_reply(200, body))

    bars, raw = yahoo.daily_bars("SKYD", date(2022, 2, 16), date(2022, 2, 18))

    assert raw == body
    assert len(fetch.calls) == 1
    assert [fields(b) for b in bars] == [
        ("SKYD", ny(date(2022, 2, 16)), 30.24, 30.33, 27.84, 29.58, 87606200, 29.25),
        ("SKYD", ny(date(2022, 2, 17)), 29.51, 30.45, 29.1, 29.35, 12100500, 29.6333),
        ("SKYD", ny(date(2022, 2, 18)), 29.0, 29.64, 27.62, 28.38, 17020000, 28.5467),
    ]
    assert all(b.trade_count is None for b in bars)
    assert all(isinstance(b.volume, int) for b in bars)
    # the Bar validator normalises t to UTC; the instant is midnight New York, like Alpaca's
    assert bars[0].t == datetime(2022, 2, 16, 5, 0, tzinfo=UTC)
    assert bars[0].t.utcoffset() is not None


def test_request_is_the_chart_endpoint_with_new_york_midnight_bounds() -> None:
    yahoo, fetch = client(reply(200, chart([])))

    yahoo.daily_bars("SKYD", date(2022, 2, 16), date(2022, 2, 18))

    [(url, params)] = fetch.calls
    assert url == CHART_URL
    assert params == {
        "period1": str(int(ny(date(2022, 2, 16)).timestamp())),  # 2022-02-16 05:00 UTC
        "period2": str(int(ny(date(2022, 2, 19)).timestamp())),  # the day after end, 05:00 UTC
        "interval": "1d",
        "includePrePost": "false",
    }
    assert params["period1"] == "1644987600" and params["period2"] == "1645246800"


def test_symbol_goes_into_the_url_and_onto_every_bar() -> None:
    yahoo, fetch = client(reply(200, chart([(T_0217, 1.0, 2.0, 0.5, 1.5, 10)])))

    bars, _ = yahoo.daily_bars("VIAC", date(2022, 2, 17), date(2022, 2, 17))

    assert fetch.urls == ["https://query1.finance.yahoo.com/v8/finance/chart/VIAC"]
    assert [b.symbol for b in bars] == ["VIAC"]


def test_bar_model_accepts_a_missing_trade_count_and_still_an_integer_one() -> None:
    bar = Bar(
        symbol="SKYD",
        t=ny(date(2022, 2, 17)),
        open=1.0,
        high=2.0,
        low=0.5,
        close=1.5,
        volume=10,
        trade_count=None,
        vwap=1.3333,
    )
    assert bar.trade_count is None
    assert make_bar().trade_count == 7  # Alpaca bars are unchanged


def test_bars_come_back_sorted_by_time_whatever_the_body_order() -> None:
    rows = [
        (T_0218, 29.0, 29.64, 27.62, 28.38, 17020000),
        (T_0216, 30.24, 30.33, 27.84, 29.58, 87606200),
        (T_0217, 29.51, 30.45, 29.1, 29.35, 12100500),
    ]
    yahoo, _ = client(reply(200, chart(rows)))

    bars, _ = yahoo.daily_bars("SKYD", date(2022, 2, 16), date(2022, 2, 18))

    assert [b.t for b in bars] == [ny(date(2022, 2, d)) for d in (16, 17, 18)]


# =============================================================================================
# boundaries
# =============================================================================================


def test_prices_are_rounded_to_four_decimals() -> None:
    # measured: Yahoo serialises 29.58 as 29.579999923706055
    rows = [
        (
            T_0216,
            30.239999771118164,
            30.329999923706055,
            27.84000015258789,
            29.579999923706055,
            87606200,
        )
    ]
    yahoo, _ = client(reply(200, chart(rows)))

    [bar], _ = yahoo.daily_bars("SKYD", date(2022, 2, 16), date(2022, 2, 16))

    assert (bar.open, bar.high, bar.low, bar.close) == (30.24, 30.33, 27.84, 29.58)
    assert bar.vwap == round((30.33 + 27.84 + 29.58) / 3, 4) == 29.25


def test_summer_timestamp_maps_to_its_new_york_session() -> None:
    # 2022-06-01 13:30 UTC is 09:30 EDT: session 2022-06-01, t = 2022-06-01 00:00-04:00
    yahoo, _ = client(reply(200, chart([(T_0601_EDT, 1.0, 2.0, 0.5, 1.5, 10)])))

    [bar], _ = yahoo.daily_bars("SKYD", date(2022, 6, 1), date(2022, 6, 1))

    assert bar.t == ny(date(2022, 6, 1))
    assert bar.t == datetime(2022, 6, 1, 4, 0, tzinfo=UTC)


def test_empty_timestamp_list_is_no_bars_and_the_body() -> None:
    body = json.dumps(chart([])).encode()
    yahoo, _ = client(raw_reply(200, body))

    assert yahoo.daily_bars("SKYD", date(2022, 2, 16), date(2022, 2, 18)) == ([], body)


def test_rows_with_a_null_close_are_skipped() -> None:
    # assumption: Yahoo pads holidays with null rows (2022-02-21 is Presidents' Day)
    t_0221 = T_0218 + 3 * 86400
    rows = [
        (T_0218, 29.0, 29.64, 27.62, 28.38, 17020000),
        (t_0221, None, None, None, None, None),
    ]
    yahoo, _ = client(reply(200, chart(rows)))

    bars, _ = yahoo.daily_bars("SKYD", date(2022, 2, 18), date(2022, 2, 22))

    assert [b.t for b in bars] == [ny(date(2022, 2, 18))]


def test_rows_outside_start_and_end_by_session_date_are_dropped() -> None:
    # measured: Yahoo may return the session before period1; the day after ``end`` is the
    # symmetric case (period2 is exclusive but the filter must not rely on that alone)
    rows = [
        (T_0216, 30.24, 30.33, 27.84, 29.58, 87606200),
        (T_0217, 29.51, 30.45, 29.1, 29.35, 12100500),
        (T_0218, 29.0, 29.64, 27.62, 28.38, 17020000),
    ]
    yahoo, _ = client(reply(200, chart(rows)))

    bars, _ = yahoo.daily_bars("SKYD", date(2022, 2, 17), date(2022, 2, 17))

    assert [b.t for b in bars] == [ny(date(2022, 2, 17))]


def test_start_and_end_are_inclusive() -> None:
    rows = [
        (T_0216, 30.24, 30.33, 27.84, 29.58, 87606200),
        (T_0218, 29.0, 29.64, 27.62, 28.38, 17020000),
    ]
    yahoo, _ = client(reply(200, chart(rows)))

    bars, _ = yahoo.daily_bars("SKYD", date(2022, 2, 16), date(2022, 2, 18))

    assert [b.t for b in bars] == [ny(date(2022, 2, 16)), ny(date(2022, 2, 18))]


def test_duplicate_timestamps_keep_the_last_row() -> None:
    rows = [
        (T_0217, 1.0, 2.0, 0.5, 1.5, 10),
        (T_0217, 29.51, 30.45, 29.1, 29.35, 12100500),
    ]
    yahoo, _ = client(reply(200, chart(rows)))

    bars, _ = yahoo.daily_bars("SKYD", date(2022, 2, 17), date(2022, 2, 17))

    assert [(b.t, b.close, b.volume) for b in bars] == [(ny(date(2022, 2, 17)), 29.35, 12100500)]


def test_error_body_raises() -> None:
    yahoo, _ = client(reply(200, ERROR_BODY))

    with pytest.raises(YahooError, match="delisted"):
        yahoo.daily_bars("SKYD", date(2022, 2, 16), date(2022, 2, 18))


def test_granularity_other_than_daily_raises() -> None:
    # measured: range=max comes back as a monthly series without complaint
    yahoo, _ = client(reply(200, chart([(T_0216, 30.24, 30.33, 27.84, 29.58, 87606200)], "1mo")))

    with pytest.raises(YahooError, match="1mo"):
        yahoo.daily_bars("SKYD", date(2022, 2, 16), date(2022, 2, 18))


def test_non_json_body_raises_a_yahoo_error_not_a_decode_error() -> None:
    yahoo, _ = client(raw_reply(200, b"<html>blocked</html>"))

    with pytest.raises(YahooError):
        yahoo.daily_bars("SKYD", date(2022, 2, 16), date(2022, 2, 18))


def test_client_error_status_raises_without_retrying() -> None:
    clock = FakeTime()
    yahoo, fetch = client(reply(404, ERROR_BODY), clock=clock)

    with pytest.raises(YahooError, match="404"):
        yahoo.daily_bars("SKYD", date(2022, 2, 16), date(2022, 2, 18))

    assert len(fetch.calls) == 1
    assert clock.sleeps == []


@pytest.mark.parametrize("status", [429, 500, 502, 503])
def test_retryable_status_is_retried_three_times_then_raises(status: int) -> None:
    clock = FakeTime()
    yahoo, fetch = client(*(reply(status) for _ in range(4)), clock=clock)

    with pytest.raises(YahooError):
        yahoo.daily_bars("SKYD", date(2022, 2, 16), date(2022, 2, 18))

    assert len(fetch.calls) == 4  # the first try plus three retries
    assert len(clock.sleeps) == 3
    assert all(s > 0 for s in clock.sleeps)


def test_retryable_status_followed_by_success_returns_the_bars() -> None:
    clock = FakeTime()
    rows = [(T_0217, 29.51, 30.45, 29.1, 29.35, 12100500)]
    yahoo, fetch = client(reply(503), reply(429), reply(200, chart(rows)), clock=clock)

    bars, _ = yahoo.daily_bars("SKYD", date(2022, 2, 17), date(2022, 2, 17))

    assert [b.close for b in bars] == [29.35]
    assert len(fetch.calls) == 3
    assert len(clock.sleeps) == 2


def test_start_after_end_is_rejected_before_any_request() -> None:
    yahoo, fetch = client()

    with pytest.raises(ValueError):
        yahoo.daily_bars("SKYD", date(2022, 2, 18), date(2022, 2, 16))

    assert fetch.calls == []


# =============================================================================================
# reviewer 2026-10-16: a row with a close but no open/high/low/volume is not data
# =============================================================================================


@pytest.mark.parametrize("missing", ["open", "high", "low", "volume"])
def test_row_with_a_close_but_another_field_null_raises(missing: str) -> None:
    row: dict[str, float | int | None] = {
        "open": 29.51,
        "high": 30.45,
        "low": 29.1,
        "close": 29.35,
        "volume": 12100500,
    }
    row[missing] = None
    body = chart([(T_0217, row["open"], row["high"], row["low"], row["close"], row["volume"])])  # type: ignore[arg-type]
    yahoo, _ = client(reply(200, body))

    with pytest.raises(YahooError, match=missing):
        yahoo.daily_bars("SKYD", date(2022, 2, 17), date(2022, 2, 17))
