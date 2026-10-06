"""Guards against a client that silently loses pages, hammers a rate-limited API or leaks keys."""

from datetime import UTC, datetime, timedelta

import pytest

from asof.ingest.alpaca import AlpacaClient, AlpacaError, Asset, Bar
from tests.ingest.fakes import FakeTime, ScriptedFetch, bar_json, reply

KEY = "PKTESTKEY"
SECRET = "testsecret"
START = datetime(2026, 10, 2, 4, 0, tzinfo=UTC)
END = datetime(2026, 10, 3, 4, 0, tzinfo=UTC)

ASSET_ROWS = [
    {
        "id": "b0b6dd9d-8b9b-48a9-ba46-b9d54906e415",
        "class": "us_equity",
        "asset_class": "us_equity",
        "exchange": "NASDAQ",
        "symbol": "AAPL",
        "name": "Apple Inc. Common Stock",
        "status": "active",
        "tradable": True,
        "marginable": True,
        "shortable": True,
        "easy_to_borrow": True,
        "fractionable": True,
    },
    {
        "asset_class": "us_equity",
        "exchange": "NYSE",
        "symbol": "SIVB",
        "name": "SVB Financial Group",
        "status": "inactive",
        "tradable": False,
    },
]


def client(fetch: ScriptedFetch, time: FakeTime | None = None, **options: int) -> AlpacaClient:
    time = time or FakeTime()
    return AlpacaClient(KEY, SECRET, fetch=fetch, sleep=time.sleep, clock=time.monotonic, **options)


def bars_page(bars: dict[str, list[dict]], next_page_token: str | None = None):
    return reply(200, {"bars": bars, "next_page_token": next_page_token})


# --- normal path -------------------------------------------------------------------------------


def test_auth_headers_are_exposed_and_never_sent_as_query_params() -> None:
    fetch = ScriptedFetch([reply(200, ASSET_ROWS)])
    c = client(fetch)

    c.assets("active")

    assert c.headers == {"APCA-API-KEY-ID": KEY, "APCA-API-SECRET-KEY": SECRET}
    for params in fetch.params:
        assert KEY not in params.values()
        assert SECRET not in params.values()
        assert KEY not in params
        assert SECRET not in params


def test_assets_hits_the_trading_api_and_parses_known_fields() -> None:
    fetch = ScriptedFetch([reply(200, ASSET_ROWS)])

    assets = client(fetch).assets("active")

    assert fetch.calls == [
        (f"{AlpacaClient.TRADING_URL}/v2/assets", {"status": "active", "asset_class": "us_equity"})
    ]
    assert assets == [
        Asset(
            symbol="AAPL",
            name="Apple Inc. Common Stock",
            exchange="NASDAQ",
            status="active",
            tradable=True,
            asset_class="us_equity",
        ),
        Asset(
            symbol="SIVB",
            name="SVB Financial Group",
            exchange="NYSE",
            status="inactive",
            tradable=False,
            asset_class="us_equity",
        ),
    ]


def test_assets_passes_status_and_asset_class_through() -> None:
    fetch = ScriptedFetch([reply(200, [])])

    assert client(fetch).assets("inactive", asset_class="crypto") == []
    assert fetch.params == [{"status": "inactive", "asset_class": "crypto"}]


def test_iter_bars_sends_the_documented_query() -> None:
    fetch = ScriptedFetch([bars_page({})])

    list(client(fetch).iter_bars(["AAPL", "MSFT"], "1Day", START, END))

    assert fetch.urls == [f"{AlpacaClient.DATA_URL}/v2/stocks/bars"]
    assert fetch.params == [
        {
            "symbols": "AAPL,MSFT",
            "timeframe": "1Day",
            "start": "2026-10-02T04:00:00Z",
            "end": "2026-10-03T04:00:00Z",
            "limit": "10000",
            "adjustment": "raw",
            "feed": "sip",
        }
    ]


def test_iter_bars_follows_next_page_token_until_null() -> None:
    fetch = ScriptedFetch(
        [
            bars_page(
                {
                    "AAPL": [bar_json("2026-10-02T04:00:00Z")],
                    "MSFT": [bar_json("2026-10-02T04:00:00Z")],
                },
                next_page_token="abc",
            ),
            bars_page({"MSFT": [bar_json("2026-10-03T04:00:00Z", c=9.0)]}, next_page_token=None),
        ]
    )

    bars = list(client(fetch).iter_bars(["AAPL", "MSFT"], "1Day", START, END + timedelta(days=1)))

    assert sorted((b.symbol, b.t) for b in bars) == [
        ("AAPL", datetime(2026, 10, 2, 4, tzinfo=UTC)),
        ("MSFT", datetime(2026, 10, 2, 4, tzinfo=UTC)),
        ("MSFT", datetime(2026, 10, 3, 4, tzinfo=UTC)),
    ]
    assert len(fetch.calls) == 2
    assert "page_token" not in fetch.params[0]
    assert fetch.params[1]["page_token"] == "abc"
    # Everything else is repeated on the continuation request.
    assert {k: v for k, v in fetch.params[1].items() if k != "page_token"} == fetch.params[0]


def test_bars_are_parsed_into_typed_utc_models() -> None:
    raw = bar_json(
        "2026-10-02T13:30:00Z", o=100.5, h=101.0, l=99.5, c=100.0, v=12345, n=678, vw=100.25
    )
    raw["unknown_future_field"] = "ignored"
    fetch = ScriptedFetch([bars_page({"AAPL": [raw]})])

    [bar] = list(client(fetch).iter_bars(["AAPL"], "1Min", START, END))

    assert bar == Bar(
        symbol="AAPL",
        t=datetime(2026, 10, 2, 13, 30, tzinfo=UTC),
        open=100.5,
        high=101.0,
        low=99.5,
        close=100.0,
        volume=12345,
        trade_count=678,
        vwap=100.25,
    )
    assert bar.t.tzinfo is not None
    assert bar.t.utcoffset() == timedelta(0)
    assert type(bar.volume) is int
    assert type(bar.trade_count) is int


# --- boundaries --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload", [{"bars": {}, "next_page_token": None}, {"next_page_token": None}]
)
def test_page_without_bars_yields_nothing(payload: dict) -> None:
    fetch = ScriptedFetch([reply(200, payload)])

    assert list(client(fetch).iter_bars(["ZZZZ"], "1Day", START, END)) == []
    assert len(fetch.calls) == 1


def test_empty_page_in_the_middle_does_not_stop_paging() -> None:
    fetch = ScriptedFetch(
        [
            bars_page({}, next_page_token="more"),
            bars_page({"AAPL": [bar_json("2026-10-02T04:00:00Z")]}),
        ]
    )

    bars = list(client(fetch).iter_bars(["AAPL"], "1Day", START, END))

    assert [b.symbol for b in bars] == ["AAPL"]
    assert fetch.params[1]["page_token"] == "more"


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (datetime(2026, 10, 2, 4, 0), END),
        (START, datetime(2026, 10, 3, 4, 0)),
        (datetime(2026, 10, 2, 4, 0), datetime(2026, 10, 3, 4, 0)),
    ],
    ids=["naive_start", "naive_end", "both_naive"],
)
def test_naive_start_or_end_is_rejected_before_any_request(start: datetime, end: datetime) -> None:
    fetch = ScriptedFetch()

    with pytest.raises(ValueError):
        list(client(fetch).iter_bars(["AAPL"], "1Day", start, end))

    assert fetch.calls == []


def test_non_utc_aware_bounds_are_sent_as_utc_z_strings() -> None:
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    fetch = ScriptedFetch([bars_page({})])

    list(
        client(fetch).iter_bars(
            ["AAPL"],
            "1Min",
            datetime(2026, 10, 2, 9, 30, tzinfo=et),
            datetime(2026, 10, 2, 16, 0, tzinfo=et),
        )
    )

    assert fetch.params[0]["start"] == "2026-10-02T13:30:00Z"
    assert fetch.params[0]["end"] == "2026-10-02T20:00:00Z"


# --- retry -------------------------------------------------------------------------------------


def test_429_without_retry_after_backs_off_exponentially_then_succeeds() -> None:
    time = FakeTime()
    fetch = ScriptedFetch([reply(429, {"message": "slow down"}), reply(200, ASSET_ROWS[:1])])

    assets = client(fetch, time).assets("active")

    assert [a.symbol for a in assets] == ["AAPL"]
    assert time.sleeps == [1.0]
    assert len(fetch.calls) == 2


def test_429_honours_retry_after_header() -> None:
    time = FakeTime()
    fetch = ScriptedFetch([reply(429, headers={"Retry-After": "3"}), reply(200, [])])

    client(fetch, time).assets("active")

    assert time.sleeps == [3.0]


@pytest.mark.parametrize("status", [500, 502, 503])
def test_server_errors_are_retried_with_growing_backoff(status: int) -> None:
    time = FakeTime()
    fetch = ScriptedFetch([reply(status), reply(status), reply(200, [])])

    client(fetch, time).assets("active")

    assert time.sleeps == [1.0, 2.0]
    assert len(fetch.calls) == 3


def test_retries_stop_after_max_retries_and_raise() -> None:
    time = FakeTime()
    fetch = ScriptedFetch([reply(503)] * 10)

    with pytest.raises(AlpacaError):
        client(fetch, time, max_retries=2).assets("active")

    # One initial attempt plus max_retries retries, then give up.
    assert len(fetch.calls) == 3
    assert time.sleeps == [1.0, 2.0]


def test_failures_up_to_max_retries_still_succeed() -> None:
    time = FakeTime()
    fetch = ScriptedFetch([reply(503), reply(503), reply(200, [])])

    assert client(fetch, time, max_retries=2).assets("active") == []


def test_retry_resumes_the_same_page_during_paging() -> None:
    time = FakeTime()
    fetch = ScriptedFetch(
        [
            bars_page({"AAPL": [bar_json("2026-10-02T04:00:00Z")]}, next_page_token="p2"),
            reply(500),
            bars_page({"AAPL": [bar_json("2026-10-03T04:00:00Z")]}),
        ]
    )

    bars = list(client(fetch, time).iter_bars(["AAPL"], "1Day", START, END + timedelta(days=1)))

    assert len(bars) == 2
    assert fetch.params[1]["page_token"] == "p2"
    assert fetch.params[2]["page_token"] == "p2"
    assert time.sleeps == [1.0]


@pytest.mark.parametrize("status", [401, 403, 404, 422])
def test_client_errors_fail_immediately_without_retry(status: int) -> None:
    time = FakeTime()
    fetch = ScriptedFetch([reply(status, {"message": "nope"}), reply(200, [])])

    with pytest.raises(AlpacaError):
        client(fetch, time).assets("active")

    assert len(fetch.calls) == 1
    assert time.sleeps == []


# --- rate limit --------------------------------------------------------------------------------


def test_fourth_request_within_a_minute_waits_for_the_window() -> None:
    time = FakeTime()
    fetch = ScriptedFetch([reply(200, [])] * 4)
    c = client(fetch, time, max_requests_per_minute=3)

    for _ in range(3):
        c.assets("active")
    assert time.sleeps == []

    c.assets("active")

    assert len(time.sleeps) == 1
    assert time.sleeps[0] == pytest.approx(60.0, abs=1.0)
    assert len(fetch.calls) == 4


def test_requests_spread_over_time_never_wait() -> None:
    time = FakeTime()
    fetch = ScriptedFetch([reply(200, [])] * 5)
    c = client(fetch, time, max_requests_per_minute=3)

    for _ in range(5):
        c.assets("active")
        time.advance(61.0)

    assert time.sleeps == []
    assert len(fetch.calls) == 5


def test_each_page_counts_as_a_request_for_the_rate_limit() -> None:
    time = FakeTime()
    fetch = ScriptedFetch(
        [
            bars_page({}, next_page_token="a"),
            bars_page({}, next_page_token="b"),
            bars_page({}, next_page_token="c"),
            bars_page({}),
        ]
    )
    c = client(fetch, time, max_requests_per_minute=3)

    list(c.iter_bars(["AAPL"], "1Day", START, END))

    assert len(fetch.calls) == 4
    assert len(time.sleeps) == 1
    assert time.sleeps[0] == pytest.approx(60.0, abs=1.0)


# --- leakage -----------------------------------------------------------------------------------


def test_client_requests_raw_prices_so_future_splits_cannot_rewrite_history() -> None:
    # Adjusted prices are rewritten by splits that happen after the bar: a form of look-ahead.
    fetch = ScriptedFetch([bars_page({})])

    list(client(fetch).iter_bars(["NVDA"], "1Day", START, END))

    assert fetch.params[0]["adjustment"] == "raw"


def test_bar_timestamps_keep_their_instant_when_the_feed_uses_an_offset() -> None:
    # Alpaca may serialise with "+00:00" or "Z"; both must land on the same UTC instant.
    fetch = ScriptedFetch(
        [
            bars_page(
                {
                    "AAPL": [
                        bar_json("2026-10-02T13:30:00Z"),
                        bar_json("2026-10-02T13:31:00+00:00"),
                        bar_json("2026-10-02T09:32:00-04:00"),
                    ]
                }
            )
        ]
    )

    bars = list(client(fetch).iter_bars(["AAPL"], "1Min", START, END))

    assert [b.t for b in bars] == [
        datetime(2026, 10, 2, 13, 30, tzinfo=UTC),
        datetime(2026, 10, 2, 13, 31, tzinfo=UTC),
        datetime(2026, 10, 2, 13, 32, tzinfo=UTC),
    ]
    assert all(b.t.utcoffset() == timedelta(0) for b in bars)


# --- transport failures ------------------------------------------------------------------------


class FlakyFetch(ScriptedFetch):
    """Raises the scripted exceptions before returning scripted results."""

    def __init__(self, errors: list[Exception], results: list) -> None:
        super().__init__(results)
        self.errors = list(errors)

    def __call__(self, url: str, params) -> object:  # type: ignore[override]
        if self.errors:
            self.calls.append((url, dict(params)))
            raise self.errors.pop(0)
        return super().__call__(url, params)


def test_connection_errors_are_retried_with_backoff() -> None:
    import httpx

    time = FakeTime()
    fetch = FlakyFetch([httpx.ConnectError("reset"), httpx.ReadTimeout("slow")], [reply(200, [])])

    assert client(fetch, time).assets("active") == []
    assert time.sleeps == [1.0, 2.0]
    assert len(fetch.calls) == 3


def test_connection_errors_exhaust_into_alpaca_error() -> None:
    import httpx

    time = FakeTime()
    fetch = FlakyFetch([httpx.ConnectError("reset")] * 5, [])

    with pytest.raises(AlpacaError):
        client(fetch, time, max_retries=2).assets("active")

    assert len(fetch.calls) == 3


def test_non_numeric_retry_after_falls_back_to_exponential_backoff() -> None:
    time = FakeTime()
    fetch = ScriptedFetch(
        [reply(429, headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}), reply(200, [])]
    )

    client(fetch, time).assets("active")

    assert time.sleeps == [1.0]


def test_retry_after_is_capped() -> None:
    time = FakeTime()
    fetch = ScriptedFetch([reply(429, headers={"Retry-After": "86400"}), reply(200, [])])

    client(fetch, time).assets("active")

    assert time.sleeps == [120.0]


def test_bar_without_timezone_is_rejected() -> None:
    from pydantic import ValidationError

    fetch = ScriptedFetch([bars_page({"AAPL": [bar_json("2026-10-02T04:00:00")]})])

    with pytest.raises(ValidationError):
        list(client(fetch).iter_bars(["AAPL"], "1Day", START, END))
