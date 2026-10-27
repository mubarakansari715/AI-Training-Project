import json
import math

import pytest
import requests
from ddgs.exceptions import DDGSException

from app.agent_tools import TOOLS, run_tool, tool_schemas
from app.agent_tools import currency, weather, web_search
from app.agent_tools.base import ToolError, ToolInputError
from app.agent_tools.currency import convert_currency
from app.agent_tools.multiply import multiply
from app.agent_tools.weather import get_weather


class FakeResponse:
    def __init__(self, data=None, status_code=200):
        self.data = data or {}
        self.status_code = status_code
        self.ok = status_code < 400

    def json(self):
        return self.data

    def raise_for_status(self):
        if not self.ok:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def fake_get(*responses):
    """A requests.get replacement that replays `responses` and records each call."""
    calls = []
    queue = list(responses)

    def get(url, params=None, timeout=None):
        calls.append({"url": url, "params": params, "timeout": timeout})
        reply = queue.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    get.calls = calls
    return get


# --- multiply ---

def test_multiply_returns_the_exact_product():
    assert multiply(250, 4) == "250 × 4 = 1,000"
    assert multiply(0.1, 0.2) == "0.1 × 0.2 = 0.02"
    assert multiply("1,200", "-3") == "1200 × -3 = -3,600"


@pytest.mark.parametrize("a, b", [(True, 2), ("abc", 2), (None, 2), (math.inf, 2), (math.nan, 2), (2e12, 1)])
def test_multiply_rejects_bad_numbers(a, b):
    with pytest.raises(ToolInputError):
        multiply(a, b)


# --- convert_currency ---

def test_convert_currency_calls_frankfurter(monkeypatch):
    get = fake_get(FakeResponse({"amount": 250.0, "base": "USD", "date": "2026-10-26", "rates": {"INR": 23796}}))
    monkeypatch.setattr(currency.requests, "get", get)

    result = convert_currency(250, "usd", " inr ")

    assert result.startswith("250.00 USD = 23,796.00 INR")
    assert "2026-10-26" in result
    assert get.calls[0]["params"] == {"base": "USD", "symbols": "INR", "amount": 250}
    assert get.calls[0]["timeout"]


def test_convert_currency_same_currency_skips_the_api(monkeypatch):
    monkeypatch.setattr(currency.requests, "get", fake_get())
    assert convert_currency(10, "EUR", "eur").startswith("10.00 EUR = 10.00 EUR")


@pytest.mark.parametrize("amount, source, target", [
    (0, "USD", "INR"), (-5, "USD", "INR"), (2e12, "USD", "INR"),
    (10, "US", "INR"), (10, "dollars", "INR"), (10, "USD", "IN1"), (10, 840, "INR"),
])
def test_convert_currency_rejects_bad_input(monkeypatch, amount, source, target):
    monkeypatch.setattr(currency.requests, "get", fake_get())
    with pytest.raises(ToolInputError):
        convert_currency(amount, source, target)


def test_convert_currency_unknown_code_is_an_input_error(monkeypatch):
    monkeypatch.setattr(currency.requests, "get", fake_get(FakeResponse(status_code=404)))
    with pytest.raises(ToolInputError, match="Unsupported currency"):
        convert_currency(10, "USD", "XYZ")


@pytest.mark.parametrize("reply", [requests.ConnectionError("down"), FakeResponse(status_code=500),
                                   FakeResponse({"rates": {}})])
def test_convert_currency_service_failures(monkeypatch, reply):
    monkeypatch.setattr(currency.requests, "get", fake_get(reply))
    with pytest.raises(ToolError):
        convert_currency(10, "USD", "INR")


# --- get_weather ---

MUMBAI = {"results": [{"name": "Mumbai", "admin1": "Maharashtra", "country": "India",
                       "latitude": 19.07, "longitude": 72.88}]}
CURRENT = {"current": {"time": "2026-10-27T13:45", "temperature_2m": 33.1, "relative_humidity_2m": 55,
                       "wind_speed_10m": 12.4, "weather_code": 2}}


def test_get_weather_geocodes_then_reads_current_conditions(monkeypatch):
    get = fake_get(FakeResponse(MUMBAI), FakeResponse(CURRENT))
    monkeypatch.setattr(weather.requests, "get", get)

    result = get_weather("  Mumbai, India ")

    assert result == ("Mumbai, Maharashtra, India: 33.1 °C, partly cloudy, humidity 55%, "
                      "wind 12.4 km/h (local time 2026-10-27T13:45)")
    assert get.calls[0]["url"] == weather.GEOCODING_URL and get.calls[0]["params"]["name"] == "Mumbai"
    assert get.calls[1]["params"]["latitude"] == 19.07


@pytest.mark.parametrize("city", ["São Paulo", "St. John's", "Clermont-Ferrand", "東京"])
def test_get_weather_accepts_real_place_names(monkeypatch, city):
    monkeypatch.setattr(weather.requests, "get", fake_get(FakeResponse(MUMBAI), FakeResponse(CURRENT)))
    assert get_weather(city).startswith("Mumbai")


@pytest.mark.parametrize("city", ["", "   ", "x" * 101, "Paris; DROP TABLE", "<script>", "12345", 42])
def test_get_weather_rejects_bad_city_names(monkeypatch, city):
    monkeypatch.setattr(weather.requests, "get", fake_get())
    with pytest.raises(ToolInputError):
        get_weather(city)


def test_get_weather_unknown_city(monkeypatch):
    monkeypatch.setattr(weather.requests, "get", fake_get(FakeResponse({})))
    with pytest.raises(ToolInputError, match="City not found"):
        get_weather("Atlantis")


def test_get_weather_service_failure(monkeypatch):
    monkeypatch.setattr(weather.requests, "get", fake_get(requests.Timeout("slow")))
    with pytest.raises(ToolError):
        get_weather("Mumbai")


# --- web_search ---

class FakeDDGS:
    results = []
    error = None
    queries = []

    def __init__(self, timeout=None):
        self.timeout = timeout

    def text(self, query, max_results=None):
        FakeDDGS.queries.append((query, max_results))
        if FakeDDGS.error:
            raise FakeDDGS.error
        return FakeDDGS.results


@pytest.fixture
def ddgs(monkeypatch):
    FakeDDGS.results, FakeDDGS.error, FakeDDGS.queries = [], None, []
    monkeypatch.setattr(web_search, "DDGS", FakeDDGS)
    return FakeDDGS


def test_web_search_formats_numbered_results(ddgs):
    ddgs.results = [{"title": "Streamlit 1.50", "href": "https://example.com/a", "body": "Released today."},
                    {"title": "Docs", "href": "https://example.com/b", "body": "Release notes."}]

    result = web_search.web_search("  latest   streamlit ")

    assert result == ("[1] Streamlit 1.50\nhttps://example.com/a\nReleased today.\n\n"
                      "[2] Docs\nhttps://example.com/b\nRelease notes.")
    assert ddgs.queries == [("latest streamlit", web_search.MAX_RESULTS)]


def test_web_search_no_results(ddgs):
    ddgs.error = DDGSException("No results found.")
    assert web_search.web_search("zzqq").startswith("No web results found")


def test_web_search_failure(ddgs):
    ddgs.error = DDGSException("ratelimit")
    with pytest.raises(ToolError):
        web_search.web_search("news")


@pytest.mark.parametrize("query", ["", "q" * 301, None])
def test_web_search_rejects_bad_queries(ddgs, query):
    with pytest.raises(ToolInputError):
        web_search.web_search(query)


# --- registry and run_tool ---

def test_registry_has_the_four_tools_with_valid_schemas():
    assert list(TOOLS) == ["multiply", "convert_currency", "get_weather", "web_search"]
    for schema in tool_schemas():
        function = schema["function"]
        assert schema["type"] == "function" and function["description"]
        assert set(function["parameters"]["required"]) <= set(function["parameters"]["properties"])
    assert [s["function"]["name"] for s in tool_schemas(["get_weather"])] == ["get_weather"]


def test_run_tool_accepts_json_string_arguments():
    assert run_tool("multiply", json.dumps({"a": 6, "b": 7})) == "6 × 7 = 42"


@pytest.mark.parametrize("name, arguments, expected", [
    ("divide", {}, "unknown tool"),
    ("multiply", "{not json", "not valid JSON"),
    ("multiply", "[1, 2]", "must be a JSON object"),
    ("multiply", {"a": 2}, "missing b"),
    ("multiply", {"a": 2, "b": 3, "c": 4}, "unexpected c"),
    ("multiply", {"a": "two", "b": 3}, "invalid input for multiply"),
])
def test_run_tool_turns_bad_calls_into_error_text(name, arguments, expected):
    result = run_tool(name, arguments)
    assert result.startswith("Error:") and expected in result


def test_run_tool_reports_service_failures_as_text(monkeypatch):
    monkeypatch.setattr(weather.requests, "get", fake_get(requests.ConnectionError("down")))
    assert run_tool("get_weather", {"city": "Mumbai"}).startswith("Error: get_weather failed")
