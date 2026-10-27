"""get_weather(city) via Open-Meteo: free, no API key.

Two calls: the geocoding API turns the city name into coordinates, then
the forecast API returns the current conditions there.
"""

from __future__ import annotations

import re

import requests

from app.agent_tools.base import HTTP_TIMEOUT_SECONDS, Tool, ToolError, ToolInputError, require_text

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
MAX_CITY_LENGTH = 100
# Letters in any script, plus the spaces and punctuation real place names use
# ("St. John's", "Rio de Janeiro", "Clermont-Ferrand").
_CITY = re.compile(r"[^\W\d_]+(?:[ .,'’-]+[^\W\d_]+)*\.?")

# WMO weather interpretation codes, as returned in `weather_code`.
WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    56: "light freezing drizzle", 57: "dense freezing drizzle",
    61: "slight rain", 63: "moderate rain", 65: "heavy rain",
    66: "light freezing rain", 67: "heavy freezing rain",
    71: "slight snow", 73: "moderate snow", 75: "heavy snow", 77: "snow grains",
    80: "slight rain showers", 81: "moderate rain showers", 82: "violent rain showers",
    85: "slight snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with slight hail", 99: "thunderstorm with heavy hail",
}


def _get_json(url: str, params: dict) -> dict:
    try:
        response = requests.get(url, params=params, timeout=HTTP_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise ToolError(f"Weather service error: {exc}") from exc


def get_weather(city) -> str:
    """Current temperature, conditions, humidity and wind for `city`."""
    city = require_text("city", city, MAX_CITY_LENGTH, _CITY)

    # "Paris, France" → search "Paris"; the geocoder matches the name only.
    places = _get_json(GEOCODING_URL, {"name": city.split(",")[0], "count": 1}).get("results") or []
    if not places:
        raise ToolInputError(f"City not found: {city!r}")
    place = places[0]

    current = _get_json(FORECAST_URL, {
        "latitude": place["latitude"], "longitude": place["longitude"], "timezone": "auto",
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
    }).get("current")
    if not current:
        raise ToolError(f"No current weather for {place['name']}")

    location = ", ".join(part for part in (place["name"], place.get("admin1"), place.get("country")) if part)
    conditions = WEATHER_CODES.get(current.get("weather_code"), "unknown conditions")
    return (f"{location}: {current['temperature_2m']} °C, {conditions}, "
            f"humidity {current['relative_humidity_2m']}%, wind {current['wind_speed_10m']} km/h "
            f"(local time {current['time']})")


WEATHER_TOOL = Tool(
    name="get_weather",
    description="Get the current weather (temperature, conditions, humidity, wind) for a city.",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name, e.g. Mumbai or Paris, France."},
        },
        "required": ["city"],
    },
    function=get_weather,
)
