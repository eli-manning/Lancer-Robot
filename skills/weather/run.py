"""
Campus weather from Open-Meteo.

Open-Meteo is keyless and free for non-commercial use, which is the whole
reason it is here: a skill that needs an API key someone has to provision is a
skill that is broken on demo day.
"""

import json
import sys

import requests

# California Baptist University, Riverside CA.
LATITUDE = 33.9281
LONGITUDE = -117.4256
ENDPOINT = "https://api.open-meteo.com/v1/forecast"

# WMO weather interpretation codes, in words a robot can say out loud.
CONDITIONS = {
    0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    56: "freezing drizzle", 57: "heavy freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "freezing rain", 67: "heavy freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light showers", 81: "showers", 82: "violent showers",
    85: "snow showers", 86: "heavy snow showers",
    95: "a thunderstorm", 96: "a thunderstorm with hail",
    99: "a severe thunderstorm with hail",
}


def main() -> int:
    try:
        args = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        args = {}
    when = str(args.get("when") or "now")

    try:
        response = requests.get(
            ENDPOINT,
            params={
                "latitude": LATITUDE,
                "longitude": LONGITUDE,
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,"
                           "wind_speed_10m,weather_code",
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,"
                         "precipitation_probability_max",
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "timezone": "America/Los_Angeles",
                "forecast_days": 3,
            },
            timeout=8,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        json.dump({"ok": False, "error": f"weather service unreachable: {e}"},
                  sys.stdout)
        return 0

    current = data.get("current", {})
    daily = data.get("daily", {})
    code = int(current.get("weather_code", -1))

    result = {
        "ok": True,
        "location": "Riverside, California",
        "now": {
            "conditions": CONDITIONS.get(code, "unclear"),
            "temperature_f": round(float(current.get("temperature_2m", 0))),
            "feels_like_f": round(float(current.get("apparent_temperature", 0))),
            "humidity_pct": current.get("relative_humidity_2m"),
            "wind_mph": round(float(current.get("wind_speed_10m", 0))),
        },
    }

    days = daily.get("time") or []
    forecast = []
    for i, day in enumerate(days):
        forecast.append({
            "date": day,
            "conditions": CONDITIONS.get(
                int((daily.get("weather_code") or [-1] * len(days))[i]), "unclear"),
            "high_f": round(float((daily.get("temperature_2m_max") or [0])[i])),
            "low_f": round(float((daily.get("temperature_2m_min") or [0])[i])),
            "rain_chance_pct": (daily.get("precipitation_probability_max") or [0])[i],
        })

    if when == "tomorrow":
        result["forecast"] = forecast[1:2]
    elif when in ("today",):
        result["forecast"] = forecast[:1]
    else:
        result["forecast"] = forecast

    json.dump(result, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
