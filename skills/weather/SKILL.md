---
name: weather
description: Use when asked about the weather, temperature, rain, or whether to bring a jacket. Returns live conditions and a three-day forecast for campus in Riverside, California. Triggers on - weather, temperature, forecast, is it raining, will it rain, hot, cold, jacket, umbrella.
---

# Campus weather

Live weather for California Baptist University (Riverside, CA) from Open-Meteo.
No API key, no account, no rate limit worth worrying about.

## Running it

```
echo '{"when":"now"}' | python3 run.py
```

`when` is one of `now`, `today`, `tomorrow`, `forecast`. It defaults to `now`,
which still includes the three-day forecast in the reply.

## Reading the result

- `now.conditions` is already plain English ("partly cloudy"), so it can be
  said out loud as-is.
- Temperatures are Fahrenheit, wind is mph, and both are rounded to whole
  numbers -- do not read out decimals to someone standing in a hallway.
- `forecast[].rain_chance_pct` is the day's maximum probability, not an average.

## When it fails

`ok: false` means the service was unreachable. Say you could not get the
weather. Do not answer from memory: whatever you think the weather is, you are
guessing, and the person asking is standing next to a window.
