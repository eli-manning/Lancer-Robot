---
name: current_datetime
description: Use for anything that depends on today's date or the current time - what day is it, what time is it, is it still morning, how many days until a date. Returns local date and time for Riverside, California. Triggers on - what day, what time, today, tomorrow, date, how long until, days away.
---

# Current date and time

Reads the clock. No network, so it is cheap enough to call whenever the answer
touches time at all.

## Running it

```
echo '{}' | python3 run.py
echo '{"until":"2026-12-12"}' | python3 run.py
```

Passing `until` (ISO `YYYY-MM-DD`) adds `days_away`, counted from today.

## Why this exists

A language model has no clock. Left to itself it will answer "what day is it"
with a date from its training data, stated with total confidence and off by
months. Call this instead of guessing, every time -- including when you only
need the date in order to work something else out.

## Reading the result

`spoken` is preformatted for saying out loud ("Wednesday, September 2, 2026,
12:40 AM"). Prefer it over assembling the parts yourself.
