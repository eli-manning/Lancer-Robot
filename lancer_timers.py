"""
Timers, reminders, and anything else Lancer needs to say unprompted.

Everything Lancer said until now was an answer to something. A timer is the
first thing he has to bring up himself, and that needs a channel that does not
exist in a request/response pipeline: nobody is asking, so there is no reply to
put it in.

The store is a file rather than memory in the server process because skills run
as subprocesses -- the skill that sets a timer and the watcher that fires it are
different programs. A file is also what makes a timer survive a server restart,
which matters when the restart takes twenty seconds and the timer was for three
minutes.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any

BASE = Path(__file__).parent
STORE = BASE / "lancer_timers.json"

# Spoken durations, in the shapes people actually say them.
_UNITS = {
    "second": 1, "seconds": 1, "sec": 1, "secs": 1, "s": 1,
    "minute": 60, "minutes": 60, "min": 60, "mins": 60, "m": 60,
    "hour": 3600, "hours": 3600, "hr": 3600, "hrs": 3600, "h": 3600,
}
_WORD_NUMBERS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40,
    "forty-five": 45, "fifty": 50, "sixty": 60, "ninety": 90,
    "half": 0.5, "quarter": 0.25,
}


def parse_duration(text: str) -> float | None:
    """
    Seconds from "5 minutes", "an hour and a half", "90 seconds", "2m30s".

    Returns None when there is no duration in the text at all, which the caller
    reports rather than guessing -- a timer set for the wrong length is worse
    than one that was refused.
    """
    lowered = str(text or "").strip().lower()
    if not lowered:
        return None

    total = 0.0
    found = False
    # Fractions of a unit, spelled out before it. These are rewritten rather
    # than pattern-matched: "half an hour" otherwise matches "an hour" and the
    # "half" is silently dropped, which sets a timer for twice as long as asked.
    for phrase, plain in (
        ("half an hour", "30 minutes"), ("half a hour", "30 minutes"),
        ("quarter of an hour", "15 minutes"), ("quarter an hour", "15 minutes"),
        ("three quarters of an hour", "45 minutes"),
        ("half a minute", "30 seconds"),
    ):
        lowered = lowered.replace(phrase, plain)
    # "an hour and a half" -- the half belongs to the unit before it.
    lowered = lowered.replace("and a half", "+half").replace("and a quarter",
                                                             "+quarter")
    for amount, unit in re.findall(
            r"([0-9]*\.?[0-9]+|\+?[a-z-]+)\s*"
            r"(seconds?|secs?|minutes?|mins?|hours?|hrs?|[smh])\b", lowered):
        try:
            count = float(amount)
        except ValueError:
            key = amount.lstrip("+")
            if key not in _WORD_NUMBERS:
                continue
            count = _WORD_NUMBERS[key]
        seconds = _UNITS.get(unit)
        if seconds is None:
            continue
        total += count * seconds
        found = True

    # "+half" after a unit: "an hour +half" -> another half hour.
    for word, unit in re.findall(r"\+(half|quarter)\s*(?:an?\s*)?"
                                 r"(hour|minute|hr|min)?", lowered):
        base = _UNITS.get(unit or "", 0) or (3600 if "hour" in lowered else 60)
        total += _WORD_NUMBERS[word] * base
        found = True

    return total if found and total > 0 else None


def _load() -> list[dict[str, Any]]:
    if not STORE.exists():
        return []
    try:
        data = json.loads(STORE.read_text())
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def _save(timers: list[dict]) -> None:
    STORE.write_text(json.dumps(timers, indent=2))


def add(seconds: float, label: str = "") -> dict:
    timers = _load()
    timer = {
        "id": uuid.uuid4().hex[:8],
        "label": (label or "").strip(),
        "seconds": float(seconds),
        "due": time.time() + float(seconds),
        "set_at": time.time(),
    }
    timers.append(timer)
    _save(timers)
    return timer


def active() -> list[dict]:
    now = time.time()
    return sorted((t for t in _load() if t["due"] > now), key=lambda t: t["due"])


def due() -> list[dict]:
    """Timers that have finished. Removes them: a timer fires once."""
    timers = _load()
    now = time.time()
    finished = [t for t in timers if t["due"] <= now]
    if finished:
        _save([t for t in timers if t["due"] > now])
    return finished


def cancel(which: str = "") -> list[dict]:
    """
    Cancel by label, by id, or all of them.

    Matching on label is deliberately loose: someone who set "the pasta timer"
    will cancel it by saying "pasta".
    """
    timers = _load()
    which = (which or "").strip().lower()
    if which in ("", "all", "every", "everything"):
        _save([])
        return timers
    removed = [t for t in timers
               if which == t["id"] or (t["label"] and which in t["label"].lower())]
    if removed:
        kept = [t for t in timers if t not in removed]
        _save(kept)
    return removed


def describe(seconds: float) -> str:
    """A duration said the way a person would say it."""
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds} second{'s' if seconds != 1 else ''}"
    minutes, rest = divmod(seconds, 60)
    if minutes < 60:
        if rest and minutes < 5:
            return f"{minutes} minute{'s' if minutes != 1 else ''} {rest} seconds"
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    hours, minutes = divmod(minutes, 60)
    if minutes:
        return f"{hours} hour{'s' if hours != 1 else ''} {minutes} minutes"
    return f"{hours} hour{'s' if hours != 1 else ''}"
