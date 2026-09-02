"""Current local date and time. No network, so it is always cheap to call."""

import json
import sys
from datetime import date, datetime
from zoneinfo import ZoneInfo

TZ = ZoneInfo("America/Los_Angeles")


def main() -> int:
    try:
        args = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        args = {}

    now = datetime.now(TZ)
    result = {
        "ok": True,
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%-I:%M %p"),
        "weekday": now.strftime("%A"),
        "spoken": now.strftime("%A, %B %-d, %Y, %-I:%M %p"),
        "timezone": "America/Los_Angeles",
    }

    target = args.get("until")
    if target:
        try:
            days = (date.fromisoformat(str(target)) - now.date()).days
            result["until"] = target
            result["days_away"] = days
        except ValueError:
            result["until_error"] = f"not an ISO date: {target}"

    json.dump(result, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
