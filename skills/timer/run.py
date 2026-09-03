"""Countdown timers that Lancer announces himself when they finish."""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

import lancer_timers as timers                      # noqa: E402

# Below this a timer is over before he finishes saying he set it.
MIN_SECONDS = 3


def main() -> int:
    try:
        args = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        args = {}

    action = str(args.get("action") or "set").strip().lower()
    label = str(args.get("label") or "").strip()

    if action == "list":
        running = timers.active()
        import time as _time

        return _out({
            "ok": True,
            "timers": [
                {"label": t["label"], "remaining": timers.describe(
                    max(0, t["due"] - _time.time()))}
                for t in running
            ],
            "detail": (f"{len(running)} timer(s) running" if running
                       else "no timers running"),
        })

    if action == "cancel":
        removed = timers.cancel(label or str(args.get("duration") or ""))
        return _out({
            "ok": True, "cancelled": len(removed),
            "detail": (f"cancelled {len(removed)} timer(s)" if removed
                       else "there was no timer matching that"),
        })

    spoken = str(args.get("duration") or "")
    seconds = timers.parse_duration(spoken)
    if seconds is None:
        return _out({"ok": False,
                     "error": "no duration in that -- ask how long they want"})
    if seconds < MIN_SECONDS:
        return _out({"ok": False,
                     "error": f"{timers.describe(seconds)} is too short to be useful"})

    timer = timers.add(seconds, label)
    return _out({"ok": True, "label": timer["label"],
                 "duration": timers.describe(seconds),
                 "detail": f"timer set for {timers.describe(seconds)}"
                           + (f" ({timer['label']})" if timer["label"] else "")
                           + ". Confirm it briefly; you will announce it "
                             "yourself when it finishes."})


def _out(payload: dict) -> int:
    json.dump(payload, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
