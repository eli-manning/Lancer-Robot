"""Put the robot to sleep or wake it up."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _robot import robot_post                       # noqa: E402


def main() -> int:
    try:
        args = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        args = {}

    action = str(args.get("action") or "sleep").strip().lower()
    if action in ("sleep", "rest", "nap", "go_to_sleep"):
        ok, body = robot_post("/api/move/play/goto_sleep", timeout=20)
        state = "asleep"
    elif action in ("wake", "wake_up", "get_up"):
        ok, body = robot_post("/api/move/play/wake_up", timeout=20)
        state = "awake"
    else:
        json.dump({"ok": False, "error": f"unknown action {action!r}; "
                   "use sleep or wake"}, sys.stdout)
        return 0

    json.dump({"ok": ok, "state": state if ok else None,
               "detail": body if not ok else f"going {state}"}, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
