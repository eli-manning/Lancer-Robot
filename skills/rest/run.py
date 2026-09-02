"""Put the robot to sleep or wake it up."""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _robot import robot_post                       # noqa: E402

# Long enough for the app's tracking loops to see the move and stand down, short
# enough that "go to sleep" still feels like an answer rather than a delay.
_HANDOVER_PAUSE = 0.7


def main() -> int:
    try:
        args = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        args = {}

    action = str(args.get("action") or "sleep").strip().lower()
    if action in ("sleep", "rest", "nap", "go_to_sleep"):
        # Take the joints away from everything else before folding up.
        #
        # Three things drive this robot: the daemon's face tracker, whatever
        # move is already playing, and the app's own tracking loops. Starting
        # goto_sleep while any of them is still running means two writers
        # fighting over the same servos on the way down, which is the shaking.
        # The app notices within a poll interval on its own; the other two have
        # to be told, and the pause gives all three time to let go.
        robot_post("/api/media/tracking/disable", timeout=10)
        robot_post("/api/move/stop", timeout=10)
        time.sleep(_HANDOVER_PAUSE)
        ok, body = robot_post("/api/move/play/goto_sleep", timeout=20)
        state = "asleep"
    elif action in ("wake", "wake_up", "get_up"):
        # Motors first. goto_sleep powers them down once the robot is folded,
        # and with them off wake_up is accepted and then does nothing at all --
        # so "go to sleep" left a robot that could not be woken by asking, and
        # looked broken rather than asleep.
        robot_post("/api/motors/set_mode/enabled", timeout=15)
        ok, body = robot_post("/api/move/play/wake_up", timeout=20)
        # Tracking was switched off on the way down; without this the robot
        # wakes up and then stares straight ahead, which looks like it did not
        # really wake.
        time.sleep(_HANDOVER_PAUSE)
        robot_post("/api/media/tracking/enable", {"weight": 1.0}, timeout=10)
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
