"""
Describe what the robot's camera can see.

Two modes, and the difference matters. A forward glance answers "what am I
holding". It cannot answer "how many people are in this room" -- the camera
sees maybe a third of the space, and a model handed one frame will confidently
answer anyway. For those, turn the head and look properly first.
"""

import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

import requests                                     # noqa: E402

import lancer_llm as llm                            # noqa: E402
import lancer_runtime as rt                         # noqa: E402
from _robot import robot_get, robot_post            # noqa: E402

SERVER = os.environ.get("LANCER_SERVER_SELF", "http://127.0.0.1:7870")

# The body, not the neck. The neck manages about 35 degrees either side, which
# is a glance; the base turns to roughly 113 degrees each way before it stops,
# and with the camera's own field of view that covers essentially the whole
# room. Measured on the hardware: a full three-radian swing completes in about
# two and a half seconds, so this costs seconds rather than the minute a
# cautious guess would have budgeted.
# (body yaw, head offset) for each look, far left to far right. The offset is
# measured from the body, not the room: the daemon takes head yaw in the world
# frame, so the head's actual target is body yaw + offset (see _aim). Sending
# the offset straight through asked the head to keep facing the room's front,
# and the neck counter-rotated against every body turn -- the body swung right
# while the camera stayed staring ahead, and the survey saw one view five times.
# The base stops at about 113 degrees each way; the neck's 35 on top at the two
# extremes reaches past that, so the pair sweeps roughly 285 degrees of aim.
SWEEP = ((1.85, 0.6), (1.0, 0.0), (0.0, 0.0), (-1.0, 0.0), (-1.85, -0.6))
# One deliberate turn for "what is to your right". A five-stop survey for that
# costs twenty seconds and spends four of its five frames somewhere else.
# Positive yaw is to the robot's left, as in SWEEP.
DIRECTIONS = {
    "left": ((1.1, 0.35),),
    "right": ((-1.1, -0.35),),
    "ahead": (),
}
BODY_LIMIT = 1.9
HEAD_LIMIT = 0.65
MOVE_SECONDS = 1.2
# The app posts a frame about once a second, so a shorter wait than this hands
# the model the previous angle's picture and the survey silently sees nothing new.
SETTLE_SECONDS = 1.1
# Pitch for a deliberate look, a little above level (negative is up): eye
# height for someone standing at the desk and for a screen on the wall. The
# head's pitch at the moment of asking is whatever face tracking last left it
# at, and turning with that kept a robot that had been looking up at someone
# photographing the ceiling from every angle. Roll goes to zero for the same
# reason.
LOOK_PITCH = -0.15

PROMPT = (
    "You are looking through your own camera, as a small desk robot. "
    "Answer in one or two short spoken sentences. Describe only what is "
    "actually visible. If you cannot make something out, say so plainly rather "
    "than guessing. Do not mention the photograph, the image, the camera, "
    "lighting or resolution -- just say what is there. Describe what people are "
    "doing, holding or wearing. Do not guess at anyone's ethnicity, gender, age "
    "or anything else you cannot actually see. Say 'a person' when you do not "
    "know: a robot on a campus desk guessing at those is worse than one that "
    "simply counts."
)
SURVEY_NOTE = (
    " These are several views taken by turning right around, from your far "
    "left to your far right, so between them they cover the whole room. "
    "Neighbouring views overlap, so the same person often appears in two of "
    "them -- count each person once, and answer about the whole space rather "
    "than describing each picture."
)


def _frame() -> bytes | None:
    try:
        response = requests.get(f"{SERVER}/api/frame", timeout=6)
    except requests.RequestException:
        return None
    return response.content if response.ok and response.content else None


def _aim(rest_pose: dict, body_yaw: float, head_offset: float) -> dict:
    """The goto payload that points the camera at body_yaw + head_offset."""
    body = max(-BODY_LIMIT, min(BODY_LIMIT, body_yaw))
    head = {k: rest_pose.get(k, 0.0) for k in ("x", "y", "z")}
    head["roll"] = 0.0
    head["pitch"] = LOOK_PITCH
    # World frame: the head's yaw has to include the body's turn, or the
    # kinematics hold the camera on the room's front.
    head["yaw"] = body + max(-HEAD_LIMIT, min(HEAD_LIMIT, head_offset))
    return {"body_yaw": body, "head_pose": head, "duration": MOVE_SECONDS}


def _pin_robot_address() -> None:
    """
    Resolve the daemon once and put it first for the rest of this run.

    Every robot call re-resolves the address once lancer_net's twenty-second
    cache lapses, which a sweep outlasts -- and with the USB link unplugged
    each re-resolve waited out the dead 10.5.1.1 probe before trying Tailscale.
    """
    try:
        import lancer_net

        base = lancer_net.base()
        if base:
            os.environ["LANCER_ROBOT"] = base
    except Exception:
        pass


def _look_around(stops) -> list[bytes]:
    """
    Turn to each (body yaw, head offset) stop, keeping a frame from each, then
    turn back to where he was facing.

    Tracking is switched off for the duration for the same reason the sleep
    sequence does it: the daemon's face tracker and this are both steering the
    head, and two writers on the same servos is how the head ends up shaking.
    """
    # Two things steer this head: the daemon's face tracker and the app's own
    # aiming loop. Disabling only the first left the second pulling the head
    # back between steps. The hold rides the app's heartbeat, so give it a
    # beat to land.
    try:
        requests.post(f"{SERVER}/api/hold", json={"seconds": 30}, timeout=5)
        time.sleep(1.1)
    except requests.RequestException:
        pass
    ok, started_at = robot_get("/api/state/present_body_yaw", timeout=6)
    try:
        started_at = float(started_at)
    except (TypeError, ValueError):
        started_at = 0.0
    ok, pose = robot_get("/api/state/present_head_pose", timeout=6)
    rest_pose = pose if ok and isinstance(pose, dict) else {}
    robot_post("/api/media/tracking/disable", timeout=6)
    frames: list[bytes] = []
    try:
        for body_yaw, head_offset in stops:
            robot_post("/api/move/goto", _aim(rest_pose, body_yaw, head_offset),
                       timeout=10)
            time.sleep(MOVE_SECONDS + SETTLE_SECONDS)
            jpeg = _frame()
            if jpeg:
                frames.append(jpeg)
    finally:
        # Always turn back to where he was facing and hand tracking over again,
        # even if a move failed -- otherwise he is left staring at a wall. The
        # head goes back to the pose it had, which is already in the world
        # frame, so it lines up with the body it started on.
        home = {k: rest_pose.get(k, 0.0)
                for k in ("x", "y", "z", "roll", "pitch")}
        home["yaw"] = float(rest_pose.get("yaw", started_at) or 0.0)
        home_seconds = MOVE_SECONDS + 0.6
        robot_post("/api/move/goto",
                   {"body_yaw": started_at, "head_pose": home,
                    "duration": home_seconds}, timeout=10)
        # Wait for the way back before handing the joints over. Enabling the
        # daemon's tracker and releasing the app's hold while the return move
        # was still running put three writers on the same servos at once, and
        # the robot shook its way home. Hand back one at a time, and only once
        # nothing of ours is still moving.
        time.sleep(home_seconds + 0.4)
        robot_post("/api/media/tracking/enable", {"weight": 1.0}, timeout=6)
        time.sleep(0.3)
        try:
            requests.post(f"{SERVER}/api/hold", json={"seconds": 0}, timeout=5)
        except requests.RequestException:
            pass
    return frames


def main() -> int:
    try:
        args = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        args = {}

    cfg = rt.load()
    provider = str(cfg.get("llm_provider") or llm.OLLAMA)
    if not llm.sees(provider):
        return _out({"ok": False,
                     "error": f"I cannot see on {provider} -- it is a text-only "
                              "model. Switch to Gemini or Anthropic."})

    survey = bool(args.get("survey"))
    direction = str(args.get("direction") or "").strip().lower()
    stops = SWEEP if survey else DIRECTIONS.get(direction, ())
    if stops:
        _pin_robot_address()
        frames = _look_around(stops)
    else:
        frames = [f for f in (_frame(),) if f]
    if not frames:
        return _out({"ok": False,
                     "error": "no camera frame right now -- the robot app may "
                              "not be running, or my eyes are closed"})

    question = str(args.get("question") or "").strip()
    ask = PROMPT + (SURVEY_NOTE if len(frames) > 1 else "")
    if not survey and direction in ("left", "right"):
        ask += f" You have turned to look to your {direction}."
    if question:
        ask += f"\n\nThe person asked: {question}"
    try:
        seen = llm.describe_images(frames, ask, cfg.get("llm_model") or "",
                                   provider, {"temperature": 0.3})
    except Exception as e:
        return _out({"ok": False, "error": f"could not look: {e}"[:220]})

    return _out({"ok": True, "seen": seen, "views": len(frames),
                 "surveyed": survey, "direction": direction or "ahead",
                 "detail": "This is what you can actually see right now. "
                           "Say it in your own voice; do not add details that "
                           "are not in it."})


def _out(payload: dict) -> int:
    json.dump(payload, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
