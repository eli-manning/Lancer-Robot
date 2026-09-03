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

# Head yaw for each look, in radians. Roughly 35 degrees either side: wide
# enough to cover a room from a desk, inside what the neck will do comfortably.
SWEEP = (0.62, 0.0, -0.62)
MOVE_SECONDS = 0.5
# The app posts a frame about once a second, so a shorter wait than this hands
# the model the previous angle's picture and the survey silently sees nothing new.
SETTLE_SECONDS = 1.2

PROMPT = (
    "You are looking through your own camera, as a small desk robot. "
    "Answer in one or two short spoken sentences. Describe only what is "
    "actually visible. If you cannot make something out, say so plainly rather "
    "than guessing. Do not mention the photograph, the image, the camera, "
    "lighting or resolution -- just say what is there. Describe what people are "
    "doing, holding or wearing; do not guess at anyone's ethnicity, age, or "
    "anything else you cannot actually see."
)
SURVEY_NOTE = (
    " These are several views taken by turning your head from your left, to "
    "the centre, to your right. Together they are one room, so do not count "
    "the same person twice if they appear in more than one view, and answer "
    "about the whole space rather than describing each picture."
)


def _frame() -> bytes | None:
    try:
        response = requests.get(f"{SERVER}/api/frame", timeout=6)
    except requests.RequestException:
        return None
    return response.content if response.ok and response.content else None


def _sweep() -> list[bytes]:
    """
    Look left, centre and right, keeping a frame from each.

    Tracking is switched off for the duration for the same reason the sleep
    sequence does it: the daemon's face tracker and this are both steering the
    head, and two writers on the same servos is how the head ends up shaking.
    """
    ok, pose = robot_get("/api/state/present_head_pose", timeout=6)
    base = pose if ok and isinstance(pose, dict) else {}
    robot_post("/api/media/tracking/disable", timeout=6)
    frames: list[bytes] = []
    try:
        for yaw in SWEEP:
            head = {
                "x": base.get("x", 0.0), "y": base.get("y", 0.0),
                "z": base.get("z", 0.0), "roll": base.get("roll", 0.0),
                "pitch": base.get("pitch", 0.0), "yaw": yaw,
            }
            robot_post("/api/move/goto",
                       {"head_pose": head, "duration": MOVE_SECONDS},
                       timeout=8)
            time.sleep(MOVE_SECONDS + SETTLE_SECONDS)
            jpeg = _frame()
            if jpeg:
                frames.append(jpeg)
    finally:
        # Always put the head back and hand tracking over again, even if a
        # move failed -- otherwise he is left staring at a wall.
        head = {"x": base.get("x", 0.0), "y": base.get("y", 0.0),
                "z": base.get("z", 0.0), "roll": base.get("roll", 0.0),
                "pitch": base.get("pitch", 0.0), "yaw": 0.0}
        robot_post("/api/move/goto",
                   {"head_pose": head, "duration": MOVE_SECONDS}, timeout=8)
        robot_post("/api/media/tracking/enable", {"weight": 1.0}, timeout=6)
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
    frames = _sweep() if survey else [f for f in (_frame(),) if f]
    if not frames:
        return _out({"ok": False,
                     "error": "no camera frame right now -- the robot app may "
                              "not be running, or my eyes are closed"})

    question = str(args.get("question") or "").strip()
    ask = PROMPT + (SURVEY_NOTE if len(frames) > 1 else "")
    if question:
        ask += f"\n\nThe person asked: {question}"
    try:
        seen = llm.describe_images(frames, ask, cfg.get("llm_model") or "",
                                   provider, {"temperature": 0.3})
    except Exception as e:
        return _out({"ok": False, "error": f"could not look: {e}"[:220]})

    return _out({"ok": True, "seen": seen, "views": len(frames),
                 "surveyed": survey,
                 "detail": "This is what you can actually see right now. "
                           "Say it in your own voice; do not add details that "
                           "are not in it."})


def _out(payload: dict) -> int:
    json.dump(payload, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
