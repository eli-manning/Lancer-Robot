"""Describe what the robot's camera can see right now."""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

import requests                                     # noqa: E402

import lancer_llm as llm                            # noqa: E402
import lancer_runtime as rt                         # noqa: E402

SERVER = os.environ.get("LANCER_SERVER_SELF", "http://127.0.0.1:7870")

# Short, spoken, and about what is actually there. Without the last line the
# model narrates image quality -- "the image is slightly blurry" -- which is
# true, useless, and not how a person answers "what do you see".
PROMPT = (
    "You are looking through your own camera, as a small desk robot. "
    "Answer in one or two short spoken sentences. Describe only what is "
    "actually visible. If you cannot make something out, say so plainly rather "
    "than guessing. Do not mention the photograph, the image, the camera, "
    "lighting or resolution -- just say what is there."
)


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

    try:
        response = requests.get(f"{SERVER}/api/frame", timeout=6)
    except requests.RequestException as e:
        return _out({"ok": False, "error": f"could not reach the camera: {e}"})
    if not response.ok or not response.content:
        return _out({"ok": False,
                     "error": "no camera frame right now -- the robot app may "
                              "not be running, or my eyes are closed"})

    question = str(args.get("question") or "").strip()
    ask = f"{PROMPT}\n\nThe person asked: {question}" if question else PROMPT
    try:
        seen = llm.describe_image(response.content, ask,
                                  cfg.get("llm_model") or "", provider,
                                  {"temperature": 0.3})
    except Exception as e:
        return _out({"ok": False, "error": f"could not look: {e}"[:220]})

    return _out({"ok": True, "seen": seen,
                 "detail": "This is what you can actually see right now. "
                           "Say it in your own voice; do not add details that "
                           "are not in it."})


def _out(payload: dict) -> int:
    json.dump(payload, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
