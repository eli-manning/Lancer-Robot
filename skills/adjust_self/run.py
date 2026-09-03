"""
Let Lancer change things about himself.

"Turn it down, you're too loud" is a request a robot should be able to act on
itself, in the same breath as answering. Routing it through a person at a
dashboard makes the robot feel like equipment rather than something you are
talking to.

What is exposed here is deliberately a short list: presentation, not judgement.
Volume, pace, voice, brevity, how strongly he follows a face, and whether he
makes a working noise. Every one of them is reversible, obvious from the
outside when it changes, and affects only how Lancer comes across.

What is deliberately NOT here, and why:

  barge-in       Turning off interruption would let Lancer talk over someone
                 with no way to stop him. The one control that must stay with
                 the person in the room.
  grounding      Whether answers are checked against the documents is not his
                 to relax, and a model that finds verification inconvenient has
                 an obvious incentive to.
  retrieval      Thresholds decide when he says "I don't know". Same reasoning.
  provider, keys, addresses
                 Infrastructure, not self. Nothing here should be able to move
                 the conversation to a different company's servers.

Relative words are supported because that is how people actually ask. Nobody
says "set volume to 70"; they say "turn it up".
"""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))                    # skills/, for _robot
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))   # repo root, for config

import lancer_runtime as rt                          # noqa: E402
from _robot import robot_get, robot_post             # noqa: E402

UP = {"up", "louder", "higher", "more", "increase", "faster", "quicker", "raise"}
DOWN = {"down", "quieter", "softer", "lower", "less", "decrease", "slower",
        "reduce", "shorter"}
ON = {"on", "yes", "true", "enable", "enabled"}
OFF = {"off", "no", "false", "disable", "disabled", "mute", "silent"}

# setting -> (config key, low, high, step). Volume is not a config key: it lives
# on the robot's daemon, so it is handled separately.
NUMERIC = {
    "speech_speed": ("tts_speed", 0.5, 2.0, 0.1),
    "talkativeness": ("max_sentences", 1, 10, 1),
    "face_tracking": ("head_tracking_weight", 0.0, 1.0, 0.2),
}
VOLUME_STEP = 15


def _clamp(value, low, high):
    return min(max(value, low), high)


def _volume_now():
    ok, body = robot_get("/api/volume/current", timeout=6)
    if ok and isinstance(body, dict) and body.get("volume") is not None:
        return int(body["volume"])
    return None


def _set_volume(value):
    current = _volume_now()
    if current is None:
        return {"ok": False, "setting": "volume",
                "error": "the robot is not reachable, so I cannot change my volume"}
    if value is None:
        return {"ok": True, "setting": "volume", "value": current,
                "detail": f"volume is {current} out of 100"}

    word = str(value).strip().lower()
    if word in UP:
        target = current + VOLUME_STEP
    elif word in DOWN:
        target = current - VOLUME_STEP
    elif word in ("max", "maximum", "full"):
        target = 100
    elif word in ("min", "minimum"):
        target = 10          # not 0: a robot that has muted itself looks broken
    else:
        try:
            target = int(float(word.rstrip("%")))
        except ValueError:
            return {"ok": False, "setting": "volume",
                    "error": f"cannot read {value!r} as a volume"}
    target = int(_clamp(target, 0, 100))
    ok, body = robot_post("/api/volume/set", {"volume": target}, timeout=8)
    if not ok:
        return {"ok": False, "setting": "volume", "error": str(body)[:200]}
    return {"ok": True, "setting": "volume", "was": current, "value": target,
            "detail": f"volume {current} -> {target}"}


def _set_numeric(setting, value):
    key, low, high, step = NUMERIC[setting]
    current = float(rt.get(key))
    if value is None:
        return {"ok": True, "setting": setting, "value": current,
                "detail": f"{setting} is {current}"}

    word = str(value).strip().lower()
    if word in UP:
        target = current + step
    elif word in DOWN:
        target = current - step
    elif word in ("max", "maximum", "full"):
        target = high
    elif word in ("min", "minimum"):
        target = low
    else:
        try:
            target = float(word)
        except ValueError:
            return {"ok": False, "setting": setting,
                    "error": f"cannot read {value!r} as a number"}
    target = _clamp(target, low, high)
    if isinstance(rt.DEFAULTS.get(key), int):
        target = int(round(target))
    rt.update({key: target})
    return {"ok": True, "setting": setting, "was": current, "value": target,
            "detail": f"{setting} {current} -> {target}"}


def _set_voice(value):
    current = rt.get("tts_voice")
    if value is None:
        return {"ok": True, "setting": "voice", "value": current,
                "detail": f"my voice is {current}"}
    name = str(value).strip()
    # Not validated against the installed voice list on purpose: that list lives
    # in the server process, and a wrong name is obvious the moment he speaks
    # and is fixed by asking again.
    rt.update({"tts_voice": name})
    return {"ok": True, "setting": "voice", "was": current, "value": name,
            "detail": f"voice {current} -> {name}"}


def _set_thinking_sound(value):
    current = bool(rt.get("thinking_sound"))
    if value is None:
        return {"ok": True, "setting": "thinking_sound", "value": current,
                "detail": f"thinking sound is {'on' if current else 'off'}"}
    word = str(value).strip().lower()
    if word in ON:
        target = True
    elif word in OFF:
        target = False
    else:
        return {"ok": False, "setting": "thinking_sound",
                "error": f"cannot read {value!r} as on or off"}
    rt.update({"thinking_sound": target})
    return {"ok": True, "setting": "thinking_sound", "was": current,
            "value": target,
            "detail": f"thinking sound {'on' if target else 'off'}"}


def _report_all():
    settings = {"volume": _volume_now(), "voice": rt.get("tts_voice"),
                "thinking_sound": bool(rt.get("thinking_sound"))}
    for name, (key, _low, _high, _step) in NUMERIC.items():
        settings[name] = rt.get(key)
    return {"ok": True, "settings": settings,
            "detail": "everything I can change about myself"}


def main() -> int:
    try:
        args = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        args = {}

    setting = str(args.get("setting") or "").strip().lower()
    value = args.get("value")
    if isinstance(value, str) and not value.strip():
        value = None

    if not setting:
        result = _report_all()
    elif setting == "volume":
        result = _set_volume(value)
    elif setting == "voice":
        result = _set_voice(value)
    elif setting == "thinking_sound":
        result = _set_thinking_sound(value)
    elif setting in NUMERIC:
        result = _set_numeric(setting, value)
    else:
        result = {"ok": False,
                  "error": f"I cannot change {setting!r}. I can change: volume, "
                           "speech_speed, voice, talkativeness, face_tracking, "
                           "thinking_sound."}

    json.dump(result, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
