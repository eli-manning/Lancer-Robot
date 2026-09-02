"""
Runtime configuration and conversation history for Lancer.

Settings live in a JSON file so the dashboard can change model, voice, retrieval
depth and thresholds without editing code or restarting the server. History is
appended to JSONL so the transcript survives restarts and can be reviewed later.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

BASE = Path(__file__).parent
CONFIG_PATH = BASE / "lancer_config.json"
HISTORY_PATH = BASE / "lancer_history.jsonl"
HISTORY_LIMIT = 500          # entries returned to the dashboard

_lock = threading.Lock()

DEFAULTS: dict[str, Any] = {
    # generation
    "llm_model": "qwen2.5:3b",
    "temperature": 0.3,
    "max_sentences": 3,
    "history_turns": 6,
    # retrieval
    "top_k": 5,
    "dense_k": 20,
    "lexical_k": 20,
    "fuse_k": 20,
    "rerank_enabled": True,
    "verify_grounding": True,
    "relevance_min": -6.0,
    # skills, memory and streaming
    # "native" runs the pipeline in server.py. "zeroclaw" hands the
    # conversation to the gateway over its WebSocket. The rollback from one to
    # the other is this line.
    "conversation_backend": "native",
    "skills_enabled": True,
    # Gestures fired by rule rather than by tool call: confused when it does not
    # know, welcoming when greeted. No latency cost, works on every route.
    "auto_react": True,
    "skill_timeout": 12,
    "memory_enabled": True,
    "stream_enabled": True,
    # A sentence cut off by a false barge-in is resumed from the start of that
    # sentence rather than abandoned. Off means the old behaviour: drop it.
    "resume_after_false_barge": True,
    "resume_grace": 6.0,        # seconds a dropped remainder stays resumable
    "thinking_sound": True,     # audible cue while the server is working
    "thinking_sound_volume": 0.06,
    # speech
    "tts_voice": "bm_george",
    "tts_speed": 1.0,
    "whisper_model": "small.en",
    # robot behaviour (read by the app at startup)
    "speech_multiplier": 2.0,
    "speech_rms_ceiling": 0.045,
    "barge_enabled": True,
    "barge_echo_headroom": 1.4,
    "barge_grace": 0.8,
    "doa_enabled": True,
    "yolo_tracking": True,
    "aim_smoothing": 0.45,
    "aim_deadband_frac": 0.022,
    "doa_priority": True,
    "doa_deadband": 0.28,
    # Which way a positive yaw turns depends on how the mic array is mounted
    # relative to the body. Flip this rather than redeploying if Lancer turns
    # away from the speaker instead of toward them.
    "doa_yaw_sign": 1.0,
    # Share of a sound-turn the head takes. Zero: the daemon tracker overwrites
    # head poses within a frame, so any share here is simply taken away from the
    # body, which is the only actuator whose turn persists.
    "doa_head_share": 0.0,
    # Automatic body yaw is the daemon solving for the body on every head move.
    # It is what keeps head and body within their mechanical limits, so it stays
    # on unless body_assist is deliberately taking over the body -- and the two
    # must not both be enabled, because they fight and the robot oscillates.
    "auto_body_yaw": True,
    # Body assist: bring the body into a turn well before automatic body yaw's
    # own 65-degree threshold would. Off until it is proven on hardware.
    "body_assist": False,
    "body_assist_deadband": 0.20,
    "body_assist_gain": 0.22,
    "body_assist_max_step": 0.05,
    "body_assist_interval": 0.15,
    "head_tracking_weight": 1.0,
    "persona": "",
}

_cache: dict[str, Any] | None = None
_cache_mtime: float = 0.0


def load() -> dict[str, Any]:
    """
    Return the current config, merging defaults over anything on disk.

    The file is re-read whenever it changes on disk. Caching it for the life of
    the process meant an edit from anywhere else was silently overwritten the
    next time this process saved.
    """
    global _cache, _cache_mtime
    with _lock:
        try:
            mtime = CONFIG_PATH.stat().st_mtime
        except OSError:
            mtime = 0.0
        if _cache is not None and mtime != _cache_mtime:
            _cache = None
        _cache_mtime = mtime
        if _cache is None:
            data = {}
            if CONFIG_PATH.exists():
                try:
                    data = json.loads(CONFIG_PATH.read_text())
                except (OSError, json.JSONDecodeError):
                    data = {}
            _cache = {**DEFAULTS, **data}
        return dict(_cache)


def get(key: str) -> Any:
    return load().get(key, DEFAULTS.get(key))


# Ranges for settings where an out-of-range value breaks something downstream
# rather than merely tuning it badly.
LIMITS: dict[str, tuple[float, float]] = {
    "tts_speed": (0.5, 2.0),          # Kokoro raises outside this
    "temperature": (0.0, 2.0),
    "top_k": (1, 20),
    "dense_k": (1, 100),
    "lexical_k": (1, 100),
    "fuse_k": (1, 100),
    "max_sentences": (1, 10),
    "history_turns": (0, 40),
    "speech_multiplier": (1.0, 10.0),
    "speech_rms_ceiling": (0.005, 0.5),
    "barge_echo_headroom": (1.0, 5.0),
    "barge_grace": (0.0, 5.0),
    "head_tracking_weight": (0.0, 1.0),
    "aim_smoothing": (0.05, 1.0),
    "aim_deadband_frac": (0.0, 0.3),
    "skill_timeout": (2, 60),
    "resume_grace": (0.0, 60.0),
    "thinking_sound_volume": (0.0, 0.5),   # louder trips the barge-in gate
    "doa_deadband": (0.05, 1.5),
    "doa_yaw_sign": (-1.0, 1.0),
    "doa_head_share": (0.0, 1.0),
    "body_assist_deadband": (0.05, 1.2),
    "body_assist_gain": (0.0, 0.6),        # higher hunts instead of converging
    "body_assist_max_step": (0.01, 0.2),   # radians per step; a velocity cap
    "body_assist_interval": (0.05, 5.0),
}


def _clamp(key: str, value: Any) -> Any:
    limit = LIMITS.get(key)
    if limit is None or isinstance(value, bool):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    low, high = limit
    clamped = min(max(number, low), high)
    return int(clamped) if isinstance(DEFAULTS.get(key), int) else clamped


def update(changes: dict[str, Any]) -> dict[str, Any]:
    """Apply changes, keeping only known keys, and persist them."""
    global _cache
    # Must read the file first: in a fresh process _cache is None, and building
    # from DEFAULTS alone silently discarded every setting already on disk.
    existing = load()
    with _lock:
        current = {**DEFAULTS, **existing}
        for key, value in changes.items():
            if key in DEFAULTS:
                current[key] = _clamp(key, value)
        CONFIG_PATH.write_text(json.dumps(current, indent=2))
        _cache = current
        try:
            globals()["_cache_mtime"] = CONFIG_PATH.stat().st_mtime
        except OSError:
            pass
        return dict(current)


def reset() -> dict[str, Any]:
    global _cache
    with _lock:
        _cache = dict(DEFAULTS)
        CONFIG_PATH.write_text(json.dumps(_cache, indent=2))
        return dict(_cache)


def record(entry: dict[str, Any]) -> None:
    """Append one exchange to the transcript."""
    entry = {"ts": time.time(), **entry}
    with _lock:
        with HISTORY_PATH.open("a") as fh:
            fh.write(json.dumps(entry) + "\n")


def history(limit: int = HISTORY_LIMIT) -> list[dict[str, Any]]:
    if not HISTORY_PATH.exists():
        return []
    with _lock:
        lines = HISTORY_PATH.read_text().splitlines()
    out = []
    for line in lines[-limit:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def clear_history() -> None:
    with _lock:
        HISTORY_PATH.write_text("")
