"""
Shared helper for skills that move the robot.

The Reachy Mini daemon exposes movement over plain HTTP, so a skill that makes
the robot do something is an ordinary subprocess skill like any other -- no
separate on-robot command channel, no new protocol. The Mac reaches the daemon
over the USB-Ethernet link, and falls back to Tailscale when that is unplugged.

The emotion library is the one installed alongside Pollen's conversation app
(pollen-robotics/reachy-mini-emotions-library, Apache-2.0). It is a dataset of
recorded moves the daemon plays by name, which is why Lancer can have eighty-odd
expressions without a line of animation code.
"""

from __future__ import annotations

import os

import requests

ROBOT_HOSTS = [h for h in (os.environ.get("LANCER_ROBOT"),
                           "10.5.1.1", "100.116.136.5") if h]
EMOTION_LIBRARY = "pollen-robotics/reachy-mini-emotions-library"
TIMEOUT = 8


def robot_base() -> str | None:
    """First daemon that answers. None when the robot is unreachable."""
    for host in ROBOT_HOSTS:
        try:
            if requests.get(f"http://{host}:8000/api/daemon/status",
                            timeout=3).ok:
                return f"http://{host}:8000"
        except requests.RequestException:
            continue
    return None


def robot_post(path: str, payload: dict | None = None,
               timeout: int = TIMEOUT) -> tuple[bool, dict | str]:
    """POST to the daemon. Returns (ok, body-or-error)."""
    base = robot_base()
    if base is None:
        return False, "robot is not reachable"
    try:
        response = requests.post(f"{base}{path}", json=payload, timeout=timeout)
    except requests.RequestException as e:
        return False, f"request failed: {e}"
    if not response.ok:
        return False, f"daemon returned {response.status_code}: {response.text[:160]}"
    try:
        return True, response.json()
    except ValueError:
        return True, response.text[:200]


def robot_get(path: str, timeout: int = TIMEOUT) -> tuple[bool, object]:
    base = robot_base()
    if base is None:
        return False, "robot is not reachable"
    try:
        response = requests.get(f"{base}{path}", timeout=timeout)
        response.raise_for_status()
        return True, response.json()
    except Exception as e:
        return False, f"request failed: {e}"


def emotions() -> list[str]:
    """Every recorded move the library offers, straight from the daemon."""
    from urllib.parse import quote

    ok, body = robot_get(
        f"/api/move/recorded-move-datasets/list/{quote(EMOTION_LIBRARY, safe='')}"
    )
    return body if ok and isinstance(body, list) else []


def play_recorded(name: str) -> tuple[bool, dict | str]:
    from urllib.parse import quote

    return robot_post(
        f"/api/move/play/recorded-move-dataset/"
        f"{quote(EMOTION_LIBRARY, safe='')}/{quote(name, safe='')}"
    )
