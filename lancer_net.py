"""
Finding the robot.

There is more than one way to reach a Reachy Mini from this Mac -- a USB
ethernet link when it is plugged in, Tailscale when it is not, plain wifi at a
demo where neither is set up -- and which one works changes between rooms. So
the address is a setting rather than a constant, and reaching the robot means
trying the list in order and remembering what answered.

Everything that talks to the daemon comes through here: the dashboard, and the
skills that move the robot. Two copies of a host list is how you get a robot
that answers status polls on one address while refusing to nod on another.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import time

import requests

logger = logging.getLogger(__name__)

# Tried when nothing is configured: the wired link, then Tailscale.
FALLBACK_HOSTS = ("10.5.1.1", "100.116.136.5")
STATUS_PATH = "/api/daemon/status"

# How long a working address is trusted before it is checked again. Long enough
# that a burst of dashboard polls costs one probe, short enough that unplugging
# the cable is noticed while you are still standing next to the robot.
_CACHE_SECONDS = 20.0
# Per-host probe timeout. This is a wired LAN or a Tailscale hop, so anything
# slower than this is not a robot that will keep up with a conversation -- and a
# generous timeout here is felt as a frozen dashboard when the robot is off.
_PROBE_TIMEOUT = 1.5

_lock = threading.Lock()
_cached_base: str | None = None
_cached_at: float = 0.0


def _config() -> dict:
    """Read settings without importing the runtime at module load."""
    try:
        import lancer_runtime as rt

        return rt.load()
    except Exception:            # skills run standalone; config is optional
        return {}


def hosts() -> list[str]:
    """
    Candidate addresses, most likely first.

    LANCER_ROBOT always wins, so a demo machine can be pointed somewhere else
    for one session without editing anything that persists.
    """
    found: list[str] = []
    for source in (os.environ.get("LANCER_ROBOT", ""),
                   str(_config().get("robot_hosts") or "")):
        for host in source.replace(";", ",").split(","):
            host = host.strip()
            if host and host not in found:
                found.append(host)
    for host in FALLBACK_HOSTS:
        if host not in found:
            found.append(host)
    return found


def port() -> int:
    try:
        return int(_config().get("robot_port") or 8000)
    except (TypeError, ValueError):
        return 8000


def url_for(host: str) -> str:
    """Accept a bare address, a host:port, or a full URL."""
    host = host.strip().rstrip("/")
    if host.startswith(("http://", "https://")):
        return host
    if ":" in host and not host.count(":") > 1:   # host:port, not IPv6
        return f"http://{host}"
    return f"http://{host}:{port()}"


def check(host: str, timeout: float = _PROBE_TIMEOUT) -> dict:
    """Probe one address. Never raises -- unreachable is a normal answer."""
    base = url_for(host)
    started = time.monotonic()
    try:
        response = requests.get(f"{base}{STATUS_PATH}", timeout=timeout)
        elapsed = round((time.monotonic() - started) * 1000)
        if not response.ok:
            return {"host": host, "base": base, "reachable": False,
                    "ms": elapsed, "error": f"daemon returned {response.status_code}"}
        body = response.json() if response.content else {}
        return {"host": host, "base": base, "reachable": True, "ms": elapsed,
                "state": body.get("state"), "version": body.get("version"),
                "wlan_ip": body.get("wlan_ip")}
    except requests.RequestException as e:
        return {"host": host, "base": base, "reachable": False,
                "ms": round((time.monotonic() - started) * 1000),
                "error": type(e).__name__}


def base(force: bool = False) -> str | None:
    """
    The first address that answers, or None when the robot is unreachable.

    Cached, because without it every dashboard poll paid for a full sweep of
    dead addresses -- which is exactly the situation where the page most needs
    to stay responsive.
    """
    global _cached_base, _cached_at
    with _lock:
        fresh = (time.monotonic() - _cached_at) < _CACHE_SECONDS
        if _cached_base and fresh and not force:
            return _cached_base
        remembered = _cached_base

    candidates = hosts()
    if remembered:
        # Try what worked last time first, wherever it sits in the list.
        candidates = [remembered] + [h for h in candidates
                                     if url_for(h) != remembered]
    for host in candidates:
        result = check(host)
        if result["reachable"]:
            with _lock:
                _cached_base = result["base"]
                _cached_at = time.monotonic()
            return result["base"]

    with _lock:
        _cached_base = None
        _cached_at = time.monotonic()
    return None


def forget() -> None:
    """Drop the cached address, so the next call probes again."""
    global _cached_base, _cached_at
    with _lock:
        _cached_base = None
        _cached_at = 0.0


def probe_all() -> list[dict]:
    """Every candidate with its result, for the dashboard's Test button."""
    return [check(host) for host in hosts()]


def local_addresses() -> list[str]:
    """
    This machine's addresses, for the robot to dial back on.

    The robot app needs the Mac's IP to reach the RAG server, and "which of my
    IPs is the robot actually able to see" is the question people get wrong when
    a robot is silent for no visible reason.
    """
    found: list[str] = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None,
                                       socket.AF_INET):
            address = info[4][0]
            if address not in found and not address.startswith("127."):
                found.append(address)
    except OSError:
        pass
    # getaddrinfo misses interfaces that have no reverse name, which on this
    # machine is usually the USB-ethernet link the robot is actually on.
    for target in ("10.5.1.1", "8.8.8.8"):
        try:
            probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            probe.settimeout(0.3)
            probe.connect((target, 80))
            address = probe.getsockname()[0]
            probe.close()
            if address not in found and not address.startswith("127."):
                found.append(address)
        except OSError:
            continue
    return found
