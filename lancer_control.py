"""
Lancer control plane: the one process you start by hand.

Everything else -- Ollama, Chroma, the RAG server, the ZeroClaw stack -- is
started, stopped and watched from the dashboard this serves. That is the whole
reason it exists as a separate process. The dashboard used to be served by the
RAG server itself, which meant stopping the server also took away the page you
would use to start it again; a supervisor cannot be one of the things it
supervises.

It deliberately stays small and dependency-light so it is always startable: no
model loading, no torch, no embedding. Starting it costs a few hundred
milliseconds, so a machine that cannot run Lancer can still run its controls.

    uvicorn lancer_control:app --host 0.0.0.0 --port 7871
    (or ./lancer, which does the same thing and prints the URL)

Two kinds of route live here. Process control -- /api/services/* -- is native.
Everything the old dashboard already did is reused from dashboard.py unchanged,
except the handful of endpoints that genuinely need the loaded models (Whisper,
the reranker, the Chroma client); those are forwarded to the RAG server on 7870
and return a plain "offline" object when it is not running, so the page degrades
into a start button instead of a stack trace.
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

import dashboard

logger = logging.getLogger(__name__)

BASE = Path(__file__).parent
RUN_DIR = BASE / ".run"                 # pidfiles and logs; untracked
VENV_BIN = Path(sys.executable).parent
SERVER_PORT = 7870
SERVER_URL = f"http://127.0.0.1:{SERVER_PORT}"

# Endpoints that need the RAG server's loaded models rather than just files on
# disk. Everything else in dashboard.py works standalone and is served directly,
# which is what keeps config, providers, history and robot control usable while
# the server is stopped.
PROXIED = {
    "GET": ["/api/traces", "/api/voices", "/api/stats", "/state"],
    "POST": ["/api/retrieve", "/api/ask", "/api/say", "/api/cue",
             "/api/alarm/stop"],
}


def _which(name: str, venv_first: bool = True) -> str:
    """Prefer the virtualenv's copy of a tool, fall back to PATH."""
    if venv_first:
        local = VENV_BIN / name
        if local.exists():
            return str(local)
    return shutil.which(name) or name


@dataclass
class Service:
    """One managed process, identified by the port it listens on."""

    name: str
    label: str
    why: str                             # shown in the UI: what breaks without it
    command: list[str] = field(default_factory=list)
    port: int | None = None
    kind: str = "process"                # "process" or "compose"
    ready_path: str | None = None        # HTTP path that means "actually serving"
    boot_seconds: int = 2                # how long start-up usually takes

    @property
    def pidfile(self) -> Path:
        return RUN_DIR / f"{self.name}.pid"

    @property
    def logfile(self) -> Path:
        return RUN_DIR / f"{self.name}.log"


def _services() -> list[Service]:
    return [
        Service(
            name="ollama", label="Ollama", port=11434,
            why="Local generation and embeddings. Without it nothing answers.",
            command=[_which("ollama", venv_first=False), "serve"],
            ready_path="/api/tags",
        ),
        Service(
            name="chroma", label="Chroma", port=8001,
            why="The vector index. Without it retrieval returns nothing.",
            command=[_which("chroma"), "run", "--host", "localhost",
                     "--port", "8001", "--path", "./chroma_data"],
            ready_path="/api/v2/heartbeat", boot_seconds=4,
        ),
        Service(
            name="server", label="RAG server", port=SERVER_PORT,
            why="Speech, retrieval and generation. The robot talks to this.",
            command=[_which("uvicorn"), "server:app",
                     "--host", "0.0.0.0", "--port", str(SERVER_PORT)],
            ready_path="/api/stats", boot_seconds=25,
        ),
        Service(
            name="stack", label="ZeroClaw stack", kind="compose",
            why="Gateway, Postgres memory and the reflection worker. Optional.",
            command=["docker", "compose", "-f", "stack/docker-compose.yml"],
            boot_seconds=10,
        ),
    ]


SERVICES = {s.name: s for s in _services()}


# --- Inspecting what is running -----------------------------------------
# Status is read from the world rather than from our own bookkeeping. A
# supervisor that trusts its pidfiles reports "running" for a process that died
# an hour ago, and reports "stopped" for one somebody started in a terminal.
# The port is the truth; the pidfile is only a hint about who to signal.

def _listening(port: int) -> bool:
    try:
        with socket.create_connection(("localhost", port), timeout=0.4):
            return True
    except OSError:
        return False


def _pid_on_port(port: int) -> int | None:
    """Whoever is listening, whether or not we started them."""
    try:
        out = subprocess.run(["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
                             capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.split():
        try:
            return int(line)
        except ValueError:
            continue
    return None


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _uptime(pid: int) -> str:
    try:
        out = subprocess.run(["ps", "-o", "etime=", "-p", str(pid)],
                             capture_output=True, text=True, timeout=5).stdout
        return out.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _compose_state() -> dict[str, Any]:
    """Container names and states, without needing the compose JSON schema."""
    try:
        out = subprocess.run(
            ["docker", "ps", "-a", "--filter", "name=lancer-",
             "--format", "{{.Names}}\t{{.State}}"],
            capture_output=True, text=True, timeout=15, cwd=BASE)
    except (OSError, subprocess.SubprocessError) as e:
        return {"running": False, "detail": f"docker unavailable: {e}"}
    rows = [line.split("\t") for line in out.stdout.strip().splitlines() if line]
    up = [name for name, state in rows if state == "running"]
    return {"running": bool(up),
            "containers": [{"name": n, "state": s} for n, s in rows],
            "detail": ", ".join(up) if up else "no containers running"}


def _ready(service: Service) -> bool:
    """Listening is not the same as serving; uvicorn binds before models load."""
    if not service.ready_path:
        return True
    try:
        return requests.get(f"http://localhost:{service.port}{service.ready_path}",
                            timeout=2).ok
    except requests.RequestException:
        return False


def _status(service: Service) -> dict[str, Any]:
    out: dict[str, Any] = {"name": service.name, "label": service.label,
                           "why": service.why, "kind": service.kind,
                           "port": service.port, "managed": False}
    if service.kind == "compose":
        out.update(_compose_state())
        out["ready"] = out["running"]
        return out

    pid = _pid_on_port(service.port) if service.port else None
    running = pid is not None or (service.port and _listening(service.port))
    out["pid"] = pid
    out["running"] = bool(running)
    out["ready"] = _ready(service) if running else False
    out["uptime"] = _uptime(pid) if pid else ""
    # Whether this supervisor is the one that started it. Adopted processes can
    # still be stopped -- we find them by port -- but it is worth showing.
    try:
        out["managed"] = (service.pidfile.exists()
                          and int(service.pidfile.read_text().strip()) == pid)
    except (OSError, ValueError):
        pass
    if not running and service.logfile.exists():
        out["last_log"] = _tail(service.logfile, 3)
    return out


def _tail(path: Path, lines: int) -> str:
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            block = min(size, lines * 400 + 2048)
            fh.seek(size - block)
            text = fh.read().decode("utf-8", "replace")
        return "\n".join(text.splitlines()[-lines:])
    except OSError:
        return ""


# --- Starting and stopping ----------------------------------------------

def _start(service: Service) -> dict[str, Any]:
    if service.kind == "compose":
        result = subprocess.run(service.command + ["up", "-d"], cwd=BASE,
                                capture_output=True, text=True, timeout=300)
        if result.returncode:
            raise HTTPException(status_code=500,
                                detail=result.stderr.strip()[-400:] or "compose failed")
        return {"started": True, "detail": result.stderr.strip()[-300:]}

    if service.port and _listening(service.port):
        return {"started": False, "detail": f"already listening on {service.port}"}
    if not Path(service.command[0]).exists() and not shutil.which(service.command[0]):
        raise HTTPException(status_code=500,
                            detail=f"{service.command[0]} not found")

    RUN_DIR.mkdir(exist_ok=True)
    # Truncate rather than append: the log is for diagnosing this run, and a
    # file that grows across every restart is one nobody reads.
    log = service.logfile.open("w")
    process = subprocess.Popen(
        service.command, cwd=BASE, stdout=log, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        # Its own process group, so stopping it never signals this supervisor,
        # and so it survives if the supervisor is restarted.
        start_new_session=True,
    )
    service.pidfile.write_text(str(process.pid))
    logger.info("started %s (pid %d)", service.name, process.pid)

    # Wait briefly for an immediate failure -- a missing model, a port clash --
    # so the dashboard can show the reason instead of a service that silently
    # never appears.
    for _ in range(20):
        time.sleep(0.1)
        if process.poll() is not None:
            return {"started": False, "exited": process.returncode,
                    "detail": _tail(service.logfile, 6) or "exited immediately"}
    return {"started": True, "pid": process.pid,
            "detail": f"starting, ready in about {service.boot_seconds}s"}


def _stop(service: Service) -> dict[str, Any]:
    if service.kind == "compose":
        result = subprocess.run(service.command + ["down", "--remove-orphans"],
                                cwd=BASE, capture_output=True, text=True, timeout=180)
        return {"stopped": result.returncode == 0,
                "detail": result.stderr.strip()[-300:] or "down"}

    pid = (_pid_on_port(service.port) if service.port else None)
    if pid is None:
        try:
            pid = int(service.pidfile.read_text().strip())
        except (OSError, ValueError):
            pid = None
    if pid is None or not _alive(pid):
        service.pidfile.unlink(missing_ok=True)
        return {"stopped": False, "detail": "not running"}

    # SIGTERM first so uvicorn closes its sockets and Chroma flushes; SIGKILL
    # only for something that will not leave.
    os.kill(pid, signal.SIGTERM)
    for _ in range(50):
        time.sleep(0.1)
        if not _alive(pid):
            break
    else:
        os.kill(pid, signal.SIGKILL)
        time.sleep(0.3)
    service.pidfile.unlink(missing_ok=True)
    logger.info("stopped %s (pid %d)", service.name, pid)
    return {"stopped": True, "pid": pid}


# --- Routes --------------------------------------------------------------

control = APIRouter()


@control.get("/api/services")
async def list_services() -> dict:
    import asyncio

    names = list(SERVICES)
    results = await asyncio.gather(
        *(asyncio.to_thread(_status, SERVICES[n]) for n in names))
    return {"services": results, "control_port": 7871}


# Registered before the {name} route below: FastAPI matches in registration
# order, so a literal path has to come first or "all" is read as a service name.
@control.post("/api/services/all/stop")
async def stop_everything() -> dict:
    """Stop in dependency order: the server first, so it is not left querying
    a Chroma that has gone away."""
    import asyncio

    done = []
    for name in ("server", "stack", "chroma", "ollama"):
        done.append({"name": name,
                     **await asyncio.to_thread(_stop, SERVICES[name])})
    return {"stopped": done}


@control.post("/api/services/all/start")
async def start_everything() -> dict:
    """Bring up what Lancer needs, in the order it needs it."""
    import asyncio

    done = []
    for name in ("ollama", "chroma", "server"):
        done.append({"name": name,
                     **await asyncio.to_thread(_start, SERVICES[name])})
        await asyncio.sleep(1.0)
    return {"started": done}


@control.post("/api/services/{name}/{action}")
async def control_service(name: str, action: str) -> dict:
    import asyncio

    service = SERVICES.get(name)
    if service is None:
        raise HTTPException(status_code=404, detail=f"unknown service {name!r}")
    if action == "start":
        return await asyncio.to_thread(_start, service)
    if action == "stop":
        return await asyncio.to_thread(_stop, service)
    if action == "restart":
        await asyncio.to_thread(_stop, service)
        await asyncio.sleep(0.5)
        return await asyncio.to_thread(_start, service)
    raise HTTPException(status_code=400, detail="action must be start/stop/restart")


@control.post("/skills/{name}")
async def run_skill(name: str, request: Request) -> JSONResponse:
    """
    Run one skill from the dashboard.

    Forwarded rather than run here: skills that move the robot expect the
    server's view of sleep state, and running two copies of that state in two
    processes is how they end up disagreeing.
    """
    import asyncio

    body = await request.body()
    return await asyncio.to_thread(_forward, "POST", f"/skills/{name}", body, "")


@control.get("/api/services/{name}/log")
async def service_log(name: str, lines: int = 80) -> dict:
    service = SERVICES.get(name)
    if service is None:
        raise HTTPException(status_code=404, detail=f"unknown service {name!r}")
    if service.kind == "compose":
        try:
            out = subprocess.run(
                service.command + ["logs", "--tail", str(lines)], cwd=BASE,
                capture_output=True, text=True, timeout=30)
            return {"name": name, "log": (out.stdout or out.stderr)[-20000:]}
        except (OSError, subprocess.SubprocessError) as e:
            return {"name": name, "log": f"could not read compose logs: {e}"}
    return {"name": name, "log": _tail(service.logfile, min(lines, 500))
            or "no log yet"}


def _forward(method: str, path: str, body: bytes, query: str) -> JSONResponse:
    url = f"{SERVER_URL}{path}" + (f"?{query}" if query else "")
    try:
        response = requests.request(method, url, data=body or None, timeout=180,
                                    headers={"content-type": "application/json"})
    except requests.RequestException as e:
        # Not an error the page should crash on: the RAG server being stopped is
        # a normal state here, and the Services panel is how you fix it.
        return JSONResponse(status_code=200, content={
            "offline": True, "error": f"RAG server not running ({type(e).__name__})",
            "traces": [], "voices": [], "results": [], "history": []})
    try:
        return JSONResponse(status_code=response.status_code, content=response.json())
    except ValueError:
        return JSONResponse(status_code=response.status_code,
                            content={"error": response.text[:500]})


def _register_proxies(target: FastAPI) -> None:
    """Forward the model-backed endpoints, keeping their paths identical."""
    import asyncio

    for method, paths in PROXIED.items():
        for path in paths:
            async def handler(request: Request, _p=path, _m=method) -> JSONResponse:
                body = await request.body()
                return await asyncio.to_thread(
                    _forward, _m, _p, body, request.url.query)
            target.add_api_route(path, handler, methods=[method])


app = FastAPI(title="Lancer Control", docs_url=None, redoc_url=None)


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    page = BASE / "dashboard.html"
    return page.read_text() if page.exists() else "<h1>dashboard.html missing</h1>"


# Order matters: our own routes and the proxies are registered before the
# dashboard router, so the model-backed paths forward to 7870 instead of being
# answered here by importing the entire RAG stack into this process.
app.include_router(control)
_register_proxies(app)
app.include_router(dashboard.router)
