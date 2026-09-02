"""
Lancer dashboard: configuration, transcript, retrieval inspection, robot control.

Mounted by server.py. Everything here is read/write against the same runtime
config the voice pipeline uses, so changes apply to the next question without a
restart -- except the Whisper model, which is reloaded on demand.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import requests
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import lancer_net
import lancer_runtime as rt

router = APIRouter()

def robot_api() -> str:
    """
    Base URL of the first reachable robot daemon.

    The address list is a setting, not a constant -- see lancer_net, which every
    caller shares so the dashboard and the movement skills can never disagree
    about where the robot is.
    """
    return lancer_net.base() or lancer_net.url_for(lancer_net.hosts()[0])
APP_NAME = "reachy_mini_lancer_app"
DASHBOARD_HTML = Path(__file__).parent / "dashboard.html"


class ConfigUpdate(BaseModel):
    changes: dict[str, Any]


class QueryRequest(BaseModel):
    query: str


@router.get("/", response_class=HTMLResponse)
async def dashboard() -> str:
    if not DASHBOARD_HTML.exists():
        return "<h1>dashboard.html missing</h1>"
    return DASHBOARD_HTML.read_text()


@router.get("/api/config")
async def get_config() -> dict:
    # Limits go out with the config so the dashboard can show a setting's range
    # and refuse a bad value in the field. Without them a number outside the
    # range was quietly clamped on save, which looks like the save failed.
    return {"config": rt.load(), "defaults": rt.DEFAULTS,
            "limits": {k: list(v) for k, v in rt.LIMITS.items()}}


@router.post("/api/config")
async def set_config(update: ConfigUpdate) -> dict:
    return {"config": rt.update(update.changes)}


@router.post("/api/config/reset")
async def reset_config() -> dict:
    return {"config": rt.reset()}


@router.get("/api/history")
async def get_history(limit: int = 200) -> dict:
    return {"history": rt.history(limit)}


@router.delete("/api/history")
async def delete_history() -> dict:
    rt.clear_history()
    return {"status": "ok"}


# --- Reflection review ---------------------------------------------------
# The reflection worker proposes skills and applies nothing. This is where a
# human reads a proposal and decides, which is the half of that loop that makes
# it safe to leave running next to a robot students talk to.


def _proposals_db():
    """Connect to the memory database the worker writes proposals into."""
    import lancer_memory

    lancer_memory.init()
    return lancer_memory


@router.get("/api/proposals")
async def list_proposals(status: str = "pending") -> dict:
    memory = _proposals_db()
    rows = memory._execute(
        "SELECT id, created_at, name, description, rationale, examples, "
        "data_source, status FROM skill_proposals WHERE status = ? "
        "ORDER BY created_at DESC LIMIT 50", (status,)
    )
    import json as _json

    return {"proposals": [
        {"id": r[0], "created_at": r[1], "name": r[2], "description": r[3],
         "rationale": r[4], "examples": _json.loads(r[5] or "[]"),
         "data_source": r[6], "status": r[7]}
        for r in rows
    ]}


class ProposalReview(BaseModel):
    status: str          # "approved" or "rejected"
    reviewer: str = "acm"


@router.post("/api/proposals/{proposal_id}")
async def review_proposal(proposal_id: int, review: ProposalReview) -> dict:
    """
    Record a decision. Approving does not create anything.

    Someone still writes the skill by hand -- approval means "this is worth
    building", not "build it". The worker has no write access to skills/ and
    this endpoint does not give it any.
    """
    if review.status not in ("approved", "rejected"):
        raise HTTPException(status_code=400, detail="status must be approved or rejected")
    import time

    memory = _proposals_db()
    memory._execute(
        "UPDATE skill_proposals SET status = ?, reviewed_at = ?, reviewer = ? "
        "WHERE id = ?", (review.status, time.time(), review.reviewer, proposal_id)
    )
    return {"id": proposal_id, "status": review.status}


@router.get("/api/skills")
async def dashboard_skills() -> dict:
    import lancer_skills

    return {"skills": lancer_skills.describe()}


@router.get("/api/memory")
async def dashboard_memory(subject: str = "student") -> dict:
    import lancer_memory

    return {"stats": lancer_memory.stats(),
            "facts": lancer_memory.recall(subject, limit=25)}


# --- Providers and keys --------------------------------------------------

class ApiKeyUpdate(BaseModel):
    provider: str
    key: str = ""


@router.get("/api/providers")
async def providers() -> dict:
    import lancer_llm

    return {"providers": lancer_llm.status(),
            "current": rt.get("llm_provider"),
            "model": rt.get("llm_model")}


@router.post("/api/providers/key")
async def set_api_key(update: ApiKeyUpdate) -> dict:
    """
    Store or clear a provider's API key.

    The key goes to lancer_secrets.json, which is untracked and mode 600 -- not
    to lancer_config.json, which is committed and which this same dashboard
    writes to. Keys are never read back out to the browser; the UI only ever
    learns whether one is set.
    """
    import lancer_llm

    if update.provider not in lancer_llm.PROVIDERS:
        raise HTTPException(status_code=400,
                            detail=f"unknown provider {update.provider!r}")
    lancer_llm.save_key(update.provider, update.key.strip())
    return {"provider": update.provider,
            "key_set": bool(lancer_llm.api_key(update.provider))}


@router.post("/api/providers/test")
async def test_provider(update: ApiKeyUpdate) -> dict:
    """Ask the provider to say one word, so a bad key fails here and not mid-answer."""
    import lancer_llm

    provider = update.provider
    model = rt.get("llm_model") if provider == rt.get("llm_provider") else None
    model = model or lancer_llm.SUGGESTED_MODELS.get(provider, [""])[0]
    try:
        reply = lancer_llm.complete(
            [{"role": "user", "content": "Reply with the single word: ready"}],
            model=model, provider=provider, options={"temperature": 0},
        )
        return {"ok": True, "provider": provider, "model": model,
                "reply": reply.strip()[:120]}
    except Exception as e:
        return {"ok": False, "provider": provider, "model": model,
                "error": f"{type(e).__name__}: {e}"[:300]}


# --- Debug traces --------------------------------------------------------

@router.get("/api/traces")
async def traces(limit: int = 10) -> dict:
    """
    The last few turns, with every decision that produced them.

    This is the view that turns "the answer was wrong" into a specific stage:
    what was transcribed, whether the addressing gate let it through, which
    route it took, every retrieval candidate with its score and why it was
    dropped, which skills ran with what arguments, and the grounding verdict on
    each sentence.
    """
    import server

    return {"traces": list(server._traces)[:limit]}


@router.get("/api/models")
async def list_models() -> dict:
    """Ollama models available for generation (embedding models filtered out)."""
    try:
        res = requests.get("http://localhost:11434/api/tags", timeout=5)
        names = [m["name"] for m in res.json().get("models", [])]
        return {"models": [n for n in names if "embed" not in n]}
    except requests.RequestException as e:
        return {"models": [], "error": str(e)}


@router.get("/api/voices")
async def list_voices() -> dict:
    """Kokoro voices, grouped so the picker is navigable."""
    try:
        import server

        voices = sorted(server._get_kokoro().get_voices())
    except Exception as e:
        return {"voices": [], "error": str(e)}
    return {"voices": voices}


@router.get("/api/robot/probe")
async def probe_robot() -> dict:
    """
    Try every configured address and report what each one did.

    "The robot is offline" is usually "the robot is on the other address", and
    without this the only way to tell them apart was a terminal. Shows this
    machine's own addresses too: the robot app has to dial back to the RAG
    server, and picking the wrong one of these is a robot that hears you and
    never answers.
    """
    import asyncio

    lancer_net.forget()
    candidates = await asyncio.to_thread(lancer_net.probe_all)
    return {"candidates": candidates,
            "local_addresses": lancer_net.local_addresses(),
            "server_port": 7870}


@router.get("/api/robot/status")
async def robot_status() -> dict:
    base = lancer_net.base()
    out: dict[str, Any] = {"host": base or lancer_net.hosts()[0],
                           "hosts": lancer_net.hosts(),
                           "reachable_base": base}
    try:
        daemon = requests.get(f"{robot_api()}/api/daemon/status", timeout=4).json()
        out["online"] = True
        out["state"] = daemon.get("state")
        out["version"] = daemon.get("version")
        out["wlan_ip"] = daemon.get("wlan_ip")
        backend = daemon.get("backend_status") or {}
        out["motors"] = backend.get("motor_control_mode")
        stats = backend.get("control_loop_stats") or {}
        out["control_hz"] = round(stats.get("mean_control_loop_frequency", 0), 1)
    except requests.RequestException as e:
        out["online"] = False
        # str(e) here is a nested HTTPConnectionPool/NewConnectionError dump.
        # It says nothing the host and the failure kind do not, and it fills the
        # panel with noise at the exact moment you want to read it.
        out["error"] = (f"no daemon answering at "
                        f"{', '.join(out['hosts'])} ({type(e).__name__})")
        return out
    try:
        app = requests.get(f"{robot_api()}/api/apps/current-app-status", timeout=4).json()
        out["app"] = (app or {}).get("info", {}).get("name")
        out["app_state"] = (app or {}).get("state")
    except requests.RequestException:
        out["app"] = None
    try:
        vol = requests.get(f"{robot_api()}/api/volume/current", timeout=4).json()
        out["volume"] = vol.get("volume")
    except requests.RequestException:
        pass
    return out


@router.post("/api/robot/{action}")
async def robot_control(action: str, volume: int | None = None) -> dict:
    """start / stop the Lancer app, or set speaker volume."""
    try:
        if action == "start":
            r = requests.post(f"{robot_api()}/api/apps/start-app/{APP_NAME}", timeout=60)
        elif action == "stop":
            r = requests.post(f"{robot_api()}/api/apps/stop-current-app", timeout=30)
        elif action == "volume" and volume is not None:
            r = requests.post(f"{robot_api()}/api/volume/set",
                              json={"volume": int(volume)}, timeout=10)
        elif action == "test-sound":
            r = requests.post(f"{robot_api()}/api/volume/test-sound", timeout=20)
        else:
            raise HTTPException(status_code=400, detail=f"unknown action {action}")
        return {"status": "ok", "code": r.status_code}
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=str(e)) from e


@router.post("/api/retrieve")
async def inspect_retrieval(req: QueryRequest) -> dict:
    """Run retrieval only, so chunk quality can be judged without generating."""
    import asyncio

    import server

    context, sources, relevant, score = await asyncio.to_thread(
        server.retrieve, req.query
    )
    chunks = context.split("\n\n---\n\n") if context else []
    return {
        "query": req.query,
        "relevant": relevant,
        "best_score": score,
        "results": [
            {"source": Path(s).name, "text": c.strip()[:600]}
            for s, c in zip(sources, chunks)
        ],
    }


@router.post("/api/ask")
async def ask(req: QueryRequest) -> dict:
    """Text chat through the same RAG path the voice pipeline uses."""
    import server

    answer, sources = await server._process_query(req.query, server._history)
    answer = server.sanitize_for_speech(answer)
    server._history.append({"role": "user", "content": req.query})
    server._history.append({"role": "assistant", "content": answer})
    del server._history[:-int(rt.get("history_turns"))]
    rt.record({"heard": req.query, "answer": answer, "sources": sources,
               "unknown": server.NO_ANSWER in answer, "model": rt.get("llm_model"),
               "via": "dashboard"})
    return {"answer": answer, "sources": [Path(s).name for s in sources]}


@router.get("/api/stats")
async def stats() -> dict:
    import server

    return {
        "chunks": server.collection.count(),
        "lexical_chunks": len(server._bm25_docs),
        "history_entries": len(rt.history(10_000)),
        "reranker_loaded": server._reranker is not None,
        "whisper_loaded": server._whisper_model is not None,
        "kokoro_loaded": server._kokoro is not None,
    }
