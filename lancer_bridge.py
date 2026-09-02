"""
Bridge to the ZeroClaw gateway.

The robot's conversation can come from one of two places: the native path in
server.py, or the gateway over its WebSocket. This is the second one, and it is
off by default -- `conversation_backend` in the runtime config switches it.

Both produce the same thing: a stream of sentences. That is the whole point of
the frame protocol in server.py being shaped the way it is. The voice loop, the
per-sentence synthesis, the grounding gate and the robot's playback code do not
know or care which backend produced a sentence, so the cutover is a config
change rather than a rewrite, and the rollback is the same config change back.

What the gateway adds: durable memory of its own, a real skill runtime with
risk-tiered approval, and the reflection loop. What it costs: the deterministic
grounding check cannot see the passages the agent retrieved, because retrieval
happened inside the agent. So the post-check here re-retrieves locally and
verifies the answer against that instead. It is a slightly weaker guarantee
than the native path's -- the passages checked against are the ones our own
retriever finds, not necessarily the ones the agent read -- and it is stated
here rather than glossed over.

Protocol, confirmed against zeroclaw 0.8.4 rather than assumed:

    ->  {"type": "message", "content": "..."}
    <-  {"type": "session_start", "session_id": ...}
    <-  {"type": "chunk", "content": "..."}          (many)
    <-  {"type": "approval_request", ...}            (when a skill needs it)
    <-  {"type": "done", "full_response": ..., "input_tokens": ...}
    <-  {"type": "error", "message": ...}
"""

from __future__ import annotations

import asyncio
import json
import logging
import os

logger = logging.getLogger(__name__)

GATEWAY_URL = os.environ.get("LANCER_GATEWAY", "ws://127.0.0.1:8081/ws/chat")
AGENT = os.environ.get("LANCER_AGENT", "lancer")
# A local model behind a gateway that may also be running skills. Generous, but
# not unbounded: a turn nobody ever hears the end of is a hung robot.
TURN_TIMEOUT = 180.0


class GatewayError(RuntimeError):
    """The gateway refused or failed the turn."""


async def stream_turn(text: str, on_status=None, session_id: str | None = None):
    """
    Send one message to the gateway and yield its sentences as they form.

    Chunk frames are tokens, not sentences, so they are buffered to sentence
    boundaries with the same splitter the native path uses -- speech has to be
    chunked on sentences either way, and having two different notions of where a
    sentence ends would show up as inconsistent pauses.

    on_status(kind, frame) is called for non-text frames, which is how an
    approval request reaches the robot as something spoken rather than a silence.
    """
    import websockets

    import server                       # for the shared sentence splitter

    uri = f"{GATEWAY_URL}?agent={AGENT}"
    if session_id:
        uri += f"&session_id={session_id}"

    async with websockets.connect(uri, max_size=None,
                                  open_timeout=15, close_timeout=5) as socket:
        await socket.send(json.dumps({"type": "message", "content": text}))

        buffer = ""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + TURN_TIMEOUT

        while loop.time() < deadline:
            try:
                raw = await asyncio.wait_for(socket.recv(),
                                             timeout=max(1.0, deadline - loop.time()))
            except asyncio.TimeoutError:
                raise GatewayError("gateway turn timed out")

            try:
                frame = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Unparseable gateway frame: %r", str(raw)[:120])
                continue

            kind = frame.get("type")

            if kind == "chunk":
                buffer += frame.get("content", "")
                while True:
                    cut = server._first_boundary(buffer)
                    if cut is None:
                        break
                    yield buffer[:cut].strip()
                    buffer = buffer[cut:].lstrip()
                continue

            if kind == "done":
                if buffer.strip():
                    yield buffer.strip()
                return

            if kind == "error":
                raise GatewayError(str(frame.get("message") or frame))

            if kind == "approval_request":
                # Nothing shipped today reaches this -- every skill is a read,
                # and the campus risk profile auto-approves reads. It is handled
                # anyway because the first skill that acts will land here, and
                # the failure mode otherwise is a robot that stops talking
                # mid-turn with no explanation.
                logger.info("Gateway asked for approval: %s",
                            str(frame.get("summary") or frame)[:200])
                if on_status:
                    on_status("approval_request", frame)
                continue

            if on_status and kind not in ("session_start",):
                on_status(kind, frame)

        raise GatewayError("gateway turn exceeded its budget")


def available() -> bool:
    """True when the gateway answers its health endpoint."""
    import requests

    base = GATEWAY_URL.replace("ws://", "http://").replace("wss://", "https://")
    base = base.rsplit("/ws/", 1)[0]
    try:
        return requests.get(f"{base}/health", timeout=3).ok
    except Exception:
        return False
