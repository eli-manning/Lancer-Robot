"""
Retrieval as a skill.

The same hybrid retrieval the deterministic path uses (dense + BM25 fused with
RRF, then a cross-encoder rerank), exposed so the model can choose to look
something up instead of having retrieval forced in front of every turn. That is
the Phase 2 shape: identical ranking, different caller.

It reaches the running server over HTTP rather than importing it. Importing
server.py from a subprocess would load Chroma, the reranker and Whisper a
second time, which is both slow and a second copy of the model in memory. The
endpoint is already there for the dashboard's retrieval inspector.
"""

import json
import os
import sys

import requests

SERVER = os.environ.get("LANCER_SERVER_SELF", "http://127.0.0.1:7870")
MAX_CHUNKS = 4
MAX_CHUNK_CHARS = 900


def main() -> int:
    try:
        args = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        args = {}

    question = str(args.get("question") or "").strip()
    if not question:
        json.dump({"ok": False, "error": "no question given"}, sys.stdout)
        return 0

    try:
        response = requests.post(
            f"{SERVER}/debug/retrieve", json={"query": question}, timeout=20
        )
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        json.dump({"ok": False, "error": f"retrieval unavailable: {e}"}, sys.stdout)
        return 0

    if not data.get("relevant"):
        json.dump({
            "ok": True,
            "found": False,
            "question": question,
            "note": "Nothing in CBU's documents covers this. Say you do not have "
                    "it -- do not answer from memory.",
        }, sys.stdout)
        return 0

    chunks = [c[:MAX_CHUNK_CHARS] for c in (data.get("chunks") or [])[:MAX_CHUNKS]]
    json.dump({
        "ok": True,
        "found": True,
        "question": question,
        "sources": data.get("sources", [])[:MAX_CHUNKS],
        "passages": chunks,
        "note": "Every factual claim in your answer must appear in these passages.",
    }, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
