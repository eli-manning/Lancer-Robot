"""
Durable memory for Lancer.

Two things live here that a JSONL transcript cannot do: a queryable record of
every turn, and facts about people that survive a restart -- so a student who
said their name and major last week is not a stranger on their next visit.

Backend is Postgres when one is reachable (the container the compose stack
brings up, matching what ZeroClaw itself uses for memory) and SQLite otherwise.
The fallback is not a nicety: Lancer has to work on a desk with Docker stopped,
and a memory layer that takes the robot down when a container is missing is
worse than no memory layer. Same schema either way, so moving between them is
a copy rather than a migration.

The JSONL transcript keeps being written by lancer_runtime.record(); this sits
alongside it rather than replacing it, so the dashboard's existing history view
is untouched.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

BASE = Path(__file__).parent
SQLITE_PATH = BASE / "lancer_memory.db"
# Set by the compose stack; unset on a bare laptop, which is the common case.
POSTGRES_URL = os.environ.get("LANCER_DB_URL", "")

_lock = threading.Lock()
_conn: Any = None
_backend = "none"

# Postgres uses %s placeholders, SQLite uses ?. One flag, translated at the
# single point where SQL is executed, keeps every query below written once.
_PLACEHOLDER = "?"

SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS sessions (
        id          TEXT PRIMARY KEY,
        started_at  DOUBLE PRECISION NOT NULL,
        last_seen   DOUBLE PRECISION NOT NULL,
        speaker     TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS turns (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        ts          DOUBLE PRECISION NOT NULL,
        session_id  TEXT,
        heard       TEXT,
        answer      TEXT,
        sources     TEXT,
        skills      TEXT,
        unknown     INTEGER DEFAULT 0,
        model       TEXT,
        latency_ms  INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS facts (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        subject     TEXT NOT NULL,
        key         TEXT NOT NULL,
        value       TEXT NOT NULL,
        first_seen  DOUBLE PRECISION NOT NULL,
        last_seen   DOUBLE PRECISION NOT NULL,
        mentions    INTEGER DEFAULT 1
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_turns_ts ON turns(ts)",
    "CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject)",
]

# Postgres has no AUTOINCREMENT and wants a serial type instead.
_PG_SCHEMA_FIXUP = [
    ("INTEGER PRIMARY KEY AUTOINCREMENT", "BIGSERIAL PRIMARY KEY"),
]


def _connect() -> tuple[Any, str]:
    """Open Postgres if configured and reachable, else SQLite."""
    if POSTGRES_URL:
        try:
            import psycopg

            conn = psycopg.connect(POSTGRES_URL, connect_timeout=5, autocommit=True)
            logger.info("Memory backend: postgres")
            return conn, "postgres"
        except Exception as e:
            logger.warning("Postgres unavailable (%s); using SQLite", e)

    conn = sqlite3.connect(str(SQLITE_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    logger.info("Memory backend: sqlite at %s", SQLITE_PATH)
    return conn, "sqlite"


def _sql(query: str) -> str:
    """Translate the ?-style queries below for whichever driver is connected."""
    if _backend == "postgres":
        return query.replace("?", "%s")
    return query


def init() -> str:
    """Connect and create the schema. Safe to call repeatedly."""
    global _conn, _backend
    with _lock:
        if _conn is not None:
            return _backend
        _conn, _backend = _connect()
        for statement in SCHEMA:
            sql = statement
            if _backend == "postgres":
                for old, new in _PG_SCHEMA_FIXUP:
                    sql = sql.replace(old, new)
            try:
                _conn.execute(sql) if _backend == "sqlite" else _conn.cursor().execute(sql)
            except Exception as e:
                logger.warning("Schema statement failed: %s", e)
        if _backend == "sqlite":
            _conn.commit()
        return _backend


def _execute(query: str, params: tuple = ()) -> list[tuple]:
    """Run one statement, returning rows for selects and [] otherwise."""
    if _conn is None:
        init()
    with _lock:
        try:
            if _backend == "sqlite":
                cursor = _conn.execute(_sql(query), params)
                rows = cursor.fetchall() if cursor.description else []
                _conn.commit()
                return rows
            cursor = _conn.cursor()
            cursor.execute(_sql(query), params)
            rows = cursor.fetchall() if cursor.description else []
            return rows
        except Exception as e:
            logger.warning("Memory query failed: %s", e)
            return []


def backend() -> str:
    return _backend if _conn is not None else init()


# --- Sessions ------------------------------------------------------------
# A "session" is one stretch of conversation. Someone who walks away and comes
# back an hour later is a new session, which is what makes "last time we spoke"
# mean anything.

SESSION_GAP = 15 * 60      # seconds of silence that end a session

_session_id: str = ""
_session_touched: float = 0.0


def session() -> str:
    """Return the current session id, starting a new one after a long gap."""
    global _session_id, _session_touched
    now = time.time()
    if not _session_id or now - _session_touched > SESSION_GAP:
        _session_id = f"s{int(now)}"
        _execute(
            "INSERT INTO sessions (id, started_at, last_seen) VALUES (?, ?, ?)",
            (_session_id, now, now),
        )
        logger.info("New conversation session %s", _session_id)
    else:
        _execute("UPDATE sessions SET last_seen = ? WHERE id = ?", (now, _session_id))
    _session_touched = now
    return _session_id


def record_turn(
    heard: str,
    answer: str,
    sources: list[str] | None = None,
    skills: list[dict] | None = None,
    unknown: bool = False,
    model: str = "",
    latency_ms: int = 0,
) -> None:
    """Store one exchange."""
    _execute(
        "INSERT INTO turns (ts, session_id, heard, answer, sources, skills, "
        "unknown, model, latency_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (time.time(), session(), heard, answer,
         json.dumps(sources or []),
         json.dumps([{"skill": c.get("skill"), "arguments": c.get("arguments")}
                     for c in (skills or [])]),
         1 if unknown else 0, model, int(latency_ms)),
    )


def recent_turns(limit: int = 20) -> list[dict]:
    rows = _execute(
        "SELECT ts, session_id, heard, answer, sources, skills, unknown, model, "
        "latency_ms FROM turns ORDER BY ts DESC LIMIT ?", (limit,)
    )
    out = []
    for r in rows:
        out.append({
            "ts": r[0], "session_id": r[1], "heard": r[2], "answer": r[3],
            "sources": json.loads(r[4] or "[]"), "skills": json.loads(r[5] or "[]"),
            "unknown": bool(r[6]), "model": r[7], "latency_ms": r[8],
        })
    return out


# --- Facts ---------------------------------------------------------------
# Deliberately regex-driven rather than model-driven. An LLM asked to "extract
# facts" from a noisy Whisper transcript will invent a major and a graduation
# year, and a robot that confidently misremembers a student is worse than one
# that remembers nothing. These patterns only fire on someone stating something
# about themselves in plain terms.

_I_AM = r"(?:i'?m|i am)"

_FACT_PATTERNS: list[tuple[str, str]] = [
    (r"\bmy name is ([A-Z][a-z]+(?: [A-Z][a-z]+)?)", "name"),
    (rf"\b{_I_AM} ([A-Z][a-z]+)(?:[,.]|\s+and\b|\s*$)", "name"),
    (r"\bcall me ([A-Z][a-z]+)", "name"),
    (rf"\b{_I_AM} (?:a |an )?(freshman|sophomore|junior|senior|grad student|graduate student)\b", "year"),
    (r"\b(?:majoring in|major(?:ing)? in|studying) ([a-z ]{3,40}?)(?:[,.]|\s+and\b|$)", "major"),
    (r"\bmy major is ([a-z ]{3,40}?)(?:[,.]|\s+and\b|$)", "major"),
    (r"\b(?:i'?m|i am) in the ([A-Z]{4,6}) program\b", "program"),
    (r"\bi transferred from ([A-Za-z ]{3,40}?)(?:[,.]|\s+and\b|$)", "transferred_from"),
]

# Whisper mishears constantly, and "I'm good" must not become a student named
# Good. Anything on this list is never stored as a name.
_NAME_STOPLIST = {
    "good", "fine", "okay", "ok", "sorry", "here", "back", "done", "sure",
    "not", "just", "really", "trying", "looking", "asking", "wondering",
    "confused", "lost", "curious", "interested", "thinking", "hungry", "tired",
}


def learn_from(text: str, subject: str = "student") -> list[dict]:
    """Pull durable facts out of something a person said. Returns what was stored."""
    learned: list[dict] = []
    for pattern, key in _FACT_PATTERNS:
        match = re.search(pattern, text, re.I if key != "name" else 0)
        if not match:
            continue
        value = match.group(1).strip().rstrip(".,!?")
        if not value:
            continue
        if key == "name" and value.lower() in _NAME_STOPLIST:
            continue
        remember(subject, key, value)
        learned.append({"key": key, "value": value})
    if learned:
        logger.info("Learned %s", learned)
    return learned


def remember(subject: str, key: str, value: str) -> None:
    """Store or refresh one fact."""
    now = time.time()
    existing = _execute(
        "SELECT id, value, mentions FROM facts WHERE subject = ? AND key = ? "
        "ORDER BY last_seen DESC LIMIT 1", (subject, key)
    )
    if existing and str(existing[0][1]).lower() == value.lower():
        _execute("UPDATE facts SET last_seen = ?, mentions = ? WHERE id = ?",
                 (now, int(existing[0][2] or 1) + 1, existing[0][0]))
        return
    _execute(
        "INSERT INTO facts (subject, key, value, first_seen, last_seen, mentions) "
        "VALUES (?, ?, ?, ?, ?, 1)", (subject, key, value, now, now)
    )


def recall(subject: str = "student", limit: int = 8) -> list[dict]:
    """
    Most recently confirmed facts about someone, one per key.

    People correct themselves ("computer science -- well, applied AI now"), and
    both statements are stored so the history is intact. Only the newest of each
    key is recalled, or Lancer would be told two different majors and pick one
    at random.
    """
    rows = _execute(
        "SELECT key, value, last_seen, mentions FROM facts WHERE subject = ? "
        "ORDER BY last_seen DESC", (subject,)
    )
    seen: set[str] = set()
    out: list[dict] = []
    for key, value, last_seen, mentions in rows:
        if key in seen:
            continue
        seen.add(key)
        out.append({"key": key, "value": value,
                    "last_seen": last_seen, "mentions": mentions})
        if len(out) >= limit:
            break
    return out


def recall_prompt(subject: str = "student") -> str:
    """
    Facts formatted for a system message, or "" when there is nothing to say.

    Kept short on purpose: a wall of remembered trivia in the prompt makes the
    model bring it up unprompted, which is unsettling rather than helpful.
    """
    facts = recall(subject, limit=5)
    if not facts:
        return ""
    parts = [f"{f['key'].replace('_', ' ')}: {f['value']}" for f in facts]
    return (
        "You have talked to this person before. What you know about them: "
        + "; ".join(parts)
        + ". Use it only if it is relevant -- greet them by name if you have "
        "one, but do not recite the rest back at them."
    )


def forget(subject: str = "student") -> int:
    """Drop everything remembered about someone. The dashboard exposes this."""
    rows = _execute("SELECT COUNT(*) FROM facts WHERE subject = ?", (subject,))
    count = int(rows[0][0]) if rows else 0
    _execute("DELETE FROM facts WHERE subject = ?", (subject,))
    return count


def stats() -> dict:
    turns = _execute("SELECT COUNT(*) FROM turns")
    facts = _execute("SELECT COUNT(*) FROM facts")
    sessions = _execute("SELECT COUNT(*) FROM sessions")
    return {
        "backend": backend(),
        "turns": int(turns[0][0]) if turns else 0,
        "facts": int(facts[0][0]) if facts else 0,
        "sessions": int(sessions[0][0]) if sessions else 0,
    }
