"""
Reflection worker.

Reads what Lancer was actually asked and could not answer, groups the misses,
and proposes skills that would have answered them. It writes proposals. It does
not write skills, does not touch the running agent, and has no path to doing so
-- the skills volume is mounted read-only and nothing here shells out.

That restraint is the point. A loop that edits its own capabilities while a
robot is talking to students is not a demo, it is an incident. A human reads
the proposal, decides, and writes the skill.

Modelled on the WISS reflect-worker's shape (capture, distill, propose, gate)
at a size an ACM chapter can actually read in one sitting.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import psycopg
import requests

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("reflect")

DB_URL = os.environ["LANCER_DB_URL"]
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://host.docker.internal:11434")
MODEL = os.environ.get("REFLECT_MODEL", "qwen2.5:14b")
INTERVAL = int(os.environ.get("REFLECT_INTERVAL_SECS", "1800"))
MIN_TURNS = int(os.environ.get("REFLECT_MIN_TURNS", "15"))
PROPOSALS = Path("/proposals")
SKILLS = Path("/skills")

# A miss has to recur before it is worth a skill. One person asking one odd
# question is not a capability gap, it is a person.
MIN_CLUSTER = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS reflection_runs (
    id          BIGSERIAL PRIMARY KEY,
    ran_at      DOUBLE PRECISION NOT NULL,
    turns_seen  INTEGER NOT NULL,
    proposed    INTEGER NOT NULL,
    notes       TEXT
);
CREATE TABLE IF NOT EXISTS skill_proposals (
    id          BIGSERIAL PRIMARY KEY,
    created_at  DOUBLE PRECISION NOT NULL,
    name        TEXT NOT NULL,
    description TEXT,
    rationale   TEXT,
    examples    TEXT,
    data_source TEXT,
    status      TEXT NOT NULL DEFAULT 'pending',
    reviewed_at DOUBLE PRECISION,
    reviewer    TEXT
);
"""

PROMPT = """You are reviewing what a campus robot at California Baptist \
University was asked and could not answer.

Here are questions it failed on, grouped by topic:

{clusters}

The robot already has these skills: {existing}.

Propose AT MOST {limit} new skills that would have answered these questions. A \
good proposal is narrow, has an obvious data source, and covers more than one \
of the questions above.

Rules:
- Do not propose a skill for something the existing skills already cover.
- Do not propose anything requiring a data source you cannot name. "Ask the \
registrar for an API" is not a data source; "the public academic calendar page" is.
- Do not invent facts about CBU. You are proposing where to GET information, \
not what the information says.
- If none of these questions justify a new skill, return an empty list. That is \
a valid and common answer.

Reply with JSON only, in this shape:
{{"proposals": [{{"name": "snake_case_name", "description": "when to use it, \
one sentence", "rationale": "which questions it answers and why", \
"data_source": "where the data actually comes from"}}]}}"""


def connect() -> psycopg.Connection:
    for attempt in range(30):
        try:
            return psycopg.connect(DB_URL, autocommit=True)
        except Exception as e:
            logger.info("Waiting for postgres (%s/30): %s", attempt + 1, e)
            time.sleep(2)
    raise SystemExit("postgres never came up")


def existing_skills() -> list[str]:
    if not SKILLS.exists():
        return []
    return sorted(d.name for d in SKILLS.iterdir()
                  if d.is_dir() and (d / "skill.json").exists())


# Cheap topical grouping: the content words a question shares with others. A
# vector clustering step would be more principled, but the volume here is a few
# hundred turns a week and this is inspectable by whoever reads the proposal.
_STOP = set("""a an and are as at be by can could did do does for from get give
had has have how i in into is it its me my no not of on or should show tell
that the their them there these this to was were what when where which who why
will with would you your about does can""".split())


def _keywords(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in _STOP}


def cluster(questions: list[str]) -> list[list[str]]:
    """Group questions that share a distinctive keyword."""
    counts: Counter[str] = Counter()
    for question in questions:
        counts.update(_keywords(question))

    grouped: dict[str, list[str]] = {}
    for question in questions:
        words = _keywords(question)
        if not words:
            continue
        # The most frequent shared keyword is the group; a word only this
        # question uses is not a topic.
        best = max(words, key=lambda w: (counts[w], -len(w)))
        if counts[best] < MIN_CLUSTER:
            continue
        grouped.setdefault(best, []).append(question)

    return [qs for qs in grouped.values() if len(qs) >= MIN_CLUSTER]


def transcript_ready(conn: psycopg.Connection) -> bool:
    """
    True once the `turns` table exists.

    lancer_memory owns that schema and creates it on the server's first write,
    so on a fresh stack this worker starts before the table does. That is
    normal, not an error -- schema duplicated here to avoid one log line would
    be two definitions to keep in step forever.
    """
    row = conn.execute("SELECT to_regclass('public.turns')").fetchone()
    return bool(row and row[0])


def unanswered(conn: psycopg.Connection, since: float) -> list[str]:
    rows = conn.execute(
        "SELECT heard FROM turns WHERE ts > %s AND unknown = 1 "
        "ORDER BY ts DESC LIMIT 400", (since,)
    ).fetchall()
    return [r[0] for r in rows if r[0]]


def turns_seen(conn: psycopg.Connection, since: float) -> int:
    row = conn.execute("SELECT COUNT(*) FROM turns WHERE ts > %s",
                       (since,)).fetchone()
    return int(row[0]) if row else 0


def ask_model(clusters: list[list[str]], existing: list[str]) -> list[dict]:
    rendered = "\n\n".join(
        f"Topic {i + 1}:\n" + "\n".join(f"  - {q}" for q in group[:8])
        for i, group in enumerate(clusters)
    )
    prompt = PROMPT.format(clusters=rendered,
                           existing=", ".join(existing) or "none",
                           limit=min(3, len(clusters)))
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.2},
            },
            timeout=180,
        )
        response.raise_for_status()
        content = response.json()["message"]["content"]
        return json.loads(content).get("proposals", []) or []
    except Exception as e:
        logger.warning("Reflection model call failed: %s", e)
        return []


def store(conn: psycopg.Connection, proposal: dict, examples: list[str]) -> bool:
    """Record a proposal unless an open one already says the same thing."""
    name = str(proposal.get("name") or "").strip()
    if not name:
        return False
    already = conn.execute(
        "SELECT 1 FROM skill_proposals WHERE name = %s AND status = 'pending'",
        (name,)
    ).fetchone()
    if already:
        logger.info("Proposal %s already pending; skipping", name)
        return False

    now = time.time()
    conn.execute(
        "INSERT INTO skill_proposals (created_at, name, description, rationale, "
        "examples, data_source) VALUES (%s, %s, %s, %s, %s, %s)",
        (now, name, proposal.get("description", ""), proposal.get("rationale", ""),
         json.dumps(examples[:8]), proposal.get("data_source", "")),
    )

    PROPOSALS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    (PROPOSALS / f"{stamp}-{name}.md").write_text(
        f"# Proposed skill: {name}\n\n"
        f"_Proposed {datetime.now(timezone.utc).isoformat()} by the reflection "
        f"worker. Nothing has been created. A human writes the skill, or closes "
        f"this._\n\n"
        f"## What it would do\n\n{proposal.get('description', '')}\n\n"
        f"## Why\n\n{proposal.get('rationale', '')}\n\n"
        f"## Where the data comes from\n\n{proposal.get('data_source', '')}\n\n"
        f"## Questions it would have answered\n\n"
        + "\n".join(f"- {q}" for q in examples[:8]) + "\n"
    )
    logger.info("Proposed skill %s", name)
    return True


def run_once(conn: psycopg.Connection) -> None:
    if not transcript_ready(conn):
        logger.info("No transcript table yet -- waiting for Lancer's first turn")
        return

    since = time.time() - 7 * 24 * 3600
    seen = turns_seen(conn, since)
    if seen < MIN_TURNS:
        logger.info("Only %d turns this week; need %d before reflecting",
                    seen, MIN_TURNS)
        conn.execute(
            "INSERT INTO reflection_runs (ran_at, turns_seen, proposed, notes) "
            "VALUES (%s, %s, 0, %s)", (time.time(), seen, "below turn threshold")
        )
        return

    questions = unanswered(conn, since)
    clusters = cluster(questions)
    logger.info("%d turns, %d unanswered, %d clusters",
                seen, len(questions), len(clusters))

    proposed = 0
    if clusters:
        flat = [q for group in clusters for q in group]
        for proposal in ask_model(clusters, existing_skills()):
            if store(conn, proposal, flat):
                proposed += 1

    conn.execute(
        "INSERT INTO reflection_runs (ran_at, turns_seen, proposed, notes) "
        "VALUES (%s, %s, %s, %s)",
        (time.time(), seen, proposed, f"{len(clusters)} clusters"),
    )


def main() -> None:
    conn = connect()
    for statement in SCHEMA.strip().split(";"):
        if statement.strip():
            conn.execute(statement)
    logger.info("Reflection worker up; every %ss, model %s", INTERVAL, MODEL)
    while True:
        try:
            run_once(conn)
        except Exception as e:
            logger.exception("Reflection cycle failed: %s", e)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
