"""
Skills: the things Lancer can actually do, as opposed to things it knows.

A skill is a directory under skills/ holding a manifest (skill.json) and an
executable (run.py). The manifest carries the JSON-schema the model sees, so
adding a capability means adding a directory -- no edits here, and no server
restart beyond a reload.

This is deliberately the same shape as a ZeroClaw shell skill: a directory, a
description the model reads, a script that takes JSON on stdin and writes JSON
to stdout. When the gateway takes over the conversation the same directories
mount straight in as a skill bundle, so nothing written here is throwaway.

Risk tiers mirror the gateway's too. "read" skills only look things up and are
auto-approved; "act" skills change something in the world and are refused
unless the caller passes approve=True. Nothing shipped today is an "act" skill,
but a robot in a hallway will eventually want one, and the gate should exist
before the first one does rather than after.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SKILLS_DIR = Path(__file__).parent / "skills"
SKILL_TIMEOUT = 12          # seconds; a slow API must not stall a spoken reply
MAX_RESULT_CHARS = 4000     # what comes back gets read by a model, not a human

# Tool loops go wrong in two directions: the model calls nothing and answers
# from imagination, or it calls the same skill forever. Cap the iterations.
MAX_TOOL_ITERATIONS = 4

# Risk tiers that run without asking.
#
#   read   -- looks something up and changes nothing
#   motion -- moves the robot on its own desk: a nod, a look, an emotion. It is
#             physical, so it is not "read", but nothing it does persists or
#             reaches anyone else, and requiring approval to nod would make the
#             gesture arrive long after the sentence it belonged to.
#
# Anything that acts on the world -- writes a file, sends a message, calls an
# API that changes something -- gets no tier here and must be approved. Nothing
# shipped today is in that category; the gate exists so the first skill that is
# does not quietly inherit a nod's permissions.
AUTO_APPROVED_RISKS = {"read", "motion"}


class Skill:
    """One loaded skill directory."""

    def __init__(self, path: Path, manifest: dict[str, Any]):
        self.path = path
        self.name: str = manifest["name"]
        self.description: str = manifest.get("description", "")
        self.parameters: dict = manifest.get("parameters") or {
            "type": "object", "properties": {},
        }
        self.risk: str = manifest.get("risk", "read")
        self.entrypoint: str = manifest.get("entrypoint", "run.py")
        self.enabled: bool = bool(manifest.get("enabled", True))
        # Cheap skills (a clock read) should not be announced; slow ones
        # (a network call) let the robot say something while it waits.
        self.spoken_wait: str | None = manifest.get("spoken_wait")

    @property
    def tool_schema(self) -> dict:
        """The OpenAI-style function schema handed to the model."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def run(self, arguments: dict[str, Any], approve: bool = False) -> dict:
        """
        Execute the skill and return its JSON result.

        Errors are returned rather than raised: the model is the caller here,
        and it handles "that lookup failed" far better than the turn dying.
        """
        if self.risk not in AUTO_APPROVED_RISKS and not approve:
            return {"ok": False, "error": "approval_required", "risk": self.risk}

        script = self.path / self.entrypoint
        if not script.exists():
            return {"ok": False, "error": f"missing entrypoint {self.entrypoint}"}

        started = time.monotonic()
        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                input=json.dumps(arguments or {}),
                capture_output=True,
                text=True,
                timeout=SKILL_TIMEOUT,
                cwd=str(self.path),
            )
        except subprocess.TimeoutExpired:
            logger.warning("Skill %s timed out after %ss", self.name, SKILL_TIMEOUT)
            return {"ok": False, "error": "timeout"}

        elapsed = time.monotonic() - started
        if proc.returncode != 0:
            logger.warning("Skill %s exited %s: %s",
                           self.name, proc.returncode, proc.stderr[:300])
            return {"ok": False, "error": (proc.stderr or "failed").strip()[:300]}

        raw = (proc.stdout or "").strip()
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            # A skill that prints prose instead of JSON is still useful; wrap it
            # rather than failing the whole turn over a formatting slip.
            result = {"ok": True, "text": raw[:MAX_RESULT_CHARS]}

        logger.info("Skill %s ran in %.2fs", self.name, elapsed)
        return result


_registry: dict[str, Skill] = {}
_loaded_at: float = 0.0


def load_skills(force: bool = False) -> dict[str, Skill]:
    """Load (or reload) every skill directory. Bad manifests are skipped loudly."""
    global _registry, _loaded_at
    if _registry and not force:
        return _registry

    found: dict[str, Skill] = {}
    if SKILLS_DIR.exists():
        for directory in sorted(SKILLS_DIR.iterdir()):
            manifest_path = directory / "skill.json"
            if not directory.is_dir() or not manifest_path.exists():
                continue
            try:
                manifest = json.loads(manifest_path.read_text())
                skill = Skill(directory, manifest)
            except Exception as e:
                logger.warning("Skipping skill %s: %s", directory.name, e)
                continue
            if skill.enabled:
                found[skill.name] = skill

    _registry = found
    _loaded_at = time.time()
    logger.info("Loaded %d skills: %s", len(found), ", ".join(sorted(found)) or "none")
    return _registry


def tool_schemas(names: list[str] | None = None) -> list[dict]:
    """Schemas for the model. Passing names limits it to a subset."""
    skills = load_skills()
    chosen = [s for n, s in skills.items() if names is None or n in names]
    return [s.tool_schema for s in chosen]


def run_skill(name: str, arguments: dict, approve: bool = False) -> dict:
    skills = load_skills()
    skill = skills.get(name)
    if skill is None:
        return {"ok": False, "error": f"no such skill: {name}"}
    return skill.run(arguments, approve=approve)


def describe() -> list[dict]:
    """Human-readable listing, for the dashboard."""
    return [
        {"name": s.name, "description": s.description, "risk": s.risk,
         "path": str(s.path), "parameters": s.parameters}
        for s in sorted(load_skills().values(), key=lambda s: s.name)
    ]


# --- The tool loop -------------------------------------------------------
# Ollama speaks the OpenAI tool-calling protocol, so this is the standard
# shape: offer the schemas, run whatever comes back, feed results in as `tool`
# messages, repeat until the model answers in prose instead of calling.


def gather(
    model: str,
    messages: list[dict],
    options: dict | None = None,
    allow: list[str] | None = None,
    approve: bool = False,
    on_call=None,
    provider: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Run only the tool phase, and hand back the messages the answer builds on.

    Splitting this out is what lets the spoken answer stream. The tool phase
    cannot stream -- a half-parsed tool call is not something you can act on --
    so it runs to completion here, and the caller then streams the final answer
    with the results already in context and the tools withdrawn. The model gets
    live data; the listener still hears the first sentence in under a second.

    Returns (messages, calls). When the model asks for no tools, messages comes
    back unchanged and the caller streams exactly as it would have anyway.
    """
    import lancer_llm as llm

    schemas = tool_schemas(allow)
    if not schemas:
        return list(messages), []

    if provider is None:
        import lancer_runtime as rt

        provider = str(rt.get("llm_provider") or llm.OLLAMA)
    # Sending schemas to a provider that cannot act on them costs a round trip
    # and returns nothing, so skip straight to the answer. The rule-driven
    # skills the caller already ran are unaffected.
    if not llm.supports_tool_calls(provider):
        return list(messages), []

    calls: list[dict] = []
    working = list(messages)

    for _ in range(MAX_TOOL_ITERATIONS):
        try:
            message = llm.chat_with_tools(working, model, provider, schemas,
                                          options or {})
        except Exception as e:
            # A model without tool support, or a malformed call it could not
            # recover from. Answering without live data beats failing the turn.
            logger.warning("Tool phase failed (%s); continuing without skills", e)
            return list(messages), calls

        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            # It answered instead of calling. Drop that draft: the caller is
            # about to generate the real answer, and keeping it would make the
            # model repeat itself.
            return working, calls

        working.append(message)
        for call in tool_calls:
            function = call.get("function", {})
            name = function.get("name", "")
            arguments = function.get("arguments") or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            if on_call:
                try:
                    on_call(name, arguments)
                except Exception:
                    pass
            result = run_skill(name, arguments, approve=approve)
            calls.append({"skill": name, "arguments": arguments, "result": result})
            logger.info("Tool call %s(%s) -> ok=%s", name, arguments, result.get("ok"))
            working.append({
                "role": "tool",
                "name": name,
                "content": json.dumps(result)[:MAX_RESULT_CHARS],
            })

    return working, calls


def chat_with_skills(
    model: str,
    messages: list[dict],
    options: dict | None = None,
    allow: list[str] | None = None,
    approve: bool = False,
    on_call=None,
) -> tuple[str, list[dict]]:
    """
    Run a chat turn in which the model may call skills.

    Returns (answer_text, calls) where calls records what ran, so the caller can
    log it, show it on the dashboard, and -- importantly for a talking robot --
    know whether the answer came from a live source or from the model's memory.

    on_call(skill_name, arguments) is invoked before each call, which is how the
    voice path gets to say "let me check that" instead of going silent for two
    seconds.
    """
    import ollama

    schemas = tool_schemas(allow)
    calls: list[dict] = []
    working = list(messages)

    for _ in range(MAX_TOOL_ITERATIONS):
        response = ollama.chat(
            model=model,
            messages=working,
            tools=schemas or None,
            options=options or {},
        )
        message = response["message"]
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            return (message.get("content") or "").strip(), calls

        # Keep the assistant's tool-call turn in history; without it the model
        # sees tool results answering a question it never asked.
        working.append(message)
        for call in tool_calls:
            function = call.get("function", {})
            name = function.get("name", "")
            arguments = function.get("arguments") or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            if on_call:
                try:
                    on_call(name, arguments)
                except Exception:
                    pass

            result = run_skill(name, arguments, approve=approve)
            calls.append({"skill": name, "arguments": arguments, "result": result})
            logger.info("Tool call %s(%s) -> ok=%s",
                        name, arguments, result.get("ok"))
            working.append({
                "role": "tool",
                "name": name,
                "content": json.dumps(result)[:MAX_RESULT_CHARS],
            })

    # Out of iterations: ask once more with tools withdrawn so it has to answer.
    response = ollama.chat(model=model, messages=working, options=options or {})
    return (response["message"].get("content") or "").strip(), calls
