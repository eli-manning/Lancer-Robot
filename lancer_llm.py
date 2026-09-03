"""
Where Lancer's words come from.

Three providers behind one interface: the local Ollama that has answered every
question so far, and hosted Anthropic and Gemini for when a bigger model is
worth the round trip. The pipeline does not know which is answering -- it asks
for a stream of text and gets one.

The trade is honest and worth stating. Local costs nothing, works with the wifi
down, and keeps every student question on this machine; a 14B on an M3 Max
answers in seconds. Hosted is faster and markedly smarter, and sends the
conversation to someone else's servers. For a robot in a hallway that mostly
answers from CBU's own documents, that is a real decision rather than an
obvious one, so it is a setting rather than a rewrite.

Keys live in lancer_secrets.json (untracked) or the environment. Never in
lancer_config.json -- that file is committed, and the dashboard writes to it.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

BASE = Path(__file__).parent
SECRETS_PATH = BASE / "lancer_secrets.json"

OLLAMA = "ollama"
ANTHROPIC = "anthropic"
GEMINI = "gemini"
PROVIDERS = (OLLAMA, ANTHROPIC, GEMINI)

# Sensible starting points. The dashboard offers these and accepts anything.
SUGGESTED_MODELS = {
    OLLAMA: ["qwen2.5:14b", "qwen2.5:32b", "qwen2.5:3b", "mistral:latest"],
    ANTHROPIC: ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"],
    GEMINI: ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
}

# Short answers said out loud. Not a cost measure: the sentence limit downstream
# throws away anything past a few sentences anyway, so generating more is time
# somebody spends waiting for words that get discarded.
MAX_OUTPUT_TOKENS = 1024

ENV_KEYS = {
    ANTHROPIC: "ANTHROPIC_API_KEY",
    GEMINI: "GEMINI_API_KEY",
}


# --- Credentials ---------------------------------------------------------

def load_secrets() -> dict:
    if not SECRETS_PATH.exists():
        return {}
    try:
        return json.loads(SECRETS_PATH.read_text())
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not read %s: %s", SECRETS_PATH.name, e)
        return {}


def save_key(provider: str, key: str) -> None:
    """Store an API key outside the tracked config, readable only by this user."""
    secrets = load_secrets()
    if key:
        secrets[f"{provider}_api_key"] = key
    else:
        secrets.pop(f"{provider}_api_key", None)
    SECRETS_PATH.write_text(json.dumps(secrets, indent=2))
    SECRETS_PATH.chmod(0o600)
    logger.info("%s API key %s", provider, "saved" if key else "cleared")


def api_key(provider: str) -> str:
    """The key for a provider, from the secrets file or the environment."""
    return (load_secrets().get(f"{provider}_api_key")
            or os.environ.get(ENV_KEYS.get(provider, ""), "")
            or "")


def configured(provider: str) -> bool:
    """Whether this provider could answer right now."""
    if provider == OLLAMA:
        return True
    return bool(api_key(provider))


def status() -> list[dict]:
    """What the dashboard shows: who is available, and whether a key is set."""
    return [
        {
            "provider": name,
            "configured": configured(name),
            "needs_key": name != OLLAMA,
            "key_set": bool(api_key(name)) if name != OLLAMA else True,
            # Model-chosen tool calls, not "does this vendor's API support
            # them". Both tool loops in lancer_skills call ollama.chat
            # directly, so a hosted model never gets offered the schemas --
            # Gemini and Anthropic alike. Claiming otherwise here sent people
            # looking for a Gemini limitation that is ours.
            "supports_tool_calls": supports_tool_calls(name),
            "models": SUGGESTED_MODELS[name],
        }
        for name in PROVIDERS
    ]


# --- Message conversion --------------------------------------------------
# Everything upstream speaks the OpenAI-ish shape Ollama uses: a flat list of
# {role, content} with system messages anywhere. The other two want something
# different, and the conversion is where the bugs would live, so it is one
# function each with the rules written down.

def _split_system(messages: list[dict]) -> tuple[str, list[dict]]:
    """Pull system messages out into one block, keeping the rest in order."""
    system_parts: list[str] = []
    rest: list[dict] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content") or ""
        if role == "system":
            if content:
                system_parts.append(str(content))
        elif role == "tool":
            # A tool result with nowhere to go: providers that are not running
            # the tool loop get it as context rather than losing it.
            rest.append({"role": "user",
                         "content": f"[result from {message.get('name', 'a tool')}]"
                                    f" {content}"})
        else:
            rest.append({"role": role or "user", "content": str(content)})
    return "\n\n".join(system_parts), rest


def _alternating(messages: list[dict]) -> list[dict]:
    """
    Force strict user/assistant alternation, starting with user.

    Anthropic accepts consecutive same-role messages by merging them, but a
    leading assistant turn is an error -- and our history can start with one
    after a truncation. Merging here keeps that from ever reaching the wire.
    """
    out: list[dict] = []
    for message in messages:
        role = "assistant" if message["role"] == "assistant" else "user"
        text = message["content"]
        if not text:
            continue
        if out and out[-1]["role"] == role:
            out[-1]["content"] = f"{out[-1]['content']}\n\n{text}"
        else:
            out.append({"role": role, "content": text})
    while out and out[0]["role"] == "assistant":
        out.pop(0)
    return out


# --- Gemini's conversation shape -----------------------------------------
# Gemini does not speak the OpenAI tool protocol the rest of this app uses, so
# the whole conversation is converted on every request: a tool call is a
# functionCall part on a "model" turn, and its result is a functionResponse part
# on a "user" turn. Getting this wrong does not error -- the model simply
# behaves as though it never called anything -- so it is one function with the
# mapping written down.

# Keys Gemini accepts in a parameter schema. Anything else (additionalProperties,
# $schema, default) is rejected outright, and our manifests are free to grow
# them, so the schema is filtered rather than trusted.
_SCHEMA_KEYS = {"type", "description", "enum", "properties", "required",
                "items", "nullable", "format"}


def _clean_schema(schema: dict) -> dict:
    out = {}
    for key, value in (schema or {}).items():
        if key not in _SCHEMA_KEYS:
            continue
        if key == "properties" and isinstance(value, dict):
            out[key] = {k: _clean_schema(v) for k, v in value.items()}
        elif key == "items" and isinstance(value, dict):
            out[key] = _clean_schema(value)
        else:
            out[key] = value
    return out


def _gemini_tools(schemas: list[dict]) -> list[dict]:
    """OpenAI-style function schemas as Gemini functionDeclarations."""
    declarations = []
    for schema in schemas or []:
        function = schema.get("function", schema)
        name = function.get("name")
        if not name:
            continue
        declaration = {"name": name,
                       "description": function.get("description", "")}
        parameters = _clean_schema(function.get("parameters") or {})
        # A declaration with an empty parameter object is rejected; a skill that
        # takes no arguments simply has no parameters field.
        if parameters.get("properties"):
            declaration["parameters"] = parameters
        declarations.append(declaration)
    return [{"functionDeclarations": declarations}] if declarations else []


def _gemini_contents(messages: list[dict]) -> tuple[str, list[dict]]:
    """Convert our message list into (systemInstruction, contents)."""
    system_parts: list[str] = []
    contents: list[dict] = []

    def add(role: str, part: dict) -> None:
        # Consecutive turns of the same role are merged rather than sent
        # separately: a model turn that calls two skills is one turn with two
        # parts, and the paired results have to line up with it.
        if contents and contents[-1]["role"] == role:
            contents[-1]["parts"].append(part)
        else:
            contents.append({"role": role, "parts": [part]})

    for message in messages:
        role = message.get("role")
        content = message.get("content") or ""
        if role == "system":
            if content:
                system_parts.append(str(content))
            continue
        if role == "tool":
            body = content
            if isinstance(body, str):
                try:
                    body = json.loads(body)
                except json.JSONDecodeError:
                    body = {"result": body}
            if not isinstance(body, dict):
                body = {"result": body}
            add("user", {"functionResponse": {
                "name": message.get("name") or "tool", "response": body}})
            continue
        if role == "assistant":
            calls = message.get("tool_calls") or []
            for call in calls:
                function = call.get("function", {}) if isinstance(call, dict) else {}
                arguments = function.get("arguments") or {}
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                add("model", {"functionCall": {"name": function.get("name", ""),
                                               "args": arguments}})
            if content:
                add("model", {"text": str(content)})
            continue
        if content:
            add("user", {"text": str(content)})

    # A conversation may not open on a model turn.
    while contents and contents[0]["role"] == "model":
        contents.pop(0)
    return "\n\n".join(system_parts), contents


def _gemini_body(messages: list[dict], options: dict,
                 tools: list[dict] | None = None) -> dict:
    system, contents = _gemini_contents(messages)
    body: dict = {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": MAX_OUTPUT_TOKENS,
            "temperature": float(options.get("temperature", 0.4)),
        },
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    declarations = _gemini_tools(tools or [])
    if declarations:
        body["tools"] = declarations
    return body


def _gemini_url(model: str, method: str) -> str:
    name = model or SUGGESTED_MODELS[GEMINI][0]
    return (f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{name}:{method}")


# --- Tool calling --------------------------------------------------------

def chat_with_tools(messages: list[dict], model: str, provider: str,
                    tools: list[dict], options: dict | None = None) -> dict:
    """
    One non-streaming turn in which the model may call skills.

    Returns a message in the shape the rest of the app already uses -- Ollama's
    -- whoever produced it: {"role", "content", "tool_calls": [{"function":
    {"name", "arguments"}}]}. The tool loop then does not care who answered.
    """
    options = options or {}
    if provider == GEMINI:
        return _gemini_chat(messages, model, tools, options)
    import ollama

    response = ollama.chat(model=model, messages=messages,
                           tools=tools or None, options=options)
    return response["message"]


def supports_tool_calls(provider: str) -> bool:
    """Providers that can be handed skill schemas and choose among them."""
    return provider in (OLLAMA, GEMINI)


def _gemini_chat(messages: list[dict], model: str, tools: list[dict],
                 options: dict) -> dict:
    import requests

    key = api_key(GEMINI)
    if not key:
        raise RuntimeError("no Gemini API key configured")

    response = requests.post(
        _gemini_url(model, "generateContent"),
        json=_gemini_body(messages, options, tools), timeout=60,
        headers={"x-goog-api-key": key})
    if not response.ok:
        raise RuntimeError(
            f"Gemini returned {response.status_code}: {response.text[:300]}")

    candidates = response.json().get("candidates") or []
    parts = (candidates[0].get("content", {}).get("parts", [])
             if candidates else [])
    text = "".join(p["text"] for p in parts if p.get("text"))
    calls = [{"function": {"name": p["functionCall"].get("name", ""),
                           "arguments": p["functionCall"].get("args") or {}}}
             for p in parts if p.get("functionCall")]
    return {"role": "assistant", "content": text, "tool_calls": calls}


# --- Streaming -----------------------------------------------------------

def stream_text(messages: list[dict], model: str, provider: str = OLLAMA,
                options: dict | None = None) -> Iterator[str]:
    """
    Yield the answer as it is written, whoever is writing it.

    Chunks are whatever the provider sends -- tokens, words, fragments. The
    caller reassembles them into sentences, which is the only unit speech
    synthesis cares about.
    """
    options = options or {}
    if provider == ANTHROPIC:
        yield from _stream_anthropic(messages, model, options)
    elif provider == GEMINI:
        yield from _stream_gemini(messages, model, options)
    else:
        yield from _stream_ollama(messages, model, options)


def _stream_ollama(messages: list[dict], model: str,
                   options: dict) -> Iterator[str]:
    import ollama

    for chunk in ollama.chat(model=model, messages=messages,
                             options=options, stream=True):
        text = chunk.get("message", {}).get("content", "")
        if text:
            yield text


def _stream_anthropic(messages: list[dict], model: str,
                      options: dict) -> Iterator[str]:
    import anthropic

    key = api_key(ANTHROPIC)
    if not key:
        raise RuntimeError("no Anthropic API key configured")

    system, rest = _split_system(messages)
    client = anthropic.Anthropic(api_key=key)

    request: dict = {
        "model": model or SUGGESTED_MODELS[ANTHROPIC][0],
        "max_tokens": MAX_OUTPUT_TOKENS,
        "messages": _alternating(rest),
    }
    if system:
        request["system"] = system
    # Effort, not thinking budget. Someone is standing in front of the robot
    # waiting for it to talk, so depth is worth less here than latency; the
    # answer is three sentences either way.
    effort = str(options.get("effort") or "low")
    request["output_config"] = {"effort": effort}
    if "temperature" in options:
        # Sampling controls are rejected on current Claude models; effort is the
        # equivalent lever and is already set above.
        logger.debug("Ignoring temperature for Anthropic: not a supported field")

    with client.messages.stream(**request) as stream:
        for text in stream.text_stream:
            if text:
                yield text


def _stream_gemini(messages: list[dict], model: str,
                   options: dict) -> Iterator[str]:
    """
    Gemini over its REST streaming endpoint.

    Raw HTTP rather than the SDK on purpose: this is one request shape, and it
    keeps the robot's dependencies to what is already installed rather than
    adding a client library to a Pi-adjacent environment for a single call.
    """
    import requests

    key = api_key(GEMINI)
    if not key:
        raise RuntimeError("no Gemini API key configured")

    # The same conversion the tool phase uses, so a conversation that already
    # contains tool calls and their results streams with those intact rather
    # than flattened into prose.
    body = _gemini_body(messages, options)
    url = _gemini_url(model, "streamGenerateContent") + "?alt=sse"

    with requests.post(url, json=body, timeout=120, stream=True,
                       headers={"x-goog-api-key": key}) as response:
        if not response.ok:
            raise RuntimeError(
                f"Gemini returned {response.status_code}: {response.text[:200]}")
        for line in response.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if not payload or payload == b"[DONE]":
                continue
            try:
                frame = json.loads(payload)
            except json.JSONDecodeError:
                continue
            for candidate in frame.get("candidates", []):
                for part in candidate.get("content", {}).get("parts", []):
                    text = part.get("text")
                    if text:
                        yield text


def complete(messages: list[dict], model: str, provider: str = OLLAMA,
             options: dict | None = None) -> str:
    """The whole answer at once, for callers that are not speaking it."""
    return "".join(stream_text(messages, model, provider, options))
