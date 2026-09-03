"""
Gemini's tool-calling wire format.

Worth testing offline because getting it wrong does not raise: Gemini accepts a
malformed conversation and simply behaves as though no skill was ever called,
which on a robot looks like a model that will not use its tools.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import lancer_llm as llm


def test_schema_is_filtered_to_what_gemini_accepts():
    tools = llm._gemini_tools([{
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Live weather",
            "parameters": {
                "type": "object",
                "additionalProperties": False,      # rejected by Gemini
                "$schema": "http://json-schema.org/draft-07/schema#",
                "properties": {"when": {"type": "string",
                                        "enum": ["now", "tomorrow"],
                                        "default": "now"}},
            },
        },
    }])
    declaration = tools[0]["functionDeclarations"][0]
    assert declaration["name"] == "weather"
    assert "additionalProperties" not in declaration["parameters"]
    assert "$schema" not in declaration["parameters"]
    assert "default" not in declaration["parameters"]["properties"]["when"]
    assert declaration["parameters"]["properties"]["when"]["enum"] == \
        ["now", "tomorrow"]


def test_skill_without_arguments_declares_no_parameters():
    # An empty parameter object is rejected outright, so it must be omitted.
    tools = llm._gemini_tools([{"function": {"name": "current_datetime",
                                             "description": "The time",
                                             "parameters": {"type": "object"}}}])
    assert "parameters" not in tools[0]["functionDeclarations"][0]


def test_tool_call_and_result_become_paired_turns():
    system, contents = llm._gemini_contents([
        {"role": "system", "content": "You are Lancer."},
        {"role": "user", "content": "What is the weather?"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"function": {"name": "weather",
                                      "arguments": {"when": "now"}}}]},
        {"role": "tool", "name": "weather",
         "content": '{"ok": true, "temp_f": 78}'},
    ])
    assert system == "You are Lancer."
    assert [c["role"] for c in contents] == ["user", "model", "user"]
    assert contents[1]["parts"][0]["functionCall"] == {
        "name": "weather", "args": {"when": "now"}}
    response = contents[2]["parts"][0]["functionResponse"]
    assert response["name"] == "weather"
    # Must be an object, not a JSON string.
    assert response["response"] == {"ok": True, "temp_f": 78}


def test_string_arguments_are_parsed():
    _, contents = llm._gemini_contents([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "tool_calls": [
            {"function": {"name": "play_emotion",
                          "arguments": '{"emotion": "happy"}'}}]},
    ])
    assert contents[1]["parts"][0]["functionCall"]["args"] == {"emotion": "happy"}


def test_two_calls_in_one_turn_stay_one_turn():
    _, contents = llm._gemini_contents([
        {"role": "user", "content": "weather and time?"},
        {"role": "assistant", "tool_calls": [
            {"function": {"name": "weather", "arguments": {}}},
            {"function": {"name": "current_datetime", "arguments": {}}}]},
    ])
    assert len(contents) == 2
    assert len(contents[1]["parts"]) == 2


def test_conversation_never_opens_on_a_model_turn():
    # History can start mid-answer after a truncation; Gemini rejects that.
    _, contents = llm._gemini_contents([
        {"role": "assistant", "content": "...as I was saying."},
        {"role": "user", "content": "go on"},
    ])
    assert contents[0]["role"] == "user"


def test_unparseable_tool_output_is_still_an_object():
    # response must be an object even when the skill printed plain text.
    _, contents = llm._gemini_contents([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "tool_calls": [
            {"function": {"name": "web_lookup", "arguments": {}}}]},
        {"role": "tool", "name": "web_lookup", "content": "not json at all"},
    ])
    assert contents[2]["parts"][0]["functionResponse"]["response"] == \
        {"result": "not json at all"}


def test_only_ollama_and_gemini_are_offered_schemas():
    assert llm.supports_tool_calls(llm.OLLAMA)
    assert llm.supports_tool_calls(llm.GEMINI)
    assert not llm.supports_tool_calls(llm.ANTHROPIC)
