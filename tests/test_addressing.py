"""
Tests for the addressing gate.

Every case marked "real transcript" is one Whisper actually produced during a
session where Lancer appeared to be ignoring the person in front of it. It was
not ignoring them: it heard all three, decided none were addressed to it, and
answered with silence, which from the outside is indistinguishable from a
microphone that is not working.

    python -m pytest tests/test_addressing.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server                                          # noqa: E402


def addressed(text: str, face: bool = False) -> tuple[bool, str]:
    return server.is_addressed(server.normalize_transcript(text), face)


# --- what people actually say to a robot facing the wrong way ------------

@pytest.mark.parametrize("said", [
    "Hey, look at me.",                     # real transcript
    "Mancer Belt. Look over here.",         # real transcript
    "Answer Bot. Lue.",                     # real transcript
    "look over here",
    "turn around",
    "face me",
    "over here",
    "wake up",
    "pay attention",
])
def test_commands_reach_lancer(said):
    ok, why = addressed(said)
    assert ok, f"{said!r} was discarded as {why}"


def test_commands_work_with_nobody_in_frame():
    """
    Standing off-camera is the reason the command is being given.

    Requiring a visible face here would mean the one situation the feature
    exists for is the one situation it refuses to work in.
    """
    assert addressed("look over here", face=False)[0]


@pytest.mark.parametrize("said", [
    "stop talking",
    "be quiet",
    "never mind",
])
def test_lancer_can_be_told_to_stop(said):
    assert addressed(said)[0]


# --- Whisper cannot spell Lancer -----------------------------------------

@pytest.mark.parametrize("mangled", [
    "Mancer Belt. Look over here.",
    "Answer Bot. Lue.",
    "Lancer Belt, what is BASAI?",
    "lanser what time is it",
])
def test_mangled_wake_word_still_wakes_it(mangled):
    assert "lancer" in server.normalize_transcript(mangled).lower()


def test_the_ordinary_word_answer_is_left_alone():
    """
    "Answer" is a normal English word.

    It only becomes the robot's name when it opens the utterance, or the gate
    would wake on someone saying "the answer was four".
    """
    normalized = server.normalize_transcript("yeah dude the answer was like four")
    assert "lancer" not in normalized.lower()


# --- overheard conversation is still overheard conversation --------------

@pytest.mark.parametrize("chatter", [
    "and then he said look at me",          # recounting, not commanding
    "yeah dude the answer was like four",
    "so like it was over here somewhere",
    "sure dude",
])
def test_overheard_speech_is_still_ignored(chatter):
    ok, _ = addressed(chatter, face=True)
    assert not ok, f"{chatter!r} should not be treated as addressed"


def test_questions_still_get_through():
    assert addressed("What classes do I take in BASAI?")[0]


# --- and they are answered without a document lookup ---------------------

@pytest.mark.parametrize("command,expected", [
    ("look at me", "Over here?"),
    ("turn around", "Over here?"),
    ("never mind", "Fine."),
])
def test_commands_get_an_instant_reply(command, expected):
    """
    Someone telling the robot where to look wants it to look now.

    Routing that through retrieval and the model costs seconds and produces a
    non-answer at the end of them.
    """
    reply = server.social_reply(command)
    assert reply is not None, f"{command!r} fell through to retrieval"
    assert reply.startswith(expected)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
