"""
Wake-word mode: the smart-speaker interaction model.

His name every time, except in the short window right after he finishes
speaking. Both halves matter -- without the window you have to say his name to
answer his own question, which is what makes smart speakers tiring; without the
requirement he is back to guessing from the heuristics.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import server


def addressed(text, **kw):
    kw.setdefault("face_present", True)
    kw.setdefault("mode", "wake_word")
    return server.is_addressed(text, kw.pop("face_present"), **kw)


def test_his_name_always_wakes_him():
    ok, why = addressed("Lancer, what time is it?")
    assert ok and why == "wake-word"


def test_a_question_without_his_name_is_ignored():
    # In ambient mode this would get through on face + question shape.
    ok, why = addressed("What time is it?")
    assert not ok
    assert "name" in why


def test_overheard_conversation_is_ignored():
    for line in ("so I told her it was fine",
                 "yeah, the meeting is at four",
                 "what do you think about the new building?"):
        assert not addressed(line)[0]


def test_follow_up_needs_no_name_inside_the_window():
    ok, why = addressed("What about tomorrow?", attentive=True)
    assert ok and why == "follow-up window"


def test_follow_up_needs_the_name_once_the_window_closes():
    assert not addressed("What about tomorrow?", attentive=False)[0]


def test_interrupting_him_counts_as_addressing_him():
    # Barge-in is by definition someone talking to him while he talks.
    assert addressed("no, the other one", interrupted=True)[0]


def test_ambient_mode_is_unchanged():
    # The default heuristics must still work, since this is opt-in.
    assert server.is_addressed("What time is it?", True, mode="ambient")[0]
    assert not server.is_addressed("so anyway, I told her", True,
                                   mode="ambient")[0]


def test_window_is_measured_from_when_playback_ends():
    # Not from when generation finished: the robot plays the first sentence
    # while later ones are still being made, so those differ by seconds.
    import time

    server._attention["until"] = time.monotonic() + 5
    assert server.attentive()
    server._attention["until"] = time.monotonic() - 0.1
    assert not server.attentive()
