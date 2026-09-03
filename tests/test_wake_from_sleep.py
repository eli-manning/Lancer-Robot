"""
Getting him out of a nap.

Sleep is the one state where a mistranscription strands him: any other ignored
utterance costs a repeat, but here the only way back is the dashboard. So the
wake check is deliberately looser than the addressing gate -- while still not
waking on ordinary conversation, which is what he hears most of while folded.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import server


def test_his_names_wake_him():
    for line in ("Lancer, wake up", "Reachy?", "wake up", "good morning",
                 "rise and shine", "get up"):
        assert server.looks_like_wake(line), line


def test_common_manglings_wake_him():
    # All of these are Whisper renderings of his name seen in real logs.
    for line in ("Richie", "reach he", "Lance", "Ritchie, wake",
                 "Lanser", "preachy"):
        assert server.looks_like_wake(line), line


def test_room_conversation_does_not_wake_him():
    # Every one of these was transcribed while he was actually asleep.
    for line in ("I remember it was like, why is it started?",
                 "And I was like, oh man, just transition.",
                 "I left the day before. It was like, beautiful.",
                 "They're really, like, they kind of miss.",
                 "It's like there's like a power up and stuff.",
                 "There's actually a couple more.",
                 "I wasn't that bad."):
        assert not server.looks_like_wake(line), line


def test_a_near_miss_inside_a_long_sentence_is_a_coincidence():
    assert not server.looks_like_wake(
        "we should probably balance the load across the other machines today")


def test_answer_is_not_lancer():
    # The nearest common false friend; it must not wake him.
    assert not server.looks_like_wake("answer")
