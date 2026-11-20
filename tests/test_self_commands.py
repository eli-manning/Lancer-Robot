"""
Asking the robot to do something is not a documents question.

"Nod at me" retrieved a CBU chunk, scored relevant and took the grounded route,
which deliberately offers no tools -- so the gesture was unreachable and he
replied that his head was fixed in place. It is not, and that was not the
question.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import server


def test_physical_commands_are_commands():
    for line in ("nod at me", "Lancer, dance", "wave at us", "look around",
                 "shake your head", "spin"):
        assert server.is_self_command(line), line


def test_self_adjustments_are_commands():
    for line in ("turn it down", "you're too loud", "speak up", "slow down",
                 "be quiet", "louder please", "turn your volume up"):
        assert server.is_self_command(line), line


def test_timers_and_sleep_are_commands():
    for line in ("set a timer for five minutes", "go to sleep", "wake up",
                 "cancel the timer", "take a nap"):
        assert server.is_self_command(line), line


def test_real_questions_are_not_commands():
    # These must still reach retrieval and the grounded route.
    for line in ("who is the Dean of the College of Engineering?",
                 "what are the non-course requirements?",
                 "how many units is the CS degree",
                 "when is the internship deadline",
                 "what is the weather like"):
        assert not server.is_self_command(line), line


def test_reachy_is_his_name_not_another_robot():
    # "Reachy, go to sleep" was answered as if Reachy were someone else.
    assert server.normalize_transcript("Reachy, go to sleep.") == "Lancer, go to sleep."
    assert server.normalize_transcript("Richie, what's up?") == "Lancer, what's up?"
    # The robot model keeps its name.
    assert "Reachy Mini" in server.normalize_transcript("Is that a Reachy Mini?")


def test_go_to_sleep_does_not_depend_on_the_model():
    for line in ("Lancer, go to sleep.", "time for bed", "take a nap",
                 "Lancer, power down"):
        assert dict(server.prefetch_plan(line)).get("rest") == {"action": "sleep"}, line
    assert "rest" not in dict(server.prefetch_plan("how much sleep do students get"))


def test_asking_about_the_name_reachy_keeps_it():
    said = "Why do they call you Reachy?"
    assert server.normalize_transcript(said) == said
    assert server.normalize_transcript("what's up, Reachy?") == "what's up, Lancer?"


def test_he_knows_reachy_is_his_name():
    for prompt in (server.SYSTEM_PROMPT, server.GENERAL_PROMPT,
                   server.CREATIVE_PROMPT, server.QUIZ_PROMPT):
        assert "also call you Reachy" in prompt
