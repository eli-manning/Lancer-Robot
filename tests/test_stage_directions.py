"""
Narrated actions must not be spoken.

Models write stage directions constantly for a character who talks, and
text-to-speech reads them out -- so "(Lancer laughs.)" came out of the speaker
as him saying the words "Lancer laughs". Parentheses cannot simply be stripped:
the corpus is full of course names that belong in the answer.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import server


def clean(text):
    return server.strip_stage_directions(text)


def test_narrated_actions_are_removed():
    # All of these were actually spoken by the robot.
    assert clean("Ha! (Lancer laughs.) That's a good one.") == \
        "Ha! That's a good one."
    assert clean("(Lancer does a little dance.) Ready when you are.") == \
        "Ready when you are."
    assert clean("Sure thing. *nods* What else?") == "Sure thing. What else?"
    assert "snore" not in clean("Zzzz... (snore) Sweet dreams!")


def test_bare_action_words_count_too():
    for line in ("*sighs*", "(chuckles)", "(clears his throat)", "(beep)",
                 "(whirs)", "[laughs]"):
        assert clean(f"Right. {line} Anyway.") == "Right. Anyway."


def test_real_parentheticals_survive():
    # These are part of the answer, not narration.
    for line in ("You need MAT 3350 (Mathematical Statistics I).",
                 "That's CSC 4370 (Machine & Deep Learning).",
                 "Take a GE course (GE-BTC) before you apply.",
                 "The Dean (as of this year) is Philip Van Haaster."):
        assert clean(line) == line


def test_emphasis_is_left_alone():
    # Markdown emphasis is unwrapped later; removing the span would delete the
    # word, which is how "*laughs*" and "*definitely*" need opposite handling.
    assert clean("The answer is *definitely* yes.") == \
        "The answer is *definitely* yes."


def test_an_ellipsis_is_not_collapsed():
    assert clean("Well... let me think.") == "Well... let me think."


def test_punctuation_left_by_a_removal_is_tidied():
    assert clean("Done. (nods). Next question.") == "Done. Next question."


def test_the_speech_path_applies_it():
    assert "Lancer laughs" not in server._clean_for_speech(
        "Sure. (Lancer laughs.) Here you go.")


def test_a_tool_call_written_as_words_is_not_spoken():
    # Gemini wrote "(playemotion: happy)" instead of calling the tool, then
    # copied itself from history every turn after.
    for raw in ("(playemotion: happy) That's correct!",
                "[play_emotion: sad] Oh, not quite.",
                "(emotion: curious) Tell me more."):
        text, meant = server.lift_pseudo_calls(raw)
        assert "emotion" not in text.lower(), text
        assert len(meant) == 1
    assert server.strip_stage_directions("(playemotion: sad) Oh, not quite.") \
        == "Oh, not quite."
    assert server.lift_pseudo_calls("(playemotion: happy) Hi")[1] == ["happy"]


def test_real_parentheticals_survive():
    text, meant = server.lift_pseudo_calls("It moved to Riverside (in 1955).")
    assert text == "It moved to Riverside (in 1955)."
    assert meant == []
