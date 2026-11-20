"""
Which questions must reach the camera.

Vision is prefetched rather than left to the model because the failure mode is
not silence. Asked how many people were in the room, he answered "two of you"
with the camera untouched; the sweep found three. A wrong number said
confidently is worse than a pause.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import server


def plan(query):
    return dict(server.prefetch_plan(query))


def test_room_questions_survey():
    for line in ("how many people are in this room right now?",
                 "who else is in the room?",
                 "who's all in this room",
                 "is anybody else here?",
                 "what's behind you?",
                 "describe the room",
                 "look around"):
        assert plan(line).get("look") == {"survey": True}, line


def test_straight_ahead_questions_do_not_sweep():
    # Six seconds of head-turning before answering "a blue mug" is worse than
    # the narrower view.
    for line in ("what do you see", "what am I holding", "can you see",
                 "what does my shirt say", "read this"):
        assert plan(line).get("look") == {}, line


def test_questions_with_nothing_to_look_at_do_not_use_the_camera():
    for line in ("what time is it", "do I need a jacket",
                 "who is the Dean of the College of Engineering?",
                 "tell me a joke"):
        assert "look" not in plan(line), line


def test_arguments_reach_the_skill():
    # The table always carried arguments and they were being discarded, so a
    # rule that fired precisely because the question was about the whole room
    # still took a single forward glance.
    assert server.prefetch_plan("how many people are here") == [
        ("look", {"survey": True})]


def test_a_skill_is_not_prefetched_twice():
    names = [n for n, _ in server.prefetch_plan(
        "what do you see in this room right now")]
    assert len(names) == len(set(names))


def test_gather_refuses_a_tool_it_did_not_offer(monkeypatch):
    """
    The allowlist has to decide what runs, not just what the model is shown.

    Gemini called `look` on a turn where look was deliberately withheld -- it
    had already run by rule -- and the loop ran it a second time, sweeping the
    head twice for one question. A withheld skill must be refused, not merely
    absent from the schemas.
    """
    import lancer_skills

    ran = []

    def fake_chat(messages, model, provider, tools, options):
        # A model asking for something outside the offered set.
        if not ran:
            return {"role": "assistant", "content": "",
                    "tool_calls": [{"function": {"name": "look",
                                                 "arguments": {}}}]}
        return {"role": "assistant", "content": "done", "tool_calls": []}

    def fake_run(name, arguments, approve=False):
        ran.append(name)
        return {"ok": True}

    monkeypatch.setattr("lancer_llm.chat_with_tools", fake_chat)
    monkeypatch.setattr(lancer_skills, "run_skill", fake_run)

    _messages, calls = lancer_skills.gather(
        "m", [{"role": "user", "content": "hi"}],
        allow=["current_datetime"], provider="ollama")

    assert ran == [], "a withheld skill must not run"
    assert calls and calls[0]["skill"] == "look"
    assert calls[0]["result"]["ok"] is False


def test_one_side_turns_that_way_instead_of_sweeping():
    # "To your right" swept the whole room, ran out of time, and saw nothing.
    for line, side in (("Lancer, look at the TV. To your right.", "right"),
                       ("what's on the screen on your right?", "right"),
                       ("look right", "right"),
                       ("can you see the whiteboard to your left", "left"),
                       ("look over left", "left")):
        assert plan(line).get("look") == {"direction": side}, line


def test_right_now_is_not_a_direction():
    # "right now" must not turn the head.
    assert plan("how many people are here right now").get("look") == {
        "survey": True}
    assert "look" not in plan("what time is it right now")
