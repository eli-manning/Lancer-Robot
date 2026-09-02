"""
Tests for the streaming reply path and interruption recovery.

This logic runs on the robot, where it is expensive to test and awkward to
reproduce: you cannot easily make a room drop a bag at the right moment. So the
robot's SDK and the server are faked here and the real code is driven through
each case that matters -- which is also the only way this got tested before it
was ever installed on hardware.

    python -m pytest tests/test_resume.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "lancer_app" / "src" / "reachy_mini_lancer_app"))

import main                                            # noqa: E402
import sounds                                          # noqa: E402


class FakeMedia:
    """Just enough of reachy_mini.media to run the playback paths."""

    def __init__(self):
        self.pushed: list[np.ndarray] = []
        self.restarts = 0

    def push_audio_sample(self, stereo):
        self.pushed.append(stereo)

    def get_audio_sample(self):
        return None

    def stop_playing(self):
        self.restarts += 1

    def start_playing(self):
        pass


class FakeMini:
    def __init__(self):
        self.media = FakeMedia()
        self.antennas: list[list[float]] = []

    def set_target_antenna_joint_positions(self, positions):
        self.antennas.append(positions)


def make_lancer(**cfg):
    return main.Lancer(FakeMini(), {"thinking_sound": False, **cfg})


def audio(seconds: float = 0.4) -> np.ndarray:
    return np.zeros(int(main.SAMPLE_RATE * seconds), dtype=np.float32)


def sentence_frame(index: int) -> dict:
    return {"type": "sentence", "index": index, "text": f"sentence {index}",
            "audio": audio()}


def reply_frames(count: int = 3, skill: str | None = None) -> list[dict]:
    frames: list[dict] = [{"type": "status", "state": "thinking"}]
    if skill:
        frames.append({"type": "status", "state": "skill", "skill": skill,
                       "ok": True})
    frames += [sentence_frame(i) for i in range(count)]
    frames.append({"type": "done", "answer": "an answer", "sources": [],
                   "unknown": False, "skills": [skill] if skill else []})
    return frames


# --- play_reply ----------------------------------------------------------

def test_uninterrupted_reply_speaks_everything_and_keeps_nothing():
    lancer = make_lancer()
    spoken: list[int] = []
    lancer.play = lambda a, e, **kw: spoken.append(len(a)) or False
    lancer.drain_microphone = lambda: None

    interrupted, pending, meta = lancer.play_reply(reply_frames(3), Stop())

    assert not interrupted
    assert pending == []
    assert len(spoken) == 3
    assert meta["answer"] == "an answer"


def test_interruption_keeps_the_cut_sentence_and_everything_after():
    """
    The sentence being spoken when the interrupt lands is kept whole.

    Resuming from the middle of a waveform would restart mid-word, which sounds
    worse than not resuming at all.
    """
    lancer = make_lancer()
    calls = {"n": 0}

    def play(a, e, **kw):
        calls["n"] += 1
        return calls["n"] == 2          # cut during the second sentence

    lancer.play = play
    lancer.drain_microphone = lambda: None

    interrupted, pending, _ = lancer.play_reply(reply_frames(4), Stop())

    assert interrupted
    # sentence 1 (interrupted) plus sentences 2 and 3, still unspoken.
    assert len(pending) == 3
    assert calls["n"] == 2, "nothing should be spoken after the interrupt"


def test_microphone_is_not_drained_after_an_interrupt():
    """The buffer holds the interrupter's first words; dropping it loses them."""
    lancer = make_lancer()
    drained = {"n": 0}
    lancer.drain_microphone = lambda: drained.__setitem__("n", drained["n"] + 1)
    lancer.play = lambda a, e, **kw: True

    lancer.play_reply(reply_frames(2), Stop())
    assert drained["n"] == 0

    lancer.play = lambda a, e, **kw: False
    lancer.play_reply(reply_frames(2), Stop())
    assert drained["n"] == 1


def test_first_sentence_calibrates_and_the_rest_reuse_the_threshold():
    lancer = make_lancer()
    seen: list[bool] = []
    lancer.play = lambda a, e, **kw: seen.append(kw.get("calibrate")) or False
    lancer.drain_microphone = lambda: None

    lancer.play_reply(reply_frames(3), Stop())
    assert seen == [True, False, False]


# --- converse: was the interruption real? --------------------------------

class Stop:
    """Stand-in for threading.Event that never fires."""

    def is_set(self):
        return False


class Recorder:
    """Records which recovery path converse took."""

    def __init__(self, lancer):
        self.resumed: list[int] = []
        lancer.play_pending = self.play_pending

    def play_pending(self, pending, stop_event):
        self.resumed.append(len(pending))
        return False, []


def run_converse(lancer, frames, follow_up, server_reply):
    """Drive one converse() with listen() and the server both faked."""
    lancer.listen = lambda e, max_wait=None: follow_up
    lancer.face_recently_seen = lambda: False
    original = main.stream_server
    main.stream_server = lambda a, f=False, interrupted=False: server_reply
    try:
        main.LancerApp.converse(main.LancerApp(), lancer, frames, False, Stop())
    finally:
        main.stream_server = original


def interrupted_lancer():
    """A Lancer whose first reply always gets cut off at sentence one."""
    lancer = make_lancer()
    state = {"first": True}

    def play(a, e, **kw):
        if state["first"]:
            state["first"] = False
            return True
        return False

    lancer.play = play
    lancer.drain_microphone = lambda: None
    return lancer


def test_noise_interrupt_resumes_the_answer():
    """Nothing was said afterwards, so nobody interrupted -- finish talking."""
    lancer = interrupted_lancer()
    recorder = Recorder(lancer)

    run_converse(lancer, reply_frames(3), follow_up=None, server_reply=None)

    assert recorder.resumed == [3], "should resume the cut sentence and the rest"


def test_speech_not_addressed_to_lancer_resumes_the_answer():
    """
    Someone talked over Lancer, to somebody else.

    The server's addressing gate answers 204 for that, which stream_server
    reports as None -- the same signal, reused rather than re-derived.
    """
    lancer = interrupted_lancer()
    recorder = Recorder(lancer)

    run_converse(lancer, reply_frames(3), follow_up=audio(1.0), server_reply=None)

    assert recorder.resumed == [3]


def test_a_real_question_replaces_the_unfinished_answer():
    lancer = interrupted_lancer()
    recorder = Recorder(lancer)
    new_reply = reply_frames(2)

    run_converse(lancer, reply_frames(3), follow_up=audio(1.0),
                 server_reply=iter(new_reply))

    assert recorder.resumed == [], "the old answer should be abandoned"


def test_resume_can_be_turned_off():
    lancer = make_lancer(resume_after_false_barge=False)
    lancer.play = lambda a, e, **kw: True
    lancer.drain_microphone = lambda: None
    recorder = Recorder(lancer)

    run_converse(lancer, reply_frames(3), follow_up=None, server_reply=None)

    assert recorder.resumed == []


# --- cues ----------------------------------------------------------------

def test_cues_stay_quiet_enough_not_to_trip_barge_in():
    """
    The cue plays through the same speaker the mic hears.

    Louder than the barge-in floor and Lancer interrupts itself, which looks
    exactly like a bug and is very hard to diagnose from the outside.
    """
    volume = 0.06
    for cue in (sounds.thinking_pulse(volume), sounds.skill_chirp(volume),
                sounds.error_blip(volume), sounds.heard_blip(volume)):
        assert cue.dtype == np.float32
        assert len(cue) > 0
        assert np.max(np.abs(cue)) <= volume * 1.5
        # No clicks: a cue that starts at full amplitude pops in the speaker.
        assert abs(float(cue[0])) < 0.01


def test_cues_do_nothing_when_disabled():
    mini = FakeMini()
    cues = sounds.Cues(mini, volume=0.06, enabled=False)
    cues.skill()
    cues.start_thinking()
    cues.stop_thinking()
    assert mini.media.pushed == []


def test_thinking_loop_starts_and_stops_cleanly():
    mini = FakeMini()
    cues = sounds.Cues(mini, volume=0.05, enabled=True)
    cues.start_thinking()
    cues.start_thinking()          # a second call must not start a second thread
    cues.stop_thinking()
    assert cues._thread is None
    assert mini.media.restarts >= 1, "queued cue audio must be flushed"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
