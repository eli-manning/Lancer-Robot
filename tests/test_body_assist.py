"""
Tests for the body-follows-head controller.

The first version of this shipped straight to hardware and made the robot
thrash: it and the daemon's automatic body yaw were both solving for the body,
so one turned it and the other unwound it, several times a second. These
simulate the loop instead, because the cost of being wrong here is a machine
shaking on a desk rather than a red test.

    python -m pytest tests/test_body_assist.py -q
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "lancer_app" / "src" / "reachy_mini_lancer_app"))

import main                                            # noqa: E402


def step(head_yaw: float, gain: float = main._BODY_ASSIST_GAIN,
         max_step: float = main._BODY_ASSIST_MAX_STEP) -> float:
    """One correction, exactly as the loop computes it."""
    return max(-max_step, min(max_step, head_yaw * gain))


def simulate(start: float, ticks: int = 200) -> list[float]:
    """
    Run the loop against an ideal plant.

    The head tracker holds the head on the person, so rotating the body by x
    reduces the head's offset by x. That is the whole feedback path.
    """
    head = start
    history = [head]
    assisting = False
    for _ in range(ticks):
        if not assisting and abs(head) < main._BODY_ASSIST_DEADBAND:
            break
        exit_band = main._BODY_ASSIST_DEADBAND * main._BODY_ASSIST_EXIT_RATIO
        if assisting and abs(head) < exit_band:
            break
        assisting = True
        head -= step(head)
        history.append(head)
    return history


@pytest.mark.parametrize("start", [0.25, 0.5, 0.9, -0.4, -1.0])
def test_it_converges_and_never_overshoots(start):
    history = simulate(start)
    assert abs(history[-1]) < main._BODY_ASSIST_DEADBAND

    # Never crosses zero: overshoot is what turns into hunting on hardware.
    signs = {h > 0 for h in history if abs(h) > 1e-9}
    assert len(signs) <= 1, f"changed direction: {history[:12]}"

    # Monotonically toward centre.
    for before, after in zip(history, history[1:]):
        assert abs(after) <= abs(before) + 1e-12


def test_it_settles_in_a_reasonable_time():
    """
    Converging is not enough; it has to look deliberate rather than glacial.

    At 0.15s a tick, staying under ~40 ticks keeps a big turn inside six
    seconds, and small ones inside two.
    """
    for start in (0.3, 0.6, 1.0):
        assert len(simulate(start)) < 40, f"too slow from {start}"


def test_the_step_is_velocity_capped():
    """A person appearing at 90 degrees must not snap the body round."""
    assert abs(step(math.pi / 2)) <= main._BODY_ASSIST_MAX_STEP
    assert abs(step(-math.pi)) <= main._BODY_ASSIST_MAX_STEP


def test_hysteresis_band_is_inside_the_deadband():
    """
    Entering and leaving on the same threshold makes it chatter.

    It starts on a clear offset and keeps going until well centred, so it
    cannot sit on the boundary switching on and off.
    """
    exit_band = main._BODY_ASSIST_DEADBAND * main._BODY_ASSIST_EXIT_RATIO
    assert 0 < exit_band < main._BODY_ASSIST_DEADBAND


def test_a_gain_that_would_hunt_is_not_reachable_from_config():
    """
    Above a gain of 1 each correction overshoots further than the last.

    The clamp in lancer_runtime keeps config well below that, because this is
    exactly the knob that produced the thrashing.
    """
    import lancer_runtime as rt

    low, high = rt.LIMITS["body_assist_gain"]
    assert high < 1.0
    assert rt._clamp("body_assist_gain", 5.0) == high

    # And prove the unstable region really is unstable, so the clamp is not
    # cargo-culted. Exactly 2.0 flips the sign every step at constant
    # amplitude -- already a robot rocking in place; above it the amplitude
    # grows. Neither ever settles.
    for bad_gain, expected in ((2.0, "rocks"), (2.5, "diverges")):
        head = 0.3
        for _ in range(6):
            head -= head * bad_gain
        assert abs(head) >= 0.3, f"gain {bad_gain} should not converge ({expected})"


def test_default_gain_and_deadband_agree_with_the_runtime_defaults():
    """The app's fallbacks and the dashboard's defaults must not drift apart."""
    import lancer_runtime as rt

    assert rt.DEFAULTS["body_assist_gain"] == main._BODY_ASSIST_GAIN
    assert rt.DEFAULTS["body_assist_deadband"] == main._BODY_ASSIST_DEADBAND
    assert rt.DEFAULTS["body_assist_max_step"] == main._BODY_ASSIST_MAX_STEP
    assert rt.DEFAULTS["body_assist_interval"] == main._BODY_ASSIST_INTERVAL


def test_body_assist_is_off_by_default_and_refuses_to_share_the_body():
    """
    It must not start alongside automatic body yaw.

    That combination is the one that shook the robot, and a config edit should
    not be able to reach it.
    """
    import lancer_runtime as rt

    assert rt.DEFAULTS["body_assist"] is False
    assert rt.DEFAULTS["auto_body_yaw"] is True

    class FakeMini:
        media = None

    robot = main.Lancer(FakeMini(), {"body_assist": True, "auto_body_yaw": True})
    started = {"yes": False}

    class Stop:
        def is_set(self):
            started["yes"] = True      # only reached if the loop actually runs
            return True

    robot.body_assist_loop(Stop())
    assert not started["yes"], "loop must refuse to start against automatic yaw"


def test_counter_rotation_keeps_the_head_pointing_the_same_way():
    """
    Turning the body must not drag the gaze off the person.

    Head pose is in the body's frame, so a body step of +x moves the head +x in
    the room unless the head is rotated -x to compensate. Without this the head
    overshoots on every step and the tracker hauls it back, which is what "the
    head resets when the body moves" looks like from outside.
    """
    import numpy as np

    class FakeMini:
        media = None

    robot = main.Lancer(FakeMini(), {})
    step = 0.05

    head_yaw_in_body = 0.30
    pose = np.eye(4)
    pose[0, 0] = math.cos(head_yaw_in_body)
    pose[0, 1] = -math.sin(head_yaw_in_body)
    pose[1, 0] = math.sin(head_yaw_in_body)
    pose[1, 1] = math.cos(head_yaw_in_body)

    body_before = 0.10
    heading_before = body_before + head_yaw_in_body

    compensated = robot._rotate_pose_z(pose, -step)
    new_head_yaw = math.atan2(compensated[1, 0], compensated[0, 0])
    heading_after = (body_before + step) + new_head_yaw

    assert heading_after == pytest.approx(heading_before, abs=1e-9)
    # And the head is now less turned relative to the body, which is the point.
    assert abs(new_head_yaw) < abs(head_yaw_in_body)


def test_counter_rotation_leaves_the_pose_a_valid_rigid_transform():
    import numpy as np

    class FakeMini:
        media = None

    robot = main.Lancer(FakeMini(), {})
    pose = np.eye(4)
    pose[0, 3], pose[1, 3], pose[2, 3] = 0.01, 0.02, 0.15

    out = robot._rotate_pose_z(pose, -0.05)
    rotation = out[:3, :3]

    assert out.shape == (4, 4)
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
    assert float(np.linalg.det(rotation)) == pytest.approx(1.0)
    assert out[3, 3] == pytest.approx(1.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
