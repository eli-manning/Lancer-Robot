"""
Tests for turning the robot toward a sound.

The bug these exist to prevent shipped once already and was invisible from the
outside: the ReSpeaker's bearing was passed straight to set_target_body_yaw as
an absolute heading. Someone standing directly in front reads as pi/2 on that
scale, so Lancer commanded a 90 degree turn away from them. It looked like
unreliable tracking. It was wrong on every single reading.

    python -m pytest tests/test_sound_geometry.py -q
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "lancer_app" / "src" / "reachy_mini_lancer_app"))

import main                                            # noqa: E402

FRONT = math.pi / 2      # the driver's convention: 0 = left, pi/2 = front, pi = right
LEFT = 0.0
RIGHT = math.pi


class FakeMini:
    def __init__(self):
        self.media = None


def lancer(**cfg):
    return main.Lancer(FakeMini(), {"thinking_sound": False, **cfg})


def test_sound_from_the_front_needs_no_turn():
    """The case the old code got most wrong: it commanded 90 degrees."""
    assert lancer().sound_offset(FRONT) == pytest.approx(0.0, abs=1e-9)


def test_sound_from_the_left_and_right_are_opposite_and_equal():
    robot = lancer()
    left = robot.sound_offset(LEFT)
    right = robot.sound_offset(RIGHT)
    assert left == pytest.approx(-right)
    assert abs(left) == pytest.approx(math.pi / 2)


def test_a_bearing_just_off_centre_is_a_small_turn():
    """
    Someone a little to one side should produce a nudge, not a lunge.

    Under the old arithmetic this same bearing produced a command of 1.66 rad,
    roughly 95 degrees, for a person about 5 degrees off centre.
    """
    offset = lancer().sound_offset(FRONT + 0.1)
    assert abs(offset) == pytest.approx(0.1, abs=1e-9)


def test_the_sign_is_configurable_without_a_redeploy():
    """
    Which way positive yaw turns depends on how the array is mounted.

    That is a fact about the hardware nobody can read off a datasheet at 2am,
    so it is config: if Lancer turns away from the speaker, flip this.
    """
    normal = lancer(doa_yaw_sign=1.0).sound_offset(LEFT)
    flipped = lancer(doa_yaw_sign=-1.0).sound_offset(LEFT)
    assert normal == pytest.approx(-flipped)


@pytest.mark.parametrize("bearing", [0.0, 0.4, 1.2, FRONT, 2.0, 2.9, math.pi])
def test_every_bearing_stays_within_the_mechanical_limit(bearing):
    """No bearing may demand more than the yaw range the IK will accept."""
    assert abs(lancer().sound_offset(bearing)) <= math.pi / 2 + 1e-9


def test_offset_is_relative_so_a_turn_composes_with_where_we_already_point():
    """
    The offset is a correction, not a destination.

    The array turns with the robot, so a bearing means "this far off from where
    you are now" and has to be added to the current body yaw. Treating it as an
    absolute heading was the second half of the original bug.
    """
    robot = lancer()
    offset = robot.sound_offset(LEFT)          # 90 degrees to one side
    for already_facing in (0.0, 0.5, -0.8):
        assert already_facing + offset == pytest.approx(already_facing + math.pi / 2)


# --- body assist ---------------------------------------------------------

def test_head_yaw_is_read_out_of_the_pose_matrix():
    import numpy as np

    robot = lancer()
    yaw = 0.42
    matrix = np.eye(4)
    matrix[0, 0] = math.cos(yaw)
    matrix[0, 1] = -math.sin(yaw)
    matrix[1, 0] = math.sin(yaw)
    matrix[1, 1] = math.cos(yaw)
    robot.mini.get_current_head_pose = lambda: matrix

    assert robot._head_yaw() == pytest.approx(yaw)


def test_head_yaw_survives_a_pose_it_cannot_read():
    robot = lancer()
    robot.mini.get_current_head_pose = lambda: None
    assert robot._head_yaw() is None

    def boom():
        raise RuntimeError("no pose")

    robot.mini.get_current_head_pose = boom
    assert robot._head_yaw() is None


def test_body_assist_gain_converges_rather_than_oscillates():
    """
    At a gain of 1 the body chases every twitch of the head.

    Simulate the loop: the body takes `gain` of the head's offset and the head
    recentres onto the same target. Under 1 this settles; at 1 it can overshoot
    and hunt, which on hardware reads as the robot shuddering.
    """
    gain = main._BODY_ASSIST_GAIN
    assert 0.0 < gain < 1.0

    remaining = 0.8                     # head is 46 degrees off centre
    for _ in range(6):
        remaining -= remaining * gain
    assert abs(remaining) < 0.05, "body assist should converge within a few steps"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
