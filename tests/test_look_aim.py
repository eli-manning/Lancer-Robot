"""
Where the look skill points the camera.

The daemon takes head yaw in the world frame. The sweep used to send the head
yaw as an offset from the body, so the neck counter-rotated against every body
turn: the body swung right and the camera kept staring ahead.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "skills" / "look"))
sys.path.insert(0, str(ROOT / "skills"))
sys.path.insert(0, str(ROOT))

import run as look  # noqa: E402


def test_head_turns_with_the_body():
    goto = look._aim({}, -1.0, 0.0)
    assert goto["body_yaw"] == -1.0
    assert goto["head_pose"]["yaw"] == -1.0


def test_head_offset_adds_to_the_body_turn():
    goto = look._aim({}, 1.85, 0.6)
    assert abs(goto["head_pose"]["yaw"] - 2.45) < 1e-9


def test_limits_apply_to_body_and_neck_separately():
    goto = look._aim({}, 5.0, 5.0)
    assert goto["body_yaw"] == look.BODY_LIMIT
    assert goto["head_pose"]["yaw"] == look.BODY_LIMIT + look.HEAD_LIMIT


def test_looks_level_whatever_the_head_was_doing():
    # Tracking had left the head pitched up at someone; the turn kept that
    # pitch and every view was ceiling.
    rest = {"x": 0.01, "z": -0.02, "pitch": -0.59, "roll": 0.1, "yaw": 0.9}
    head = look._aim(rest, 0.0, 0.0)["head_pose"]
    assert head["pitch"] == look.LOOK_PITCH and head["roll"] == 0.0
    assert head["z"] == -0.02 and head["yaw"] == 0.0


def test_right_is_negative_yaw_and_left_positive():
    (right_body, _), = look.DIRECTIONS["right"]
    (left_body, _), = look.DIRECTIONS["left"]
    assert right_body < 0 < left_body
    assert look.DIRECTIONS["ahead"] == ()
