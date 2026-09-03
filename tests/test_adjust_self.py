"""
What Lancer is allowed to change about himself.

The point of these tests is the exclusions. Everything in the allowlist is
reversible and cosmetic; the settings left out are the ones a model would have
an incentive to relax -- being interruptible, having its answers checked -- and
a later edit that quietly adds one should fail here rather than ship.
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location(
    "adjust_self_run", ROOT / "skills" / "adjust_self" / "run.py")
adjust = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adjust)

import lancer_runtime as rt


def _reachable_config_keys():
    """Config keys the skill can write, by any route."""
    return {key for key, *_ in adjust.NUMERIC.values()} | {
        "tts_voice", "thinking_sound"}


def test_safety_settings_are_not_reachable():
    forbidden = {
        "barge_enabled",        # you must always be able to interrupt him
        "barge_grace",
        "barge_echo_headroom",
        "verify_grounding",     # not his to switch off
        "relevance_min",        # decides when he admits he does not know
        "llm_provider",         # nothing here moves the conversation off-machine
        "llm_model",
        "robot_hosts",
        "conversation_backend",
        "skills_enabled",
        "memory_enabled",
    }
    assert not (_reachable_config_keys() & forbidden)


def test_every_reachable_key_is_a_real_setting():
    for key in _reachable_config_keys():
        assert key in rt.DEFAULTS, f"{key} is not a config key"


def test_declared_settings_match_the_manifest():
    import json

    manifest = json.loads(
        (ROOT / "skills" / "adjust_self" / "skill.json").read_text())
    declared = set(manifest["parameters"]["properties"]["setting"]["enum"])
    handled = set(adjust.NUMERIC) | {"volume", "voice", "thinking_sound"}
    assert declared == handled


def test_values_are_clamped_to_the_setting_range():
    assert adjust._clamp(99.0, 0.5, 2.0) == 2.0
    assert adjust._clamp(-4.0, 0.5, 2.0) == 0.5


def test_relative_words_people_actually_use_are_understood():
    for word in ("up", "louder", "quieter", "down", "slower", "faster"):
        assert word in adjust.UP or word in adjust.DOWN


def test_minimum_volume_is_not_silence():
    # A robot that has muted itself is indistinguishable from a broken one, and
    # cannot be asked to turn itself back up.
    assert adjust._clamp(0, 10, 100) == 10
