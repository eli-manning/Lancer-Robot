"""
Spoken durations.

A timer set for the wrong length is worse than one that was refused, so the
parser returns None rather than guessing -- and the fraction cases are here
because "half an hour" originally matched "an hour" and set sixty minutes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import lancer_timers as timers


def test_plain_durations():
    assert timers.parse_duration("5 minutes") == 300
    assert timers.parse_duration("90 seconds") == 90
    assert timers.parse_duration("3 hours") == 10800
    assert timers.parse_duration("2 mins") == 120


def test_spelled_out_numbers():
    assert timers.parse_duration("ten minutes") == 600
    assert timers.parse_duration("twenty seconds") == 20


def test_fractions_do_not_double_the_time():
    assert timers.parse_duration("half an hour") == 1800
    assert timers.parse_duration("quarter of an hour") == 900
    assert timers.parse_duration("half a minute") == 30
    assert timers.parse_duration("an hour and a half") == 5400


def test_compound_durations_add_up():
    assert timers.parse_duration("1 minute 30 seconds") == 90


def test_no_duration_is_refused_not_guessed():
    assert timers.parse_duration("set a timer") is None
    assert timers.parse_duration("for the pasta") is None
    assert timers.parse_duration("") is None


def test_durations_are_described_the_way_people_say_them():
    assert timers.describe(90) == "1 minute 30 seconds"
    assert timers.describe(300) == "5 minutes"
    assert timers.describe(3600) == "1 hour"
    assert timers.describe(5400) == "1 hour 30 minutes"
    assert timers.describe(1) == "1 second"
