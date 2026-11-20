"""
Names in an answer are checked against the retrieved text.

The check used to read "Dr. Dan Grissom" as "Dr. Dan G" -- the surname's first
letter taken for a middle initial -- so it looked for the surname "g", never
found it, and blocked every answer that named a titled professor.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import server

CONTEXT = ("Meet Some of Our Faculty Dan Grissom, PhD Professor of Computer "
           "Science Department Chair for Computing, Software, and Data Sciences")


def test_full_surname_is_matched():
    assert server._NAME_RE.findall("Dr. Dan Grissom chairs it.") == [
        "Dr. Dan Grissom"]


def test_true_name_passes():
    assert server._unsupported_claims(
        "Dr. Dan Grissom chairs the department.", CONTEXT) == []


def test_invented_name_is_still_caught():
    assert server._unsupported_claims(
        "Dr. Mark Keller teaches the intro course.", CONTEXT)


def test_real_middle_initials_still_work():
    assert server._NAME_RE.findall("Dr. Dan G. Grissom") == ["Dr. Dan G. Grissom"]
    assert server._NAME_RE.findall("Prof. Ann K Lee") == ["Prof. Ann K Lee"]
