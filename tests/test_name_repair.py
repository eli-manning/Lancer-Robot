"""
Tests for rescuing a search that a mis-heard name broke.

From a real session: someone asked "who is Philip Van Haaster" and Whisper
wrote "Philip Van Hoster". That one syllable moved the rerank score from 3.68
to -5.84, under the relevance floor -- so the correct document was retrieved,
ranked first, and then discarded as irrelevant. The robot said it had no idea,
about a name printed in its own corpus.

The dangerous fix is to fuzzy-match names in the transcript. Measured against
this corpus that rewrites "Turing" into "During" and "Costco" into "Cost", so a
question about Alan Turing becomes a question about the engineering faculty.
These tests pin the safe shape instead: repair only after a search has failed,
and only keep it when it demonstrably worked.

Retrieval tests need Chroma and the reranker, so they skip when those are not
running rather than failing the suite on a laptop.

    python -m pytest tests/test_name_repair.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server                                          # noqa: E402


@pytest.fixture(scope="module")
def corpus():
    try:
        server.build_lexical_index()
    except Exception as e:                              # no Chroma running
        pytest.skip(f"retrieval stack unavailable: {e}")
    if not server._bm25_docs:
        pytest.skip("no documents indexed")
    return True


# --- the transcript itself is never rewritten ----------------------------

def test_the_transcript_is_left_alone(corpus):
    """
    What the person said is what gets answered and logged.

    Name repair happens on the retrieval query, never here -- rewriting the
    transcript would mean the answer quietly addresses a different question
    than the one asked.
    """
    for said in ("Who was Alan Turing?", "is there a Costco near campus",
                 "who is Philip Van Hoster"):
        assert server.normalize_transcript(said) == said


def test_the_robots_own_name_is_protected(corpus):
    """CBU's mascot is the Lancers, so the corpus is full of the plural."""
    assert "Lancers," not in server.repair_proper_nouns("Lancer, who is that?")


# --- repair rescues a real miss ------------------------------------------

@pytest.mark.parametrize("misheard", [
    "who is Philip Van Hoster",       # real transcript
    "who is Philip Van Hooster",
    "who is Philip Van Haastner",     # real transcript
])
def test_a_misheard_name_still_finds_the_answer(corpus, misheard):
    _context, _sources, relevant, _score = server.retrieve_or_repair(misheard)
    assert relevant, f"{misheard!r} should still reach the dean's page"


def test_repair_actually_improves_the_score(corpus):
    _c, _s, _relevant, before = server.retrieve("who is Philip Van Hoster")
    _c, _s, relevant, after = server.retrieve_or_repair("who is Philip Van Hoster")
    assert relevant
    assert after > before + server._REPAIR_MIN_GAIN


# --- and never invents a campus answer for an outside question -----------

@pytest.mark.parametrize("outside", [
    "Who was Alan Turing?",
    "who is Osama bin Laden",
    "is there a Costco near campus",
    "what is the tuition for underwater basket weaving",
])
def test_outside_questions_are_still_refused(corpus, outside):
    """
    The matcher will happily mangle these into campus words.

    It is allowed to try. It is not allowed to have the result kept, because
    the rewritten query has to actually retrieve something good, and it does
    not. This is the property that lets the cutoff be generous.
    """
    _context, _sources, relevant, _score = server.retrieve_or_repair(outside)
    assert not relevant, f"{outside!r} must not be answered from CBU documents"


# --- questions that already work are untouched ---------------------------

@pytest.mark.parametrize("question", [
    "What classes do I take in year one fall of BASAI",
    "Can I transfer credits from a community college",
    "tell me about the machine learning concentration",
    "How many internship hours do I need to graduate",
])
def test_working_questions_are_not_disturbed(corpus, question):
    _c, _s, plain, plain_score = server.retrieve(question)
    _c, _s, through, through_score = server.retrieve_or_repair(question)
    assert plain and through
    assert plain_score == through_score, "a hit must not be re-run or altered"


# --- the thing it used to refuse to do -----------------------------------

def test_lancer_no_longer_claims_it_cannot_dance():
    """
    It can dance now, through play_emotion.

    The canned "can't do that yet" reply fired before the skill was ever
    reached, so the robot insisted it could not do a thing it had just been
    taught.
    """
    assert server.social_reply("do a dance") is None
    assert server.social_reply("dance for me") is None
    assert server.social_reply("tell me a joke") is None
    # Things it genuinely cannot do should still say so.
    assert server.social_reply("take a picture of me") is not None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
