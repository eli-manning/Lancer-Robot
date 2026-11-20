"""
The CBU trivia quiz.

Grading is deterministic and lives outside the model because this is the one
place where being generous costs someone else money: a model told the answer
is 1950 will pass 1955 on a good day. And while a quiz is running, the words
coming in are answers, not questions -- "Riverside" must not go to retrieval.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import lancer_quiz as quiz
import server


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(quiz, "STORE", tmp_path / "quiz.json")
    monkeypatch.setattr(quiz, "RESULTS", tmp_path / "results.jsonl")
    yield


# --- Grading ---------------------------------------------------------------


def test_spoken_numbers_become_digits():
    assert quiz.normalize("nineteen fifty") == "1950"
    assert quiz.normalize("twenty twenty five") == "2025"
    assert quiz.normalize("two thousand") == "2000"
    assert quiz.normalize("fifty five") == "55"
    assert quiz.normalize("1998") == "1998"


def test_answer_inside_a_sentence_counts():
    q = {"q": "city?", "answer": "Riverside"}
    assert quiz.grade("I think it's Riverside", q)
    assert quiz.grade("riverside, california", q)
    assert not quiz.grade("San Bernardino", q)


def test_years_must_match_exactly():
    q = {"q": "year?", "answer": "1950"}
    assert quiz.grade("nineteen fifty", q)
    assert quiz.grade("it was 1950", q)
    assert not quiz.grade("1955", q)
    assert not quiz.grade("nineteen fifty five", q)


def test_alternates_and_near_misses():
    q = {"q": "conference?", "answer": "the Big West Conference",
         "accept": ["big west"]}
    assert quiz.grade("the big west", q)
    assert quiz.grade("big west conference", q)
    # A transcription slip on a long answer is still that answer.
    assert quiz.grade("big west conferents", q)
    # Where they used to play is not where they play.
    assert not quiz.grade("the WAC", q)


def test_multiple_choice_by_letter_ordinal_or_text():
    q = {"q": "year?", "options": ["1985", "1998", "2004", "2011"],
         "answer": "1998"}
    assert quiz.grade("B", q)
    assert quiz.grade("b.", q)
    assert quiz.grade("I think it's B", q)
    assert quiz.grade("the second one", q)
    assert quiz.grade("nineteen ninety eight", q)
    assert not quiz.grade("A", q)
    assert not quiz.grade("the third one", q)


def test_a_wrong_letter_is_wrong_even_if_the_words_match_elsewhere():
    q = {"q": "motto?", "options": ["Live Your Purpose", "Lead With Faith"],
         "answer": "Live Your Purpose"}
    assert not quiz.grade("option b", q)


# --- Playing ---------------------------------------------------------------


def test_a_round_scores_and_gets_a_title():
    started = quiz.start(count=3, seed=1)
    assert started["total"] == 3
    assert quiz.active()
    questions = quiz.current()["questions"]

    result = None
    for question in questions:
        result = quiz.answer(question["answer"])
        assert result["correct"], question["q"]
    assert result["finished"]
    assert result["score"] == 3
    assert result["title"] == "an honorary campus tour guide"
    assert "claim_code" not in result and "reward" not in result
    assert not quiz.active()

    logged = [json.loads(line) for line in quiz.RESULTS.read_text().splitlines()]
    assert logged[-1]["score"] == 3
    assert logged[-1]["title"] == result["title"]


def test_a_perfect_five_is_a_legend():
    quiz.start(count=5, seed=2)
    for question in list(quiz.current()["questions"]):
        result = quiz.answer(question["answer"])
    assert result["title"] == "a CBU trivia legend"
    assert "perfect" in result["detail"].lower()


def test_a_bad_score_still_gets_a_title_and_no_prize_talk():
    quiz.start(count=3, seed=3)
    for _ in range(3):
        result = quiz.answer("no idea")
    assert result["finished"]
    assert result["title"] == "a lost freshman on their first day"
    for word in ("discount", "keychain", "code"):
        assert word not in result["detail"].lower()


def test_quitting_ends_it():
    quiz.start(count=3, seed=4)
    result = quiz.answer("I give up")
    assert result["quit"]
    assert not quiz.active()


def test_a_stale_quiz_is_over(monkeypatch):
    quiz.start(count=3, seed=5)
    state = quiz._read()
    state["updated"] -= quiz.IDLE_SECONDS + 1
    quiz._write(state)
    assert not quiz.active()


# --- Routing ---------------------------------------------------------------


def test_asking_for_a_quiz_starts_one():
    for line in ("quiz me on CBU", "can you quiz me", "give me some CBU trivia",
                 "let's play trivia", "test me"):
        assert server.wants_quiz(line), line
        assert dict(server.prefetch_plan(line)) == {"quiz": {"action": "start"}}


def test_ordinary_questions_do_not_start_one():
    for line in ("what classes do I take in BASAI", "what's the weather",
                 "nod at me", "hello"):
        assert not server.wants_quiz(line), line
        assert "quiz" not in dict(server.prefetch_plan(line))


def test_everything_said_during_a_quiz_is_an_answer():
    quiz.start(count=3, seed=6)
    for line in ("Riverside", "nineteen fifty", "B", "what day is it today"):
        plan = dict(server.prefetch_plan(line))
        assert plan["quiz"] == {"action": "answer", "answer": line}, line


def test_asking_again_mid_quiz_restarts_rather_than_grades():
    quiz.start(count=3, seed=7)
    assert dict(server.prefetch_plan("quiz me again")) == {
        "quiz": {"action": "start"}}


# --- Addressing ------------------------------------------------------------


def test_quiz_answers_do_not_need_his_name():
    # "Riverside" is not question-shaped and never will be.
    for line in ("Riverside", "East Campus, Riverside.", "B", "nineteen fifty"):
        ok, why = server.is_addressed(line, False, mode="wake_word",
                                      attentive=True, quiz_running=True)
        assert ok, (line, why)
        assert why == "quiz answer"


def test_outside_the_window_the_quiz_still_needs_a_name():
    ok, _ = server.is_addressed("Riverside", False, mode="wake_word",
                                attentive=False, quiz_running=True)
    assert not ok
    ok, _ = server.is_addressed("Lancer, Riverside", False, mode="wake_word",
                                attentive=False, quiz_running=True)
    assert ok


def test_asking_for_the_quiz_counts_as_addressing_him():
    # Dropped as "not-a-question" in the room: no name, no question mark.
    for line in ("Start the quiz for CBU trivia.",
                 "quiz me", "Enter, start the quiz."):
        ok, why = server.is_addressed(line, True, mode="ambient")
        assert ok, (line, why)


def test_a_long_sentence_opening_with_and_then_is_not_go_on(monkeypatch):
    import time
    monkeypatch.setitem(server._remainder, "text", "the rest of an answer")
    monkeypatch.setitem(server._remainder, "at", time.time())
    assert server.wants_continuation("and then?")
    assert not server.wants_continuation(
        "And then we're time to do it. This guy's with teaching, with the "
        "half of the class. Give this guy a quiz on CBU trivia.")


def test_the_keychain_is_gone():
    # The quiz is just for fun now; nothing about it mentions a prize.
    assert not server.wants_quiz("how do I get a keychain")
    assert "keychain" not in server.QUIZ_PROMPT.lower()
    assert "titles" in quiz.load_questions() and "rewards" not in quiz.load_questions()


# --- SAE questions ---------------------------------------------------------


def _sae():
    return [q for q in quiz.load_questions()["questions"] if q.get("topic") == "sae"]


def test_there_are_sae_questions_and_each_grades_its_own_answer():
    questions = _sae()
    assert len(questions) >= 10
    for q in questions:
        assert quiz.grade(q["answer"], q), q["q"]
        for alt in q.get("accept", []):
            assert quiz.grade(alt, q), (q["q"], alt)
        if q.get("options"):
            assert q["answer"] in q["options"], q["q"]
            right = "abcd"[q["options"].index(q["answer"])]
            assert quiz.grade(right, q), q["q"]
            wrong = next(o for o in q["options"] if o != q["answer"])
            assert not quiz.grade(wrong, q), (q["q"], wrong)


def test_how_people_actually_say_the_sae_answers():
    by_q = {q["q"]: q for q in _sae()}

    def q(start):
        return next(v for k, v in by_q.items() if k.startswith(start))

    assert quiz.grade("charger six", q("What is the name of CBU Motorsports' newest"))
    assert quiz.grade("I think it was E-Lancer One", q("What was the name of CBU's very first"))
    assert quiz.grade("four thirty two", q("How many battery cells")) or \
        quiz.grade("C", q("How many battery cells"))
    assert quiz.grade("drag reduction system", q("What does DRS"))
    assert quiz.grade("twenty twenty four", q("In what year did CBU's combustion"))
    assert not quiz.grade("Daytona", q("Every June, where"))


def test_an_sae_quiz_only_asks_sae_questions():
    quiz.start(count=5, seed=7, topic="sae")
    assert all(x.get("topic") == "sae" for x in quiz.current()["questions"])
    quiz.start(count=5, seed=7, topic="cbu")
    assert all(not x.get("topic") for x in quiz.current()["questions"])
    quiz.quit()


def test_asking_for_sae_trivia_picks_the_topic():
    for line in ("quiz me on SAE", "give me some race car trivia",
                 "Lancer, quiz me about CBU Motorsports"):
        assert dict(server.prefetch_plan(line))["quiz"] == {
            "action": "start", "topic": "sae"}, line
    assert dict(server.prefetch_plan("quiz me"))["quiz"] == {"action": "start"}
