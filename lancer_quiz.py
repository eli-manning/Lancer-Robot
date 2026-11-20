"""
The CBU trivia quiz: a few questions, a score, and a silly title. Just for fun.

Most of what Lancer does is one turn long. A quiz is the first thing that
spans several: he asks, they answer, he grades and asks the next one, and the
whole time the words coming in -- "1950", "Riverside", "B" -- look nothing
like questions. Sent through the usual routing they would go hunting for a
course plan about Riverside. So the state of the quiz lives here, and while
one is running the server sends every utterance to it as an answer.

The store is a file for the same reason the timers' is: the skill that grades
an answer is a subprocess, and the server that decides whether a quiz is
running is not. The results log is just a record of how people did; there is
no prize, so nothing is redeemed against it.

Grading is deterministic and lives here rather than with the model. A model
told the answer is "1950" will pass "1955" on a generous day, and this is the
one place where being generous would make the score meaningless.
"""

from __future__ import annotations

import difflib
import json
import random
import re
import string
import time
from pathlib import Path
from typing import Any

BASE = Path(__file__).parent
STORE = BASE / "lancer_quiz.json"
RESULTS = BASE / "lancer_quiz_results.jsonl"
QUESTIONS = BASE / "skills" / "quiz" / "questions.json"

# A quiz nobody has answered in this long is over, whatever the file says.
# Otherwise someone walks off mid-quiz and the next person to say "hello" is
# told it was wrong.
IDLE_SECONDS = 300

# Things people say when they want out. Checked before grading, because
# "I give up" graded as an answer is both wrong and rude.
_QUIT_RE = re.compile(
    r"\b(stop|quit|give up|i'?m done|never ?mind|forget it|end (the|this) quiz|"
    r"cancel|no more|enough)\b", re.I,
)

# Multiple-choice by letter, in the ways people say a letter out loud.
_LETTER_RE = re.compile(
    r"^(?:(?:option|answer|letter|it'?s|is it|i think|maybe|probably|the)\s+)*"
    r"([a-d])\b[.?!]*$", re.I,
)
_ORDINALS = {"first": 0, "second": 1, "third": 2, "fourth": 3, "last": 3,
             "1st": 0, "2nd": 1, "3rd": 2, "4th": 3}

_WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
    "thousand": 1000,
}
_FILLER = {"the", "a", "an", "it", "its", "it's", "is", "was", "in", "of",
           "um", "uh", "i", "think", "guess", "maybe", "probably", "answer",
           "that", "would", "be", "they", "are", "called"}


# --- Questions ------------------------------------------------------------


def load_questions(path: Path = QUESTIONS) -> dict[str, Any]:
    """The question pool and the score titles, from the skill's data file."""
    data = json.loads(path.read_text())
    if not data.get("questions"):
        raise ValueError(f"{path} has no questions")
    return data


# --- Store ------------------------------------------------------------------


def _read() -> dict | None:
    try:
        return json.loads(STORE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _write(state: dict | None) -> None:
    if state is None:
        STORE.unlink(missing_ok=True)
        return
    tmp = STORE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state))
    tmp.replace(STORE)


def current() -> dict | None:
    """The running quiz, or None when there is not one (or it went stale)."""
    state = _read()
    if not state or state.get("finished"):
        return None
    if time.time() - float(state.get("updated", 0)) > IDLE_SECONDS:
        _write(None)
        return None
    return state


def active() -> bool:
    return current() is not None


# --- Playing ----------------------------------------------------------------


TOPICS = ("cbu", "sae")


def start(count: int | None = None, seed: int | None = None,
          topic: str | None = None) -> dict:
    """
    Begin a fresh quiz, throwing away any that was in progress.

    `topic` narrows the pool: "sae" for the CBU Motorsports questions, "cbu"
    for the campus ones (anything untagged). No topic mixes everything.
    """
    data = load_questions()
    pool = list(data["questions"])
    topic = (topic or "").strip().lower()
    if topic in TOPICS:
        pool = [q for q in pool if (q.get("topic") or "cbu") == topic] or pool
    count = int(count or data.get("questions_per_quiz", 5))
    rng = random.Random(seed)
    rng.shuffle(pool)
    chosen = pool[:max(1, min(count, len(pool)))]
    state = {
        "questions": chosen,
        "index": 0,
        "score": 0,
        "results": [],
        "started": time.time(),
        "updated": time.time(),
        "finished": False,
    }
    _write(state)
    return {
        "ok": True,
        "started": True,
        "total": len(chosen),
        "titles": data.get("titles", []),
        **_next_block(state),
    }


def answer(text: str) -> dict:
    """Grade one spoken answer against the current question and advance."""
    state = current()
    if state is None:
        return {"ok": False, "error": "no quiz running -- offer to start one"}

    if _QUIT_RE.search(text or ""):
        _write(None)
        return {"ok": True, "quit": True, "score": state["score"],
                "asked": state["index"], "total": len(state["questions"]),
                "detail": "they stopped the quiz. Be a good sport about it "
                          "and say they can play again any time."}

    question = state["questions"][state["index"]]
    correct = grade(text, question)
    state["results"].append({"question": question["q"], "heard": text,
                             "correct": correct})
    if correct:
        state["score"] += 1
    state["index"] += 1
    state["updated"] = time.time()

    result = {
        "ok": True,
        "correct": correct,
        "correct_answer": question["answer"],
        "score": state["score"],
        "asked": state["index"],
        "total": len(state["questions"]),
    }
    if question.get("fact"):
        result["fact"] = question["fact"]

    if state["index"] >= len(state["questions"]):
        state["finished"] = True
        _write(None)
        result.update(_finish(state))
    else:
        _write(state)
        result.update(_next_block(state))
    return result


def status() -> dict:
    state = current()
    if state is None:
        return {"ok": True, "running": False}
    return {"ok": True, "running": True, "score": state["score"],
            "asked": state["index"], "total": len(state["questions"]),
            **_next_block(state)}


def quit() -> dict:
    state = current()
    _write(None)
    if state is None:
        return {"ok": True, "quit": False, "detail": "no quiz was running"}
    return {"ok": True, "quit": True, "score": state["score"],
            "asked": state["index"], "total": len(state["questions"])}


def _next_block(state: dict) -> dict:
    question = state["questions"][state["index"]]
    block = {
        "number": state["index"] + 1,
        "total": len(state["questions"]),
        "question": question["q"],
    }
    if question.get("options"):
        block["options"] = question["options"]
        block["spoken"] = _spoken(question)
    else:
        block["spoken"] = question["q"]
    return {"next": block, "finished": False}


def _spoken(question: dict) -> str:
    """The question with its options, in a shape that reads aloud cleanly."""
    letters = "ABCD"
    options = ", ".join(f"{letters[i]}: {opt}"
                        for i, opt in enumerate(question["options"]))
    return f"{question['q']} {options}."


def _finish(state: dict) -> dict:
    data = load_questions()
    score, total = state["score"], len(state["questions"])
    title = title_for(score, data.get("titles", []))
    record = {
        "ts": time.time(),
        "score": score,
        "total": total,
        "title": title,
        "results": state["results"],
    }
    with RESULTS.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
    detail = (f"Quiz over: {score} of {total}. "
              + (f"Crown them {title} -- say it with some ceremony. "
                 if title else "")
              + ("A perfect round: make a big deal of it. "
                 if score == total else
                 "Keep it light and playful, never disappointed, and offer "
                 "another round if they want to beat it. ")
              + "There is no prize; it is just for fun.")
    return {"finished": True, "score": score, "total": total,
            "title": title, "detail": detail}


def title_for(score: int, tiers: list[dict]) -> str:
    """The best title the score reaches, or "" when none is set."""
    best = ""
    best_min = -1
    for tier in tiers:
        needed = int(tier.get("min_correct", 0))
        if score >= needed and needed > best_min:
            best, best_min = str(tier.get("title", "")), needed
    return best


# --- Grading ----------------------------------------------------------------


def normalize(text: str) -> str:
    """Lowercase, no punctuation, spoken numbers as digits, filler dropped."""
    lowered = str(text or "").lower().replace("-", " ")
    lowered = lowered.translate(str.maketrans("", "", string.punctuation
                                              .replace("'", "")))
    words = [w.strip("'") for w in lowered.split()]
    words = _digits(words)
    words = [w for w in words if w and w not in _FILLER]
    return " ".join(words)


def _digits(words: list[str]) -> list[str]:
    """
    Spoken numbers to digits, including years.

    Whisper gives "nineteen fifty" for 1950 about as often as the digits, and
    "nineteen" "fifty" is 19 and 50 unless the two are joined the way a year
    is said.
    """
    out: list[str] = []
    run: list[int] = []

    def flush() -> None:
        if not run:
            return
        # "nineteen fifty" -> 1950; "twenty twenty five" -> 2025;
        # "two thousand" -> 2000; "fifty five" -> 55.
        if len(run) >= 2 and run[0] in range(10, 100) and run[1] >= 10:
            value = run[0] * 100 + run[1] + sum(run[2:])
        else:
            value = 0
            group = 0
            for n in run:
                if n == 100:
                    group = (group or 1) * 100
                elif n == 1000:
                    value += (group or 1) * 1000
                    group = 0
                else:
                    group += n
            value += group
        out.append(str(value))
        run.clear()

    for word in words:
        if word in _WORD_NUMBERS:
            run.append(_WORD_NUMBERS[word])
        elif word == "and" and run:
            continue
        else:
            flush()
            out.append(word)
    flush()
    return out


def grade(heard: str, question: dict) -> bool:
    """
    Whether what they said counts.

    Exact after normalising, or the accepted answer appears whole inside what
    they said ("I think it's Riverside"), or it is close enough that the
    difference is a transcription slip rather than a different answer.
    """
    said = normalize(heard)
    if not said:
        return False

    accepted = [question["answer"], *question.get("accept", [])]
    options = question.get("options") or []

    # Multiple choice: a bare letter, an ordinal, or the option's own words.
    if options:
        letter = _LETTER_RE.match(heard.strip())
        index = None
        if letter:
            index = "abcd".index(letter.group(1).lower())
        else:
            for word, position in _ORDINALS.items():
                if re.search(rf"\b{word}\b", heard, re.I):
                    index = position
                    break
        if index is not None and index < len(options):
            return normalize(options[index]) == normalize(question["answer"])

    for candidate in accepted:
        target = normalize(candidate)
        if not target:
            continue
        if said == target:
            return True
        if re.search(rf"(^|\s){re.escape(target)}(\s|$)", said):
            return True
        # Fuzzy only for things long enough that a near-miss is a slip. Two
        # years that differ by one digit are not a slip, so numbers are exact.
        if len(target) >= 5 and not target.isdigit():
            if difflib.SequenceMatcher(None, said, target).ratio() >= 0.85:
                return True
    return False
