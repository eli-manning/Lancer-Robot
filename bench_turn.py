"""
Time a streaming turn the way a listener experiences it.

Total latency is the wrong number for a robot that talks: what matters is how
long someone stands in silence before Lancer starts saying anything. That is
time-to-first-sentence, and it is the number this prints.

    python bench_turn.py                 # the default question set
    python bench_turn.py "your question"
"""

import json
import sys
import time

import requests

SERVER = "http://127.0.0.1:7870"

QUESTIONS = [
    "What classes do I take in year one fall of BASAI?",
    "How many internship hours do I need to graduate?",
    "What day is it today?",
    "What is the weather like right now?",
    "Who was Alan Turing?",
    "Tell me a joke about computer science majors.",
]


def run(question: str, speak: bool = False) -> dict:
    started = time.monotonic()
    first_sentence = None
    sentences = 0
    skills: list[str] = []
    answer = ""

    response = requests.post(
        f"{SERVER}/chat/stream",
        params={"speak": str(speak).lower()},
        json={"query": question},
        stream=True,
        timeout=300,
    )
    response.raise_for_status()
    for line in response.iter_lines():
        if not line:
            continue
        frame = json.loads(line)
        if frame["type"] == "sentence":
            sentences += 1
            if first_sentence is None:
                first_sentence = time.monotonic() - started
        elif frame["type"] == "status" and frame.get("state") == "skill":
            skills.append(frame["skill"])
        elif frame["type"] == "done":
            answer = frame["answer"]

    return {
        "question": question,
        "first_sentence_s": first_sentence,
        "total_s": time.monotonic() - started,
        "sentences": sentences,
        "skills": skills,
        "answer": answer,
    }


def main() -> int:
    questions = sys.argv[1:] or QUESTIONS
    speak = "--speak" in questions
    questions = [q for q in questions if not q.startswith("--")] or QUESTIONS

    rows = []
    for question in questions:
        result = run(question, speak)
        rows.append(result)
        first = result["first_sentence_s"]
        print(f"\n  {question}")
        print(f"    first sentence {first:.2f}s   total {result['total_s']:.2f}s"
              f"   sentences {result['sentences']}"
              f"   skills {result['skills'] or '-'}"
              if first else "    no sentences produced")
        print(f"    {result['answer'][:150]}")

    timed = [r["first_sentence_s"] for r in rows if r["first_sentence_s"]]
    if timed:
        print(f"\n  mean time-to-first-sentence {sum(timed)/len(timed):.2f}s"
              f"   worst {max(timed):.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
