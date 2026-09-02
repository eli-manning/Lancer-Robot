"""
Compare candidate models on the turns Lancer actually takes.

The brief proposed qwen2.5-coder:32b. This measures the alternatives on the
number that decides it -- how long someone stands there before the robot starts
talking -- and prints the answers next to each other so the quality half of the
decision is made by reading rather than by assertion.

Run against a live server:

    python bench_models.py
    python bench_models.py qwen2.5:14b qwen2.5:32b
"""

import statistics
import sys
import time

import requests

from bench_turn import run

SERVER = "http://127.0.0.1:7870"
DEFAULT_MODELS = ["qwen2.5:14b", "qwen2.5:32b"]

# A spread of routes, because they cost different amounts: grounded retrieval
# carries ~2,000 tokens of context, a skill call adds a whole extra round trip,
# and a social reply never reaches the model at all.
QUESTIONS = [
    ("grounded", "What classes do I take in year one fall of BASAI?"),
    ("grounded", "How many internship hours do I need to graduate?"),
    ("grounded", "Can I transfer credits from a community college?"),
    ("skill", "What is the weather like right now?"),
    ("skill", "What day is it today?"),
    ("general", "Who was Alan Turing?"),
    ("creative", "Tell me a joke about computer science majors."),
    ("refusal", "Who is the university president and what is their phone number?"),
]


def set_model(name: str) -> None:
    response = requests.post(f"{SERVER}/api/config",
                             json={"changes": {"llm_model": name}}, timeout=15)
    response.raise_for_status()


def warm(name: str) -> float:
    """Load the model into memory and return how long that took."""
    started = time.monotonic()
    requests.post(f"{SERVER}/chat/stream", json={"query": "Say hello."},
                  timeout=600, stream=True).content
    return time.monotonic() - started


def main() -> int:
    models = sys.argv[1:] or DEFAULT_MODELS
    original = requests.get(f"{SERVER}/api/config",
                            timeout=10).json()["config"]["llm_model"]
    results: dict[str, list[dict]] = {}

    try:
        for model in models:
            print(f"\n=== {model} ===")
            set_model(model)
            print(f"  warm-up (includes load): {warm(model):.1f}s")
            rows = []
            for route, question in QUESTIONS:
                result = run(question)
                result["route"] = route
                rows.append(result)
                first = result["first_sentence_s"] or float("nan")
                print(f"  {route:9s} first {first:5.2f}s  total "
                      f"{result['total_s']:5.2f}s  {question[:44]}")
                print(f"            {result['answer'][:130]}")
            results[model] = rows
    finally:
        set_model(original)
        print(f"\nRestored model: {original}")

    print("\n\n  route      " + "".join(f"{m:>22s}" for m in models))
    print("  " + "-" * (11 + 22 * len(models)))
    for index, (route, question) in enumerate(QUESTIONS):
        cells = ""
        for model in models:
            row = results[model][index]
            first = row["first_sentence_s"]
            cells += f"{first:>13.2f}s first" if first else f"{'-':>22s}"
        print(f"  {route:9s}" + cells + f"   {question[:38]}")

    print()
    for model in models:
        firsts = [r["first_sentence_s"] for r in results[model]
                  if r["first_sentence_s"]]
        totals = [r["total_s"] for r in results[model]]
        print(f"  {model:16s} median first sentence "
              f"{statistics.median(firsts):5.2f}s   "
              f"median total {statistics.median(totals):5.2f}s   "
              f"worst first {max(firsts):5.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
