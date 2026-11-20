"""The CBU trivia quiz. State and grading live in lancer_quiz; this is the door."""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

import lancer_quiz as quiz                          # noqa: E402


def main() -> int:
    try:
        args = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        args = {}

    action = str(args.get("action") or "status").strip().lower()
    try:
        if action == "start":
            result = quiz.start()
            result["detail"] = (
                f"Quiz started: {result['total']} questions about CBU, just for "
                "fun -- there is no prize, only bragging rights and a title at "
                "the end. Say so in one short, playful line, then ask question "
                "1 exactly as written in next.spoken.")
        elif action == "answer":
            result = quiz.answer(str(args.get("answer") or ""))
        elif action == "quit":
            result = quiz.quit()
        else:
            result = quiz.status()
    except FileNotFoundError as e:
        result = {"ok": False, "error": f"quiz questions missing: {e}"}
    except ValueError as e:
        result = {"ok": False, "error": str(e)}

    json.dump(result, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
