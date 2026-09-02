"""Play one of the robot's recorded emotion moves."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _robot import emotions, play_recorded          # noqa: E402

# What the model is likely to ask for, mapped onto what the library actually
# has. Without this it invents plausible names -- "happy", "excited", "nod" --
# and every one of them fails, which reads as the robot ignoring the request.
ALIASES = {
    "happy": "cheerful1", "joy": "cheerful1", "excited": "enthusiastic1",
    "enthusiastic": "enthusiastic1", "pleased": "proud1", "proud": "proud1",
    "sad": "sad1", "unhappy": "sad1", "disappointed": "downcast1",
    "confused": "confused1", "puzzled": "confused1", "unsure": "uncertain1",
    "curious": "curious1", "interested": "attentive1", "thinking": "thoughtful1",
    "thoughtful": "thoughtful1", "surprised": "surprised1", "amazed": "amazed1",
    "shocked": "surprised2", "laugh": "laughing1", "laughing": "laughing1",
    "funny": "laughing1", "yes": "yes1", "nod": "yes1", "agree": "yes1",
    "no": "no1", "shake": "no1", "disagree": "no1", "hello": "welcoming1",
    "hi": "welcoming1", "welcome": "welcoming1", "greeting": "welcoming1",
    "bye": "go_away1", "goodbye": "go_away1", "sorry": "oops1",
    "oops": "oops1", "mistake": "oops1", "tired": "tired1", "bored": "boredom1",
    "annoyed": "irritated1", "angry": "furious1", "frustrated": "frustrated1",
    "scared": "scared1", "afraid": "fear1", "worried": "anxiety1",
    "grateful": "grateful1", "thanks": "grateful1", "helpful": "helpful1",
    "success": "success1", "celebrate": "success1", "relief": "relief1",
    "calm": "serenity1", "shy": "shy1", "lost": "lost1", "waiting": "waiting",
    "understand": "understanding1", "got it": "understanding1",
    "sleep": "sleep1", "wake": "wake-mini-up", "dance": "dance1",
}


def main() -> int:
    try:
        args = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        args = {}

    wanted = str(args.get("emotion") or "").strip().lower().replace(" ", "_")
    if not wanted:
        json.dump({"ok": False, "error": "no emotion given"}, sys.stdout)
        return 0

    available = emotions()
    if not available:
        json.dump({"ok": False, "error": "robot unreachable or library missing"},
                  sys.stdout)
        return 0

    name = None
    if wanted in available:
        name = wanted
    elif ALIASES.get(wanted) in available:
        name = ALIASES[wanted]
    else:
        # "curious" should still find "curious1" even if the alias table misses.
        matches = [e for e in available if e.lower().startswith(wanted)]
        name = matches[0] if matches else None

    if name is None:
        json.dump({
            "ok": False,
            "error": f"no emotion called {wanted!r}",
            "try_one_of": sorted(available)[:40],
        }, sys.stdout)
        return 0

    ok, body = play_recorded(name)
    json.dump({"ok": ok, "played": name if ok else None,
               "requested": wanted, "detail": body if not ok else "playing"},
              sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
