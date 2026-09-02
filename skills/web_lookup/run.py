"""
Wikipedia lookup.

Two hops on purpose: search first, then fetch the summary of the top title.
Going straight to the summary endpoint with the user's phrasing 404s on
anything that is not already an exact article title, which is most of what
someone says out loud.

Keyless, and the summary endpoint returns a couple of sentences rather than an
article -- which is roughly the length Lancer can say aloud anyway.
"""

import json
import sys
import urllib.parse

import requests

SEARCH = "https://en.wikipedia.org/w/api.php"
SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"
# Wikimedia asks for a descriptive User-Agent and rate-limits generic ones.
HEADERS = {"User-Agent": "LancerCampusRobot/1.0 (CBU ACM student chapter)"}


def main() -> int:
    try:
        args = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        args = {}

    topic = str(args.get("topic") or "").strip()
    if not topic:
        json.dump({"ok": False, "error": "no topic given"}, sys.stdout)
        return 0

    try:
        found = requests.get(
            SEARCH,
            params={"action": "query", "list": "search", "srsearch": topic,
                    "srlimit": 1, "format": "json"},
            headers=HEADERS,
            timeout=8,
        )
        found.raise_for_status()
        hits = found.json().get("query", {}).get("search", [])
        if not hits:
            json.dump({"ok": True, "found": False, "topic": topic,
                       "note": "Wikipedia has no article matching that."},
                      sys.stdout)
            return 0

        title = hits[0]["title"]
        page = requests.get(
            SUMMARY + urllib.parse.quote(title.replace(" ", "_"), safe=""),
            headers=HEADERS,
            timeout=8,
        )
        page.raise_for_status()
        summary = page.json()
    except Exception as e:
        json.dump({"ok": False, "error": f"lookup failed: {e}"}, sys.stdout)
        return 0

    json.dump({
        "ok": True,
        "found": True,
        "title": summary.get("title", title),
        "summary": (summary.get("extract") or "")[:1200],
        "source": "Wikipedia",
        "url": (summary.get("content_urls", {})
                .get("desktop", {}).get("page", "")),
    }, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
