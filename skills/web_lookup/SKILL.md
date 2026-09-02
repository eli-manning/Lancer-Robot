---
name: web_lookup
description: Use to look up general knowledge on Wikipedia - people, places, history, science, technology, definitions. NOT for California Baptist University questions, which come from cbu_documents. Triggers on - who was, what is, tell me about, look up, explain, history of.
---

# Wikipedia lookup

Searches Wikipedia and returns a short summary of the best-matching article.

## Running it

```
echo '{"topic":"Alan Turing"}' | python3 run.py
```

`topic` must be a short noun phrase, not the whole spoken question. "Alan
Turing", not "hey can you tell me who Alan Turing was". The search matches
titles; a conversational sentence matches nothing useful.

## Reading the result

- `found: false` means Wikipedia has no matching article. Say you do not know.
  Do not fill the gap from memory -- that is the failure this skill exists to
  prevent.
- `summary` is a couple of paragraphs at most. Compress it to two or three
  spoken sentences rather than reading it out.
- `title` is the article actually found, which may not be what was asked for.
  Check it matches the intent before answering; a search for a person can land
  on a film of the same name.

## Not for CBU

Wikipedia's article on California Baptist University is not the university's
own catalog, and using it for degree requirements produces confident, wrong
answers about someone's graduation. Use `cbu_documents` for anything about CBU.
