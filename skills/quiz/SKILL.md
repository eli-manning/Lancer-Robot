---
name: quiz
description: Use when someone asks to be quizzed, tested, or wants to play trivia. Runs a short, just-for-fun CBU trivia quiz and crowns them with a silly title for their score. No prize. Triggers on - quiz me, test me, trivia, play a game.
---

# CBU trivia quiz

Five questions drawn at random from `questions.json`, graded here rather than
by the model, and a title at the end that depends on the score. It is just
for fun: there is no prize.

## Running it

```
echo '{"action":"start"}' | python3 run.py
echo '{"action":"answer","answer":"nineteen fifty"}' | python3 run.py
echo '{"action":"status"}' | python3 run.py
echo '{"action":"quit"}' | python3 run.py
```

While a quiz is running the server routes every utterance to `answer` itself,
so in conversation the model only ever needs `start`.

## Reading the result

- `next.spoken` is the question with its options, ready to read out. Read it
  word for word. Do not paraphrase it, do not drop an option, and never hint
  at the answer.
- `correct`, `correct_answer` and `fact` describe the answer just graded. Say
  whether they got it; if they missed, say what it was.
- `finished: true` carries `score`, `total` and `title`. Give the score and
  crown them with the title. Keep a low score playful.
- `quit: true` means they asked to stop.

## Editing the questions

`questions.json` is the whole pool. Each entry has `q`, `answer`, an optional
`accept` list of other phrasings that count, optional `options` for multiple
choice (the answer must match one option exactly), and an optional `fact` to
say after grading. `titles` maps a minimum score to the title it earns.

Answers are normalised before comparison: case and punctuation are ignored,
spoken numbers become digits ("nineteen fifty" is 1950), and the accepted
answer only needs to appear inside what they said, so "I think it's
Riverside" passes. Years must match exactly.

## Results

Every finished quiz is appended to `lancer_quiz_results.jsonl` at the repo
root with its score and title.
