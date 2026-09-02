---
name: play_emotion
description: Express something with the robot's body - nod, shake head, look curious, laugh, celebrate, greet. Use alongside speech to react physically. Triggers on - nod, agree, disagree, react, gesture, look happy, look confused, celebrate, say hello, dance.
---

# Play an emotion

Plays one of ~85 recorded moves from
`pollen-robotics/reachy-mini-emotions-library` through the robot daemon. The
moves are animations recorded on a real Reachy Mini, which is why they read as
alive rather than as a servo sweep.

## Running it

```
echo '{"emotion":"curious"}' | python3 run.py
```

Plain words are mapped onto library names, so `happy` finds `cheerful1` and
`nod` finds `yes1`. Exact names work too. If nothing matches, the result lists
what is available rather than failing silently.

## When to use it

Alongside an answer, not instead of one. A nod while confirming, a confused
tilt when a question does not parse, a welcome when someone walks up. It is the
difference between a robot that talks and a speaker that talks.

Do not announce it. "Let me nod at you" is worse than nodding.

## What it costs

Nothing to speak over: the move plays on the robot while the answer streams, so
it does not delay speech. It does take over the head for a second or two, which
means face tracking pauses for that long.

## Vocabulary

Roughly: yes, no, curious, attentive, thoughtful, confused, uncertain, amazed,
surprised, laughing, cheerful, enthusiastic, proud, grateful, helpful, success,
relief, serenity, welcoming, shy, lost, tired, boredom, impatient, irritated,
frustrated, furious, rage, sad, downcast, lonely, anxiety, fear, scared,
disgusted, contempt, displeased, resigned, exhausted, oops, dance1-3.
