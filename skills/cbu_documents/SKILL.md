---
name: cbu_documents
description: Use for every question about California Baptist University - degree programs, course plans, units, general education and non-course requirements, transfer and AP credit policy, department policies, variances, CSDS and engineering material. The only acceptable source for a CBU fact. Triggers on - CBU, California Baptist, BASAI, MSAI, CSDS, major, degree, course, class, units, requirement, prerequisite, transfer, catalog, variance, internship, graduate.
---

# CBU documents

Hybrid retrieval over the university's own material: dense vector search and
BM25, fused with Reciprocal Rank Fusion, then reranked with a cross-encoder.
Measured at 87% top-1 and 100% in-context on a 15-case benchmark
(`eval_retrieval.py`), which is why this is the source of record rather than
anything the model remembers.

## Running it

```
echo '{"question":"What classes do I take in year one fall of BASAI?"}' | python3 run.py
```

Pass the question roughly as the student asked it. Keep campus vocabulary
intact -- BASAI, MSAI, CSDS, course codes like CSCI 110 -- because those are
exact lexical matches and they are what the BM25 half of the search keys on.
Paraphrasing them away measurably degrades the result.

## Reading the result

- `found: false` means nothing in the corpus covers it. Say you do not have it
  and point them at their advisor. Do not answer anyway.
- `passages` are the source text. Every number, course code, name and
  requirement in your answer must appear in them verbatim. If you want to say
  "about 120 units" and the passages do not contain 120, you do not know that.
- `sources` names the documents the passages came from, for citation.

## The rule that matters

There is a grounding check downstream that compares your answer's specifics
against these passages and refuses the answer if they are not there. Inventing
a plausible-sounding course code does not get past it; it just costs the
student their answer. Say less and stay inside the passages.
