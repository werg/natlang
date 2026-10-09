---
description: Choose which search hits can answer the question.
args:
  question: string
  found: SearchResult
  problem?: string
returns: HitIds
---
Choose the hits of found that can answer question.

1. For each hit in found.hits, read its preview and decide whether it states something that bears on the question: an
   answer, a condition, a qualification or a contradiction.
2. Collect the id of each such hit, once.
3. When found.truncated is true, or when no hit bears on the question, return the ids you collected (possibly none);
   the gap stage reports the shortfall.
4. problem, when given, says why an earlier list was refused. Collect the ids again so that each one is the id of a hit
   in found.hits.
