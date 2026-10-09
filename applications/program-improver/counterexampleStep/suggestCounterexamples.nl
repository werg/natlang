---
args:
  goal: string
  evidence: unknown[]
  maxSuggestions?: number
returns: CounterexampleSuggestions
---
Suggest concrete deployment inputs that distinguish a plausible faulty implementation from the requested behavior.

1. Read goal and the training failures in evidence.
2. Name the boundary or interaction that the failures suggest is untested.
3. Write at most maxSuggestions argument lists (four when it is absent) that exercise it, in inputs.
4. Write reason: one sentence on what the inputs probe.

Expected outcomes belong to the independent oracle, which checks the inputs; return the inputs alone.
