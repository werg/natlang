---
description: Judge one observation against the caller's criterion, without inventing evidence.
args:
  observation: Untrusted<string>
  criterion: string
returns: Assessment
---
Decide the verdict with verdict, then return it with a brief reason grounded in the observation.
Do not infer missing measurements.
