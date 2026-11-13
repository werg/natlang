---
description: Whether one observation supports or contradicts the criterion, or leaves it uncertain.
args:
  observation: Untrusted<string>
  criterion: string
returns: '"supported" | "contradicted" | "uncertain"'
readout: decision
---
Decide whether the observation supports or contradicts the criterion.
If it is irrelevant, ambiguous, or insufficient to decide, the answer is uncertain. Do not infer missing measurements.
