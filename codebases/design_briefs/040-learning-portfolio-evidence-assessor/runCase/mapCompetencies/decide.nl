---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Compare evidence with the actual competency definition and performance level. Recognize alternative demonstrations, avoid matching by keywords alone, and identify which facets are supported versus merely mentioned in a reflection.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
