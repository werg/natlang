---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Compare reports by location, asset dependencies, timing, and symptom mechanism. Group likely shared issues for coordinated investigation while retaining unrelated complaints that use similar language; do not close a new report solely because a nearby task was completed.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
