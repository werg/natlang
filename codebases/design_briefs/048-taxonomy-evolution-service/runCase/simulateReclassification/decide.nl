---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Reclassify only potentially affected records under baseline and candidate definitions, compare changes, and inspect boundary cases. Build explicit bridges for historical reporting and retain cases that cannot be mapped uniquely after a split.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
