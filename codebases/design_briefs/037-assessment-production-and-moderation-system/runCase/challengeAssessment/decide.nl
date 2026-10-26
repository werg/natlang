---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Independently solve each item, attempt reasonable alternate readings, and test rubric consistency on borderline responses. Repair items and rubrics together, then rerun coverage and duration checks before releasing a frozen assessment edition.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
