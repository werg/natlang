---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Select a complication appropriate to the learner’s progress and learning objective. Ensure it changes a meaningful decision, preserves a safe recovery path, and does not retroactively invalidate an action that was correct under earlier visible facts.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
