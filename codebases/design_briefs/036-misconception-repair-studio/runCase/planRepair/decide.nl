---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Select a suitable explanation and contrastive examples, then create practice with gradually reduced scaffolding. Adapt to observed responses without simply repeating the original answer, and escalate persistent ambiguity to the instructor.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
