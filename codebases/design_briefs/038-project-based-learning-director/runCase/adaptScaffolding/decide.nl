---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Interpret progress artifacts to identify where support is needed without doing the substantive work for learners. Provide targeted examples, prompts, or task decomposition, and remove support when evidence shows independent capability.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
