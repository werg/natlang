---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Interpret local names and descriptive phrases against project terminology and context. Retain multiple plausible identities when evidence is insufficient, and avoid forcing unusual observations into familiar categories merely to complete the dataset.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
