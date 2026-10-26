---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Generate a small set of plausible explanations for the observed pattern, including task ambiguity and accidental error. Identify a question whose answers would distinguish these explanations instead of treating one wrong response as a stable misconception.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
