---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Compare actual implemented configurations with approved applicability and acceptance criteria. Preserve authorized exceptions and unresolved field cases, then close the change only when required consumers and records reflect the correct revision state.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
