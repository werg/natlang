---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Construct a short diagnostic task varying the relevant conceptual feature while controlling unrelated difficulty. Include an independently checked answer space and accessible wording; avoid hints that reveal which hypothesis the system expects to confirm.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
