---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Read occupant descriptions to identify location, symptom, timing, and practical impact. Distinguish observed behavior from guessed causes, resolve ambiguous room names, and preserve access or communication preferences without inferring sensitive personal circumstances.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
