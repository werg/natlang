---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Extract checkable statements about inputs, outputs, defaults, errors, and supported workflows from documentation and examples. Preserve version and audience scope, distinguishing normative promises from illustrative suggestions or historical notes.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
