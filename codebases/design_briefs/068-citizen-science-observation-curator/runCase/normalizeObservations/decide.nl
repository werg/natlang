---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Parse field notes and structured submissions while preserving original wording, location precision, and observation method. Distinguish what was directly seen from inferred identity or cause, and flag missing metadata without inventing exact coordinates or times.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
