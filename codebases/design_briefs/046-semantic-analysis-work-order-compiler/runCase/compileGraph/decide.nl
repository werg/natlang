---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Construct a typed DAG of exact filters, joins, bounded semantic classification, temporal matching, and aggregation. Push exact restrictive filters before semantic work when meaning is preserved; attach provenance and missing-data rules to every edge and reject unsupported operations.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
