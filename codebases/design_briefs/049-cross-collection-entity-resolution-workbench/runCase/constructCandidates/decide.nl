---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Generate candidate pairs using exact normalized keys, aliases, and bounded semantic retrieval. Preserve blocking provenance and include audit samples outside candidate blocks so recall failures are measurable rather than hidden by the candidate generator.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
