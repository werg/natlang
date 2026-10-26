---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Inspect candidate records for task relevance, data quality, provenance, consent, and leakage risks. Distinguish missing coverage from invalid records, and quarantine material whose rights or source integrity cannot be established.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
