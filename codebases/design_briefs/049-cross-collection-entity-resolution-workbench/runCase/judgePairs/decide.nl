---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Compare names, attributes, chronology, and context while distinguishing identity from affiliation or succession. Treat missing attributes as unknown rather than matching, and record both positive support and hard cannot-link evidence.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
