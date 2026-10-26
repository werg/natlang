---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Match tasks to volunteers or specialists using stated skills, verified qualifications where required, availability, and consent. Do not infer capability from identity or enthusiasm, and preserve supervision requirements for trainees.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
