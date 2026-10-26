---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Compare observer, time, location, media hashes, and narrative context to identify repeated submissions of one event. Preserve independent observations of the same phenomenon and avoid merging nearby events solely because they share a likely identity.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
