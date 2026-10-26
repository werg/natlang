---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Read each bid, qualification, schedule, and addendum as a package. Extract functional scope rather than relying on trade headings, distinguish allowances from firm inclusions, and retain conditional exclusions that affect interfaces with other bidders.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
