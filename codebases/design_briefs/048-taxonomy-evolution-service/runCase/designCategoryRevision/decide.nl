---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Draft revised definitions, inclusion and exclusion rules, and parent-child relationships. Preserve stable category IDs where meaning remains stable, create new identities when meaning changes, and explicitly map split or merged historical categories.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
