---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Reconcile site notes, messages, and drawing annotations to identify the actual requested action and its issuer. Distinguish clarification, proposed change, and authorized instruction; preserve conflicting versions and do not infer commercial authority from site presence.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
