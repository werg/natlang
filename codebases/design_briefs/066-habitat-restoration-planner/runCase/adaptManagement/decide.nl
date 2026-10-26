---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Interpret new observations against baseline and monitoring rules, including alternative explanations such as seasonality or survey changes. Propose bounded plan revisions, preserve unaffected work, and obtain expert approval before changing interventions.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
