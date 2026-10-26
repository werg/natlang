---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Process new evidence and reviewer corrections as reversible identity changes. Trace downstream uses, prepare split or merge migrations, and preserve old identifiers as redirects with explicit validity rather than silently changing historical joins.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
