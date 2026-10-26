---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Compare failures with requirements, execution traces, and fixture health. Distinguish a product defect from a bad oracle, flaky setup, or unavailable dependency; propose the smallest repair without weakening a valid assertion to match defective behavior.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
