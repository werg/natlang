---
args: { prepared: PreparedScopedClaim, goal: GoalContract, policy: PolicySelection }
returns: DecisionsScopedClaim
---
1. Apply this domain algorithm to each relevant evidence unit: Extract independently checkable claims with qualifications and page context. Resolve entity references conservatively, attach author and effective period, and keep instructions separate from descriptions of historical behavior or quoted third-party views.
2. Return field-level atoms for the actual ScopedClaim record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
