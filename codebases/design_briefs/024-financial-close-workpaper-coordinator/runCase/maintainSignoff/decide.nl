---
args: { prepared: PreparedWorkpaperSignoff, goal: GoalContract, policy: PolicySelection }
returns: DecisionsWorkpaperSignoff
---
1. Apply this domain algorithm to each relevant evidence unit: Bind sign-off to exact source and conclusion versions. When a late journal or source correction arrives, trace affected assertions and invalidate only dependent approvals; preserve the prior signed workpaper as historical evidence.
2. Return field-level atoms for the actual WorkpaperSignoff record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
