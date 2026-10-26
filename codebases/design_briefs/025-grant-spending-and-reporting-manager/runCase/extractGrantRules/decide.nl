---
args: { prepared: PreparedGrantCondition, goal: GoalContract, policy: PolicySelection }
returns: DecisionsGrantCondition
---
1. Apply this domain algorithm to each relevant evidence unit: Parse the agreement, budget schedules, and amendments into spending, milestone, and reporting conditions. Distinguish required outcomes from aspirational impact language, and preserve restrictions that apply only to specific funding tranches or time periods.
2. Return field-level atoms for the actual GrantCondition record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
