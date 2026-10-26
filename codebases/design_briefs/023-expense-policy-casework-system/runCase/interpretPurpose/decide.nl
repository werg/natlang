---
args: { prepared: PreparedPurposeAllocation, goal: GoalContract, policy: PolicySelection }
returns: DecisionsPurposeAllocation
---
1. Apply this domain algorithm to each relevant evidence unit: Identify mixed business and personal components using explicit facts and supplied allocation rules. Request a targeted explanation when purpose changes eligibility, and avoid inferring personal behavior from merchant stereotypes or unrelated account history.
2. Return field-level atoms for the actual PurposeAllocation record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
