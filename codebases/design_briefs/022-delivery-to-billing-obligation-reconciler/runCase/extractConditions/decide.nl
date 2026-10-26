---
args: { prepared: PreparedBillingCondition, goal: GoalContract, policy: PolicySelection }
returns: DecisionsBillingCondition
---
1. Apply this domain algorithm to each relevant evidence unit: Read contract schedules and amendments to identify triggers, acceptance requirements, partial billing rules, and exclusions. Distinguish work performed from the specific event authorizing billing and retain conditions that apply only to particular deliverables.
2. Return field-level atoms for the actual BillingCondition record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
