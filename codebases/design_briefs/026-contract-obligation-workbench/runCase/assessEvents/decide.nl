---
args: { prepared: PreparedTriggerFinding, goal: GoalContract, policy: PolicySelection }
returns: DecisionsTriggerFinding
---
1. Apply this domain algorithm to each relevant evidence unit: Compare incoming operational events with contractual triggers and conditions. Distinguish an allegation that a trigger occurred from established evidence, and request the smallest missing fact when activation would materially change a duty or deadline.
2. Return field-level atoms for the actual TriggerFinding record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
