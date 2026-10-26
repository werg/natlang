---
args: { prepared: PreparedDeliveryEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionsDeliveryEvidence
---
1. Apply this domain algorithm to each relevant evidence unit: Relate project artifacts, service logs, and customer messages to the promised deliverables. Identify partial or alternative delivery and distinguish a draft shared for comment from an accepted contractual handover.
2. Return field-level atoms for the actual DeliveryEvidence record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
