---
args: { prepared: PreparedBillingPacket, goal: GoalContract, policy: PolicySelection }
returns: DecisionsBillingPacket
---
1. Apply this domain algorithm to each relevant evidence unit: Assemble a customer-readable explanation and internal approval packet linking each amount to satisfied conditions. Recheck late amendments and acceptance changes before release, then reconcile the actual invoice identifier back to consumed obligations.
2. Return field-level atoms for the actual BillingPacket record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
