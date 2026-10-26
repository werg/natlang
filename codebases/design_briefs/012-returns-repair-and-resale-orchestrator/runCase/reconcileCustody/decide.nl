---
args: { prepared: PreparedCustodyEvent, goal: GoalContract, policy: PolicySelection }
returns: DecisionsCustodyEvent
---
1. Apply this domain algorithm to each relevant evidence unit: Track collection, receipt, repair transfer, and return as distinct custody events. Compare condition at each handoff and isolate new discrepancies rather than attributing all damage to the customer or overwriting the original inspection.
2. Return field-level atoms for the actual CustodyEvent record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
