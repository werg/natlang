---
args: { prepared: PreparedPurchaseDisposition, goal: GoalContract, policy: PolicySelection }
returns: DecisionsPurchaseDisposition
---
1. Apply this domain algorithm to each relevant evidence unit: Build a review packet with scope evidence, pricing, alternatives, and unresolved conditions. Route to the correct authority based on cumulative commitment and exception type; recheck terms and request versions immediately before order release.
2. Return field-level atoms for the actual PurchaseDisposition record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
