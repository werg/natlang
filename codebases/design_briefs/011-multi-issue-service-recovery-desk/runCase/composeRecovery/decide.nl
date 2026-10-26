---
args: { prepared: PreparedRemedyBundle, goal: GoalContract, policy: PolicySelection }
returns: DecisionsRemedyBundle
---
1. Apply this domain algorithm to each relevant evidence unit: Construct alternative bundles that jointly solve the issues, accounting for dependencies and customer preferences. Detect double compensation for the same loss, but do not erase separate entitlements merely because they share one order.
2. Return field-level atoms for the actual RemedyBundle record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
