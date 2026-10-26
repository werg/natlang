---
args: { prepared: PreparedSupplierOffer, goal: GoalContract, policy: PolicySelection }
returns: DecisionsSupplierOffer
---
1. Apply this domain algorithm to each relevant evidence unit: Reconcile quotations into comparable units and scopes while preserving bundles, exclusions, and conditional discounts. Distinguish an indicative quote from a binding offer and expose missing service or logistics costs rather than assuming comparability.
2. Return field-level atoms for the actual SupplierOffer record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
