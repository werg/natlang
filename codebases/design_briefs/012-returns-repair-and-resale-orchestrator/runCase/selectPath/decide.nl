---
args: { prepared: PreparedReturnDisposition, goal: GoalContract, policy: PolicySelection }
returns: DecisionsReturnDisposition
---
1. Apply this domain algorithm to each relevant evidence unit: Apply purchase terms and approved product-handling rules to condition evidence and customer preference. Identify cases needing specialist inspection, compare repair and replacement options, and avoid promising a remedy before required checks are satisfied.
2. Return field-level atoms for the actual ReturnDisposition record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
