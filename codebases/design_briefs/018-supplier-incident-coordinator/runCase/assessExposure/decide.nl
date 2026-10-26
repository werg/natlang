---
args: { prepared: PreparedExposureFinding, goal: GoalContract, policy: PolicySelection }
returns: DecisionsExposureFinding
---
1. Apply this domain algorithm to each relevant evidence unit: Trace potentially affected lots and orders through authoritative supply records. Distinguish confirmed exposure from precautionary scope, identify downstream commitments, and ask for missing traceability evidence before narrowing a containment boundary.
2. Return field-level atoms for the actual ExposureFinding record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
