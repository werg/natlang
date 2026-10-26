---
args: { prepared: PreparedAgreementScope, goal: GoalContract, policy: PolicySelection }
returns: DecisionsAgreementScope
---
1. Apply this domain algorithm to each relevant evidence unit: Resolve the applicable contract, schedules, amendments, and purchasing entity. Interpret coverage under dates, territory, and intended use; do not apply a parent agreement to an affiliate or service category without supporting terms.
2. Return field-level atoms for the actual AgreementScope record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
