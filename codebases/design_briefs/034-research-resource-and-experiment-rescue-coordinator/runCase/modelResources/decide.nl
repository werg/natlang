---
args: { prepared: PreparedResourceCapability, goal: GoalContract, policy: PolicySelection }
returns: DecisionsResourceCapability
---
1. Apply this domain algorithm to each relevant evidence unit: Reconcile calendars, instrument status, maintenance records, and operator qualifications. Distinguish booked availability from actual readiness and identify resources that appear interchangeable but differ in capability or validated use.
2. Return field-level atoms for the actual ResourceCapability record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
