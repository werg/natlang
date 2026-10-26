---
args: { prepared: PreparedStudyObservation, goal: GoalContract, policy: PolicySelection }
returns: DecisionsStudyObservation
---
1. Apply this domain algorithm to each relevant evidence unit: Apply eligibility to available evidence, escalating insufficient abstracts to full-text review. Extract outcomes, methods, and limitations with source locations; keep author interpretation separate from directly reported observations and exact numerical results.
2. Return field-level atoms for the actual StudyObservation record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
