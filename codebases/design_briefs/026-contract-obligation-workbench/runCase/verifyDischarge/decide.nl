---
args: { prepared: PreparedObligationStatus, goal: GoalContract, policy: PolicySelection }
returns: DecisionsObligationStatus
---
1. Apply this domain algorithm to each relevant evidence unit: Compare completed actions with the actual duty and any acceptance condition. Distinguish performance from waiver or settlement, record reviewer-approved interpretations, and keep disputed discharge visible rather than closing on an internal completion flag.
2. Return field-level atoms for the actual ObligationStatus record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
