---
args: { prepared: PreparedTenderRequirement, goal: GoalContract, policy: PolicySelection }
returns: DecisionsTenderRequirement
---
1. Apply this domain algorithm to each relevant evidence unit: Read the main tender, schedules, forms, and addenda together. Extract explicit requirements and conditional submission instructions, distinguish mandatory eligibility from scored preferences, and preserve exact formatting or attachment obligations separately from substantive answers.
2. Return field-level atoms for the actual TenderRequirement record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
