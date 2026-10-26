---
args: { prepared: PreparedExceptionRecord, goal: GoalContract, policy: PolicySelection }
returns: DecisionsExceptionRecord
---
1. Apply this domain algorithm to each relevant evidence unit: Extract the exact deviation, beneficiary, dates, and conditions of each granted exception. Keep the requirement that was waived separate from prerequisites that remained binding, and record whether the decision explicitly denied precedential value.
2. Return field-level atoms for the actual ExceptionRecord record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
