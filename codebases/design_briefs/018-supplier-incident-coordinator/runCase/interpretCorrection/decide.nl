---
args: { prepared: PreparedCorrectiveAction, goal: GoalContract, policy: PolicySelection }
returns: DecisionsCorrectiveAction
---
1. Apply this domain algorithm to each relevant evidence unit: Evaluate the supplier’s explanation against incident evidence and rival mechanisms. Separate immediate correction from prevention of recurrence, identify testable claims, and request evidence that the proposed change addresses the actual mechanism rather than just the symptoms.
2. Return field-level atoms for the actual CorrectiveAction record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
