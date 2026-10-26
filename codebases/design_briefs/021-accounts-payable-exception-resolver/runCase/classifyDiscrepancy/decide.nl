---
args: { prepared: PreparedPayableDiscrepancy, goal: GoalContract, policy: PolicySelection }
returns: DecisionsPayableDiscrepancy
---
1. Apply this domain algorithm to each relevant evidence unit: Interpret differences using contract terms and receiving evidence. Distinguish partial delivery, unauthorized substitution, freight charges, and genuine duplication; a repeated invoice number is a signal to investigate rather than a sufficient payment decision.
2. Return field-level atoms for the actual PayableDiscrepancy record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
