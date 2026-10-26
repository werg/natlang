---
args: { prepared: PreparedReconciliationBreak, goal: GoalContract, policy: PolicySelection }
returns: DecisionsReconciliationBreak
---
1. Apply this domain algorithm to each relevant evidence unit: Consume exact reconciliation differences and inspect supporting narratives. Distinguish timing, mapping, missing postings, and unsupported explanations; group related breaks only when they share a demonstrated mechanism rather than the same amount.
2. Return field-level atoms for the actual ReconciliationBreak record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
