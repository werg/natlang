---
args: { prepared: PreparedAnalysisOrder, goal: GoalContract, policy: PolicySelection }
returns: DecisionsAnalysisOrder
---
1. Apply this domain algorithm to each relevant evidence unit: Convert unresolved issues into typed analysis plans with cohorts, time windows, allowed operations, and interpretation rules. Separate confirmatory follow-up from exploratory searching and require a fresh data or holdout plan where the proposed claim needs one.
2. Return field-level atoms for the actual AnalysisOrder record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
