---
args: { prepared: PreparedResearchAttempt, goal: GoalContract, policy: PolicySelection }
returns: DecisionsResearchAttempt
---
1. Apply this domain algorithm to each relevant evidence unit: Read plans, execution logs, and researcher notes to reconstruct the attempted test. Separate intended from actual procedure and identify incomplete steps, measurement failures, or deviations that limit what the result can establish.
2. Return field-level atoms for the actual ResearchAttempt record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
