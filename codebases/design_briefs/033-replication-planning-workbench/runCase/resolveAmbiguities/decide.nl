---
args: { prepared: PreparedMethodQuestion, goal: GoalContract, policy: PolicySelection }
returns: DecisionsMethodQuestion
---
1. Apply this domain algorithm to each relevant evidence unit: Enumerate plausible interpretations of missing details and assess whether each changes the result. Seek author clarification or documented evidence for consequential choices; retain sensitivity branches where ambiguity cannot be resolved.
2. Return field-level atoms for the actual MethodQuestion record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
