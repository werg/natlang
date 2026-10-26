---
args: { prepared: PreparedResearchFinding, goal: GoalContract, policy: PolicySelection }
returns: DecisionsResearchFinding
---
1. Apply this domain algorithm to each relevant evidence unit: Build findings around supported mechanisms and recurring patterns, actively search for counterexamples, and distinguish an absent observation from evidence of absence. Return unresolved questions and a traceable path from each conclusion to the coded passages.
2. Return field-level atoms for the actual ResearchFinding record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
