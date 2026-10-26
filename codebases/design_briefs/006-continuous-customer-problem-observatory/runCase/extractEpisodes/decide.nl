---
args: { prepared: PreparedProblemEpisode, goal: GoalContract, policy: PolicySelection }
returns: DecisionsProblemEpisode
---
1. Apply this domain algorithm to each relevant evidence unit: Read enough surrounding conversation to identify the attempted job, the obstacle, and its consequence. Separate a product complaint from a requested feature, and split independently resolvable problems while retaining links to the original interaction.
2. Return field-level atoms for the actual ProblemEpisode record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
