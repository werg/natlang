---
args: { prepared: PreparedDiscriminatingExperiment, goal: GoalContract, policy: PolicySelection }
returns: DecisionsDiscriminatingExperiment
---
1. Apply this domain algorithm to each relevant evidence unit: Propose bounded low-risk follow-ups that distinguish rival explanations with minimal changes. Specify what each outcome would mean before execution and avoid repeating the same invalid procedure or adding so many changes that the result becomes uninterpretable.
2. Return field-level atoms for the actual DiscriminatingExperiment record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
