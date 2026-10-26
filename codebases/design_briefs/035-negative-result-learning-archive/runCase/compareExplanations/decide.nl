---
args: { prepared: PreparedFailureExplanation, goal: GoalContract, policy: PolicySelection }
returns: DecisionsFailureExplanation
---
1. Apply this domain algorithm to each relevant evidence unit: Construct competing explanations grounded in observed deviations and results. Separate demonstrated causes from plausible hypotheses, seek contradictory observations, and identify shared mechanisms across attempts without assuming all similar outcomes have one cause.
2. Return field-level atoms for the actual FailureExplanation record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
