---
args: { prepared: PreparedExperimentSlot, goal: GoalContract, policy: PolicySelection }
returns: DecisionsExperimentSlot
---
1. Apply this domain algorithm to each relevant evidence unit: Submit hard resource and timing constraints to an exact scheduler, then evaluate feasible schedules against supplied priorities and scientific dependencies. Keep optional preferences visible and explain unavoidable conflicts instead of silently dropping experiments.
2. Return field-level atoms for the actual ExperimentSlot record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
