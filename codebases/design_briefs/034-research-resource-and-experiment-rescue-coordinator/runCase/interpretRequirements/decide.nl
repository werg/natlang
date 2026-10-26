---
args: { prepared: PreparedExperimentRequirement, goal: GoalContract, policy: PolicySelection }
returns: DecisionsExperimentRequirement
---
1. Apply this domain algorithm to each relevant evidence unit: Read approved protocols and researcher requests to distinguish essential experimental conditions from convenient preferences. Identify sample timing, equipment capabilities, and required personnel without changing scientific procedures or inferring unapproved substitutions.
2. Return field-level atoms for the actual ExperimentRequirement record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
