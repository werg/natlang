---
args: { prepared: PreparedSamplingPlan, goal: GoalContract, policy: PolicySelection }
returns: DecisionsSamplingPlan
---
1. Apply this domain algorithm to each relevant evidence unit: Define strata from the inquiry, not from convenient examples, then request exact selection within each stratum. Include negative and atypical cases, record nonresponse, and freeze a sampling manifest before coding the main set.
2. Return field-level atoms for the actual SamplingPlan record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
