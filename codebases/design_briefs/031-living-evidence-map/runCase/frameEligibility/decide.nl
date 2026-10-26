---
args: { prepared: PreparedEligibilityCriterion, goal: GoalContract, policy: PolicySelection }
returns: DecisionsEligibilityCriterion
---
1. Apply this domain algorithm to each relevant evidence unit: Translate the research question into explicit screening criteria and distinguish mandatory eligibility from useful descriptive metadata. Define ambiguous boundaries with examples before full screening and preserve questions requiring expert methodological judgment.
2. Return field-level atoms for the actual EligibilityCriterion record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
