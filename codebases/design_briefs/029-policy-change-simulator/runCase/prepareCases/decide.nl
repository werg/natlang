---
args: { prepared: PreparedPolicyCaseFacts, goal: GoalContract, policy: PolicySelection }
returns: DecisionsPolicyCaseFacts
---
1. Apply this domain algorithm to each relevant evidence unit: Normalize historical or synthetic cases into evidence-backed facts without adding missing information. Preserve uncertainty, policy-relevant context, and data-access restrictions; do not infer sensitive attributes merely to complete a simulation.
2. Return field-level atoms for the actual PolicyCaseFacts record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
