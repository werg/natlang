---
args: { prepared: PreparedPolicyOption, goal: GoalContract, policy: PolicySelection }
returns: DecisionsPolicyOption
---
1. Apply this domain algorithm to each relevant evidence unit: Summarize who and what changes under each candidate, then propose clarifications or alternatives for the policy owner. Preserve minority and exceptional cases, report coverage limits, and avoid choosing normative priorities not supplied by the organization.
2. Return field-level atoms for the actual PolicyOption record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
