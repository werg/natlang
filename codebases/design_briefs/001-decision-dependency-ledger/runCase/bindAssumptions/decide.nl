---
args: { prepared: PreparedAssumptionBinding, goal: GoalContract, policy: PolicySelection }
returns: DecisionsAssumptionBinding
---
1. Apply this domain algorithm to each relevant evidence unit: For each decision ask what would have to change to make reconsideration necessary. Separate essential premises from background rationale; emit a directed dependency with a counterfactual explanation and a passage establishing the connection.
2. Return field-level atoms for the actual AssumptionBinding record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
