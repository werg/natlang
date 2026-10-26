---
args: { prepared: PreparedRequirementConflict, goal: GoalContract, policy: PolicySelection }
returns: DecisionsRequirementConflict
---
1. Apply this domain algorithm to each relevant evidence unit: Identify requirements that cannot jointly hold under the supplied technical, time, or resource constraints. Distinguish true contradictions from terminology differences and propose the smallest clarification or relaxation that would make the pair compatible.
2. Return field-level atoms for the actual RequirementConflict record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
