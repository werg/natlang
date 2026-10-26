---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Parse the request into population, entities, semantic predicates, temporal relations, grouping, and desired output. Generate competing interpretations only where they change the answer; resolve decisive ambiguity before choosing a convenient available column.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
