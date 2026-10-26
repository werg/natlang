---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for formalizeQuestion. Parse the request into population, entities, semantic predicates, temporal relations, grouping, and desired output. Generate competing interpretations only where they change the answer; resolve decisive ambiguity before choosing a convenient available column. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
