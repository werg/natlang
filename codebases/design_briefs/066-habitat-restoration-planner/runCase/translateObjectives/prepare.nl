---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: HabitatObservation[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for translateObjectives. Convert landholder and ecological goals into observable objectives under supplied guidance. Separate desired long-term outcomes from near-term actions, identify conflicting uses, and preserve uncertainty about causal pathways rather than promising guaranteed ecological results. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
