---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: RestorationObjective[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for constructInterventions. Propose staged interventions from approved guidance and known site conditions. Explain the hypothesized mechanism, dependencies, resource needs, and stop conditions; route unfamiliar or potentially harmful interventions to qualified specialists instead of improvising. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
