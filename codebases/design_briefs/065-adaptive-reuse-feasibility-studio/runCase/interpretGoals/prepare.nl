---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: ReuseSiteFact[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for interpretGoals. Translate stakeholder objectives into required functions and explicit tradeoffs. Separate mandatory supplied constraints from preferences, identify incompatible goals, and avoid choosing political or commercial priorities not authorized by the project brief. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
