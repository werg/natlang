---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for recoverSiteFacts. Reconcile inventories, surveys, historical plans, and stakeholder descriptions into verified facts and unknowns. Distinguish current observations from outdated drawings and preserve uncertainties that could materially change feasibility or cost. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
