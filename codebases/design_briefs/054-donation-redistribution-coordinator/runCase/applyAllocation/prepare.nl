---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: DonationBundle[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for applyAllocation. Apply supplied fairness and priority principles to feasible bundles, using exact inventory and transport constraints. Explain tradeoffs and preserve unmet needs without inventing deservingness criteria or optimizing only the easiest deliveries. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
