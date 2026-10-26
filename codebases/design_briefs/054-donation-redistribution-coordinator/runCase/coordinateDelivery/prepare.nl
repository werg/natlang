---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: DonationAllocation[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for coordinateDelivery. Arrange consented collection or delivery, verify availability before dispatch, and track actual recipient acceptance. Reallocate declined or unavailable items through the same policy rather than informal side agreements, and preserve privacy in donor communications. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
