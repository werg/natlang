---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: DonationOffer[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for interpretNeeds. Recover recipient needs from descriptions and prior confirmations without inferring sensitive circumstances. Distinguish necessary features from preferences, record accessibility and timing constraints explicitly supplied by recipients, and preserve consent for contact or delivery. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
