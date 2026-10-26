---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: AssetIdentity[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for matchArtifacts. Compare submitted manuals, inspection records, certificates, and training logs with the actual requirement. Distinguish a blank form from completed evidence and a generic manual from the installed configuration; retain missing signatures or scope mismatches. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
