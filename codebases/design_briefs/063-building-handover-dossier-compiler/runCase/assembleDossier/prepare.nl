---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: HandoverClosureTask[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for assembleDossier. Produce a navigable asset-linked dossier and a separate outstanding-obligations register. Verify search, links, file integrity, and recipient access, then record documentary acceptance independently from any qualified technical approval or occupancy decision. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
