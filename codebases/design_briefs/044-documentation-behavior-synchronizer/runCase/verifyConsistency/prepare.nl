---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: DocumentationPatch[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for verifyConsistency. Re-extract changed promises, rerun examples and targeted behavior probes, and inspect linked pages for stale repetitions. Publish the reviewed bundle only when its examples and claims correspond to the same release version. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
