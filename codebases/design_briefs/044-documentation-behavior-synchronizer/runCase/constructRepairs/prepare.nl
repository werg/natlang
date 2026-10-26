---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: DocumentationMismatch[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for constructRepairs. Draft coordinated changes to pages, examples, tests, and release notes according to the reviewed repair direction. Preserve unaffected claims, update cross-links and terminology, and keep code examples executable under their stated environment. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
