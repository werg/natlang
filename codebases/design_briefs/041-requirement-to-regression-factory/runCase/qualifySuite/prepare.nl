---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: TestFailureFinding[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for qualifySuite. Run independent seeded defects and mutation tests to evaluate whether the suite detects meaningful violations. Remove redundant tests only when coverage remains, preserve regression cases for repaired bugs, and report uncovered requirements explicitly. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
