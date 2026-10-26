---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: RegressionTest[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for interpretFailures. Compare failures with requirements, execution traces, and fixture health. Distinguish a product defect from a bad oracle, flaky setup, or unavailable dependency; propose the smallest repair without weakening a valid assertion to match defective behavior. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
