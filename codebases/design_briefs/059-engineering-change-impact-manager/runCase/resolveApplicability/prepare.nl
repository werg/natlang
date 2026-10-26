---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: DesignDelta[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for resolveApplicability. Interpret effective serials, lots, dates, configurations, and authorized exceptions. Distinguish new production from retrofit obligations and preserve fielded units outside scope rather than applying the latest revision universally. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
