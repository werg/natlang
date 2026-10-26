---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: HandoverRequirement[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for resolveAssets. Reconcile asset tags, drawing labels, manuals, and contractor naming conventions. Distinguish identical models at different locations, preserve serial and revision data, and flag ambiguous identity before attaching a certificate or manual to the wrong asset. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
