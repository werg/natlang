---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for interpretDisruption. Read carrier notices and operational reports to determine what is unavailable, for how long, and with what certainty. Distinguish a delayed service from a closed route and preserve exceptions that may allow some shipment classes to continue. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
