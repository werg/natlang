---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: BehavioralRequirement[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for modelPartitions. Partition input and state space by behaviorally meaningful distinctions, including invalid inputs and interacting requirements. Identify boundary cases and previously untested combinations rather than generating many superficial variants of the same happy path. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
