---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: InstructionVariant[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for replayProcedure. Generate ordinary and adverse scenarios involving local exceptions, missing resources, and interrupted work. Replay the adapted procedure to detect ambiguous instructions, skipped prerequisites, or branches that cannot reach a safe terminal state. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
