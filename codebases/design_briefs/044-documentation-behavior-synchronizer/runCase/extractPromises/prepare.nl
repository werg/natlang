---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for extractPromises. Extract checkable statements about inputs, outputs, defaults, errors, and supported workflows from documentation and examples. Preserve version and audience scope, distinguishing normative promises from illustrative suggestions or historical notes. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
